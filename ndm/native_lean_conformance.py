"""Differential runner for the Lean 4 and production native coordinators.

The canonical input is ``formal/resilient/trace-schema-v1.json``.  Lean is the
pure protocol oracle.  Native execution reaches the persistent compiled
service through ``ndp_coord_step_v1`` and the production RPC/service/kernel
call path.  This module translates identities and emits a common state view;
it contains no admission, closure, receipt, commit, apply, or recovery policy.

The native ABI intentionally omits human-readable worker names, allocation
names, and some proof-only identities.  The adapter retains those opaque names
beside the ABI result.  Every mutable decision and every ABI-visible value in
the common view comes from the native result after the real mutation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Iterator, Mapping, Sequence

from ndm.native_artifacts import (
    BuildAttestation,
    sha256_file,
    validate_build_manifest,
)
from ndm.native_coordination import (
    DEADLINE_EXPIRED,
    FINITE_CLOSE,
    NativeCoordinationAuthority,
)
from ndm.native_dataplane import (
    ABI_V1,
    Client,
    CoordinationEventKind,
    CoordinationEventV1,
    CoordinationMemberV1,
    CoordinationResultV1,
    NativeLibrary,
    Role,
)


TRACE_SCHEMA = "emender-resilient-coordination-trace-v1"
TRACE_SCHEMA_DIGEST = (
    "cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4"
)
VIEW_VERSION = "emender-native-lean-authority-view-v1"
RUN_SCHEMA = "emender-native-lean-conformance-run-v1"
IDENTITY_SCHEMA = "emender-native-lean-conformance-identities-v1"
TOOLCHAIN = (
    "leanprover/lean4:v4.26.0"
    "@d8204c9fd894f91bbb2cdfec5912ec8196fd8562"
)
_ZERO_DIGEST = "0" * 64
_TOKEN_HEX = "7c" * 32
_DIGEST_MODULUS = 1 << 64


class TraceFormatError(ValueError):
    """A canonical trace failed before either authority could mutate."""


class ConformanceDivergence(AssertionError):
    """Lean and native first disagreed at one canonical trace step."""

    def __init__(self, report: Mapping[str, object]):
        self.report = dict(report)
        index = self.report.get("eventIndex", "?")
        event_id = self.report.get("eventId", "?")
        differences = self.report.get("differences", [])
        super().__init__(
            f"native/Lean divergence at event {index} ({event_id}): "
            f"{differences}"
        )


def canonical_json(value: object) -> str:
    """Return the one v1 JSON representation used by both runners."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _reject_duplicate_pairs(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise TraceFormatError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def load_canonical_trace(path: str | Path) -> dict[str, object]:
    """Load canonical JSON and fail closed on aliases or ambiguous objects."""
    trace_path = Path(path)
    try:
        raw = trace_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TraceFormatError(f"cannot read trace: {trace_path}") from error
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (json.JSONDecodeError, TraceFormatError) as error:
        raise TraceFormatError(f"malformed trace JSON: {error}") from error
    if not isinstance(value, dict):
        raise TraceFormatError("trace root must be an object")
    if raw.strip() != canonical_json(value):
        raise TraceFormatError(
            "trace is not the deterministic compact, sorted-key v1 encoding"
        )
    required_keys = {
        "schemaVersion",
        "schemaDigest",
        "policyName",
        "policySchema",
        "policyDigest",
        "toolchain",
        "sourceSchema",
        "sourceDigest",
        "initialState",
        "steps",
    }
    if set(value) != required_keys:
        raise TraceFormatError(
            "trace top-level identity set is incomplete or unknown"
        )
    if value.get("schemaVersion") != TRACE_SCHEMA:
        raise TraceFormatError("trace schema version mismatch")
    digest = _value(value.get("schemaDigest"), field="schemaDigest")
    if digest != TRACE_SCHEMA_DIGEST:
        raise TraceFormatError("trace schema digest mismatch")
    if value.get("toolchain") != TOOLCHAIN:
        raise TraceFormatError("trace Lean toolchain identity mismatch")
    return value


def coordination_digest(domain: str, material: str) -> str:
    """Byte-for-byte Python rendition of Lean ``coordinationDigest``."""
    domain_material = f"{domain}\0{material}"
    lanes = []
    for seed in (
        1469598103934665603,
        1099511628211,
        7809847782465536322,
        9650029242287828579,
    ):
        lane = seed
        for character in domain_material:
            lane = (
                lane * 1099511628211
                + ord(character)
                + 1469598103934665603
            ) % _DIGEST_MODULUS
        lanes.append(f"{lane:016x}")
    return "".join(lanes)


def state_digest(state: Mapping[str, object]) -> str:
    return coordination_digest(VIEW_VERSION, canonical_json(state))


def _value(value: object, *, field: str) -> object:
    if not isinstance(value, dict) or set(value) != {"value"}:
        raise TraceFormatError(f"{field} must be a single-field identity object")
    return value["value"]


def _text(value: object, *, field: str) -> str:
    result = _value(value, field=field)
    if not isinstance(result, str) or not result:
        raise TraceFormatError(f"{field} must be a nonempty string identity")
    return result


def _nat(value: object, *, field: str) -> int:
    result = _value(value, field=field)
    if (
        not isinstance(result, int)
        or isinstance(result, bool)
        or result < 0
        or result >= 1 << 64
    ):
        raise TraceFormatError(f"{field} must be a bounded natural identity")
    return result


def _digest(value: object, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != 64 or any(character not in "0123456789abcdef"
                                for character in result):
        raise TraceFormatError(f"{field} must be a lowercase SHA-256 identity")
    return result


def _event(step: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    wrapper = step.get("event")
    if not isinstance(wrapper, dict) or len(wrapper) != 1:
        raise TraceFormatError("trace event must have exactly one typed variant")
    kind, encoded = next(iter(wrapper.items()))
    if not isinstance(encoded, dict):
        raise TraceFormatError(f"{kind} event wrapper must be an object")
    body = encoded.get("value", encoded)
    if not isinstance(body, dict):
        raise TraceFormatError(f"{kind} event body must be an object")
    return kind, body


def _context(body: Mapping[str, object]) -> Mapping[str, object]:
    value = body.get("context")
    if not isinstance(value, dict):
        raise TraceFormatError("typed event is missing context")
    return value


def _artifact(attestation: BuildAttestation, name: str) -> Path:
    record = attestation.artifacts[name]
    return (attestation.path.parent / str(record["path"])).resolve()


@contextmanager
def native_service(
    service_binary: str | Path,
) -> Iterator[tuple[str, bytes, subprocess.Popen[bytes]]]:
    """Launch the real persistent service on a model-free local provider."""
    with tempfile.TemporaryDirectory(prefix="emender-native-lean-") as directory:
        socket_path = str(Path(directory) / "ndp.sock")
        token = bytes.fromhex(_TOKEN_HEX)
        environment = dict(os.environ)
        process = subprocess.Popen(
            [
                str(Path(service_binary).resolve()),
                "--provider", "tcp;ofi_rxm",
                "--test-only", "--serve",
                "--bind-node", "127.0.0.1",
                "--socket", socket_path,
                "--admission-token-hex", _TOKEN_HEX,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            env=environment,
        )
        try:
            target = Path(socket_path)
            for _ in range(300):
                if target.is_socket():
                    break
                return_code = process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        "ndp_cxi_service exited before binding its socket: "
                        f"{return_code}"
                    )
                time.sleep(0.05)
            else:
                raise TimeoutError(
                    "ndp_cxi_service did not bind its control socket"
                )
            yield socket_path, token, process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def run_lean_oracle(
    trace_path: str | Path, lean_runner: str | Path
) -> dict[str, object]:
    completed = subprocess.run(
        [str(Path(lean_runner).resolve()), str(Path(trace_path).resolve())],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise TraceFormatError(f"Lean rejected canonical trace: {detail}")
    try:
        value = json.loads(
            completed.stdout, object_pairs_hook=_reject_duplicate_pairs
        )
    except (json.JSONDecodeError, TraceFormatError) as error:
        raise RuntimeError("Lean oracle emitted invalid JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("viewVersion") != VIEW_VERSION
        or value.get("schemaVersion") != TRACE_SCHEMA
        or value.get("schemaDigest") != TRACE_SCHEMA_DIGEST
        or value.get("toolchain") != TOOLCHAIN
    ):
        raise RuntimeError("Lean oracle identity envelope mismatch")
    return value


@dataclass
class _ContributionIdentity:
    sequence: int


@dataclass
class _NodeApplyIdentity:
    generation: int
    worker: str
    node: str
    incarnation: str
    result_digest: str
    receipt_digest: str


class NativeTraceAdapter:
    """Identity adapter over one live ``NativeCoordinationAuthority``."""

    def __init__(
        self,
        authority: NativeCoordinationAuthority,
        trace: Mapping[str, object],
        *,
        fault_event_index: int | None = None,
    ):
        initial = trace.get("initialState")
        if not isinstance(initial, dict):
            raise TraceFormatError("trace initialState must be an object")
        authority_identity = initial.get("authority")
        policy = initial.get("policy")
        if not isinstance(authority_identity, dict) or not isinstance(policy, dict):
            raise TraceFormatError("trace initial authority/policy is missing")
        self.authority = authority
        self.policy = policy
        self.static = {
            "viewVersion": VIEW_VERSION,
            "traceSchema": str(authority_identity["traceSchema"]),
            "traceSchemaDigest": _digest(
                authority_identity["traceSchemaDigest"],
                field="initial.authority.traceSchemaDigest",
            ),
            "toolchain": str(authority_identity["toolchain"]),
            "sourceSchema": str(authority_identity["sourceSchema"]),
            "sourceDigest": _digest(
                authority_identity["sourceDigest"],
                field="initial.authority.sourceDigest",
            ),
            "run": _text(authority_identity["run"], field="initial.authority.run"),
            "allocation": _text(
                authority_identity["allocation"],
                field="initial.authority.allocation",
            ),
            "fence": _nat(
                authority_identity["fence"], field="initial.authority.fence"
            ),
            "policyName": str(authority_identity["policyName"]),
            "policySchema": str(authority_identity["policySchema"]),
            "policyDigest": _digest(
                authority_identity["policyDigest"],
                field="initial.authority.policyDigest",
            ),
            "layoutDigest": _digest(
                authority_identity["layoutDigest"],
                field="initial.authority.layoutDigest",
            ),
            "codeDigest": _digest(
                authority_identity["codeDigest"],
                field="initial.authority.codeDigest",
            ),
        }
        self.authority_receipt = _text(
            initial["baseReceipt"], field="initial.baseReceipt"
        )
        self.result_id = (
            None
            if initial.get("lastResult") is None
            else _text(initial["lastResult"], field="initial.lastResult")
        )
        self.node_workers: dict[str, str] = {}
        self.node_synced_generation: dict[str, int] = {}
        self.node_control_sequence: dict[str, int] = {}
        self.contribution_identities: dict[str, _ContributionIdentity] = {}
        self.node_apply_identities: dict[str, _NodeApplyIdentity] = {}
        self.cohort_digest: str | None = None
        self.close_ticks: dict[tuple[int, int], int] = {}
        self.fault_event_index = fault_event_index
        self.last_result: dict[str, object] | None = None
        self.native_calls: list[dict[str, object]] = []

    def _record_call(
        self,
        event_index: int,
        projection: str,
        result: dict[str, object],
    ) -> dict[str, object]:
        trace = json.loads(str(result["trace"]))
        record = {
            "canonicalEventIndex": event_index,
            "projection": projection,
            "event": trace["event"],
            "disposition": str(result["disposition"]).replace("-", "_"),
            "preStateDigest": result["pre_state_digest"],
            "postStateDigest": result["post_state_digest"],
        }
        self.native_calls.append(record)
        self.last_result = result
        return result

    def _step(
        self,
        event_index: int,
        projection: str,
        kind: CoordinationEventKind,
        **fields: object,
    ) -> dict[str, object]:
        return self._record_call(
            event_index,
            projection,
            self.authority.step(kind, **fields),
        )

    def _query(self, event_index: int, projection: str) -> dict[str, object]:
        generation = (
            int(self.last_result["committed_generation"])
            if self.last_result is not None
            else 0
        )
        return self._step(
            event_index,
            projection,
            CoordinationEventKind.QUERY_COMMIT,
            generation=generation,
        )

    def _worker_for_node(self, node: str) -> str:
        return self.node_workers.get(node, node)

    def _node_for_worker(self, worker: str) -> str:
        for node, candidate in self.node_workers.items():
            if candidate == worker:
                return node
        return worker

    def execute(
        self,
        event_index: int,
        kind: str,
        body: Mapping[str, object],
    ) -> tuple[str, Mapping[str, object], Sequence[Mapping[str, object]]]:
        """Translate one typed event, call production, and return its view."""
        first_call = len(self.native_calls)
        context = _context(body) if kind != "claimFence" else None
        fields: dict[str, object] = {}
        if context is not None:
            fields = {
                "run_id": _text(context["run"], field=f"{kind}.context.run"),
                "fence": _nat(context["fence"], field=f"{kind}.context.fence"),
                "policy_digest": _digest(
                    context["policyDigest"],
                    field=f"{kind}.context.policyDigest",
                ),
            }
        if kind == "peerTransition":
            node = _text(body["node"], field="peerTransition.node")
            worker = _text(body["worker"], field="peerTransition.worker")
            incarnation = _text(
                body["incarnation"], field="peerTransition.incarnation"
            )
            generation = _nat(
                body["syncedGeneration"],
                field="peerTransition.syncedGeneration",
            )
            sequence = self.node_control_sequence.get(node, 1)
            to_phase = str(body["toPhase"])
            if to_phase == "discover":
                result = self._step(
                    event_index,
                    "peer-admission/recovery",
                    CoordinationEventKind.RECOVER_PEER,
                    **fields,
                    generation=generation,
                    node_id=node,
                    incarnation=incarnation,
                    sequence=sequence,
                    receipt_digest=str(
                        self.last_result["commit_receipt"]
                        if self.last_result is not None
                        else _ZERO_DIGEST
                    ),
                )
            elif to_phase == "ready":
                native_member = next(
                    (
                        item
                        for item in self.last_result["members"]
                        if item["worker_id"] == node
                    ),
                    None,
                )
                ready_receipt = (
                    _ZERO_DIGEST
                    if generation == 0
                    else (
                        str(native_member["apply_receipt"])
                        if native_member is not None
                        else str(self.last_result["commit_receipt"])
                    )
                )
                result = self._step(
                    event_index,
                    "leased-ready",
                    CoordinationEventKind.READY,
                    **fields,
                    generation=generation,
                    node_id=node,
                    incarnation=incarnation,
                    sequence=sequence,
                    receipt_digest=ready_receipt,
                )
            elif to_phase in {"drain", "expire"}:
                result = self._step(
                    event_index,
                    "peer-expiry",
                    CoordinationEventKind.EXPIRE_PEER,
                    **fields,
                    node_id=node,
                    incarnation=incarnation,
                    sequence=sequence,
                )
            else:
                result = self._query(event_index, "peer-bootstrap/sync")
            if _native_name(result) in {"accepted", "identical_duplicate"}:
                self.node_workers[node] = worker
                self.node_synced_generation[node] = generation
                self.node_control_sequence[node] = sequence
        elif kind == "registerTrainer":
            result = self._query(event_index, "trainer-registration-identity")
        elif kind == "openGeneration":
            generation = _nat(
                context["generation"], field="openGeneration.context.generation"
            )
            attempt = _nat(
                context["attempt"], field="openGeneration.context.attempt"
            )
            result = self._step(
                event_index,
                "generation-admission",
                CoordinationEventKind.OPEN_GENERATION,
                **fields,
                generation=generation,
                attempt=attempt + 1,
            )
            if _native_name(result) in {"accepted", "identical_duplicate"}:
                self.close_ticks[(generation, attempt)] = _nat(
                    body["closeTick"], field="openGeneration.closeTick"
                )
                self.cohort_digest = None
                self.contribution_identities.clear()
                self.node_apply_identities.clear()
        elif kind == "contribution":
            node = _text(body["node"], field="contribution.node")
            worker = _text(body["worker"], field="contribution.worker")
            generation = _nat(
                context["generation"], field="contribution.context.generation"
            )
            attempt = _nat(
                context["attempt"], field="contribution.context.attempt"
            )
            exact_tokens = int(body["exactTokens"])
            payload = _digest(
                body["payloadDigest"], field="contribution.payloadDigest"
            )
            if not (
                body.get("finite") is True
                and body.get("checksumValid") is True
                and body.get("layoutValid") is True
            ):
                payload = _ZERO_DIGEST
            if event_index == self.fault_event_index:
                exact_tokens += 1
            result = self._step(
                event_index,
                "contribution-receipt",
                CoordinationEventKind.CONTRIBUTION,
                **fields,
                generation=generation,
                attempt=attempt + 1,
                node_id=node,
                incarnation=_text(
                    body["incarnation"], field="contribution.incarnation"
                ),
                sequence=_nat(
                    body["sequence"], field="contribution.sequence"
                ),
                exact_tokens=exact_tokens,
                payload_digest=payload,
            )
            if _native_name(result) == "accepted":
                self.node_workers[node] = worker
                self.contribution_identities[node] = _ContributionIdentity(
                    _nat(body["sequence"], field="contribution.sequence")
                )
                actual = next(
                    (
                        item
                        for item in result["members"]
                        if item["worker_id"] == node
                    ),
                    None,
                )
                expected_receipt = _digest(
                    body["receiptDigest"], field="contribution.receiptDigest"
                )
                if (
                    actual is None
                    or actual["contribution_receipt"] != expected_receipt
                ):
                    result = dict(result)
                    result["_translation_error"] = {
                        "field": "contribution.receiptDigest",
                        "canonical": expected_receipt,
                        "native": (
                            None
                            if actual is None
                            else actual["contribution_receipt"]
                        ),
                    }
                    self.last_result = result
        elif kind == "closeGeneration":
            generation = _nat(
                context["generation"], field="closeGeneration.context.generation"
            )
            attempt = _nat(
                context["attempt"], field="closeGeneration.context.attempt"
            )
            observed = _nat(
                body["observedAt"], field="closeGeneration.observedAt"
            )
            close_tick = self.close_ticks.get((generation, attempt))
            closure_mode = str(self.policy.get("closureMode", ""))
            flags = 0
            if close_tick is not None and observed >= close_tick:
                flags |= DEADLINE_EXPIRED
            if closure_mode == "floorOrDeadline":
                flags |= FINITE_CLOSE
            result = self._step(
                event_index,
                "finite-closure",
                CoordinationEventKind.CLOSE_GENERATION,
                **fields,
                generation=generation,
                attempt=attempt + 1,
                flags=flags,
            )
            if _native_name(result) == "accepted":
                self.cohort_digest = _digest(
                    body["cohortDigest"], field="closeGeneration.cohortDigest"
                )
        elif kind == "ownerLoss":
            worker = _text(body["owner"], field="ownerLoss.owner")
            node = self._node_for_worker(worker)
            result = self._step(
                event_index,
                "owner-replay/reassignment",
                CoordinationEventKind.OWNER_LOST,
                **fields,
                generation=_nat(
                    context["generation"], field="ownerLoss.context.generation"
                ),
                attempt=_nat(
                    context["attempt"], field="ownerLoss.context.attempt"
                ) + 1,
                node_id=node,
                incarnation=_text(
                    body["ownerIncarnation"],
                    field="ownerLoss.ownerIncarnation",
                ),
                sequence=_nat(
                    context["ownerEpoch"], field="ownerLoss.context.ownerEpoch"
                ) + 1,
            )
        elif kind == "commitGeneration":
            generation = _nat(
                context["generation"],
                field="commitGeneration.context.generation",
            )
            attempt = _nat(
                context["attempt"], field="commitGeneration.context.attempt"
            )
            if (
                self.last_result is not None
                and int(self.last_result["committed_generation"]) == generation
                and str(self.last_result["phase"]) == "closed"
            ):
                contributed = [
                    member
                    for member in self.last_result["members"]
                    if member["contributed"]
                ]
                exact_tokens = sum(
                    int(member["exact_tokens"]) for member in contributed
                )
                result_digest = _digest(
                    body["resultDigest"],
                    field="commitGeneration.resultDigest",
                )
                for member in contributed:
                    node = str(member["worker_id"])
                    identity = self.contribution_identities[node]
                    self._step(
                        event_index,
                        "result-receipt",
                        CoordinationEventKind.RESULT_RECEIPT,
                        **fields,
                        generation=generation,
                        attempt=attempt + 1,
                        node_id=node,
                        incarnation=str(member["cohort_incarnation"]),
                        sequence=identity.sequence,
                        exact_tokens=exact_tokens,
                        result_digest=result_digest,
                    )
            assert self.last_result is not None
            generation_tokens = sum(
                int(member["exact_tokens"])
                for member in self.last_result["members"]
                if member["contributed"]
            )
            target_generation = generation + 1
            if int(self.last_result["committed_generation"]) == target_generation:
                accepted_token_clock = int(
                    self.last_result["accepted_token_clock"]
                )
            else:
                accepted_token_clock = (
                    int(self.last_result["accepted_token_clock"])
                    + generation_tokens
                )
            result_digest = _digest(
                body["resultDigest"], field="commitGeneration.resultDigest"
            )
            result = self._step(
                event_index,
                "commit-authority",
                CoordinationEventKind.COMMIT,
                **fields,
                generation=target_generation,
                attempt=attempt + 1,
                exact_tokens=accepted_token_clock,
                previous_receipt_digest=str(
                    self.last_result["commit_receipt"]
                ),
                receipt_digest=_digest(
                    body["receiptDigest"],
                    field="commitGeneration.receiptDigest",
                ),
                manifest_digest=result_digest,
                result_digest=result_digest,
            )
            if _native_name(result) in {"accepted", "identical_duplicate"}:
                self.authority_receipt = _text(
                    body["receipt"], field="commitGeneration.receipt"
                )
                self.result_id = _text(
                    body["result"], field="commitGeneration.result"
                )
                if self.cohort_digest is None:
                    self.cohort_digest = _digest(
                        body["cohortDigest"],
                        field="commitGeneration.cohortDigest",
                    )
        elif kind == "publishResult":
            result = self._query(event_index, "result-publication-authority")
        elif kind == "trainerApply":
            result = self._query(event_index, "trainer-apply-identity")
        elif kind == "reduceNodeApply":
            node = _text(body["node"], field="reduceNodeApply.node")
            sequence = self.node_control_sequence.get(node, 0) + 1
            generation = _nat(
                context["generation"],
                field="reduceNodeApply.context.generation",
            ) + 1
            incarnation = _text(
                body["peerIncarnation"],
                field="reduceNodeApply.peerIncarnation",
            )
            receipt_digest = _digest(
                body["receiptDigest"],
                field="reduceNodeApply.receiptDigest",
            )
            result_digest = _digest(
                body["resultDigest"], field="reduceNodeApply.resultDigest"
            )
            result = self._step(
                event_index,
                "eight-to-one-node-apply",
                CoordinationEventKind.NODE_APPLY,
                **fields,
                generation=generation,
                node_id=node,
                incarnation=incarnation,
                sequence=sequence,
                trainer_count=8,
                receipt_digest=receipt_digest,
                previous_receipt_digest=str(
                    self.last_result["commit_receipt"]
                ),
            )
            if _native_name(result) == "accepted":
                worker = _text(body["worker"], field="reduceNodeApply.worker")
                self.node_workers[node] = worker
                self.node_control_sequence[node] = sequence
                self.node_synced_generation[node] = generation
                self.node_apply_identities[node] = _NodeApplyIdentity(
                    generation - 1,
                    worker,
                    node,
                    incarnation,
                    result_digest,
                    receipt_digest,
                )
        elif kind == "expirePeer":
            node = _text(body["node"], field="expirePeer.node")
            result = self._step(
                event_index,
                "lease-expiry",
                CoordinationEventKind.EXPIRE_PEER,
                **fields,
                node_id=node,
                incarnation=_text(
                    body["incarnation"], field="expirePeer.incarnation"
                ),
                sequence=self.node_control_sequence.get(node, 1),
            )
        elif kind == "loss":
            node = _text(body["node"], field="loss.node")
            if str(body["role"]) == "trainer":
                incarnation = _text(
                    body["peerIncarnation"], field="loss.peerIncarnation"
                )
                sequence = self.node_control_sequence.get(node, 1)
                expired = self._step(
                    event_index,
                    "trainer-loss-expire-node-authority",
                    CoordinationEventKind.EXPIRE_PEER,
                    **fields,
                    node_id=node,
                    incarnation=incarnation,
                    sequence=sequence,
                )
                if _native_name(expired) not in {
                    "accepted", "identical_duplicate",
                }:
                    result = expired
                else:
                    sequence += 1
                    result = self._step(
                        event_index,
                        "trainer-loss-recover-node-authority",
                        CoordinationEventKind.RECOVER_PEER,
                        **fields,
                        generation=int(
                            self.last_result["committed_generation"]
                        ),
                        node_id=node,
                        incarnation=incarnation,
                        sequence=sequence,
                        receipt_digest=str(
                            self.last_result["commit_receipt"]
                        ),
                    )
                    if _native_name(result) == "accepted":
                        self.node_control_sequence[node] = sequence
            else:
                result = self._step(
                    event_index,
                    f"{body['role']}-loss",
                    CoordinationEventKind.EXPIRE_PEER,
                    **fields,
                    node_id=node,
                    incarnation=_text(
                        body["peerIncarnation"], field="loss.peerIncarnation"
                    ),
                    sequence=self.node_control_sequence.get(node, 1),
                )
        elif kind == "restartPeer":
            node = _text(body["node"], field="restartPeer.node")
            sequence = self.node_control_sequence.get(node, 0) + 1
            generation = _nat(
                body["syncedGeneration"], field="restartPeer.syncedGeneration"
            )
            result = self._step(
                event_index,
                "peer-restart/recovery",
                CoordinationEventKind.RECOVER_PEER,
                **fields,
                generation=generation,
                node_id=node,
                incarnation=_text(
                    body["newIncarnation"], field="restartPeer.newIncarnation"
                ),
                sequence=sequence,
                receipt_digest=str(self.last_result["commit_receipt"]),
            )
            if _native_name(result) == "accepted":
                self.node_workers[node] = _text(
                    body["worker"], field="restartPeer.worker"
                )
                self.node_control_sequence[node] = sequence
                self.node_synced_generation[node] = generation
        elif kind == "claimFence":
            claim_authority = body.get("authority")
            claim_policy = body.get("policy")
            if not isinstance(claim_authority, dict) or not isinstance(
                claim_policy, dict
            ):
                raise TraceFormatError("claimFence authority/policy is missing")
            result = self._step(
                event_index,
                "fresh-fence-recovery",
                CoordinationEventKind.RECOVER_AUTHORITY,
                run_id=_text(
                    claim_authority["run"], field="claimFence.authority.run"
                ),
                fence=_nat(
                    claim_authority["fence"],
                    field="claimFence.authority.fence",
                ),
                policy_digest=_digest(
                    claim_authority["policyDigest"],
                    field="claimFence.authority.policyDigest",
                ),
                generation=_nat(
                    body["baseGeneration"], field="claimFence.baseGeneration"
                ),
                exact_tokens=int(body["acceptedTokenClock"]),
                receipt_digest=_digest(
                    body["baseReceiptDigest"],
                    field="claimFence.baseReceiptDigest",
                ),
                previous_receipt_digest=str(
                    self.last_result["commit_receipt"]
                ),
                manifest_digest=_digest(
                    body["baseDigest"], field="claimFence.baseDigest"
                ),
                result_digest=_digest(
                    body["baseDigest"], field="claimFence.baseDigest"
                ),
                minimum_nodes=int(claim_policy["qMin"]),
                minimum_tokens=int(claim_policy["tMin"]),
            )
            if _native_name(result) == "accepted":
                self.authority.fence = int(result["fence"])
                self.static.update(
                    {
                        "allocation": _text(
                            claim_authority["allocation"],
                            field="claimFence.authority.allocation",
                        ),
                        "fence": int(result["fence"]),
                        "policyName": str(claim_authority["policyName"]),
                        "policySchema": str(claim_authority["policySchema"]),
                        "policyDigest": str(result["policy_digest"]),
                        "sourceDigest": _digest(
                            claim_authority["sourceDigest"],
                            field="claimFence.authority.sourceDigest",
                        ),
                        "layoutDigest": _digest(
                            claim_authority["layoutDigest"],
                            field="claimFence.authority.layoutDigest",
                        ),
                        "codeDigest": _digest(
                            claim_authority["codeDigest"],
                            field="claimFence.authority.codeDigest",
                        ),
                    }
                )
                self.policy = claim_policy
                self.authority_receipt = _text(
                    body["baseReceipt"], field="claimFence.baseReceipt"
                )
                self.result_id = (
                    None
                    if body.get("lastResult") is None
                    else _text(
                        body["lastResult"], field="claimFence.lastResult"
                    )
                )
                self.node_workers.clear()
                self.node_synced_generation.clear()
                self.node_control_sequence.clear()
                self.contribution_identities.clear()
                self.node_apply_identities.clear()
                self.cohort_digest = None
                self.close_ticks.clear()
        else:
            raise TraceFormatError(f"unsupported canonical event variant: {kind}")

        disposition = self._normalize_disposition(kind, body, result)
        view = self._state_view(result)
        calls = self.native_calls[first_call:]
        return disposition, view, calls

    def _normalize_disposition(
        self,
        kind: str,
        body: Mapping[str, object],
        result: Mapping[str, object],
    ) -> str:
        disposition = _native_name(result)
        if disposition == "corrupt":
            return "corrupt_nonfinite"
        if (
            disposition == "generation_closed"
            and kind == "contribution"
            and _nat(
                _context(body)["generation"],
                field="contribution.context.generation",
            )
            < int(result["committed_generation"])
        ):
            return "catch_up"
        if disposition == "deferred" and kind == "contribution":
            return "retry_next_generation"
        if (
            disposition == "retry_next_generation"
            and str(result["phase"]) == "aborted"
            and kind == "ownerLoss"
        ):
            return "aborted"
        return disposition

    def _state_view(
        self, result: Mapping[str, object]
    ) -> dict[str, object]:
        members = []
        cohort = []
        contributions = []
        result_receipt_workers = []
        for item in result["members"]:
            node = str(item["worker_id"])
            worker = self._worker_for_node(node)
            incarnation = str(item["incarnation"])
            if not item["live"]:
                lifecycle = "expired"
            elif item["recovering"]:
                lifecycle = "recovering"
            else:
                lifecycle = "leased_ready"
            members.append(
                {
                    "worker": worker,
                    "node": node,
                    "incarnation": incarnation,
                    "lifecycle": lifecycle,
                    "syncedGeneration": self.node_synced_generation.get(
                        node, int(result["committed_generation"])
                    ),
                }
            )
            if item["cohort"]:
                cohort.append(
                    {
                        "worker": worker,
                        "node": node,
                        "incarnation": str(item["cohort_incarnation"]),
                    }
                )
            if item["contributed"]:
                identity = self.contribution_identities.get(node)
                if identity is None:
                    raise RuntimeError(
                        f"native contribution lacks translated sequence: {node}"
                    )
                contributions.append(
                    {
                        "worker": worker,
                        "node": node,
                        "incarnation": str(item["cohort_incarnation"]),
                        "sequence": identity.sequence,
                        "exactTokens": int(item["exact_tokens"]),
                        "payloadDigest": str(item["payload_digest"]),
                        "receiptDigest": str(
                            item["contribution_receipt"]
                        ),
                    }
                )
            if item["result_receipt"]:
                result_receipt_workers.append(worker)
        members.sort(key=lambda item: (
            item["worker"], item["node"], item["incarnation"]
        ))
        cohort.sort(key=lambda item: (
            item["worker"], item["node"], item["incarnation"]
        ))
        contributions.sort(key=lambda item: (
            item["worker"], item["node"], item["incarnation"],
            str(item["sequence"]),
        ))
        result_receipt_workers.sort()
        node_applies = [
            {
                "generation": item.generation,
                "worker": item.worker,
                "node": item.node,
                "incarnation": item.incarnation,
                "resultDigest": item.result_digest,
                "receiptDigest": item.receipt_digest,
            }
            for item in self.node_apply_identities.values()
        ]
        node_applies.sort(key=lambda item: (
            str(item["generation"]),
            item["worker"],
            item["node"],
            item["incarnation"],
        ))
        phase = str(result["phase"]).replace("-", "_")
        if phase == "none":
            generation = int(result["committed_generation"])
            attempt = 0
            generation_status = "applied"
        else:
            generation = int(result["active_generation"])
            attempt = max(0, int(result["active_attempt"]) - 1)
            generation_status = phase
        receipt_digest = str(result["commit_receipt"])
        committed_result = str(result["committed_result"])
        view: dict[str, object] = {
            **self.static,
            "generation": generation,
            "attempt": attempt,
            "generationStatus": generation_status,
            "ownerEpoch": max(0, int(result["owner_epoch"]) - 1),
            "ownerReassignments": int(result["owner_reassignments"]),
            "acceptedTokenClock": int(result["accepted_token_clock"]),
            "acceptedCount": len(contributions),
            "acceptedTokens": sum(
                int(item["exactTokens"]) for item in contributions
            ),
            "cohortDigest": (
                self.cohort_digest if phase != "none" else None
            ),
            "authorityReceipt": self.authority_receipt,
            "authorityReceiptDigest": receipt_digest,
            "result": self.result_id,
            "resultDigest": (
                None if committed_result == _ZERO_DIGEST else committed_result
            ),
            "members": members,
            "cohort": cohort,
            "contributions": contributions,
            "resultReceiptWorkers": result_receipt_workers,
            "nodeApplies": node_applies,
        }
        return view


def _native_name(result: Mapping[str, object]) -> str:
    return str(result["disposition"]).replace("-", "_")


def _first_peer_identities(
    trace: Mapping[str, object],
) -> list[tuple[str, str]]:
    identities: dict[str, str] = {}
    steps = trace.get("steps")
    if not isinstance(steps, list):
        raise TraceFormatError("trace steps must be an array")
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            raise TraceFormatError("trace step must be an object")
        kind, body = _event(raw_step)
        if kind == "peerTransition" and str(body.get("toPhase")) == "discover":
            node = _text(body["node"], field="peerTransition.node")
            identities.setdefault(
                node,
                _text(
                    body["incarnation"], field="peerTransition.incarnation"
                ),
            )
    return sorted(identities.items())


def runtime_identity_manifest(
    *,
    repository: str | Path,
    trace: Mapping[str, object],
    attestation: BuildAttestation,
    lean_runner: str | Path,
) -> dict[str, object]:
    root = Path(repository).resolve()
    source_paths = [
        "ndm/native_lean_conformance.py",
        "ndm/native_coordination.py",
        "ndm/native_dataplane.py",
        "src/native_resilient_dataplane/src/client.cpp",
        "src/native_resilient_dataplane/src/rpc_server.cpp",
        "src/native_resilient_dataplane/src/service_core.hpp",
        "src/native_resilient_dataplane/src/ndp.cpp",
        "src/native_resilient_dataplane/src/coordination_kernel.cpp",
        "formal/resilient/ResilientProtocol/Types.lean",
        "formal/resilient/ResilientProtocol/Kernel.lean",
        "formal/resilient/ResilientProtocol/Trace.lean",
        "formal/resilient/ResilientProtocol/Conformance.lean",
        "formal/resilient/trace-schema-v1.json",
    ]
    source_identities = [
        {"path": path, "sha256": sha256_file(root / path)}
        for path in source_paths
    ]
    initial = trace["initialState"]
    assert isinstance(initial, dict)
    policy = initial["policy"]
    authority = initial["authority"]
    assert isinstance(policy, dict) and isinstance(authority, dict)
    return {
        "schema": IDENTITY_SCHEMA,
        "native": {
            "sourceCommit": attestation.source_commit,
            "sourceTreeDirty": attestation.source_tree_dirty,
            "bundleSha256": attestation.bundle_sha256,
            "localAbi": ABI_V1,
            "eventStructBytes": __import__("ctypes").sizeof(
                CoordinationEventV1
            ),
            "memberStructBytes": __import__("ctypes").sizeof(
                CoordinationMemberV1
            ),
            "resultStructBytes": __import__("ctypes").sizeof(
                CoordinationResultV1
            ),
            "artifacts": {
                name: {
                    "path": str(_artifact(attestation, name)),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for name, record in sorted(attestation.artifacts.items())
            },
        },
        "lean": {
            "toolchain": TOOLCHAIN,
            "runner": str(Path(lean_runner).resolve()),
            "runnerSha256": sha256_file(lean_runner),
        },
        "trace": {
            "schema": TRACE_SCHEMA,
            "schemaDigest": TRACE_SCHEMA_DIGEST,
            "viewVersion": VIEW_VERSION,
            "policyName": str(policy["name"]),
            "policySchema": str(policy["schema"]),
            "policyDigest": _digest(policy["digest"], field="policy.digest"),
            "sourceSchema": str(authority["sourceSchema"]),
            "sourceDigest": _digest(
                authority["sourceDigest"], field="authority.sourceDigest"
            ),
        },
        "callPath": [
            "NativeTraceAdapter._step",
            "NativeCoordinationAuthority.step",
            "Client.coordination_step",
            "ndp_coord_step_v1",
            "rpc::Opcode::CoordinationStep",
            "LocalServiceCore::coordination_step",
            "Service::coordination_step",
            "coordination::step",
        ],
        "sourceIdentities": source_identities,
        "scope": {
            "provider": "tcp;ofi_rxm",
            "serviceMode": "test-only persistent compiled service",
            "modelTensors": False,
            "gpu": False,
            "libfabricPeers": 0,
            "externalServices": False,
            "slurm": False,
        },
    }


def audit_production_call_path(repository: str | Path) -> None:
    """Fail if the documented runtime chain no longer exists in source."""
    root = Path(repository).resolve()
    required = {
        "ndm/native_lean_conformance.py": (
            "self.authority.step(kind, **fields)",
        ),
        "ndm/native_coordination.py": (
            "result, trace = self.client.coordination_step(event)",
        ),
        "ndm/native_dataplane.py": (
            "native.library.ndp_coord_step_v1(",
        ),
        "src/native_resilient_dataplane/src/client.cpp": (
            "ndp_coord_step_v1(",
            "Opcode::CoordinationStep",
        ),
        "src/native_resilient_dataplane/src/rpc_server.cpp": (
            "Opcode::CoordinationStep",
            "coordination_step(",
        ),
        "src/native_resilient_dataplane/src/service_core.hpp": (
            "coordination_step(",
        ),
        "src/native_resilient_dataplane/src/ndp.cpp": (
            "int Service::coordination_step(",
            "int LocalServiceCore::coordination_step(",
            "coordination::step(coordination_state_, event)",
        ),
        "src/native_resilient_dataplane/src/coordination_kernel.cpp": (
            "Transition step(const AuthorityState& state",
        ),
    }
    for relative, needles in required.items():
        source = (root / relative).read_text(encoding="utf-8")
        missing = [needle for needle in needles if needle not in source]
        if missing:
            raise RuntimeError(
                f"production coordination call-path audit failed in "
                f"{relative}: {missing}"
            )


def _differences(expected: object, actual: object, path: str = "$") -> list[str]:
    if type(expected) is not type(actual):
        return [
            f"{path}: type {type(actual).__name__} != "
            f"{type(expected).__name__}"
        ]
    if isinstance(expected, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                differences.append(f"{path}.{key}: unexpected")
            elif key not in actual:
                differences.append(f"{path}.{key}: missing")
            else:
                differences.extend(
                    _differences(expected[key], actual[key], f"{path}.{key}")
                )
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        differences = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(
                _differences(left, right, f"{path}[{index}]")
            )
        return differences
    if expected != actual:
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


def _write_replay_prefix(
    trace: Mapping[str, object],
    event_index: int,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    steps = trace["steps"]
    assert isinstance(steps, list)
    prefix = dict(trace)
    prefix["steps"] = steps[: event_index + 1]
    trace_id = str(steps[0]["causality"]["traceId"]) if steps else "empty"
    replay_path = output / f"{trace_id}.first-divergence-{event_index}.json"
    replay_path.write_text(canonical_json(prefix) + "\n", encoding="utf-8")
    report_path = output / f"{trace_id}.first-divergence-{event_index}.report.json"
    return replay_path, report_path


def run_differential_trace(
    *,
    trace_path: str | Path,
    build_manifest: str | Path,
    lean_runner: str | Path,
    repository: str | Path,
    divergence_directory: str | Path,
    fault_event_index: int | None = None,
) -> dict[str, object]:
    """Replay one trace and fail immediately with a replayable prefix."""
    trace = load_canonical_trace(trace_path)
    oracle = run_lean_oracle(trace_path, lean_runner)
    audit_production_call_path(repository)
    attestation = validate_build_manifest(build_manifest)
    identity = runtime_identity_manifest(
        repository=repository,
        trace=trace,
        attestation=attestation,
        lean_runner=lean_runner,
    )
    initial = trace["initialState"]
    assert isinstance(initial, dict)
    initial_authority = initial["authority"]
    policy = initial["policy"]
    assert isinstance(initial_authority, dict) and isinstance(policy, dict)
    base_generation = _nat(
        initial["baseGeneration"], field="initial.baseGeneration"
    )
    base_receipt_digest = _digest(
        initial["baseReceiptDigest"], field="initial.baseReceiptDigest"
    )
    recovered_applies = [
        (node, incarnation, base_receipt_digest)
        for node, incarnation in _first_peer_identities(trace)
        if base_generation != 0
    ]
    service = _artifact(attestation, "service_binary")
    library = _artifact(attestation, "local_library")
    steps = trace["steps"]
    oracle_steps = oracle["steps"]
    assert isinstance(steps, list) and isinstance(oracle_steps, list)
    if len(steps) != len(oracle_steps):
        raise RuntimeError("Lean oracle step cardinality mismatch")
    executed: list[dict[str, object]] = []
    with native_service(service) as (socket_path, token, _process):
        native = NativeLibrary(library)
        client = Client.open(
            library=native,
            role=Role.CONTROLLER,
            run_key=_text(
                initial_authority["run"], field="initial.authority.run"
            ),
            fence_epoch=_nat(
                initial_authority["fence"], field="initial.authority.fence"
            ),
            worker_key="lean-conformance-controller",
            incarnation="lean-conformance-controller-incarnation",
            admission_token=token,
            socket_path=socket_path,
        )
        authority = None
        try:
            authority = NativeCoordinationAuthority(
                client,
                run_id=_text(
                    initial_authority["run"],
                    field="initial.authority.run",
                ),
                fence=_nat(
                    initial_authority["fence"],
                    field="initial.authority.fence",
                ),
                q_min=int(policy["qMin"]),
                t_min=int(policy["tMin"]),
                policy_digest=_digest(
                    policy["digest"], field="initial.policy.digest"
                ),
                committed_generation=base_generation,
                committed_receipt_digest=base_receipt_digest,
                committed_accepted_tokens=int(initial["acceptedTokenClock"]),
                committed_manifest_digest=_digest(
                    initial["baseDigest"], field="initial.baseDigest"
                ),
                committed_result_root=(
                    _digest(initial["baseDigest"], field="initial.baseDigest")
                    if initial.get("lastResult") is not None
                    else ""
                ),
                committed_apply_receipts=recovered_applies,
            )
            adapter = NativeTraceAdapter(
                authority, trace, fault_event_index=fault_event_index
            )
            adapter.last_result = authority.step(
                CoordinationEventKind.QUERY_COMMIT,
                generation=base_generation,
            )
            for raw_step, expected in zip(steps, oracle_steps):
                assert isinstance(raw_step, dict) and isinstance(expected, dict)
                event_index = int(expected["eventIndex"])
                event_id = str(expected["eventId"])
                kind, body = _event(raw_step)
                disposition, native_state, calls = adapter.execute(
                    event_index, kind, body
                )
                native_state_digest = state_digest(native_state)
                differences = []
                if disposition != expected["disposition"]:
                    differences.append(
                        "$.disposition: "
                        f"{disposition!r} != {expected['disposition']!r}"
                    )
                differences.extend(
                    _differences(expected["state"], native_state, "$.state")
                )
                if native_state_digest != expected["stateDigest"]:
                    differences.append(
                        "$.stateDigest: "
                        f"{native_state_digest!r} != "
                        f"{expected['stateDigest']!r}"
                    )
                if adapter.last_result is not None and (
                    "_translation_error" in adapter.last_result
                ):
                    differences.append(
                        "$.translation: "
                        f"{adapter.last_result['_translation_error']!r}"
                    )
                record = {
                    "eventIndex": event_index,
                    "eventId": event_id,
                    "eventKind": kind,
                    "disposition": disposition,
                    "stateDigest": native_state_digest,
                    "nativePostStateDigest": adapter.last_result[
                        "post_state_digest"
                    ],
                    "nativeCalls": list(calls),
                }
                executed.append(record)
                if differences:
                    replay_path, report_path = _write_replay_prefix(
                        trace, event_index, divergence_directory
                    )
                    report = {
                        "schema": RUN_SCHEMA,
                        "verdict": "first_divergence",
                        "trace": str(Path(trace_path).resolve()),
                        "eventIndex": event_index,
                        "eventId": event_id,
                        "eventKind": kind,
                        "differences": differences,
                        "expected": expected,
                        "native": {
                            "disposition": disposition,
                            "state": native_state,
                            "stateDigest": native_state_digest,
                            "calls": list(calls),
                        },
                        "identityManifest": identity,
                        "replayTrace": str(replay_path.resolve()),
                        "replayCommand": (
                            "scripts/conformance/run_native_lean_conformance.py "
                            f"--trace {replay_path.resolve()} "
                            f"--build-manifest {Path(build_manifest).resolve()} "
                            f"--lean-runner {Path(lean_runner).resolve()}"
                        ),
                    }
                    report_path.write_text(
                        canonical_json(report) + "\n", encoding="utf-8"
                    )
                    report["reportPath"] = str(report_path.resolve())
                    raise ConformanceDivergence(report)
        finally:
            if authority is not None:
                authority.close()
            client.close()
    return {
        "schema": RUN_SCHEMA,
        "verdict": "agreement",
        "trace": str(Path(trace_path).resolve()),
        "traceSha256": sha256_file(trace_path),
        "traceId": oracle["traceId"],
        "events": len(executed),
        "finalStateDigest": (
            executed[-1]["stateDigest"] if executed else None
        ),
        "identityManifest": identity,
        "executed": executed,
    }


__all__ = [
    "ConformanceDivergence",
    "IDENTITY_SCHEMA",
    "RUN_SCHEMA",
    "TRACE_SCHEMA",
    "TRACE_SCHEMA_DIGEST",
    "TOOLCHAIN",
    "TraceFormatError",
    "VIEW_VERSION",
    "audit_production_call_path",
    "canonical_json",
    "coordination_digest",
    "load_canonical_trace",
    "run_differential_trace",
    "run_lean_oracle",
    "runtime_identity_manifest",
    "state_digest",
]
