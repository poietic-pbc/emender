"""Live control and distributed shard-owner plane for Compute Pool v1.

The production control server carries endpoint leases and effect metadata
around a pure transition kernel owned by the persistent compiled service.  The
separate Python control implementation is an explicit debug/reference backend.
Dense E97 bytes travel directly from model-free node managers to deterministic
shard owners and back; the allocation holder is never a model broker. All
socket waits and retained byte windows are explicit and bounded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import socket
import socketserver
import threading
import time
from typing import Mapping, Sequence

from ndm.native_artifacts import (
    NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG, validate_backend,
)
from ndm.native_coordination import (
    DEADLINE_EXPIRED, FINITE_CLOSE, NativeCoordinationAuthority,
)
from ndm.native_dataplane import CoordinationEventKind
from ndm.native_transport import decode_endpoint_record
from ndm.resilient_e97_reducer import (
    ExactWeightedShardReducer, ShardChunk, TensorLayout,
)
from ndm.resilient_node_quorum import (
    Contribution, GenerationAdmission, GenerationClosePolicy, GenerationFence,
)
from ndm.resilient_node_transport import recv_frame, send_frame
from ndm.resilient_peer_membership import PeerMembership, PeerState, StageDeadlines


MAX_CONTROL_PAYLOAD = 64 * 1024
_OWNER_CONNECT_TIMEOUT_S = 1.0
_OWNER_MIN_IO_TIMEOUT_S = 1.0
_OWNER_MAX_IO_TIMEOUT_S = 15.0
_OWNER_MIN_BYTES_PER_SECOND = 8 * 1024 * 1024


def _bounded_owner_io_timeout(payload_bytes: int, response_bytes: int, *,
                              remaining_s: float) -> float:
    """Bound one established owner stream operation by size and run deadline.

    Connection establishment remains independently capped at one second.  A
    dense owner frame needs enough time to cross the socket and be reduced by
    the receiver, however: Frontier's approved 64 MiB frame cannot reliably do
    that under the former one-second read/write timeout.  Budget at least an
    8 MiB/s transfer rate, cap any individual stall at 15 seconds, and never
    extend the caller's overall exchange/commit deadline.
    """
    frame_bytes = max(0, int(payload_bytes), int(response_bytes))
    scaled_s = frame_bytes / _OWNER_MIN_BYTES_PER_SECOND
    operation_s = min(_OWNER_MAX_IO_TIMEOUT_S,
                      max(_OWNER_MIN_IO_TIMEOUT_S, scaled_s))
    return max(.001, min(float(remaining_s), operation_s))


@dataclass(frozen=True)
class PoolStageSLO:
    """Stage bounds derived from the measured 212-215s production K=40 cadence."""

    generation_expected_s: float
    generation_hard_s: float
    first_heartbeat_s: float
    sync_s: float
    freeze_s: float
    transport_s: float
    apply_s: float
    shutdown_s: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("pool stage SLOs must be finite and positive")
        if self.generation_hard_s < self.generation_expected_s:
            raise ValueError("generation hard bound cannot precede its expected bound")

    @classmethod
    def production(cls) -> "PoolStageSLO":
        # Downstream's first 20m gate requires READY<=180s, K40<=420s,
        # exchange+commit<=180s, and the first atomic commit<=720s.  The bulk
        # sub-stages total 165s and retain the measured 215s expected cadence.
        return cls(215.0, 720.0, 180.0, 180.0, 15.0, 90.0, 60.0, 300.0)

    @property
    def training_hard_s(self) -> float:
        """The explicit downstream K40 bound, capped by fixture hard bounds."""
        return min(420.0, self.generation_hard_s)

    @property
    def ready_lease_s(self) -> float:
        """Span every independently bounded phase before next READY.

        ``generation_hard_s`` covers admission through the immutable commit.
        Candidate preparation and the safe-boundary rendezvous each retain a
        separate 420-second bound, followed by the 60-second atomic apply.
        Expiring an unchanged peer between its node-apply receipt and its next
        READY advertisement would turn harmless node skew into a false stale
        incarnation.
        """
        return (
            self.generation_hard_s
            + 2 * self.training_hard_s
            + self.apply_s
        )


@dataclass(frozen=True)
class OwnerEndpoint:
    worker_id: str
    incarnation: str
    host: str
    port: int
    backend: str = PYTHON_TCP_DEBUG
    endpoint_record: str = ""
    provider: str = ""
    endpoint_epoch: int = 0
    expires_unix_ns: int = 0
    artifact_bundle_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.worker_id or not self.incarnation or not self.host:
            raise ValueError("owner endpoint identity/address is invalid")
        validate_backend(self.backend, production=self.backend == NATIVE_CXI,
                         full_layout=False)
        if self.backend == PYTHON_TCP_DEBUG:
            if not 0 < self.port < 65536 or any((
                    self.endpoint_record, self.provider, self.endpoint_epoch,
                    self.expires_unix_ns, self.artifact_bundle_sha256)):
                raise ValueError("Python TCP debug endpoint must not claim native evidence")
            return
        if self.port != 0 or not self.endpoint_record or not self.provider:
            raise ValueError("native endpoint requires an opaque record and no TCP port")
        try:
            decoded = decode_endpoint_record(bytes.fromhex(self.endpoint_record))
        except (ValueError, TypeError) as error:
            raise ValueError("native endpoint record is invalid") from error
        expected_worker = hashlib.sha256(self.worker_id.encode()).digest()[:16]
        expected_incarnation = hashlib.sha256(self.incarnation.encode()).digest()[:16]
        if (decoded.worker_key != expected_worker
                or decoded.incarnation != expected_incarnation
                or decoded.endpoint_epoch != self.endpoint_epoch
                or decoded.expires_unix_ns != self.expires_unix_ns
                or decoded.provider != self.provider
                or decoded.expires_unix_ns <= time.time_ns()):
            raise ValueError("native endpoint record identity/provider/expiry mismatch")
        if self.backend == NATIVE_CXI and self.provider != "cxi":
            raise ValueError("production endpoint did not attest exact provider cxi")
        if len(self.artifact_bundle_sha256) != 64:
            raise ValueError("native endpoint must bind the compiled artifact bundle")


@dataclass(frozen=True)
class PoolControlConfig:
    run_id: str
    fence: int
    q_min: int
    t_min: int
    ready_fraction: float | None
    base_digest: str
    policy_digest: str
    layout_digest: str
    code_digest: str
    slo: PoolStageSLO
    dataplane_backend: str = PYTHON_TCP_DEBUG
    production: bool = False
    full_layout: bool = False
    artifact_bundle_sha256: str = ""
    scale_close_offset_s: float | None = None
    scale_stable_diversity_floor: int | None = None
    scale_per_ready_worker_token_floor: int | None = None
    scale_closure_digest: str = ""
    committed_generation: int = 0
    committed_receipt_digest: str = ""
    committed_accepted_tokens: int = 0
    committed_manifest_digest: str = ""
    committed_result_root: str = ""
    committed_apply_receipts: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        GenerationClosePolicy(self.q_min, self.t_min, self.ready_fraction)
        validate_backend(self.dataplane_backend, production=self.production,
                         full_layout=self.full_layout)
        if self.fence <= 0 or not all((self.run_id, self.base_digest,
                                      self.policy_digest, self.layout_digest,
                                      self.code_digest)):
            raise ValueError("complete fenced control identity is required")
        if self.dataplane_backend != PYTHON_TCP_DEBUG:
            if len(self.artifact_bundle_sha256) != 64:
                raise ValueError("native pool requires an attested artifact bundle digest")
        elif self.artifact_bundle_sha256:
            raise ValueError("Python TCP debug pool cannot claim a native artifact bundle")
        if (
            self.committed_generation < 0
            or self.committed_accepted_tokens < 0
            or (
                self.dataplane_backend != PYTHON_TCP_DEBUG
                and self.committed_generation > 0
                and self.committed_accepted_tokens == 0
            )
            or (
                self.committed_generation > 0
                and any((
                    self.committed_receipt_digest,
                    self.committed_manifest_digest,
                    self.committed_result_root,
                ))
                and any(
                    len(value) != 64 for value in (
                        self.committed_receipt_digest,
                        self.committed_manifest_digest,
                        self.committed_result_root,
                    )
                )
            )
            or (
                self.committed_generation == 0
                and any((
                    self.committed_receipt_digest,
                    self.committed_manifest_digest,
                    self.committed_result_root,
                ))
            )
            or any(
                len(item) not in {2, 3}
                or not item[0]
                or (len(item) == 3 and not item[1])
                or len(item[-1]) != 64
                for item in self.committed_apply_receipts
            )
            or (
                self.dataplane_backend != PYTHON_TCP_DEBUG
                and any(len(item) != 3
                        for item in self.committed_apply_receipts)
            )
            or len({
                item[0] for item in self.committed_apply_receipts
            }) != len(self.committed_apply_receipts)
        ):
            raise ValueError("initial peer commit authority is invalid")
        scale = (
            self.scale_close_offset_s,
            self.scale_stable_diversity_floor,
            self.scale_per_ready_worker_token_floor,
            self.scale_closure_digest,
        )
        if any(item not in (None, "") for item in scale):
            if (
                self.scale_close_offset_s is None
                or not math.isfinite(self.scale_close_offset_s)
                or not 0 < self.scale_close_offset_s <= self.slo.training_hard_s
                or self.scale_stable_diversity_floor is None
                or self.scale_stable_diversity_floor < 2
                or self.scale_per_ready_worker_token_floor is None
                or self.scale_per_ready_worker_token_floor <= 0
                or len(self.scale_closure_digest) != 64
            ):
                raise ValueError("complete bounded V21S17 scale closure is required")


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            header, payload = recv_frame(self.rfile, max_payload_bytes=MAX_CONTROL_PAYLOAD)
            result = self.server.dispatch(header, payload)
            send_frame(self.wfile, {"op": "result", "result": result})
        except Exception as error:
            send_frame(self.wfile, {"op": "error", "error": str(error)})


class PoolControlServer(socketserver.ThreadingTCPServer):
    """Explicit Python-TCP debug/reference coordinator.

    Production native managers use :class:`NativePoolControlServer`; keeping
    this deterministic reference path is useful for local fixtures but it is
    never the authoritative native service.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: PoolControlConfig, *,
                 evidence_root: str | Path):
        self.config = config
        self.strict_commit_authority = (
            config.dataplane_backend != PYTHON_TCP_DEBUG)
        self.evidence_root = Path(evidence_root)
        self.membership = PeerMembership(StageDeadlines(
            config.slo.first_heartbeat_s,
            config.slo.first_heartbeat_s,
            config.slo.sync_s,
            config.slo.ready_lease_s,
            config.slo.shutdown_s,
        ))
        self.endpoints: dict[tuple[str, str], OwnerEndpoint] = {}
        self.seen_incarnations: dict[str, set[str]] = {}
        self.snapshots: dict[tuple[int, int], dict[str, object]] = {}
        self.admissions: dict[tuple[int, int], GenerationAdmission] = {}
        self.result_roots: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
        self.owner_results: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
        self.route_readiness: dict[
            tuple[int, int, tuple[str, str]], dict[str, tuple[str, str]]
        ] = {}
        self.committed_generation = config.committed_generation
        self.committed_receipt_digest = config.committed_receipt_digest
        self.committed_accepted_tokens = config.committed_accepted_tokens
        self.committed_result_root = config.committed_result_root
        self.committed_manifest_digest = config.committed_manifest_digest
        self.commit_history: dict[int, dict[str, object]] = {}
        self.node_applies: dict[tuple[int, str], dict[str, object]] = {}
        self.recovered_apply_receipts = {
            item[0]: item[-1] for item in config.committed_apply_receipts
        }
        self.pending_recoveries: dict[str, str] = {}
        self.apply_required_generation: int | None = None
        self._lock = threading.RLock()
        super().__init__(address, _ControlHandler)

    def dispatch(self, request: Mapping[str, object], payload: bytes) -> dict[str, object]:
        op = str(request.get("op"))
        if request.get("run_id") != self.config.run_id:
            raise ValueError("control run identity mismatch")
        if int(request.get("fence", -1)) != self.config.fence:
            raise ValueError("control allocation fence mismatch")
        with self._lock:
            if op == "ready":
                return self._ready(request)
            if op == "open":
                return self._open(request)
            if op == "contribute":
                return self._contribute(request, payload)
            if op == "close":
                return self._close(request)
            if op == "route_ready":
                return self._route_ready(request)
            if op == "owner_result":
                return self._owner_result(request)
            if op == "result_root":
                return self._result_root(request)
            if op == "commit":
                return self._commit(request)
            if op == "commit_state":
                return self._commit_state(request)
            if op == "recover":
                return self._recover(request)
            if op == "node_apply":
                return self._node_apply(request)
            if op == "expire":
                return self._expire(request)
            raise ValueError("unsupported pool control operation")

    def _ready(self, request: Mapping[str, object]) -> dict[str, object]:
        endpoint = OwnerEndpoint(
            str(request["worker_id"]), str(request["incarnation"]),
            str(request["host"]), int(request["port"]),
            str(request.get("backend", PYTHON_TCP_DEBUG)),
            str(request.get("endpoint_record", "")),
            str(request.get("provider", "")),
            int(request.get("endpoint_epoch", 0)),
            int(request.get("expires_unix_ns", 0)),
            str(request.get("artifact_bundle_sha256", "")))
        if endpoint.backend != self.config.dataplane_backend:
            raise ValueError("READY endpoint backend differs from pool policy")
        if endpoint.backend != PYTHON_TCP_DEBUG:
            decoded = decode_endpoint_record(bytes.fromhex(endpoint.endpoint_record))
            expected_run = hashlib.sha256(self.config.run_id.encode()).digest()[:16]
            if (decoded.run_key != expected_run
                    or decoded.fence_epoch != self.config.fence
                    or endpoint.artifact_bundle_sha256
                    != self.config.artifact_bundle_sha256):
                raise ValueError("READY native endpoint run/fence/artifact mismatch")
        generation = int(request["generation"])
        if self.strict_commit_authority and generation != self.committed_generation:
            raise ValueError(
                "READY generation differs from native peer commit authority")
        if self.strict_commit_authority and self.apply_required_generation == generation:
            apply_digest = str(request.get("apply_receipt_digest", ""))
            applied = self.node_applies.get((generation, endpoint.worker_id))
            if (
                applied is None
                or applied.get("receipt_digest") != apply_digest
                or applied.get("incarnation") != endpoint.incarnation
            ):
                raise ValueError(
                    "READY requires this incarnation's atomic node-apply receipt")
        current = self.membership.records.get(endpoint.worker_id)
        if (current is not None and current.incarnation == endpoint.incarnation
                and current.state is PeerState.READY):
            if current.base_generation != generation:
                self.membership.catch_up(endpoint.worker_id, endpoint.incarnation,
                                         committed_generation=generation)
                self.membership.ready(endpoint.worker_id, endpoint.incarnation,
                                      base_generation=generation)
            else:
                self.membership.renew(endpoint.worker_id, endpoint.incarnation,
                                      base_generation=generation)
        else:
            if endpoint.incarnation in self.seen_incarnations.get(endpoint.worker_id, set()):
                raise ValueError("superseded peer incarnation rejected")
            record = self.membership.discover(endpoint.worker_id,
                                              incarnation=endpoint.incarnation)
            self.membership.begin_boot(record.worker_id, record.incarnation)
            self.membership.begin_sync(record.worker_id, record.incarnation)
            self.membership.ready(record.worker_id, record.incarnation,
                                  base_generation=generation)
            self.seen_incarnations.setdefault(endpoint.worker_id, set()).add(
                endpoint.incarnation)
        self.endpoints[(endpoint.worker_id, endpoint.incarnation)] = endpoint
        self.pending_recoveries.pop(endpoint.worker_id, None)
        return {"status": "READY", "worker_id": endpoint.worker_id,
                "incarnation": endpoint.incarnation, "generation": generation}

    def _open(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        if self.strict_commit_authority and generation != self.committed_generation:
            raise ValueError(
                "generation open does not extend native peer commit authority")
        key = (generation, attempt)
        prior = self.snapshots.get(key)
        if prior is not None:
            return prior
        snapshot = self.membership.active_snapshot(generation)
        if snapshot.size < self.config.q_min:
            raise RuntimeError("READY floor is not yet available")
        peers = []
        for peer in snapshot.peers:
            endpoint = self.endpoints[(peer.worker_id, peer.incarnation)]
            peers.append({**asdict(endpoint), "lease_expiry": peer.lease_expiry})
        value: dict[str, object] = {
            "run_id": self.config.run_id, "fence": self.config.fence,
            "generation": generation, "attempt": attempt,
            "observed_at": snapshot.observed_at, "peers": peers,
        }
        # The READY snapshot opens before local K40 work. At two nodes, close
        # is immediate once the reviewed token/diversity floor is present. At
        # scale, V21S17 pins a finite empirical close over this exact leased
        # snapshot so every admissible pre-close arrival is included.
        opened = time.monotonic()
        deadline = opened + self.config.slo.training_hard_s
        scale = self.config.scale_close_offset_s is not None
        close_not_before = (
            opened + float(self.config.scale_close_offset_s)
            if scale else None)
        close_policy = GenerationClosePolicy(
            int(self.config.scale_stable_diversity_floor)
            if scale else self.config.q_min,
            int(self.config.scale_per_ready_worker_token_floor) * snapshot.size
            if scale else self.config.t_min,
            None if scale else self.config.ready_fraction,
        )
        self.admissions[key] = GenerationAdmission.open(
            GenerationFence(self.config.run_id, generation, attempt, self.config.fence),
            ready_snapshot=tuple((peer.worker_id, peer.incarnation) for peer in snapshot.peers),
            policy=close_policy,
            deadline=deadline, base_digest=self.config.base_digest,
            policy_digest=self.config.policy_digest,
            layout_digest=self.config.layout_digest, code_digest=self.config.code_digest,
            close_not_before=close_not_before,
            evidence_path=self.evidence_root / f"generation-{generation:08d}.jsonl")
        value["deadline_after_s"] = self.config.slo.training_hard_s
        if scale:
            value["scale_closure"] = {
                "schema": "emender-v21s17-runtime-close-v1",
                "close_after_s": self.config.scale_close_offset_s,
                "stable_diversity_floor":
                    self.config.scale_stable_diversity_floor,
                "per_ready_worker_token_floor":
                    self.config.scale_per_ready_worker_token_floor,
                "closure_digest": self.config.scale_closure_digest,
                "close_on_q_min": False,
                "uses_launched_ranks": False,
                "wait_for_all_ready": False,
            }
        self.snapshots[key] = value
        return value

    def _admission_for_identity(self, generation: int, attempt: int) -> GenerationAdmission:
        exact = self.admissions.get((generation, attempt))
        if exact is not None:
            return exact
        # Feed a wrong-attempt identity to the live generation admission so it
        # produces the normative stale-fence receipt rather than an RPC error.
        candidates = [value for (item_generation, _), value in self.admissions.items()
                      if item_generation == generation]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("generation is not open")

    def _contribute(self, request: Mapping[str, object], payload: bytes) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        # Generation admissions are deliberately volatile.  After peer-control
        # reconstruction, a still-live cohort may hold either the generation
        # immediately preceding authoritative commit or a current-generation
        # snapshot opened by the lost control incarnation.  Older work catches
        # up from immutable authority.  Equal-generation work first rejoins
        # rebuilt READY membership and then retries under a newly opened
        # admission; its already sealed local bytes remain immutable.  Neither
        # response recreates an accumulator or mutates membership/apply state.
        if (generation, attempt) not in self.admissions and (
                generation <= self.committed_generation):
            if (
                generation == self.committed_generation
                and not any(
                    open_generation == generation
                    for open_generation, _ in self.admissions
                )
            ):
                return {
                    "status": "rejoin",
                    "generation": generation,
                    "attempt": attempt,
                    "authoritative_generation": self.committed_generation,
                    "receipt_digest": self.committed_receipt_digest,
                    "manifest_digest": self.committed_manifest_digest,
                    "result_root": self.committed_result_root,
                    "accepted_tokens": self.committed_accepted_tokens,
                    "apply_receipts": [
                        {"worker_id": node_id, "receipt_digest": digest}
                        for node_id, digest in sorted(
                            self.recovered_apply_receipts.items())
                    ],
                    "requires_rejoin": True,
                    "requires_reload": False,
                }
            if generation == self.committed_generation:
                return {"status": "rejected_stale_fence"}
            return {
                "status": "catch_up",
                "generation": generation,
                "attempt": attempt,
                "authoritative_generation": self.committed_generation,
                "receipt_digest": self.committed_receipt_digest,
                "manifest_digest": self.committed_manifest_digest,
                "result_root": self.committed_result_root,
                "accepted_tokens": self.committed_accepted_tokens,
                "requires_reload": True,
            }
        admission = self._admission_for_identity(generation, attempt)
        if "aggregation_weight" in request:
            raise ValueError(
                "v2.1 contributions forbid aggregation_weight; accepted_tokens is exact")
        contribution = Contribution.create(
            GenerationFence(self.config.run_id, generation, attempt, self.config.fence),
            str(request["worker_id"]), str(request["incarnation"]),
            int(request["contribution_seq"]), int(request["accepted_tokens"]), payload,
            base_digest=str(request.get("base_digest", self.config.base_digest)),
            policy_digest=str(request.get("policy_digest", self.config.policy_digest)),
            layout_digest=str(request.get("layout_digest", self.config.layout_digest)),
            code_digest=str(request.get("code_digest", self.config.code_digest)))
        receipt = admission.admit(contribution, now=time.monotonic())
        return {"identity": asdict(receipt.identity), "status": receipt.status,
                "content_digest": receipt.content_digest}

    def _close(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        admission = self.admissions[(generation, attempt)]
        now = time.monotonic()
        try:
            close = admission.close(
                now=now, run_deadline=admission.deadline + self.config.slo.shutdown_s)
        except RuntimeError as error:
            if "still open" not in str(error):
                raise
            return {"status": "open", "accepted_tokens": sum(
                item.accepted_tokens for item in admission._accepted.values())}
        return {
            "status": close.status, "reason": close.reason,
            "accepted_tokens": close.accepted_tokens,
            "required_contributions": close.required_contributions,
            "ready_snapshot": [list(item) for item in close.ready_snapshot],
            "frozen_identities": [asdict(item) for item in close.frozen_identities],
            "accepted_payloads": {
                item.identity.worker_id: item.payload.decode("ascii")
                for item in admission._accepted.values()
                if item.identity in close.frozen_identities
            },
            "exact_tokens_by_worker": {
                item.identity.worker_id: item.accepted_tokens
                for item in admission._accepted.values()
                if item.identity in close.frozen_identities
            },
        }

    def _expire(self, request: Mapping[str, object]) -> dict[str, object]:
        worker_id, incarnation = str(request["worker_id"]), str(request["incarnation"])
        record = self.membership.records.get(worker_id)
        if record is not None and record.incarnation == incarnation and record.state is PeerState.READY:
            self.membership.drain(worker_id, incarnation)
        return {"status": "DRAINING", "worker_id": worker_id}

    def _commit(self, request: Mapping[str, object]) -> dict[str, object]:
        """Advance volatile peer authority only after an immutable receipt."""
        result_generation = int(request["result_generation"])
        source_generation = result_generation - 1
        attempt = int(request["attempt"])
        if result_generation <= 0:
            raise ValueError("peer commit generation must be positive")
        record = {
            "result_generation": result_generation,
            "source_generation": source_generation,
            "attempt": attempt,
            "receipt_digest": str(request["receipt_digest"]),
            "previous_receipt_digest": str(
                request.get("previous_receipt_digest", "")),
            "manifest_digest": str(request["manifest_digest"]),
            "result_root": str(request["result_root"]),
            "accepted_tokens": int(request["accepted_tokens"]),
        }
        if any(
            len(str(record[field])) != 64
            for field in ("receipt_digest", "manifest_digest", "result_root")
        ):
            raise ValueError("peer commit digest identity is invalid")
        prior = self.commit_history.get(result_generation)
        if prior is not None:
            if prior != record:
                raise ValueError("conflicting peer commit replay")
            return {"status": "committed", **prior}
        if record["previous_receipt_digest"] != self.committed_receipt_digest:
            raise ValueError("peer commit does not extend current receipt")
        if (
            source_generation != self.committed_generation
            or record["accepted_tokens"] < self.committed_accepted_tokens
        ):
            raise ValueError("peer commit generation/token clock is stale")
        admission = self.admissions.get((source_generation, attempt))
        if (
            admission is None
            or admission.close_result is None
            or admission.close_result.status != "commit_ready"
        ):
            raise RuntimeError("peer commit cannot precede frozen generation")
        roots = self.result_roots.get((source_generation, attempt), {})
        accepted = {
            item.worker_id for item in admission.close_result.frozen_identities
        }
        if set(roots) != accepted:
            raise RuntimeError("peer commit cannot precede native root agreement")
        agreed_roots = {str(item["result_root"]) for item in roots.values()}
        if agreed_roots != {record["result_root"]}:
            raise ValueError("peer commit receipt differs from native result")
        self.committed_generation = result_generation
        self.committed_receipt_digest = str(record["receipt_digest"])
        self.committed_manifest_digest = str(record["manifest_digest"])
        self.committed_result_root = str(record["result_root"])
        self.committed_accepted_tokens = int(record["accepted_tokens"])
        self.commit_history[result_generation] = record
        # The configured receipts describe only the allocation's restored base
        # commit.  Once this live authority advances, recovery must expose
        # receipts for the new commit rather than silently retaining that base.
        self.recovered_apply_receipts.clear()
        self.apply_required_generation = result_generation
        return {"status": "committed", **record}

    def _commit_state(self, request: Mapping[str, object]) -> dict[str, object]:
        """Expose volatile commit state for bounded point-to-point discovery."""
        generation = int(request["result_generation"])
        if generation <= 0:
            raise ValueError("peer commit generation must be positive")
        record = self.commit_history.get(generation)
        if record is not None:
            return {"status": "committed", **record}
        if generation <= self.committed_generation:
            raise ValueError("requested peer commit is absent below current state")
        if generation != self.committed_generation + 1:
            raise ValueError("requested peer commit is discontinuous")
        return {
            "status": "pending",
            "result_generation": generation,
            "current_generation": self.committed_generation,
        }

    def _recover(self, request: Mapping[str, object]) -> dict[str, object]:
        """Return the bounded volatile state a peer must match before READY."""
        worker_id = str(request.get("worker_id", ""))
        incarnation = str(request.get("incarnation", ""))
        if not worker_id or not incarnation:
            raise ValueError("recovery handshake requires a peer incarnation")
        known_generation = int(
            request.get("known_generation", self.committed_generation))
        known_receipt = str(
            request.get("known_receipt_digest", self.committed_receipt_digest))
        if (
            known_generation != self.committed_generation
            or known_receipt != self.committed_receipt_digest
        ):
            raise ValueError(
                "peer recovery state differs from commit authority")
        current = self.membership.records.get(worker_id)
        if (
            incarnation in self.seen_incarnations.get(worker_id, set())
            and (current is None or current.incarnation != incarnation)
        ):
            raise ValueError("stale incarnation recovery handshake rejected")
        pending = self.pending_recoveries.get(worker_id)
        if pending is not None and pending != incarnation:
            # A recovery incarnation has no authority before READY.  If its
            # local manager dies during the bounded handshake, the supervisor
            # must be able to advance to one replacement without leaving the
            # stable worker permanently fenced.  Superseding the pending token
            # makes every later command from it stale; it never entered READY,
            # an accepted set, an apply transaction, or a commit.
            self.seen_incarnations.setdefault(worker_id, set()).add(pending)
        self.pending_recoveries[worker_id] = incarnation
        return {
            "status": "recover",
            "generation": self.committed_generation,
            "receipt_digest": self.committed_receipt_digest,
            "manifest_digest": self.committed_manifest_digest,
            "result_root": self.committed_result_root,
            "accepted_tokens": self.committed_accepted_tokens,
            "apply_receipts": [
                {"worker_id": node_id, "receipt_digest": digest}
                for node_id, digest in sorted(
                    self.recovered_apply_receipts.items())
            ],
            "requires_node_apply":
                self.apply_required_generation == self.committed_generation,
        }

    def _node_apply(self, request: Mapping[str, object]) -> dict[str, object]:
        """Admit one immutable all-eight-trainer receipt for current commit."""
        generation = int(request["generation"])
        worker_id = str(request["worker_id"])
        incarnation = str(request["incarnation"])
        receipt_digest = str(request["receipt_digest"])
        commit_digest = str(request["commit_receipt_digest"])
        trainer_count = int(request["trainer_count"])
        current = self.membership.records.get(worker_id)
        admitted_incarnation = (
            current is not None and current.incarnation == incarnation)
        recovering_incarnation = (
            self.pending_recoveries.get(worker_id) == incarnation)
        if (
            generation != self.committed_generation
            or commit_digest != self.committed_receipt_digest
            or len(receipt_digest) != 64
            or trainer_count != 8
            or not (admitted_incarnation or recovering_incarnation)
        ):
            raise ValueError("atomic node-apply peer receipt is invalid")
        record = {
            "incarnation": incarnation,
            "receipt_digest": receipt_digest,
            "commit_receipt_digest": commit_digest,
            "trainer_count": trainer_count,
        }
        key = (generation, worker_id)
        prior = self.node_applies.get(key)
        if prior is not None and prior != record:
            raise ValueError("conflicting atomic node-apply peer receipt")
        self.node_applies[key] = record
        self.recovered_apply_receipts[worker_id] = receipt_digest
        commit = self.commit_history.get(generation)
        admission = (
            None
            if commit is None
            else self.admissions.get(
                (int(commit["source_generation"]), int(commit["attempt"])))
        )
        if (
            admission is not None
            and admission.close_result is not None
            and admission.close_result.status == "commit_ready"
        ):
            required_workers = {
                item.worker_id
                for item in admission.close_result.frozen_identities
            }
            applied_workers = {
                item_worker
                for item_generation, item_worker in self.node_applies
                if item_generation == generation
            }
            if required_workers and required_workers <= applied_workers:
                self.apply_required_generation = None
        return {
            "status": "node_applied",
            "generation": generation,
            "worker_id": worker_id,
            "incarnation": incarnation,
            "receipt_digest": receipt_digest,
        }

    def _route_ready(self, request: Mapping[str, object]) -> dict[str, object]:
        """Release one native peer pair only after both routes are installed.

        A libfabric endpoint can receive before Python has installed the
        current-fence address-vector entry.  Such frames are correctly
        rejected, but an initial credit rejection can strand an otherwise
        healthy exact transfer.  This is deliberately a reciprocal pairwise
        control-plane rendezvous, not a launched-rank or accepted-set barrier:
        unrelated peers never participate and can progress independently.
        """
        generation, attempt = int(request["generation"]), int(request["attempt"])
        admission = self.admissions[(generation, attempt)]
        close = admission.close_result
        if close is None or close.status != "commit_ready":
            raise RuntimeError("route readiness cannot precede the frozen accepted set")
        worker_id = str(request["worker_id"])
        incarnation = str(request["incarnation"])
        peer_worker_id = str(request["peer_worker_id"])
        peer_incarnation = str(request["peer_incarnation"])
        if worker_id == peer_worker_id:
            raise ValueError("native route readiness requires a remote peer")
        accepted = {(item.worker_id, item.incarnation)
                    for item in close.frozen_identities}
        if ((worker_id, incarnation) not in accepted
                or (peer_worker_id, peer_incarnation) not in accepted):
            raise ValueError("route readiness reporter is outside the frozen accepted set")
        pair = tuple(sorted((worker_id, peer_worker_id)))
        reports = self.route_readiness.setdefault(
            (generation, attempt, pair), {})
        report = (incarnation, peer_incarnation)
        prior = reports.get(worker_id)
        if prior is not None and prior != report:
            raise ValueError("conflicting native route readiness replay")
        reports[worker_id] = report
        if set(reports) != set(pair):
            return {"status": "waiting", "workers": sorted(reports),
                    "required": list(pair)}
        # Both directional reports must name the same exact frozen
        # incarnations; this prevents a superseded endpoint from releasing a
        # current route.
        left, right = pair
        if (reports[left] != (dict(accepted)[left], dict(accepted)[right])
                or reports[right] != (dict(accepted)[right], dict(accepted)[left])):
            raise ValueError("native route readiness incarnation mismatch")
        return {"status": "ready", "workers": list(pair)}

    def _result_root(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        admission = self.admissions[(generation, attempt)]
        if admission.close_result is None or admission.close_result.status != "commit_ready":
            raise RuntimeError("result root cannot precede the frozen accepted set")
        worker_id, incarnation = str(request["worker_id"]), str(request["incarnation"])
        accepted = {(item.worker_id, item.incarnation)
                    for item in admission.close_result.frozen_identities}
        if (worker_id, incarnation) not in accepted:
            raise ValueError("result root reporter is outside the frozen accepted set")
        root = str(request["result_root"])
        weight, result_bytes = int(request["global_weight"]), int(request["result_bytes"])
        if len(root) != 64 or root == "00" * 32 or weight <= 0 or result_bytes <= 0:
            raise ValueError("result root metadata is invalid")
        values = self.result_roots.setdefault((generation, attempt), {})
        record = {"incarnation": incarnation, "result_root": root,
                  "global_weight": weight, "result_bytes": result_bytes}
        prior = values.get(worker_id)
        if prior is not None and prior != record:
            raise ValueError("conflicting result root replay")
        values[worker_id] = record
        accepted_workers = {item[0] for item in accepted}
        if set(values) != accepted_workers:
            return {"status": "waiting", "reported": len(values),
                    "required": len(accepted_workers)}
        identities = {(item["result_root"], item["global_weight"], item["result_bytes"])
                      for item in values.values()}
        if len(identities) != 1:
            raise ValueError("native result-root validation mismatch")
        validated_root, validated_weight, validated_bytes = identities.pop()
        frozen = set(admission.close_result.frozen_identities)
        exact_tokens = sum(
            item.accepted_tokens for item in admission._accepted.values()
            if item.identity in frozen)
        if exact_tokens <= 0 or validated_weight != exact_tokens:
            raise ValueError("native result-root token accounting mismatch")
        return {"status": "validated", "result_root": validated_root,
                "global_weight": validated_weight, "result_bytes": validated_bytes,
                "workers": sorted(values)}

    def _owner_result(self, request: Mapping[str, object]) -> dict[str, object]:
        """Collect distinct immutable shard-owner roots before redistribution."""
        generation, attempt = int(request["generation"]), int(request["attempt"])
        admission = self.admissions[(generation, attempt)]
        close = admission.close_result
        if close is None or close.status != "commit_ready":
            raise RuntimeError("owner result cannot precede the frozen accepted set")
        worker_id, incarnation = str(request["worker_id"]), str(request["incarnation"])
        accepted = {(item.worker_id, item.incarnation) for item in close.frozen_identities}
        if (worker_id, incarnation) not in accepted:
            raise ValueError("owner result reporter is outside the frozen accepted set")
        frozen = set(close.frozen_identities)
        exact_tokens = sum(
            item.accepted_tokens for item in admission._accepted.values()
            if item.identity in frozen)
        if exact_tokens <= 0:
            raise ValueError("frozen exact-token total is invalid")
        root = str(request["result_root"])
        layout_digest = str(request["layout_digest"])
        weight, result_bytes = int(request["global_weight"]), int(request["result_bytes"])
        if (len(root) != 64 or root == "00" * 32
                or len(layout_digest) != 64 or layout_digest == "00" * 32
                or weight != exact_tokens
                or result_bytes <= 0):
            raise ValueError("owner result metadata is invalid")
        values = self.owner_results.setdefault((generation, attempt), {})
        record = {"incarnation": incarnation, "result_root": root,
                  "layout_digest": layout_digest,
                  "global_weight": weight, "result_bytes": result_bytes}
        prior = values.get(worker_id)
        if prior is not None and prior != record:
            raise ValueError("conflicting owner result replay")
        values[worker_id] = record
        accepted_workers = {item[0] for item in accepted}
        if set(values) != accepted_workers:
            return {"status": "waiting", "reported": len(values),
                    "required": len(accepted_workers)}
        return {"status": "ready", "global_weight": exact_tokens,
                "roots": {worker: str(values[worker]["result_root"])
                          for worker in sorted(values)},
                "owners": {worker: {
                    "result_root": str(values[worker]["result_root"]),
                    "layout_digest": str(values[worker]["layout_digest"]),
                    "result_bytes": int(values[worker]["result_bytes"]),
                } for worker in sorted(values)}}


class NativePoolControlServer(socketserver.ThreadingTCPServer):
    """Effect executor for the live service's pure coordination kernel.

    This class intentionally retains only external data: endpoint records,
    clocks, payload presentation, and pairwise transport rendezvous.  It has no
    independently mutable membership, generation, commit, apply, or recovery
    authority.  Every such decision is a serialized typed event applied by
    ``NativeCoordinationAuthority`` in the persistent compiled service.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: PoolControlConfig, *,
                 authority: NativeCoordinationAuthority,
                 evidence_root: str | Path):
        if config.dataplane_backend == PYTHON_TCP_DEBUG:
            raise ValueError("native control server requires a compiled backend")
        if (authority.run_id != config.run_id
                or authority.fence != config.fence):
            raise ValueError("native control authority/config identity mismatch")
        self.config = config
        self.authority = authority
        self.evidence_root = Path(evidence_root)
        self.endpoints: dict[tuple[str, str], OwnerEndpoint] = {}
        self.endpoint_leases: dict[tuple[str, str], float] = {}
        self.endpoint_sequences: dict[tuple[str, str], int] = {}
        self.snapshots: dict[tuple[int, int], dict[str, object]] = {}
        self.opened: dict[tuple[int, int], tuple[float, float, float | None]] = {}
        self.accepted_payloads: dict[
            tuple[int, int, str], dict[str, object]
        ] = {}
        self.owner_results: dict[
            tuple[int, int], dict[str, dict[str, object]]
        ] = {}
        self.route_readiness: dict[
            tuple[int, int, tuple[str, str]], dict[str, tuple[str, str]]
        ] = {}
        self.commit_records: dict[int, dict[str, object]] = {}
        self.apply_receipts = {
            item[0]: {
                "incarnation": item[1] if len(item) == 3 else "",
                "receipt_digest": item[-1],
            }
            for item in config.committed_apply_receipts
        }
        self._lock = threading.RLock()
        super().__init__(address, _ControlHandler)

    @staticmethod
    def _member(result: Mapping[str, object], worker_id: str
                ) -> Mapping[str, object] | None:
        return next((
            item for item in result["members"]  # type: ignore[index]
            if item["worker_id"] == worker_id  # type: ignore[index]
        ), None)

    @staticmethod
    def _fatal(result: Mapping[str, object]) -> None:
        if result["disposition"] == "fatal-invariant":
            raise RuntimeError(
                "native coordination authority invariant violation")

    def _step(self, request: Mapping[str, object],
              kind: CoordinationEventKind, **fields: object
              ) -> dict[str, object]:
        result = self.authority.step(
            kind,
            run_id=str(request.get("run_id", "")),
            fence=int(request.get("fence", 0)),
            **fields,
        )
        self._fatal(result)
        return result

    def dispatch(self, request: Mapping[str, object],
                 payload: bytes) -> dict[str, object]:
        op = str(request.get("op"))
        with self._lock:
            if op == "ready":
                return self._ready(request)
            if op == "open":
                return self._open(request)
            if op == "contribute":
                return self._contribute(request, payload)
            if op == "close":
                return self._close(request)
            if op == "route_ready":
                return self._route_ready(request)
            if op == "owner_result":
                return self._owner_result(request)
            if op == "result_root":
                return self._result_root(request)
            if op == "commit":
                return self._commit(request)
            if op == "commit_state":
                return self._commit_state(request)
            if op == "recover":
                return self._recover(request)
            if op == "node_apply":
                return self._node_apply(request)
            if op == "expire":
                return self._expire(request)
            if op == "owner_lost":
                return self._owner_lost(request)
            raise ValueError("unsupported native pool control operation")

    def _ready(self, request: Mapping[str, object]) -> dict[str, object]:
        endpoint = OwnerEndpoint(
            str(request["worker_id"]), str(request["incarnation"]),
            str(request["host"]), int(request["port"]),
            str(request.get("backend", PYTHON_TCP_DEBUG)),
            str(request.get("endpoint_record", "")),
            str(request.get("provider", "")),
            int(request.get("endpoint_epoch", 0)),
            int(request.get("expires_unix_ns", 0)),
            str(request.get("artifact_bundle_sha256", "")))
        if endpoint.backend != self.config.dataplane_backend:
            raise ValueError("READY endpoint backend differs from pool policy")
        decoded = decode_endpoint_record(bytes.fromhex(endpoint.endpoint_record))
        expected_run = hashlib.sha256(self.config.run_id.encode()).digest()[:16]
        if (decoded.run_key != expected_run
                or decoded.fence_epoch != self.config.fence
                or endpoint.artifact_bundle_sha256
                != self.config.artifact_bundle_sha256):
            raise ValueError("READY native endpoint run/fence/artifact mismatch")
        remaining_endpoint_s = (
            endpoint.expires_unix_ns - time.time_ns()) / 1e9
        if remaining_endpoint_s <= 0:
            raise ValueError("READY native endpoint expired before admission")
        lease_expiry = time.monotonic() + min(
            self.config.slo.ready_lease_s, remaining_endpoint_s)
        generation = int(request["generation"])
        sequence = int(request.get("incarnation_sequence", 0))
        result = self._step(
            request, CoordinationEventKind.READY,
            generation=generation,
            node_id=endpoint.worker_id,
            incarnation=endpoint.incarnation,
            sequence=sequence,
            receipt_digest=str(request.get("apply_receipt_digest", "")),
        )
        disposition = str(result["disposition"])
        if disposition in {"accepted", "identical-duplicate"}:
            identity = (endpoint.worker_id, endpoint.incarnation)
            self.endpoints[identity] = endpoint
            self.endpoint_leases[identity] = lease_expiry
            self.endpoint_sequences[identity] = sequence
            frozen_incarnations = {
                str(item["cohort_incarnation"])
                for item in result["members"]  # type: ignore[index]
                if (item["worker_id"] == endpoint.worker_id  # type: ignore[index]
                    and item["cohort"])  # type: ignore[index]
            }
            for identity in tuple(self.endpoints):
                if (identity[0] == endpoint.worker_id
                        and identity[1] != endpoint.incarnation
                        and identity[1] not in frozen_incarnations):
                    self.endpoints.pop(identity, None)
                    self.endpoint_leases.pop(identity, None)
                    self.endpoint_sequences.pop(identity, None)
            return {
                "status": "READY", "disposition": disposition,
                "worker_id": endpoint.worker_id,
                "incarnation": endpoint.incarnation,
                "generation": generation,
            }
        return {
            "status": disposition, "disposition": disposition,
            "worker_id": endpoint.worker_id,
            "incarnation": endpoint.incarnation,
            "generation": generation,
            "authoritative_generation": result["committed_generation"],
        }

    def _open(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        key = (generation, attempt)
        prior = self.snapshots.get(key)
        now = time.monotonic()
        if prior is None:
            # The clock is external to the pure kernel.  Convert every elapsed
            # lease into a typed, serialized event before the kernel snapshots
            # READY; never discover expiry after authoritative open mutation.
            expired = sorted(
                identity for identity, expiry
                in self.endpoint_leases.items() if expiry <= now)
            for worker_id, incarnation in expired:
                expired_result = self._step(
                    request, CoordinationEventKind.EXPIRE_PEER,
                    generation=generation, node_id=worker_id,
                    incarnation=incarnation,
                    sequence=self.endpoint_sequences.get(
                        (worker_id, incarnation), 0),
                )
                if expired_result["disposition"] in {
                    "accepted", "identical-duplicate",
                }:
                    identity = (worker_id, incarnation)
                    self.endpoints.pop(identity, None)
                    self.endpoint_leases.pop(identity, None)
                    self.endpoint_sequences.pop(identity, None)
        result = self._step(
            request, CoordinationEventKind.OPEN_GENERATION,
            generation=generation, attempt=attempt)
        disposition = str(result["disposition"])
        if disposition not in {"accepted", "identical-duplicate"}:
            return {
                "status": disposition,
                "generation": generation,
                "attempt": attempt,
                "authoritative_generation": result["committed_generation"],
            }
        if prior is not None:
            return prior
        members = [
            item for item in result["members"]  # type: ignore[index]
            if item["cohort"]  # type: ignore[index]
        ]
        peers = []
        for member in members:
            identity = (str(member["worker_id"]), str(member["incarnation"]))
            endpoint = self.endpoints.get(identity)
            lease_expiry = self.endpoint_leases.get(identity, 0.0)
            if endpoint is None or lease_expiry <= now:
                raise RuntimeError(
                    "native cohort endpoint effect is absent or lease-expired")
            peers.append({**asdict(endpoint), "lease_expiry": lease_expiry})
        peers.sort(key=lambda item: str(item["worker_id"]))
        deadline = now + self.config.slo.training_hard_s
        close_not_before = (
            None if self.config.scale_close_offset_s is None
            else now + self.config.scale_close_offset_s)
        value: dict[str, object] = {
            "run_id": self.config.run_id,
            "fence": self.config.fence,
            "generation": generation,
            "attempt": attempt,
            "observed_at": now,
            "peers": peers,
            "deadline_after_s": self.config.slo.training_hard_s,
        }
        if close_not_before is not None:
            value["scale_closure"] = {
                "schema": "emender-v21s17-runtime-close-v1",
                "close_after_s": self.config.scale_close_offset_s,
                "stable_diversity_floor":
                    self.config.scale_stable_diversity_floor,
                "per_ready_worker_token_floor":
                    self.config.scale_per_ready_worker_token_floor,
                "closure_digest": self.config.scale_closure_digest,
                "close_on_q_min": False,
                "uses_launched_ranks": False,
                "wait_for_all_ready": False,
            }
        self.opened[key] = (now, deadline, close_not_before)
        self.snapshots[key] = value
        return value

    def _contribute(self, request: Mapping[str, object],
                    payload: bytes) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        worker_id = str(request["worker_id"])
        incarnation = str(request["incarnation"])
        sequence = int(request["contribution_seq"])
        if "aggregation_weight" in request:
            raise ValueError(
                "v2.1 contributions forbid aggregation_weight; "
                "accepted_tokens is exact")
        try:
            payload_digest = bytes.fromhex(payload.decode("ascii"))
        except (UnicodeError, ValueError) as error:
            raise ValueError(
                "native contribution payload must be one SHA-256 digest") from error
        if len(payload_digest) != 32:
            raise ValueError(
                "native contribution payload must be one SHA-256 digest")
        result = self._step(
            request, CoordinationEventKind.CONTRIBUTION,
            generation=generation, attempt=attempt,
            node_id=worker_id, incarnation=incarnation,
            sequence=sequence,
            exact_tokens=int(request["accepted_tokens"]),
            payload_digest=payload_digest,
            policy_digest=str(request.get(
                "policy_digest", self.config.policy_digest)),
        )
        disposition = str(result["disposition"])
        if disposition == "generation-closed":
            return {
                "status": "catch_up",
                "disposition": disposition,
                "generation": generation,
                "attempt": attempt,
                "authoritative_generation": result["committed_generation"],
                "receipt_digest": result["commit_receipt"],
                "manifest_digest": result["commit_manifest"],
                "result_root": result["committed_result"],
                "accepted_tokens": result["accepted_token_clock"],
                "requires_reload": True,
            }
        if disposition in {"accepted", "identical-duplicate"}:
            member = self._member(result, worker_id)
            if member is None:
                raise RuntimeError(
                    "native accepted contribution lacks member evidence")
            self.accepted_payloads[(generation, attempt, worker_id)] = {
                "incarnation": incarnation,
                "sequence": sequence,
                "exact_tokens": int(request["accepted_tokens"]),
                "payload_digest": payload.decode("ascii"),
            }
            return {
                "identity": {
                    "run_id": self.config.run_id,
                    "coordinator_epoch": self.config.fence,
                    "generation": generation,
                    "attempt": attempt,
                    "worker_id": worker_id,
                    "incarnation": incarnation,
                    "contribution_seq": sequence,
                },
                # A byte-identical replay remains an accepted receipt so a
                # lost RPC response can continue into close.
                "status": "accepted",
                "disposition": disposition,
                "content_digest": member["contribution_receipt"],
            }
        return {
            "status": disposition,
            "disposition": disposition,
            "generation": generation,
            "attempt": attempt,
        }

    def _close(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        opened = self.opened.get((generation, attempt))
        now = time.monotonic()
        flags = 0
        if opened is not None:
            _started, deadline, close_not_before = opened
            if close_not_before is None or now >= close_not_before:
                flags |= FINITE_CLOSE
            if now >= deadline:
                flags |= DEADLINE_EXPIRED
        result = self._step(
            request, CoordinationEventKind.CLOSE_GENERATION,
            generation=generation, attempt=attempt, flags=flags)
        disposition = str(result["disposition"])
        if disposition in {"deferred", "insufficient-cohort"}:
            return {
                "status": "open", "disposition": disposition,
                "accepted_tokens": sum(
                    int(item["exact_tokens"])
                    for item in result["members"]  # type: ignore[index]
                    if item["contributed"]),  # type: ignore[index]
            }
        if disposition == "retry-next-generation":
            return {
                "status": "aborted", "disposition": disposition,
                "authoritative_generation": result["committed_generation"],
            }
        if (disposition not in {"accepted", "identical-duplicate"}
                or result["phase"] not in {"closed", "committed", "applied"}):
            return {"status": disposition, "disposition": disposition}
        cohort = [
            item for item in result["members"]  # type: ignore[index]
            if item["cohort"]  # type: ignore[index]
        ]
        accepted = [
            item for item in cohort if item["contributed"]  # type: ignore[index]
        ]
        frozen_identities = []
        payloads = {}
        exact_tokens = {}
        for member in accepted:
            worker_id = str(member["worker_id"])
            cached = self.accepted_payloads.get(
                (generation, attempt, worker_id))
            if cached is None:
                raise RuntimeError(
                    "native frozen contribution lacks payload effect cache")
            frozen_identities.append({
                "run_id": self.config.run_id,
                "coordinator_epoch": self.config.fence,
                "generation": generation,
                "attempt": attempt,
                "worker_id": worker_id,
                "incarnation": member["cohort_incarnation"],
                "contribution_seq": int(cached["sequence"]),
            })
            payloads[worker_id] = str(cached["payload_digest"])
            exact_tokens[worker_id] = int(member["exact_tokens"])
        return {
            "status": "commit_ready",
            "disposition": disposition,
            "reason": "finite_native_close",
            "accepted_tokens": sum(exact_tokens.values()),
            "required_contributions": self.config.q_min,
            "ready_snapshot": [
                [str(item["worker_id"]), str(item["cohort_incarnation"])]
                for item in cohort
            ],
            "frozen_identities": frozen_identities,
            "accepted_payloads": payloads,
            "exact_tokens_by_worker": exact_tokens,
        }

    def _accepted(self, result: Mapping[str, object]
                  ) -> dict[str, Mapping[str, object]]:
        return {
            str(item["worker_id"]): item
            for item in result["members"]  # type: ignore[index]
            if item["contributed"]  # type: ignore[index]
        }

    def _route_ready(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        state = self._step(
            request, CoordinationEventKind.CLOSE_GENERATION,
            generation=generation, attempt=attempt, flags=0)
        if state["phase"] not in {"closed", "committed", "applied"}:
            raise RuntimeError(
                "route readiness cannot precede the frozen accepted set")
        accepted = self._accepted(state)
        worker_id, incarnation = (
            str(request["worker_id"]), str(request["incarnation"]))
        peer_worker_id, peer_incarnation = (
            str(request["peer_worker_id"]),
            str(request["peer_incarnation"]))
        if worker_id == peer_worker_id:
            raise ValueError("native route readiness requires a remote peer")
        if (worker_id not in accepted or peer_worker_id not in accepted
                or accepted[worker_id]["cohort_incarnation"] != incarnation
                or accepted[peer_worker_id]["cohort_incarnation"]
                != peer_incarnation):
            raise ValueError(
                "route readiness reporter is outside the frozen accepted set")
        pair = tuple(sorted((worker_id, peer_worker_id)))
        reports = self.route_readiness.setdefault(
            (generation, attempt, pair), {})
        report = (incarnation, peer_incarnation)
        prior = reports.get(worker_id)
        if prior is not None and prior != report:
            raise ValueError("conflicting native route readiness replay")
        reports[worker_id] = report
        if set(reports) != set(pair):
            return {
                "status": "waiting", "workers": sorted(reports),
                "required": list(pair),
            }
        left, right = pair
        if (reports[left] != (
                str(accepted[left]["cohort_incarnation"]),
                str(accepted[right]["cohort_incarnation"]))
                or reports[right] != (
                    str(accepted[right]["cohort_incarnation"]),
                    str(accepted[left]["cohort_incarnation"]))):
            raise ValueError("native route readiness incarnation mismatch")
        return {"status": "ready", "workers": list(pair)}

    def _owner_result(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        state = self._step(
            request, CoordinationEventKind.CLOSE_GENERATION,
            generation=generation, attempt=attempt, flags=0)
        accepted = self._accepted(state)
        worker_id, incarnation = (
            str(request["worker_id"]), str(request["incarnation"]))
        if (worker_id not in accepted
                or accepted[worker_id]["cohort_incarnation"] != incarnation):
            raise ValueError(
                "owner result reporter is outside the frozen accepted set")
        exact_tokens = sum(
            int(item["exact_tokens"]) for item in accepted.values())
        root = str(request["result_root"])
        layout_digest = str(request["layout_digest"])
        weight, result_bytes = (
            int(request["global_weight"]), int(request["result_bytes"]))
        if (len(root) != 64 or root == "00" * 32
                or len(layout_digest) != 64
                or layout_digest == "00" * 32
                or weight != exact_tokens or result_bytes <= 0):
            raise ValueError("owner result metadata is invalid")
        values = self.owner_results.setdefault((generation, attempt), {})
        record = {
            "incarnation": incarnation, "result_root": root,
            "layout_digest": layout_digest, "global_weight": weight,
            "result_bytes": result_bytes,
        }
        prior = values.get(worker_id)
        if prior is not None and prior != record:
            raise ValueError("conflicting owner result replay")
        values[worker_id] = record
        if set(values) != set(accepted):
            return {
                "status": "waiting", "reported": len(values),
                "required": len(accepted),
            }
        return {
            "status": "ready", "global_weight": exact_tokens,
            "roots": {
                worker: str(values[worker]["result_root"])
                for worker in sorted(values)
            },
            "owners": {
                worker: {
                    "result_root": str(values[worker]["result_root"]),
                    "layout_digest": str(values[worker]["layout_digest"]),
                    "result_bytes": int(values[worker]["result_bytes"]),
                } for worker in sorted(values)
            },
        }

    def _result_root(self, request: Mapping[str, object]) -> dict[str, object]:
        generation, attempt = int(request["generation"]), int(request["attempt"])
        worker_id, incarnation = (
            str(request["worker_id"]), str(request["incarnation"]))
        cached = self.accepted_payloads.get((generation, attempt, worker_id))
        if cached is None:
            raise ValueError(
                "result root reporter is outside the frozen accepted set")
        result = self._step(
            request, CoordinationEventKind.RESULT_RECEIPT,
            generation=generation, attempt=attempt,
            node_id=worker_id, incarnation=incarnation,
            sequence=int(cached["sequence"]),
            exact_tokens=int(request["global_weight"]),
            result_digest=str(request["result_root"]),
        )
        disposition = str(result["disposition"])
        if disposition not in {"accepted", "identical-duplicate"}:
            return {"status": disposition, "disposition": disposition}
        if int(result["result_receipt_count"]) != int(
                result["contribution_count"]):
            return {
                "status": "waiting",
                "disposition": disposition,
                "reported": result["result_receipt_count"],
                "required": result["contribution_count"],
            }
        return {
            "status": "validated", "disposition": disposition,
            "result_root": str(request["result_root"]),
            "global_weight": int(request["global_weight"]),
            "result_bytes": int(request["result_bytes"]),
            "workers": sorted(self._accepted(result)),
        }

    def _commit(self, request: Mapping[str, object]) -> dict[str, object]:
        result_generation = int(request["result_generation"])
        record = {
            "result_generation": result_generation,
            "source_generation": result_generation - 1,
            "attempt": int(request["attempt"]),
            "receipt_digest": str(request["receipt_digest"]),
            "previous_receipt_digest":
                str(request.get("previous_receipt_digest", "")),
            "manifest_digest": str(request["manifest_digest"]),
            "result_root": str(request["result_root"]),
            "accepted_tokens": int(request["accepted_tokens"]),
        }
        result = self._step(
            request, CoordinationEventKind.COMMIT,
            generation=result_generation,
            attempt=int(record["attempt"]),
            exact_tokens=int(record["accepted_tokens"]),
            receipt_digest=str(record["receipt_digest"]),
            previous_receipt_digest=
                str(record["previous_receipt_digest"]),
            manifest_digest=str(record["manifest_digest"]),
            result_digest=str(record["result_root"]),
        )
        disposition = str(result["disposition"])
        if disposition in {"accepted", "identical-duplicate"}:
            prior = self.commit_records.get(result_generation)
            if prior is not None and prior != record:
                raise RuntimeError(
                    "native accepted commit cache conflicts with authority")
            self.commit_records[result_generation] = record
            # The compiled state retains only the one active generation.  Its
            # effect-side Python caches mirror that fixed bound: once commit is
            # authoritative, no prior payload, timer, or route record can be
            # consulted by a legal next event.
            self.commit_records = {
                key: value for key, value in self.commit_records.items()
                if key == result_generation
            }
            self.snapshots = {
                key: value for key, value in self.snapshots.items()
                if key[0] >= result_generation
            }
            self.opened = {
                key: value for key, value in self.opened.items()
                if key[0] >= result_generation
            }
            self.accepted_payloads = {
                key: value for key, value in self.accepted_payloads.items()
                if key[0] >= result_generation
            }
            self.owner_results = {
                key: value for key, value in self.owner_results.items()
                if key[0] >= result_generation
            }
            self.route_readiness = {
                key: value for key, value in self.route_readiness.items()
                if key[0] >= result_generation
            }
            return {
                "status": "committed", "disposition": disposition, **record,
            }
        return {
            "status": disposition, "disposition": disposition,
            "result_generation": result_generation,
            "current_generation": result["committed_generation"],
        }

    def _commit_state(self, request: Mapping[str, object]) -> dict[str, object]:
        generation = int(request["result_generation"])
        result = self._step(
            request, CoordinationEventKind.QUERY_COMMIT,
            generation=generation)
        disposition = str(result["disposition"])
        if (disposition == "accepted"
                and generation == int(result["committed_generation"])):
            record = self.commit_records.get(generation, {
                "result_generation": generation,
                "source_generation": generation - 1,
                "attempt": int(result["active_attempt"]),
                "receipt_digest": result["commit_receipt"],
                "manifest_digest": result["commit_manifest"],
                "result_root": result["committed_result"],
                "accepted_tokens": result["accepted_token_clock"],
            })
            return {
                "status": "committed", "disposition": disposition, **record,
            }
        if disposition == "deferred":
            return {
                "status": "pending", "disposition": disposition,
                "result_generation": generation,
                "current_generation": result["committed_generation"],
            }
        return {
            "status": disposition, "disposition": disposition,
            "result_generation": generation,
            "current_generation": result["committed_generation"],
        }

    def _recover(self, request: Mapping[str, object]) -> dict[str, object]:
        worker_id, incarnation = (
            str(request.get("worker_id", "")),
            str(request.get("incarnation", "")))
        result = self._step(
            request, CoordinationEventKind.RECOVER_PEER,
            generation=int(request.get("known_generation", 0)),
            node_id=worker_id, incarnation=incarnation,
            sequence=int(request.get("incarnation_sequence", 0)),
            receipt_digest=str(request.get("known_receipt_digest", "")),
        )
        disposition = str(result["disposition"])
        if disposition not in {"accepted", "identical-duplicate"}:
            return {
                "status": disposition, "disposition": disposition,
                "generation": result["committed_generation"],
            }
        member = self._member(result, worker_id)
        requires_apply = bool(
            int(result["committed_generation"]) > 0
            and (member is None or not member["node_applied"]))
        return {
            "status": "recover",
            "disposition": disposition,
            "generation": result["committed_generation"],
            "receipt_digest": result["commit_receipt"],
            "manifest_digest": result["commit_manifest"],
            "result_root": result["committed_result"],
            "accepted_tokens": result["accepted_token_clock"],
            "apply_receipts": [
                {
                    "worker_id": node_id,
                    "incarnation": str(record["incarnation"]),
                    "receipt_digest": str(record["receipt_digest"]),
                }
                for node_id, record in sorted(self.apply_receipts.items())
            ],
            "requires_node_apply": requires_apply,
        }

    def _node_apply(self, request: Mapping[str, object]) -> dict[str, object]:
        worker_id, incarnation = (
            str(request["worker_id"]), str(request["incarnation"]))
        result = self._step(
            request, CoordinationEventKind.NODE_APPLY,
            generation=int(request["generation"]),
            node_id=worker_id, incarnation=incarnation,
            sequence=int(request.get("incarnation_sequence", 0)),
            trainer_count=int(request["trainer_count"]),
            receipt_digest=str(request["receipt_digest"]),
            previous_receipt_digest=str(request["commit_receipt_digest"]),
        )
        disposition = str(result["disposition"])
        if disposition in {"accepted", "identical-duplicate"}:
            self.apply_receipts[worker_id] = {
                "incarnation": incarnation,
                "receipt_digest": str(request["receipt_digest"]),
            }
            return {
                "status": "node_applied", "disposition": disposition,
                "generation": int(request["generation"]),
                "worker_id": worker_id, "incarnation": incarnation,
                "receipt_digest": str(request["receipt_digest"]),
            }
        return {"status": disposition, "disposition": disposition}

    def _expire(self, request: Mapping[str, object]) -> dict[str, object]:
        worker_id, incarnation = (
            str(request["worker_id"]), str(request["incarnation"]))
        result = self._step(
            request, CoordinationEventKind.EXPIRE_PEER,
            generation=int(request.get(
                "generation", self.config.committed_generation)),
            node_id=worker_id, incarnation=incarnation,
            sequence=int(request.get("incarnation_sequence", 0)),
        )
        if result["disposition"] in {
            "accepted", "identical-duplicate",
        }:
            identity = (worker_id, incarnation)
            self.endpoints.pop(identity, None)
            self.endpoint_leases.pop(identity, None)
            self.endpoint_sequences.pop(identity, None)
        return {
            "status": (
                "DRAINING" if result["disposition"] in {
                    "accepted", "identical-duplicate",
                } else result["disposition"]),
            "disposition": result["disposition"],
            "worker_id": worker_id,
        }

    def _owner_lost(self, request: Mapping[str, object]) -> dict[str, object]:
        """Feed one transport/supervisor loss observation into authority."""
        result = self._step(
            request, CoordinationEventKind.OWNER_LOST,
            generation=int(request["generation"]),
            attempt=int(request["attempt"]),
            node_id=str(request["worker_id"]),
            incarnation=str(request["incarnation"]),
            sequence=int(request.get("incarnation_sequence", 0)),
        )
        return {
            "status": result["disposition"],
            "disposition": result["disposition"],
            "generation": int(request["generation"]),
            "owner_epoch": result["owner_epoch"],
            "owner_reassignments": result["owner_reassignments"],
            "effects": result["effects"],
        }

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.authority.close()


class PoolControlClient:
    def __init__(self, address: tuple[str, int], *, timeout_s: float):
        self.address, self.timeout_s = address, float(timeout_s)
        if self.timeout_s <= 0:
            raise ValueError("control timeout must be positive")
        self.run_id: str | None = None
        self.fence: int | None = None

    def bind(self, run_id: str, fence: int) -> "PoolControlClient":
        self.run_id, self.fence = run_id, int(fence)
        return self

    def _rpc(self, op: str, *, payload: bytes = b"", **fields: object) -> dict[str, object]:
        run_id = str(fields.pop("run_id", self.run_id or "run"))
        fence = int(fields.pop("fence", self.fence if self.fence is not None else 0))
        # Unit/local callers may not explicitly bind; the server-side values are
        # carried by public methods once their first READY call succeeds.
        deadline = time.monotonic() + self.timeout_s
        last: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection(self.address, timeout=min(1.0, self.timeout_s))
                with sock, sock.makefile("rwb", buffering=0) as stream:
                    send_frame(stream, {"op": op, "run_id": run_id, "fence": fence, **fields}, payload)
                    header, _ = recv_frame(stream, max_payload_bytes=0)
                if header["op"] == "error":
                    raise RuntimeError(str(header["error"]))
                return dict(header["result"])
            except OSError as error:
                last = error
                time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"pool control RPC deadline expired: {last}")

    def ready(self, endpoint: OwnerEndpoint, generation: int, *,
              run_id: str | None = None,
              fence: int | None = None,
              apply_receipt_digest: str = "",
              incarnation_sequence: int = 1) -> dict[str, object]:
        if run_id is not None:
            self.run_id = run_id
        elif self.run_id is None:
            self.run_id = "run"
        if fence is not None:
            self.fence = int(fence)
        if self.fence is None:
            raise ValueError("pool control client must bind the allocation fence")
        return self._rpc(
            "ready", **asdict(endpoint), generation=generation,
            apply_receipt_digest=apply_receipt_digest,
            incarnation_sequence=incarnation_sequence)

    def recover(self, *, worker_id: str, incarnation: str,
                known_generation: int,
                known_receipt_digest: str,
                incarnation_sequence: int = 1) -> dict[str, object]:
        return self._rpc(
            "recover",
            worker_id=worker_id,
            incarnation=incarnation,
            known_generation=known_generation,
            known_receipt_digest=known_receipt_digest,
            incarnation_sequence=incarnation_sequence,
        )

    def commit_authority(
            self, *, result_generation: int, attempt: int,
            receipt_digest: str, previous_receipt_digest: str,
            manifest_digest: str, result_root: str, accepted_tokens: int,
            ) -> dict[str, object]:
        return self._rpc(
            "commit",
            result_generation=result_generation,
            attempt=attempt,
            receipt_digest=receipt_digest,
            previous_receipt_digest=previous_receipt_digest,
            manifest_digest=manifest_digest,
            result_root=result_root,
            accepted_tokens=accepted_tokens,
        )

    def wait_for_commit(
            self, *, result_generation: int, deadline: float,
            ) -> dict[str, object]:
        """Wait on peer memory, never a shared manifest directory."""
        last: dict[str, object] = {}
        while time.monotonic() < deadline:
            last = self._rpc(
                "commit_state", result_generation=result_generation)
            if last.get("status") == "committed":
                return last
            time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            "native peer commit deadline expired"
            + (
                ""
                if not last
                else f" at generation {last.get('current_generation', -1)}"
            )
        )

    def node_applied(
            self, *, generation: int, worker_id: str, incarnation: str,
            receipt_digest: str, commit_receipt_digest: str,
            trainer_count: int = 8,
            incarnation_sequence: int = 1) -> dict[str, object]:
        return self._rpc(
            "node_apply",
            generation=generation,
            worker_id=worker_id,
            incarnation=incarnation,
            receipt_digest=receipt_digest,
            commit_receipt_digest=commit_receipt_digest,
            trainer_count=trainer_count,
            incarnation_sequence=incarnation_sequence,
        )

    def open_generation(self, generation: int, attempt: int, *,
                        deadline: float) -> dict[str, object]:
        last: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                result = self._rpc(
                    "open", generation=generation, attempt=attempt)
                if result.get("status") in {
                    "deferred", "insufficient-cohort",
                }:
                    time.sleep(.01)
                    continue
                return result
            except RuntimeError as error:
                last = error
                if "READY floor" not in str(error):
                    raise
                time.sleep(.01)
        raise TimeoutError(f"READY snapshot deadline expired: {last}")

    def contribute(self, generation: int, attempt: int, worker_id: str,
                   incarnation: str, contribution_seq: int, accepted_tokens: int,
                   payload_digest: str) -> dict[str, object]:
        return self._rpc("contribute", payload=payload_digest.encode(), generation=generation,
                         attempt=attempt, worker_id=worker_id, incarnation=incarnation,
                         contribution_seq=contribution_seq,
                         accepted_tokens=accepted_tokens)

    def contribute_and_freeze(self, *, generation: int, attempt: int,
                              worker_id: str, incarnation: str,
                              contribution_seq: int, accepted_tokens: int,
                              payload_digest: str, deadline: float,
                              ) -> dict[str, object]:
        receipt = self.contribute(generation, attempt, worker_id, incarnation,
                                  contribution_seq, accepted_tokens, payload_digest)
        if receipt["status"] != "accepted":
            return receipt
        while time.monotonic() < deadline:
            close = self._rpc("close", generation=generation, attempt=attempt)
            if close["status"] != "open":
                return close
            time.sleep(.01)
        raise TimeoutError("deterministic freeze deadline expired")

    def await_peer_route_ready(self, *, generation: int, attempt: int,
                               worker_id: str, incarnation: str,
                               peer_worker_id: str, peer_incarnation: str,
                               deadline: float) -> dict[str, object]:
        """Wait for reciprocal installation of one frozen native peer pair."""
        last: dict[str, object] = {"status": "waiting"}
        while time.monotonic() < deadline:
            last = self._rpc(
                "route_ready", generation=generation, attempt=attempt,
                worker_id=worker_id, incarnation=incarnation,
                peer_worker_id=peer_worker_id,
                peer_incarnation=peer_incarnation)
            if last.get("status") == "ready":
                return last
            time.sleep(.01)
        raise TimeoutError(f"native peer route-readiness deadline expired: {last}")

    def drain(self, worker_id: str, incarnation: str, *,
              generation: int = 0,
              incarnation_sequence: int = 1) -> dict[str, object]:
        return self._rpc(
            "expire", worker_id=worker_id, incarnation=incarnation,
            generation=generation,
            incarnation_sequence=incarnation_sequence)

    def owner_lost(self, *, generation: int, attempt: int,
                   worker_id: str, incarnation: str,
                   incarnation_sequence: int = 1) -> dict[str, object]:
        """Report transport/process loss and receive explicit kernel effects."""
        return self._rpc(
            "owner_lost", generation=generation, attempt=attempt,
            worker_id=worker_id, incarnation=incarnation,
            incarnation_sequence=incarnation_sequence)

    def validate_result_root(self, *, generation: int, attempt: int,
                             worker_id: str, incarnation: str, result_root: str,
                             global_weight: int, result_bytes: int,
                             deadline: float) -> dict[str, object]:
        last: dict[str, object] = {"status": "waiting"}
        while time.monotonic() < deadline:
            last = self._rpc(
                "result_root", generation=generation, attempt=attempt,
                worker_id=worker_id, incarnation=incarnation,
                result_root=result_root, global_weight=global_weight,
                result_bytes=result_bytes)
            if last.get("status") == "validated":
                return last
            time.sleep(.01)
        raise TimeoutError(f"native result-root validation deadline expired: {last}")

    def announce_owner_result(self, *, generation: int, attempt: int,
                              worker_id: str, incarnation: str,
                              result_root: str, layout_digest: str,
                              global_weight: int,
                              result_bytes: int, deadline: float
                              ) -> dict[str, object]:
        """Return the immutable per-owner root map once all frozen owners report."""
        last: dict[str, object] = {"status": "waiting"}
        while time.monotonic() < deadline:
            last = self._rpc(
                "owner_result", generation=generation, attempt=attempt,
                worker_id=worker_id, incarnation=incarnation,
                result_root=result_root, layout_digest=layout_digest,
                global_weight=global_weight,
                result_bytes=result_bytes)
            if last.get("status") == "ready":
                return last
            time.sleep(.01)
        raise TimeoutError(f"native owner-result announcement deadline expired: {last}")


class _OwnerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            header, payload = recv_frame(self.rfile,
                                         max_payload_bytes=self.server.max_owner_bytes)
            result, response = self.server.dispatch(header, payload)
            send_frame(self.wfile, {"op": "result", **result}, response)
        except Exception as error:
            send_frame(self.wfile, {"op": "error", "error": str(error)})


class DistributedOwnerServer(socketserver.ThreadingTCPServer):
    """Own only its deterministic subset of one full-state generation."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], worker_id: str, *, max_owner_bytes: int):
        self.worker_id, self.max_owner_bytes = worker_id, int(max_owner_bytes)
        self._condition = threading.Condition()
        self._layout: TensorLayout | None = None
        self._identity: tuple[str, int, int, int] | None = None
        self._owners: tuple[OwnerEndpoint, ...] = ()
        self._accepted: tuple[str, ...] = ()
        self._reducers: dict[int, ExactWeightedShardReducer] = {}
        self._received: dict[int, set[str]] = {}
        self._aggregates: dict[int, ShardChunk] = {}
        self._receipts: dict[tuple[int, str], tuple[int, str]] = {}
        self.bytes_received = self.bytes_sent = self.high_water_bytes = 0
        super().__init__(address, _OwnerHandler)

    def install(self, layout: TensorLayout, *, run_id: str, fence: int,
                generation: int, attempt: int, owners: Sequence[OwnerEndpoint],
                accepted_ids: Sequence[str]) -> None:
        ordered_owners = tuple(sorted(owners, key=lambda item: item.worker_id))
        if len({item.worker_id for item in ordered_owners}) != len(ordered_owners):
            raise ValueError("owner endpoints must have unique stable identities")
        owned = [shard for shard in range(layout.shard_count)
                 if layout.owner(shard, [item.worker_id for item in ordered_owners],
                                 run_id=run_id, generation=generation, attempt=attempt)
                 == self.worker_id]
        with self._condition:
            self._layout, self._identity = layout, (run_id, fence, generation, attempt)
            self._owners, self._accepted = ordered_owners, tuple(sorted(accepted_ids))
            self._reducers = {shard: ExactWeightedShardReducer(
                layout, shard, max_inflight_bytes=self.max_owner_bytes) for shard in owned}
            self._received = {shard: set() for shard in owned}
            self._aggregates.clear(); self._receipts.clear()
            self._condition.notify_all()

    def dispatch(self, request: Mapping[str, object], payload: bytes
                 ) -> tuple[dict[str, object], bytes]:
        op = str(request.get("op"))
        with self._condition:
            if op == "ping":
                return {"status": "live", "worker_id": self.worker_id}, b""
            if self._layout is None or self._identity is None:
                raise RuntimeError("shard owner generation is not installed")
            identity = (str(request.get("run_id")), int(request.get("fence", -1)),
                        int(request.get("generation", -1)), int(request.get("attempt", -1)))
            if identity != self._identity:
                raise ValueError("stale shard owner fence/generation rejected")
            shard = int(request["shard"])
            expected_owner = self._layout.owner(
                shard, [item.worker_id for item in self._owners], run_id=identity[0],
                generation=identity[2], attempt=identity[3])
            if expected_owner != self.worker_id or shard not in self._reducers:
                raise ValueError("shard routed to non-owner")
            if op == "submit":
                contribution_id = str(request["contribution_id"])
                weight = int(request["weight"])
                if contribution_id not in self._accepted:
                    raise ValueError("unfrozen contribution rejected by shard owner")
                checksum = str(request["checksum"])
                receipt = (weight, checksum)
                prior = self._receipts.get((shard, contribution_id))
                if prior is not None:
                    if prior != receipt:
                        raise ValueError("conflicting owner contribution replay")
                    return {"status": "duplicate"}, b""
                chunk = ShardChunk(self._layout.digest, shard,
                                   int(request["element_offset"]),
                                   int(request["elements"]), payload, checksum)
                inflight = sum(reducer.inflight_bytes for reducer in self._reducers.values())
                if inflight + chunk.nbytes > self.max_owner_bytes:
                    raise BufferError("distributed owner aggregate byte bound exceeded")
                self._reducers[shard].submit(contribution_id, weight=weight, chunk=chunk)
                self._receipts[(shard, contribution_id)] = receipt
                self._received[shard].add(contribution_id)
                self.bytes_received += len(payload)
                self.high_water_bytes = max(
                    self.high_water_bytes,
                    sum(reducer.inflight_bytes for reducer in self._reducers.values()))
                if self._received[shard] == set(self._accepted):
                    self._aggregates[shard] = self._reducers[shard].finalize(self._accepted)
                    self._condition.notify_all()
                return {"status": "accepted"}, b""
            if op == "fetch":
                wait_s = float(request.get("wait_s", 0))
                deadline = time.monotonic() + max(0.0, wait_s)
                while shard not in self._aggregates:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("owner aggregate fetch deadline expired")
                    self._condition.wait(remaining)
                chunk = self._aggregates[shard]
                self.bytes_sent += chunk.nbytes
                return {"status": "committed", "layout_digest": chunk.layout_digest,
                        "shard": chunk.shard_id, "element_offset": chunk.element_offset,
                        "elements": chunk.elements, "checksum": chunk.checksum_sha256}, chunk.payload
            raise ValueError("unsupported shard owner operation")


def _owner_rpc(endpoint: OwnerEndpoint, request: Mapping[str, object], payload: bytes,
               *, deadline: float, max_payload_bytes: int) -> tuple[dict[str, object], bytes]:
    last: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            remaining_s = deadline - time.monotonic()
            connect_timeout_s = min(_OWNER_CONNECT_TIMEOUT_S,
                                    max(.01, remaining_s))
            sock = socket.create_connection((endpoint.host, endpoint.port),
                                            timeout=connect_timeout_s)
            sock.settimeout(_bounded_owner_io_timeout(
                len(payload), max_payload_bytes,
                remaining_s=deadline - time.monotonic()))
            with sock, sock.makefile("rwb", buffering=0) as stream:
                send_frame(stream, request, payload)
                header, response = recv_frame(stream, max_payload_bytes=max_payload_bytes)
            if header["op"] == "error":
                error = RuntimeError(str(header["error"]))
                if any(reason in str(error) for reason in (
                        "not installed", "routed to non-owner")) \
                        and time.monotonic() < deadline:
                    last = error
                    time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
                    continue
                raise error
            return header, response
        except OSError as error:
            last = error
            time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
    raise TimeoutError(f"shard owner transport deadline expired: {last}")


def _endpoint_by_owner(layout: TensorLayout, endpoints: Sequence[OwnerEndpoint], shard: int,
                       *, run_id: str, generation: int, attempt: int) -> OwnerEndpoint:
    owner = layout.owner(shard, [item.worker_id for item in endpoints], run_id=run_id,
                         generation=generation, attempt=attempt)
    return next(item for item in endpoints if item.worker_id == owner)


def live_owner_endpoints(endpoints: Sequence[OwnerEndpoint], *, deadline: float
                         ) -> tuple[OwnerEndpoint, ...]:
    """Bounded deterministic liveness probe used before placement/replay."""
    live = []
    for endpoint in sorted(endpoints, key=lambda item: item.worker_id):
        try:
            _owner_rpc(endpoint, {"op": "ping"}, b"", deadline=deadline,
                       max_payload_bytes=0)
            live.append(endpoint)
        except (OSError, RuntimeError, TimeoutError):
            continue
    return tuple(live)


def submit_owned_shards(*, layout: TensorLayout, chunks: Sequence[ShardChunk],
                        contribution_id: str, weight: int,
                        endpoints: Sequence[OwnerEndpoint], run_id: str, fence: int,
                        generation: int, attempt: int, deadline: float) -> dict[str, int]:
    """Replayable sender path: retain ``chunks`` until every owner receipt."""
    if time.monotonic() >= deadline:
        raise TimeoutError("shard submission deadline expired")
    retained = sum(chunk.nbytes for chunk in chunks)
    sent = 0
    for chunk in sorted(chunks, key=lambda item: item.shard_id):
        endpoint = _endpoint_by_owner(layout, endpoints, chunk.shard_id, run_id=run_id,
                                      generation=generation, attempt=attempt)
        _owner_rpc(endpoint, {
            "op": "submit", "run_id": run_id, "fence": fence,
            "generation": generation, "attempt": attempt, "shard": chunk.shard_id,
            "contribution_id": contribution_id, "weight": weight,
            "element_offset": chunk.element_offset, "elements": chunk.elements,
            "checksum": chunk.checksum_sha256,
        }, chunk.payload, deadline=deadline, max_payload_bytes=0)
        sent += chunk.nbytes
    return {"p2p_bytes_sent": sent, "peak_retained_bytes": retained,
            "released_bytes": retained}


def fetch_owned_shards(*, layout: TensorLayout, endpoints: Sequence[OwnerEndpoint],
                       run_id: str, fence: int, generation: int, attempt: int,
                       deadline: float) -> tuple[tuple[ShardChunk, ...], dict[str, int]]:
    """Fetch each aggregate directly from its deterministic owner."""
    chunks = []
    for shard in range(layout.shard_count):
        endpoint = _endpoint_by_owner(layout, endpoints, shard, run_id=run_id,
                                      generation=generation, attempt=attempt)
        wait_s = max(0.0, deadline - time.monotonic())
        header, payload = _owner_rpc(endpoint, {
            "op": "fetch", "run_id": run_id, "fence": fence,
            "generation": generation, "attempt": attempt, "shard": shard,
            "wait_s": wait_s,
        }, b"", deadline=deadline, max_payload_bytes=layout.chunk_elements * 8)
        chunks.append(ShardChunk(str(header["layout_digest"]), int(header["shard"]),
                                 int(header["element_offset"]), int(header["elements"]),
                                 payload, str(header["checksum"])))
    total = sum(chunk.nbytes for chunk in chunks)
    return tuple(chunks), {"redistribution_bytes": total, "received_bytes": total}


def contribution_id(identity: Mapping[str, object]) -> str:
    """Compact stable sender/owner key for one frozen ContributionIdentity."""
    return (f"{identity['worker_id']}:{identity['incarnation']}:"
            f"{int(identity['contribution_seq'])}")


def chunk_manifest_digest(chunks: Sequence[ShardChunk]) -> str:
    value = [(item.shard_id, item.elements, item.checksum_sha256) for item in chunks]
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
