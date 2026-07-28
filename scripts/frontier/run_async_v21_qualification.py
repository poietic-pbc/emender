#!/usr/bin/env python3
"""Serial, fail-closed controller for async-decoupled-v2.1 qualification.

This is the only qualification/submission surface for the v2.1 production
path.  It renders a single immutable payload at a time.  Two-node gates are
always ``Nodes=2, Partition=batch, QOS=debug``.  Scale requires a separately
signed promotion manifest, a signed pass from the exact predecessor, and the
reviewed V21S17 finite close derived from retained passing two-node evidence.

The controller does not use launched ranks as training membership.  A scale
job receives only the reviewed closure description; the allocation control
plane evaluates it over the leased READY snapshot taken at group open.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POLICY_ID = "async-decoupled-v2.1-simple"
POLICY_SCHEMA = "emender-async-policy-v2.1"
PAYLOAD_SCHEMA = "emender-async-v21-qualification-payload-v1"
STATE_SCHEMA = "emender-async-v21-qualification-state-v2"
LEGACY_STATE_SCHEMA = "emender-async-v21-qualification-state-v1"
COLLECTOR_SCHEMA = "emender-async-v21-terminal-collector-v1"
FRONTIER_ACCOUNT = "bif148"
COLLECTOR_QOS = "normal"
EXECUTION_SOURCE_SCHEMA = "emender-async-v21-execution-source-v1"
# These are append-only human/machine evidence stores.  They are deliberately
# the complete exclusion list: every other tracked byte, including authority
# documents and experiment/data preparation code, remains executable identity.
EVIDENCE_ONLY_PATH_PREFIXES = (
    "docs/validation/",
    "logs/",
    "reports/",
)
AUTHORIZATION_SCHEMA = "emender-async-v21-scale-authorization-v1"
RUNG_PASS_SCHEMA = "emender-async-v21-rung-pass-v1"
CLOSURE_SCHEMA = "emender-v21s17-scale-closure-v1"
CONFIG_PATH = ROOT / "configs/frontier/e97_async_256.yaml"
LAUNCHER_PATH = ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch"
SCALE_RUNGS = (4, 8, 16, 32, 64, 256)
PREDECESSOR = {4: 2, 8: 4, 16: 8, 32: 16, 64: 32, 256: 64}
TWO_NODE_GATES = ("clean", "faults", "convergence")
ALL_GATES = (*TWO_NODE_GATES, "scale")
SEED_STEP = 2_300_930
SEED_ACCEPTED_TOKENS = 150_793_748_480
SEED_BYTES = 7_719_680_116
SEED_SHA256 = (
    "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
)
SEED_NODE_PATH = "/tmp/emender-e97-seed-$SLURM_JOB_ID"
LINUX_SUN_PATH_BYTES = 108
CLEAN_WALLTIME = "02:00:00"
CLEAN_PHASE = "clean-overlap"
# The debug QoS caps the clean gate at two hours.  Ten finalized generations
# are the acceptance authority's minimum and already contain far more than the
# required two warm-up plus ten measured K40 windows per trainer.  Requesting
# twelve made the launcher enter an eleventh generation instead of terminating
# after the gate was satisfied; job 5084736 was consequently stopped by the
# launcher's five-minute pre-timeout signal after its tenth checkpoint but
# before its tenth all-rank apply.
CLEAN_GENERATIONS = 10
CLEAN_SIGNAL = "B:TERM@60"
CLEAN_PROGRESS_DEADLINE_S = 45 * 60
CLEAN_GENERATION_DEADLINE_S = 420
FAULT_WALLTIME = "02:00:00"
FAULT_SIGNAL = "B:TERM@60"
FAULT_PROGRESS_DEADLINE_S = 45 * 60
FAULT_GENERATION_DEADLINE_S = 420
FAULT_PHASE_SPECS = (
    {
        "name": "fault-baseline",
        "initial_generation": 0,
        "generations": 2,
        "minimum_commits": 2,
        "fresh_allocation": False,
        "max_restarts": 0,
        "injections": {},
    },
    {
        "name": "fault-rejoin",
        "initial_generation": 2,
        "generations": 4,
        "minimum_commits": 4,
        "fresh_allocation": False,
        "max_restarts": 3,
        "injections": {
            "RESILIENT_E97_DELAY_READY": "1:2:45",
            "RESILIENT_E97_INJECT_TRAINER": "0:3:2",
            "RESILIENT_E97_INJECT_MANAGER":
                "1:-1:4:published_node_applied",
            "RESILIENT_E97_INJECT_NATIVE_SERVICE":
                "0:-1:4:owner_transport",
        },
    },
    {
        "name": "fresh-recovery",
        "initial_generation": 6,
        "generations": 5,
        "minimum_commits": 3,
        "fresh_allocation": True,
        "max_restarts": 0,
        "injections": {},
    },
)
FAULT_REQUIREMENTS = {
    "compute_pool": [f"R{index:02d}" for index in range(1, 17)],
    "native": [f"NDP{index:02d}" for index in range(1, 18)],
    "async_v21": [f"V21S{index:02d}" for index in range(1, 18)],
    "immutable_snapshot": [f"ISP{index:02d}" for index in range(1, 8)],
}
FAULT_SCENARIOS = (
    "delayed_or_missing_contribution",
    "lag_0_1_2_admission",
    "lag_3_drop_and_catch_up",
    "duplicate_idempotence_and_conflict_rejection",
    "checksum_nonfinite_wrong_fence_rejection",
    "local_owned_timeout",
    "trainer_loss",
    "native_service_loss",
    "manager_loss_new_incarnation",
    "owner_reassignment",
    "failed_publication_invisibility",
    "mailbox_replacement",
    "all_eight_trainer_atomic_apply",
    "fresh_allocation_model_outer_token_restore",
)
APPROVED_ENV = (
    "/lustre/orion/bif148/scratch/erikgarrison/emender/"
    ".envs/olcf-rocm711-torch210-py312"
)
DATA_PATH = Path(
    "/lustre/orion/bif148/proj-shared/commapile/"
    "commapile_mainmix_v0.1_1tb.txt"
)
TOKENIZER_PATH = Path(
    "/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/"
    "p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15"
)
TOKENIZER_SHA256 = (
    "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"
)
SEED_CACHE_ROOT = Path(
    "/lustre/orion/bif148/proj-shared/emender/bootstrap/e97-seeds"
)
CLEAN_PARAMETERS = {
    "all_eight_apply_swap_seconds_max": 60,
    "causal_phase_classes": [
        "freeze_snapshot",
        "snapshot_admission",
        "publish_network",
        "aggregation",
        "checkpoint",
        "result_wait",
        "apply_swap",
    ],
    "foreground_idle_fraction_strict_max": 0.10,
    "foreground_gap_seconds_max": 60,
    "foreground_result_wait_seconds_max": 0,
    "freeze_to_latest_seconds_max": 420,
    "immutable_snapshot_requirements": [
        f"ISP{i:02d}" for i in range(1, 8)
    ],
    "local_owned_latency_seconds_max": 1,
    "local_steps": 40,
    "measured_windows_per_trainer": 10,
    "minimum_atomic_commits": 10,
    "pause_tail_statistics": ["maximum", "p99"],
    "progress_deadline_seconds": CLEAN_PROGRESS_DEADLINE_S,
    "real_trainers": 16,
    "snapshot_admission_seconds_max": 1,
    "steady_state_cadence_multiple_max": 1.25,
    "trainers_per_node": 8,
    "warmup_windows_per_trainer": 2,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _qualification_bulk_root(*, run_id: str, phase_name: str) -> str:
    """Return a phase-unique node-local root that always fits AF_UNIX."""
    phase_identity = canonical_digest({
        "run_id": run_id,
        "phase_name": phase_name,
    })[:20]
    root = Path("/tmp") / f"emv21-{phase_identity}"
    control_socket = root / run_id / "node-0" / "control" / "ndp.sock"
    if len(os.fsencode(control_socket)) >= LINUX_SUN_PATH_BYTES:
        raise ValueError(
            "qualification node-local control socket exceeds Linux sun_path")
    return str(root)


def _verify_canonical_config() -> None:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = value.get("seed") if isinstance(value, Mapping) else None
    if (
        not isinstance(seed, Mapping)
        or seed.get("step") != SEED_STEP
        or seed.get("tokens") != SEED_ACCEPTED_TOKENS
        or seed.get("size") != SEED_BYTES
        or seed.get("sha256") != SEED_SHA256
    ):
        raise ValueError("canonical E97 config/seed identity drifted")


def _digest(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return text


def _load_json(path: str | Path, *, schema: str) -> dict[str, object]:
    target = Path(path).resolve()
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{schema} manifest identity is required")
    encoded_digest = value.get("manifest_digest")
    unsigned = {key: item for key, item in value.items()
                if key != "manifest_digest"}
    if encoded_digest != canonical_digest(unsigned):
        raise ValueError(f"{schema} manifest digest mismatch")
    return value


def _verify_prior_clean_gate(
    path: str | Path,
    *,
    expected_identities: Mapping[str, str],
) -> dict[str, object]:
    """Verify the scheduler-owned clean pass and its exact submitted payload."""
    target = Path(path).resolve()
    terminal = _load_json(
        target, schema="emender-async-v21-terminal-verdict-v1")
    scheduler = terminal.get("scheduler")
    validator_inputs = terminal.get("validator_inputs")
    payload_reference = (
        validator_inputs.get("payload")
        if isinstance(validator_inputs, Mapping) else None
    )
    semantic = (
        validator_inputs.get("semantic_verdict")
        if isinstance(validator_inputs, Mapping) else None
    )
    if (
        terminal.get("passed") is not True
        or terminal.get("verdict") != "passed"
        or not isinstance(scheduler, Mapping)
        or scheduler.get("state") != "COMPLETED"
        or scheduler.get("exit_code") != "0:0"
        or scheduler.get("derived_exit_code") != "0:0"
        or scheduler.get("nodes") != 2
        or scheduler.get("partition") != "batch"
        or scheduler.get("qos") != "debug"
        or not isinstance(semantic, Mapping)
        or semantic.get("required") is not True
        or semantic.get("passed") is not True
        or not isinstance(payload_reference, Mapping)
    ):
        raise ValueError(
            "fault qualification requires a semantic clean pass at "
            "Nodes=2, Partition=batch, QOS=debug")
    payload_path = Path(str(payload_reference.get("path", ""))).resolve()
    if (
        not payload_path.is_file()
        or payload_reference.get("sha256") != _file_sha256(payload_path)
        or payload_reference.get("bytes") != payload_path.stat().st_size
    ):
        raise ValueError("prior clean payload input is missing or changed")
    payload_input = json.loads(payload_path.read_text(encoding="utf-8"))
    payload = (
        payload_input.get("payload")
        if isinstance(payload_input, Mapping) else None
    )
    submitted_scheduler = (
        payload_input.get("scheduler")
        if isinstance(payload_input, Mapping) else None
    )
    if (
        payload_input.get("schema")
        != "emender-async-v21-collector-input-v1"
        or payload_input.get("payload_digest")
        != terminal.get("payload_digest")
        or not isinstance(payload, Mapping)
        or payload.get("schema") != PAYLOAD_SCHEMA
        or payload.get("gate") != "clean"
        or payload.get("nodes") != 2
        or payload.get("identities") != dict(expected_identities)
        or not isinstance(submitted_scheduler, Mapping)
        or submitted_scheduler.get("Nodes") != 2
        or submitted_scheduler.get("Partition") != "batch"
        or submitted_scheduler.get("QOS") != "debug"
    ):
        raise ValueError(
            "prior clean pass does not bind this exact two-node payload")
    return {
        "path": str(target),
        "sha256": _file_sha256(target),
        "payload_digest": str(terminal["payload_digest"]),
        "payload_job_id": str(terminal["payload_job_id"]),
        "identities": dict(expected_identities),
        "training_inputs": dict(payload.get("training_inputs", {})),
    }


def _next_fault_phase(
    state_path: str | Path,
    *,
    campaign_digest: str,
    prior_gate: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the only phase eligible to reconcile or submit.

    Terminal passing phases advance in the fixed order.  A failed/retired
    phase is a permanent stop for this campaign digest; later phases cannot be
    rendered, much less submitted.
    """
    _digest(campaign_digest, "fault campaign")
    state = _state(Path(state_path).resolve())
    records = [
        value for value in state["payloads"].values()
        if (
            isinstance(value, Mapping)
            and value.get("campaign_digest") == campaign_digest
        )
    ]
    by_index: dict[int, Mapping[str, object]] = {}
    for record in records:
        if record.get("prior_payload_digest") != prior_gate.get(
                "payload_digest"):
            raise ValueError("fault campaign prior clean identity changed")
        index = int(record.get("fault_phase_index", -1))
        if index in by_index:
            raise ValueError("fault campaign has duplicate phase payloads")
        if index < 0 or index >= len(FAULT_PHASE_SPECS):
            raise ValueError("fault campaign phase index is invalid")
        by_index[index] = record
    for index, phase in enumerate(FAULT_PHASE_SPECS):
        record = by_index.get(index)
        if record is None:
            if any(later > index for later in by_index):
                raise ValueError("fault campaign phase history is out of order")
            return dict(phase)
        if record.get("fault_phase") != phase["name"]:
            raise ValueError("fault campaign phase identity changed")
        status = str(record.get("status", ""))
        if status == "retired" or record.get("verdict") == "failed":
            raise ValueError(
                f"fault phase {phase['name']} failed; later phases are blocked")
        if status == "terminal" and record.get("verdict") == "passed":
            continue
        return dict(phase)
    return None


def _verify_review_signature(
    value: Mapping[str, object],
    *,
    trusted_reviewer_key: str | Path | None,
    allow_test_signatures: bool,
) -> None:
    signature = value.get("review_signature")
    if allow_test_signatures:
        if signature != "signed-for-test":
            raise ValueError("signed review attestation is required")
        return
    if not isinstance(signature, Mapping):
        raise ValueError("signed Ed25519 review attestation is required")
    if (
        signature.get("algorithm") != "ed25519"
        or trusted_reviewer_key is None
    ):
        raise ValueError("trusted Ed25519 reviewer key is required")
    public_bytes = Path(trusted_reviewer_key).resolve().read_bytes()
    key_digest = hashlib.sha256(public_bytes).hexdigest()
    if signature.get("public_key_sha256") != key_digest:
        raise ValueError("review signature does not use the trusted key")
    try:
        encoded = base64.b64decode(
            str(signature["signature_base64"]), validate=True)
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        key = load_pem_public_key(public_bytes)
        signed = {
            key_name: item
            for key_name, item in value.items()
            if key_name not in {"manifest_digest", "review_signature"}
        }
        key.verify(
            encoded,
            b"emender-async-v21-review-v1\0" + canonical_bytes(signed),
        )
    except (KeyError, ValueError, TypeError, ImportError) as error:
        raise ValueError("invalid Ed25519 review signature") from error
    except Exception as error:
        raise ValueError("invalid Ed25519 review signature") from error


def _manifest_reference(
    reference: object,
    *,
    evidence_root: Path,
    schema: str,
    samples_field: str,
) -> tuple[dict[str, object], list[int]]:
    if not isinstance(reference, Mapping):
        raise ValueError("V21S17 evidence reference is missing")
    relative = Path(str(reference.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("V21S17 evidence path must stay under retained evidence")
    path = (evidence_root / relative).resolve()
    path.relative_to(evidence_root.resolve())
    value = _load_json(path, schema=schema)
    if (
        value.get("status") != "passed"
        or value.get("nodes") != 2
        or reference.get("digest") != value.get("manifest_digest")
    ):
        raise ValueError("V21S17 evidence is not a digested passed two-node record")
    samples = value.get(samples_field)
    if (
        not isinstance(samples, list)
        or len(samples) < 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in samples
        )
    ):
        raise ValueError(f"V21S17 {samples_field} evidence is unsupported")
    return value, [int(item) for item in samples]


def _quantile(samples: Sequence[int], numerator: int, denominator: int) -> int:
    if (
        not samples
        or numerator <= 0
        or denominator <= 0
        or numerator > denominator
    ):
        raise ValueError("unsupported empirical quantile")
    ordered = sorted(int(item) for item in samples)
    index = max(0, math.ceil(len(ordered) * numerator / denominator) - 1)
    return ordered[index]


def _scaled(value: int, numerator: int, denominator: int) -> int:
    if value <= 0 or numerator < denominator or denominator <= 0:
        raise ValueError("V21S17 margin must be finite and at least one")
    return (value * numerator + denominator - 1) // denominator


def validate_scale_evidence(
    closure: Mapping[str, object],
    *,
    evidence_root: str | Path,
    ready_count: int,
) -> dict[str, object]:
    """Recompute the finite V21S17 arithmetic from retained 2-node samples."""
    if (
        not isinstance(closure, Mapping)
        or closure.get("schema") != CLOSURE_SCHEMA
        or closure.get("ready_snapshot_source")
        != "leased-ready-at-group-open"
        or closure.get("include_all_complete_admissible_preclose") is not True
        or closure.get("close_on_q_min") is not False
        or closure.get("uses_launched_ranks") is not False
        or closure.get("wait_for_all_ready") is not False
        or isinstance(ready_count, bool)
        or ready_count < 2
    ):
        raise ValueError(
            "V21S17 evidence-derived leased READY closure is required; "
            "launched ranks/Q_min early close/unbounded all-ready are forbidden")
    root = Path(evidence_root).resolve()
    _arrival_value, arrivals = _manifest_reference(
        closure.get("arrival_evidence"),
        evidence_root=root,
        schema="emender-async-v21-two-node-arrivals-v1",
        samples_field="samples_ns",
    )
    stage_value, stages = _manifest_reference(
        closure.get("stage_evidence"),
        evidence_root=root,
        schema="emender-async-v21-two-node-stages-v1",
        samples_field="close_to_latest_ns",
    )
    cadence = stage_value.get("cadence_ns")
    if (
        not isinstance(cadence, list)
        or len(cadence) < 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in cadence
        )
    ):
        raise ValueError("V21S17 cadence derivation lacks passed two-node evidence")
    quantile = closure.get("quantile")
    margin = closure.get("margin")
    if not isinstance(quantile, Mapping) or not isinstance(margin, Mapping):
        raise ValueError("V21S17 quantile/margin arithmetic is missing")
    qn, qd = int(quantile.get("numerator", 0)), int(
        quantile.get("denominator", 0))
    mn, md = int(margin.get("numerator", 0)), int(
        margin.get("denominator", 0))
    arrival_q = _quantile(arrivals, qn, qd)
    stage_q = _quantile(stages, qn, qd)
    cadence_q = _quantile([int(item) for item in cadence], qn, qd)
    close_offset_ns = _scaled(max(arrival_q, cadence_q), mn, md)
    stage_deadline_ns = _scaled(stage_q, mn, md)
    cadence_deadline_ns = _scaled(cadence_q, mn, md)
    diversity = int(closure.get("stable_diversity_floor", 0))
    tokens_per_worker = int(closure.get("per_ready_worker_token_floor", 0))
    if diversity < 2 or diversity > ready_count or tokens_per_worker <= 0:
        raise ValueError("V21S17 stable diversity/exact-token floor is invalid")
    derived = {
        "schema": "emender-v21s17-derived-close-v1",
        "ready_snapshot_source": "leased-ready-at-group-open",
        "ready_count": int(ready_count),
        "close_offset_ns": close_offset_ns,
        "stage_deadline_ns": stage_deadline_ns,
        "cadence_deadline_ns": cadence_deadline_ns,
        "stable_diversity_floor": diversity,
        "per_ready_worker_token_floor": tokens_per_worker,
        "exact_token_floor": tokens_per_worker * ready_count,
        "include_all_complete_admissible_preclose": True,
        "close_on_q_min": False,
        "uses_launched_ranks": False,
        "wait_for_all_ready": False,
        "arrival_evidence_digest": closure["arrival_evidence"]["digest"],
        "stage_evidence_digest": closure["stage_evidence"]["digest"],
        "arithmetic": {
            "quantile": {"numerator": qn, "denominator": qd},
            "margin": {"numerator": mn, "denominator": md},
            "close": "ceil(max(arrival_quantile,cadence_quantile)*margin)",
            "stage": "ceil(close_to_latest_quantile*margin)",
            "cadence": "ceil(cadence_quantile*margin)",
        },
    }
    expected = closure.get("derived")
    if expected is not None and expected != derived:
        raise ValueError("V21S17 stored derivation differs from empirical arithmetic")
    return derived


@dataclass
class V21ScaleClosure:
    """One finite group close evaluated only over an open READY snapshot."""

    open_time_ns: int
    ready_snapshot: Mapping[str, str]
    derived: Mapping[str, object]

    def __post_init__(self) -> None:
        if (
            self.open_time_ns < 0
            or len(self.ready_snapshot) < 2
            or any(not worker or not incarnation
                   for worker, incarnation in self.ready_snapshot.items())
            or self.derived.get("ready_snapshot_source")
            != "leased-ready-at-group-open"
            or int(self.derived.get("ready_count", -1))
            != len(self.ready_snapshot)
        ):
            raise ValueError("scale close requires the exact leased READY snapshot")
        self.close_time_ns = self.open_time_ns + int(
            self.derived["close_offset_ns"])
        if self.close_time_ns <= self.open_time_ns:
            raise ValueError("scale close must be finite and after group open")
        self._complete: dict[str, dict[str, object]] = {}

    def can_close(
        self, now_ns: int, *, complete_workers: set[str] | None = None,
    ) -> bool:
        del complete_workers
        # The reviewed close is immutable at group open.  Neither Q_min nor
        # all-READY completion changes it.
        return int(now_ns) >= self.close_time_ns

    def record(
        self,
        worker_id: str,
        incarnation: str,
        received_ns: int,
        *,
        exact_tokens: int,
    ) -> str:
        if (
            self.ready_snapshot.get(worker_id) != incarnation
            or exact_tokens <= 0
        ):
            raise ValueError("contribution is not admissible in the leased snapshot")
        if received_ns > self.close_time_ns:
            return "late"
        value = {
            "worker_id": worker_id,
            "incarnation": incarnation,
            "received_ns": int(received_ns),
            "exact_tokens": int(exact_tokens),
        }
        old = self._complete.get(worker_id)
        if old is not None:
            if old != value:
                raise ValueError("conflicting stable-worker contribution")
            return "duplicate"
        self._complete[worker_id] = value
        return "accepted"

    def freeze(self, now_ns: int) -> tuple[dict[str, object], ...]:
        if not self.can_close(now_ns):
            raise ValueError("reviewed finite close has not arrived")
        frozen = tuple(
            self._complete[worker] for worker in sorted(self._complete))
        if len(frozen) < int(self.derived["stable_diversity_floor"]):
            raise ValueError("scale stable-diversity safety floor is not met")
        if sum(int(item["exact_tokens"]) for item in frozen) < int(
                self.derived["exact_token_floor"]):
            raise ValueError("scale exact-token safety floor is not met")
        return frozen


def _state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"schema": STATE_SCHEMA, "payloads": {}, "active_job": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") not in {STATE_SCHEMA, LEGACY_STATE_SCHEMA}
        or not isinstance(value.get("payloads"), dict)
    ):
        raise ValueError("retained qualification state identity is invalid")
    value["schema"] = STATE_SCHEMA
    for record in value["payloads"].values():
        if isinstance(record, dict) and record.get("status") == "failed":
            record["status"] = "retired"
            record.setdefault("verdict", "failed")
    return value


def _identities_match(
    value: Mapping[str, object], expected: Mapping[str, str],
) -> bool:
    actual = value.get("identities")
    return isinstance(actual, Mapping) and dict(actual) == dict(expected)


def build_plan(
    *,
    gate: str,
    nodes: int,
    state_path: str | Path,
    evidence_root: str | Path,
    source_digest: str,
    policy_digest: str,
    bundle_digest: str,
    seed_digest: str,
    launcher_digest: str,
    parameters: Mapping[str, object],
    authorization_path: str | Path | None = None,
    predecessor_path: str | Path | None = None,
    trusted_reviewer_key: str | Path | None = None,
    allow_test_signatures: bool = False,
    clean_launch: Mapping[str, object] | None = None,
    fault_launch: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _verify_canonical_config()
    if gate not in ALL_GATES:
        raise ValueError(f"unsupported qualification gate: {gate}")
    if isinstance(nodes, bool) or nodes <= 0:
        raise ValueError("node count must be positive")
    if gate in TWO_NODE_GATES and nodes != 2:
        raise ValueError(f"{gate} qualification is exactly two nodes")
    if gate == "scale" and nodes not in SCALE_RUNGS:
        raise ValueError("scale nodes must follow 4->8->16->32->64->256")
    if gate == "scale" and (
        parameters.get("close_on_q_min") is not False
        or parameters.get("uses_launched_ranks") is not False
        or parameters.get("wait_for_all_ready") is not False
    ):
        raise ValueError(
            "scale parameters must explicitly reject Q_min early close, "
            "launched-rank membership, and an all-READY barrier")
    if clean_launch is not None and gate != "clean":
        raise ValueError("the reviewed clean launch applies only to the clean gate")
    if fault_launch is not None and gate != "faults":
        raise ValueError("the reviewed fault launch applies only to the fault gate")
    if clean_launch is not None and fault_launch is not None:
        raise ValueError("only one production launch profile may be supplied")
    launch: dict[str, str] | None = None
    rendered_parameters = dict(parameters)
    production_launch = (
        clean_launch if clean_launch is not None else fault_launch)
    if production_launch is not None:
        required_launch = {
            "repo", "seed_config", "native_build_manifest",
            "full_layout_gate", "run_dir", "acceptance_manifest",
            "seed_cache", "seed_attestation", "seed_attestation_sha256",
            "train_args", "data", "data_identity_digest", "tokenizer",
            "tokenizer_sha256", "source_commit", "native_source_commit",
            "execution_source_digest", "execution_source_schema",
            "native_build_manifest_sha256", "full_layout_gate_sha256",
            "train_args_sha256",
        }
        missing = sorted(
            name for name in required_launch if not production_launch.get(name))
        if missing:
            raise ValueError(
                "reviewed production launch is missing immutable bindings: "
                + ", ".join(missing))
        launch = {name: str(value) for name, value in production_launch.items()}
        for name in (
            "seed_attestation_sha256", "data_identity_digest",
            "execution_source_digest",
            "tokenizer_sha256", "native_build_manifest_sha256",
            "full_layout_gate_sha256", "train_args_sha256",
        ):
            _digest(launch[name], name)
        if (
            len(launch["source_commit"]) != 40
            or any(character not in "0123456789abcdef"
                   for character in launch["source_commit"])
        ):
            raise ValueError("production launch source commit must be a Git SHA-1")
        if launch["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError(
                "production launch must bind the reviewed p50k tokenizer")
        if launch["execution_source_schema"] != EXECUTION_SOURCE_SCHEMA:
            raise ValueError(
                "production launch execution-source schema is unsupported")
        if (
            len(launch["native_source_commit"]) != 40
            or any(character not in "0123456789abcdef"
                   for character in launch["native_source_commit"])
        ):
            raise ValueError(
                "production launch native source commit must be a Git SHA-1")
        if clean_launch is not None:
            conflicting = {
                name: value
                for name, value in rendered_parameters.items()
                if name in CLEAN_PARAMETERS and value != CLEAN_PARAMETERS[name]
            }
            if conflicting:
                raise ValueError(
                    "clean parameters differ from the reviewed acceptance profile")
            rendered_parameters = {**CLEAN_PARAMETERS, **rendered_parameters}
        else:
            required_fault = {
                "fault_campaign_digest", "fault_phase", "fault_phase_index",
                "initial_generation", "generations", "coordinator_epoch",
                "prior_gate", "prior_gate_sha256", "prior_payload_digest",
            }
            missing_fault = sorted(
                name for name in required_fault if not launch.get(name))
            if missing_fault:
                raise ValueError(
                    "reviewed fault launch is missing campaign bindings: "
                    + ", ".join(missing_fault))
            _digest(launch["fault_campaign_digest"], "fault campaign")
            _digest(launch["prior_gate_sha256"], "prior clean gate")
            phase_index = int(launch["fault_phase_index"])
            if (
                phase_index < 0
                or phase_index >= len(FAULT_PHASE_SPECS)
                or FAULT_PHASE_SPECS[phase_index]["name"]
                != launch["fault_phase"]
                or int(launch["initial_generation"])
                != FAULT_PHASE_SPECS[phase_index]["initial_generation"]
                or int(launch["generations"])
                != FAULT_PHASE_SPECS[phase_index]["generations"]
            ):
                raise ValueError("reviewed fault phase differs from fixed campaign")
            rendered_parameters = {
                "fault_phase": launch["fault_phase"],
                "fault_phase_index": phase_index,
                "initial_generation": int(launch["initial_generation"]),
                "generations": int(launch["generations"]),
                "minimum_commits":
                    FAULT_PHASE_SPECS[phase_index]["minimum_commits"],
                "fresh_allocation":
                    FAULT_PHASE_SPECS[phase_index]["fresh_allocation"],
                "q_min": 2,
                "one_node_commit_authority": False,
                "max_speculative_windows": 2,
                "scenarios": list(FAULT_SCENARIOS),
                "requirements": FAULT_REQUIREMENTS,
                **rendered_parameters,
            }
    identities = {
        "source_digest": _digest(source_digest, "source"),
        "policy_digest": _digest(policy_digest, "policy"),
        "bundle_digest": _digest(bundle_digest, "bundle"),
        "seed_digest": _digest(seed_digest, "seed"),
        "launcher_digest": _digest(launcher_digest, "launcher"),
    }
    if identities["seed_digest"] != SEED_SHA256:
        raise ValueError("qualification payload must bind the canonical E97 seed")
    if (
        launch is not None
        and identities["source_digest"] != launch["execution_source_digest"]
    ):
        raise ValueError(
            "qualification payload source differs from reviewed execution source")
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "policy_id": POLICY_ID,
        "policy_schema": POLICY_SCHEMA,
        "gate": gate,
        "nodes": int(nodes),
        "identities": identities,
        "parameters": rendered_parameters,
        "config": launch["seed_config"] if launch else str(CONFIG_PATH),
        "seed": {
            "step": SEED_STEP,
            "accepted_tokens": SEED_ACCEPTED_TOKENS,
            "bytes": SEED_BYTES,
            "sha256": SEED_SHA256,
            "node_path": SEED_NODE_PATH,
            "compute_node_network_fetches": 0,
        },
    }
    if launch is not None:
        training_inputs = {
            "data_identity_digest": launch["data_identity_digest"],
            "execution_source_digest": launch["execution_source_digest"],
            "execution_source_schema": launch["execution_source_schema"],
            "full_layout_gate_sha256": launch["full_layout_gate_sha256"],
            "native_build_manifest_sha256":
                launch["native_build_manifest_sha256"],
            "native_source_commit": launch["native_source_commit"],
            "seed_config": launch["seed_config"],
            "seed_attestation_sha256": launch["seed_attestation_sha256"],
            "source_commit": launch["source_commit"],
            "tokenizer_sha256": launch["tokenizer_sha256"],
            "train_args": launch["train_args"],
            "train_args_sha256": launch["train_args_sha256"],
        }
        for name in (
            "data_bytes", "data_mtime_ns",
        ):
            if name in launch:
                training_inputs[name] = (
                    int(launch[name])
                    if name in {"data_bytes", "data_mtime_ns"}
                    else launch[name]
                )
        payload["training_inputs"] = training_inputs
        if gate == "faults":
            payload["prior_gate"] = {
                "path": str(Path(launch["prior_gate"]).resolve()),
                "sha256": launch["prior_gate_sha256"],
                "payload_digest": launch["prior_payload_digest"],
            }
            payload["fault_campaign_digest"] = launch[
                "fault_campaign_digest"]
    payload_digest = canonical_digest(payload)
    retained_state = _state(Path(state_path).resolve())
    old = retained_state["payloads"].get(payload_digest)
    if isinstance(old, Mapping) and old.get("status") == "retired":
        raise ValueError("unchanged failed payload cannot be resubmitted")
    active = retained_state.get("active_job")
    if (
        isinstance(active, Mapping)
        and active.get("job_id")
        and active.get("payload_digest") != payload_digest
    ):
        raise ValueError("at most one active job is permitted")

    scale_evidence = None
    if gate == "scale":
        if authorization_path is None or predecessor_path is None:
            raise ValueError(
                "signed scale authorization and immediate predecessor pass "
                "are required")
        authorization = _load_json(
            authorization_path, schema=AUTHORIZATION_SCHEMA)
        _verify_review_signature(
            authorization,
            trusted_reviewer_key=trusted_reviewer_key,
            allow_test_signatures=allow_test_signatures,
        )
        if (
            authorization.get("status") != "passed"
            or authorization.get("authorized_nodes") != nodes
            or not _identities_match(authorization, identities)
        ):
            raise ValueError("scale authorization does not bind this exact payload/rung")
        predecessor = _load_json(predecessor_path, schema=RUNG_PASS_SCHEMA)
        _verify_review_signature(
            predecessor,
            trusted_reviewer_key=trusted_reviewer_key,
            allow_test_signatures=allow_test_signatures,
        )
        required_predecessor = PREDECESSOR[nodes]
        if predecessor.get("nodes") != required_predecessor:
            raise ValueError(
                f"exact immediate predecessor {required_predecessor} pass is required")
        if (
            predecessor.get("status") != "passed"
            or not _identities_match(predecessor, identities)
        ):
            raise ValueError("immediate predecessor is not a bound passing manifest")
        reviewed_ready_count = int(
            authorization.get("reviewed_ready_snapshot_size", 0))
        scale_evidence = validate_scale_evidence(
            authorization.get("closure", {}),
            evidence_root=evidence_root,
            ready_count=reviewed_ready_count,
        )
        payload["scale_authorization_digest"] = authorization["manifest_digest"]
        payload["predecessor_pass_digest"] = predecessor["manifest_digest"]
        payload["scale_closure"] = scale_evidence
        payload_digest = canonical_digest(payload)
        old = retained_state["payloads"].get(payload_digest)
        if isinstance(old, Mapping) and old.get("status") == "retired":
            raise ValueError("unchanged failed payload cannot be resubmitted")

    scheduler = {"Nodes": nodes, "Partition": "batch", "QOS": "debug"}
    if launch is not None:
        scheduler["TimeLimit"] = (
            CLEAN_WALLTIME if gate == "clean" else FAULT_WALLTIME)
    config_path = launch["seed_config"] if launch else str(CONFIG_PATH)
    exports = [
        "ALL",
        f"ASYNC_V21_GATE={gate}",
        f"ASYNC_V21_PAYLOAD_DIGEST={payload_digest}",
        f"ASYNC_V21_POLICY_DIGEST={policy_digest}",
        f"ASYNC_V21_CONFIG={config_path}",
        f"RESILIENT_E97_NODE_COUNT={nodes}",
        f"RESILIENT_E97_DILOCO_POLICY={POLICY_ID}",
        "RESILIENT_E97_GLOBAL_TOKEN_MIN=3934080",
        "RESILIENT_E97_ETA_OUTER=1.0",
        "RESILIENT_E97_MAX_COMMIT_LAG=2",
        "RESILIENT_E97_MAX_ANCHOR_LAG=2",
        "RESILIENT_E97_MAX_RESULT_LAG=2",
        "RESILIENT_E97_MAX_SPECULATIVE_WINDOWS=2",
        "RESILIENT_E97_COMPUTE_NODE_NETWORK_FETCHES=0",
    ]
    if launch is not None:
        execution_digest = launch.get("execution_source_digest", source_digest)
        phase_name = (
            CLEAN_PHASE if gate == "clean" else launch["fault_phase"])
        campaign_identity = (
            payload_digest
            if gate == "clean"
            else launch["fault_campaign_digest"]
        )
        run_id = f"async-v21-{gate}-{campaign_identity[:16]}"
        bulk_root = _qualification_bulk_root(
            run_id=run_id,
            phase_name=phase_name,
        )
        generations = (
            CLEAN_GENERATIONS
            if gate == "clean" else int(launch["generations"])
        )
        initial_generation = (
            0 if gate == "clean" else int(launch["initial_generation"])
        )
        coordinator_epoch = (
            1 if gate == "clean" else int(launch["coordinator_epoch"])
        )
        max_restarts = (
            0 if gate == "clean"
            else int(FAULT_PHASE_SPECS[int(
                launch["fault_phase_index"])]["max_restarts"])
        )
        exports.extend([
            f"REPO={launch['repo']}",
            "RESILIENT_E97_ACCEPTANCE_MANIFEST="
            + launch["acceptance_manifest"],
            f"RESILIENT_E97_ACCEPTANCE_PHASE={phase_name}",
            f"RUN_DIR={launch['run_dir']}",
            f"NDP_BUILD_MANIFEST={launch['native_build_manifest']}",
            f"NDP_FULL_LAYOUT_GATE_JSON={launch['full_layout_gate']}",
            "NDP_REQUIRED_GATE="
            + ("G2" if gate == "clean" else "G2-fault-rejoin-replay"),
            f"EMENDER_CONDA_ENV={APPROVED_ENV}",
            "DILOCO_DATAPLANE=native-cxi",
            "FI_PROVIDER=cxi",
            "FI_MR_CACHE_MONITOR=kdreg2",
            "FI_CXI_ATS=0",
            f"RESILIENT_E97_RUN_ID={run_id}",
            "RESILIENT_E97_SOURCE_ID="
            f"step-{SEED_STEP}-tokens-{SEED_ACCEPTED_TOKENS}-"
            f"sha256-{SEED_SHA256}",
            "RESILIENT_E97_PAYLOAD_ID=" + campaign_identity,
            f"RESILIENT_E97_CODE_ID=execution-sha256-{execution_digest}",
            f"RESILIENT_E97_SEED_CONFIG={launch['seed_config']}",
            f"RESILIENT_E97_SEED_STEP={SEED_STEP}",
            f"RESILIENT_E97_SEED_TOKENS={SEED_ACCEPTED_TOKENS}",
            f"RESILIENT_E97_SEED_SIZE={SEED_BYTES}",
            f"RESILIENT_E97_SEED_SHA256={SEED_SHA256}",
            f"RESILIENT_E97_SEED_CACHE={launch['seed_cache']}",
            f"RESILIENT_E97_SEED_ATTESTATION={launch['seed_attestation']}",
            "RESILIENT_E97_SEED_ATTESTATION_SHA256="
            + launch["seed_attestation_sha256"],
            f"RESILIENT_E97_TRAIN_ARGS_JSON={launch['train_args']}",
            f"RESILIENT_E97_DATA={launch['data']}",
            "RESILIENT_E97_DATA_IDENTITY_DIGEST="
            + launch["data_identity_digest"],
            f"RESILIENT_E97_TIKTOKEN_CACHE_FILE={launch['tokenizer']}",
            f"RESILIENT_E97_TIKTOKEN_SHA256={launch['tokenizer_sha256']}",
            f"RESILIENT_E97_GENERATIONS={generations}",
            f"RESILIENT_E97_INITIAL_GENERATION={initial_generation}",
            f"RESILIENT_E97_COORDINATOR_EPOCH={coordinator_epoch}",
            "RESILIENT_E97_GLOBAL_QUORUM=2",
            "RESILIENT_E97_STARTUP_SMOKE=0",
            "RESILIENT_E97_REQUESTED_WALLTIME="
            + (CLEAN_WALLTIME if gate == "clean" else FAULT_WALLTIME),
            "RESILIENT_E97_LAUNCH_MODE=node-local",
            "RESILIENT_E97_STARTUP_DEADLINE_S=180",
            "RESILIENT_E97_HEARTBEAT_DEADLINE_S=60",
            f"RESILIENT_E97_PROGRESS_DEADLINE_S="
            f"{CLEAN_PROGRESS_DEADLINE_S if gate == 'clean' else FAULT_PROGRESS_DEADLINE_S}",
            f"RESILIENT_E97_GENERATION_DEADLINE_S="
            f"{CLEAN_GENERATION_DEADLINE_S if gate == 'clean' else FAULT_GENERATION_DEADLINE_S}",
            f"RESILIENT_E97_MAX_RESTARTS={max_restarts}",
            f"RESILIENT_E97_BULK_ROOT={bulk_root}",
        ])
        injection_values = (
            {}
            if gate == "clean"
            else FAULT_PHASE_SPECS[int(
                launch["fault_phase_index"])]["injections"]
        )
        for variable in (
            "RESILIENT_E97_INJECT_TRAINER",
            "RESILIENT_E97_INJECT_MANAGER",
            "RESILIENT_E97_INJECT_NODE_STEP",
            "RESILIENT_E97_INJECT_NATIVE_SERVICE",
            "RESILIENT_E97_DELAY_READY",
        ):
            exports.append(f"{variable}={injection_values.get(variable, '')}")
        if gate == "faults":
            exports.extend([
                "ASYNC_V21_PRIOR_CLEAN_GATE=" + launch["prior_gate"],
                "ASYNC_V21_PRIOR_CLEAN_GATE_SHA256="
                + launch["prior_gate_sha256"],
                "ASYNC_V21_FAULT_CAMPAIGN_DIGEST="
                + launch["fault_campaign_digest"],
                "ASYNC_V21_FAULT_PHASE_INDEX="
                + launch["fault_phase_index"],
            ])
            if launch.get("resume_handoff"):
                exports.append(
                    "RESILIENT_E97_RESUME_HANDOFF="
                    + launch["resume_handoff"])
    if scale_evidence is not None:
        exports.extend([
            "ASYNC_V21_SCALE_AUTHORIZATION="
            + str(Path(authorization_path).resolve()),
            "ASYNC_V21_SCALE_AUTHORIZATION_DIGEST="
            + str(payload["scale_authorization_digest"]),
            "ASYNC_V21_PRIOR_RUNG_PASS="
            + str(Path(predecessor_path).resolve()),
            "ASYNC_V21_PRIOR_RUNG_PASS_DIGEST="
            + str(payload["predecessor_pass_digest"]),
            "ASYNC_V21_SCALE_EVIDENCE_ROOT="
            + str(Path(evidence_root).resolve()),
            "ASYNC_V21_SCALE_CLOSURE_DIGEST="
            + canonical_digest(scale_evidence),
            "ASYNC_V21_SCALE_CLOSE_OFFSET_NS="
            + str(scale_evidence["close_offset_ns"]),
            "ASYNC_V21_SCALE_STABLE_DIVERSITY_FLOOR="
            + str(scale_evidence["stable_diversity_floor"]),
            "ASYNC_V21_SCALE_PER_READY_WORKER_TOKEN_FLOOR="
            + str(scale_evidence["per_ready_worker_token_floor"]),
        ])
        if trusted_reviewer_key is not None:
            exports.append(
                "ASYNC_V21_TRUSTED_REVIEWER_KEY="
                + str(Path(trusted_reviewer_key).resolve()))
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--nodes={nodes}",
        "--partition=batch",
        "--qos=debug",
        f"--job-name=em-v21-{gate}-{payload_digest[:12]}",
        "--comment=emender-v21-payload:"
        f"{payload_digest}:{EXECUTION_SOURCE_SCHEMA}",
    ]
    launcher_path = LAUNCHER_PATH
    evidence_path = Path(evidence_root).resolve()
    model_log_root = evidence_path / "payloads" / payload_digest
    if launch is not None:
        launcher_path = Path(launch["repo"]) / (
            "scripts/frontier/resilient_e97_true_2n.sbatch")
        run_dir = Path(launch["run_dir"])
        model_log_root = run_dir
        walltime = CLEAN_WALLTIME if gate == "clean" else FAULT_WALLTIME
        signal_spec = CLEAN_SIGNAL if gate == "clean" else FAULT_SIGNAL
        command.extend([
            f"--time={walltime}",
            f"--signal={signal_spec}",
            "--network=job_vni",
            f"--chdir={launch['repo']}",
            f"--output={run_dir / 'slurm-%j.out'}",
            f"--error={run_dir / 'slurm-%j.err'}",
        ])
    else:
        command.extend([
            f"--chdir={ROOT}",
            f"--output={model_log_root / 'slurm-%j.out'}",
            f"--error={model_log_root / 'slurm-%j.err'}",
        ])
    command.extend([
        f"--export={','.join(exports)}",
        str(launcher_path),
    ])
    plan = {
        "payload": payload,
        "payload_digest": payload_digest,
        "scheduler": scheduler,
        "command": command,
        "state_path": str(Path(state_path).resolve()),
        "evidence_root": str(Path(evidence_root).resolve()),
        "repo": launch["repo"] if launch else str(ROOT),
        "collector": {
            "schema": COLLECTOR_SCHEMA,
            "evidence_dir": str(
                evidence_path / "terminal-collector" / payload_digest),
            "stdout_pattern": str(model_log_root / "slurm-%j.out"),
            "stderr_pattern": str(model_log_root / "slurm-%j.err"),
            "semantic_verdict": (
                str(model_log_root / "pipelined-performance.json")
                if launch is not None and gate == "clean"
                else (
                    str(
                        model_log_root
                        / f"{launch['fault_phase']}-verdict.json")
                    if launch is not None and gate == "faults"
                    else None
                )
            ),
            "scheduler_owned": True,
            "dependency": "afterany",
            "requires_wg_or_codex": False,
        },
    }
    if launch is not None:
        plan[
            "clean_launch" if gate == "clean" else "fault_launch"
        ] = launch
    return plan


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _verify_clean_plan_immutable(plan: Mapping[str, object]) -> None:
    launch = plan.get("clean_launch") or plan.get("fault_launch")
    if not isinstance(launch, Mapping):
        return
    repo = Path(str(launch["repo"])).resolve()
    _require_submission_source(repo)
    current_source = _source_digest(repo)
    if current_source["digest"] != launch["execution_source_digest"]:
        raise ValueError(
            "authoritative execution source changed after plan rendering")
    for path_name, digest_name in (
        ("native_build_manifest", "native_build_manifest_sha256"),
        ("full_layout_gate", "full_layout_gate_sha256"),
        ("seed_attestation", "seed_attestation_sha256"),
        ("train_args", "train_args_sha256"),
        ("tokenizer", "tokenizer_sha256"),
    ):
        if _file_sha256(Path(str(launch[path_name]))) != launch[digest_name]:
            raise ValueError(
                f"launch artifact changed after rendering: {path_name}")
    data = _data_identity(Path(str(launch["data"])))
    if data["identity_digest"] != launch["data_identity_digest"]:
        raise ValueError("reviewed E97 data object changed after rendering")
    cache = Path(str(launch["seed_cache"]))
    if (
        not cache.is_file()
        or cache.stat().st_size != SEED_BYTES
        or cache.name != f"sha256-{SEED_SHA256}.pt"
    ):
        raise ValueError("verified content-addressed seed cache is unavailable")
    if "prior_gate" in launch and (
        _file_sha256(Path(str(launch["prior_gate"])))
        != launch.get("prior_gate_sha256")
    ):
        raise ValueError("prior clean terminal verdict changed after rendering")


def _command_option(command: Sequence[object], prefix: str) -> str:
    matches = [
        str(item)[len(prefix):]
        for item in command
        if str(item).startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"scheduler command requires exactly one {prefix} option")
    return matches[0]


def _find_scheduler_job(*, name: str, comment: str) -> str | None:
    """Recover a prior scheduler side effect by its deterministic identity."""
    found: set[str] = set()
    queued = subprocess.run(
        ["squeue", "-h", "--name", name, "-o", "%i|%j|%k"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for line in queued.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[1] == name and fields[2] == comment:
            found.add(fields[0].split(".", 1)[0])
    accounting = subprocess.run(
        [
            "sacct", "-n", "-X", "--name", name,
            "--format=JobIDRaw,JobName,Comment,State", "-P",
        ],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    for line in accounting.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 4 and fields[1] == name and fields[2] == comment:
            found.add(fields[0].split(".", 1)[0])
    found = {job_id for job_id in found if job_id.isdigit()}
    if len(found) > 1:
        raise ValueError(
            f"duplicate scheduler identities exist for {name}: {sorted(found)}")
    return next(iter(found), None)


def _collector_registration(
    plan: Mapping[str, object],
    *,
    payload_job_id: str,
    payload_input: Path,
) -> tuple[list[str], dict[str, object]]:
    collector = plan.get("collector")
    if (
        not isinstance(collector, Mapping)
        or collector.get("schema") != COLLECTOR_SCHEMA
    ):
        raise ValueError("durable scheduler-owned collector plan is required")
    script = (
        Path(str(plan.get("repo", ROOT))).resolve()
        / "scripts/frontier/async_v21_terminal_collector.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"terminal collector is missing: {script}")
    evidence_dir = Path(str(collector["evidence_dir"])).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    wrap_argv = [
        sys.executable,
        str(script),
        "--state", str(Path(str(plan["state_path"])).resolve()),
        "--payload-digest", str(plan["payload_digest"]),
        "--payload-job-id", payload_job_id,
        "--evidence-dir", str(evidence_dir),
        "--payload-input", str(payload_input),
        "--stdout-pattern", str(collector["stdout_pattern"]),
        "--stderr-pattern", str(collector["stderr_pattern"]),
    ]
    semantic = collector.get("semantic_verdict")
    if semantic:
        wrap_argv.extend(["--semantic-verdict", str(semantic)])
    name = f"em-v21-collector-{str(plan['payload_digest'])[:12]}"
    script_sha256 = _file_sha256(script)
    comment = (
        "emender-v21-collector:"
        f"{plan['payload_digest']}:{payload_job_id}:{script_sha256}"
    )
    command = [
        "sbatch",
        "--parsable",
        f"--account={FRONTIER_ACCOUNT}",
        "--nodes=1",
        "--partition=batch",
        f"--qos={COLLECTOR_QOS}",
        "--time=00:10:00",
        f"--job-name={name}",
        f"--comment={comment}",
        f"--dependency=afterany:{payload_job_id}",
        f"--chdir={plan.get('repo', ROOT)}",
        f"--output={evidence_dir / 'collector-%j.out'}",
        f"--error={evidence_dir / 'collector-%j.err'}",
        "--export=NONE",
        f"--wrap={shlex.join(wrap_argv)}",
    ]
    identity = {
        "schema": COLLECTOR_SCHEMA,
        "name": name,
        "comment": comment,
        "dependency": f"afterany:{payload_job_id}",
        "script": str(script),
        "script_sha256": script_sha256,
        "wrap_argv": wrap_argv,
        "wrap_digest": canonical_digest(wrap_argv),
        "evidence_dir": str(evidence_dir),
        "scheduler": {
            "Account": _command_option(command, "--account="),
            "Nodes": int(_command_option(command, "--nodes=")),
            "Partition": _command_option(command, "--partition="),
            "QOS": _command_option(command, "--qos="),
        },
        "scheduler_owned": True,
        "requires_wg_or_codex": False,
    }
    return command, identity


def submit_plan(plan: Mapping[str, object]) -> str:
    """Reconcile one held-payload/afterany-collector/release transaction."""
    state_path = Path(str(plan["state_path"]))
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _state(state_path)
        payload_digest = str(plan["payload_digest"])
        existing = state["payloads"].get(payload_digest)
        if isinstance(existing, Mapping) and existing.get("status") in {
            "terminal", "retired",
        }:
            job_id = str(existing.get("job_id", ""))
            if not job_id.isdigit():
                raise ValueError("terminal payload lacks its scheduler identity")
            return job_id
        if isinstance(existing, Mapping) and existing.get("status") in {
            "released", "queued", "running",
        }:
            job_id = str(existing.get("job_id", ""))
            collector = existing.get("collector")
            if (
                not job_id.isdigit()
                or not isinstance(collector, Mapping)
                or not str(collector.get("job_id", "")).isdigit()
            ):
                raise ValueError(
                    "active payload lacks its durable scheduler/collector identity")
            return job_id
        active = state.get("active_job")
        if (
            isinstance(active, Mapping)
            and active.get("job_id")
            and active.get("payload_digest") != payload_digest
        ):
            raise ValueError("at most one active job is permitted")
        _verify_clean_plan_immutable(plan)
        record = (
            existing
            if isinstance(existing, dict)
            else {"status": "new", "payload_digest": payload_digest}
        )
        job_id = str(record.get("job_id", ""))
        model_command = [str(item) for item in plan["command"]]
        if "--hold" not in model_command:
            raise ValueError("payload must be submitted held")
        model_name = _command_option(model_command, "--job-name=")
        model_comment = _command_option(model_command, "--comment=")
        if not job_id:
            job_id = _find_scheduler_job(
                name=model_name, comment=model_comment) or ""
            if not job_id:
                queued = subprocess.run(
                    ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"],
                    check=True, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout.strip()
                if queued:
                    raise ValueError(
                        "serial qualification refuses to overlap another allocation")
                for pattern_name in ("stdout_pattern", "stderr_pattern"):
                    Path(str(plan["collector"][pattern_name])).parent.mkdir(
                        parents=True, exist_ok=True)
                completed = subprocess.run(
                    model_command,
                    cwd=Path(str(plan.get("repo", ROOT))),
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                job_id = completed.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise RuntimeError(
                        "held sbatch did not return one numeric job id")
            record.update({
                "status": "held",
                "job_id": job_id,
                "payload_digest": payload_digest,
                "model_scheduler_identity": {
                    "name": model_name,
                    "comment": model_comment,
                    "held": True,
                },
            })
            parameters = plan.get("payload", {}).get("parameters", {})
            if (
                isinstance(parameters, Mapping)
                and "fault_phase" in parameters
            ):
                record.update({
                    "campaign_digest":
                        plan["payload"]["fault_campaign_digest"],
                    "fault_phase": parameters["fault_phase"],
                    "fault_phase_index": parameters["fault_phase_index"],
                    "prior_payload_digest":
                        plan["payload"]["prior_gate"]["payload_digest"],
                })
            state["payloads"][payload_digest] = record
            state["active_job"] = {
                "job_id": job_id,
                "payload_digest": payload_digest,
                "status": "held",
            }
            # This is the first durable record containing both the held Slurm
            # job and exact payload identity.
            _atomic_json(state_path, state)

        if not isinstance(state.get("active_job"), dict):
            state["active_job"] = {
                "job_id": job_id,
                "payload_digest": payload_digest,
                "status": str(record.get("status", "held")),
            }
        evidence_dir = Path(str(plan["collector"]["evidence_dir"])).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        payload_input = evidence_dir / "payload-input.json"
        if not payload_input.exists():
            _atomic_json(payload_input, {
                "schema": "emender-async-v21-collector-input-v1",
                "payload_digest": payload_digest,
                "payload": plan["payload"],
                "scheduler": plan["scheduler"],
                "model_command": model_command,
            })

        collector_command, collector_identity = _collector_registration(
            plan, payload_job_id=job_id, payload_input=payload_input)
        collector_record = record.get("collector")
        collector_job_id = (
            str(collector_record.get("job_id", ""))
            if isinstance(collector_record, Mapping)
            else ""
        )
        if not collector_job_id:
            collector_job_id = _find_scheduler_job(
                name=str(collector_identity["name"]),
                comment=str(collector_identity["comment"]),
            ) or ""
            if not collector_job_id:
                try:
                    completed = subprocess.run(
                        collector_command,
                        cwd=Path(str(plan.get("repo", ROOT))),
                        check=True,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                except subprocess.CalledProcessError:
                    record["collector"] = {
                        **collector_identity,
                        "status": "registration-failed",
                    }
                    record["status"] = "held"
                    state["active_job"]["status"] = "held"
                    _atomic_json(state_path, state)
                    raise
                collector_job_id = completed.stdout.strip().split(";", 1)[0]
                if not collector_job_id.isdigit():
                    raise RuntimeError(
                        "collector sbatch did not return one numeric job id")
            record["collector"] = {
                **collector_identity,
                "status": "registered",
                "job_id": collector_job_id,
            }
            record["status"] = "collector-registered"
            state["active_job"]["status"] = "collector-registered"
            # The scheduler-owned collector identity is durable before release.
            _atomic_json(state_path, state)

        if record.get("status") != "released":
            subprocess.run(
                ["scontrol", "release", job_id],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            record["status"] = "released"
            state["active_job"]["status"] = "released"
            _atomic_json(state_path, state)
        return job_id


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_digest(
    repo: Path = ROOT, *, revision: str | None = None,
) -> dict[str, object]:
    """Hash every tracked execution byte except reviewed append-only evidence."""
    digest = hashlib.sha256(
        b"emender-async-v21-execution-source-v1\0")
    included = 0
    excluded = 0
    entries: list[tuple[bytes, bytes]] = []
    if revision is None:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo, check=True, stdout=subprocess.PIPE,
        )
        for encoded in sorted(completed.stdout.split(b"\0")):
            if not encoded:
                continue
            relative = os.fsdecode(encoded)
            if relative.startswith(EVIDENCE_ONLY_PATH_PREFIXES):
                excluded += 1
                continue
            path = repo / relative
            if path.is_symlink():
                payload = os.fsencode(os.readlink(path))
            elif path.is_file():
                payload = path.read_bytes()
            else:
                continue
            entries.append((encoded, payload))
    else:
        completed = subprocess.run(
            ["git", "ls-tree", "-r", "-z", revision, "--"],
            cwd=repo, check=True, stdout=subprocess.PIPE,
        )
        blobs: list[tuple[bytes, bytes]] = []
        for entry in completed.stdout.split(b"\0"):
            if not entry:
                continue
            metadata, encoded = entry.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split()
            relative = os.fsdecode(encoded)
            if relative.startswith(EVIDENCE_ONLY_PATH_PREFIXES):
                excluded += 1
                continue
            if object_type != b"blob":
                continue
            blobs.append((encoded, object_id))
        batch = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repo,
            check=True,
            input=b"".join(object_id + b"\n" for _path, object_id in blobs),
            stdout=subprocess.PIPE,
        ).stdout
        offset = 0
        for (encoded, expected_id) in blobs:
            header_end = batch.index(b"\n", offset)
            header = batch[offset:header_end].split()
            if (
                len(header) != 3
                or header[0] != expected_id
                or header[1] != b"blob"
            ):
                raise ValueError("Git historical execution-source blob mismatch")
            size = int(header[2])
            begin = header_end + 1
            end = begin + size
            if batch[end:end + 1] != b"\n":
                raise ValueError("Git historical execution-source framing mismatch")
            entries.append((encoded, batch[begin:end]))
            offset = end + 1
        if offset != len(batch):
            raise ValueError("Git historical execution-source batch has extra bytes")
    for encoded, payload in entries:
        included += 1
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    boundary = {
        "schema": EXECUTION_SOURCE_SCHEMA,
        "digest": digest.hexdigest(),
        "algorithm": "sha256-domain-separated-path-length-content",
        "evidence_only_path_prefixes": list(EVIDENCE_ONLY_PATH_PREFIXES),
        "included_tracked_files": included,
        "excluded_evidence_files": excluded,
    }
    return boundary


def _require_submission_source(repo: Path = ROOT) -> str:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if (
        git("branch", "--show-current") != "main"
        or git("status", "--porcelain", "--untracked-files=all")
        or commit != git("rev-parse", "origin/main")
    ):
        raise ValueError(
            "submission requires clean authoritative main at origin/main")
    return commit


def _data_identity(path: Path) -> dict[str, object]:
    """Bind the reviewed Lustre object without rereading the one-terabyte corpus."""
    metadata = path.stat()
    value = {
        "schema": "emender-reviewed-data-object-v1",
        "path": str(path.resolve()),
        "bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    value["identity_digest"] = canonical_digest(value)
    return value


def _clean_launch_context(
    *,
    repo: Path,
    source_commit: str,
    seed_config: Path,
    native_build_manifest: Path,
    full_layout_gate: Path,
    run_dir: Path,
    acceptance_manifest: Path,
    submit: bool,
    required_gate: str,
) -> tuple[dict[str, str], str]:
    for path, name in (
        (seed_config, "canonical seed config"),
        (native_build_manifest, "native build manifest"),
        (full_layout_gate, "passed full-layout G2 gate"),
        (repo / "configs/frontier/e97_resilient_split_role_flat.json",
         "approved E97 training arguments"),
        (repo / "scripts/frontier/resilient_e97_true_2n.sbatch",
         "canonical two-node launcher"),
        (DATA_PATH, "reviewed E97 dataset"),
        (TOKENIZER_PATH, "reviewed p50k tokenizer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")

    seed = json.loads(seed_config.read_text(encoding="utf-8")).get("seed")
    if (
        not isinstance(seed, Mapping)
        or seed.get("step") != SEED_STEP
        or seed.get("tokens") != SEED_ACCEPTED_TOKENS
        or seed.get("size") != SEED_BYTES
        or seed.get("sha256") != SEED_SHA256
    ):
        raise ValueError("selected seed config is not the immutable E97 authority")
    build = json.loads(native_build_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(build, Mapping)
        or build.get("schema") != "emender-native-dataplane-build-v1"
        or build.get("source_tree_dirty") is not False
    ):
        raise ValueError(
            "native build manifest is not current clean authoritative source")
    native_source_commit = str(build.get("source_commit", ""))
    current_execution_source = _source_digest(repo)
    if (
        len(native_source_commit) != 40
        or _source_digest(
            repo, revision=native_source_commit)["digest"]
        != current_execution_source["digest"]
    ):
        raise ValueError(
            "native build execution source differs from authoritative source")
    bundle_digest = _digest(build.get("bundle_sha256"), "native bundle")

    from ndm.native_artifacts import attest_launch
    attest_launch(
        backend="native-cxi",
        production=True,
        full_layout=True,
        build_manifest=native_build_manifest,
        gate_json=full_layout_gate,
        # The controller has just recomputed and compared the reviewed
        # execution-source digests for current main and the native build
        # commit.  Native/G2 remain exact to each other, while evidence-only
        # Git commits are intentionally allowed to differ.
        source_root=None,
        required_gate=required_gate,
    )
    if _file_sha256(TOKENIZER_PATH) != TOKENIZER_SHA256:
        raise ValueError("reviewed p50k tokenizer digest mismatch")

    run_dir.mkdir(parents=True, exist_ok=True)
    if submit:
        from scripts.frontier.materialize_e97_s3_seed import prefetch
        seed_cache, seed_attestation = prefetch(
            dict(seed),
            Path(os.environ.get(
                "RESILIENT_E97_SUBMIT_SEED_CACHE_ROOT",
                str(SEED_CACHE_ROOT))),
            run_dir / "seed-bootstrap-attestation.json",
        )
    else:
        seed_cache = SEED_CACHE_ROOT / f"sha256-{SEED_SHA256}.pt"
        seed_attestation = run_dir / "seed-bootstrap-attestation.json"
    data_identity = _data_identity(DATA_PATH)
    train_args = (
        repo / "configs/frontier/e97_resilient_split_role_flat.json").resolve()
    context = {
        "repo": str(repo),
        "source_commit": source_commit,
        "native_source_commit": native_source_commit,
        "execution_source_schema": EXECUTION_SOURCE_SCHEMA,
        "execution_source_digest": str(current_execution_source["digest"]),
        "seed_config": str(seed_config),
        "native_build_manifest": str(native_build_manifest),
        "native_build_manifest_sha256": _file_sha256(native_build_manifest),
        "full_layout_gate": str(full_layout_gate),
        "full_layout_gate_sha256": _file_sha256(full_layout_gate),
        "run_dir": str(run_dir),
        "acceptance_manifest": str(acceptance_manifest),
        "seed_cache": str(seed_cache),
        "seed_attestation": str(seed_attestation),
        "seed_attestation_sha256": (
            _file_sha256(seed_attestation) if submit else "0" * 64),
        "train_args": str(train_args),
        "train_args_sha256": _file_sha256(train_args),
        "data": str(DATA_PATH),
        "data_bytes": str(data_identity["bytes"]),
        "data_mtime_ns": str(data_identity["mtime_ns"]),
        "data_identity_digest": str(data_identity["identity_digest"]),
        "tokenizer": str(TOKENIZER_PATH),
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    return context, bundle_digest


def _fault_resume_handoff(
    *,
    run_dir: Path,
    campaign_digest: str,
    expected_generation: int,
) -> str:
    latest_path = run_dir.resolve() / "handoff" / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(
            "passing prior fault phase did not retain an immutable handoff")
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest = Path(str(latest.get("manifest", ""))).resolve()
    manifest.relative_to((run_dir.resolve() / "handoff").resolve())
    if (
        latest.get("generation") != expected_generation
        or not manifest.is_file()
        or latest.get("manifest_sha256") != _file_sha256(manifest)
    ):
        raise ValueError(
            "fault phase handoff does not match the required generation")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    expected_run = "async-v21-faults-" + campaign_digest[:16]
    expected_tokens = (
        SEED_ACCEPTED_TOKENS + expected_generation * 5_245_440)
    if (
        value.get("finalized") is not True
        or value.get("run_id") != expected_run
        or value.get("payload_id") != campaign_digest
        or value.get("generation") != expected_generation
        or value.get("membership") != ["node-0", "node-1"]
        or value.get("accepted_tokens") != expected_tokens
        or value.get("outer_update_state") != {
            "mode": "delta_sgd",
            "eta_outer": 1.0,
            "step": expected_generation,
            "accepted_tokens": expected_tokens,
        }
        or not Path(str(value.get("checkpoint", ""))).is_file()
        or value.get("checkpoint_sha256")
        != _file_sha256(Path(str(value.get("checkpoint", ""))))
    ):
        raise ValueError(
            "fault phase handoff model/outer/token identity is invalid")
    return str(manifest)


def _fault_campaign_verdict(
    *,
    state_path: Path,
    campaign_digest: str,
    prior_gate: Mapping[str, object],
    identities: Mapping[str, str],
) -> dict[str, object]:
    state = _state(state_path)
    records = [
        record for record in state["payloads"].values()
        if (
            isinstance(record, Mapping)
            and record.get("campaign_digest") == campaign_digest
        )
    ]
    if len(records) != len(FAULT_PHASE_SPECS):
        raise ValueError("fault campaign does not have every terminal phase")
    phases = []
    for index, spec in enumerate(FAULT_PHASE_SPECS):
        matches = [
            record for record in records
            if record.get("fault_phase_index") == index
        ]
        if len(matches) != 1:
            raise ValueError("fault campaign phase history is ambiguous")
        record = matches[0]
        terminal_reference = record.get("terminal_evidence")
        if (
            record.get("status") != "terminal"
            or record.get("verdict") != "passed"
            or record.get("fault_phase") != spec["name"]
            or not isinstance(terminal_reference, Mapping)
        ):
            raise ValueError(
                f"fault campaign phase {spec['name']} is not a pass")
        terminal_path = Path(str(terminal_reference.get("path", ""))).resolve()
        if (
            not terminal_path.is_file()
            or terminal_reference.get("sha256")
            != _file_sha256(terminal_path)
        ):
            raise ValueError("fault terminal evidence changed after collection")
        terminal = _load_json(
            terminal_path, schema="emender-async-v21-terminal-verdict-v1")
        scheduler = terminal.get("scheduler")
        semantic = terminal.get("validator_inputs", {}).get(
            "semantic_verdict", {})
        if (
            terminal.get("passed") is not True
            or terminal.get("payload_digest") != record.get("payload_digest")
            or not isinstance(scheduler, Mapping)
            or scheduler.get("state") != "COMPLETED"
            or scheduler.get("exit_code") != "0:0"
            or scheduler.get("derived_exit_code") != "0:0"
            or scheduler.get("nodes") != 2
            or scheduler.get("partition") != "batch"
            or scheduler.get("qos") != "debug"
            or not isinstance(semantic, Mapping)
            or semantic.get("required") is not True
            or semantic.get("passed") is not True
        ):
            raise ValueError(
                f"fault phase {spec['name']} lacks its exact terminal pass")
        phases.append({
            "name": spec["name"],
            "payload_digest": record["payload_digest"],
            "job_id": record["job_id"],
            "collector_job_id": record["collector"]["job_id"],
            "terminal_verdict": str(terminal_path),
            "terminal_verdict_sha256": _file_sha256(terminal_path),
            "semantic_verdict": semantic.get("retained_path"),
            "semantic_verdict_sha256": semantic.get("sha256"),
            "scheduler": dict(scheduler),
        })
    value: dict[str, object] = {
        "schema": "emender-async-v21-fault-campaign-verdict-v1",
        "status": "passed",
        "passed": True,
        "gate": "faults",
        "nodes": 2,
        "partition": "batch",
        "qos": "debug",
        "campaign_digest": campaign_digest,
        "identities": dict(identities),
        "prior_gate": {
            "path": prior_gate["path"],
            "sha256": prior_gate["sha256"],
            "payload_digest": prior_gate["payload_digest"],
        },
        "phases": phases,
        "scenarios": list(FAULT_SCENARIOS),
        "requirements": FAULT_REQUIREMENTS,
        "no_one_node_commit_authority": True,
        "maximum_speculative_windows": 2,
        "fresh_allocation_additional_windows": 5,
        "fresh_allocation_minimum_commits": 3,
    }
    value["manifest_digest"] = canonical_digest(value)
    return value


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=ALL_GATES, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--seed-config", type=Path,
        default=Path("configs/frontier/e97_async_256.yaml"))
    parser.add_argument("--native-build-manifest", type=Path)
    parser.add_argument("--full-layout-gate", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--state", required=True)
    parser.add_argument("--evidence-root")
    parser.add_argument("--authorization")
    parser.add_argument("--prior-rung")
    parser.add_argument("--prior-gate")
    parser.add_argument("--trusted-reviewer-key")
    parser.add_argument("--source-digest")
    parser.add_argument("--policy-digest")
    parser.add_argument("--bundle-digest")
    parser.add_argument("--seed-digest", default=SEED_SHA256)
    parser.add_argument("--launcher-digest")
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--output", help="retained rendered plan JSON")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    repo = args.repo.resolve()
    source_commit = (
        _require_submission_source(repo)
        if args.submit
        else subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
    )
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise ValueError("--parameters-json must encode an object")
    launcher_path = repo / "scripts/frontier/resilient_e97_true_2n.sbatch"
    seed_config = args.seed_config
    if not seed_config.is_absolute():
        seed_config = repo / seed_config
    seed_config = seed_config.resolve()
    if not seed_config.is_file() or not launcher_path.is_file():
        raise FileNotFoundError("canonical config or launcher is missing")
    if SEED_SHA256 != args.seed_digest:
        _digest(args.seed_digest, "seed")
    if args.policy_digest is None:
        from ndm.async_diloco_v2 import ASYNC_DECOUPLED_V21
        policy_digest = ASYNC_DECOUPLED_V21.digest
    else:
        policy_digest = args.policy_digest
    clean_launch = None
    fault_launch = None
    bundle_digest = args.bundle_digest
    if args.gate in {"clean", "faults"} and any((
        args.native_build_manifest,
        args.full_layout_gate,
        args.run_root,
    )):
        if not all((
            args.native_build_manifest,
            args.full_layout_gate,
            args.run_root,
            args.output,
        )):
            raise ValueError(
                "integrated production launch requires --native-build-manifest, "
                "--full-layout-gate, --run-root, and --output")
        launch_context, derived_bundle = _clean_launch_context(
            repo=repo,
            source_commit=source_commit,
            seed_config=seed_config,
            native_build_manifest=args.native_build_manifest.resolve(),
            full_layout_gate=args.full_layout_gate.resolve(),
            run_dir=(
                args.run_root.resolve()
                / (CLEAN_PHASE if args.gate == "clean" else "faults")
            ),
            acceptance_manifest=Path(args.output).resolve(),
            submit=args.submit,
            required_gate=(
                "G2"
                if args.gate == "clean"
                else "G2-fault-rejoin-replay"
            ),
        )
        if args.gate == "clean":
            clean_launch = launch_context
        else:
            fault_launch = launch_context
        if bundle_digest is not None and bundle_digest != derived_bundle:
            raise ValueError(
                "explicit bundle digest differs from the native build manifest")
        bundle_digest = derived_bundle
    if args.submit and args.gate == "clean" and clean_launch is None:
        raise ValueError(
            "clean submission requires the integrated build/G2/run launch")
    if args.submit and args.gate == "faults" and fault_launch is None:
        raise ValueError(
            "fault submission requires the integrated build/G2/run launch")
    if bundle_digest is None:
        raise ValueError(
            "--bundle-digest or an integrated native build manifest is required")
    evidence_root = (
        Path(args.evidence_root).resolve()
        if args.evidence_root is not None
        else (
            args.run_root.resolve()
            if args.run_root is not None
            else None
        )
    )
    if evidence_root is None:
        raise ValueError("--evidence-root or --run-root is required")
    source_digest = args.source_digest or str(_source_digest(repo)["digest"])
    launcher_digest = args.launcher_digest or _file_sha256(launcher_path)
    if fault_launch is not None:
        if args.prior_gate is None:
            raise ValueError(
                "integrated fault launch requires the passed --prior-gate")
        fault_gate_value = json.loads(
            Path(fault_launch["full_layout_gate"]).read_text(
                encoding="utf-8"))
        if (
            not isinstance(fault_gate_value, Mapping)
            or fault_gate_value.get("status") != "passed"
            or fault_gate_value.get("gate") != "G2-fault-rejoin-replay"
            or fault_gate_value.get("nodes") != 2
            or fault_gate_value.get("provider") != "cxi"
            or not isinstance(fault_gate_value.get("fault"), Mapping)
            or fault_gate_value["fault"].get("partial_commit") is not False
        ):
            raise ValueError(
                "fault launch requires the exact passing two-node CXI "
                "G2-fault-rejoin-replay artifact")
        identities = {
            "source_digest": source_digest,
            "policy_digest": policy_digest,
            "bundle_digest": bundle_digest,
            "seed_digest": args.seed_digest,
            "launcher_digest": launcher_digest,
        }
        prior_gate = _verify_prior_clean_gate(
            args.prior_gate, expected_identities=identities)
        campaign_digest = canonical_digest({
            "schema": "emender-async-v21-fault-campaign-v1",
            "identities": identities,
            "prior_gate_sha256": prior_gate["sha256"],
            "full_layout_gate_sha256":
                fault_launch["full_layout_gate_sha256"],
            "phases": list(FAULT_PHASE_SPECS),
            "scenarios": list(FAULT_SCENARIOS),
        })
        phase = _next_fault_phase(
            args.state,
            campaign_digest=campaign_digest,
            prior_gate=prior_gate,
        )
        if phase is None:
            verdict = _fault_campaign_verdict(
                state_path=Path(args.state).resolve(),
                campaign_digest=campaign_digest,
                prior_gate=prior_gate,
                identities=identities,
            )
            if args.output:
                _atomic_json(Path(args.output).resolve(), verdict)
            print(json.dumps(verdict, sort_keys=True, indent=2))
            return 0
        phase_index = next(
            index for index, item in enumerate(FAULT_PHASE_SPECS)
            if item["name"] == phase["name"])
        resume_handoff = ""
        if phase_index:
            resume_handoff = _fault_resume_handoff(
                run_dir=Path(fault_launch["run_dir"]),
                campaign_digest=campaign_digest,
                expected_generation=int(phase["initial_generation"]),
            )
        fault_launch.update({
            "fault_campaign_digest": campaign_digest,
            "fault_phase": str(phase["name"]),
            "fault_phase_index": str(phase_index),
            "initial_generation": str(phase["initial_generation"]),
            "generations": str(phase["generations"]),
            "coordinator_epoch": str(phase_index + 1),
            "resume_handoff": resume_handoff,
            "prior_gate": str(prior_gate["path"]),
            "prior_gate_sha256": str(prior_gate["sha256"]),
            "prior_payload_digest": str(prior_gate["payload_digest"]),
        })
    plan = build_plan(
        gate=args.gate,
        nodes=args.nodes,
        state_path=args.state,
        evidence_root=evidence_root,
        source_digest=source_digest,
        policy_digest=policy_digest,
        bundle_digest=bundle_digest,
        seed_digest=args.seed_digest,
        launcher_digest=launcher_digest,
        parameters=parameters,
        authorization_path=args.authorization,
        predecessor_path=args.prior_rung,
        trusted_reviewer_key=args.trusted_reviewer_key,
        clean_launch=clean_launch,
        fault_launch=fault_launch,
    )
    if args.output:
        _atomic_json(Path(args.output).resolve(), plan)
    print(json.dumps(plan, sort_keys=True, indent=2))
    if args.submit:
        print(f"submitted_job_id={submit_plan(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
