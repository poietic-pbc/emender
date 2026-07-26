#!/usr/bin/env python3
"""Fail-closed semantic validator for simple async DiLoCo v2.1.

This validator does not infer a generation from a thread name. It requires the
exact versioned v2.1 policy, local-window/applied-anchor identity, versioned
background work, bounded queue evidence, reload/CAS-verified application, and
independent correctness latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


POLICY_ID = "async-decoupled-v2.1-simple"
POLICY_SCHEMA = "emender-async-policy-v2.1"
K_LOCAL_STEPS = 40
MAX_COMMIT_LAG = 2
MAX_ANCHOR_LAG = 2
MAX_RESULT_LAG = 2
MAX_SPECULATIVE_WINDOWS = 2
ETA_OUTER = 1.0
EXPECTED_TRAINERS = 16
WARMUP_WINDOWS = 2
MEASURED_WINDOWS = 10
MAX_IDLE_FRACTION = 0.10
MAX_CADENCE_MULTIPLE = 1.25
MAX_LOCAL_OWNED_S = 1.0
MAX_ALL_EIGHT_APPLY_S = 60.0
MAX_FOREGROUND_GAP_S = 60.0
MAX_CORRECTNESS_S = 420.0
E97_NATIVE_RESIDENT_BYTES = 64_001_671_648

BACKGROUND_STAGES = frozenset({
    "native_local_reduction",
    "native_owner_redistribution",
    "native_trainer_apply",
    "checkpoint_publication",
    "fenced_atomic_commit",
    "control_handoff_integrity",
})
REQUIRED_STAGE_CLASSES = frozenset({
    "native_local_reduction",
    "native_owner_redistribution",
    "native_trainer_apply",
    "checkpoint_publication",
    "control_handoff_integrity",
    "fenced_atomic_commit",
})
CAUSAL_PHASES = frozenset({
    "freeze_snapshot",
    "snapshot_admission",
    "publish_network",
    "aggregation",
    "checkpoint",
    "result_wait",
    "apply_swap",
})


def _records(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    values.append(value)
    # Production policy declarations are retained as small control-plane JSON
    # documents, while the high-volume timing records above are JSONL.  Read
    # only the policy stage from standalone JSON so unrelated control records
    # cannot be mistaken for semantic telemetry.
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("stage") == "async_v21_policy":
            values.append(value)
    return values


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if isinstance(value, float) and value != parsed:
        raise ValueError(f"{name} must be an integer")
    return parsed


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _digest(value: Any, name: str) -> str:
    encoded = str(value)
    if len(encoded) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        bytes.fromhex(encoded)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    return encoded


def _percentile(values: Iterable[int | float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile from no values")
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _causal_work_id(identity: str, generation: int) -> str:
    return hashlib.sha256(
        f"{identity}|{int(generation)}".encode("utf-8")).hexdigest()


def _policy(records: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    policies = [value for value in records
                if value.get("stage") == "async_v21_policy"]
    if not policies:
        raise ValueError("missing exact async-v2.1 policy declaration")
    required = {
        "policy_id": POLICY_ID,
        "policy_schema": POLICY_SCHEMA,
        "contribution_schema": "emender-native-e97-submission-v2.1",
        "manifest_schema": "emender-native-e97-generation-v2.1",
        "checkpoint_schema": "emender-async-v21-reference-checkpoint-v1",
        "native_abi": 0x00020001,
        "wire_protocol_major": 2,
        "wire_protocol_minor": 1,
        "max_commit_lag": MAX_COMMIT_LAG,
        "max_anchor_lag": MAX_ANCHOR_LAG,
        "max_result_lag": MAX_RESULT_LAG,
        "max_speculative_windows": MAX_SPECULATIVE_WINDOWS,
        "eta_outer": ETA_OUTER,
        "k_local_steps": K_LOCAL_STEPS,
        "owned_descriptor_capacity": 1,
        "mutable_interval_capacity": 1,
        "result_mailbox_capacity": 1,
        "result_staging_capacity": 1,
    }
    fences = set()
    for value in policies:
        if value.get("policy_id") != POLICY_ID:
            raise ValueError("historical or unknown async policy declaration")
        if any(value.get(name) != expected for name, expected in required.items()):
            raise ValueError("rendered async-v2.1 policy differs from reviewed constants")
        fence = _integer(value.get("allocation_fence"), "allocation fence")
        if fence <= 0:
            raise ValueError("allocation fence must be positive")
        fences.add(fence)
    if len(fences) != 1:
        raise ValueError("performance artifact crosses allocation fences")
    return required, fences.pop()


def _background(
    records: list[dict[str, Any]], *, fence: int,
) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for record in records:
        stage = str(record.get("stage", ""))
        if stage not in BACKGROUND_STAGES:
            continue
        elapsed = _finite(record.get("elapsed_s"), f"{stage} elapsed")
        ended = _finite(record.get("timestamp"), f"{stage} timestamp")
        if elapsed <= 0 or record.get("within_slo") is not True:
            raise ValueError(f"invalid or failed background telemetry: {stage}")
        if record.get("policy_id") != POLICY_ID:
            raise ValueError("background record has a non-v2.1 policy identity")
        if _integer(record.get("allocation_fence"), "background fence") != fence:
            raise ValueError("background work is not under the current fence")
        base = _integer(record.get("base_global_version"), "background base version")
        committed = _integer(
            record.get("commit_global_version"), "background commit version")
        lag = _integer(record.get("commit_lag"), "background commit lag")
        if base < 0 or committed < base or committed - base != lag:
            raise ValueError("background base/commit/lag identity is inconsistent")
        if not 0 <= lag <= MAX_COMMIT_LAG:
            raise ValueError(f"background hard lag violation: {lag}")
        exact_tokens = _integer(record.get("exact_tokens"), "background exact tokens")
        if exact_tokens <= 0 or "aggregation_weight" in record:
            raise ValueError("background must use only the exact-token quantity")
        values.append({
            "stage": stage,
            "begin": ended - elapsed,
            "end": ended,
            "base_global_version": base,
            "commit_global_version": committed,
            "commit_lag": lag,
            "contribution_digest": _digest(
                record.get("contribution_digest"), "background contribution"),
        })
    if not values:
        raise ValueError("no accurately versioned background work")
    present = {value["stage"] for value in values}
    if not REQUIRED_STAGE_CLASSES <= present:
        missing = sorted(REQUIRED_STAGE_CLASSES - present)
        raise ValueError(
            f"background stages are not independently timed: {missing}")
    return values


def _bounds(
    records: list[dict[str, Any]], trainers: set[str],
) -> dict[str, int | float]:
    by_identity: dict[str, dict[str, Any]] = {}
    for value in records:
        if value.get("stage") == "async_v21_bounds":
            identity = str(value.get("identity", ""))
            if not identity or identity in by_identity:
                raise ValueError("duplicate or missing bounded queue identity")
            by_identity[identity] = value
    if set(by_identity) != trainers:
        raise ValueError("bounded queue evidence is missing for one or more trainers")
    high = {
        "sealed_descriptor_high_water": 0,
        "mutable_interval_high_water": 0,
        "mutable_window_high_water": 0,
        "result_mailbox_high_water": 0,
        "result_staging_high_water": 0,
    }
    ownership = 0.0
    for value in by_identity.values():
        exact_caps = {
            "sealed_descriptor_capacity": 1,
            "mutable_interval_capacity": 1,
            "result_mailbox_capacity": 1,
            "result_staging_capacity": 1,
        }
        if any(_integer(value.get(name), name) != expected
               for name, expected in exact_caps.items()):
            raise ValueError("bounded queue capacity differs from reviewed v2 policy")
        limits = {
            "sealed_descriptor_high_water": 1,
            "mutable_interval_high_water": 1,
            "mutable_window_high_water": MAX_SPECULATIVE_WINDOWS,
            "result_mailbox_high_water": 1,
            "result_staging_high_water": 1,
        }
        for name, limit in limits.items():
            observed = _integer(value.get(name), name)
            if not 0 <= observed <= limit:
                raise ValueError(f"bounded queue high-water violation: {name}")
            high[name] = max(high[name], observed)
        local_owned = _finite(
            value.get("native_owned_seconds_max"), "local OWNED latency")
        if not 0 <= local_owned <= MAX_LOCAL_OWNED_S:
            raise ValueError("local OWNED acknowledgement exceeded one second")
        ownership = max(ownership, local_owned)
        admitted = _integer(
            value.get("resident_admission_bytes"),
            "resident admission bytes")
        resident_limit = _integer(
            value.get("resident_limit_bytes"), "resident limit bytes")
        headroom = _integer(
            value.get("resident_headroom_bytes"),
            "resident headroom bytes")
        if (
            admitted != E97_NATIVE_RESIDENT_BYTES
            or resident_limit < admitted
            or headroom != resident_limit - admitted
        ):
            raise ValueError(
                "native resident formula/cap is incomplete or unbounded")
        for forbidden in (
            "python_dense_socket_bytes", "lustre_dense_hot_path_bytes",
        ):
            if _integer(value.get(forbidden), forbidden) != 0:
                raise ValueError(f"forbidden dense hot path observed: {forbidden}")
        for count in ("pause_count", "drop_count"):
            if _integer(value.get(count), count) < 0:
                raise ValueError(f"{count} cannot be negative")
        for bootstrap in (
            "model_build", "optimizer_build", "data_iterator_build",
        ):
            if _integer(value.get(bootstrap), bootstrap) != 1:
                raise ValueError(
                    "training lane is not one persistent resident session")
        if _integer(value.get("windows_completed"), "windows completed") < (
                WARMUP_WINDOWS + MEASURED_WINDOWS):
            raise ValueError(
                "persistent training session lacks the measured K windows")
    return {**high, "native_owned_seconds_max": ownership}


def _applications(
    records: list[dict[str, Any]], *, trainers: set[str], fence: int,
    measured_windows: Mapping[str, set[int]],
) -> tuple[list[int], list[int], list[int]]:
    applications: dict[tuple[str, int], dict[str, Any]] = {}
    for value in records:
        if value.get("stage") != "safe_boundary_apply":
            continue
        identity = str(value.get("identity", ""))
        window = _integer(value.get("local_window"), "apply local window")
        key = (identity, window)
        if key in applications:
            raise ValueError("duplicate safe-boundary application receipt")
        applications[key] = value
    anchor_lags: list[int] = []
    result_lags: list[int] = []
    speculative_lags: list[int] = []
    for identity in trainers:
        measured = measured_windows[identity]
        first_window, last_window = min(measured), max(measured)
        selected = [
            value
            for (owner, window), value in sorted(applications.items())
            if owner == identity and first_window <= window <= last_window + 1
        ]
        # Results are latest-only, not mandatory per-window barriers.  Requiring
        # one receipt for every K would silently recreate the serial policy.
        # Every application that does occur in the measured interval must be
        # independently versioned and verified, and at least one is required.
        if not selected:
            raise ValueError(
                "unverifiable application: measured interval has no "
                "safe-boundary receipt")
        for value in selected:
            if (
                _integer(value.get("allocation_fence"), "apply fence") != fence
                or value.get("reload_verified") is not True
                or value.get("latest_cas_verified") is not True
            ):
                raise ValueError("unverifiable application: reload/latest CAS absent")
            _digest(value.get("result_digest"), "applied result")
            _digest(value.get("manifest_digest"), "applied manifest")
            known = _integer(
                value.get("known_global_version"), "known global version")
            result = _integer(value.get("result_version"), "result version")
            anchor = _integer(
                value.get("applied_anchor_version"), "applied anchor version")
            anchor_lag = _integer(
                value.get("anchor_lag_before_apply"), "anchor lag")
            result_lag = _integer(
                value.get("result_version_lag_at_apply"), "result lag")
            speculative = _integer(
                value.get("speculative_window_lag"), "speculative lag")
            if result > known or anchor > known or result_lag != known - result:
                raise ValueError("unverifiable application version identity")
            if not 0 <= anchor_lag <= MAX_ANCHOR_LAG:
                raise ValueError(f"application hard lag violation: {anchor_lag}")
            if not 0 <= result_lag <= MAX_RESULT_LAG:
                raise ValueError(f"result hard lag violation: {result_lag}")
            if not 0 <= speculative <= MAX_SPECULATIVE_WINDOWS:
                raise ValueError(f"speculative hard lag violation: {speculative}")
            anchor_lags.append(anchor_lag)
            result_lags.append(result_lag)
            speculative_lags.append(speculative)
    return anchor_lags, result_lags, speculative_lags


def _correctness(records: list[dict[str, Any]], *, fence: int) -> dict[str, Any]:
    values = [value for value in records
              if value.get("stage") == "async_v21_correctness"]
    if not values:
        raise ValueError("missing separate freeze-to-latest correctness evidence")
    latencies = []
    for value in values:
        if (
            value.get("policy_id") != POLICY_ID
            or _integer(value.get("allocation_fence"), "correctness fence") != fence
            or value.get("reload_verified") is not True
            or value.get("latest_cas_verified") is not True
        ):
            raise ValueError("correctness publication is not reload/CAS verified")
        latency = _finite(
            value.get("freeze_to_latest_s"), "freeze-to-latest latency")
        if not 0 <= latency <= MAX_CORRECTNESS_S:
            raise ValueError("freeze-to-latest correctness deadline exceeded")
        if _integer(value.get("checkpoint_bytes"), "checkpoint bytes") <= 0:
            raise ValueError("correctness checkpoint extent is missing")
        _digest(value.get("manifest_digest"), "correctness manifest")
        latencies.append(latency)
    return {
        "freeze_to_latest_seconds_max": max(latencies),
        "deadline_seconds": MAX_CORRECTNESS_S,
        "passed": True,
    }


def _causal_phase_timing(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, float | int]],
    dict[str, dict[str, float | int | bool]],
]:
    """Require directly measured intervals for every pipeline phase.

    These records make a synchronized foreground gap attributable without
    inferring a cause from checkpoint counts or median cadence.  All phase
    intervals come from the same monotonic clock used by the runtime.
    """
    by_phase: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in CAUSAL_PHASES
    }
    for value in records:
        phase = str(value.get("causal_phase", ""))
        if phase not in by_phase:
            continue
        stage = str(value.get("stage", ""))
        started = _finite(
            value.get("monotonic_start_s"), f"{stage} monotonic start")
        ended = _finite(
            value.get("monotonic_end_s"), f"{stage} monotonic end")
        elapsed = _finite(value.get("elapsed_s"), f"{stage} elapsed")
        if (
            ended < started
            or elapsed < 0
            or not math.isclose(
                ended - started, elapsed, rel_tol=1e-9, abs_tol=1e-6)
        ):
            raise ValueError(f"{stage} causal timing interval is inconsistent")
        identity = str(value.get("identity", ""))
        generation = _integer(
            value.get("generation"), f"{stage} causal generation")
        causal_id = _digest(value.get("causal_id"), f"{stage} causal identity")
        if (
            not identity
            or generation < 0
            or causal_id != _causal_work_id(identity, generation)
        ):
            raise ValueError(f"{stage} has a mismatched causal identity")
        foreground = _finite(
            value.get("foreground_component_s"),
            f"{stage} foreground component")
        if foreground < 0 or foreground > elapsed + 1e-6:
            raise ValueError(
                f"{stage} foreground component double counts its interval")
        by_phase[phase].append(value)
    missing = sorted(phase for phase, values in by_phase.items() if not values)
    if missing:
        raise ValueError(
            f"causal pipeline phases are not independently timed: {missing}")

    for value in by_phase["freeze_snapshot"]:
        if (
            value.get("foreground_interruption") != "snapshot_capture"
            or value.get("phase_scope") != "trainer_snapshot"
            or value.get("snapshot_coherent") is not True
            or _integer(value.get("snapshot_slots"), "snapshot slots") != 2
            or value.get("live_model_read_after_snapshot") is not False
        ):
            raise ValueError("snapshot capture is not coherent and bounded")
    snapshot_pauses: list[float] = []
    for value in by_phase["snapshot_admission"]:
        if (
            value.get("foreground_interruption") != "snapshot_admission"
            or value.get("phase_scope") != "snapshot_owned"
            or value.get("owned") is not True
            or value.get("immutable_snapshot") is not True
            or value.get("mutable_training_resumed") is not True
            or _finite(value.get("policy_bound_s"), "snapshot policy bound")
            != MAX_LOCAL_OWNED_S
        ):
            raise ValueError("mutable training did not resume after snapshot")
        snapshot_pauses.append(_finite(
            value.get("foreground_pause_s"), "snapshot foreground pause"))
    if not snapshot_pauses:
        raise ValueError("snapshot admission lacks every-event pause evidence")
    if (
        max(snapshot_pauses) > MAX_LOCAL_OWNED_S
        or _percentile(snapshot_pauses, .99) > MAX_LOCAL_OWNED_S
    ):
        raise ValueError(
            "snapshot capture/admission maximum or p99 exceeded one second")

    if not any(
        value.get("stage") == "native_direct_memfd"
        and value.get("immutable_snapshot") is True
        and value.get("mutable_training_already_resumed") is True
        and value.get("foreground_blocking") is False
        for value in by_phase["publish_network"]
    ):
        raise ValueError(
            "snapshot publication lacks proof of prior foreground resume")
    for phase in ("publish_network", "aggregation", "checkpoint"):
        if any(
            _finite(
                value.get("foreground_component_s"),
                f"{phase} foreground component") != 0.0
            for value in by_phase[phase]
        ):
            raise ValueError(f"{phase} entered the foreground training lane")
    if any(
        _finite(
            value.get("foreground_component_s"),
            "result_wait foreground component") != 0.0
        or value.get("foreground_blocking") is not False
        for value in by_phase["result_wait"]
    ):
        raise ValueError("result_wait has a nonzero foreground component")
    if not any(
        value.get("mutable_training_active") is True
        and value.get("foreground_blocking") is False
        for value in by_phase["result_wait"]
    ):
        raise ValueError("result readiness recreated a foreground wait")

    for value in by_phase["apply_swap"]:
        if (
            value.get("foreground_interruption") != "verified_result_apply"
            or value.get("foreground_blocking") is not True
            or value.get("atomic_live_model_swap") is not True
        ):
            raise ValueError("verified result apply is not an atomic boundary")

    node_applies = [
        value for value in by_phase["apply_swap"]
        if value.get("phase_scope") == "node_all_eight"
    ]
    if not node_applies:
        raise ValueError("missing complete all-eight apply/swap timing")
    apply_pauses: list[float] = []
    for value in node_applies:
        if (
            _integer(value.get("trainer_count"), "apply trainer count") != 8
            or _finite(value.get("policy_bound_s"), "apply policy bound")
            != MAX_ALL_EIGHT_APPLY_S
        ):
            raise ValueError("apply/swap is not one bounded all-eight transaction")
        apply_pauses.append(_finite(
            value.get("elapsed_s"), "all-eight apply/swap pause"))
    if (
        max(apply_pauses) > MAX_ALL_EIGHT_APPLY_S
        or _percentile(apply_pauses, .99) > MAX_ALL_EIGHT_APPLY_S
    ):
        raise ValueError(
            "all-eight apply/swap maximum or p99 exceeded 60 seconds")

    # Foreground intervals for one causal identity are disjoint.  A record
    # cannot make the same pause simultaneously count as two phase classes.
    foreground_by_work: dict[str, list[tuple[float, float, str]]] = {}
    for phase, values in by_phase.items():
        for value in values:
            if _finite(
                value.get("foreground_component_s"),
                f"{phase} foreground component") <= 0:
                continue
            foreground_by_work.setdefault(str(value["causal_id"]), []).append((
                _finite(value["monotonic_start_s"], "foreground start"),
                _finite(value["monotonic_end_s"], "foreground end"),
                phase,
            ))
    for intervals in foreground_by_work.values():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if current[0] < previous[1] - 1e-6:
                raise ValueError(
                    "causal foreground phases overlap or double count time")

    phase_summary = {
        phase: {
            "count": len(values),
            "median_seconds": statistics.median(
                _finite(value["elapsed_s"], f"{phase} elapsed")
                for value in values),
            "p99_seconds": _percentile(
                (_finite(value["elapsed_s"], f"{phase} elapsed")
                 for value in values),
                .99,
            ),
            "maximum_seconds": max(
                _finite(value["elapsed_s"], f"{phase} elapsed")
                for value in values),
        }
        for phase, values in sorted(by_phase.items())
    }
    pause_bounds = {
        "snapshot_admission": {
            "count": len(snapshot_pauses),
            "p99_seconds": _percentile(snapshot_pauses, .99),
            "maximum_seconds": max(snapshot_pauses),
            "policy_bound_seconds": MAX_LOCAL_OWNED_S,
            "passed": True,
        },
        "all_eight_apply_swap": {
            "count": len(apply_pauses),
            "p99_seconds": _percentile(apply_pauses, .99),
            "maximum_seconds": max(apply_pauses),
            "policy_bound_seconds": MAX_ALL_EIGHT_APPLY_S,
            "passed": True,
        },
        "result_wait": {
            "count": len(by_phase["result_wait"]),
            "foreground_seconds": 0.0,
            "passed": True,
        },
    }
    return phase_summary, pause_bounds


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    policy, fence = _policy(records)
    backgrounds = _background(records, fence=fence)
    causal_phases, foreground_pause_bounds = _causal_phase_timing(records)
    atomic_commit_versions = {
        int(value["commit_global_version"])
        for value in backgrounds
        if value["stage"] == "fenced_atomic_commit"
    }
    if len(atomic_commit_versions) < 10:
        raise ValueError(
            "clean promotion requires at least ten distinct atomic commits")

    trainer: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    for value in records:
        phase = str(value.get("phase", ""))
        if phase not in {"optimizer_step_start", "optimizer_step_end"}:
            continue
        identity = str(value.get("identity", ""))
        window = _integer(
            value.get("local_window", value.get("generation", -1)),
            "local window",
        )
        if (
            not identity
            or window < 0
            or "monotonic_s" not in value
            or "timestamp" not in value
        ):
            raise ValueError(
                "trainer step telemetry lacks identity/window/paired timestamps")
        if value.get("policy_id") != POLICY_ID:
            raise ValueError("trainer record has a non-v2.1 policy identity")
        trainer.setdefault(identity, {}).setdefault(window, {}).setdefault(
            phase, []).append(value)
    if len(trainer) != EXPECTED_TRAINERS:
        raise ValueError(
            f"semantic performance gate requires exactly {EXPECTED_TRAINERS} trainers")

    raw: list[float] = []
    cadence: list[float] = []
    idle: list[float] = []
    overlaps: list[dict[str, Any]] = []
    measured_windows: dict[str, set[int]] = {}
    for identity, windows_by_id in trainer.items():
        ordered = sorted(windows_by_id)
        if len(ordered) < WARMUP_WINDOWS + MEASURED_WINDOWS:
            raise ValueError(
                f"{identity} lacks two warm-up plus ten measured K40 windows")
        selected = ordered[-MEASURED_WINDOWS:]
        measured_windows[identity] = set(selected)
        windows: dict[int, tuple[float, float, float]] = {}
        for window in selected:
            phases = windows_by_id[window]
            starts = phases.get("optimizer_step_start", [])
            ends = phases.get("optimizer_step_end", [])
            if len(starts) != K_LOCAL_STEPS or len(ends) != K_LOCAL_STEPS:
                raise ValueError(
                    f"{identity} local window {window} lacks exact K40 step timestamps")
            monotonic_start = min(
                _finite(value["monotonic_s"], "K40 monotonic start")
                for value in starts)
            monotonic_end = max(
                _finite(value["monotonic_s"], "K40 monotonic end")
                for value in ends)
            if monotonic_end <= monotonic_start:
                raise ValueError("non-positive K40 compute interval")
            first = min(starts, key=lambda item: float(item["monotonic_s"]))
            if (
                first.get("local_model_basis") != "worker_local"
                or _integer(first.get("applied_anchor_version"),
                            "window applied anchor") < 0
            ):
                raise ValueError(
                    "K-window pretends it used an unavailable global base")
            offset = _finite(first["timestamp"], "K40 wall timestamp") - _finite(
                first["monotonic_s"], "K40 monotonic timestamp")
            windows[window] = (monotonic_start, monotonic_end, offset)
            raw.append(monotonic_end - monotonic_start)
        identity_matches = 0
        for previous, current in zip(selected, selected[1:]):
            previous_window = windows[previous]
            current_window = windows[current]
            observed_cadence = current_window[0] - previous_window[0]
            if observed_cadence <= 0:
                raise ValueError("training local-window order is not monotonic")
            control_idle = max(0.0, current_window[0] - previous_window[1])
            cadence.append(observed_cadence)
            idle.append(control_idle)
            wall = (
                current_window[0] + current_window[2],
                current_window[1] + current_window[2],
            )
            matched = [
                value for value in backgrounds
                if value["begin"] < wall[1] and value["end"] > wall[0]
            ]
            if not matched:
                raise ValueError(
                    f"{identity} local window {current} lacks true versioned "
                    "background overlap")
            identity_matches += 1
            overlaps.append({
                "identity": identity,
                "local_window": current,
                "applied_anchor_version": _integer(
                    min(
                        windows_by_id[current]["optimizer_step_start"],
                        key=lambda item: float(item["monotonic_s"]),
                    ).get("applied_anchor_version"),
                    "window applied anchor",
                ),
                "background": [
                    {
                        "stage": value["stage"],
                        "base_global_version": value["base_global_version"],
                        "commit_global_version": value["commit_global_version"],
                        "commit_lag": value["commit_lag"],
                        "contribution_digest": value["contribution_digest"],
                    }
                    for value in matched
                ],
            })
        if identity_matches != MEASURED_WINDOWS - 1:
            raise ValueError("training lane overlap sample is incomplete")

    raw_k40 = statistics.median(raw)
    cadence_s = statistics.median(cadence)
    cadence_multiple = cadence_s / raw_k40
    idle_fraction = sum(idle) / sum(cadence)
    idle_p99 = _percentile(idle, .99)
    idle_max = max(idle)
    if idle_max > MAX_FOREGROUND_GAP_S or idle_p99 > MAX_FOREGROUND_GAP_S:
        raise ValueError(
            "training-lane every-event foreground tail exceeds the "
            "60-second interruption bound")
    if cadence_multiple > MAX_CADENCE_MULTIPLE:
        raise ValueError(
            f"training-lane cadence {cadence_multiple:.6f}x exceeds 1.25x raw K40")
    if idle_fraction >= MAX_IDLE_FRACTION:
        raise ValueError(
            f"training-lane foreground idle {idle_fraction:.6f} is not below 0.10")

    bounds = _bounds(records, set(trainer))
    anchor_lags, result_lags, speculative_lags = _applications(
        records,
        trainers=set(trainer),
        fence=fence,
        measured_windows=measured_windows,
    )
    commit_lags = [int(value["commit_lag"]) for value in backgrounds]
    for name, values, target in (
        ("commit", commit_lags, MAX_COMMIT_LAG),
        ("anchor", anchor_lags, MAX_ANCHOR_LAG),
        ("speculative", speculative_lags, MAX_SPECULATIVE_WINDOWS),
    ):
        if max(values) > target or _percentile(values, .99) > target:
            raise ValueError(f"{name} lag exceeds the clean promotion target")
    correctness = _correctness(records, fence=fence)
    stage_seconds = {
        stage: [
            value["end"] - value["begin"]
            for value in backgrounds if value["stage"] == stage
        ]
        for stage in sorted({value["stage"] for value in backgrounds})
    }
    return {
        "schema": "emender-async-decoupled-e97-performance-v2.1",
        "status": "passed",
        "policy": policy,
        "allocation_fence": fence,
        "atomic_commits": len(atomic_commit_versions),
        "measured_trainers": len(trainer),
        "warmup_windows_per_trainer": WARMUP_WINDOWS,
        "measured_windows_per_trainer": MEASURED_WINDOWS,
        "raw_k40_compute_seconds": raw_k40,
        "steady_state_cadence_seconds": cadence_s,
        "steady_state_cadence_multiple": cadence_multiple,
        "foreground_control_plane_idle_fraction": idle_fraction,
        "foreground_control_plane_idle_seconds_p99": idle_p99,
        "foreground_control_plane_idle_seconds_max": idle_max,
        "overlaps": overlaps,
        "stage_seconds": {
            name: {
                "count": len(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }
            for name, values in stage_seconds.items()
        },
        "causal_phase_seconds": causal_phases,
        "foreground_pause_bounds": foreground_pause_bounds,
        "lag": {
            "commit_p99": _percentile(commit_lags, .99),
            "commit_max": max(commit_lags),
            "anchor_p99": _percentile(anchor_lags, .99),
            "anchor_max": max(anchor_lags),
            "result_version_p99": _percentile(result_lags, .99),
            "result_version_max": max(result_lags),
            "speculative_p99": _percentile(speculative_lags, .99),
            "speculative_max": max(speculative_lags),
        },
        "bounds": bounds,
        "correctness": correctness,
        "training_lane": {
            "cadence_multiple_max": MAX_CADENCE_MULTIPLE,
            "foreground_idle_fraction_strict_max": MAX_IDLE_FRACTION,
            "foreground_gap_seconds_max": MAX_FOREGROUND_GAP_S,
            "passed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(_records(args.telemetry_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
