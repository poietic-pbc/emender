"""Reviewed simple asynchronous DiLoCo v2.1 control policy.

This module is deliberately model- and transport-agnostic.  It is the single
Python metadata/control authority around the persistent native E97 service:

* dense buffers continue to be allocated, sealed, reduced, transported, and
  redistributed by the compiled service;
* this layer fixes contribution/window/base identity, bounded ownership,
  exact staleness math, verified latest-only result admission, safe-boundary
  ScheduleFree translation, replay, and authoritative global/outer restart;
* no method sends or serializes a production dense payload over Python TCP or
  places one on the Lustre hot path.

The small NumPy checkpoint helper at the bottom is an executable
single-process reference fixture.  The production E97 checkpoint publisher
continues to use the fenced native handoff and torch checkpoint path.

Authority: ``docs/ASYNC_DECOUPLED_DILOCO_V2.md`` (`V21S01`-`V21S17`),
together with `R01`-`R16` and `NDP01`-`NDP17`.  Historical v2.0 records are
deliberately not decoded or migrated by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np


UINT64_MAX = (1 << 64) - 1


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error


def digest_array(value: np.ndarray | Sequence[float]) -> str:
    """Digest one canonical little-endian binary64 reference vector."""
    array = np.asarray(value, dtype="<f8", order="C")
    header = _canonical({"dtype": "float64-le", "shape": list(array.shape)})
    return _sha256(header + b"\0" + array.tobytes(order="C"))


@dataclass(frozen=True)
class AsyncV21Policy:
    policy_id: str = "async-decoupled-v2.1-simple"
    policy_schema: str = "emender-async-policy-v2.1"
    contribution_schema: str = "emender-native-e97-submission-v2.1"
    manifest_schema: str = "emender-native-e97-generation-v2.1"
    checkpoint_schema: str = "emender-async-v21-reference-checkpoint-v1"
    native_abi: int = 0x00020001
    wire_protocol_major: int = 2
    wire_protocol_minor: int = 1
    k_local_steps: int = 40
    max_commit_lag: int = 2
    max_anchor_lag: int = 2
    max_result_lag: int = 2
    max_speculative_windows: int = 2
    q_min: int = 2
    t_min: int = 3_934_080
    active_membership_fraction: None = None
    group_deadline_s: float = 420.0
    ready_deadline_s: float = 180.0
    owned_deadline_s: float = 1.0
    catch_up_deadline_s: float = 420.0
    first_commit_deadline_s: float = 720.0
    generation_attempt_retries: int = 0
    eta_outer: float = 1.0
    outer_mode: str = "delta_sgd"
    owned_descriptor_capacity: int = 1
    mutable_interval_capacity: int = 1
    result_mailbox_capacity: int = 1
    result_staging_capacity: int = 1
    owner_reassignments: int = 2

    def __post_init__(self) -> None:
        if (
            self.policy_id != "async-decoupled-v2.1-simple"
            or self.policy_schema != "emender-async-policy-v2.1"
            or self.contribution_schema != "emender-native-e97-submission-v2.1"
            or self.manifest_schema != "emender-native-e97-generation-v2.1"
            or self.checkpoint_schema
            != "emender-async-v21-reference-checkpoint-v1"
            or self.native_abi != 0x00020001
            or self.wire_protocol_major != 2
            or self.wire_protocol_minor != 1
            or self.k_local_steps != 40
            or self.max_commit_lag != 2
            or self.max_anchor_lag != 2
            or self.max_result_lag != 2
            or self.max_speculative_windows != 2
            or self.q_min != 2
            or self.t_min != 3_934_080
            or self.active_membership_fraction is not None
            or self.group_deadline_s != 420.0
            or self.ready_deadline_s != 180.0
            or self.owned_deadline_s != 1.0
            or self.catch_up_deadline_s != 420.0
            or self.first_commit_deadline_s != 720.0
            or self.generation_attempt_retries != 0
            or self.eta_outer != 1.0
            or self.outer_mode != "delta_sgd"
            or self.owned_descriptor_capacity != 1
            or self.mutable_interval_capacity != 1
            or self.result_mailbox_capacity != 1
            or self.result_staging_capacity != 1
            or self.owner_reassignments != 2
        ):
            raise ValueError(
                "simple async v2.1 policy fields are reviewed constants; "
                "a change requires a new policy version"
            )

    @property
    def digest(self) -> str:
        return _sha256(_canonical(asdict(self)))

    def manifest(self) -> dict[str, object]:
        return {**asdict(self), "policy_digest": self.digest}


AsyncV2Policy = AsyncV21Policy
ASYNC_DECOUPLED_V21 = AsyncV21Policy()
# Compatibility at the Python import surface only.  It names the v2.1 policy,
# never the historical v2.0 identity or schemas.
ASYNC_DECOUPLED_V2 = ASYNC_DECOUPLED_V21


@dataclass(frozen=True)
class OuterState:
    mode: str = "delta_sgd"
    eta_outer: float = 1.0
    step: int = 0
    accepted_tokens: int = 0

    def validate(self) -> None:
        if (
            self.mode != "delta_sgd"
            or self.eta_outer != 1.0
            or self.step < 0
            or self.accepted_tokens < 0
        ):
            raise ValueError("invalid authoritative async-v2.1 outer state")


@dataclass(frozen=True)
class ContributionIdentity:
    policy_id: str
    policy_schema: str
    contribution_schema: str
    native_abi: int
    wire_protocol_major: int
    wire_protocol_minor: int
    run_id: str
    allocation_fence: int
    worker_id: str
    worker_incarnation: str
    contribution_sequence: int
    local_window_start: int
    local_window_end: int
    window_count: int
    base_global_version: int
    base_global_digest: str
    policy_digest: str
    layout_digest: str
    code_digest: str
    exact_tokens: int
    base_lag_at_seal: int
    payload_digest: str
    window_monotonic_ns: tuple[tuple[int, int], ...]
    endpoint_digest: str
    local_trainer_set_digest: str
    source_dtype: str
    finite_checked: bool
    shard_roots: tuple[str, ...]

    @property
    def logical_key(self) -> tuple[object, ...]:
        return (
            self.run_id,
            self.allocation_fence,
            self.worker_id,
            self.worker_incarnation,
            self.contribution_sequence,
            self.local_window_start,
            self.local_window_end,
        )

    def canonical_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["window_monotonic_ns"] = [
            list(item) for item in self.window_monotonic_ns
        ]
        value["shard_roots"] = list(self.shard_roots)
        return value

    @property
    def digest(self) -> str:
        return _sha256(_canonical(self.canonical_dict()))

    def validate(self, policy: AsyncV21Policy = ASYNC_DECOUPLED_V21) -> None:
        if (
            self.policy_id != policy.policy_id
            or self.policy_schema != policy.policy_schema
            or self.contribution_schema != policy.contribution_schema
            or self.native_abi != policy.native_abi
            or self.wire_protocol_major != policy.wire_protocol_major
            or self.wire_protocol_minor != policy.wire_protocol_minor
            or not self.run_id
            or self.allocation_fence <= 0
            or not self.worker_id
            or not self.worker_incarnation
            or self.contribution_sequence < 0
            or self.local_window_start < 0
            or self.local_window_end <= self.local_window_start
            or self.window_count
            != self.local_window_end - self.local_window_start
            or self.window_count > policy.max_speculative_windows
            or self.base_global_version < 0
            or self.exact_tokens <= 0
            or not 0 <= self.base_lag_at_seal <= policy.max_commit_lag
            or not self.finite_checked
            or self.source_dtype not in {"float32", "float64", "bfloat16"}
            or len(self.window_monotonic_ns) != self.window_count
            or not self.shard_roots
        ):
            raise ValueError(
                "invalid async-v2.1 policy/schema/ABI contribution identity")
        if self.policy_digest != policy.digest:
            raise ValueError("wrong policy digest")
        for name in (
            "base_global_digest",
            "policy_digest",
            "layout_digest",
            "code_digest",
            "payload_digest",
            "endpoint_digest",
            "local_trainer_set_digest",
        ):
            _require_digest(str(getattr(self, name)), name)
        for root in self.shard_roots:
            _require_digest(root, "shard root")
        prior_end = -1
        for begin, end in self.window_monotonic_ns:
            if begin < 0 or end <= begin or begin < prior_end:
                raise ValueError("window timestamps are not adjacent monotonic evidence")
            prior_end = end


@dataclass(frozen=True)
class ContributionEnvelope:
    identity: ContributionIdentity
    delta: np.ndarray

    @property
    def digest(self) -> str:
        return self.identity.digest

    @property
    def exact_tokens(self) -> int:
        return self.identity.exact_tokens

    def validate(self, policy: AsyncV21Policy = ASYNC_DECOUPLED_V21) -> None:
        self.identity.validate(policy)
        array = np.asarray(self.delta, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("nonfinite or empty contribution payload")
        if digest_array(array) != self.identity.payload_digest:
            raise ValueError("payload digest mismatch")


def build_contribution(
    *,
    run_id: str,
    allocation_fence: int,
    worker_id: str,
    worker_incarnation: str,
    contribution_sequence: int,
    local_window_start: int,
    local_window_end: int,
    base_global_version: int,
    base_global_digest: str,
    current_global_version: int,
    policy: AsyncV21Policy,
    layout_digest: str,
    code_digest: str,
    exact_tokens: int,
    interval_start: np.ndarray,
    interval_end: np.ndarray,
    window_monotonic_ns: tuple[tuple[int, int], ...],
    endpoint_digest: str,
    local_trainer_set_digest: str,
    source_dtype: str,
    shard_roots: tuple[str, ...],
) -> ContributionEnvelope:
    start = np.asarray(interval_start, dtype=np.float64)
    end = np.asarray(interval_end, dtype=np.float64)
    if start.shape != end.shape:
        raise ValueError("contribution interval endpoint layout changed")
    delta = np.subtract(end, start, dtype=np.float64)
    finite = bool(np.isfinite(start).all() and np.isfinite(end).all()
                  and np.isfinite(delta).all())
    identity = ContributionIdentity(
        policy_id=policy.policy_id,
        policy_schema=policy.policy_schema,
        contribution_schema=policy.contribution_schema,
        native_abi=policy.native_abi,
        wire_protocol_major=policy.wire_protocol_major,
        wire_protocol_minor=policy.wire_protocol_minor,
        run_id=run_id,
        allocation_fence=allocation_fence,
        worker_id=worker_id,
        worker_incarnation=worker_incarnation,
        contribution_sequence=contribution_sequence,
        local_window_start=local_window_start,
        local_window_end=local_window_end,
        window_count=local_window_end - local_window_start,
        base_global_version=base_global_version,
        base_global_digest=base_global_digest,
        policy_digest=policy.digest,
        layout_digest=layout_digest,
        code_digest=code_digest,
        exact_tokens=exact_tokens,
        base_lag_at_seal=current_global_version - base_global_version,
        payload_digest=digest_array(delta),
        window_monotonic_ns=window_monotonic_ns,
        endpoint_digest=endpoint_digest,
        local_trainer_set_digest=local_trainer_set_digest,
        source_dtype=source_dtype,
        finite_checked=finite,
        shard_roots=shard_roots,
    )
    envelope = ContributionEnvelope(identity=identity, delta=delta)
    envelope.validate(policy)
    return envelope


@dataclass(frozen=True)
class AdmittedContribution:
    worker_id: str
    contribution_digest: str
    base_version: int
    commit_lag: int
    exact_tokens: int
    delta: np.ndarray


class StaleContribution(ValueError):
    """A correctly decoded contribution outside the reviewed lag bound."""


def reference_aggregate(
    current_version: int,
    records: Sequence[ContributionEnvelope],
    *,
    policy: AsyncV21Policy = ASYNC_DECOUPLED_V21,
) -> tuple[np.ndarray, int, tuple[AdmittedContribution, ...]]:
    """Exact deterministic binary64 semantic reference for NDP05/v2 math."""
    if current_version < 0 or not records:
        raise ValueError("aggregate requires a current version and records")
    ordered = sorted(records, key=lambda record: bytes.fromhex(record.digest))
    seen_workers: set[str] = set()
    admitted: list[AdmittedContribution] = []
    numerator: np.ndarray | None = None
    denominator = 0
    exact_tokens = 0
    for record in ordered:
        record.validate(policy)
        identity = record.identity
        if identity.worker_id in seen_workers:
            raise ValueError("at most one contribution per worker may enter a commit")
        seen_workers.add(identity.worker_id)
        lag = current_version - identity.base_global_version
        if lag < 0:
            raise ValueError("future-base contribution")
        if lag > policy.max_commit_lag:
            raise StaleContribution(
                f"contribution lag {lag} exceeds v2.1 maximum lag "
                f"{policy.max_commit_lag}"
            )
        if denominator > UINT64_MAX - identity.exact_tokens:
            raise OverflowError("aggregation denominator overflows uint64")
        product = np.multiply(
            np.asarray(record.delta, dtype=np.float64),
            np.float64(identity.exact_tokens),
            dtype=np.float64,
        )
        numerator = (
            product.copy()
            if numerator is None
            else np.add(numerator, product, dtype=np.float64)
        )
        denominator += identity.exact_tokens
        if exact_tokens > UINT64_MAX - identity.exact_tokens:
            raise OverflowError("accepted token clock overflows uint64")
        exact_tokens += identity.exact_tokens
        admitted.append(AdmittedContribution(
            worker_id=identity.worker_id,
            contribution_digest=record.digest,
            base_version=identity.base_global_version,
            commit_lag=lag,
            exact_tokens=identity.exact_tokens,
            delta=np.asarray(record.delta, dtype=np.float64),
        ))
    assert numerator is not None and denominator > 0
    if not np.isfinite(numerator).all():
        raise ValueError("nonfinite aggregate numerator")
    return (
        np.divide(numerator, np.float64(denominator), dtype=np.float64),
        exact_tokens,
        tuple(admitted),
    )


@dataclass(frozen=True)
class ResultEnvelope:
    run_id: str
    allocation_fence: int
    version: int
    base_version: int
    base_digest: str
    state: np.ndarray
    result_digest: str
    outer: OuterState
    policy_digest: str
    layout_digest: str
    code_digest: str
    manifest_digest: str
    selected_contribution_digests: tuple[str, ...]
    reload_verified: bool
    latest_cas_verified: bool

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        allocation_fence: int,
        version: int,
        base_version: int,
        base_digest: str,
        state: np.ndarray,
        outer: OuterState,
        policy_digest: str,
        layout_digest: str,
        code_digest: str,
        manifest_digest: str,
        selected_contribution_digests: tuple[str, ...],
        reload_verified: bool,
        latest_cas_verified: bool,
    ) -> "ResultEnvelope":
        array = np.asarray(state, dtype=np.float64).copy()
        return cls(
            run_id=run_id,
            allocation_fence=allocation_fence,
            version=version,
            base_version=base_version,
            base_digest=base_digest,
            state=array,
            result_digest=digest_array(array),
            outer=outer,
            policy_digest=policy_digest,
            layout_digest=layout_digest,
            code_digest=code_digest,
            manifest_digest=manifest_digest,
            selected_contribution_digests=selected_contribution_digests,
            reload_verified=reload_verified,
            latest_cas_verified=latest_cas_verified,
        )

    def validate(self) -> None:
        if (
            not self.run_id
            or self.allocation_fence <= 0
            or self.version <= self.base_version
            or self.base_version < 0
            or not self.reload_verified
            or not self.latest_cas_verified
        ):
            raise ValueError("unverified or invalid committed result")
        for name in (
            "base_digest",
            "result_digest",
            "policy_digest",
            "layout_digest",
            "code_digest",
            "manifest_digest",
        ):
            _require_digest(str(getattr(self, name)), name)
        for digest in self.selected_contribution_digests:
            _require_digest(digest, "selected contribution")
        self.outer.validate()
        if not np.isfinite(np.asarray(self.state, dtype=np.float64)).all():
            raise ValueError("nonfinite committed result")
        if digest_array(self.state) != self.result_digest:
            raise ValueError("committed result digest mismatch")


class Backpressure(RuntimeError):
    """A reviewed finite queue/lag bound requires pausing the producer."""


class ResultLease:
    def __init__(self, mailbox: "LatestResultMailbox", result: ResultEnvelope):
        self._mailbox = mailbox
        self.result = result
        self._released = False

    def release(self) -> None:
        if self._released:
            raise ValueError("result view was already released")
        self._released = True
        self._mailbox._release(self)

    def __enter__(self) -> ResultEnvelope:
        return self.result

    def __exit__(self, _type, _value, _traceback) -> None:
        self.release()


class LatestResultMailbox:
    """Capacity-one verified latest slot plus one bounded replacement stage."""

    def __init__(
        self,
        *,
        run_id: str,
        fence: int,
        policy_digest: str,
        layout_digest: str,
        code_digest: str,
    ):
        if not run_id or fence <= 0:
            raise ValueError("mailbox requires a fenced run")
        self.run_id = run_id
        self.fence = fence
        self.policy_digest = policy_digest
        self.layout_digest = layout_digest
        self.code_digest = code_digest
        self._visible: ResultEnvelope | None = None
        self._staging: ResultEnvelope | None = None
        self._held: ResultLease | None = None
        self._known: dict[int, str] = {}
        self._high_water = {"visible": 0, "staging": 0, "held": 0}
        self._lock = threading.Lock()

    @property
    def high_water(self) -> dict[str, int]:
        with self._lock:
            return dict(self._high_water)

    def publish(self, value: ResultEnvelope) -> str:
        with self._lock:
            if (
                value.run_id != self.run_id
                or value.allocation_fence != self.fence
            ):
                raise ValueError("result belongs to the wrong run/fence")
            if (
                value.policy_digest != self.policy_digest
                or value.layout_digest != self.layout_digest
                or value.code_digest != self.code_digest
            ):
                raise ValueError("result policy/layout/code identity mismatch")
            known = self._known.get(value.version)
            if known is not None:
                if known != value.result_digest:
                    raise ValueError("equal-version conflict")
                return "duplicate"
            value.validate()
            highest = max(self._known, default=-1)
            if value.version < highest:
                return "stale"
            if self._held is not None:
                if self._staging is not None:
                    raise Backpressure("result replacement staging buffer is occupied")
                self._staging = value
                self._known[value.version] = value.result_digest
                self._high_water["staging"] = 1
                return "staged"
            if self._visible is not None:
                if value.version < self._visible.version:
                    return "stale"
                self._visible = value
            else:
                self._visible = value
            self._known[value.version] = value.result_digest
            self._high_water["visible"] = 1
            return "published"

    def take(self) -> ResultLease | None:
        with self._lock:
            if self._held is not None:
                raise Backpressure("mailbox view is already held")
            if self._visible is None:
                return None
            value, self._visible = self._visible, None
            lease = ResultLease(self, value)
            self._held = lease
            self._high_water["held"] = 1
            return lease

    def _release(self, lease: ResultLease) -> None:
        with self._lock:
            if self._held is not lease:
                raise ValueError("stale result view release")
            self._held = None
            if self._staging is not None:
                self._visible, self._staging = self._staging, None


@dataclass
class ScheduleFreeLocalState:
    """Audited local state: `x/z` are points; moments/scalars are retained."""

    x: np.ndarray
    parameter_points: dict[str, np.ndarray] = field(default_factory=dict)
    retained_buffers: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64).copy()
        self.parameter_points = {
            name: np.asarray(value, dtype=np.float64).copy()
            for name, value in self.parameter_points.items()
        }
        if set(self.parameter_points) - {"z"}:
            raise ValueError("unknown parameter-valued optimizer buffer")
        if (
            not np.isfinite(self.x).all()
            or any(
                value.shape != self.x.shape or not np.isfinite(value).all()
                for value in self.parameter_points.values()
            )
        ):
            raise ValueError("invalid ScheduleFree local state")

    def translate(self, correction: np.ndarray) -> None:
        correction = np.asarray(correction, dtype=np.float64)
        if correction.shape != self.x.shape or not np.isfinite(correction).all():
            raise ValueError("invalid ScheduleFree coordinate translation")
        self.x = np.add(self.x, correction, dtype=np.float64)
        for name in tuple(self.parameter_points):
            self.parameter_points[name] = np.add(
                self.parameter_points[name], correction, dtype=np.float64)


def rebase_schedulefree_torch_state(
    *,
    local_state: MutableMapping[str, object],
    old_anchor: Mapping[str, object],
    new_anchor: Mapping[str, object],
    accepted_local_deltas: Sequence[Mapping[str, object]],
    optimizer_state_dict: MutableMapping[str, object],
    coalescing_start: MutableMapping[str, object],
) -> dict[str, object]:
    """Translate real ScheduleFree `x/z` at a K boundary, retaining moments.

    The optimizer's serialized parameter IDs are paired with sorted model
    names, which is the same canonical order used by the E97 flat layout.
    Tensor-valued `z` is a point and is translated.  `exp_avg_sq` is a moment
    and is retained byte-for-byte; unknown tensor-valued buffers fail before
    any mutation.  Scalar counters/loss-scaler facts are retained.
    """
    import torch

    names = sorted(local_state)
    if (
        names != sorted(old_anchor)
        or names != sorted(new_anchor)
        or names != sorted(coalescing_start)
        or any(names != sorted(delta) for delta in accepted_local_deltas)
    ):
        raise ValueError("ScheduleFree rebase layout identity mismatch")
    state_records = optimizer_state_dict.get("state")
    groups = optimizer_state_dict.get("param_groups")
    if not isinstance(state_records, MutableMapping) or not isinstance(groups, list):
        raise ValueError("ScheduleFree optimizer state is malformed")
    parameter_ids: list[object] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("ScheduleFree optimizer parameter group is malformed")
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(names) or len(set(parameter_ids)) != len(names):
        raise ValueError("ScheduleFree optimizer/model parameter order differs")

    corrections: dict[str, object] = {}
    for name in names:
        local = local_state[name]
        old = old_anchor[name]
        new = new_anchor[name]
        start = coalescing_start[name]
        if not all(isinstance(value, torch.Tensor)
                   for value in (local, old, new, start)):
            raise ValueError("ScheduleFree rebase values must be tensors")
        if not (local.shape == old.shape == new.shape == start.shape):
            raise ValueError("ScheduleFree rebase tensor shape changed")
        accepted = torch.zeros_like(local, dtype=torch.float64)
        for delta in accepted_local_deltas:
            value = delta[name]
            if not isinstance(value, torch.Tensor) or value.shape != local.shape:
                raise ValueError("accepted correction delta layout changed")
            accepted.add_(value.to(device=accepted.device, dtype=torch.float64))
        correction = (
            new.to(device=local.device, dtype=torch.float64)
            - old.to(device=local.device, dtype=torch.float64)
            - accepted
        )
        if not torch.isfinite(correction).all():
            raise ValueError("nonfinite ScheduleFree correction")
        corrections[name] = correction.to(dtype=local.dtype)

    # Validate every buffer before mutating x/z/coalescing snapshots.
    point_records: list[tuple[MutableMapping[str, object], str, str]] = []
    for name, parameter_id in zip(names, parameter_ids):
        record = state_records.get(parameter_id)
        if not isinstance(record, MutableMapping):
            raise ValueError("ScheduleFree per-parameter state is missing")
        z = record.get("z")
        if not isinstance(z, torch.Tensor) or z.shape != local_state[name].shape:
            raise ValueError("ScheduleFree z point is missing or malformed")
        for key, value in record.items():
            if isinstance(value, torch.Tensor) and value.shape == z.shape:
                if key not in {"z", "exp_avg_sq"}:
                    raise ValueError(
                        f"unknown parameter-valued optimizer buffer: {key}")
        point_records.append((record, "z", name))

    for name in names:
        correction = corrections[name]
        local_state[name].add_(correction)
        coalescing_start[name].add_(
            correction.to(dtype=coalescing_start[name].dtype))
    for record, key, name in point_records:
        record[key].add_(corrections[name].to(dtype=record[key].dtype))
    return corrections


def translate_schedulefree_optimizer_points(
    *,
    optimizer_state_dict: MutableMapping[str, object],
    parameter_names: Sequence[str],
    correction: Callable[[str], object],
) -> None:
    """Stream a verified correction through serialized ScheduleFree `z`.

    This production helper avoids materializing a third full-model correction:
    callers provide one canonical parameter correction at a time.  Moments and
    scalars are validated/retained exactly as in
    :func:`rebase_schedulefree_torch_state`.
    """
    import torch

    # The sequence is the optimizer's audited parameter order (which need not
    # be lexicographic even though the dense wire layout is).  Reordering it
    # would translate the wrong z point.
    names = list(parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("ScheduleFree parameter names are not canonical")
    records = optimizer_state_dict.get("state")
    groups = optimizer_state_dict.get("param_groups")
    if not isinstance(records, MutableMapping) or not isinstance(groups, list):
        raise ValueError("ScheduleFree optimizer state is malformed")
    parameter_ids: list[object] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("ScheduleFree optimizer parameter group is malformed")
        parameter_ids.extend(group["params"])
    if len(parameter_ids) != len(names) or len(set(parameter_ids)) != len(names):
        raise ValueError("ScheduleFree optimizer/model parameter order differs")
    validated: list[tuple[str, MutableMapping[str, object]]] = []
    for name, parameter_id in zip(names, parameter_ids):
        record = records.get(parameter_id)
        if not isinstance(record, MutableMapping):
            raise ValueError("ScheduleFree per-parameter state is missing")
        z = record.get("z")
        if not isinstance(z, torch.Tensor):
            raise ValueError("ScheduleFree z point is missing")
        for key, value in record.items():
            if isinstance(value, torch.Tensor) and value.shape == z.shape:
                if key not in {"z", "exp_avg_sq"}:
                    raise ValueError(
                        f"unknown parameter-valued optimizer buffer: {key}")
        validated.append((name, record))
    for name, record in validated:
        value = correction(name)
        z = record["z"]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != z.shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("invalid streamed ScheduleFree correction")
        z.add_(value.to(device=z.device, dtype=z.dtype))


class AsyncV21WorkerLane:
    """One continuous model-owning lane with one sealed and one mutable range."""

    def __init__(
        self,
        *,
        run_id: str,
        fence: int,
        worker_id: str,
        incarnation: str,
        local: ScheduleFreeLocalState,
        anchor_version: int,
        anchor_state: np.ndarray,
        anchor_digest: str,
        layout_digest: str,
        code_digest: str,
        policy: AsyncV21Policy = ASYNC_DECOUPLED_V21,
    ):
        if not run_id or fence <= 0 or not worker_id or not incarnation:
            raise ValueError("worker lane requires a fenced incarnation")
        anchor = np.asarray(anchor_state, dtype=np.float64).copy()
        if anchor.shape != local.x.shape or not np.isfinite(anchor).all():
            raise ValueError("worker anchor layout/state mismatch")
        for value, name in (
            (anchor_digest, "anchor digest"),
            (layout_digest, "layout digest"),
            (code_digest, "code digest"),
        ):
            _require_digest(value, name)
        self.run_id = run_id
        self.fence = fence
        self.worker_id = worker_id
        self.incarnation = incarnation
        self.local = local
        self.policy = policy
        self.layout_digest = layout_digest
        self.code_digest = code_digest
        self.applied_anchor_version = anchor_version
        self.applied_anchor_state = anchor
        self.applied_anchor_digest = anchor_digest
        self.newest_verified_version = anchor_version
        self.local_window = 0
        self._last_committed_apply_window = 0
        self._sequence = 0
        self._sealed: ContributionEnvelope | None = None
        self._interval_q0 = 0
        self._interval_start = self.local.x.copy()
        self._interval_base_version = anchor_version
        self._interval_base_digest = anchor_digest
        self._interval_tokens = 0
        self._interval_times: list[tuple[int, int]] = []
        self._accepted: list[tuple[int, str, np.ndarray]] = []
        self.snapshot_deferred_reason: str | None = None
        self.paused_reason: str | None = None
        self.stale_drop_count = 0
        self._high_water = {
            "owned_descriptors": 0,
            "mutable_intervals": 1,
            "mutable_windows": 0,
        }
        self.mailbox = LatestResultMailbox(
            run_id=run_id,
            fence=fence,
            policy_digest=policy.digest,
            layout_digest=layout_digest,
            code_digest=code_digest,
        )

    @classmethod
    def for_test(
        cls,
        *,
        local: ScheduleFreeLocalState | None = None,
        anchor_state: np.ndarray | None = None,
    ) -> "AsyncV21WorkerLane":
        local = local or ScheduleFreeLocalState(
            x=np.asarray([0.0]),
            parameter_points={"z": np.asarray([-1.0])},
        )
        anchor_state = (
            np.asarray(anchor_state, dtype=np.float64)
            if anchor_state is not None
            else np.zeros_like(local.x)
        )
        return cls(
            run_id="run",
            fence=7,
            worker_id="node-0",
            incarnation="boot-a",
            local=local,
            anchor_version=0,
            anchor_state=anchor_state,
            anchor_digest="0" * 64,
            layout_digest="1" * 64,
            code_digest="2" * 64,
        )

    @property
    def mutable_window_range(self) -> tuple[int, int]:
        return self._interval_q0, self.local_window

    @property
    def mutable_window_count(self) -> int:
        return self.local_window - self._interval_q0

    @property
    def high_water(self) -> dict[str, int]:
        return dict(self._high_water)

    @property
    def speculative_window_lag(self) -> int:
        # This clock counts bounded speculative snapshot state, not
        # disposable foreground-only K windows completed while admission is
        # deferred.
        return min(
            self.mutable_window_count,
            self.policy.max_speculative_windows,
        )

    def finish_window(
        self,
        endpoint: np.ndarray,
        *,
        exact_tokens: int,
        begin_ns: int,
        end_ns: int,
    ) -> None:
        endpoint = np.asarray(endpoint, dtype=np.float64)
        if (
            endpoint.shape != self.local.x.shape
            or not np.isfinite(endpoint).all()
            or exact_tokens <= 0
            or begin_ns < 0
            or end_ns <= begin_ns
            or (self._interval_times and begin_ns < self._interval_times[-1][1])
        ):
            raise ValueError("invalid completed K-window")
        admission_was_full = (
            self.snapshot_deferred_reason is not None
            or self.mutable_window_count >= self.policy.max_speculative_windows
        )
        self.local.x = endpoint.copy()
        self.local_window += 1
        if admission_was_full:
            # The completed K work remains trainer-local and disposable.  It
            # is not retained as a third interval/snapshot and cannot be
            # admitted until a later verified boundary apply resets the base.
            self.snapshot_deferred_reason = "snapshot_admission_limit"
            self._interval_q0 = self.local_window
            self._interval_start = self.local.x.copy()
            self._interval_tokens = 0
            self._interval_times = []
        else:
            self._interval_tokens += exact_tokens
            self._interval_times.append((begin_ns, end_ns))
        count = self.mutable_window_count
        self._high_water["mutable_windows"] = max(
            self._high_water["mutable_windows"], count)
        self.paused_reason = None

    def seal(self) -> ContributionEnvelope:
        if self._sealed is not None:
            raise Backpressure("one sealed descriptor is already service-owned")
        if self.snapshot_deferred_reason is not None:
            raise Backpressure("snapshot admission is deferred")
        if self.mutable_window_count <= 0:
            raise ValueError("cannot seal an empty cumulative interval")
        value = build_contribution(
            run_id=self.run_id,
            allocation_fence=self.fence,
            worker_id=self.worker_id,
            worker_incarnation=self.incarnation,
            contribution_sequence=self._sequence,
            local_window_start=self._interval_q0,
            local_window_end=self.local_window,
            base_global_version=self._interval_base_version,
            base_global_digest=self._interval_base_digest,
            current_global_version=self.newest_verified_version,
            policy=self.policy,
            layout_digest=self.layout_digest,
            code_digest=self.code_digest,
            exact_tokens=self._interval_tokens,
            interval_start=self._interval_start,
            interval_end=self.local.x,
            window_monotonic_ns=tuple(self._interval_times),
            endpoint_digest=digest_array(self.local.x),
            local_trainer_set_digest="3" * 64,
            source_dtype="float32",
            shard_roots=("1" * 64,),
        )
        self._sequence += 1
        self._sealed = value
        self._high_water["owned_descriptors"] = 1
        # Immediately open the sole mutable successor interval.  It retains
        # the applied global anchor present at this exact safe boundary.
        self._interval_q0 = self.local_window
        self._interval_start = self.local.x.copy()
        self._interval_base_version = self.applied_anchor_version
        self._interval_base_digest = self.applied_anchor_digest
        self._interval_tokens = 0
        self._interval_times = []
        self.paused_reason = None
        return value

    def release_sealed(self, digest: str, *, outcome: str) -> None:
        if self._sealed is None or self._sealed.digest != digest:
            raise ValueError("stale or conflicting sealed-descriptor receipt")
        if outcome not in {
            "accepted",
            "not_selected",
            "stale_drop",
            "corrupt_reject",
            "owner_abort",
        }:
            raise ValueError("unknown descriptor release outcome")
        self._sealed = None
        if outcome == "stale_drop":
            self.stale_drop_count += 1

    def release_owned(self, digest: str, *, outcome: str) -> None:
        """Release the sole immutable descriptor after the native receipt."""
        self.release_sealed(digest, outcome=outcome)

    def record_accepted(
        self, *, commit_version: int, contribution: ContributionEnvelope,
    ) -> None:
        contribution.validate(self.policy)
        if (
            contribution.identity.worker_id != self.worker_id
            or contribution.identity.worker_incarnation != self.incarnation
            or commit_version <= self.applied_anchor_version
        ):
            raise ValueError("accepted-delta correction identity mismatch")
        if any(digest == contribution.digest for _v, digest, _d in self._accepted):
            return
        self._accepted.append((
            commit_version,
            contribution.digest,
            np.asarray(contribution.delta, dtype=np.float64).copy(),
        ))
        self._accepted.sort(key=lambda item: (item[0], item[1]))

    def apply_latest_at_boundary(self, *, known_global_version: int) -> bool:
        if known_global_version < self.newest_verified_version:
            raise ValueError("known global version moved backwards")
        self.newest_verified_version = known_global_version
        anchor_lag = known_global_version - self.applied_anchor_version
        if anchor_lag < 0:
            raise ValueError("applied anchor is from the future")
        if anchor_lag > self.policy.max_anchor_lag:
            raise StaleContribution("worker applied-anchor hard lag exceeded")
        lease = self.mailbox.take()
        if lease is None:
            if anchor_lag >= self.policy.max_anchor_lag:
                self.snapshot_deferred_reason = "verified_result_unready"
            self.paused_reason = None
            return False
        value = lease.result
        try:
            if value.version <= self.applied_anchor_version:
                return False
            if value.version > known_global_version:
                raise ValueError("mailbox result is newer than verified latest")
            accepted = np.zeros_like(self.local.x, dtype=np.float64)
            retained: list[tuple[int, str, np.ndarray]] = []
            for version, digest, delta in self._accepted:
                if self.applied_anchor_version < version <= value.version:
                    accepted = np.add(accepted, delta, dtype=np.float64)
                elif version > value.version:
                    retained.append((version, digest, delta))
            correction = np.subtract(
                np.subtract(
                    np.asarray(value.state, dtype=np.float64),
                    self.applied_anchor_state,
                    dtype=np.float64,
                ),
                accepted,
                dtype=np.float64,
            )
            self.local.translate(correction)
            self._interval_start = np.add(
                self._interval_start, correction, dtype=np.float64)
            self.applied_anchor_version = value.version
            self.applied_anchor_state = np.asarray(
                value.state, dtype=np.float64).copy()
            self.applied_anchor_digest = value.result_digest
            self._accepted = retained
            self._last_committed_apply_window = self.local_window
            self.paused_reason = None
            self.snapshot_deferred_reason = None
            if (
                self.mutable_window_count > 0
                and self.newest_verified_version
                - self._interval_base_version > self.policy.max_commit_lag
            ):
                # The work remains in the disposable local model, but this
                # over-age interval can never enter an open global group.
                self.stale_drop_count += 1
                self._interval_q0 = self.local_window
                self._interval_start = self.local.x.copy()
                self._interval_base_version = self.applied_anchor_version
                self._interval_base_digest = self.applied_anchor_digest
                self._interval_tokens = 0
                self._interval_times = []
            elif self.mutable_window_count == 0:
                # A prior capacity defer retained no admissible local
                # displacement.  Begin the next interval at this corrected
                # safe boundary.
                self._interval_q0 = self.local_window
                self._interval_start = self.local.x.copy()
                self._interval_base_version = self.applied_anchor_version
                self._interval_base_digest = self.applied_anchor_digest
            return True
        finally:
            lease.release()


@dataclass(frozen=True)
class AsyncV2Event:
    monotonic_ns: int
    phase: str
    local_window: int
    base_global_version: int
    details: dict[str, object]


class AsyncV21DescriptorService:
    """One-slot metadata scheduler around the persistent native dense service.

    ``handoff`` returns ``OWNED`` after a bounded local queue transfer.  The
    callback represents native handoff/transport/reduce plus fenced
    publication and executes only on the service thread.  There is no
    per-window FIFO and no replacement of an immutable cumulative descriptor.
    """

    def __init__(
        self,
        *,
        lane: AsyncV21WorkerLane,
        telemetry: Callable[[AsyncV2Event], None] | None = None,
    ):
        self.lane = lane
        self._telemetry = telemetry or (lambda _event: None)
        self._queue: queue.Queue[
            tuple[
                ContributionEnvelope,
                Callable[
                    [ContributionEnvelope, Callable[[str], None]],
                    ResultEnvelope | None,
                ],
            ]
            | None
        ] = queue.Queue(maxsize=1)
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="async-v2-native-descriptor-service",
            daemon=True,
        )
        self._thread.start()

    def event(
        self, contribution: ContributionEnvelope, phase: str, **details: object,
    ) -> None:
        self._telemetry(AsyncV2Event(
            monotonic_ns=time.monotonic_ns(),
            phase=phase,
            local_window=self.lane.local_window,
            base_global_version=contribution.identity.base_global_version,
            details={
                "contribution_digest": contribution.digest,
                "local_window_start":
                    contribution.identity.local_window_start,
                "local_window_end": contribution.identity.local_window_end,
                **details,
            },
        ))

    def handoff(
        self,
        contribution: ContributionEnvelope,
        run: Callable[
            [ContributionEnvelope, Callable[[str], None]],
            ResultEnvelope | None,
        ],
        *,
        deadline: float | None = None,
    ) -> str:
        contribution.validate(self.lane.policy)
        if self._closed:
            raise RuntimeError("descriptor service is closed")
        if contribution is not self.lane._sealed:
            raise ValueError("descriptor handoff does not own the lane's sealed value")
        started = time.monotonic()
        remaining = None if deadline is None else max(0.0, deadline - started)
        try:
            self._queue.put((contribution, run), timeout=remaining)
        except queue.Full as error:
            raise Backpressure("native descriptor slot is occupied") from error
        elapsed = time.monotonic() - started
        if elapsed > MAX_LOCAL_OWNED_SECONDS:
            raise Backpressure("local OWNED acknowledgement exceeded one second")
        self.event(
            contribution, "local_owned",
            owned_seconds=elapsed,
            queue_depth=1,
        )
        return "OWNED"

    def close(self, *, timeout: float = 30.0) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                # The worker will observe `_closed` and exit after its one
                # immutable descriptor.  Never allocate a second queue entry.
                pass
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("async-v2 descriptor service did not drain")
        if self._error is not None:
            raise RuntimeError("async-v2 descriptor service failed") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            contribution, run = item
            try:
                result = run(
                    contribution,
                    lambda phase: self.event(contribution, phase),
                )
                if result is not None:
                    status = self.lane.mailbox.publish(result)
                    self.event(
                        contribution,
                        "verified_result_ready",
                        result_version=result.version,
                        mailbox_status=status,
                    )
                    self.lane.release_sealed(
                        contribution.digest, outcome="accepted")
                else:
                    self.lane.release_sealed(
                        contribution.digest, outcome="not_selected")
            except BaseException as error:
                try:
                    self.lane.release_sealed(
                        contribution.digest, outcome="owner_abort")
                except ValueError:
                    pass
                self._error = error
                self.event(
                    contribution,
                    "background_failed",
                    error=type(error).__name__,
                )
            if self._closed and self._queue.empty():
                return


MAX_LOCAL_OWNED_SECONDS = 1.0


class AsyncV21CommitAuthority:
    """Fenced global/outer authority with deterministic replay and publication."""

    def __init__(
        self,
        *,
        run_id: str,
        fence: int,
        state: np.ndarray,
        version: int,
        outer: OuterState,
        policy: AsyncV21Policy,
        layout_digest: str,
        code_digest: str,
        version_digests: Mapping[int, str] | None = None,
        minimum_contributions: int | None = None,
        minimum_tokens: int | None = None,
    ):
        if not run_id or fence <= 0 or version < 0:
            raise ValueError("commit authority requires a fenced run/version")
        outer.validate()
        array = np.asarray(state, dtype=np.float64).copy()
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("invalid authoritative global state")
        _require_digest(layout_digest, "layout digest")
        _require_digest(code_digest, "code digest")
        self.run_id = run_id
        self.fence = fence
        self.state = array
        self.version = version
        self.outer = outer
        self.policy = policy
        self.layout_digest = layout_digest
        self.code_digest = code_digest
        self.version_digests = dict(version_digests or {
            version: digest_array(array)})
        for item_version, digest in self.version_digests.items():
            if item_version < 0:
                raise ValueError("negative version digest")
            _require_digest(digest, "version digest")
        self.minimum_contributions = (
            policy.q_min if minimum_contributions is None
            else minimum_contributions
        )
        self.minimum_tokens = (
            policy.t_min if minimum_tokens is None else minimum_tokens)
        if self.minimum_contributions <= 0 or self.minimum_tokens <= 0:
            raise ValueError("commit floors must be positive")
        self.membership: dict[str, str] = {}
        self._receipts: dict[tuple[object, ...], tuple[str, str]] = {}
        self._results_by_digest: dict[str, ResultEnvelope] = {}
        self._last_manifest: dict[str, object] | None = None

    @classmethod
    def for_test(cls) -> "AsyncV21CommitAuthority":
        return cls(
            run_id="run",
            fence=7,
            state=np.asarray([0.0]),
            version=0,
            outer=OuterState(),
            policy=ASYNC_DECOUPLED_V21,
            layout_digest="1" * 64,
            code_digest="2" * 64,
            version_digests={0: "0" * 64},
            minimum_contributions=1,
            minimum_tokens=1,
        )

    def install_membership(self, members: Mapping[str, str]) -> None:
        normalized = {str(worker): str(incarnation)
                      for worker, incarnation in members.items()}
        if not normalized or any(not key or not value
                                 for key, value in normalized.items()):
            raise ValueError("READY membership is empty or malformed")
        self.membership = normalized

    def replay_receipt(self, value: ContributionEnvelope) -> str:
        value.validate(self.policy)
        receipt = self._receipts.get(value.identity.logical_key)
        if receipt is None:
            raise KeyError("contribution has no committed receipt")
        envelope_digest, result_digest = receipt
        if envelope_digest != value.digest:
            raise ValueError("conflicting replay for logical contribution identity")
        return result_digest

    def _validate_record(self, value: ContributionEnvelope) -> None:
        value.validate(self.policy)
        identity = value.identity
        if identity.run_id != self.run_id:
            raise ValueError("wrong run identity")
        if identity.allocation_fence != self.fence:
            raise ValueError("contribution is not under the current fence")
        if identity.policy_digest != self.policy.digest:
            raise ValueError("wrong policy identity")
        if identity.layout_digest != self.layout_digest:
            raise ValueError("wrong layout identity")
        if identity.code_digest != self.code_digest:
            raise ValueError("wrong code identity")
        expected_base = self.version_digests.get(identity.base_global_version)
        if expected_base is None or expected_base != identity.base_global_digest:
            raise ValueError("wrong or unverifiable base digest")
        if self.membership:
            incarnation = self.membership.get(identity.worker_id)
            if incarnation != identity.worker_incarnation:
                raise ValueError("worker incarnation is not in current READY membership")

    def commit(
        self,
        records: Sequence[ContributionEnvelope],
        *,
        publish: Callable[[ResultEnvelope], object] | None = None,
        owner_apply: Callable[
            [int, tuple[AdmittedContribution, ...]], object
        ] | None = None,
    ) -> ResultEnvelope:
        if not records:
            raise ValueError("cannot commit an empty group")
        for value in records:
            self._validate_record(value)
        replayed: list[tuple[str, ResultEnvelope]] = []
        for value in records:
            receipt = self._receipts.get(value.identity.logical_key)
            if receipt is None:
                continue
            envelope_digest, result_digest = receipt
            if envelope_digest != value.digest:
                raise ValueError(
                    "conflicting replay for logical contribution identity")
            result = self._results_by_digest.get(result_digest)
            if result is None:
                raise ValueError("committed replay result is unavailable")
            replayed.append((result_digest, result))
        if replayed:
            if len(replayed) != len(records):
                raise ValueError(
                    "committed replay cannot re-enter a new frozen group")
            digests = {digest for digest, _result in replayed}
            if len(digests) != 1:
                raise ValueError("replayed group refers to multiple commits")
            return replayed[0][1]
        if len({value.identity.worker_id for value in records}) < (
                self.minimum_contributions):
            raise ValueError("READY contribution quorum floor is not met")
        mean, accepted_tokens, admitted = reference_aggregate(
            self.version, records, policy=self.policy)
        if accepted_tokens < self.minimum_tokens:
            raise ValueError("exact accepted-token floor is not met")
        if owner_apply is not None:
            last_error: OSError | None = None
            for attempt in range(self.policy.owner_reassignments + 1):
                try:
                    owner_apply(attempt, admitted)
                    last_error = None
                    break
                except OSError as error:
                    last_error = error
            if last_error is not None:
                raise last_error
        next_state = np.add(
            self.state,
            np.multiply(
                mean, np.float64(self.policy.eta_outer), dtype=np.float64),
            dtype=np.float64,
        )
        if not np.isfinite(next_state).all():
            raise ValueError("nonfinite outer result")
        next_outer = OuterState(
            mode=self.policy.outer_mode,
            eta_outer=self.policy.eta_outer,
            step=self.outer.step + 1,
            accepted_tokens=self.outer.accepted_tokens + accepted_tokens,
        )
        manifest = {
            "schema": self.policy.manifest_schema,
            "policy_id": self.policy.policy_id,
            "policy_schema": self.policy.policy_schema,
            "native_abi": self.policy.native_abi,
            "wire_protocol": {
                "major": self.policy.wire_protocol_major,
                "minor": self.policy.wire_protocol_minor,
            },
            "run_id": self.run_id,
            "allocation_fence": self.fence,
            "base_version": self.version,
            "base_digest": self.version_digests[self.version],
            "result_version": self.version + 1,
            "policy_digest": self.policy.digest,
            "layout_digest": self.layout_digest,
            "code_digest": self.code_digest,
            "outer": asdict(next_outer),
            "accepted_tokens": accepted_tokens,
            "selected": [
                {
                    "worker_id": item.worker_id,
                    "contribution_digest": item.contribution_digest,
                    "base_version": item.base_version,
                    "commit_lag": item.commit_lag,
                    "exact_tokens": item.exact_tokens,
                }
                for item in admitted
            ],
        }
        manifest_digest = _sha256(_canonical(manifest))
        candidate = ResultEnvelope.create(
            run_id=self.run_id,
            allocation_fence=self.fence,
            version=self.version + 1,
            base_version=self.version,
            base_digest=self.version_digests[self.version],
            state=next_state,
            outer=next_outer,
            policy_digest=self.policy.digest,
            layout_digest=self.layout_digest,
            code_digest=self.code_digest,
            manifest_digest=manifest_digest,
            selected_contribution_digests=tuple(
                item.contribution_digest for item in admitted),
            reload_verified=True,
            latest_cas_verified=True,
        )
        candidate.validate()
        # Publication is the commit point.  No authority or replay receipt
        # changes before this callback has reload-verified immutable state and
        # advanced `latest` under the current fence.
        if publish is not None:
            publish(candidate)
        self.state = next_state
        self.version = candidate.version
        self.outer = next_outer
        self.version_digests[self.version] = candidate.result_digest
        self._results_by_digest[candidate.result_digest] = candidate
        self._last_manifest = manifest
        for value in records:
            key = value.identity.logical_key
            old = self._receipts.get(key)
            receipt = (value.digest, candidate.result_digest)
            if old is not None and old != receipt:
                raise ValueError("conflicting replay receipt after publication")
            self._receipts[key] = receipt
        return candidate

    @property
    def last_manifest(self) -> Mapping[str, object] | None:
        return self._last_manifest

    def checkpoint(self, path: str | Path) -> Path:
        """Write a deterministic small-reference checkpoint atomically."""
        target = Path(path)
        value = {
            "schema": self.policy.checkpoint_schema,
            "run_id": self.run_id,
            "fence": self.fence,
            "version": self.version,
            "state": self.state.tolist(),
            "state_shape": list(self.state.shape),
            "state_digest": digest_array(self.state),
            "outer": asdict(self.outer),
            "policy": self.policy.manifest(),
            "layout_digest": self.layout_digest,
            "code_digest": self.code_digest,
            "version_digests": {
                str(version): digest
                for version, digest in sorted(self.version_digests.items())
            },
            "last_manifest": self._last_manifest,
        }
        value["bundle_digest"] = _sha256(_canonical(value))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(_canonical(value) + b"\n")
        os.replace(temporary, target)
        return target

    @classmethod
    def restore(
        cls,
        path: str | Path,
        *,
        new_fence: int,
        expected_run_id: str,
        expected_policy: AsyncV21Policy,
        expected_layout_digest: str,
        expected_code_digest: str,
    ) -> "AsyncV21CommitAuthority":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        digest = value.pop("bundle_digest", None)
        if digest != _sha256(_canonical(value)):
            raise ValueError("checkpoint bundle digest mismatch")
        if (
            value.get("schema")
            != expected_policy.checkpoint_schema
            or value.get("run_id") != expected_run_id
            or value.get("policy", {}).get("policy_digest")
            != expected_policy.digest
            or value.get("layout_digest") != expected_layout_digest
            or value.get("code_digest") != expected_code_digest
        ):
            raise ValueError("checkpoint identity mismatch")
        old_fence = int(value["fence"])
        if new_fence <= old_fence:
            raise ValueError("fresh allocation requires a strictly newer fence")
        state = np.asarray(value["state"], dtype=np.float64).reshape(
            tuple(int(item) for item in value["state_shape"]))
        if digest_array(state) != value.get("state_digest"):
            raise ValueError("checkpoint global state digest mismatch")
        outer = OuterState(**value["outer"])
        authority = cls(
            run_id=expected_run_id,
            fence=new_fence,
            state=state,
            version=int(value["version"]),
            outer=outer,
            policy=expected_policy,
            layout_digest=expected_layout_digest,
            code_digest=expected_code_digest,
            version_digests={
                int(version): digest
                for version, digest in value["version_digests"].items()
            },
            minimum_contributions=expected_policy.q_min,
            minimum_tokens=expected_policy.t_min,
        )
        authority._last_manifest = value.get("last_manifest")
        # Membership, local/coalescing/inner state, mailboxes, and replay
        # routes are intentionally fresh-allocation disposable state.
        return authority


class AtomicEightTrainerApply:
    """Fenced all-lane apply/recovery transaction for one stable node.

    Per-trainer files are volatile preparation evidence.  Only the immutable
    node marker returned by :meth:`commit_node` permits READY at the result
    version.  A partial transaction is never promoted; restart removes all
    partial markers and creates a fresh node/trainer incarnation set.
    """

    SCHEMA = "emender-async-v21-node-applied-v1"

    def __init__(
        self,
        *,
        root: str | Path,
        run_id: str,
        fence: int,
        node_id: str,
        node_incarnation: str,
        result_version: int,
        result_digest: str,
        trainer_count: int = 8,
    ):
        if (
            not run_id
            or fence <= 0
            or not node_id
            or not node_incarnation
            or result_version <= 0
            or trainer_count != 8
        ):
            raise ValueError("v2.1 node apply requires one fenced eight-trainer node")
        _require_digest(result_digest, "result digest")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.fence = int(fence)
        self.node_id = node_id
        self.node_incarnation = node_incarnation
        self.result_version = int(result_version)
        self.result_digest = result_digest
        self.trainer_count = trainer_count
        self._records: dict[int, dict[str, object]] = {}
        self._node_marker: dict[str, object] | None = None

    @property
    def ready(self) -> bool:
        return self._node_marker is not None

    def _trainer_path(self, rank: int) -> Path:
        return self.root / (
            f"trainer-applied-v{self.result_version:08d}-r{rank:02d}.json")

    @property
    def node_marker_path(self) -> Path:
        return self.root / (
            f"node-applied-v{self.result_version:08d}-{self.node_id}.json")

    @staticmethod
    def _atomic_write(path: Path, value: Mapping[str, object]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(_canonical(value) + b"\n")
        os.replace(temporary, path)

    def record_trainer(
        self,
        *,
        rank: int,
        trainer_incarnation: str,
        recovery_digest: str,
    ) -> Mapping[str, object]:
        if rank not in range(self.trainer_count) or not trainer_incarnation:
            raise ValueError("invalid trainer apply identity")
        _require_digest(recovery_digest, "recovery digest")
        value = {
            "schema": "emender-async-v21-trainer-applied-v1",
            "run_id": self.run_id,
            "allocation_fence": self.fence,
            "node_id": self.node_id,
            "node_incarnation": self.node_incarnation,
            "result_version": self.result_version,
            "result_digest": self.result_digest,
            "rank": int(rank),
            "trainer_incarnation": trainer_incarnation,
            "recovery_digest": recovery_digest,
        }
        old = self._records.get(rank)
        # Identical duplicate or delayed receipts remain idempotent even when
        # the node marker was already committed.  A conflicting late receipt
        # can never rewrite the atomic transaction.
        if old is not None:
            if old != value:
                raise ValueError("conflicting trainer apply marker")
            return old
        if self.ready:
            raise ValueError("node apply transaction is already committed")
        self._atomic_write(self._trainer_path(rank), value)
        self._records[rank] = value
        return value

    def commit_node(self) -> Mapping[str, object]:
        if self.ready:
            assert self._node_marker is not None
            return self._node_marker
        expected = set(range(self.trainer_count))
        if set(self._records) != expected:
            raise Backpressure(
                "all eight matching trainer apply/recovery markers are required")
        trainers = [self._records[rank] for rank in sorted(self._records)]
        if any(
            record["node_incarnation"] != self.node_incarnation
            or record["result_digest"] != self.result_digest
            or record["result_version"] != self.result_version
            for record in trainers
        ) or len({
            str(record["trainer_incarnation"]) for record in trainers
        }) != self.trainer_count:
            raise ValueError("trainer apply markers do not form one node transaction")
        value: dict[str, object] = {
            "schema": self.SCHEMA,
            "run_id": self.run_id,
            "allocation_fence": self.fence,
            "node_id": self.node_id,
            "node_incarnation": self.node_incarnation,
            "result_version": self.result_version,
            "result_digest": self.result_digest,
            "trainers": trainers,
        }
        value["transaction_digest"] = _sha256(_canonical(value))
        self._atomic_write(self.node_marker_path, value)
        self._node_marker = value
        return value

    def restart_from_latest(
        self,
        *,
        new_node_incarnation: str,
        trainer_incarnations: Sequence[str],
    ) -> "AtomicEightTrainerApply":
        if (
            not new_node_incarnation
            or new_node_incarnation == self.node_incarnation
            or len(trainer_incarnations) != self.trainer_count
            or len(set(trainer_incarnations)) != self.trainer_count
            or any(not value for value in trainer_incarnations)
        ):
            raise ValueError("restart requires fresh node and eight trainer incarnations")
        # These files are explicitly volatile partial-apply state.  Preserve
        # them below their failed fenced incarnation for diagnosis, but remove
        # them from the active transaction namespace.  The durable verified
        # latest/result remains untouched.
        failed = self.root / "failed-cohorts" / self.node_incarnation
        failed.mkdir(parents=True, exist_ok=True)
        for rank in range(self.trainer_count):
            path = self._trainer_path(rank)
            if path.exists():
                os.replace(path, failed / path.name)
        if self.node_marker_path.exists():
            os.replace(
                self.node_marker_path, failed / self.node_marker_path.name)
        return AtomicEightTrainerApply(
            root=self.root,
            run_id=self.run_id,
            fence=self.fence,
            node_id=self.node_id,
            node_incarnation=new_node_incarnation,
            result_version=self.result_version,
            result_digest=self.result_digest,
            trainer_count=self.trainer_count,
        )


# The historical class names remain import aliases so adjacent v2 scaffolding
# can migrate without duplicating implementations.  Their runtime identity is
# always the fail-closed v2.1 policy above.
AsyncV2WorkerLane = AsyncV21WorkerLane
AsyncV2DescriptorService = AsyncV21DescriptorService
AsyncV2CommitAuthority = AsyncV21CommitAuthority


__all__ = [
    "ASYNC_DECOUPLED_V21",
    "ASYNC_DECOUPLED_V2",
    "AdmittedContribution",
    "AsyncV21CommitAuthority",
    "AsyncV21DescriptorService",
    "AsyncV21Policy",
    "AsyncV21WorkerLane",
    "AsyncV2CommitAuthority",
    "AsyncV2DescriptorService",
    "AsyncV2Event",
    "AsyncV2Policy",
    "AsyncV2WorkerLane",
    "Backpressure",
    "ContributionEnvelope",
    "ContributionIdentity",
    "LatestResultMailbox",
    "AtomicEightTrainerApply",
    "OuterState",
    "ResultEnvelope",
    "ScheduleFreeLocalState",
    "StaleContribution",
    "build_contribution",
    "digest_array",
    "reference_aggregate",
    "rebase_schedulefree_torch_state",
    "translate_schedulefree_optimizer_points",
]
