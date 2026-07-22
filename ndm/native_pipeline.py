"""Bounded foreground/background handoff for the native DiLoCo runtime.

This module owns no tensors and performs no transport.  It is the small,
thread-safe policy layer between model-owning trainers and the persistent
native service: two immutable contribution slots and one latest-only,
durably-committed result slot.  Keeping policy here makes the safe-boundary
rules independently testable without creating a second dense data plane.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import threading
import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class SlotState(Enum):
    FREE = "free"
    WRITING = "writing"
    HANDED_OFF = "handed_off"


@dataclass(frozen=True)
class GenerationIdentity:
    run_id: str
    fence: int
    generation: int
    attempt: int
    incarnation: str
    layout_digest: str
    base_digest: str
    source_id: str = "local"
    route_id: str = "native"
    lease_id: str = "allocation"

    def validate(self) -> None:
        if (not self.run_id or not self.incarnation or self.fence <= 0
                or self.generation < 0 or self.attempt <= 0
                or len(self.layout_digest) != 64 or len(self.base_digest) != 64
                or not self.source_id or not self.route_id or not self.lease_id):
            raise ValueError("invalid pipelined generation identity")


@dataclass(frozen=True)
class Handoff(Generic[T]):
    slot: int
    identity: GenerationIdentity
    payload: T
    weight: int
    digest: str
    handed_off_ns: int


@dataclass(frozen=True)
class CommittedResult(Generic[T]):
    identity: GenerationIdentity
    payload: T
    result_digest: str
    membership_root: str
    global_weight: int
    committed_ns: int
    complete: bool = True

    def validate(self) -> None:
        self.identity.validate()
        if (not self.complete or self.global_weight <= 0
                or len(self.result_digest) != 64
                or len(self.membership_root) != 64):
            raise ValueError("partial or malformed pipelined result")


@dataclass(frozen=True)
class PipelineMetrics:
    handoffs: int = 0
    handoff_replacements: int = 0
    result_replacements: int = 0
    stale_results: int = 0
    rejected_results: int = 0
    applied_results: int = 0
    foreground_wait_ns: int = 0
    handoff_high_water: int = 0
    result_high_water: int = 0
    backpressure_events: int = 0
    dropped_handoffs: int = 0


@dataclass(frozen=True)
class PipelineEvent:
    monotonic_ns: int
    generation: int
    phase: str
    identity: GenerationIdentity
    details: dict[str, object]


@dataclass(frozen=True)
class BackgroundWork(Generic[T]):
    """Immutable work captured at a K boundary.

    ``run`` owns collection through durable checkpoint publication.  It must
    return only a completely published result; an exception is non-
    participation for that round and never poisons the foreground thread.
    """
    identity: GenerationIdentity
    payload: T
    run: Callable[[T, Callable[[str], None]], CommittedResult[T] | None]


class LiveNativeGenerationScheduler(Generic[T]):
    """One-worker, latest-only native generation scheduler.

    The training thread performs only an immutable snapshot and ``enqueue``.
    Native collection/reduce/scan/redistribution/checkpoint work executes on
    the service thread.  At most one queued item and one running item exist.
    When the queue is occupied a newer generation replaces it; when the
    configured safety policy requires backpressure, the wait is deadline
    bounded and reported in both metrics and telemetry.
    """

    BACKGROUND_PHASES = ("collection", "reduction", "integrity_scan",
                         "redistribution", "checkpoint_publication")

    def __init__(self, pipeline: "NativeGenerationPipeline[T]", *,
                 telemetry: Callable[[PipelineEvent], None] | None = None,
                 result_delay: int = 0):
        if result_delay not in (0, 1):
            raise ValueError("native result delay must be zero or one generation")
        self.pipeline = pipeline
        self.result_delay = result_delay
        self._telemetry = telemetry or (lambda event: None)
        self._condition = threading.Condition()
        self._queued: BackgroundWork[T] | None = None
        self._running: BackgroundWork[T] | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._service_loop,
                                        name="native-generation-service", daemon=True)
        self._thread.start()

    def event(self, identity: GenerationIdentity, phase: str, **details: object) -> None:
        self._telemetry(PipelineEvent(time.monotonic_ns(), identity.generation,
                                      phase, identity, details))

    def enqueue(self, work: BackgroundWork[T], *, deadline: float | None = None,
                require_backpressure: bool = False) -> bool:
        work.identity.validate()
        if (work.identity.run_id, work.identity.fence, work.identity.incarnation) != (
                self.pipeline.run_id, self.pipeline.fence, self.pipeline.incarnation):
            raise ValueError("background work belongs to an obsolete identity")
        started = time.monotonic_ns()
        with self._condition:
            while require_backpressure and self._queued is not None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self.pipeline._increment_metric("backpressure_events")
                    self.event(work.identity, "backpressure_timeout",
                               waited_ns=time.monotonic_ns() - started)
                    return False
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("scheduler is closed")
            old = self._queued
            if old is not None:
                if work.identity.generation <= old.identity.generation:
                    self.pipeline._increment_metric("dropped_handoffs")
                    self.event(work.identity, "enqueue_rejected_stale")
                    return False
                self.pipeline._increment_metric("handoff_replacements")
                self.pipeline._increment_metric("dropped_handoffs")
                self.event(old.identity, "enqueue_replaced",
                           replacement_generation=work.identity.generation)
            self._queued = work
            self.event(work.identity, "enqueue", queue_depth=1)
            self._condition.notify_all()
            return True

    def apply_at_safe_boundary(self, identity: GenerationIdentity, *,
                               apply: Callable[[CommittedResult[T]], None]) -> bool:
        value = self.pipeline.take_at_boundary(
            trainer_generation=identity.generation - self.result_delay,
            boundary_generation=identity.generation, fence=identity.fence,
            incarnation=identity.incarnation, base_digest=identity.base_digest)
        if value is None:
            return False
        apply(value)
        self.event(identity, "safe_boundary_apply",
                   source_generation=value.identity.generation,
                   result_delay=self.result_delay,
                   result_digest=value.result_digest)
        return True

    def close(self, *, drain: bool = True, timeout: float = 30.0) -> None:
        with self._condition:
            self._closed = True
            if not drain:
                self._queued = None
            self._condition.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("native generation service did not stop within bound")

    def _service_loop(self) -> None:
        while True:
            with self._condition:
                while self._queued is None and not self._closed:
                    self._condition.wait()
                if self._queued is None and self._closed:
                    return
                work, self._queued = self._queued, None
                self._running = work
                self._condition.notify_all()
            assert work is not None
            try:
                result = work.run(work.payload,
                                  lambda phase: self.event(work.identity, phase))
                if result is not None and self.pipeline.publish_committed(
                        result, verify=finite_result_verifier):
                    self.event(work.identity, "accepted_result_ready",
                               result_digest=result.result_digest)
                elif result is not None:
                    self.event(work.identity, "accepted_result_rejected")
            except Exception as error:  # round failure is non-participation
                self.event(work.identity, "background_failed",
                           error=type(error).__name__)
            finally:
                with self._condition:
                    self._running = None
                    self._condition.notify_all()


class NativeGenerationPipeline(Generic[T]):
    """Thread-safe bounded ownership and safe-boundary admission policy.

    Producers reserve one of exactly two slots, seal it, and hand ownership to
    the background service.  A service receipt/abort releases that slot.  Only
    a complete, verified *and durably committed* result enters the mailbox.
    Trainers call :meth:`take_at_boundary` only after finishing a K window.
    """

    def __init__(self, *, run_id: str, fence: int, incarnation: str):
        if not run_id or fence <= 0 or not incarnation:
            raise ValueError("pipeline requires a fenced incarnation")
        self.run_id, self.fence, self.incarnation = run_id, fence, incarnation
        self._states = [SlotState.FREE, SlotState.FREE]
        self._handoffs: list[Handoff[T] | None] = [None, None]
        self._mailbox: CommittedResult[T] | None = None
        self._metrics = PipelineMetrics()
        self._condition = threading.Condition()
        self._closed = False

    @property
    def metrics(self) -> PipelineMetrics:
        with self._condition:
            return self._metrics

    def reserve(self, *, deadline: float | None = None) -> int:
        """Reserve a free writer slot; wait is bounded when both are owned."""
        started = time.monotonic_ns()
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("pipeline is closed")
                for index, state in enumerate(self._states):
                    if state is SlotState.FREE:
                        self._states[index] = SlotState.WRITING
                        self._metrics = replace(
                            self._metrics,
                            foreground_wait_ns=self._metrics.foreground_wait_ns
                            + time.monotonic_ns() - started)
                        return index
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._metrics = replace(
                        self._metrics,
                        foreground_wait_ns=self._metrics.foreground_wait_ns
                        + time.monotonic_ns() - started)
                    raise TimeoutError("both native handoff slots remain owned")
                self._condition.wait(remaining)

    def cancel_reservation(self, slot: int) -> None:
        with self._condition:
            self._require_slot(slot, SlotState.WRITING)
            self._states[slot] = SlotState.FREE
            self._condition.notify_all()

    def handoff(self, slot: int, identity: GenerationIdentity, payload: T, *,
                weight: int, digest: str) -> Handoff[T]:
        identity.validate()
        if (identity.run_id, identity.fence, identity.incarnation) != (
                self.run_id, self.fence, self.incarnation):
            raise ValueError("handoff belongs to an obsolete fence/incarnation")
        if weight <= 0 or len(digest) != 64:
            raise ValueError("handoff weight/digest is invalid")
        with self._condition:
            self._require_slot(slot, SlotState.WRITING)
            value = Handoff(slot, identity, payload, weight, digest, time.monotonic_ns())
            self._handoffs[slot] = value
            self._states[slot] = SlotState.HANDED_OFF
            depth = sum(state is SlotState.HANDED_OFF for state in self._states)
            self._metrics = replace(
                self._metrics, handoffs=self._metrics.handoffs + 1,
                handoff_high_water=max(self._metrics.handoff_high_water, depth))
            self._condition.notify_all()
            return value

    def release(self, handoff: Handoff[T]) -> None:
        """Release only the exact immutable ownership token (replay-safe)."""
        with self._condition:
            self._require_slot(handoff.slot, SlotState.HANDED_OFF)
            if self._handoffs[handoff.slot] is not handoff:
                raise ValueError("stale handoff receipt")
            self._handoffs[handoff.slot] = None
            self._states[handoff.slot] = SlotState.FREE
            self._condition.notify_all()

    def publish_committed(self, result: CommittedResult[T], *,
                          verify: Callable[[CommittedResult[T]], bool]) -> bool:
        """Admit a verified result after the durable publisher's CAS succeeds."""
        try:
            result.validate()
            if (result.identity.run_id != self.run_id
                    or result.identity.fence != self.fence
                    or result.identity.incarnation != self.incarnation
                    or not verify(result)):
                raise ValueError("result verification failed")
        except (ValueError, OverflowError):
            with self._condition:
                self._metrics = replace(
                    self._metrics, rejected_results=self._metrics.rejected_results + 1)
            return False
        with self._condition:
            old = self._mailbox
            if old is not None and result.identity.generation <= old.identity.generation:
                self._metrics = replace(
                    self._metrics, stale_results=self._metrics.stale_results + 1)
                return False
            self._mailbox = result
            self._metrics = replace(
                self._metrics,
                result_replacements=self._metrics.result_replacements
                + int(old is not None), result_high_water=1)
            return True

    def take_at_boundary(self, *, trainer_generation: int, fence: int,
                         boundary_generation: int | None = None,
                         incarnation: str, base_digest: str) -> CommittedResult[T] | None:
        """Return the newest admissible result without waiting for control work."""
        if boundary_generation is None:
            boundary_generation = trainer_generation
        if boundary_generation < trainer_generation:
            raise ValueError("safe boundary cannot precede the result generation")
        with self._condition:
            value = self._mailbox
            if value is None:
                return None
            identity = value.identity
            if (fence != self.fence or incarnation != self.incarnation
                    or identity.generation != trainer_generation
                    or identity.base_digest != base_digest):
                if identity.generation < trainer_generation:
                    self._mailbox = None
                    self._metrics = replace(
                        self._metrics, stale_results=self._metrics.stale_results + 1)
                return None
            self._mailbox = None
            self._metrics = replace(
                self._metrics, applied_results=self._metrics.applied_results + 1)
            return value

    def rebind(self, *, fence: int, incarnation: str) -> None:
        """Invalidate all volatile state on owner restart/new allocation fence."""
        if fence < self.fence or fence <= 0 or not incarnation:
            raise ValueError("pipeline fence cannot move backwards")
        with self._condition:
            self.fence, self.incarnation = fence, incarnation
            self._states = [SlotState.FREE, SlotState.FREE]
            self._handoffs = [None, None]
            self._mailbox = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _increment_metric(self, name: str) -> None:
        """Internal atomic counter shared with the scheduler policy layer."""
        with self._condition:
            if name not in PipelineMetrics.__dataclass_fields__:
                raise ValueError(f"unknown pipeline metric {name}")
            self._metrics = replace(
                self._metrics, **{name: getattr(self._metrics, name) + 1})

    def _require_slot(self, slot: int, expected: SlotState) -> None:
        if slot not in (0, 1) or self._states[slot] is not expected:
            raise ValueError(f"slot {slot} is not {expected.value}")


def finite_result_verifier(result: CommittedResult[object]) -> bool:
    """Small-value fixture verifier; dense native buffers verify while streaming."""
    payload = result.payload
    if isinstance(payload, (int, float)):
        return math.isfinite(float(payload))
    return True
