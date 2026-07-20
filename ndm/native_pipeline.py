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

    def validate(self) -> None:
        if (not self.run_id or not self.incarnation or self.fence <= 0
                or self.generation < 0 or self.attempt <= 0
                or len(self.layout_digest) != 64 or len(self.base_digest) != 64):
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
                         incarnation: str, base_digest: str) -> CommittedResult[T] | None:
        """Return the newest admissible result without waiting for control work."""
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

    def _require_slot(self, slot: int, expected: SlotState) -> None:
        if slot not in (0, 1) or self._states[slot] is not expected:
            raise ValueError(f"slot {slot} is not {expected.value}")


def finite_result_verifier(result: CommittedResult[object]) -> bool:
    """Small-value fixture verifier; dense native buffers verify while streaming."""
    payload = result.payload
    if isinstance(payload, (int, float)):
        return math.isfinite(float(payload))
    return True
