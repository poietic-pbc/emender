"""Leased peer lifecycle for the resilient DiLoCo compute pool.

Conforms to ``RESILIENT_DILOCO_COMPUTE_POOL.md`` version 1 requirements
R02, R03, R11, and R14.  This is an in-memory control-plane primitive: it has
no scheduler-size input, collective communication, tensor broker, or durable
filesystem hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import secrets
import time
from typing import Callable, Mapping


class PeerState(str, Enum):
    DISCOVER = "DISCOVER"
    BOOTING = "BOOTING"
    SYNCING = "SYNCING"
    READY = "READY"
    DRAINING = "DRAINING"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class StageDeadlines:
    first_heartbeat: float
    boot: float
    sync: float
    lease: float
    drain: float

    def __post_init__(self) -> None:
        if any(value <= 0 for value in self.__dict__.values()):
            raise ValueError("stage deadlines must be positive")


@dataclass(frozen=True)
class PeerRecord:
    worker_id: str
    incarnation: str
    state: PeerState
    base_generation: int | None
    entered_at: float
    deadline: float
    lease_expiry: float | None = None


@dataclass(frozen=True)
class ActivePeer:
    worker_id: str
    incarnation: str
    base_generation: int
    lease_expiry: float


@dataclass(frozen=True)
class ActiveSnapshot:
    generation: int
    observed_at: float
    peers: tuple[ActivePeer, ...]

    @property
    def size(self) -> int:
        return len(self.peers)


class PeerMembership:
    """State machine whose active set is exactly live synchronized READY peers."""

    def __init__(self, deadlines: StageDeadlines, *,
                 clock: Callable[[], float] = time.monotonic):
        self.deadlines = deadlines
        self._clock = clock
        self._records: dict[str, PeerRecord] = {}

    @property
    def records(self) -> Mapping[str, PeerRecord]:
        return dict(self._records)

    def discover(self, worker_id: str, *, incarnation: str | None = None) -> PeerRecord:
        if not worker_id:
            raise ValueError("worker_id must be stable and non-empty")
        now = self._clock()
        old = self._records.get(worker_id)
        if old is not None and old.state is not PeerState.EXPIRED:
            self._records[worker_id] = replace(old, state=PeerState.EXPIRED,
                                                entered_at=now, deadline=now,
                                                lease_expiry=None)
        token = incarnation or secrets.token_hex(16)
        if old is not None and token == old.incarnation:
            raise ValueError("a returning worker requires a new incarnation")
        record = PeerRecord(worker_id, token, PeerState.DISCOVER, None, now,
                            now + self.deadlines.first_heartbeat)
        self._records[worker_id] = record
        return record

    def begin_boot(self, worker_id: str, incarnation: str) -> PeerRecord:
        return self._transition(worker_id, incarnation, PeerState.DISCOVER,
                                PeerState.BOOTING, self.deadlines.boot)

    def begin_sync(self, worker_id: str, incarnation: str) -> PeerRecord:
        return self._transition(worker_id, incarnation, PeerState.BOOTING,
                                PeerState.SYNCING, self.deadlines.sync)

    def ready(self, worker_id: str, incarnation: str, *,
              base_generation: int) -> PeerRecord:
        if base_generation < 0:
            raise ValueError("base_generation must be non-negative")
        record = self._require(worker_id, incarnation, PeerState.SYNCING)
        now = self._clock()
        ready = replace(record, state=PeerState.READY,
                        base_generation=base_generation, entered_at=now,
                        deadline=now + self.deadlines.lease,
                        lease_expiry=now + self.deadlines.lease)
        self._records[worker_id] = ready
        return ready

    def renew(self, worker_id: str, incarnation: str, *,
              base_generation: int) -> PeerRecord:
        self.expire_due()
        record = self._require(worker_id, incarnation, PeerState.READY)
        if record.base_generation != base_generation:
            raise ValueError("lease renewal generation differs; peer must catch up")
        now = self._clock()
        renewed = replace(record, deadline=now + self.deadlines.lease,
                          lease_expiry=now + self.deadlines.lease)
        self._records[worker_id] = renewed
        return renewed

    def catch_up(self, worker_id: str, incarnation: str, *,
                 committed_generation: int) -> PeerRecord:
        """Move a live READY peer back through SYNCING after a late/stale result."""
        record = self._require(worker_id, incarnation, PeerState.READY)
        now = self._clock()
        syncing = replace(record, state=PeerState.SYNCING,
                          base_generation=committed_generation, entered_at=now,
                          deadline=now + self.deadlines.sync, lease_expiry=None)
        self._records[worker_id] = syncing
        return syncing

    def drain(self, worker_id: str, incarnation: str) -> PeerRecord:
        record = self._require(worker_id, incarnation, PeerState.READY)
        now = self._clock()
        draining = replace(record, state=PeerState.DRAINING, entered_at=now,
                           deadline=now + self.deadlines.drain, lease_expiry=None)
        self._records[worker_id] = draining
        return draining

    def expire_due(self) -> tuple[PeerRecord, ...]:
        now = self._clock()
        expired = []
        for worker_id, record in tuple(self._records.items()):
            if record.state is not PeerState.EXPIRED and now >= record.deadline:
                record = replace(record, state=PeerState.EXPIRED, entered_at=now,
                                 deadline=now, lease_expiry=None)
                self._records[worker_id] = record
                expired.append(record)
        return tuple(expired)

    def active_snapshot(self, generation: int) -> ActiveSnapshot:
        """Freeze live READY peers synchronized to ``generation``; never launch size."""
        self.expire_due()
        now = self._clock()
        peers = tuple(sorted((ActivePeer(r.worker_id, r.incarnation,
                                         r.base_generation, r.lease_expiry)
                              for r in self._records.values()
                              if r.state is PeerState.READY
                              and r.base_generation == generation
                              and r.lease_expiry is not None
                              and r.lease_expiry > now),
                             key=lambda peer: (peer.worker_id, peer.incarnation)))
        return ActiveSnapshot(generation, now, peers)

    def _require(self, worker_id: str, incarnation: str,
                 state: PeerState) -> PeerRecord:
        record = self._records.get(worker_id)
        if record is None or record.incarnation != incarnation:
            raise ValueError("unknown or superseded peer incarnation")
        if record.state is not state:
            raise ValueError(f"peer must be {state.value}, not {record.state.value}")
        if self._clock() >= record.deadline:
            self.expire_due()
            raise TimeoutError(f"{state.value} stage deadline expired")
        return record

    def _transition(self, worker_id: str, incarnation: str, source: PeerState,
                    target: PeerState, duration: float) -> PeerRecord:
        record = self._require(worker_id, incarnation, source)
        now = self._clock()
        changed = replace(record, state=target, entered_at=now,
                          deadline=now + duration, lease_expiry=None)
        self._records[worker_id] = changed
        return changed
