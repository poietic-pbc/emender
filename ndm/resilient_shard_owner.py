"""Owner replay, redistribution, and catch-up for Compute Pool v1 R08/R11/R14.

The implementation is deliberately transport-neutral and memory-only: callers move
the returned chunks over independent point-to-point connections.  It models the
protocol boundary at which sender-retained chunks become an atomic committed
generation, without using Lustre or assembling a full-model broker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
from typing import Callable, Mapping, Sequence

from ndm.resilient_e97_reducer import ExactWeightedShardReducer, ShardChunk, TensorLayout
from ndm.resilient_peer_membership import ActiveSnapshot


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class OwnerDeadlines:
    transport: float
    replay: float
    apply: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError("transport, replay, and apply deadlines must be finite and positive")


@dataclass(frozen=True)
class TrafficMetrics:
    p2p_bytes_sent: int = 0
    replay_bytes_sent: int = 0
    redistribution_bytes_sent: int = 0
    peak_retained_bytes: int = 0
    peak_owner_bytes: int = 0
    released_bytes: int = 0


@dataclass(frozen=True)
class CommittedGeneration:
    run_id: str
    generation: int
    attempt: int
    layout_digest: str
    accepted_ids: tuple[str, ...]
    chunks: tuple[ShardChunk, ...]
    manifest_digest: str


class ShardOwnerGeneration:
    """One bounded generation attempt with deterministic replaceable owners."""

    def __init__(self, layout: TensorLayout, snapshot: ActiveSnapshot, owners: Sequence[str],
                 *, run_id: str, generation: int, attempt: int,
                 max_retained_bytes: int, max_owner_bytes: int,
                 deadlines: OwnerDeadlines, clock: Callable[[], float] = time.monotonic):
        if snapshot.generation != generation:
            raise ValueError("READY snapshot generation mismatch")
        if max_retained_bytes <= 0 or max_owner_bytes <= 0:
            raise ValueError("byte bounds must be positive")
        unique_owners = tuple(sorted(set(owners)))
        if not unique_owners:
            raise ValueError("at least one live shard owner is required")
        self.layout, self.snapshot = layout, snapshot
        self.run_id, self.generation, self.attempt = run_id, generation, attempt
        self.max_retained_bytes, self.max_owner_bytes = max_retained_bytes, max_owner_bytes
        self.deadlines, self._clock = deadlines, clock
        self._owners = unique_owners
        self._placement = self._place(unique_owners)
        self._retained: dict[str, tuple[int, tuple[ShardChunk, ...]]] = {}
        self._reducers = self._new_reducers()
        self._committed: CommittedGeneration | None = None
        self._applied: dict[tuple[str, str], int] = {}
        self.metrics = TrafficMetrics()

    @property
    def placement(self) -> Mapping[int, str]:
        return dict(self._placement)

    @property
    def retained_bytes(self) -> int:
        return sum(chunk.nbytes for _, chunks in self._retained.values() for chunk in chunks)

    @property
    def committed(self) -> CommittedGeneration | None:
        return self._committed

    def submit(self, worker_id: str, incarnation: str, contribution_id: str, *,
               weight: int, chunks: Sequence[ShardChunk], deadline: float) -> None:
        self._bounded(deadline, self.deadlines.transport, "transport")
        if (worker_id, incarnation) not in {(p.worker_id, p.incarnation) for p in self.snapshot.peers}:
            raise ValueError("contribution source is not in the leased READY snapshot")
        ordered = tuple(sorted(chunks, key=lambda item: item.shard_id))
        if len(ordered) != self.layout.shard_count or tuple(c.shard_id for c in ordered) != tuple(
                range(self.layout.shard_count)):
            raise ValueError("contribution chunks are incomplete or duplicated")
        prior = self._retained.get(contribution_id)
        signature = (weight, tuple((c.checksum_sha256, c.nbytes) for c in ordered))
        if prior is not None:
            old_signature = (prior[0], tuple((c.checksum_sha256, c.nbytes) for c in prior[1]))
            if signature != old_signature:
                raise ValueError("conflicting contribution replay")
            return
        incoming = sum(chunk.nbytes for chunk in ordered)
        if self.retained_bytes + incoming > self.max_retained_bytes:
            raise BufferError("sender retained-byte backpressure bound exceeded")
        # Retain before sending so owner loss can never strand an accepted partial.
        self._retained[contribution_id] = (weight, ordered)
        try:
            for chunk in ordered:
                self._reducers[chunk.shard_id].submit(contribution_id, weight=weight, chunk=chunk)
        except Exception:
            self._retained.pop(contribution_id, None)
            self._rebuild_reducers()
            raise
        self.metrics = replace(self.metrics,
            p2p_bytes_sent=self.metrics.p2p_bytes_sent + incoming,
            peak_retained_bytes=max(self.metrics.peak_retained_bytes, self.retained_bytes),
            peak_owner_bytes=max(self.metrics.peak_owner_bytes,
                                 sum(r.inflight_bytes for r in self._reducers.values())))

    def lose_owner(self, owner: str, replacements: Sequence[str], *, deadline: float) -> None:
        self._bounded(deadline, self.deadlines.replay, "owner replay")
        if owner not in self._owners:
            return
        live = tuple(sorted((set(self._owners) - {owner}) | set(replacements)))
        if not live:
            raise TimeoutError("owner replay has no replacement before deadline")
        replay = self.retained_bytes
        self._owners, self._placement = live, self._place(live)
        self._rebuild_reducers()
        self.metrics = replace(self.metrics,
            p2p_bytes_sent=self.metrics.p2p_bytes_sent + replay,
            replay_bytes_sent=self.metrics.replay_bytes_sent + replay,
            peak_owner_bytes=max(self.metrics.peak_owner_bytes,
                                 sum(r.inflight_bytes for r in self._reducers.values())))

    def commit(self, accepted_ids: Sequence[str], *, deadline: float,
               fail_after_shards: int | None = None) -> CommittedGeneration:
        self._bounded(deadline, self.deadlines.apply, "aggregate/apply")
        if self._committed is not None:
            if tuple(sorted(accepted_ids)) != self._committed.accepted_ids:
                raise ValueError("generation already committed with another frozen set")
            return self._committed
        accepted = tuple(sorted(accepted_ids))
        outputs = []
        try:
            for shard_id in range(self.layout.shard_count):
                if fail_after_shards is not None and len(outputs) >= fail_after_shards:
                    raise TimeoutError("injected owner loss before atomic publication")
                outputs.append(self._reducers[shard_id].finalize(accepted))
        except Exception:
            # Finalize releases individual reducers. Reconstitute the entire attempt
            # from sender retention, leaving no partially published generation.
            self._rebuild_reducers()
            raise
        manifest_digest = self._manifest_digest(accepted, outputs)
        committed = CommittedGeneration(self.run_id, self.generation + 1, self.attempt,
                                        self.layout.digest, accepted, tuple(outputs),
                                        manifest_digest)
        # This is the sole publication point. Everything above is attempt-local.
        self._committed = committed
        released = self.retained_bytes
        self._retained.clear()
        self._reducers.clear()
        self.metrics = replace(self.metrics, released_bytes=self.metrics.released_bytes + released)
        return committed

    def catch_up(self, worker_id: str, incarnation: str, *, local_generation: int,
                 committed: CommittedGeneration, deadline: float,
                 chunks: Sequence[ShardChunk] | None = None) -> bool:
        self._bounded(deadline, self.deadlines.apply, "catch-up apply")
        key = (worker_id, incarnation)
        if committed.run_id != self.run_id or committed.layout_digest != self.layout.digest:
            raise ValueError("catch-up commit identity mismatch")
        if committed.generation != self.generation + 1 or local_generation > committed.generation:
            raise ValueError("stale catch-up generation rejected")
        expected_manifest = self._manifest_digest(committed.accepted_ids, committed.chunks)
        if committed.manifest_digest != expected_manifest:
            raise ValueError("committed generation manifest checksum mismatch")
        if local_generation == committed.generation or self._applied.get(key) == committed.generation:
            return False
        supplied = tuple(chunks if chunks is not None else committed.chunks)
        # unpack performs completeness, duplicate, bounds, checksum, and nonfinite checks.
        self.layout.unpack(supplied)
        if tuple((c.shard_id, c.checksum_sha256, c.nbytes) for c in supplied) != tuple(
                (c.shard_id, c.checksum_sha256, c.nbytes) for c in committed.chunks):
            raise ValueError("catch-up chunks differ from committed generation")
        sent = sum(c.nbytes for c in supplied)
        self._applied[key] = committed.generation
        self.metrics = replace(self.metrics,
            p2p_bytes_sent=self.metrics.p2p_bytes_sent + sent,
            redistribution_bytes_sent=self.metrics.redistribution_bytes_sent + sent)
        return True

    def _place(self, owners: Sequence[str]) -> dict[int, str]:
        return {shard: self.layout.owner(shard, owners, run_id=self.run_id,
                                         generation=self.generation, attempt=self.attempt)
                for shard in range(self.layout.shard_count)}

    def _new_reducers(self) -> dict[int, ExactWeightedShardReducer]:
        return {shard: ExactWeightedShardReducer(self.layout, shard,
                    max_inflight_bytes=self.max_owner_bytes)
                for shard in range(self.layout.shard_count)}

    def _rebuild_reducers(self) -> None:
        reducers = self._new_reducers()
        for identity in sorted(self._retained):
            weight, chunks = self._retained[identity]
            for chunk in chunks:
                reducers[chunk.shard_id].submit(identity, weight=weight, chunk=chunk)
        self._reducers = reducers

    def _manifest_digest(self, accepted: Sequence[str], chunks: Sequence[ShardChunk]) -> str:
        manifest = {
            "run_id": self.run_id, "generation": self.generation + 1,
            "attempt": self.attempt, "layout_digest": self.layout.digest,
            "accepted_ids": tuple(accepted),
            "chunks": [(c.shard_id, c.checksum_sha256, c.nbytes) for c in chunks],
        }
        return _digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())

    def _bounded(self, deadline: float, budget: float, stage: str) -> None:
        now = self._clock()
        if not math.isfinite(deadline) or deadline <= now or deadline - now > budget:
            raise TimeoutError(f"{stage} wait must have a live bounded deadline")
