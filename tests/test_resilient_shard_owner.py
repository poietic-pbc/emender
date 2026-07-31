from dataclasses import replace
import hashlib

import pytest
import torch

from ndm.resilient_e97_reducer import ShardChunk, TensorLayout
from ndm.resilient_peer_membership import ActivePeer, ActiveSnapshot
from ndm.resilient_shard_owner import OwnerDeadlines, ShardOwnerGeneration


class Clock:
    now = 10.0

    def __call__(self):
        return self.now


def fixture():
    states = [
        {"a": torch.arange(12, dtype=torch.float32).reshape(3, 4) + offset,
         "b": torch.tensor([offset, offset + 2], dtype=torch.float32)}
        for offset in (1, 5)
    ]
    layout = TensorLayout.from_state(states[0], max_chunk_bytes=32)
    peers = (ActivePeer("n0", "i0", 4, 100), ActivePeer("n1", "i1", 4, 100))
    snapshot = ActiveSnapshot(4, 10, peers)
    clock = Clock()
    protocol = ShardOwnerGeneration(layout, snapshot, ("owner-a", "owner-b"),
        run_id="run", generation=4, attempt=0, max_retained_bytes=4096,
        max_owner_bytes=4096, deadlines=OwnerDeadlines(5, 5, 5), clock=clock)
    return protocol, layout, states, clock


def submit_all(protocol, layout, states, deadline=12):
    protocol.submit("n0", "i0", "c0", weight=3, chunks=layout.pack(states[0]), deadline=deadline)
    protocol.submit("n1", "i1", "c1", weight=7, chunks=layout.pack(states[1]), deadline=deadline)


def test_owner_loss_replays_retained_chunks_then_commits_and_releases():
    protocol, layout, states, _ = fixture()
    submit_all(protocol, layout, states)
    before = protocol.placement
    retained = protocol.retained_bytes
    lost = before[0]
    protocol.lose_owner(lost, ("owner-c",), deadline=12)
    assert protocol.metrics.replay_bytes_sent == retained
    assert lost not in protocol.placement.values()
    committed = protocol.commit(("c0", "c1"), deadline=12)
    actual = layout.unpack(committed.chunks)
    for name in actual:
        expected = (states[0][name].double() * 3 + states[1][name].double() * 7) / 10
        assert torch.allclose(actual[name].double(), expected)
    assert protocol.retained_bytes == 0
    assert protocol.metrics.released_bytes == retained
    assert protocol.metrics.peak_retained_bytes <= protocol.max_retained_bytes
    assert protocol.metrics.peak_owner_bytes <= layout.shard_count * protocol.max_owner_bytes


def test_owner_loss_during_finalize_has_no_partial_publication_and_replays():
    protocol, layout, states, _ = fixture()
    submit_all(protocol, layout, states)
    with pytest.raises(TimeoutError, match="before atomic publication"):
        protocol.commit(("c0", "c1"), deadline=12, fail_after_shards=1)
    assert protocol.committed is None
    assert protocol.retained_bytes > 0
    committed = protocol.commit(("c0", "c1"), deadline=12)
    assert committed.generation == 5


def test_catch_up_validates_committed_checksums_and_rejects_duplicate_stale_corrupt():
    protocol, layout, states, _ = fixture()
    submit_all(protocol, layout, states)
    committed = protocol.commit(("c0", "c1"), deadline=12)
    assert protocol.catch_up("late", "new", local_generation=4,
                             committed=committed, deadline=12)
    assert not protocol.catch_up("late", "new", local_generation=4,
                                 committed=committed, deadline=12)
    with pytest.raises(ValueError, match="stale"):
        protocol.catch_up("future", "i", local_generation=6,
                          committed=committed, deadline=12)
    chunk = committed.chunks[0]
    corrupt = ShardChunk(chunk.layout_digest, chunk.shard_id, chunk.element_offset,
                         chunk.elements, b"x" + chunk.payload[1:], chunk.checksum_sha256)
    with pytest.raises(ValueError, match="checksum"):
        protocol.catch_up("bad", "i", local_generation=4, committed=committed,
                          deadline=12, chunks=(corrupt,) + committed.chunks[1:])
    with pytest.raises(ValueError, match="manifest checksum"):
        protocol.catch_up("manifest-bad", "i", local_generation=4,
                          committed=replace(committed, manifest_digest="0" * 64), deadline=12)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        protocol.catch_up("dup", "i", local_generation=4, committed=committed,
                          deadline=12, chunks=(committed.chunks[0],) * layout.shard_count)
    assert protocol.metrics.redistribution_bytes_sent == sum(c.nbytes for c in committed.chunks)


def test_ready_identity_backpressure_and_all_waits_are_bounded():
    protocol, layout, states, clock = fixture()
    with pytest.raises(ValueError, match="READY"):
        protocol.submit("n0", "stale-incarnation", "bad", weight=1,
                        chunks=layout.pack(states[0]), deadline=12)
    with pytest.raises(TimeoutError, match="bounded deadline"):
        protocol.submit("n0", "i0", "late", weight=1,
                        chunks=layout.pack(states[0]), deadline=20)
    tiny = ShardOwnerGeneration(layout, protocol.snapshot, ("owner",), run_id="run",
        generation=4, attempt=0, max_retained_bytes=1, max_owner_bytes=4096,
        deadlines=protocol.deadlines, clock=clock)
    with pytest.raises(BufferError, match="backpressure"):
        tiny.submit("n0", "i0", "c", weight=1, chunks=layout.pack(states[0]), deadline=12)
    submit_all(protocol, layout, states)
    committed = protocol.commit(("c0", "c1"), deadline=12)
    clock.now = 12
    with pytest.raises(TimeoutError, match="bounded deadline"):
        protocol.catch_up("late", "i", local_generation=4,
                          committed=committed, deadline=12)


def test_corrupt_submission_is_not_retained_or_partially_admitted():
    protocol, layout, states, _ = fixture()
    chunks = list(layout.pack(states[0]))
    chunk = chunks[-1]
    chunks[-1] = ShardChunk(chunk.layout_digest, chunk.shard_id, chunk.element_offset,
                            chunk.elements, chunk.payload, hashlib.sha256(b"wrong").hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        protocol.submit("n0", "i0", "c0", weight=1, chunks=chunks, deadline=12)
    assert protocol.retained_bytes == 0
    assert protocol.metrics.p2p_bytes_sent == 0
