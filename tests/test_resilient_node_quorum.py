import json
import multiprocessing
import time

import pytest

from ndm.resilient_node_quorum import (
    BoundedBucketStore, BucketUpdate, MetadataCoordinator, reassign_and_replay,
)


def _node(store, fence, node, values, weight=1):
    for bucket, value in enumerate(values):
        store.put(BucketUpdate.create(fence, node, bucket, weight, bytes(value)))


def test_missing_and_stuck_node_advances_exact_weighted_quorum(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "run", 1)
    fence = coordinator.fence(0)
    store = BoundedBucketStore(1024)
    _node(store, fence, "n0", ([10, 20], [30]), 1)
    _node(store, fence, "n1", ([20, 40], [60]), 3)
    # n2 never arrives: no collective, wait, or unanimity is involved.
    manifest = coordinator.commit(fence, store.generation(fence), quorum=2, expected_buckets=2)
    assert manifest["accepted_nodes"] == ["n0", "n1"]
    assert bytes.fromhex(manifest["buckets"]["0"]["payload_hex"]) == bytes([18, 35])
    assert json.loads((tmp_path / "latest.json").read_text())["generation"] == 0


def test_real_stuck_node_process_is_killed_while_quorum_continues(tmp_path):
    stuck = multiprocessing.Process(target=time.sleep, args=(60,))
    stuck.start()
    coordinator = MetadataCoordinator(tmp_path, "process-fault", 1)
    fence = coordinator.fence(0)
    store = BoundedBucketStore(32)
    _node(store, fence, "healthy-0", ([2],))
    _node(store, fence, "healthy-1", ([4],))
    stuck.terminate()
    stuck.join(timeout=5)
    assert not stuck.is_alive() and stuck.exitcode != 0
    committed = coordinator.commit(fence, store.generation(fence), quorum=2,
                                   expected_buckets=1)
    assert committed["accepted_nodes"] == ["healthy-0", "healthy-1"]
    assert bytes.fromhex(committed["buckets"]["0"]["payload_hex"]) == b"\x03"


def test_late_stale_attempt_rejected_and_quorum_loss_fails_closed(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "run", 1)
    current, stale = coordinator.fence(2, 1), coordinator.fence(2, 0)
    store = BoundedBucketStore(64)
    _node(store, stale, "late", ([99],))
    _node(store, current, "good", ([7],))
    with pytest.raises(TimeoutError, match="quorum lost"):
        coordinator.commit(current, store.generation(stale) + store.generation(current),
                           quorum=2, expected_buckets=1)


def test_shard_owner_failure_reassignment_replays_retained_buckets(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "run", 1)
    fence = coordinator.fence(0)
    sender0, sender1 = BoundedBucketStore(32), BoundedBucketStore(32)
    _node(sender0, fence, "n0", ([1], [2]))
    _node(sender1, fence, "n1", ([3], [4]))
    replacement = BoundedBucketStore(32)
    assert reassign_and_replay(fence, (sender0, sender1), replacement) == 4
    assert coordinator.commit(fence, replacement.generation(fence), quorum=2,
                              expected_buckets=2)["accepted_nodes"] == ["n0", "n1"]
    sender0.release(fence)
    assert sender0.bytes_used == 0


def test_coordinator_failover_fences_old_writer_and_restart_catches_up(tmp_path):
    old = MetadataCoordinator(tmp_path, "run", 1)
    old_fence = old.fence(0)
    new = MetadataCoordinator(tmp_path, "run", 2)
    with pytest.raises(RuntimeError, match="stale coordinator"):
        old.commit(old_fence, (), quorum=1, expected_buckets=1)
    fence = new.fence(0)
    store = BoundedBucketStore(8)
    _node(store, fence, "n0", ([42],))
    new.commit(fence, store.generation(fence), quorum=1, expected_buckets=1)
    assert new.catch_up(-1)["generation"] == 0
    assert new.catch_up(0) is None


def test_bucket_store_checksum_and_memory_bound(tmp_path):
    fence = MetadataCoordinator(tmp_path, "run", 1).fence(0)
    store = BoundedBucketStore(2)
    store.put(BucketUpdate.create(fence, "n0", 0, 1, b"ab"))
    with pytest.raises(BufferError, match="retention limit"):
        store.put(BucketUpdate.create(fence, "n0", 1, 1, b"c"))
    corrupt = BucketUpdate(fence, "n1", 0, 1, b"x", "bad")
    with pytest.raises(ValueError, match="checksum"):
        store.put(corrupt)
