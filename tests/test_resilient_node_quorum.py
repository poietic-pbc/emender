import json
import multiprocessing
import time
import threading

import pytest

from ndm.resilient_node_quorum import (
    BoundedBucketStore, BucketUpdate, MetadataCoordinator, reassign_and_replay,
    ShardOwnerServer, send_bucket, supervise_until_quorum, Contribution,
    ContributionIdentity, GenerationAdmission, GenerationClosePolicy, GenerationFence,
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
    # A loaded Frontier login node can delay SIGTERM delivery beyond the
    # bounded grace period. Exercise the supervisor's fail-closed escalation
    # and never leave the synthetic 60-second peer alive after this test.
    if stuck.is_alive():
        stuck.kill()
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


def _stall_forever():
    time.sleep(60)


def test_network_owner_and_deadline_supervisor_advance_without_unanimity(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "network-run", 1)
    fence = coordinator.fence(0)
    owner_store = BoundedBucketStore(128)
    owner = ShardOwnerServer(("127.0.0.1", 0), owner_store, fence,
                             max_frame_bytes=4096)
    thread = threading.Thread(target=owner.serve, daemon=True)
    thread.start()
    retained = BoundedBucketStore(128)
    for node, value in (("n0", 10), ("n1", 30)):
        update = BucketUpdate.create(fence, node, 0, 1, bytes([value]))
        retained.put(update)
        send_bucket(owner.address, update, timeout=1, max_frame_bytes=4096)
    stuck = multiprocessing.Process(target=_stall_forever)
    stuck.start()
    accepted = supervise_until_quorum({"n2": stuck}, owner_store, fence, quorum=2,
                                      expected_buckets=1,
                                      deadline=time.monotonic() + .2)
    # Once quorum is frozen, the supervisor kills only the incomplete step.
    assert accepted == ("n0", "n1") and stuck.exitcode != 0
    manifest = coordinator.commit(fence, owner_store.generation(fence), quorum=2,
                                  expected_buckets=1)
    assert bytes.fromhex(manifest["buckets"]["0"]["payload_hex"]) == b"\x14"
    retained.release(fence)
    assert retained.bytes_used == 0
    owner.close()


def test_network_owner_rejects_stale_fence_and_oversize_frame(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "network-run", 1)
    current = coordinator.fence(1, 1)
    owner = ShardOwnerServer(("127.0.0.1", 0), BoundedBucketStore(16), current,
                             max_frame_bytes=512)
    threading.Thread(target=owner.serve, daemon=True).start()
    stale = BucketUpdate.create(coordinator.fence(1, 0), "late", 0, 1, b"x")
    with pytest.raises(RuntimeError, match="stale or late"):
        send_bucket(owner.address, stale, timeout=1, max_frame_bytes=512)
    huge = BucketUpdate.create(current, "n0", 0, 1, b"x" * 512)
    with pytest.raises(ValueError, match="configured maximum"):
        send_bucket(owner.address, huge, timeout=1, max_frame_bytes=512)
    owner.close()


def _contribution(admission, worker, incarnation, seq, tokens, payload=b"delta"):
    return Contribution.create(
        admission.fence, worker, incarnation, seq, tokens, payload,
        base_digest=admission.base_digest, policy_digest=admission.policy_digest,
        layout_digest=admission.layout_digest, code_digest=admission.code_digest,
    )


def test_contribution_identity_fencing_replay_conflict_stale_and_corrupt_rejection(tmp_path):
    coordinator = MetadataCoordinator(tmp_path, "identity-run", 3)
    admission = GenerationAdmission.open(
        coordinator.fence(7, 2), ready_snapshot=(("n0", "boot-a"),),
        policy=GenerationClosePolicy(q_min=1, t_min=10), deadline=20.0,
        base_digest="base", policy_digest="policy", layout_digest="layout",
        code_digest="code",
    )
    original = _contribution(admission, "n0", "boot-a", 4, 10)
    receipt = admission.admit(original, now=10.0)
    assert receipt.status == "accepted"
    assert admission.admit(original, now=11.0) == receipt

    conflict = _contribution(admission, "n0", "boot-a", 4, 11, b"other")
    assert admission.admit(conflict, now=12.0).status == "rejected_conflicting_duplicate"
    stale = _contribution(admission, "n0", "boot-a", 5, 10)
    stale = Contribution.create(
        coordinator.fence(6, 2), stale.identity.worker_id, stale.identity.incarnation,
        stale.identity.contribution_seq, stale.accepted_tokens, stale.payload,
        base_digest="base", policy_digest="policy", layout_digest="layout", code_digest="code",
    )
    assert admission.admit(stale, now=13.0).status == "rejected_stale_fence"
    corrupt_source = _contribution(admission, "n0", "boot-a", 6, 10)
    corrupt = Contribution(corrupt_source.identity, original.accepted_tokens, original.payload,
                           "bad", original.base_digest, original.policy_digest,
                           original.layout_digest, original.code_digest)
    assert admission.admit(corrupt, now=14.0).status == "rejected_corrupt"
    wrong_incarnation = _contribution(admission, "n0", "boot-old", 9, 10)
    assert admission.admit(wrong_incarnation, now=15.0).status == "rejected_not_ready"


def test_token_floor_ready_snapshot_fraction_and_deterministic_freeze():
    policy = GenerationClosePolicy(q_min=1, t_min=25, ready_fraction=.75)
    admission = GenerationAdmission.open(
        GenerationFence("run", 4, 0, 8),
        ready_snapshot=(("n3", "i3"), ("n1", "i1"), ("n2", "i2"), ("n0", "i0")),
        policy=policy, deadline=100.0, base_digest="b", policy_digest="p",
        layout_digest="l", code_digest="c",
    )
    # The fraction is computed once from the four-member READY snapshot: ceil(3).
    assert admission.required_contributions == 3
    for worker, incarnation, seq, tokens in (
        ("n2", "i2", 2, 10), ("n0", "i0", 0, 10), ("n1", "i1", 1, 10)
    ):
        admission.admit(_contribution(admission, worker, incarnation, seq, tokens), now=50.0)
    close = admission.close(now=50.0, run_deadline=200.0)
    assert close.status == "commit_ready"
    assert close.accepted_tokens == 30
    assert [identity.worker_id for identity in close.frozen_identities] == ["n0", "n1", "n2"]
    assert close.ready_snapshot == (("n0", "i0"), ("n1", "i1"), ("n2", "i2"), ("n3", "i3"))
    late = admission.admit(_contribution(admission, "n3", "i3", 3, 100), now=51.0)
    assert late.status == "rejected_stale_fence"
    assert admission.close(now=51.0, run_deadline=200.0) == close


def test_quorum_collapse_deadline_defers_then_aborts_without_commit(tmp_path):
    admission = GenerationAdmission.open(
        GenerationFence("run", 9, 1, 2), ready_snapshot=(("n0", "i0"), ("n1", "i1")),
        policy=GenerationClosePolicy(q_min=2, t_min=20, ready_fraction=1.0),
        deadline=10.0, base_digest="b", policy_digest="p", layout_digest="l", code_digest="c",
        evidence_path=tmp_path / "close-evidence.jsonl",
    )
    admission.admit(_contribution(admission, "n0", "i0", 0, 10), now=5.0)
    deferred = admission.close(now=10.0, run_deadline=20.0)
    assert deferred.status == "deferred" and deferred.frozen_identities == ()
    assert not (tmp_path / "latest.json").exists()
    aborted = admission.close(now=20.0, run_deadline=20.0)
    assert aborted.status == "aborted" and aborted.frozen_identities == ()
    evidence = [json.loads(line) for line in (tmp_path / "close-evidence.jsonl").read_text().splitlines()]
    assert [item["status"] for item in evidence] == ["deferred", "aborted"]
    assert all(item["reason"] == "generation_deadline_floor_unavailable" for item in evidence)
