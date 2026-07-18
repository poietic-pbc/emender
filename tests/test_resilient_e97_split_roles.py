import hashlib
import time

import pytest
import torch

from ndm.resilient_e97_roles import CpuNodeManager, LocalFence, LocalTrainerSpool
from ndm.resilient_e97_runtime import apply_delta


def fence():
    return LocalFence("run", 10, 0, 3, "payload-code-seed")


def test_eight_real_trainers_publish_and_cpu_manager_exactly_weights(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    for trainer in range(8):
        spool.publish(fence(), trainer, [torch.tensor([trainer, trainer + 1.])],
                      weight=trainer + 1, source_id="source")
    members, weight, shards = CpuNodeManager(spool, quorum=6).collect(
        fence(), deadline=time.monotonic() + 1, expected_source_id="source")
    assert members == tuple(range(6))
    assert weight == 21
    assert torch.equal(shards[0], torch.tensor([70 / 21, 91 / 21], dtype=torch.float64))


def test_failed_trainers_are_evicted_by_dynamic_quorum(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    for trainer in (0, 1, 2, 4, 6, 7):
        spool.publish(fence(), trainer, [torch.tensor([float(trainer)])],
                      weight=1, source_id="source")
    members, _, _ = CpuNodeManager(spool, quorum=6).collect(
        fence(), deadline=time.monotonic() + 1, expected_source_id="source")
    assert members == (0, 1, 2, 4, 6, 7)


def test_manager_rejects_corrupt_stale_nonfinite_duplicate_and_overflow(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1000)
    spool.publish(fence(), 0, [torch.tensor([1.])], weight=1, source_id="source")
    with pytest.raises(ValueError, match="duplicate"):
        spool.publish(fence(), 0, [torch.tensor([1.])], weight=1, source_id="source")
    with pytest.raises(ValueError, match="finite"):
        spool.publish(fence(), 1, [torch.tensor([float("nan")])], weight=1, source_id="source")
    shard = next(tmp_path.rglob("*.data")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        CpuNodeManager(spool, quorum=1).collect(
            fence(), deadline=time.monotonic() + 1, expected_source_id="source")


def test_consumed_generation_is_promptly_released(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    spool.publish(fence(), 0, [torch.ones(64)], weight=1, source_id="source")
    assert spool.bytes_used > 0
    spool.release_generation(fence())
    assert spool.bytes_used == 0


def test_large_stream_uses_two_files_per_trainer_not_one_per_chunk(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 8 << 20)
    chunks = (torch.full((4096,), float(index)) for index in range(128))
    manifest = spool.publish(fence(), 0, chunks, weight=7, source_id="source")
    assert sorted(path.name for path in manifest.parent.iterdir()) == [
        "contribution.data", "manifest.json"]
    assert spool.file_count == 2
    assert spool.files_published == 2
    assert spool.bytes_written == 128 * 4096 * 4


def test_global_aggregate_projects_once_to_model_dtype_and_streams_apply(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    exact = torch.tensor([1 / 3, -7 / 11, 5 / 17], dtype=torch.float64)
    path = spool.publish_aggregate(
        fence(), [0, 1], [exact], weight=19, source_id="source",
        accepted_peers=("node-0", "node-1"), storage_dtype=torch.float32)
    manifest = __import__("json").loads(path.read_text())
    assert manifest["data_bytes"] == exact.numel() * 4
    assert manifest["shards"][0]["dtype"] == "float32"

    loaded_manifest, stream = spool.stream_aggregate(
        fence(), deadline=time.monotonic() + 1, expected_source_id="source")
    base = {"weight": torch.ones(3, dtype=torch.float32)}
    pointer = base["weight"].data_ptr()
    actual = apply_delta(base, stream, eta_outer=.25, in_place=True)
    expected = torch.ones(3, dtype=torch.float32)
    expected.add_(exact.to(torch.float32), alpha=.25)
    assert loaded_manifest == manifest
    assert actual["weight"].data_ptr() == pointer
    assert torch.equal(actual["weight"], expected)


def test_same_fence_aggregate_is_idempotent_only_for_identical_payload(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    first = spool.publish_aggregate(
        fence(), [0], [torch.tensor([1.0])], weight=7, source_id="source")
    assert spool.publish_aggregate(
        fence(), [0], [torch.tensor([1.0])], weight=7, source_id="source") == first
    with pytest.raises(ValueError, match="conflicting aggregate"):
        spool.publish_aggregate(
            fence(), [0], [torch.tensor([2.0])], weight=7, source_id="source")


def test_byte_limit_is_shared_across_role_process_spool_instances(tmp_path):
    trainer_spool = LocalTrainerSpool(tmp_path, 1700)
    manager_spool = LocalTrainerSpool(tmp_path, 1700)
    trainer_spool.publish(
        fence(), 0, [torch.ones(256)], weight=1, source_id="source")
    occupied = trainer_spool.bytes_used
    assert 1024 < occupied <= 1700
    assert manager_spool.bytes_used == occupied

    with pytest.raises(BufferError, match="byte limit"):
        manager_spool.publish(
            fence(), 1, [torch.ones(256)], weight=1, source_id="source")
    assert trainer_spool.bytes_used == occupied

    trainer_spool.release_generation(fence())
    assert manager_spool.bytes_used == 0
    manager_spool.publish(
        fence(), 1, [torch.ones(256)], weight=1, source_id="source")
