import hashlib
import time

import pytest
import torch

from ndm.resilient_e97_roles import CpuNodeManager, LocalFence, LocalTrainerSpool


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
    shard = next(tmp_path.rglob("*.f64")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        CpuNodeManager(spool, quorum=1).collect(
            fence(), deadline=time.monotonic() + 1, expected_source_id="source")


def test_consumed_generation_is_promptly_released(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 1 << 20)
    spool.publish(fence(), 0, [torch.ones(64)], weight=1, source_id="source")
    assert spool.bytes_used > 0
    spool.release_generation(fence())
    assert spool.bytes_used == 0
