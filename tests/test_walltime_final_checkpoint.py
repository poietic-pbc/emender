import math
import time
from argparse import Namespace
from pathlib import Path

import pytest
import torch

import train


def _args(**overrides):
    base = dict(
        disable_walltime_final_checkpoint=False,
        walltime_minutes=None,
        walltime_final_checkpoint_margin_seconds=60.0,
        walltime_check_every=1,
        level='E97',
        params='100m',
        mlp_ratio=1.0,
        gdn2_mlp_ratio=6208 / 2304,
    )
    base.update(overrides)
    return Namespace(**base)


def test_parse_slurm_timelimit_seconds():
    assert train.parse_slurm_timelimit_seconds('30') == 1800
    assert train.parse_slurm_timelimit_seconds('12:34') == 754
    assert train.parse_slurm_timelimit_seconds('01:02:03') == 3723
    assert train.parse_slurm_timelimit_seconds('2-01:02:03') == 176523
    assert train.parse_slurm_timelimit_seconds('UNLIMITED') is None
    assert train.parse_slurm_timelimit_seconds('bad') is None


def test_resolve_walltime_deadline_prefers_explicit_minutes(monkeypatch):
    monkeypatch.setattr(train, 'STARTUP_TIME', 1000.0)
    monkeypatch.setenv('SLURM_JOB_END_TIME', '999999')
    deadline, source = train.resolve_walltime_deadline(_args(walltime_minutes=2.5), now=1001.0)
    assert deadline == 1150.0
    assert source == '--walltime_minutes'


def test_resolve_walltime_deadline_uses_slurm_job_end(monkeypatch):
    monkeypatch.delenv('SLURM_TIMELIMIT', raising=False)
    monkeypatch.setenv('SLURM_JOB_END_TIME', '1234.5')
    deadline, source = train.resolve_walltime_deadline(_args(), now=1000.0)
    assert deadline == 1234.5
    assert source == 'SLURM_JOB_END_TIME'


def test_final_checkpoint_controller_triggers_inside_margin(monkeypatch):
    monkeypatch.setattr(train, 'STARTUP_TIME', time.time() - 100.0)
    args = _args(walltime_minutes=2.0, walltime_final_checkpoint_margin_seconds=30.0)
    controller = train.FinalCheckpointController(args, torch.device('cpu'))

    stop, reason, remaining = controller.maybe_request_stop(step=1, dist_enabled=False)

    assert stop is True
    assert reason == 'walltime:--walltime_minutes'
    assert remaining is not None
    assert remaining <= 30.0


def test_final_checkpoint_controller_respects_check_interval(monkeypatch):
    monkeypatch.setattr(train, 'STARTUP_TIME', time.time() - 100.0)
    args = _args(
        walltime_minutes=2.0,
        walltime_final_checkpoint_margin_seconds=30.0,
        walltime_check_every=4,
    )
    controller = train.FinalCheckpointController(args, torch.device('cpu'))

    stop, _, _ = controller.maybe_request_stop(step=2, dist_enabled=False)
    assert stop is False
    stop, reason, _ = controller.maybe_request_stop(step=4, dist_enabled=False)
    assert stop is True
    assert reason == 'walltime:--walltime_minutes'


def test_multinode_diloco_final_request_uses_non_collective_peer_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'STARTUP_TIME', time.time() - 100.0)

    class NoCollectiveDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(*_args, **_kwargs):
            raise AssertionError("final-checkpoint stop propagation must not use a NCCL collective")

    monkeypatch.setattr(train, 'dist', NoCollectiveDist)

    controllers = []
    for rank in range(16):
        args = _args(
            output=str(tmp_path),
            walltime_minutes=2.0 if rank == 0 else None,
            walltime_final_checkpoint_margin_seconds=30.0,
            walltime_check_every=1,
            _rank=rank,
            _world_size=16,
        )
        controllers.append(train.FinalCheckpointController(args, torch.device('cpu')))

    controllers[0].reset_coordination_files()

    stop, reason, remaining = controllers[0].maybe_request_stop(step=197, dist_enabled=True)
    assert stop is True
    assert reason == 'walltime:--walltime_minutes'
    assert remaining is not None

    for rank in range(1, 16):
        stop, reason, _ = controllers[rank].maybe_request_stop(step=197, dist_enabled=True)
        assert stop is True, f"rank {rank} missed peer final-checkpoint shutdown"
        assert reason == 'peer_final_checkpoint_request'

    controllers[0].mark_finalization_ready(step=197)
    with pytest.raises(TimeoutError):
        controllers[0].wait_for_all_finalization_ready(timeout_s=0.01, poll_s=0.001)

    for controller in controllers[1:]:
        controller.mark_finalization_ready(step=197)

    for controller in controllers:
        controller.wait_for_all_finalization_ready(timeout_s=0.5, poll_s=0.001)


def test_distributed_any_uses_collective_max(monkeypatch):
    class CollectiveMaxDist:
        class ReduceOp:
            MAX = object()

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op=None):
            assert op is CollectiveMaxDist.ReduceOp.MAX
            tensor.fill_(1)

    monkeypatch.setattr(train, 'dist', CollectiveMaxDist)

    assert train.distributed_any(False, torch.device('cpu'), dist_enabled=True) is True


def test_final_checkpoint_stop_consensus_promotes_peer_stop(tmp_path, monkeypatch):
    class PeerStopDist:
        class ReduceOp:
            MAX = object()

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op=None):
            tensor.fill_(1)

    monkeypatch.setattr(train, 'dist', PeerStopDist)

    controller = train.FinalCheckpointController(
        _args(output=str(tmp_path), _rank=8, _world_size=16),
        torch.device('cpu'),
    )

    stop, reason, remaining = train.consensus_final_checkpoint_stop(
        controller,
        local_stop=False,
        reason=None,
        remaining=None,
        device=torch.device('cpu'),
        dist_enabled=True,
    )

    assert stop is True
    assert reason == 'peer_final_checkpoint_request'
    assert remaining is None
    assert controller.triggered is True
    assert controller.reason == 'peer_final_checkpoint_request'


def test_save_checkpoint_metadata_and_latest_roundtrip(tmp_path: Path):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metadata = {
        'kind': 'final',
        'reason': 'walltime:SLURM_TIMELIMIT',
        'model_variant': 'level=E97,params=100m,mlp_ratio=1.0',
        'rank': 0,
        'world_size': 8,
        'is_head': True,
        'walltime_remaining_s': 42.5,
    }

    ckpt_path = train.save_checkpoint(
        model,
        optimizer,
        step=12,
        loss=math.pi,
        output_dir=tmp_path,
        keep_n=1,
        metadata=metadata,
        total_tokens=123456,
    )

    latest = tmp_path / 'latest.pt'
    assert latest.is_symlink()
    assert latest.resolve() == ckpt_path.resolve()

    loaded = torch.load(ckpt_path, map_location='cpu')
    assert loaded['checkpoint_metadata'] == metadata | {'total_tokens': 123456}
    assert loaded['total_tokens'] == 123456
    assert loaded['step'] == 12
    assert loaded['loss'] == math.pi
