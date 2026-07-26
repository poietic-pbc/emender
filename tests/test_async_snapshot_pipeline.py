from __future__ import annotations

from pathlib import Path
from queue import Full, Queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import ndm.async_diloco_real as real
from ndm.async_diloco_real import (
    PersistentAsyncTrainingLane,
    PersistentRealWorkerSession,
    RealAsyncWorkerSpec,
)
from ndm.async_diloco_v2 import AtomicEightTrainerApply, Backpressure


class _AdvancingSession:
    """Small model-owner fixture for foreground/background boundary tests."""

    def __init__(self) -> None:
        self.completed = 0
        self.value = torch.zeros(1)

    def run_window(self, local_window, **_unused):
        self.value.add_(1)
        self.completed += 1
        return SimpleNamespace(
            generation=local_window,
            tokens=1,
            losses=(1.0,),
        )

    def translate(self, corrections):
        self.value.add_(corrections["weight"])

    def snapshot(self):
        return {"weight": self.value.clone()}


def _start_advancing_lane(session: _AdvancingSession) -> PersistentAsyncTrainingLane:
    lane = PersistentAsyncTrainingLane(session, max_windows=2)
    lane.start(
        local_window_start=0,
        start_state={"weight": session.value.clone()},
        admission_deadline=time.monotonic() + 1.0,
    )
    return lane


def _wait_for_window(session: _AdvancingSession) -> None:
    deadline = time.monotonic() + 1.0
    while session.completed < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.completed >= 1


def test_snapshot_capture_is_coherent_and_background_never_reads_live_state(
    monkeypatch,
):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    class ScheduleFreeFixture:
        def __init__(self, parameters):
            parameter = tuple(parameters)[0]
            self.param_groups = [{"params": [parameter], "lr": 0.1}]
            self.state = {
                parameter: {
                    "z": parameter.detach().clone(),
                    "exp_avg_sq": torch.zeros(1),
                },
            }

    monkeypatch.setattr(
        real.train, "build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr(
        real.train,
        "build_training_optimizer",
        lambda model, _args: ScheduleFreeFixture(model.parameters()),
    )
    monkeypatch.setattr(
        real, "_build_batch_iter", lambda *_args, **_kwargs: object())
    session = PersistentRealWorkerSession(
        base_state={"weight": torch.tensor([1.0])},
        train_args=real.Namespace(seed=7, bf16=False, lr=0.1),
        spec=RealAsyncWorkerSpec("trainer", "node-0", "cpu", 1, 0),
        synthetic_token_stream=False,
        synthetic_vocab_size=8,
    )

    admitted = session.snapshot()
    admitted_bytes = admitted["weight"].numpy().tobytes()
    with torch.no_grad():
        session.model.weight.fill_(9.0)
    # The background-facing object is the immutable mapping, never the model.
    background_root = admitted["weight"].numpy().tobytes()

    assert admitted_bytes == background_root
    torch.testing.assert_close(admitted["weight"], torch.tensor([1.0]))
    torch.testing.assert_close(session.model.weight, torch.tensor([9.0]))
    assert session.snapshot_slot_count == 2
    session.close()


def test_snapshot_admission_deadline_excludes_telemetry_io():
    """Persist causal records only after the next mutable K lane owns state."""
    source = (
        Path(__file__).parents[1]
        / "scripts/frontier/resilient_e97_role.py"
    ).read_text(encoding="utf-8")
    trainer = source[source.index("def trainer(args) -> int:"):]
    snapshot = trainer.index(
        "retained_endpoint = persistent_worker.snapshot()")
    snapshot_completed = trainer.index(
        "endpoint_snapshot_completed = time.monotonic()", snapshot)
    lane_start = trainer.index("async_training_lane.start(", snapshot_completed)
    lane_completed = trainer.index(
        "lane_admission_completed = time.monotonic()", lane_start)
    snapshot_telemetry = trainer.index(
        '"async_v21_endpoint_snapshot"', lane_completed)
    admission_telemetry = trainer.index(
        '"async_v21_snapshot_admission"', snapshot_telemetry)

    assert (
        snapshot < snapshot_completed < lane_start < lane_completed
        < snapshot_telemetry < admission_telemetry
    )
    pre_owned = trainer[snapshot_completed:lane_start]
    assert '"async_v21_endpoint_snapshot"' not in pre_owned
    assert "lookahead_anchor_digest = state_digest(" not in pre_owned
    assert (
        "ended=endpoint_snapshot_completed"
        in trainer[snapshot_telemetry:admission_telemetry]
    )
    assert (
        "ended=lane_admission_completed"
        in trainer[admission_telemetry:admission_telemetry + 1200]
    )


@pytest.mark.parametrize(
    "blocked_phase",
    [
        "publish_network",
        "aggregation",
        "checkpoint",
        "result_wait",
    ],
)
def test_admitted_snapshot_resumes_next_k_before_every_background_phase(
    blocked_phase,
):
    background_release = threading.Event()
    background_started = threading.Event()

    def blocked_background() -> None:
        background_started.set()
        assert background_release.wait(2.0), blocked_phase

    worker = threading.Thread(
        target=blocked_background, name=f"blocked-{blocked_phase}")
    worker.start()
    assert background_started.wait(1.0)
    session = _AdvancingSession()
    admission_started = time.monotonic()
    lane = _start_advancing_lane(session)
    admission_elapsed = time.monotonic() - admission_started

    _wait_for_window(session)
    assert worker.is_alive()
    assert admission_elapsed <= 1.0

    lane.finish_at_boundary(
        deadline=time.monotonic() + 1.0,
        corrections={"weight": torch.zeros(1)},
    )
    background_release.set()
    worker.join(1.0)
    assert not worker.is_alive()


def test_background_pipeline_uses_only_immutable_snapshots():
    live = torch.tensor([1.0])
    immutable = live.clone()
    background_started = threading.Event()
    background_release = threading.Event()
    observed: list[bytes] = []

    def checkpoint_background(snapshot: torch.Tensor) -> None:
        background_started.set()
        assert background_release.wait(1.0)
        observed.append(snapshot.numpy().tobytes())

    worker = threading.Thread(
        target=checkpoint_background, args=(immutable,))
    worker.start()
    assert background_started.wait(1.0)
    live.add_(8.0)
    background_release.set()
    worker.join(1.0)

    assert observed == [torch.tensor([1.0]).numpy().tobytes()]
    torch.testing.assert_close(live, torch.tensor([9.0]))


@pytest.mark.parametrize(
    "capacity_edge",
    [
        "snapshot_slot",
        "mailbox_view",
        "mailbox_staging",
        "native_credit",
        "replay",
        "receipt",
    ],
)
def test_snapshot_and_mailbox_capacity_never_blocks_foreground(capacity_edge):
    bounded = Queue(maxsize=1)
    bounded.put_nowait(object())
    with pytest.raises(Full):
        bounded.put_nowait(capacity_edge)

    session = _AdvancingSession()
    lane = _start_advancing_lane(session)
    _wait_for_window(session)
    report = lane.finish_at_boundary(
        deadline=time.monotonic() + 1.0,
        corrections={"weight": torch.zeros(1)},
    )

    assert session.completed >= 1
    assert bounded.qsize() == 1
    assert report.elapsed_s < 1.0


def test_result_apply_is_atomic_bounded_and_nonblocking(tmp_path):
    transaction = AtomicEightTrainerApply(
        root=tmp_path,
        run_id="run",
        fence=7,
        node_id="node-0",
        node_incarnation="node-incarnation",
        result_version=1,
        result_digest="a" * 64,
    )
    started = time.monotonic()
    for rank in range(7):
        transaction.record_trainer(
            rank=rank,
            trainer_incarnation=f"trainer-{rank}",
            recovery_digest=f"{rank + 1:064x}",
        )
    with pytest.raises(Backpressure, match="all eight"):
        transaction.commit_node()
    assert not transaction.ready
    assert not transaction.node_marker_path.exists()

    transaction.record_trainer(
        rank=7,
        trainer_incarnation="trainer-7",
        recovery_digest=f"{8:064x}",
    )
    marker = transaction.commit_node()
    elapsed = time.monotonic() - started

    assert elapsed < 60.0
    assert transaction.ready
    assert len(marker["trainers"]) == 8
    assert transaction.node_marker_path.exists()
    assert transaction.commit_node() == marker
