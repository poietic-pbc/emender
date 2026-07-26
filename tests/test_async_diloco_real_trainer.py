import json
import inspect
import math
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoCheckpointCadence
from ndm.async_diloco import AsyncDiLoCoGenerationMetrics, AsyncDiLoCoUpdate
from ndm.async_diloco_compiled_mpich import COMPILED_MPICH_TRANSPORT
from ndm.async_diloco_real import (
    PersistentAsyncTrainingLane,
    PersistentRealWorkerSession,
    RealAsyncDiLoCoConfig,
    RealAsyncFileRankConfig,
    RealAsyncNodeResult,
    RealAsyncWorkerSpec,
    RealAsyncWorkerReport,
    default_tiny_e97_train_args,
    run_real_async_diloco,
    run_real_async_diloco_file_rank,
    _release_consumed_optimizer_state,
    _run_real_worker,
)


def _args(**overrides):
    values = {
        "seed": 123,
        "batch_size": 2,
        "chunk_size": 8,
        "steps": 8,
        "lr": 1e-3,
    }
    values.update(overrides)
    return default_tiny_e97_train_args(**values)


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_consumed_checkpoint_optimizer_state_releases_mutable_mapping():
    tensor = torch.ones(4)
    checkpoint_state = {
        "state": {0: {"exp_avg_sq": tensor}},
        "param_groups": [{"params": [0]}],
    }

    _release_consumed_optimizer_state(checkpoint_state)

    assert checkpoint_state == {}
    assert torch.equal(tensor, torch.ones(4))


def test_consumed_checkpoint_optimizer_state_keeps_immutable_mapping_compatible():
    from types import MappingProxyType

    checkpoint_state = MappingProxyType({"state": {}})
    _release_consumed_optimizer_state(checkpoint_state)
    assert checkpoint_state["state"] == {}


def test_real_async_trainer_one_node_reduces_real_worker_updates(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-one-node",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=tuple(
            RealAsyncWorkerSpec(
                worker_id=f"worker-{idx}",
                node_id="node-0",
                local_steps=1,
                seed_offset=idx,
            )
            for idx in range(2)
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    assert result.latest_generation == 0
    assert (tmp_path / "run" / "latest.json").exists()
    assert len(result.node_results) == 1
    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.metrics.accepted_updates == 2
    assert global_generation.metrics.quorum_status == "advanced"
    assert global_generation.metrics.accepted_updates == 1
    assert global_generation.metrics.latest_advanced is True
    assert node.metrics.tokens_per_generation == 2 * 2 * 9
    assert node.metrics.tokens_per_sec > 0.0
    assert global_generation.metrics.tokens_per_sec > 0.0
    assert node.metrics.update_bytes["worker"] > 0
    assert global_generation.metrics.update_bytes["node"] > 0

    losses = [
        report.losses[0]
        for report in node.worker_reports
        if report.losses
    ]
    assert len(losses) == 2
    assert all(math.isfinite(loss) and loss > 1.0 for loss in losses)
    assert node.metrics.loss_moving_average["loss"] == pytest.approx(sum(losses) / 2)
    assert node.metrics.loss_moving_average["loss"] != pytest.approx(0.99)

    payload = json.loads(Path(result.metrics_json).read_text(encoding="utf-8"))
    assert payload["metrics_summary"]["quorum_status"]["advanced"] == 2
    assert payload["metrics_summary"]["totals"]["accepted_updates"] == 3
    assert payload["global_generations"][0]["metrics"]["checkpoint_paths"]


def test_real_async_trainer_local_quorum_defers_without_process_failure(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-local-defer",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="failed", fail_before_submit=True),
            RealAsyncWorkerSpec(worker_id="timed-out", timed_out=True),
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.node_update is None
    assert node.metrics.quorum_status == "deferred"
    assert node.metrics.failed_updates == 1
    assert node.metrics.timed_out_updates == 1
    assert global_generation.metrics.quorum_status == "deferred"
    assert global_generation.metrics.latest_advanced is False
    assert not (tmp_path / "run" / "latest.json").exists()
    assert Path(result.metrics_json).exists()


def test_real_async_trainer_global_quorum_defers_without_process_failure(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-global-defer",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
            RealAsyncWorkerSpec(worker_id="worker-1", local_steps=1, seed_offset=1),
        ),
        local_quorum=2,
        global_quorum=2,
        global_node_count=2,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.node_update is not None
    assert global_generation.metrics.quorum_status == "deferred"
    assert global_generation.metrics.accepted_updates == 1
    assert global_generation.metrics.timed_out_updates == 1
    assert global_generation.metrics.latest_advanced is False
    assert not (tmp_path / "run" / "latest.json").exists()


def test_real_async_trainer_rejects_stale_worker_but_collects_fresh_quorum(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-stale-accounting",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="stale", stale_generation=-1),
            RealAsyncWorkerSpec(worker_id="fresh-0", seed_offset=1),
            RealAsyncWorkerSpec(worker_id="fresh-1", seed_offset=2),
        ),
        local_quorum=2,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    node = result.node_results[0]
    global_generation = result.generations[0]
    assert node.metrics.quorum_status == "advanced"
    assert node.metrics.stale_updates == 1
    assert node.metrics.accepted_updates == 2
    assert node.node_update is not None
    assert global_generation.metrics.quorum_status == "advanced"
    assert global_generation.metrics.latest_advanced is True


def test_real_async_trainer_cli_smoke_runs_one_generation(tmp_path):
    metrics_json = tmp_path / "metrics.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/frontier/e97_async_diloco_train.py",
            "--run-id",
            "cli-smoke",
            "--run-dir",
            str(tmp_path / "run"),
            "--metrics-json",
            str(metrics_json),
            "--synthetic-token-stream",
            "--worker-count",
            "1",
            "--local-quorum",
            "1",
            "--global-quorum",
            "1",
            "--generations",
            "1",
            "--local-steps",
            "1",
            "--dim",
            "8",
            "--depth",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    stdout = json.loads(completed.stdout)
    assert stdout["latest_generation"] == 0
    assert metrics_json.exists()
    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert payload["global_generations"][0]["metrics"]["latest_advanced"] is True
    assert payload["node_generations"][0]["metrics"]["loss_moving_average"]["loss"] > 1.0


def test_compiled_mpich_file_rank_runs_two_generations_in_one_process(tmp_path, monkeypatch):
    """The production entrypoint must perform two K40 merges without reloading its seed."""
    load_calls = []
    generations = []
    merges = []

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.load_checkpoint",
        lambda *args, **kwargs: load_calls.append(args[0]) or (None, None, {"step": 10, "optimizer_state_dict": {}}),
    )

    def fake_supervisor(**kwargs):
        generation = kwargs["generation"]
        spec = kwargs["worker_specs"][0]
        generations.append((generation, spec.local_steps, kwargs["optimizer_state_dict"]))
        update = AsyncDiLoCoUpdate(
            worker_id=spec.worker_id,
            base_generation=generation,
            delta={"weight": torch.ones(1)},
            tokens=spec.local_steps,
            local_steps=spec.local_steps,
            loss_moving_average={"loss": 1.0, "loss_100": 1.0},
        )
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=generation,
            update=update,
            elapsed_s=0.1,
            tokens=spec.local_steps,
            optimizer_state_dict={"generation": generation},
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"], generation=generation, requested_workers=1,
            participating_workers=1, quorum_threshold=1, quorum_size=1,
            accepted_updates=1, stale_updates=0, timed_out_updates=0,
            failed_updates=0, invalid_updates=0, generation_duration_s=0.1,
            merge_duration_s=0.0, rebase_duration_s=0.0, checkpoint_duration_s=0.0,
            tokens_per_sec=400.0, tokens_per_generation=spec.local_steps,
            loss_moving_average={"loss": 1.0, "loss_100": 1.0},
        )
        return RealAsyncNodeResult(kwargs["node_id"], generation, update, (worker,), metrics)

    def fake_coordinate(**kwargs):
        generation = kwargs["generation"]
        merges.append(generation)
        state = {"weight": kwargs["base_state"]["weight"] + 1}
        return {
            "latest_generation": generation,
            "global_generations": [{"generation": generation, "metrics": {"latest_advanced": True}}],
            "_private_global_state": state,
        }

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_supervisor)
    monkeypatch.setattr("ndm.async_diloco_real._coordinate_compiled_mpich_dense_rank", fake_coordinate)
    seed = tmp_path / "seed.pt"
    seed.write_bytes(b"seed")
    result = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
        run_id="two-generations", run_dir=tmp_path / "run", metrics_json=tmp_path / "metrics.json",
        train_args=_args(steps=80), node_rank=0, node_count=1, global_quorum=1,
        local_steps=40, generations=2, initial_checkpoint=seed,
        transport=COMPILED_MPICH_TRANSPORT, compiled_mpich_helper_bin=tmp_path / "helper",
    ))

    assert load_calls == [str(seed)]
    assert [item[0] for item in generations] == [0, 1]
    assert [item[1] for item in generations] == [40, 40]
    assert generations[1][2] == {"generation": 0}
    assert merges == [0, 1]
    assert sum(item[1] for item in generations) == 80
    assert result["global_result"]["latest_generation"] == 1


def test_file_rank_generation_loop_is_lazy_and_memory_bounded():
    source = inspect.getsource(run_real_async_diloco_file_rank)
    assert "for generation in range(config.generations):" in source
    assert "list(range(config.generations))" not in source
    assert "generation_payloads" not in source


def test_real_async_trainer_accepts_initial_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed.pt"
    seed_args = _args(seed=999)
    model = __import__("train").build_training_model(seed_args)
    torch.save({"model_state_dict": model.state_dict(), "step": 1065000}, checkpoint)

    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-checkpoint-seed",
        run_dir=tmp_path / "run",
        train_args=seed_args,
        initial_checkpoint=checkpoint,
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        ),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))

    assert result.latest_generation == 0
    assert Path(result.metrics_json).exists()


def test_real_async_two_hop_chain_uses_run_local_latest_pt_with_optimizer_state(tmp_path):
    train_mod = __import__("train")
    seed_args = _args(seed=999, steps=1, optimizer="schedulefree", weight_decay=0.01, warmup_steps=0)
    train_mod.normalize_training_args(seed_args)
    model = train_mod.build_training_model(seed_args)
    optimizer = train_mod.build_training_optimizer(model, seed_args)
    seed_checkpoint = tmp_path / "seed" / "checkpoint_step_001000_loss_2.5000.pt"
    seed_checkpoint.parent.mkdir()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 1000,
            "loss": 2.5,
            "checkpoint_metadata": {
                "tokenizer": "p50k_base",
                "model": getattr(seed_args, "_model_metadata", None),
            },
        },
        seed_checkpoint,
    )

    hop_a = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="chain-hop-a",
        run_dir=tmp_path / "hop-a",
        train_args=seed_args,
        initial_checkpoint=seed_checkpoint,
        worker_specs=(RealAsyncWorkerSpec(worker_id="worker-a", local_steps=1),),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))
    hop_a_latest_pt = tmp_path / "hop-a" / "latest.pt"
    assert hop_a.latest_generation == 0
    assert hop_a_latest_pt.is_symlink()
    hop_a_checkpoint = hop_a_latest_pt.resolve()
    hop_a_payload = torch.load(hop_a_checkpoint, map_location="cpu")
    assert hop_a_payload["step"] == 1001
    assert "optimizer_state_dict" in hop_a_payload
    assert hop_a_payload["checkpoint_metadata"]["source_checkpoint"] == str(seed_checkpoint)
    reload_model = train_mod.build_training_model(seed_args)
    reload_optimizer = train_mod.build_training_optimizer(reload_model, seed_args)
    train_mod.load_checkpoint(hop_a_latest_pt, reload_model, reload_optimizer)
    before_train = {
        name: tensor.detach().clone()
        for name, tensor in reload_model.state_dict().items()
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
    }
    reload_optimizer.train()
    after_train = {
        name: tensor.detach()
        for name, tensor in reload_model.state_dict().items()
        if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
    }
    assert all(torch.equal(before_train[name], after_train[name]) for name in before_train)

    hop_b = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="chain-hop-b",
        run_dir=tmp_path / "hop-b",
        train_args=seed_args,
        initial_checkpoint=hop_a_latest_pt,
        worker_specs=(RealAsyncWorkerSpec(worker_id="worker-b", local_steps=1),),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
    ))
    hop_b_latest_pt = tmp_path / "hop-b" / "latest.pt"
    hop_b_payload = torch.load(hop_b_latest_pt.resolve(), map_location="cpu")

    assert hop_b.latest_generation == 0
    assert hop_b_payload["step"] == 1002
    assert "optimizer_state_dict" in hop_b_payload
    assert hop_b_payload["checkpoint_metadata"]["source_checkpoint"] == str(hop_a_latest_pt)
    assert hop_b_payload["checkpoint_metadata"]["source_checkpoint"] != str(seed_checkpoint)
    assert json.loads((tmp_path / "hop-b" / "latest.json").read_text(encoding="utf-8"))[
        "model_checkpoint_path"
    ] == str(hop_b_latest_pt.resolve())


def test_real_async_worker_converts_model_to_bf16_before_training(monkeypatch):
    observed = {}
    progress = []

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyOptimizer:
        pass

    args = _args(bf16=True)

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_model",
        lambda _args: OneParamModel(),
    )
    monkeypatch.setattr(
        "ndm.async_diloco_real.train.build_training_optimizer",
        lambda model, _args: DummyOptimizer(),
    )

    def fake_train_one_optimizer_step(model, optimizer, _args, **kwargs):
        del optimizer, kwargs
        observed["dtype"] = next(model.parameters()).dtype
        with torch.no_grad():
            model.weight.add_(1.0)
        return {"loss": 2.0, "tokens_processed": 9, "hidden_state": None}

    monkeypatch.setattr(
        "ndm.async_diloco_real.train.train_one_optimizer_step",
        fake_train_one_optimizer_step,
    )

    report = _run_real_worker(
        run_id="bf16-worker",
        generation=0,
        base_state={"weight": torch.zeros(1)},
        train_args=args,
        spec=RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        synthetic_token_stream=True,
        synthetic_vocab_size=8,
        progress_callback=lambda step, metrics: progress.append((step, metrics["loss"])),
    )

    assert observed["dtype"] is torch.bfloat16
    assert report.update is not None
    assert report.update.delta["weight"].dtype is torch.float32
    assert report.losses == (2.0,)
    assert progress == [(1, 2.0)]


def test_persistent_worker_bootstraps_once_across_multiple_exact_windows(monkeypatch):
    import ndm.async_diloco_real as real

    counts = {"model": 0, "optimizer": 0, "iterator": 0}
    hidden_seen = []

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class ScheduleFreeFixture:
        def __init__(self, parameters):
            parameter = tuple(parameters)[0]
            self.param_groups = [{"params": [parameter], "lr": 0.1}]
            self.state = {
                parameter: {
                    "z": parameter.detach().clone(),
                    "exp_avg_sq": torch.tensor([9.0]),
                    "step": 0,
                },
            }

    def build_model(_args):
        counts["model"] += 1
        return OneParamModel()

    def build_optimizer(model, _args):
        counts["optimizer"] += 1
        return ScheduleFreeFixture(model.parameters())

    iterator = object()

    def build_iterator(*_args, **_kwargs):
        counts["iterator"] += 1
        return iterator

    def train_step(model, optimizer, _args, **kwargs):
        assert kwargs["batch_iter"] is iterator
        hidden_seen.append(kwargs["hidden_state"])
        with torch.no_grad():
            model.weight.add_(1.0)
        optimizer.state[model.weight]["step"] += 1
        return {
            "loss": 1.0,
            "tokens_processed": 3,
            "hidden_state": len(hidden_seen),
        }

    monkeypatch.setattr(real.train, "build_training_model", build_model)
    monkeypatch.setattr(real.train, "build_training_optimizer", build_optimizer)
    monkeypatch.setattr(real, "_build_batch_iter", build_iterator)
    monkeypatch.setattr(real.train, "train_one_optimizer_step", train_step)
    phases = []
    session = PersistentRealWorkerSession(
        base_state={"weight": torch.zeros(1)},
        train_args=real.Namespace(seed=7, bf16=False, lr=0.1),
        spec=RealAsyncWorkerSpec("trainer", "node-0", "cpu", 2, 0),
        synthetic_token_stream=False,
        synthetic_vocab_size=8,
        bootstrap_phase_callback=lambda phase, _details: phases.append(phase),
    )
    model_identity = id(session.model)
    optimizer_identity = id(session.optimizer)
    first = session.run_window(0)
    second = session.run_window(1)

    assert first.tokens == second.tokens == 6
    assert session.windows_completed == 2
    assert id(session.model) == model_identity
    assert id(session.optimizer) == optimizer_identity
    assert counts == {"model": 1, "optimizer": 1, "iterator": 1}
    assert session.bootstrap_counts == {
        "model_build": 1,
        "optimizer_build": 1,
        "data_iterator_build": 1,
    }
    assert phases.count("model_build_start") == 1
    assert phases.count("optimizer_built") == 1
    assert phases.count("data_iterator_ready") == 1
    assert hidden_seen == [None, 1, 2, 3]

    moment = session.optimizer.state[session.model.weight][
        "exp_avg_sq"].clone()
    session.translate({"weight": torch.tensor([5.0])})
    torch.testing.assert_close(
        session.model.weight.detach(), torch.tensor([9.0]))
    torch.testing.assert_close(
        session.optimizer.state[session.model.weight]["z"],
        torch.tensor([5.0]))
    torch.testing.assert_close(
        session.optimizer.state[session.model.weight]["exp_avg_sq"], moment)
    session.close()


def test_persistent_real_worker_materializes_lazy_schedulefree_z(monkeypatch):
    """Sparse first-window state still permits full-model x/z translation."""
    import ndm.async_diloco_real as real

    class TwoParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.active = torch.nn.Parameter(torch.tensor([1.0]))
            self.sparse = torch.nn.Parameter(torch.tensor([2.0]))

    class LazyScheduleFreeFixture:
        def __init__(self, parameters):
            active, sparse = tuple(parameters)
            self.param_groups = [{
                "params": [active, sparse], "lr": 0.1, "train_mode": True,
            }]
            self.state = {
                active: {
                    "z": torch.tensor([10.0]),
                    "exp_avg_sq": torch.tensor([4.0]),
                },
                sparse: {},
            }

    monkeypatch.setattr(
        real.train, "build_training_model", lambda _args: TwoParamModel())
    monkeypatch.setattr(
        real.train, "build_training_optimizer",
        lambda model, _args: LazyScheduleFreeFixture(model.parameters()))
    monkeypatch.setattr(
        real, "_build_batch_iter", lambda *_args, **_kwargs: object())

    session = PersistentRealWorkerSession(
        base_state={
            "active": torch.tensor([1.0]),
            "sparse": torch.tensor([2.0]),
        },
        train_args=real.Namespace(
            seed=7, bf16=False, lr=0.1, optimizer="schedulefree"),
        spec=RealAsyncWorkerSpec("trainer", "node-0", "cpu", 1, 0),
        synthetic_token_stream=False,
        synthetic_vocab_size=8,
    )
    active, sparse = tuple(session.model.parameters())

    # Preserve existing live state and initialize only the point that the
    # sparse first window has not touched.
    torch.testing.assert_close(
        session.optimizer.state[active]["z"], torch.tensor([10.0]))
    torch.testing.assert_close(
        session.optimizer.state[sparse]["z"], torch.tensor([2.0]))
    session.translate({
        "active": torch.tensor([3.0]),
        "sparse": torch.tensor([5.0]),
    })
    torch.testing.assert_close(active, torch.tensor([4.0]))
    torch.testing.assert_close(sparse, torch.tensor([7.0]))
    torch.testing.assert_close(
        session.optimizer.state[active]["z"], torch.tensor([13.0]))
    torch.testing.assert_close(
        session.optimizer.state[sparse]["z"], torch.tensor([7.0]))


def test_persistent_snapshot_uses_two_preallocated_coherent_slots(monkeypatch):
    """Adjacent immutable endpoints alternate without per-window allocation."""
    import ndm.async_diloco_real as real

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
                    "exp_avg_sq": torch.tensor([0.0]),
                },
            }

    monkeypatch.setattr(
        real.train, "build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr(
        real.train, "build_training_optimizer",
        lambda model, _args: ScheduleFreeFixture(model.parameters()))
    monkeypatch.setattr(
        real, "_build_batch_iter", lambda *_args, **_kwargs: object())

    session = PersistentRealWorkerSession(
        base_state={"weight": torch.tensor([1.0])},
        train_args=real.Namespace(seed=7, bf16=False, lr=0.1),
        spec=RealAsyncWorkerSpec("trainer", "node-0", "cpu", 1, 0),
        synthetic_token_stream=False,
        synthetic_vocab_size=8,
    )
    first = session.snapshot()
    with torch.no_grad():
        session.model.weight.fill_(2.0)
    second = session.snapshot()
    with torch.no_grad():
        session.model.weight.fill_(3.0)
    third = session.snapshot()

    assert first["weight"].data_ptr() == third["weight"].data_ptr()
    assert first["weight"].data_ptr() != second["weight"].data_ptr()
    torch.testing.assert_close(second["weight"], torch.tensor([2.0]))
    torch.testing.assert_close(third["weight"], torch.tensor([3.0]))
    assert session.snapshot_slot_count == 2
    session.close()


def test_persistent_lane_progresses_and_defers_snapshot_while_result_is_delayed(
        monkeypatch):
    import ndm.async_diloco_real as real

    first_forward = threading.Event()
    allow_forward = threading.Event()
    result_completed = threading.Event()

    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class ScheduleFreeFixture:
        def __init__(self, parameters):
            parameter = tuple(parameters)[0]
            self.param_groups = [{"params": [parameter], "lr": 0.1}]
            self.state = {
                parameter: {
                    "z": parameter.detach().clone(),
                    "exp_avg_sq": torch.tensor([4.0]),
                },
            }

    iterator = iter(())
    monkeypatch.setattr(
        real.train, "build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr(
        real.train, "build_training_optimizer",
        lambda model, _args: ScheduleFreeFixture(model.parameters()))
    monkeypatch.setattr(real, "_build_batch_iter", lambda *_args, **_kwargs: iterator)

    def train_step(model, _optimizer, _args, **_kwargs):
        first_forward.set()
        assert allow_forward.wait(1)
        with torch.no_grad():
            model.weight.add_(1.0)
        return {"loss": 1.0, "tokens_processed": 5, "hidden_state": None}

    monkeypatch.setattr(real.train, "train_one_optimizer_step", train_step)
    session = PersistentRealWorkerSession(
        base_state={"weight": torch.zeros(1)},
        train_args=real.Namespace(seed=7, bf16=False, lr=0.1),
        spec=RealAsyncWorkerSpec("trainer", "node-0", "cpu", 1, 0),
        synthetic_token_stream=False,
        synthetic_vocab_size=8,
    )
    lane = PersistentAsyncTrainingLane(session, max_windows=3)
    lane.start(
        local_window_start=1,
        start_state=session.snapshot(),
        admission_deadline=time.monotonic() + 1,
    )

    assert first_forward.wait(1)
    assert not result_completed.is_set()
    allow_forward.set()
    deadline = time.monotonic() + 1
    while session.windows_completed < 4 and time.monotonic() < deadline:
        time.sleep(.001)
    assert session.windows_completed >= 4
    assert not result_completed.is_set()

    result_completed.set()
    interval = lane.finish_at_boundary(
        deadline=time.monotonic() + 1,
        corrections={"weight": torch.tensor([7.0])},
    )
    assert interval.local_window_start == 1
    assert interval.local_window_end == session.windows_completed + 1
    assert interval.window_count == 0
    assert interval.exact_tokens == 0
    assert interval.reached_hard_bound
    assert interval.snapshot_deferred
    torch.testing.assert_close(
        session.model.weight.detach(),
        torch.tensor([float(session.windows_completed + 7)]))
    torch.testing.assert_close(
        session.optimizer.state[session.model.weight]["z"],
        torch.tensor([7.0]))
    torch.testing.assert_close(
        lane.start_state["weight"],
        torch.tensor([float(session.windows_completed + 7)]))
    assert session.bootstrap_counts == {
        "model_build": 1,
        "optimizer_build": 1,
        "data_iterator_build": 1,
    }
    session.close()


def test_real_async_trainer_checkpoint_cadence_records_recovery_and_finalization(tmp_path):
    result = run_real_async_diloco(RealAsyncDiLoCoConfig(
        run_id="real-cadence",
        run_dir=tmp_path / "run",
        train_args=_args(),
        worker_specs=(
            RealAsyncWorkerSpec(worker_id="worker-0", local_steps=1),
        ),
        local_quorum=1,
        global_quorum=1,
        global_node_count=1,
        synthetic_token_stream=True,
        walltime_remaining_s=0.0,
        estimated_finalization_duration_s=0.0,
        checkpoint_cadence=AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=1,
            recovery_every_seconds=None,
            export_every_generations=None,
            export_every_seconds=None,
            finalization_reserve_seconds=1200.0,
        ),
    ))

    paths = result.generations[0].metrics.checkpoint_paths
    assert len(paths) >= 3
    assert any("/recovery_checkpoints/" in path and "initial" in path for path in paths)
    assert any(
        "/recovery_checkpoints/" in path and "walltime_finalization" in path
        for path in paths
    )


def test_actual_multinode_tcp_rank_writes_progress_and_global_quorum(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr("ndm.async_diloco_real.train.load_checkpoint", lambda *_args, **_kwargs: None)

    def fake_node_supervisor(**kwargs):
        spec = kwargs["worker_specs"][0]
        generation = kwargs["generation"]
        delta = {"weight": torch.ones(1) * (int(spec.seed_offset) + 1)}
        update = AsyncDiLoCoUpdate(
            worker_id=kwargs["node_id"],
            base_generation=generation,
            delta=delta,
            tokens=8192,
            local_steps=2,
            loss_moving_average={"loss": 2.0 + int(spec.seed_offset), "loss_100": 2.0 + int(spec.seed_offset)},
        )
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=generation,
            update=update,
            elapsed_s=0.1,
            tokens=8192,
            losses=(2.0 + int(spec.seed_offset),),
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=generation,
            requested_workers=1,
            participating_workers=1,
            quorum_threshold=1,
            quorum_size=1,
            accepted_updates=1,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=0,
            invalid_updates=0,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=81920.0,
            tokens_per_generation=8192,
            update_bytes={"worker": 4, "node": 4},
            loss_moving_average={"loss": 2.0 + int(spec.seed_offset), "loss_100": 2.0 + int(spec.seed_offset)},
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=generation,
            node_update=update,
            worker_reports=(worker,),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_node_supervisor)

    base = {
        "run_id": "tcp-rank",
        "run_dir": tmp_path / "run",
        "metrics_json": tmp_path / "metrics.json",
        "train_args": _args(data=str(tmp_path / "real_tokens.txt")),
        "node_count": 2,
        "global_quorum": 2,
        "local_steps": 2,
        "timeout_s": 3.0,
        "synthetic_token_stream": False,
        "coordinator_host": "127.0.0.1",
        "coordinator_bind_host": "127.0.0.1",
        "coordinator_port": _free_tcp_port(),
        "walltime_remaining_s": 0.0,
        "estimated_finalization_duration_s": 0.0,
        "checkpoint_cadence": AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=1,
            finalization_reserve_seconds=1200.0,
        ),
    }
    (tmp_path / "real_tokens.txt").write_text("real token source placeholder\n", encoding="utf-8")

    result_box = {}

    def run_rank0():
        result_box["rank0"] = run_real_async_diloco_file_rank(
            RealAsyncFileRankConfig(node_rank=0, **base)
        )

    rank0_thread = threading.Thread(target=run_rank0)
    rank0_thread.start()
    rank1 = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(node_rank=1, **base))
    rank0_thread.join(timeout=5.0)
    assert not rank0_thread.is_alive()
    rank0 = result_box["rank0"]

    assert rank1["node_update_submitted"] is True
    assert rank0["coordinator"] is True
    assert rank1["transport"] == "tcp"
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "actual_multinode_tcp_quorum_debug"
    assert payload["transport"]["filesystem_live_quorum"] is False
    assert payload["transport"]["bytes_sent"] > 0
    assert payload["transport"]["submit_latency_s"]["count"] >= 1
    assert payload["synthetic_token_stream"] is False
    assert payload["latest_generation"] == 0
    assert payload["global_generations"][0]["metrics"]["quorum_status"] == "advanced"
    assert payload["global_generations"][0]["metrics"]["accepted_updates"] == 2
    assert payload["global_generations"][0]["metrics"]["tokens_per_generation"] == 16384
    assert payload["global_generations"][0]["metrics"]["tokens_per_sec"] > 0.0
    assert payload["accepted_node_ids"] == ["node-00000", "node-00001"]
    assert (tmp_path / "run" / "latest.json").exists()
    assert (tmp_path / "run" / "progress" / "node-00000.heartbeat.json").exists()
    assert (tmp_path / "run" / "progress" / "node-00001.heartbeat.json").exists()
    assert (tmp_path / "run" / "node_update_artifacts" / "node-00000.json").exists()
    assert (tmp_path / "run" / "node_update_artifacts" / "node-00001.json").exists()


def test_actual_multinode_tcp_rank_rejects_synthetic_stream(tmp_path):
    with pytest.raises(ValueError, match="synthetic_token_stream is disabled"):
        run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
            run_id="reject-synthetic",
            run_dir=tmp_path / "run",
            metrics_json=tmp_path / "metrics.json",
            train_args=_args(),
            node_rank=0,
            node_count=1,
            global_quorum=1,
            local_steps=1,
            synthetic_token_stream=True,
        ))


def test_actual_multinode_tcp_rank_serializes_failed_rank_payload(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())

    def failed_node_supervisor(**kwargs):
        spec = kwargs["worker_specs"][0]
        generation = kwargs["generation"]
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=generation,
            update=None,
            elapsed_s=0.1,
            tokens=0,
            failed=True,
            error="configured failure",
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=generation,
            requested_workers=1,
            participating_workers=1,
            quorum_threshold=1,
            quorum_size=0,
            accepted_updates=0,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=1,
            invalid_updates=0,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=0.0,
            tokens_per_generation=0,
            loss_moving_average={},
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=generation,
            node_update=None,
            worker_reports=(worker,),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", failed_node_supervisor)

    result = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
        run_id="failed-tcp-rank",
        run_dir=tmp_path / "run",
        metrics_json=tmp_path / "metrics.json",
        train_args=_args(data=str(tmp_path / "real_tokens.txt")),
        node_rank=0,
        node_count=1,
        global_quorum=1,
        local_steps=1,
        timeout_s=0.1,
    ))

    assert result["node_update_submitted"] is False
    node_payload = json.loads((tmp_path / "run" / "node_update_artifacts" / "node-00000.json").read_text(encoding="utf-8"))
    assert node_payload["loss"] is None
    assert node_payload["worker_reports"][0]["failed"] is True
    assert node_payload["worker_reports"][0]["error"] == "configured failure"


def test_actual_multinode_tcp_rank_allows_explicit_synthetic_fallback(tmp_path, monkeypatch):
    class OneParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr("ndm.async_diloco_real.train.build_training_model", lambda _args: OneParamModel())
    monkeypatch.setattr("ndm.async_diloco_real.train.load_checkpoint", lambda *_args, **_kwargs: None)

    def fake_node_supervisor(**kwargs):
        spec = kwargs["worker_specs"][0]
        generation = kwargs["generation"]
        delta = {"weight": torch.ones(1)}
        update = AsyncDiLoCoUpdate(
            worker_id=kwargs["node_id"],
            base_generation=generation,
            delta=delta,
            tokens=128,
            local_steps=1,
            loss_moving_average={"loss": 1.0, "loss_100": 1.0},
        )
        worker = RealAsyncWorkerReport(
            worker_id=spec.worker_id,
            node_id=kwargs["node_id"],
            base_generation=generation,
            update=update,
            elapsed_s=0.1,
            tokens=128,
            losses=(1.0,),
        )
        metrics = AsyncDiLoCoGenerationMetrics(
            run_id=kwargs["run_id"],
            generation=generation,
            requested_workers=1,
            participating_workers=1,
            quorum_threshold=1,
            quorum_size=1,
            accepted_updates=1,
            stale_updates=0,
            timed_out_updates=0,
            failed_updates=0,
            invalid_updates=0,
            generation_duration_s=0.1,
            merge_duration_s=0.0,
            rebase_duration_s=0.0,
            checkpoint_duration_s=0.0,
            tokens_per_sec=1280.0,
            tokens_per_generation=128,
            update_bytes={"worker": 4, "node": 4},
            loss_moving_average={"loss": 1.0, "loss_100": 1.0},
        )
        return RealAsyncNodeResult(
            node_id=kwargs["node_id"],
            generation=generation,
            node_update=update,
            worker_reports=(worker,),
            metrics=metrics,
        )

    monkeypatch.setattr("ndm.async_diloco_real._run_real_node_supervisor", fake_node_supervisor)

    result = run_real_async_diloco_file_rank(RealAsyncFileRankConfig(
        run_id="synthetic-tcp-rank",
        run_dir=tmp_path / "run",
        metrics_json=tmp_path / "metrics.json",
        train_args=_args(),
        node_rank=0,
        node_count=1,
        global_quorum=1,
        local_steps=1,
        synthetic_token_stream=True,
        allow_synthetic_token_stream=True,
    ))

    assert result["node_update_submitted"] is True
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "actual_multinode_tcp_quorum_debug"
    assert payload["synthetic_token_stream"] is True
    assert payload["latest_generation"] == 0


def test_multinode_entrypoint_no_longer_imports_synthetic_debug_harness():
    source = Path("scripts/frontier/async_diloco_e97_multinode.py").read_text(encoding="utf-8")
    assert "from e97_async_diloco_train import main" in source
    assert "from async_diloco_e97_2n8n_debug import main" not in source
