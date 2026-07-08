import json
from pathlib import Path
import re

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoUpdate
from ndm.async_diloco_local import (
    LocalAsyncCheckpointManager,
    LocalScheduleFreeDelta,
    LocalScheduleFreeState,
    rebase_local_schedulefree_state,
)
from ndm.async_diloco_mpi import (
    DenseTransportQuorumConfig,
    collect_dense_quorum_from_envelopes,
    pack_dense_update,
)


def _dense_base():
    return {
        "x": torch.zeros(2, dtype=torch.float32),
        "z": torch.zeros(2, dtype=torch.float32),
    }


def _dense_update(worker_id, value, *, base_generation=0, tokens=1):
    return AsyncDiLoCoUpdate(
        worker_id=str(worker_id),
        base_generation=base_generation,
        delta={
            "x": torch.tensor([float(value), 0.0], dtype=torch.float32),
            "z": torch.tensor([0.0, float(value)], dtype=torch.float32),
        },
        tokens=tokens,
        local_steps=1,
        loss_moving_average={"loss": float(value)},
    )


def test_failure_injection_missing_and_stuck_ranks_advance_without_unanimity(tmp_path):
    base = _dense_base()
    envelopes = (
        pack_dense_update(_dense_update("rank-0", 1.0), run_id="missing-stuck", rank=0, generation=0),
        pack_dense_update(_dense_update("rank-1", 3.0), run_id="missing-stuck", rank=1, generation=0),
    )

    result = collect_dense_quorum_from_envelopes(
        base,
        envelopes,
        config=DenseTransportQuorumConfig(
            run_id="missing-stuck",
            generation=0,
            base_generation=0,
            requested_ranks=4,
            quorum=2,
            timeout_s=0.01,
        ),
        timed_out_ranks=(2, 3),
    )
    evidence = {
        "scenario": "missing_and_stuck_before_merge",
        "quorum_status": result.metrics.quorum_status,
        "accepted_updates": result.metrics.accepted_updates,
        "timed_out_updates": result.metrics.timed_out_updates,
        "timed_out_ranks": list(result.transport_metrics.timed_out_ranks),
        "transport": result.to_payload()["transport"],
    }
    evidence_path = tmp_path / "missing-stuck-evidence.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")

    assert result.metrics.quorum_status == "advanced"
    assert result.metrics.quorum_size == 2
    assert result.metrics.accepted_updates == 2
    assert result.metrics.timed_out_updates == 2
    assert result.transport_metrics.timed_out_ranks == (2, 3)
    assert result.to_payload()["transport"]["filesystem_live_quorum"] is False
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["timed_out_ranks"] == [2, 3]
    torch.testing.assert_close(result.state["x"], torch.tensor([2.0, 0.0]))


def test_late_base_generation_policy_accepts_current_and_rejects_old_with_metrics():
    base = _dense_base()
    old_late = pack_dense_update(
        _dense_update("late-old", 100.0, base_generation=1),
        run_id="late-policy",
        rank=1,
        generation=2,
        staleness=1,
    )
    current0 = pack_dense_update(
        _dense_update("current-0", 2.0, base_generation=2),
        run_id="late-policy",
        rank=0,
        generation=2,
    )
    current2 = pack_dense_update(
        _dense_update("current-2", 4.0, base_generation=2),
        run_id="late-policy",
        rank=2,
        generation=2,
    )

    result = collect_dense_quorum_from_envelopes(
        base,
        (old_late, current0, current2),
        config=DenseTransportQuorumConfig(
            run_id="late-policy",
            generation=2,
            base_generation=2,
            requested_ranks=3,
            quorum=2,
            stale_policy="reject",
        ),
    )

    assert result.metrics.quorum_status == "advanced"
    assert [update.worker_id for update in result.accepted_updates] == ["current-0", "current-2"]
    assert [update.worker_id for update in result.stale_updates] == ["late-old"]
    assert result.metrics.accepted_updates == 2
    assert result.metrics.stale_updates == 1
    assert result.transport_metrics.stale_ranks == (1,)
    torch.testing.assert_close(result.state["x"], torch.tensor([3.0, 0.0]))


def test_stale_worker_catchup_loads_latest_rebases_and_resets_base_generation(tmp_path):
    old_base = LocalScheduleFreeState(
        x=(torch.tensor([1.0, 2.0]),),
        z=(torch.tensor([3.0, 4.0]),),
    )
    local_state = LocalScheduleFreeState(
        x=(torch.tensor([1.5, 1.25]),),
        z=(torch.tensor([2.75, 4.5]),),
    )
    current_global = LocalScheduleFreeState(
        x=(torch.tensor([10.0, 20.0]),),
        z=(torch.tensor([30.0, 40.0]),),
    )
    manager = LocalAsyncCheckpointManager(tmp_path / "run-local")
    manager.save_generation(5, current_global, {"kind": "global", "global_generation": 5})

    observed_global_generation = manager.latest_generation()
    assert observed_global_generation == 5
    _generation, loaded_current, manifest = manager.load_latest()
    caught_up_state = rebase_local_schedulefree_state(local_state, old_base, loaded_current)
    catchup_delta = LocalScheduleFreeDelta(
        dx=tuple(caught - base for caught, base in zip(caught_up_state.x, loaded_current.x)),
        dz=tuple(caught - base for caught, base in zip(caught_up_state.z, loaded_current.z)),
    )
    resumed = {
        "worker_id": 2,
        "detected_global_generation": observed_global_generation,
        "loaded_manifest_generation": manifest["generation"],
        "reset_base_generation": observed_global_generation,
        "catchup_delta_norm": float(torch.linalg.vector_norm(catchup_delta.dx[0])),
    }

    torch.testing.assert_close(caught_up_state.x[0] - loaded_current.x[0], local_state.x[0] - old_base.x[0])
    torch.testing.assert_close(caught_up_state.z[0] - loaded_current.z[0], local_state.z[0] - old_base.z[0])
    assert resumed == {
        "worker_id": 2,
        "detected_global_generation": 5,
        "loaded_manifest_generation": 5,
        "reset_base_generation": 5,
        "catchup_delta_norm": pytest.approx(0.901388, abs=1e-6),
    }


def test_run_local_latest_is_isolated_from_production_latest_guard(tmp_path):
    seed_checkpoint = tmp_path / "seed.pt"
    torch.save({"model": "seed"}, seed_checkpoint)
    production_latest = tmp_path / "production_latest.pt"
    production_latest.symlink_to(seed_checkpoint)
    production_before = production_latest.resolve()

    manager = LocalAsyncCheckpointManager(tmp_path / "debug-run")
    manager.save_generation(
        0,
        LocalScheduleFreeState(
            x=(torch.tensor([1.0]),),
            z=(torch.tensor([2.0]),),
        ),
        {"kind": "debug", "latest_scope": "run-local"},
    )

    assert manager.latest_generation() == 0
    assert (tmp_path / "debug-run" / "latest").exists()
    assert production_latest.resolve() == production_before
    assert not (tmp_path / "latest").exists()


def test_resilient_dense_transport_and_strict_collective_paths_are_both_present():
    repo = Path(__file__).resolve().parents[1]
    mpi_path = (repo / "ndm" / "async_diloco_mpi.py").read_text(encoding="utf-8")
    helper_path = (repo / "scripts" / "frontier" / "compiled_mpich_dense_helper.cpp").read_text(encoding="utf-8")

    assert "comm.Irecv" in mpi_path
    assert "comm.Isend" in mpi_path
    assert "collect_dense_quorum_from_envelopes" in mpi_path
    assert "MPI_Reduce" in helper_path
    assert re.search(r'\\"strict_collective_all_launched_ranks\\"\s*:\s*true', helper_path)
