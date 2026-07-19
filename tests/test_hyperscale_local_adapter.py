from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ndm.fenced_admission import SQLiteFencedControlStore
from ndm.hyperscale_local_adapter import HyperscaleLocalAdapter, HyperscaleLocalConfig
from ndm.native_artifacts import NATIVE_CXI, NATIVE_TEST, validate_backend


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_native_pool_hyperscale_local.py"
BUILD_MANIFEST = ROOT / "build/native-resilient-dataplane/native-artifacts.json"


def _module():
    spec = importlib.util.spec_from_file_location(
        "validate_native_pool_hyperscale_local", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, manifest: Path) -> HyperscaleLocalConfig:
    return HyperscaleLocalConfig(
        run_id="run", allocation_id="allocation-b",
        allocation_incarnation="allocation-b-boot",
        protocol_id="pool-v1", config_id="config-v1",
        control_db=tmp_path / "control.sqlite3",
        evidence_root=tmp_path / "loser-evidence",
        build_manifest=manifest, source_root=ROOT,
    )


def test_exclusive_lease_loser_exits_before_manifest_or_native_work(tmp_path):
    config = _config(tmp_path, tmp_path / "manifest-must-not-be-read.json")
    store = SQLiteFencedControlStore(config.control_db)
    assert store.acquire(
        run_id=config.run_id, allocation_id="allocation-a",
        incarnation="allocation-a-boot", protocol_id=config.protocol_id,
        config_id=config.config_id, ttl_s=60) is not None
    starts = []

    def forbidden_factory(**kwargs):
        starts.append(kwargs)
        raise AssertionError("lease loser started a native session")

    assert HyperscaleLocalAdapter.try_start(
        config, session_factory=forbidden_factory) is None
    assert starts == []
    assert not config.evidence_root.exists()


def test_repeated_fenced_control_transactions_close_descriptors(tmp_path):
    store = SQLiteFencedControlStore(tmp_path / "bounded-control.sqlite3")
    lease = store.acquire(
        run_id="run", allocation_id="allocation", incarnation="boot",
        protocol_id="pool-v1", config_id="config-v1", ttl_s=60)
    assert lease is not None
    before = len(tuple(Path("/proc/self/fd").iterdir()))
    for generation in range(32):
        store.assert_current(lease)
        store.publish(
            lease, kind="checkpoint", name=f"g{generation}",
            payload={"generation": generation})
        assert store.read_publication(
            "run", "checkpoint", f"g{generation}") == {"generation": generation}
    after = len(tuple(Path("/proc/self/fd").iterdir()))
    assert after == before


def test_local_adapter_cannot_weaken_frontier_provider_selection():
    assert validate_backend(NATIVE_TEST, production=False, full_layout=False) == NATIVE_TEST
    with pytest.raises(ValueError, match="cannot be promoted"):
        validate_backend(NATIVE_TEST, production=True, full_layout=False)
    assert validate_backend(NATIVE_CXI, production=True, full_layout=True) == NATIVE_CXI


def test_repeated_real_native_failure_restart_gate_uses_dynamic_membership(tmp_path):
    result = _module().run_gate(
        build_manifest=BUILD_MANIFEST,
        output_root=tmp_path / "gate",
        failure_restart_cycles=4,
        elements=12)
    assert result["status"] == "passed"
    assert result["exclusive_lease"]["loser_native_starts"] == 0
    assert result["membership"]["launched_world_size"] is None
    assert result["membership"]["late_join_excluded_from_open_generation"] is True
    assert result["failure_restart_cycles"] == 4
    assert all(result["resource_bounds"]["checks"].values())
    assert result["forbidden_path_counters"] == {
        "python_dense_socket_bytes": 0,
        "trainer_spool_bytes": 0,
        "handoff_full_copy_bytes": 0,
        "slurm_jobs_submitted": 0,
    }


def test_gate_refuses_a_single_failure_as_repeated_restart_evidence(tmp_path):
    with pytest.raises(ValueError, match="at least 4"):
        _module().run_gate(
            build_manifest="unused", output_root=tmp_path,
            failure_restart_cycles=1, elements=12)
