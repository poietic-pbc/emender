from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_native_dataplane_local.py"


def _module():
    spec = importlib.util.spec_from_file_location("validate_native_dataplane_local", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_e97_frontier_layout_and_resident_bounds_are_machine_checked():
    accounting = _module().full_layout_accounting()
    assert accounting["total_elements"] == 688_346_312
    assert accounting["layout_bytes"] == 5_506_770_496
    assert accounting["shard_count"] == 83
    assert accounting["last_shard_bytes"] == 3_843_648
    assert accounting["logical_contribution_bytes"] == 1_409_733_246_976
    assert accounting["logical_redistribution_bytes"] == 1_409_733_246_976
    assert accounting["resident_bound_two_owners_bytes"] == 14_440_737_184
    assert accounting["eight_trainer_lane_bytes"] == 22_027_081_984
    assert accounting["eight_lanes_fit_local_default"] is False


def test_local_gate_refuses_a_partial_stress_run():
    module = _module()
    try:
        module.run_gate(
            library_path="unused", generations=99, warmup_generations=1,
            elements=8, workers=2,
        )
    except ValueError as error:
        assert "at least 100" in str(error)
    else:  # pragma: no cover - fail clearly if the hard gate is weakened
        raise AssertionError("partial local stress run was accepted")
