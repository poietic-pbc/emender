from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/validate_pipelined_e97_performance.py"
SPEC = importlib.util.spec_from_file_location("pipelined_perf", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _records(*, next_start=105.0, background=(102.0, 112.0)):
    values = []
    for generation, start in ((0, 0.0), (1, next_start)):
        for step in range(40):
            began = start + step * 2.5
            values.extend([
                {"identity": "trainer-0", "generation": generation,
                 "phase": "optimizer_step_start", "monotonic_s": began,
                 "timestamp": 1000.0 + began},
                {"identity": "trainer-0", "generation": generation,
                 "phase": "optimizer_step_end", "monotonic_s": began + 2.0,
                 "timestamp": 1002.0 + began},
            ])
    began, ended = background
    values.append({"identity": "manager-0", "generation": 0,
                   "stage": "native_owner_redistribution", "elapsed_s": ended - began,
                   "timestamp": 1000.0 + ended, "within_slo": True})
    return values


def test_live_k40_accepts_overlap_low_idle_and_bounded_cadence():
    result = MODULE.validate(_records())
    assert result["status"] == "passed"
    assert result["foreground_control_plane_idle_fraction"] < 0.10
    assert result["steady_state_cadence_multiple"] <= 1.25
    assert result["overlaps"][0]["stages"] == ["native_owner_redistribution"]


def test_live_k40_rejects_missing_background_overlap():
    with pytest.raises(ValueError, match="did not overlap"):
        MODULE.validate(_records(background=(1.0, 2.0)))


def test_live_k40_rejects_ten_percent_or_more_foreground_idle():
    with pytest.raises(ValueError, match="idle"):
        MODULE.validate(_records(next_start=120.0, background=(121.0, 130.0)))


def test_live_k40_rejects_slow_cadence_when_background_fits():
    # Keep idle below 10% while exceeding the 1.25 cadence bound by making raw
    # K40 98.5 s and cadence 124 s; the aggregate idle ratio is checked first.
    records = _records(next_start=126.0, background=(127.0, 132.0))
    for value in records:
        if value.get("phase") == "optimizer_step_end":
            value["monotonic_s"] += 0.45
            value["timestamp"] += 0.45
    with pytest.raises(ValueError, match="cadence"):
        MODULE.validate(records)


def test_live_k40_requires_exact_step_timestamps():
    records = _records()
    records.pop(0)
    with pytest.raises(ValueError, match="exact K40"):
        MODULE.validate(records)
