#!/usr/bin/env python3
"""Fail-closed live K40 overlap, foreground-idle, and cadence validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


MAX_IDLE_FRACTION = 0.10
MAX_CADENCE_MULTIPLE = 1.25
BACKGROUND_STAGES = frozenset({
    "native_local_reduction", "native_owner_redistribution",
    "native_trainer_apply", "fenced_atomic_commit",
})


def _records(root: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    values.append(value)
    return values


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    trainer: dict[str, dict[int, dict[str, list[float]]]] = {}
    backgrounds: list[tuple[int, str, float, float]] = []
    for value in records:
        generation = int(value.get("generation", -1))
        if generation < 0:
            continue
        phase = str(value.get("phase", ""))
        if phase in {"optimizer_step_start", "optimizer_step_end"}:
            identity = str(value.get("identity", ""))
            if not identity or "monotonic_s" not in value:
                raise ValueError("trainer step telemetry lacks identity/monotonic timestamp")
            trainer.setdefault(identity, {}).setdefault(generation, {}).setdefault(
                phase, []).append(float(value["monotonic_s"]))
        stage = str(value.get("stage", ""))
        if stage in BACKGROUND_STAGES:
            elapsed = float(value.get("elapsed_s", -1))
            ended = float(value.get("timestamp", -1))
            if elapsed < 0 or ended < 0 or value.get("within_slo") is not True:
                raise ValueError(f"invalid or failed background telemetry: {stage}")
            backgrounds.append((generation, stage, ended - elapsed, ended))
    if not trainer:
        raise ValueError("no real trainer step telemetry")

    raw: list[float] = []
    cadence: list[float] = []
    idle: list[float] = []
    overlaps: list[dict[str, Any]] = []
    for identity, generations in trainer.items():
        ordered = sorted(generations)
        if len(ordered) < 2:
            raise ValueError(f"{identity} lacks two steady-state K40 generations")
        windows: dict[int, tuple[float, float]] = {}
        for generation in ordered:
            phases = generations[generation]
            starts, ends = phases.get("optimizer_step_start", []), phases.get("optimizer_step_end", [])
            if len(starts) != 40 or len(ends) != 40:
                raise ValueError(f"{identity} generation {generation} lacks exact K40 step timestamps")
            start, end = min(starts), max(ends)
            if end <= start:
                raise ValueError("non-positive K40 compute interval")
            windows[generation] = (start, end)
            raw.append(end - start)
        for previous, current in zip(ordered, ordered[1:]):
            previous_window, current_window = windows[previous], windows[current]
            observed_cadence = current_window[0] - previous_window[0]
            control_idle = max(0.0, current_window[0] - previous_window[1])
            cadence.append(observed_cadence)
            idle.append(control_idle)
            # Background telemetry uses wall clock while trainer events retain
            # both clocks. Translate the next K40 window through its first
            # step's paired wall/monotonic timestamp.
            current_starts = [
                value for value in records
                if value.get("identity") == identity
                and int(value.get("generation", -1)) == current
                and value.get("phase") == "optimizer_step_start"
            ]
            first = min(current_starts, key=lambda value: float(value["monotonic_s"]))
            offset = float(first["timestamp"]) - float(first["monotonic_s"])
            next_wall = (current_window[0] + offset, current_window[1] + offset)
            matched = [stage for generation, stage, began, ended in backgrounds
                       if generation == previous and began < next_wall[1] and ended > next_wall[0]]
            if not matched:
                raise ValueError(
                    f"generation {previous} background did not overlap {current} K40 compute"
                )
            overlaps.append({"identity": identity, "background_generation": previous,
                             "foreground_generation": current, "stages": sorted(set(matched))})

    raw_k40 = statistics.median(raw)
    cadence_s = statistics.median(cadence)
    idle_fraction = sum(idle) / sum(cadence)
    background_max = max((ended - began for _g, _s, began, ended in backgrounds), default=0.0)
    background_fits = background_max <= raw_k40
    cadence_multiple = cadence_s / raw_k40
    if background_fits and cadence_multiple > MAX_CADENCE_MULTIPLE:
        raise ValueError(
            f"steady-state cadence {cadence_multiple:.6f}x exceeds 1.25x raw K40"
        )
    if idle_fraction >= MAX_IDLE_FRACTION:
        raise ValueError(f"foreground control-plane idle {idle_fraction:.6f} is not below 0.10")
    return {
        "schema": "emender-pipelined-e97-performance-v1", "status": "passed",
        "raw_k40_compute_seconds": raw_k40, "steady_state_cadence_seconds": cadence_s,
        "steady_state_cadence_multiple": cadence_multiple,
        "foreground_control_plane_idle_fraction": idle_fraction,
        "background_max_stage_seconds": background_max,
        "background_fits_k40_window": background_fits, "overlaps": overlaps,
        "policy": {"foreground_idle_fraction_strict_max": MAX_IDLE_FRACTION,
                   "cadence_multiple_max_when_background_fits": MAX_CADENCE_MULTIPLE},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(_records(args.telemetry_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
