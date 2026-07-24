#!/usr/bin/env python3
"""Compare GDN2-MLP and E97 DiLoCo losses at a matched token cutoff.

This script is intentionally read-only with respect to training outputs. It
copies bounded source snapshots into an ops directory, reconstructs finite
rank-0/main loss records, aligns exact optimizer steps, applies one shared
trailing moving-average window, and writes an overlay PNG plus JSON summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.]+)"
    r".*?\|\s+tok/s\s+(?P<tok_s>[0-9.]+)"
    r".*?\|\s+global_tok/s\s+(?P<global_tok_s>[0-9.]+)"
    r".*?\|\s+time\s+(?P<time>\S+)"
)
RESUME_RE = re.compile(r"Resumed at step\s+(?P<step>\d+)")


@dataclasses.dataclass(frozen=True)
class Point:
    step: int
    loss: float
    tok_s: float
    global_tok_s: float
    timestamp: dt.datetime
    source: str
    order: int
    source_line: int

    @property
    def token(self) -> int:
        return self.step * TOKENS_PER_STEP


TOKENS_PER_STEP = 65_536
DEFAULT_CUTOFF_STEP = 78_100
DEFAULT_CUTOFF_TOKENS = DEFAULT_CUTOFF_STEP * TOKENS_PER_STEP
DEFAULT_SMOOTHING_WINDOW = 78


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tokens_per_step_from_args(path: Path) -> dict:
    args = load_json(path)
    world_size = int(args["_world_size"])
    batch_size = int(args["batch_size"])
    chunk_size = int(args["chunk_size"])
    grad_accum = int(args["grad_accum"])
    tokens_per_step = world_size * batch_size * chunk_size * grad_accum
    return {
        "path": str(path),
        "level": args.get("level"),
        "world_size": world_size,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "grad_accum": grad_accum,
        "log_every": int(args.get("log_every", 0)),
        "tokens_per_step": tokens_per_step,
        "formula": f"{world_size} * {batch_size} * {chunk_size} * {grad_accum}",
    }


def copy_snapshot(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "snapshot": str(dst),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def parse_log(path: Path, source_name: str, start_order: int) -> tuple[list[Point], int, bool, list[dict]]:
    raw = path.read_bytes()
    dropped_final_partial_line = False
    if raw and not raw.endswith(b"\n"):
        dropped_final_partial_line = True
        raw = raw.rsplit(b"\n", 1)[0] + b"\n"
    text = raw.decode("utf-8", errors="replace")
    points: list[Point] = []
    malformed: list[dict] = []
    order = start_order
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = STEP_RE.search(line)
        if match:
            values = {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "tok_s": float(match.group("tok_s")),
                "global_tok_s": float(match.group("global_tok_s")),
                "timestamp": parse_time(match.group("time")),
            }
            if all(
                math.isfinite(v)
                for v in (values["loss"], values["tok_s"], values["global_tok_s"])
            ):
                points.append(
                    Point(
                        step=values["step"],
                        loss=values["loss"],
                        tok_s=values["tok_s"],
                        global_tok_s=values["global_tok_s"],
                        timestamp=values["timestamp"],
                        source=source_name,
                        order=order,
                        source_line=line_no,
                    )
                )
                order += 1
            continue
        if "step" in line and "loss" in line and "|" in line:
            malformed.append({"source": source_name, "source_line": line_no, "line_prefix": line[:200]})
    return points, order, dropped_final_partial_line, malformed


def dedupe(points: list[Point], cutoff_step: int) -> tuple[list[Point], list[Point]]:
    by_step: dict[int, Point] = {}
    superseded: list[Point] = []
    for point in sorted((p for p in points if p.step <= cutoff_step), key=lambda p: (p.timestamp, p.order)):
        previous = by_step.get(point.step)
        if previous is not None:
            superseded.append(previous)
        by_step[point.step] = point
    return [by_step[step] for step in sorted(by_step)], superseded


def moving_average(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    queue: list[float] = []
    total = 0.0
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        out.append(total / len(queue))
    return out


def assert_monotonic(points: list[Point]) -> dict:
    steps = [p.step for p in points]
    tokens = [p.token for p in points]
    return {
        "strictly_increasing_steps": all(b > a for a, b in zip(steps, steps[1:])),
        "strictly_increasing_tokens": all(b > a for a, b in zip(tokens, tokens[1:])),
        "finite": all(math.isfinite(p.loss) for p in points),
    }


def interval_summary(points: list[Point], lo: int, hi: int) -> dict:
    obs = [p for p in points if lo <= p.step <= hi]
    steps = [p.step for p in obs]
    diffs = [b - a for a, b in zip(steps, steps[1:])]
    if not obs:
        raise SystemExit(f"no aligned observations in interval {lo}..{hi}")
    cadence = "n/a"
    if diffs:
        cadence = str(diffs[0]) if all(d == diffs[0] for d in diffs) else "mixed:" + ",".join(map(str, sorted(set(diffs))))
    return {
        "bounds": [lo, hi],
        "token_bounds": [lo * TOKENS_PER_STEP, hi * TOKENS_PER_STEP],
        "observation_count": len(obs),
        "first_observed_step": steps[0],
        "last_observed_step": steps[-1],
        "first_observed_token": steps[0] * TOKENS_PER_STEP,
        "last_observed_token": steps[-1] * TOKENS_PER_STEP,
        "cadence_optimizer_steps": cadence,
        "every_optimizer_step_represented": len(obs) == (hi - lo + 1) and all(d == 1 for d in diffs),
        "coverage_note": "Arithmetic mean over aligned logged observations only; no interpolation.",
        "mean_loss": sum(p.loss for p in obs) / len(obs),
    }


def series_summary(label: str, raw_count: int, points: list[Point], superseded: list[Point], dropped: list[bool], malformed: list[dict]) -> dict:
    return {
        "label": label,
        "raw_points_through_cutoff": raw_count,
        "effective_points_through_cutoff": len(points),
        "duplicates_removed": len(superseded),
        "dropped_final_partial_line": any(dropped),
        "malformed_step_like_lines": len(malformed),
        "step_range": [points[0].step, points[-1].step],
        "token_range": [points[0].token, points[-1].token],
        "monotonic": assert_monotonic(points),
    }


def plot_overlay(aligned: list[dict], output: Path, smoothing_window: int, cutoff_tokens: int) -> None:
    tokens_b = [row["token"] / 1e9 for row in aligned]
    e97_raw = [row["e97_loss"] for row in aligned]
    gdn2_raw = [row["gdn2_loss"] for row in aligned]
    e97_sm = moving_average(e97_raw, smoothing_window)
    gdn2_sm = moving_average(gdn2_raw, smoothing_window)

    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    ax.plot(tokens_b, e97_raw, color="#2563eb", linewidth=0.75, alpha=0.23, label="E97 raw")
    ax.plot(tokens_b, gdn2_raw, color="#f97316", linewidth=0.75, alpha=0.25, label="GDN2-MLP raw")
    ax.plot(tokens_b, e97_sm, color="#1d4ed8", linewidth=2.25, label=f"E97 trailing MA ({smoothing_window} records)")
    ax.plot(tokens_b, gdn2_sm, color="#c2410c", linewidth=2.25, label=f"GDN2-MLP trailing MA ({smoothing_window} records)")
    ax.axvline(cutoff_tokens / 1e9, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(
        cutoff_tokens / 1e9,
        min(min(e97_sm), min(gdn2_sm)),
        f" cutoff {cutoff_tokens / 1e9:.6f}B",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#111827",
    )
    ax.set_title(f"GDN2-MLP vs E97 DiLoCo Training Loss through {cutoff_tokens / 1e9:.6f}B Matched Tokens")
    ax.set_xlabel("Aggregate training tokens (billions)")
    ax.set_ylabel("Rank-0/main logged training loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.margins(x=0.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    if output.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{output} is not a PNG")
    image = mpimg.imread(output)
    if image.size == 0:
        raise SystemExit(f"{output} read as blank/empty image")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops-dir", type=Path, required=True)
    ap.add_argument("--gdn2-snapshot", type=Path, required=True)
    ap.add_argument("--gdn2-args", type=Path, required=True)
    ap.add_argument("--e97-run-root", type=Path, required=True)
    ap.add_argument("--e97-args", type=Path, required=True)
    ap.add_argument("--cutoff-step", type=int, default=DEFAULT_CUTOFF_STEP)
    ap.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    ap.add_argument("--snapshot-utc", default="2026-07-23T07:58:22Z")
    ap.add_argument("--output-png", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    cutoff_step = args.cutoff_step
    cutoff_tokens = cutoff_step * TOKENS_PER_STEP

    gdn2_config = tokens_per_step_from_args(args.gdn2_args)
    e97_config = tokens_per_step_from_args(args.e97_args)
    if gdn2_config["tokens_per_step"] != TOKENS_PER_STEP or e97_config["tokens_per_step"] != TOKENS_PER_STEP:
        raise SystemExit(f"token semantics mismatch: gdn2={gdn2_config}, e97={e97_config}")

    snapshots_dir = args.ops_dir / "snapshots"
    gdn2_snap_meta = [copy_snapshot(args.gdn2_snapshot, snapshots_dir / args.gdn2_snapshot.name)]
    e97_sources = sorted(args.e97_run_root.glob("run*.log"), key=lambda p: (p.stat().st_mtime, p.name))
    if not e97_sources:
        raise SystemExit("no E97 run*.log files found")
    e97_snap_meta = [copy_snapshot(path, snapshots_dir / "e97" / path.name) for path in e97_sources]

    order = 0
    gdn2_points_raw: list[Point] = []
    gdn2_dropped: list[bool] = []
    gdn2_malformed: list[dict] = []
    for meta in gdn2_snap_meta:
        parsed, order, dropped, malformed = parse_log(Path(meta["snapshot"]), Path(meta["snapshot"]).name, order)
        gdn2_points_raw.extend(parsed)
        gdn2_dropped.append(dropped)
        gdn2_malformed.extend(malformed)
    gdn2_points, gdn2_superseded = dedupe(gdn2_points_raw, cutoff_step)

    order = 0
    e97_points_raw: list[Point] = []
    e97_dropped: list[bool] = []
    e97_malformed: list[dict] = []
    for meta in e97_snap_meta:
        parsed, order, dropped, malformed = parse_log(Path(meta["snapshot"]), Path(meta["snapshot"]).name, order)
        e97_points_raw.extend(parsed)
        e97_dropped.append(dropped)
        e97_malformed.extend(malformed)
    e97_points, e97_superseded = dedupe(e97_points_raw, cutoff_step)

    expected_window = min(80, max(5, len(gdn2_points) // 40))
    if args.smoothing_window != expected_window:
        raise SystemExit(
            f"smoothing-window {args.smoothing_window} does not match GDN2 "
            f"effective-count rule min(80,max(5,n//40))={expected_window} for n={len(gdn2_points)}"
        )

    e97_by_step = {p.step: p for p in e97_points}
    gdn2_by_step = {p.step: p for p in gdn2_points}
    common_steps = sorted(set(e97_by_step) & set(gdn2_by_step))
    if not common_steps:
        raise SystemExit("no common logged optimizer steps through cutoff")
    aligned_points = [
        {
            "step": step,
            "token": step * TOKENS_PER_STEP,
            "e97_loss": e97_by_step[step].loss,
            "gdn2_loss": gdn2_by_step[step].loss,
            "delta_gdn2_minus_e97": gdn2_by_step[step].loss - e97_by_step[step].loss,
        }
        for step in common_steps
    ]
    if common_steps[-1] != cutoff_step:
        raise SystemExit(f"latest common step {common_steps[-1]} does not equal cutoff step {cutoff_step}")

    missing = {
        "e97_only_steps_through_cutoff": sorted(set(e97_by_step) - set(gdn2_by_step)),
        "gdn2_only_steps_through_cutoff": sorted(set(gdn2_by_step) - set(e97_by_step)),
    }
    e97_aligned = [e97_by_step[step] for step in common_steps]
    gdn2_aligned = [gdn2_by_step[step] for step in common_steps]
    e97_smoothed = moving_average([p.loss for p in e97_aligned], args.smoothing_window)
    gdn2_smoothed = moving_average([p.loss for p in gdn2_aligned], args.smoothing_window)

    raw_cutoff = {
        "step": cutoff_step,
        "token": cutoff_tokens,
        "e97_loss": e97_by_step[cutoff_step].loss,
        "gdn2_loss": gdn2_by_step[cutoff_step].loss,
        "delta_gdn2_minus_e97": gdn2_by_step[cutoff_step].loss - e97_by_step[cutoff_step].loss,
    }
    smoothed_cutoff = {
        "step": cutoff_step,
        "token": cutoff_tokens,
        "window_aligned_records": args.smoothing_window,
        "e97_loss": e97_smoothed[-1],
        "gdn2_loss": gdn2_smoothed[-1],
        "delta_gdn2_minus_e97": gdn2_smoothed[-1] - e97_smoothed[-1],
    }

    intervals = {}
    for width in (100, 1000):
        lo = cutoff_step - width + 1
        hi = cutoff_step
        e97_interval = interval_summary(e97_aligned, lo, hi)
        gdn2_interval = interval_summary(gdn2_aligned, lo, hi)
        intervals[f"last_{width}_optimizer_steps"] = {
            "bounds": [lo, hi],
            "e97": e97_interval,
            "gdn2": gdn2_interval,
            "delta_gdn2_minus_e97": gdn2_interval["mean_loss"] - e97_interval["mean_loss"],
        }

    all_delta_mean = sum(row["delta_gdn2_minus_e97"] for row in aligned_points) / len(aligned_points)
    recent_1000_rows = [row for row in aligned_points if cutoff_step - 999 <= row["step"] <= cutoff_step]
    recent_1000_delta_mean = sum(row["delta_gdn2_minus_e97"] for row in recent_1000_rows) / len(recent_1000_rows)

    plot_overlay(aligned_points, args.output_png, args.smoothing_window, cutoff_tokens)
    plot_hash = sha256_file(args.output_png)
    plot_stat = args.output_png.stat()

    summary = {
        "snapshot_utc": args.snapshot_utc,
        "cutoff": {"step": cutoff_step, "tokens": cutoff_tokens, "tokens_billions": cutoff_tokens / 1e9},
        "token_semantics": {
            "validated_same_tokens_per_step": True,
            "tokens_per_step": TOKENS_PER_STEP,
            "e97": e97_config,
            "gdn2": gdn2_config,
        },
        "source_policy": {
            "e97": "canonical run-root run*.log files parsed as rank-0/main training stdout; dedupe by optimizer step keeping latest timestamp/order record",
            "gdn2": "predecessor bounded run.log snapshot parsed as rank-0/main training stdout; dedupe by optimizer step keeping latest timestamp/order record",
            "finite_complete_records_only": True,
            "no_interpolation": True,
        },
        "snapshots": {"gdn2": gdn2_snap_meta, "e97": e97_snap_meta},
        "series": {
            "e97": series_summary("E97", len([p for p in e97_points_raw if p.step <= cutoff_step]), e97_points, e97_superseded, e97_dropped, e97_malformed),
            "gdn2": series_summary("GDN2-MLP", len([p for p in gdn2_points_raw if p.step <= cutoff_step]), gdn2_points, gdn2_superseded, gdn2_dropped, gdn2_malformed),
        },
        "alignment": {
            "aligned_record_count": len(aligned_points),
            "common_step_range": [common_steps[0], common_steps[-1]],
            "common_token_range": [common_steps[0] * TOKENS_PER_STEP, common_steps[-1] * TOKENS_PER_STEP],
            "missing_or_nonoverlapping": {
                "e97_only_count": len(missing["e97_only_steps_through_cutoff"]),
                "gdn2_only_count": len(missing["gdn2_only_steps_through_cutoff"]),
                "e97_only_steps_through_cutoff": missing["e97_only_steps_through_cutoff"],
                "gdn2_only_steps_through_cutoff": missing["gdn2_only_steps_through_cutoff"],
            },
        },
        "smoothing": {
            "method": "trailing moving average over aligned effective records",
            "window_aligned_records": args.smoothing_window,
            "window_reason": (
                f"GDN2 cutoff has {len(gdn2_points)} effective records, so "
                f"min(80, max(5, n//40)) = {expected_window}; applied identically "
                "to E97 aligned records"
            ),
        },
        "cutoff_raw": raw_cutoff,
        "cutoff_smoothed": smoothed_cutoff,
        "intervals": intervals,
        "aggregate_delta_metrics": {
            "mean_signed_loss_delta_gdn2_minus_e97_all_aligned_points": all_delta_mean,
            "mean_signed_loss_delta_gdn2_minus_e97_recent_1000_step_interval": recent_1000_delta_mean,
            "definition": "Arithmetic mean of per-step GDN2 loss minus E97 loss over aligned logged records; descriptive only, no statistical significance implied.",
        },
        "plot": {
            "path": str(args.output_png),
            "sha256": plot_hash,
            "size_bytes": plot_stat.st_size,
            "readable_png": True,
            "image_shape": list(mpimg.imread(args.output_png).shape),
        },
        "confirmations": {
            "no_training_control": True,
            "no_checkpoint_modification": True,
            "no_s3_command": True,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
