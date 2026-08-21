#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
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
SAVE_RE = re.compile(r"saved checkpoint:\s+checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[0-9.]+)\.pt")
CKPT_RE = re.compile(r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[0-9.]+)\.pt$")


@dataclasses.dataclass(frozen=True)
class Point:
    step: int
    loss: float
    tok_s: float
    global_tok_s: float
    timestamp: dt.datetime
    order: int
    source_line: int


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_summary(effective: list[Point], max_step: int, width: int) -> dict:
    lo = max_step - width + 1
    hi = max_step
    obs = [p for p in effective if lo <= p.step <= hi]
    if not obs:
        raise SystemExit(f"no observations inside final {width}-step interval {lo}..{hi}")
    losses = [p.loss for p in obs]
    steps = [p.step for p in obs]
    diffs = [b - a for a, b in zip(steps, steps[1:])]
    cadence = "n/a"
    if diffs:
        cadence = str(diffs[0]) if all(d == diffs[0] for d in diffs) else "mixed:" + ",".join(map(str, sorted(set(diffs))))
    return {
        "width_optimizer_steps": width,
        "bounds": [lo, hi],
        "observation_count": len(obs),
        "first_observed_step": steps[0],
        "last_observed_step": steps[-1],
        "observed_steps": steps,
        "cadence_optimizer_steps": cadence,
        "every_optimizer_step_represented": len(obs) == width and all(d == 1 for d in diffs),
        "mean_loss_observed_records": sum(losses) / len(losses),
        "note": "Arithmetic mean over logged observations inside the optimizer-step interval; no interpolation.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--launch-manifest", type=Path, required=True)
    ap.add_argument("--args-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--snapshot-utc", required=True)
    args = ap.parse_args()

    run_args = load_json(args.args_json)
    launch_manifest = load_json(args.launch_manifest)
    world_size = int(run_args["_world_size"])
    batch_size = int(run_args["batch_size"])
    chunk_size = int(run_args["chunk_size"])
    grad_accum = int(run_args["grad_accum"])
    tokens_per_step = world_size * batch_size * chunk_size * grad_accum
    if tokens_per_step != 65_536:
        raise SystemExit(f"tokens_per_step discrepancy: config gives {tokens_per_step}, expected 65536")
    if int(run_args["log_every"]) != 25:
        raise SystemExit(f"unexpected log_every={run_args['log_every']}; interval cadence reporting assumes logged records")

    raw = args.snapshot.read_bytes()
    dropped_final_partial_line = False
    if raw and not raw.endswith(b"\n"):
        dropped_final_partial_line = True
        raw = raw.rsplit(b"\n", 1)[0] + b"\n"
    text = raw.decode("utf-8", errors="replace")

    points: list[Point] = []
    save_markers: list[dict[str, object]] = []
    malformed_step_like_lines: list[dict[str, object]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        save_match = SAVE_RE.search(line)
        if save_match:
            save_markers.append(
                {"step": int(save_match.group("step")), "loss": float(save_match.group("loss")), "source_line": line_no}
            )
        match = STEP_RE.search(line)
        if match:
            point = Point(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                tok_s=float(match.group("tok_s")),
                global_tok_s=float(match.group("global_tok_s")),
                timestamp=parse_time(match.group("time")),
                order=len(points),
                source_line=line_no,
            )
            points.append(point)
        elif "step" in line and "loss" in line and "|" in line:
            malformed_step_like_lines.append({"source_line": line_no, "line_prefix": line[:200]})

    if not points:
        raise SystemExit("no finite complete loss records parsed from snapshot")

    by_step: dict[int, Point] = {}
    superseded: list[Point] = []
    for point in sorted(points, key=lambda p: (p.timestamp, p.order)):
        old = by_step.get(point.step)
        if old is not None:
            superseded.append(old)
        by_step[point.step] = point
    effective = [by_step[step] for step in sorted(by_step)]
    steps = [p.step for p in effective]
    losses = [p.loss for p in effective]
    tokens = [p.step * tokens_per_step for p in effective]

    nonfinite = [p for p in effective if not all(math.isfinite(v) for v in (p.loss, p.tok_s, p.global_tok_s))]
    nonmonotonic_steps = [(a.step, b.step) for a, b in zip(effective, effective[1:]) if b.step <= a.step]
    nonmonotonic_tokens = [(a, b) for a, b in zip(tokens, tokens[1:]) if b <= a]
    final_record_rejected = dropped_final_partial_line or any(
        item["source_line"] == len(text.splitlines()) for item in malformed_step_like_lines
    )

    errors: list[str] = []
    if nonfinite:
        errors.append(f"nonfinite numeric values in {len(nonfinite)} effective points")
    if nonmonotonic_steps:
        errors.append(f"non-monotonic effective steps: {nonmonotonic_steps[:5]}")
    if nonmonotonic_tokens:
        errors.append(f"non-monotonic tokens: {nonmonotonic_tokens[:5]}")
    if dropped_final_partial_line:
        errors.append("snapshot ended with an incomplete final line")
    if final_record_rejected:
        errors.append("final record was incomplete or malformed")
    if errors:
        raise SystemExit("sanity check failed: " + "; ".join(errors))

    window = min(80, max(5, len(effective) // 40))
    smoothed = moving_average(losses, window)

    checkpoints: list[dict[str, object]] = []
    for path in sorted(args.run_dir.glob("checkpoint_step_*_loss_*.pt")):
        m = CKPT_RE.match(path.name)
        if not m:
            continue
        st = path.stat()
        checkpoints.append(
            {
                "step": int(m.group("step")),
                "loss": float(m.group("loss")),
                "path": str(path),
                "size_bytes": st.st_size,
                "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).isoformat(),
            }
        )
    checkpoints.extend(save_markers)

    visible_ckpts = sorted(
        {int(c["step"]): c for c in checkpoints if steps[0] <= int(c["step"]) <= steps[-1]}.values(),
        key=lambda c: int(c["step"]),
    )
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    ax.plot(tokens, losses, color="#2563eb", linewidth=0.9, alpha=0.45, label="observed loss")
    ax.plot(tokens, smoothed, color="#dc2626", linewidth=2.1, label=f"moving average ({window} pts)")
    if visible_ckpts:
        ax.scatter(
            [int(c["step"]) * tokens_per_step for c in visible_ckpts],
            [float(c["loss"]) for c in visible_ckpts],
            marker="v",
            s=34,
            color="#059669",
            edgecolors="white",
            linewidths=0.45,
            label=f"checkpoints ({len(visible_ckpts)})",
            zorder=4,
        )
    latest = effective[-1]
    ax.set_title(
        f"GDN2-MLP 8-GPU DiLoCo Progress: {tokens[-1] / 1e9:.3f}B tokens, smoothed loss {smoothed[-1]:.4f}"
    )
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Training loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.text(
        0.01,
        0.01,
        f"snapshot: {args.snapshot.name}; tokens/step={tokens_per_step}; latest raw loss={latest.loss:.4f}",
        transform=ax.transAxes,
        fontsize=8,
        color="#4b5563",
        va="bottom",
        ha="left",
    )
    ax.margins(x=0.01)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)

    if args.output.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("output is not a PNG")
    image = mpimg.imread(args.output)
    if image.size == 0:
        raise SystemExit("matplotlib image read produced an empty array")

    plot_stat = args.output.stat()
    summary = {
        "snapshot_utc": args.snapshot_utc,
        "run_id": launch_manifest["name"],
        "run_root": str(args.run_root),
        "run_dir": str(args.run_dir),
        "log_source": str(args.run_root / "run.log"),
        "snapshot": {
            "path": str(args.snapshot),
            "size_bytes": args.snapshot.stat().st_size,
            "bounded_tail_bytes": 2_097_152,
        },
        "rank_source_policy": "rank-0/main training stdout in shared run.log; one complete logged loss record per log_every optimizer steps; duplicate optimizer steps would keep the latest timestamp/order record.",
        "config": {
            "world_size": world_size,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "grad_accum": grad_accum,
            "log_every": int(run_args["log_every"]),
            "tokens_per_step": tokens_per_step,
            "tokens_per_step_formula": "world_size * batch_size * chunk_size * grad_accum = 8 * 4 * 2048 * 1",
        },
        "smoothing": {
            "method": "trailing moving average over effective plotted loss records",
            "window_formula": "min(80, max(5, effective_point_count // 40))",
            "window_points": window,
        },
        "records": {
            "raw_points": len(points),
            "effective_points": len(effective),
            "superseded_points": len(superseded),
            "duplicate_steps_removed": len(points) - len(effective),
            "step_range": [steps[0], steps[-1]],
            "token_range": [tokens[0], tokens[-1]],
        },
        "latest": {
            "step": latest.step,
            "tokens": tokens[-1],
            "tokens_billions": tokens[-1] / 1e9,
            "raw_loss": latest.loss,
            "smoothed_loss": smoothed[-1],
            "time_utc": latest.timestamp.isoformat(),
        },
        "intervals": {
            "last_100_optimizer_steps": interval_summary(effective, latest.step, 100),
            "last_1000_optimizer_steps": interval_summary(effective, latest.step, 1000),
        },
        "sanity": {
            "finite_effective_records": len(nonfinite) == 0,
            "strictly_increasing_steps": len(nonmonotonic_steps) == 0,
            "strictly_increasing_tokens": len(nonmonotonic_tokens) == 0,
            "malformed_step_like_lines": len(malformed_step_like_lines),
            "dropped_final_partial_line": dropped_final_partial_line,
            "final_record_rejected": final_record_rejected,
            "passed": True,
        },
        "plot": {
            "path": str(args.output),
            "size_bytes": plot_stat.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(plot_stat.st_mtime, dt.timezone.utc).isoformat(),
            "sha256": sha256_file(args.output),
            "readable": True,
            "image_shape": list(image.shape),
        },
        "checkpoint_steps_visible": [int(c["step"]) for c in visible_ckpts],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
