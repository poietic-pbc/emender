#!/usr/bin/env python3
"""Plot the live E97/Emender 8-GPU DiLoCo loss curve from existing logs."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.]+)"
    r".*?\|\s+tok/s\s+(?P<tok_s>[0-9.]+)"
    r".*?\|\s+global_tok/s\s+(?P<global_tok_s>[0-9.]+)"
    r".*?\|\s+time\s+(?P<time>\S+)"
)
RESUME_RE = re.compile(r"Resumed at step\s+(?P<step>\d+)")
RESUME_FROM_RE = re.compile(r"Resuming from\s+(?P<path>\S+)")
SAVE_RE = re.compile(r"saved checkpoint:\s+checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[0-9.]+)\.pt")
CKPT_NAME_RE = re.compile(r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[0-9.]+)\.pt$")


@dataclasses.dataclass(frozen=True)
class Point:
    step: int
    loss: float
    tok_s: float
    global_tok_s: float
    timestamp: dt.datetime
    source: Path
    order: int


@dataclasses.dataclass(frozen=True)
class Resume:
    step: int
    source: Path
    checkpoint_path: str | None = None


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    step: int
    loss: float
    source: Path


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_logs(paths: list[Path]) -> tuple[list[Point], list[Resume], list[Checkpoint]]:
    points: list[Point] = []
    resumes: list[Resume] = []
    saves: list[Checkpoint] = []
    order = 0
    pending_resume_path: str | None = None

    for path in paths:
        pending_resume_path = None
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if match := RESUME_FROM_RE.search(line):
                    pending_resume_path = match.group("path")
                    continue
                if match := RESUME_RE.search(line):
                    resumes.append(
                        Resume(
                            step=int(match.group("step")),
                            source=path,
                            checkpoint_path=pending_resume_path,
                        )
                    )
                    continue
                if match := SAVE_RE.search(line):
                    saves.append(
                        Checkpoint(
                            step=int(match.group("step")),
                            loss=float(match.group("loss")),
                            source=path,
                        )
                    )
                    continue
                match = STEP_RE.search(line)
                if not match:
                    continue
                points.append(
                    Point(
                        step=int(match.group("step")),
                        loss=float(match.group("loss")),
                        tok_s=float(match.group("tok_s")),
                        global_tok_s=float(match.group("global_tok_s")),
                        timestamp=parse_time(match.group("time")),
                        source=path,
                        order=order,
                    )
                )
                order += 1

    return points, resumes, saves


def checkpoint_files(run_root: Path) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for path in sorted(run_root.glob("runs/*/checkpoint_step_*_loss_*.pt")):
        match = CKPT_NAME_RE.search(path.name)
        if not match:
            continue
        checkpoints.append(
            Checkpoint(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                source=path,
            )
        )
    return checkpoints


def effective_lineage(points: list[Point]) -> tuple[list[Point], list[Point]]:
    """Keep the latest observed record for duplicated steps after resume rollbacks."""

    by_step: dict[int, Point] = {}
    superseded: list[Point] = []
    for point in sorted(points, key=lambda p: (p.timestamp, p.order)):
        previous = by_step.get(point.step)
        if previous is not None:
            superseded.append(previous)
        by_step[point.step] = point
    return [by_step[step] for step in sorted(by_step)], superseded


def moving_average(values: list[float], window: int) -> list[float]:
    averaged: list[float] = []
    total = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        averaged.append(total / len(queue))
    return averaged


def plot(
    points: list[Point],
    superseded: list[Point],
    resumes: list[Resume],
    checkpoints: list[Checkpoint],
    output: Path,
    sources: list[Path],
) -> None:
    steps = [point.step for point in points]
    losses = [point.loss for point in points]
    avg_window = min(80, max(5, len(points) // 40))
    avg = moving_average(losses, avg_window)

    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    if superseded:
        superseded_steps = [point.step for point in superseded]
        superseded_losses = [point.loss for point in superseded]
        ax.scatter(
            superseded_steps,
            superseded_losses,
            s=8,
            alpha=0.22,
            color="#9ca3af",
            label="superseded pre-resume points",
            linewidths=0,
            zorder=1,
        )
    ax.plot(steps, losses, color="#2563eb", linewidth=0.85, alpha=0.5, label="observed loss", zorder=2)
    ax.plot(
        steps,
        avg,
        color="#dc2626",
        linewidth=2.0,
        label=f"moving average ({avg_window} pts)",
        zorder=3,
    )

    checkpoint_by_step = {checkpoint.step: checkpoint for checkpoint in checkpoints}
    visible_checkpoints = [checkpoint_by_step[step] for step in sorted(checkpoint_by_step) if steps[0] <= step <= steps[-1]]
    if visible_checkpoints:
        ymin, ymax = min(losses), max(losses)
        marker_y = ymin + (ymax - ymin) * 0.055
        ax.scatter(
            [checkpoint.step for checkpoint in visible_checkpoints],
            [checkpoint.loss for checkpoint in visible_checkpoints],
            marker="v",
            s=30,
            color="#059669",
            edgecolors="white",
            linewidths=0.4,
            label=f"checkpoints ({len(visible_checkpoints)})",
            zorder=4,
        )
        for checkpoint in visible_checkpoints[:: max(1, len(visible_checkpoints) // 12)]:
            ax.text(
                checkpoint.step,
                marker_y,
                f"{checkpoint.step:,}",
                rotation=90,
                fontsize=6,
                color="#047857",
                va="bottom",
                ha="center",
                alpha=0.75,
            )

    resume_steps = sorted({resume.step for resume in resumes})
    for index, step in enumerate(resume_steps):
        if steps[0] <= step <= steps[-1]:
            ax.axvline(
                step,
                color="#7c3aed",
                linestyle="--",
                linewidth=1.1,
                alpha=0.55,
                label="resume point" if index == 0 else None,
                zorder=0,
            )

    latest = points[-1]
    source_names = ", ".join(path.name for path in sources)
    ax.set_title(
        "E97 / Emender 8-GPU DiLoCo Training Loss\n"
        f"{len(points):,} effective points, latest step {latest.step:,} loss {latest.loss:.4f}"
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.text(
        0.01,
        0.01,
        f"Sources: {source_names}",
        transform=ax.transAxes,
        fontsize=7,
        color="#4b5563",
        va="bottom",
        ha="left",
    )
    ax.margins(x=0.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/emender"),
        help="E97/Emender run root containing split logs and run directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/experiments/figures/e97_diloco_loss_curve_20260623.png"),
    )
    args = parser.parse_args()

    existing_logs = sorted(
        args.run_root.glob("run*.log"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    points, resumes, save_markers = parse_logs(existing_logs)
    effective_points, superseded = effective_lineage(points)
    checkpoints = save_markers + checkpoint_files(args.run_root)
    if not effective_points:
        raise SystemExit("No step/loss points parsed from E97 logs")

    plot(effective_points, superseded, resumes, checkpoints, args.output, existing_logs)
    latest = effective_points[-1]
    print(f"output={args.output}")
    print(f"sources={','.join(str(path) for path in existing_logs)}")
    print(f"raw_points={len(points)}")
    print(f"effective_points={len(effective_points)}")
    print(f"superseded_points={len(superseded)}")
    print(f"resume_steps={','.join(str(step) for step in sorted({resume.step for resume in resumes})) or 'none'}")
    print(f"checkpoint_steps={','.join(str(step) for step in sorted({checkpoint.step for checkpoint in checkpoints})) or 'none'}")
    print(f"latest_step={latest.step}")
    print(f"latest_loss={latest.loss:.4f}")
    print(f"latest_time={latest.timestamp.isoformat()}")


if __name__ == "__main__":
    main()
