#!/usr/bin/env python3
"""Run the documented exact-shape eight-GPU E97 4B LR interference bracket."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

CANDIDATES = [
    ("lr0400", 0.0004000),
    ("lr0474", 0.00047431158698290157),
    ("lr0550", 0.0005500),
    ("lr0630", 0.0006300),
    ("lr0720", 0.0007200),
    ("lr0820", 0.0008200),
    ("lr0920", 0.0009200),
    ("lr1007", 0.0010070),
]
STEP_RE = re.compile(
    r"^step\s+(\d+) \| loss ([0-9.eE+-]+) \| lr ([0-9.eE+-]+) "
    r"\| grad ([0-9.eE+-]+) \| tok/s ([0-9.eE+-]+) "
    r"\| global_tok/s ([0-9.eE+-]+)")
FITNESS_STEPS = tuple(range(72, 97, 4))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def command_for(root: Path, name: str, lr: float) -> list[str]:
    candidate_output = root / "candidates" / name / "runs"
    return [
        sys.executable, "scripts/numa_local_rank_exec.py", "--", "train.py",
        "--level", "E97", "--params", "4b",
        "--dim", "3840", "--depth", "18", "--n_heads", "60",
        "--n_state", "64", "--expansion", "1.0",
        "--use_gate", "1", "--gate_activation", "silu",
        "--linear_state", "0", "--mlp_ratio", "2.5",
        "--mlp_multiple", "64", "--use_triton", "1",
        "--optimizer", "schedulefree", "--offload_schedulefree_state",
        "--lr", repr(lr), "--weight_decay", "0.01", "--warmup_steps", "0",
        "--bf16", "--batch_size", "32", "--chunk_size", "2048",
        "--grad_accum", "1", "--gradient_checkpointing",
        "--gradient_checkpoint_group_size", "2", "--checkpoint_interval", "16",
        "--projection_chunk_size", "512", "--loss_chunk_size", "128",
        "--grad_clip", "1.0", "--seed", "42", "--compile_warmup_steps", "1",
        "--data", "/home/erikg/elman/data/pile.txt", "--tokenizer", "p50k_base",
        "--sampler_schema", "emender-byte-window-counter-v1",
        "--sampler_corpus_sha256",
        "5eb92c0f16157710c90e33b02fd5b7852b30713d4c754f4220ad7120155db464",
        "--sampler_tokenizer_sha256",
        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
        "--sampler_key", "42", "--sampler_data_world_size", "1",
        "--steps", "96", "--save_every", "999999", "--keep_checkpoints", "1",
        "--log_every", "4", "--output", str(candidate_output),
    ]


def mean_at(trajectory: dict[int, dict[str, float]], steps: tuple[int, ...]) -> float | None:
    if any(step not in trajectory for step in steps):
        return None
    return sum(trajectory[step]["loss"] for step in steps) / len(steps)


def parse_candidate(name: str, lr: float, gpu: int, command: list[str],
                    logfile: Path, returncode: int) -> dict:
    trajectory: dict[int, dict[str, float]] = {}
    text = logfile.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = STEP_RE.match(line)
        if not match:
            continue
        step = int(match.group(1))
        trajectory[step] = {
            "loss": float(match.group(2)),
            "lr_reported": float(match.group(3)),
            "grad_norm": float(match.group(4)),
            "tokens_per_second": float(match.group(5)),
            "global_tokens_per_second": float(match.group(6)),
        }

    failure_markers = [
        marker for marker in ("Traceback", "out of memory", "non-finite")
        if marker in text
    ]
    fused_guard = "NO eager fallback" in text
    fitness = mean_at(trajectory, FITNESS_STEPS)
    complete = (
        returncode == 0 and fitness is not None and fused_guard
        and not failure_markers and "Training complete! Final step: 96" in text
    )
    status = "complete" if complete else "failed"
    windows = {
        "steps_4_32_mean": mean_at(trajectory, tuple(range(4, 33, 4))),
        "steps_40_64_mean": mean_at(trajectory, tuple(range(40, 65, 4))),
        "steps_72_96_mean": fitness,
    }
    return {
        "name": name,
        "learning_rate": lr,
        "gpu": gpu,
        "numa_node": 0 if gpu <= 3 else 1,
        "status": status,
        "fitness": fitness if complete else None,
        "fitness_steps": list(FITNESS_STEPS),
        "windows": windows,
        "max_reported_grad_norm": (
            max(row["grad_norm"] for row in trajectory.values())
            if trajectory else None
        ),
        "returncode": returncode,
        "fused_guard_passed": fused_guard,
        "failure_markers": failure_markers,
        "trajectory": {str(step): row for step, row in sorted(trajectory.items())},
        "command": command,
        "logfile": str(logfile),
    }


def write_results(root: Path, manifest: dict, results: list[dict]) -> None:
    complete = [row for row in results if row["fitness"] is not None]
    complete.sort(key=lambda row: row["fitness"])
    rank = {row["name"]: index + 1 for index, row in enumerate(complete)}
    for row in results:
        row["rank"] = rank.get(row["name"])
    payload = {
        **manifest,
        "completed_at": utc_now(),
        "fitness_definition": "mean reported training loss at steps 72,76,...,96",
        "fitness_steps": list(FITNESS_STEPS),
        "winner": complete[0]["name"] if complete else None,
        "candidates": sorted(results, key=lambda row: (row["rank"] is None, row["rank"] or 999)),
    }
    temporary = root / ".results.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, root / "results.json")

    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "name", "learning_rate", "status", "fitness",
                         "steps_4_32_mean", "steps_40_64_mean", "max_grad_norm", "gpu"])
        for row in payload["candidates"]:
            writer.writerow([
                row["rank"], row["name"], row["learning_rate"], row["status"],
                row["fitness"], row["windows"]["steps_4_32_mean"],
                row["windows"]["steps_40_64_mean"], row["max_reported_grad_norm"],
                row["gpu"],
            ])


def _candidate_spec(value: str) -> tuple[str, float]:
    try:
        name, raw_lr = value.split("=", 1)
        lr = float(raw_lr)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be NAME=LR") from exc
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise argparse.ArgumentTypeError(f"invalid candidate name: {name!r}")
    if not math.isfinite(lr) or lr <= 0:
        raise argparse.ArgumentTypeError(f"learning rate must be positive and finite: {raw_lr!r}")
    return name, lr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", type=_candidate_spec,
        help="override default population with repeated NAME=LR entries")
    args = parser.parse_args()
    candidates = args.candidate or CANDIDATES
    if len(candidates) != 8:
        raise SystemExit(f"scan requires exactly eight candidates; got {len(candidates)}")
    names = [name for name, _lr in candidates]
    if len(set(names)) != len(names):
        raise SystemExit(f"candidate names must be unique: {names!r}")
    root = args.root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"refusing nonempty scan root: {root}")
    (root / "candidates").mkdir(parents=True, exist_ok=True)

    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if len(visible) != 8 or any(not item.isdigit() for item in visible):
        raise SystemExit(f"scan requires eight numeric leased GPUs; got {visible!r}")
    physical_gpus = [int(item) for item in visible]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "emender-e97-4b-lr-interference-scan-v1",
        "started_at": utc_now(),
        "source_commit": source_commit,
        "protocol": "docs/experiments/e97-4b-learning-rate-rapid-interference-scan.md",
        "shape": "d3840-L18-H60-n64-mlp2.5",
        "total_parameters": 4_045_972_080,
        "steps": 96,
        "seed": 42,
        "physical_gpus": physical_gpus,
        "candidate_learning_rates": [
            {"name": name, "learning_rate": lr} for name, lr in candidates
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    children: list[dict] = []
    stopping = False

    def forward(signum, _frame):
        nonlocal stopping
        stopping = True
        print(f"[scan] forwarding signal {signum} to live candidates", flush=True)
        for child in children:
            if child["process"].poll() is None:
                child["process"].send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    try:
        for (name, lr), gpu in zip(candidates, physical_gpus, strict=True):
            candidate_dir = root / "candidates" / name
            candidate_dir.mkdir(parents=True, exist_ok=False)
            logfile = candidate_dir / "run.log"
            command = command_for(root, name, lr)
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "LOCAL_RANK": "0",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "OMP_NUM_THREADS": "4",
                "TIKTOKEN_CACHE_DIR": "/tmp/data-gym-cache",
            })
            handle = logfile.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=Path(__file__).resolve().parents[1], env=env,
                stdout=handle, stderr=subprocess.STDOUT, text=True)
            children.append({
                "name": name, "lr": lr, "gpu": gpu, "command": command,
                "logfile": logfile, "handle": handle, "process": process,
            })
            print(f"[scan] launched {name} lr={lr:.17g} gpu={gpu} pid={process.pid}", flush=True)
            time.sleep(1.0)

        live = set(range(len(children)))
        failed_early = False
        while live:
            for index in tuple(live):
                code = children[index]["process"].poll()
                if code is None:
                    continue
                live.remove(index)
                children[index]["returncode"] = code
                children[index]["handle"].close()
                print(f"[scan] exited {children[index]['name']} returncode={code}", flush=True)
                if code != 0 and not stopping:
                    failed_early = True
            if failed_early:
                print("[scan] candidate failed; stopping remaining population", flush=True)
                for index in live:
                    children[index]["process"].send_signal(signal.SIGTERM)
                failed_early = False
                stopping = True
            if live:
                time.sleep(1.0)
    finally:
        for child in children:
            if child["process"].poll() is None:
                child["process"].send_signal(signal.SIGTERM)
        for child in children:
            if child["process"].poll() is None:
                try:
                    child["process"].wait(timeout=900)
                except subprocess.TimeoutExpired:
                    child["process"].kill()
                    child["process"].wait()
            if not child["handle"].closed:
                child["handle"].close()

    results = [
        parse_candidate(
            child["name"], child["lr"], child["gpu"], child["command"],
            child["logfile"], child["process"].returncode)
        for child in children
    ]
    write_results(root, manifest, results)
    complete = [row for row in results if row["fitness"] is not None]
    if complete:
        winner = min(complete, key=lambda row: row["fitness"])
        print(f"[scan] WINNER name={winner['name']} lr={winner['learning_rate']:.17g} "
              f"fitness={winner['fitness']:.6f}", flush=True)
    else:
        print("[scan] no complete candidates", flush=True)
    return 0 if len(complete) == len(candidates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
