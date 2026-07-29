#!/usr/bin/env python3
"""Refresh the GDN2 ops snapshot and publish verified loss plots.

This script is intentionally read-only for training state: it snapshots log
prefixes, parses complete finite rank-0/main records, writes ops artifacts, and
publishes PNGs through same-directory temporary files followed by remote mv.
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
import subprocess
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


TOKENS_PER_STEP = 65_536
TARGET_150B = 150_000_000_000
TARGET_E97_PARITY = 150_793_748_480
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[0-9.]+)"
    r".*?\|\s+tok/s\s+(?P<tok_s>[0-9.]+)"
    r".*?\|\s+global_tok/s\s+(?P<global_tok_s>[0-9.]+)"
    r".*?\|\s+time\s+(?P<time>\S+)"
)
CKPT_RE = re.compile(r"checkpoint_step_(?P<step>\d+)_loss_(?P<loss>[0-9.]+)\.pt$")


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


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso_z(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def snapshot_prefix(source: Path, dest: Path) -> dict:
    source_stat = source.stat()
    raw = b""
    remaining = source_stat.st_size
    with source.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            raw += chunk
            remaining -= len(chunk)
    dropped_partial = False
    if raw and not raw.endswith(b"\n"):
        dropped_partial = True
        raw = raw.rsplit(b"\n", 1)[0] + b"\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return {
        "source": str(source),
        "snapshot": str(dest),
        "source_size_bound_bytes": source_stat.st_size,
        "source_mtime_utc": iso_z(dt.datetime.fromtimestamp(source_stat.st_mtime, dt.timezone.utc)),
        "snapshot_size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "dropped_final_partial_line": dropped_partial,
    }


def copy_snapshot(source: Path, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return {
        "source": str(source),
        "snapshot": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
    }


def parse_log(path: Path, source_name: str, start_order: int = 0) -> tuple[list[Point], int, bool, list[dict]]:
    raw = path.read_bytes()
    dropped_partial = False
    if raw and not raw.endswith(b"\n"):
        dropped_partial = True
        raw = raw.rsplit(b"\n", 1)[0] + b"\n"
    text = raw.decode("utf-8", errors="replace")
    points: list[Point] = []
    malformed: list[dict] = []
    order = start_order
    for line_no, line in enumerate(text.splitlines(), start=1):
        match = STEP_RE.search(line)
        if match:
            point = Point(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                tok_s=float(match.group("tok_s")),
                global_tok_s=float(match.group("global_tok_s")),
                timestamp=parse_time(match.group("time")),
                source=source_name,
                order=order,
                source_line=line_no,
            )
            if all(math.isfinite(v) for v in (point.loss, point.tok_s, point.global_tok_s)):
                points.append(point)
                order += 1
            continue
        if "step" in line and "loss" in line and "|" in line:
            malformed.append({"source": source_name, "source_line": line_no, "line_prefix": line[:200]})
    return points, order, dropped_partial, malformed


def dedupe(points: list[Point], cutoff_step: int | None = None) -> tuple[list[Point], list[Point]]:
    by_step: dict[int, Point] = {}
    superseded: list[Point] = []
    eligible = points if cutoff_step is None else [p for p in points if p.step <= cutoff_step]
    for point in sorted(eligible, key=lambda p: (p.timestamp, p.order)):
        old = by_step.get(point.step)
        if old is not None:
            superseded.append(old)
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


def cadence(steps: list[int]) -> str:
    diffs = [b - a for a, b in zip(steps, steps[1:])]
    if not diffs:
        return "n/a"
    unique = sorted(set(diffs))
    return str(unique[0]) if len(unique) == 1 else "mixed:" + ",".join(str(x) for x in unique)


def assert_series(points: list[Point], label: str) -> None:
    if not points:
        raise SystemExit(f"{label}: no records")
    if not all(math.isfinite(p.loss) and math.isfinite(p.tok_s) and math.isfinite(p.global_tok_s) for p in points):
        raise SystemExit(f"{label}: non-finite record")
    if not all(b.step > a.step for a, b in zip(points, points[1:])):
        raise SystemExit(f"{label}: non-monotonic steps")
    if not all(b.token > a.token for a, b in zip(points, points[1:])):
        raise SystemExit(f"{label}: non-monotonic tokens")


def interval_summary(points: list[Point], max_step: int, width: int) -> dict:
    lo = max_step - width + 1
    hi = max_step
    obs = [p for p in points if lo <= p.step <= hi]
    if not obs:
        raise SystemExit(f"no observations inside interval {lo}..{hi}")
    steps = [p.step for p in obs]
    return {
        "width_optimizer_steps": width,
        "bounds": [lo, hi],
        "token_bounds": [lo * TOKENS_PER_STEP, hi * TOKENS_PER_STEP],
        "observation_count": len(obs),
        "first_observed_step": steps[0],
        "last_observed_step": steps[-1],
        "first_observed_token": steps[0] * TOKENS_PER_STEP,
        "last_observed_token": steps[-1] * TOKENS_PER_STEP,
        "cadence_optimizer_steps": cadence(steps),
        "every_optimizer_step_represented": len(obs) == width and all((b - a) == 1 for a, b in zip(steps, steps[1:])),
        "mean_loss_observed_records": sum(p.loss for p in obs) / len(obs),
        "note": "Arithmetic mean over logged observations inside the optimizer-step interval; no interpolation.",
    }


def series_summary(points_raw: list[Point], points: list[Point], superseded: list[Point], dropped: list[bool], malformed: list[dict]) -> dict:
    return {
        "raw_points": len(points_raw),
        "effective_points": len(points),
        "duplicates_removed": len(points_raw) - len(points),
        "superseded_points": len(superseded),
        "malformed_step_like_lines": len(malformed),
        "dropped_final_partial_line": any(dropped),
        "finite": all(math.isfinite(p.loss) for p in points),
        "strictly_increasing_steps": all(b.step > a.step for a, b in zip(points, points[1:])),
        "strictly_increasing_tokens": all(b.token > a.token for a, b in zip(points, points[1:])),
        "step_range": [points[0].step, points[-1].step],
        "token_range": [points[0].token, points[-1].token],
        "cadence_optimizer_steps": cadence([p.step for p in points]),
    }


def token_config(path: Path) -> dict:
    args = load_json(path)
    world_size = int(args["_world_size"])
    batch_size = int(args["batch_size"])
    chunk_size = int(args["chunk_size"])
    grad_accum = int(args["grad_accum"])
    tokens = world_size * batch_size * chunk_size * grad_accum
    if tokens != TOKENS_PER_STEP:
        raise SystemExit(f"{path}: tokens/step={tokens}, expected {TOKENS_PER_STEP}")
    return {
        "path": str(path),
        "world_size": world_size,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "grad_accum": grad_accum,
        "log_every": int(args.get("log_every", 0)),
        "tokens_per_step": tokens,
        "formula": f"{world_size} * {batch_size} * {chunk_size} * {grad_accum}",
    }


def inspect_health(run_root: Path) -> dict:
    ps_out = check_output(["ps", "-eo", "pid,ppid,stat,etimes,cmd"])
    torchrun = []
    launch_wrappers = []
    ranks = []
    for line in ps_out.splitlines()[1:]:
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        pid, ppid, stat, etimes, cmd = parts
        if "gdn2-mlp" not in cmd or "train.py" not in cmd:
            continue
        item = {"pid": int(pid), "ppid": int(ppid), "stat": stat, "etimes": int(etimes), "cmd": cmd}
        if "torchrun" in cmd and "bash -c" in cmd:
            launch_wrappers.append(item)
        elif "torchrun" in cmd:
            torchrun.append(item)
        elif "python3 -u train.py" in cmd:
            ranks.append(item)
    torchrun_pids = [p["pid"] for p in torchrun]
    worker_ranks = [p for p in ranks if p["ppid"] in torchrun_pids]
    gpu = None
    try:
        pmon = check_output(["nvidia-smi", "pmon", "-c", "1"])
        gpu = pmon
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        gpu = f"unavailable: {exc}"
    return {
        "torchrun_active": len(torchrun) >= 1,
        "torchrun_pids": torchrun_pids,
        "launch_wrapper_pids": [p["pid"] for p in launch_wrappers],
        "worker_rank_count": len(worker_ranks),
        "worker_rank_pids": [p["pid"] for p in worker_ranks],
        "all_8_ranks_active": len(worker_ranks) == 8,
        "status": "torchrun active with 8 worker ranks" if len(worker_ranks) == 8 else "not all expected ranks active",
        "ps_matches": {"torchrun": torchrun, "launch_wrappers": launch_wrappers, "worker_ranks": worker_ranks},
        "nvidia_smi_pmon": gpu,
        "run_log_exists": (run_root / "run.log").exists(),
    }


def plot_gdn2(points: list[Point], smoothed: list[float], window: int, output: Path) -> dict:
    tokens_b = [p.token / 1e9 for p in points]
    losses = [p.loss for p in points]
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    ax.plot(tokens_b, losses, color="#2563eb", linewidth=0.85, alpha=0.4, label="observed loss")
    ax.plot(tokens_b, smoothed, color="#dc2626", linewidth=2.1, label=f"trailing MA ({window} records)")
    ax.set_title(f"GDN2-MLP 8-GPU DiLoCo: {points[-1].token / 1e9:.6f}B tokens, smoothed loss {smoothed[-1]:.4f}")
    ax.set_xlabel("Aggregate training tokens (billions)")
    ax.set_ylabel("Rank-0/main logged training loss")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax.margins(x=0.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    image = mpimg.imread(output)
    if image.size == 0:
        raise SystemExit(f"{output} read as empty")
    return {"path": str(output), "sha256": sha256_file(output), "size_bytes": output.stat().st_size, "readable_png": True, "image_shape": list(image.shape)}


def plot_overlay(aligned: list[dict], e97_smoothed: list[float], gdn2_smoothed: list[float], window: int, output: Path) -> dict:
    tokens_b = [row["token"] / 1e9 for row in aligned]
    e97_raw = [row["e97_loss"] for row in aligned]
    gdn2_raw = [row["gdn2_loss"] for row in aligned]
    cutoff_tokens = aligned[-1]["token"]
    fig, ax = plt.subplots(figsize=(14, 7.5), dpi=160)
    ax.plot(tokens_b, e97_raw, color="#2563eb", linewidth=0.75, alpha=0.22, label="E97 raw")
    ax.plot(tokens_b, gdn2_raw, color="#f97316", linewidth=0.75, alpha=0.25, label="GDN2-MLP raw")
    ax.plot(tokens_b, e97_smoothed, color="#1d4ed8", linewidth=2.25, label=f"E97 trailing MA ({window})")
    ax.plot(tokens_b, gdn2_smoothed, color="#c2410c", linewidth=2.25, label=f"GDN2-MLP trailing MA ({window})")
    ax.axvline(cutoff_tokens / 1e9, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(cutoff_tokens / 1e9, min(min(e97_smoothed), min(gdn2_smoothed)), f" cutoff {cutoff_tokens / 1e9:.6f}B", rotation=90, va="bottom", ha="right", fontsize=8)
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
    image = mpimg.imread(output)
    if image.size == 0:
        raise SystemExit(f"{output} read as empty")
    return {"path": str(output), "sha256": sha256_file(output), "size_bytes": output.stat().st_size, "readable_png": True, "image_shape": list(image.shape)}


def http_artifact(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "wg-gdn2-ops-refresh/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        headers = response.headers
        return {
            "url": url,
            "status": int(response.status),
            "content_type": headers.get_content_type(),
            "content_length": int(headers.get("Content-Length", len(body))),
            "sha256": sha256_bytes(body),
        }


def ssh_hash(remote: str, path: str) -> str | None:
    try:
        out = check_output(["ssh", remote, "sha256sum", path])
    except subprocess.CalledProcessError:
        return None
    return out.split()[0]


def ssh_test(remote: str, path: str) -> bool:
    return subprocess.run(["ssh", remote, "test", "-f", path], check=False).returncode == 0


def scp_upload(local: Path, remote: str, path: str) -> None:
    subprocess.check_call(["scp", str(local), f"{remote}:{path}"])


def ssh_mv(remote: str, src: str, dst: str) -> None:
    subprocess.check_call(["ssh", remote, "mv", src, dst])


def publish_overwrite(local: Path, remote: str, target_path: str, url: str, tag: str) -> dict:
    local_hash = sha256_file(local)
    before_ssh = ssh_hash(remote, target_path)
    before_http = http_artifact(url)
    tmp = f"{str(Path(target_path).parent)}/.{Path(target_path).name}.tmp.{tag}.{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
    scp_upload(local, remote, tmp)
    tmp_hash = ssh_hash(remote, tmp)
    if tmp_hash != local_hash:
        raise SystemExit(f"temporary upload hash mismatch for {target_path}: {tmp_hash} != {local_hash}")
    ssh_mv(remote, tmp, target_path)
    after_ssh = ssh_hash(remote, target_path)
    after_http = http_artifact(url)
    if after_ssh != local_hash or after_http["sha256"] != local_hash:
        raise SystemExit(f"publication verification failed for {target_path}")
    if after_http["status"] != 200 or after_http["content_type"] != "image/png":
        raise SystemExit(f"HTTP verification failed for {url}: {after_http}")
    return {
        "target_path": target_path,
        "url": url,
        "mode": "overwrite",
        "local_sha256": local_hash,
        "before_ssh_sha256": before_ssh,
        "before_http": before_http,
        "temp_path": tmp,
        "temp_upload_same_dir_verified": True,
        "temp_ssh_sha256": tmp_hash,
        "atomic_rename": True,
        "after_ssh_sha256": after_ssh,
        "after_http": after_http,
    }


def publish_collision_safe(local: Path, remote: str, target_path: str, url: str, tag: str) -> dict:
    local_hash = sha256_file(local)
    exists = ssh_test(remote, target_path)
    actual_path = target_path
    actual_url = url
    if exists:
        existing_hash = ssh_hash(remote, target_path)
        existing_http = http_artifact(url)
        if existing_hash == local_hash and existing_http["sha256"] == local_hash:
            return {
                "target_path": target_path,
                "url": url,
                "mode": "idempotent_existing_identical",
                "target_existed": True,
                "local_sha256": local_hash,
                "before_ssh_sha256": existing_hash,
                "before_http": existing_http,
                "after_ssh_sha256": existing_hash,
                "after_http": existing_http,
            }
        suffix = utc_now().strftime("%Y%m%dT%H%M%SZ")
        actual_path = f"{target_path[:-4]}_{suffix}.png" if target_path.endswith(".png") else f"{target_path}.{suffix}"
        actual_url = f"{url[:-4]}_{suffix}.png" if url.endswith(".png") else f"{url}.{suffix}"
    tmp = f"{str(Path(actual_path).parent)}/.{Path(actual_path).name}.tmp.{tag}.{utc_now().strftime('%Y%m%dT%H%M%SZ')}"
    scp_upload(local, remote, tmp)
    tmp_hash = ssh_hash(remote, tmp)
    if tmp_hash != local_hash:
        raise SystemExit(f"temporary upload hash mismatch for {actual_path}: {tmp_hash} != {local_hash}")
    ssh_mv(remote, tmp, actual_path)
    after_ssh = ssh_hash(remote, actual_path)
    after_http = http_artifact(actual_url)
    if after_ssh != local_hash or after_http["sha256"] != local_hash:
        raise SystemExit(f"publication verification failed for {actual_path}")
    if after_http["status"] != 200 or after_http["content_type"] != "image/png":
        raise SystemExit(f"HTTP verification failed for {actual_url}: {after_http}")
    return {
        "target_path": actual_path,
        "url": actual_url,
        "mode": "publish_new_target_absent" if not exists else "publish_timestamp_suffix_due_collision",
        "target_existed": exists,
        "local_sha256": local_hash,
        "temp_path": tmp,
        "temp_upload_same_dir_verified": True,
        "temp_ssh_sha256": tmp_hash,
        "atomic_rename": True,
        "after_ssh_sha256": after_ssh,
        "after_http": after_http,
    }


def throughput_window(points: list[Point], seconds: int | None) -> dict:
    latest = points[-1]
    if seconds is None:
        obs = points
        label = "since_launch"
    else:
        cutoff = latest.timestamp - dt.timedelta(seconds=seconds)
        obs = [p for p in points if p.timestamp >= cutoff]
        label = f"recent_{seconds // 3600}h"
    if len(obs) < 2:
        raise SystemExit(f"not enough observations for throughput window {label}")
    first = obs[0]
    last = obs[-1]
    elapsed = (last.timestamp - first.timestamp).total_seconds()
    step_delta = last.step - first.step
    token_delta = step_delta * TOKENS_PER_STEP
    return {
        "label": label,
        "bounds_utc": [iso_z(first.timestamp), iso_z(last.timestamp)],
        "bounds_step": [first.step, last.step],
        "bounds_token": [first.token, last.token],
        "sample_count": len(obs),
        "elapsed_seconds": elapsed,
        "step_delta": step_delta,
        "token_delta": token_delta,
        "steps_per_sec": step_delta / elapsed,
        "tokens_per_sec": token_delta / elapsed,
    }


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {secs}s"


def target_eta(current: Point, target: int, rates: dict[str, dict]) -> dict:
    remaining_tokens = target - current.token
    remaining_steps_exact = remaining_tokens / TOKENS_PER_STEP
    step_at_or_above = math.ceil(target / TOKENS_PER_STEP)
    overshoot = step_at_or_above * TOKENS_PER_STEP - target
    per_rate = {}
    for name, rate in rates.items():
        seconds = remaining_steps_exact / rate["steps_per_sec"]
        when = current.timestamp + dt.timedelta(seconds=seconds)
        per_rate[name] = {"duration_seconds": seconds, "duration": format_duration(seconds), "completion_utc": iso_z(when)}
    completions = [parse_time(per_rate[name]["completion_utc"]) for name in ("recent_1h", "since_launch")]
    return {
        "target_tokens": target,
        "target_tokens_billions": target / 1e9,
        "percent_complete": current.token / target * 100.0,
        "remaining_tokens": remaining_tokens,
        "remaining_steps_exact": remaining_steps_exact,
        "step_at_or_above_target": step_at_or_above,
        "token_overshoot_at_step": overshoot,
        "primary_rate": "recent_6h",
        "primary_eta": per_rate["recent_6h"],
        "range_rate_sources": ["recent_1h", "since_launch"],
        "range_completion_utc": [iso_z(min(completions)), iso_z(max(completions))],
        "range_duration": [format_duration((min(completions) - current.timestamp).total_seconds()), format_duration((max(completions) - current.timestamp).total_seconds())],
        "by_rate": per_rate,
    }


def build_report(summary: dict) -> str:
    g = summary["gdn2"]
    c = summary["comparison"]
    t = summary["throughput_eta"]
    pub = summary["publication"]
    lines = [
        f"# GDN2 Ops Snapshot - {summary['task_date_formatted']}",
        "",
        f"- Snapshot UTC: `{summary['snapshot_utc']}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Health: `{summary['health']['status']}`; torchrun PIDs `{summary['health']['torchrun_pids']}`, worker rank PIDs `{summary['health']['worker_rank_pids']}`.",
        f"- Source policy: `{summary['source_policy']}`",
        f"- Token validation: `{summary['token_semantics']['formula']} = {summary['token_semantics']['tokens_per_step']}` aggregate tokens/optimizer step.",
        "",
        "## Current GDN2",
        "",
        f"- Effective records: `{g['records']['effective_points']}` from step `{g['records']['step_range'][0]}` to `{g['records']['step_range'][1]}`; raw records `{g['records']['raw_points']}`, duplicates removed `{g['records']['duplicates_removed']}`.",
        f"- Token range: `{g['records']['token_range'][0]}` to `{g['records']['token_range'][1]}`; current `{g['latest']['tokens']}` (`{g['latest']['tokens_billions']:.9f}B`), `{g['completion_percent_150b']:.9f}%` of 150B.",
        f"- Latest loss at step `{g['latest']['step']}` (`{g['latest']['time_utc']}`): raw `{g['latest']['raw_loss']:.10f}`, smoothed `{g['latest']['smoothed_loss']:.10f}` with trailing MA window `{g['smoothing']['window_points']}`.",
        f"- Last 100 optimizer-step mean: `{g['intervals']['last_100_optimizer_steps']['mean_loss_observed_records']:.10f}` over `{g['intervals']['last_100_optimizer_steps']['observation_count']}` observations, bounds `{g['intervals']['last_100_optimizer_steps']['bounds']}`, cadence `{g['intervals']['last_100_optimizer_steps']['cadence_optimizer_steps']}`, every-step coverage `{g['intervals']['last_100_optimizer_steps']['every_optimizer_step_represented']}`.",
        f"- Last 1000 optimizer-step mean: `{g['intervals']['last_1000_optimizer_steps']['mean_loss_observed_records']:.10f}` over `{g['intervals']['last_1000_optimizer_steps']['observation_count']}` observations, bounds `{g['intervals']['last_1000_optimizer_steps']['bounds']}`, cadence `{g['intervals']['last_1000_optimizer_steps']['cadence_optimizer_steps']}`, every-step coverage `{g['intervals']['last_1000_optimizer_steps']['every_optimizer_step_represented']}`.",
        "",
        "## Matched E97 Comparison",
        "",
        f"- Cutoff: step `{c['cutoff']['step']}`, tokens `{c['cutoff']['tokens']}` (`{c['cutoff']['tokens_billions']:.9f}B`).",
        f"- Alignment: `{c['alignment']['aligned_record_count']}` exact common records, step range `{c['alignment']['common_step_range']}`, token range `{c['alignment']['common_token_range']}`, E97-only `{c['alignment']['missing_or_nonoverlapping']['e97_only_count']}`, GDN2-only `{c['alignment']['missing_or_nonoverlapping']['gdn2_only_count']}`.",
        f"- Raw cutoff losses: E97 `{c['cutoff_raw']['e97_loss']:.10f}`, GDN2 `{c['cutoff_raw']['gdn2_loss']:.10f}`, delta GDN2-E97 `{c['cutoff_raw']['delta_gdn2_minus_e97']:.10f}`.",
        f"- Smoothed cutoff losses: E97 `{c['cutoff_smoothed']['e97_loss']:.10f}`, GDN2 `{c['cutoff_smoothed']['gdn2_loss']:.10f}`, delta `{c['cutoff_smoothed']['delta_gdn2_minus_e97']:.10f}`.",
        f"- Last 100 mean delta: `{c['intervals']['last_100_optimizer_steps']['delta_gdn2_minus_e97']:.10f}`; E97 `{c['intervals']['last_100_optimizer_steps']['e97']['mean_loss_observed_records']:.10f}`, GDN2 `{c['intervals']['last_100_optimizer_steps']['gdn2']['mean_loss_observed_records']:.10f}`.",
        f"- Last 1000 mean delta: `{c['intervals']['last_1000_optimizer_steps']['delta_gdn2_minus_e97']:.10f}`; E97 `{c['intervals']['last_1000_optimizer_steps']['e97']['mean_loss_observed_records']:.10f}`, GDN2 `{c['intervals']['last_1000_optimizer_steps']['gdn2']['mean_loss_observed_records']:.10f}`.",
        f"- Mean signed delta over all aligned observations: `{c['aggregate_delta_metrics']['mean_signed_loss_delta_gdn2_minus_e97_all_aligned_points']:.10f}`; recent 1000-step interval: `{c['aggregate_delta_metrics']['mean_signed_loss_delta_gdn2_minus_e97_recent_1000_step_interval']:.10f}`.",
        "",
        "## Publication",
        "",
        f"- GDN2 plot: `{g['plot']['path']}`, SHA-256 `{g['plot']['sha256']}`, URL `{pub['gdn2_stable']['url']}`.",
        f"- Comparison overlay: `{c['plot']['path']}`, SHA-256 `{c['plot']['sha256']}`, URL `{pub['comparison_overlay']['url']}`.",
        f"- GDN2 stable local/SSH/HTTP hashes: `{pub['gdn2_stable']['local_sha256']}` / `{pub['gdn2_stable']['after_ssh_sha256']}` / `{pub['gdn2_stable']['after_http']['sha256']}`; HTTP `{pub['gdn2_stable']['after_http']['status']}` `{pub['gdn2_stable']['after_http']['content_type']}`.",
        f"- Overlay local/SSH/HTTP hashes: `{pub['comparison_overlay']['local_sha256']}` / `{pub['comparison_overlay']['after_ssh_sha256']}` / `{pub['comparison_overlay']['after_http']['sha256']}`; HTTP `{pub['comparison_overlay']['after_http']['status']}` `{pub['comparison_overlay']['after_http']['content_type']}`.",
        "",
        "## Protected Artifact Hashes",
        "",
    ]
    for name, hashes in summary["protected_artifacts"].items():
        lines.append(
            f"- {name}: before refresh SSH/HTTP `{hashes['before_refresh_ssh_sha256']}` / `{hashes['before_refresh_http_sha256']}`; after GDN2 publish SSH/HTTP `{hashes['after_gdn2_publish_ssh_sha256']}` / `{hashes['after_gdn2_publish_http_sha256']}`; after overlay SSH/HTTP `{hashes['after_overlay_ssh_sha256']}` / `{hashes['after_overlay_http_sha256']}`; unchanged after overlay vs after GDN2 `{hashes['unchanged_after_overlay_vs_after_gdn2']}`."
        )
    lines.extend(
        [
        "",
        "## Throughput and ETA",
        "",
        ]
    )
    for name in ("recent_1h", "recent_6h", "since_launch"):
        rate = t["rates"][name]
        lines.append(
            f"- {name}: bounds `{rate['bounds_utc'][0]}` step `{rate['bounds_step'][0]}` to `{rate['bounds_utc'][1]}` step `{rate['bounds_step'][1]}`; samples `{rate['sample_count']}`, steps/sec `{rate['steps_per_sec']:.9f}`, tokens/sec `{rate['tokens_per_sec']:.3f}`."
        )
    for name, eta in t["targets"].items():
        lines.append(
            f"- {name}: target `{eta['target_tokens']}`, percent `{eta['percent_complete']:.9f}%`, remaining `{eta['remaining_tokens']}` tokens / `{eta['remaining_steps_exact']:.6f}` steps, step at/above `{eta['step_at_or_above_target']}`, overshoot `{eta['token_overshoot_at_step']}` tokens, primary ETA `{eta['primary_eta']['duration']}` ending `{eta['primary_eta']['completion_utc']}`, range `{eta['range_completion_utc'][0]}` to `{eta['range_completion_utc'][1]}`."
        )
    lines.extend(
        [
            "",
            "Assumptions: no downtime, unchanged 8-GPU rate, and 65,536 aggregate tokens per optimizer step.",
            "",
            "No training control, checkpoint write/modification, or S3 command was run.",
        ]
    )
    return "\n".join(lines) + "\n"


def compact_result(summary: dict) -> str:
    g = summary["gdn2"]
    c = summary["comparison"]
    t = summary["throughput_eta"]
    pub = summary["publication"]
    protected = summary["protected_artifacts"]
    return (
        "RESULT: "
        f"snapshot_utc={summary['snapshot_utc']}; run_id={summary['run_id']}; "
        f"run_health={summary['health']['status']} torchrun_pids={summary['health']['torchrun_pids']} launch_wrapper_pids={summary['health']['launch_wrapper_pids']} worker_rank_count={summary['health']['worker_rank_count']} worker_rank_pids={summary['health']['worker_rank_pids']}; "
        f"tokens_per_step={summary['token_semantics']['tokens_per_step']} validated_formula={summary['token_semantics']['formula']}; "
        f"gdn2_snapshot={g['snapshot']['snapshot']} snapshot_sha256={g['snapshot']['sha256']} size={g['snapshot']['snapshot_size_bytes']} source_size_bound={g['snapshot']['source_size_bound_bytes']}; "
        f"gdn2_records raw/effective={g['records']['raw_points']}/{g['records']['effective_points']} duplicates_removed={g['records']['duplicates_removed']} malformed={g['records']['malformed_step_like_lines']} dropped_final_partial={g['records']['dropped_final_partial_line']} finite={g['records']['finite']} monotonic_steps={g['records']['strictly_increasing_steps']} monotonic_tokens={g['records']['strictly_increasing_tokens']} step_range={g['records']['step_range'][0]}..{g['records']['step_range'][1]} token_range={g['records']['token_range'][0]}..{g['records']['token_range'][1]} cadence={g['records']['cadence_optimizer_steps']}; "
        f"current step={g['latest']['step']} tokens={g['latest']['tokens']} tokens_b={g['latest']['tokens_billions']:.9f} percent_150b={g['completion_percent_150b']:.9f} raw_loss={g['latest']['raw_loss']:.10f} smoothed_loss={g['latest']['smoothed_loss']:.10f} smoothing_window={g['smoothing']['window_points']}; "
        f"gdn2_last100 bounds={g['intervals']['last_100_optimizer_steps']['bounds'][0]}..{g['intervals']['last_100_optimizer_steps']['bounds'][1]} obs={g['intervals']['last_100_optimizer_steps']['observation_count']} first_last={g['intervals']['last_100_optimizer_steps']['first_observed_step']}..{g['intervals']['last_100_optimizer_steps']['last_observed_step']} cadence={g['intervals']['last_100_optimizer_steps']['cadence_optimizer_steps']} every_step={g['intervals']['last_100_optimizer_steps']['every_optimizer_step_represented']} mean={g['intervals']['last_100_optimizer_steps']['mean_loss_observed_records']:.10f} no_interpolation; "
        f"gdn2_last1000 bounds={g['intervals']['last_1000_optimizer_steps']['bounds'][0]}..{g['intervals']['last_1000_optimizer_steps']['bounds'][1]} obs={g['intervals']['last_1000_optimizer_steps']['observation_count']} first_last={g['intervals']['last_1000_optimizer_steps']['first_observed_step']}..{g['intervals']['last_1000_optimizer_steps']['last_observed_step']} cadence={g['intervals']['last_1000_optimizer_steps']['cadence_optimizer_steps']} every_step={g['intervals']['last_1000_optimizer_steps']['every_optimizer_step_represented']} mean={g['intervals']['last_1000_optimizer_steps']['mean_loss_observed_records']:.10f} no_interpolation; "
        f"gdn2_plot_local={g['plot']['path']} sha256={g['plot']['sha256']} size={g['plot']['size_bytes']} shape={g['plot']['image_shape']} stable_url={pub['gdn2_stable']['url']} ssh_hash={pub['gdn2_stable']['after_ssh_sha256']} http_hash={pub['gdn2_stable']['after_http']['sha256']} http_status={pub['gdn2_stable']['after_http']['status']} content_type={pub['gdn2_stable']['after_http']['content_type']} atomic_rename={pub['gdn2_stable'].get('atomic_rename')}; "
        f"comparison cutoff_step={c['cutoff']['step']} cutoff_tokens={c['cutoff']['tokens']} aligned_count={c['alignment']['aligned_record_count']} common_step_range={c['alignment']['common_step_range'][0]}..{c['alignment']['common_step_range'][1]} common_token_range={c['alignment']['common_token_range'][0]}..{c['alignment']['common_token_range'][1]} e97_only={c['alignment']['missing_or_nonoverlapping']['e97_only_count']} gdn2_only={c['alignment']['missing_or_nonoverlapping']['gdn2_only_count']}; "
        f"e97_records raw/effective={c['series']['e97']['raw_points']}/{c['series']['e97']['effective_points']} duplicates_removed={c['series']['e97']['duplicates_removed']} malformed={c['series']['e97']['malformed_step_like_lines']} finite={c['series']['e97']['finite']} monotonic_steps={c['series']['e97']['strictly_increasing_steps']} monotonic_tokens={c['series']['e97']['strictly_increasing_tokens']}; "
        f"comparison_cutoff_raw E97={c['cutoff_raw']['e97_loss']:.10f} GDN2={c['cutoff_raw']['gdn2_loss']:.10f} delta={c['cutoff_raw']['delta_gdn2_minus_e97']:.10f}; "
        f"comparison_cutoff_smoothed E97={c['cutoff_smoothed']['e97_loss']:.10f} GDN2={c['cutoff_smoothed']['gdn2_loss']:.10f} delta={c['cutoff_smoothed']['delta_gdn2_minus_e97']:.10f}; "
        f"comparison_last100 bounds={c['intervals']['last_100_optimizer_steps']['bounds'][0]}..{c['intervals']['last_100_optimizer_steps']['bounds'][1]} obs_e97={c['intervals']['last_100_optimizer_steps']['e97']['observation_count']} obs_gdn2={c['intervals']['last_100_optimizer_steps']['gdn2']['observation_count']} E97_mean={c['intervals']['last_100_optimizer_steps']['e97']['mean_loss_observed_records']:.10f} GDN2_mean={c['intervals']['last_100_optimizer_steps']['gdn2']['mean_loss_observed_records']:.10f} delta={c['intervals']['last_100_optimizer_steps']['delta_gdn2_minus_e97']:.10f} cadence={c['intervals']['last_100_optimizer_steps']['gdn2']['cadence_optimizer_steps']} no_interpolation; "
        f"comparison_last1000 bounds={c['intervals']['last_1000_optimizer_steps']['bounds'][0]}..{c['intervals']['last_1000_optimizer_steps']['bounds'][1]} obs_e97={c['intervals']['last_1000_optimizer_steps']['e97']['observation_count']} obs_gdn2={c['intervals']['last_1000_optimizer_steps']['gdn2']['observation_count']} E97_mean={c['intervals']['last_1000_optimizer_steps']['e97']['mean_loss_observed_records']:.10f} GDN2_mean={c['intervals']['last_1000_optimizer_steps']['gdn2']['mean_loss_observed_records']:.10f} delta={c['intervals']['last_1000_optimizer_steps']['delta_gdn2_minus_e97']:.10f} cadence={c['intervals']['last_1000_optimizer_steps']['gdn2']['cadence_optimizer_steps']} no_interpolation; "
        f"mean_signed_delta_all={c['aggregate_delta_metrics']['mean_signed_loss_delta_gdn2_minus_e97_all_aligned_points']:.10f} mean_signed_delta_recent1000={c['aggregate_delta_metrics']['mean_signed_loss_delta_gdn2_minus_e97_recent_1000_step_interval']:.10f}; "
        f"overlay_local={c['plot']['path']} sha256={c['plot']['sha256']} size={c['plot']['size_bytes']} shape={c['plot']['image_shape']} overlay_url={pub['comparison_overlay']['url']} mode={pub['comparison_overlay']['mode']} ssh_hash={pub['comparison_overlay']['after_ssh_sha256']} http_hash={pub['comparison_overlay']['after_http']['sha256']} http_status={pub['comparison_overlay']['after_http']['status']} content_type={pub['comparison_overlay']['after_http']['content_type']}; "
        f"throughput_1h bounds={t['rates']['recent_1h']['bounds_utc'][0]} step={t['rates']['recent_1h']['bounds_step'][0]}..{t['rates']['recent_1h']['bounds_utc'][1]} step={t['rates']['recent_1h']['bounds_step'][1]} samples={t['rates']['recent_1h']['sample_count']} steps_per_sec={t['rates']['recent_1h']['steps_per_sec']:.9f} tokens_per_sec={t['rates']['recent_1h']['tokens_per_sec']:.3f}; "
        f"throughput_6h bounds={t['rates']['recent_6h']['bounds_utc'][0]} step={t['rates']['recent_6h']['bounds_step'][0]}..{t['rates']['recent_6h']['bounds_utc'][1]} step={t['rates']['recent_6h']['bounds_step'][1]} samples={t['rates']['recent_6h']['sample_count']} steps_per_sec={t['rates']['recent_6h']['steps_per_sec']:.9f} tokens_per_sec={t['rates']['recent_6h']['tokens_per_sec']:.3f}; "
        f"throughput_since_launch bounds={t['rates']['since_launch']['bounds_utc'][0]} step={t['rates']['since_launch']['bounds_step'][0]}..{t['rates']['since_launch']['bounds_utc'][1]} step={t['rates']['since_launch']['bounds_step'][1]} samples={t['rates']['since_launch']['sample_count']} steps_per_sec={t['rates']['since_launch']['steps_per_sec']:.9f} tokens_per_sec={t['rates']['since_launch']['tokens_per_sec']:.3f}; "
        f"eta_150b percent={t['targets']['target_150b']['percent_complete']:.9f} remaining_tokens={t['targets']['target_150b']['remaining_tokens']} remaining_steps={t['targets']['target_150b']['remaining_steps_exact']:.6f} step_at_or_above={t['targets']['target_150b']['step_at_or_above_target']} overshoot={t['targets']['target_150b']['token_overshoot_at_step']} primary_duration={t['targets']['target_150b']['primary_eta']['duration']} primary_utc={t['targets']['target_150b']['primary_eta']['completion_utc']} range_utc={t['targets']['target_150b']['range_completion_utc'][0]}..{t['targets']['target_150b']['range_completion_utc'][1]}; "
        f"eta_e97_parity percent={t['targets']['target_e97_parity']['percent_complete']:.9f} target=150793748480 remaining_tokens={t['targets']['target_e97_parity']['remaining_tokens']} remaining_steps={t['targets']['target_e97_parity']['remaining_steps_exact']:.6f} step_at_or_above={t['targets']['target_e97_parity']['step_at_or_above_target']} overshoot={t['targets']['target_e97_parity']['token_overshoot_at_step']} primary_duration={t['targets']['target_e97_parity']['primary_eta']['duration']} primary_utc={t['targets']['target_e97_parity']['primary_eta']['completion_utc']} range_utc={t['targets']['target_e97_parity']['range_completion_utc'][0]}..{t['targets']['target_e97_parity']['range_completion_utc'][1]}; "
        f"protected_artifacts={json.dumps(protected, sort_keys=True)}; "
        "assumptions=no_downtime,unchanged_8gpu_rate,65536_tokens_per_step; no_training_control=true no_checkpoint_write_or_modification=true no_s3_command=true"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp"))
    ap.add_argument("--run-dir", type=Path, default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444"))
    ap.add_argument("--e97-root", type=Path, default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/emender"))
    ap.add_argument("--e97-args", type=Path, default=Path("/mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260722_055730/args.json"))
    ap.add_argument("--remote", default="erik@hypervolu.me")
    ap.add_argument("--task-date", default="20260729", help="YYYYMMDD date stamp for task-specific ops artifacts")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    snapshot_time = utc_now()
    stamp = snapshot_time.strftime("%Y%m%dT%H%M%SZ")
    task_date = args.task_date
    if not re.fullmatch(r"\d{8}", task_date):
        raise SystemExit(f"--task-date must be YYYYMMDD, got {task_date!r}")
    task_date_formatted = f"{task_date[:4]}-{task_date[4:6]}-{task_date[6:]}"
    if args.report is None:
        args.report = Path(f"docs/GDN2_OPS_SNAPSHOT_{task_date}.md")
    ops_dir = args.run_root / "ops" / f"refresh-gdn2-ops_{task_date}_{stamp}"
    snapshots_dir = ops_dir / "snapshots"
    ops_dir.mkdir(parents=True, exist_ok=False)

    run_id = load_json(args.run_root / "launch_manifest.json")["name"]
    health_before = inspect_health(args.run_root)
    token_semantics = token_config(args.run_dir / "args.json")
    e97_token_semantics = token_config(args.e97_args)

    protected_names = {
        "E97_standalone": ("www/emender/e97_diloco_loss_curve_20260623.png", "http://hypervolu.me/~erik/emender/e97_diloco_loss_curve_20260623.png"),
        "GDN2_standalone": ("www/emender/gdn2_mlp_diloco_loss_curve_20260722.png", "http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png"),
        "prior_5p118B_comparison": ("www/emender/gdn2_vs_e97_matched_5p118b_tokens_20260723.png", "http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_5p118b_tokens_20260723.png"),
        "prior_20260724_comparison": ("www/emender/gdn2_vs_e97_matched_tokens_20260724.png", "http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260724.png"),
        "prior_20260728_comparison": ("www/emender/gdn2_vs_e97_matched_tokens_20260728.png", "http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_20260728.png"),
    }
    protected_before = {
        name: {"ssh_sha256": ssh_hash(args.remote, path), "http": http_artifact(url), "path": path, "url": url}
        for name, (path, url) in protected_names.items()
    }

    gdn2_snapshot = snapshot_prefix(args.run_root / "run.log", snapshots_dir / f"run.log.snapshot.prefix_{stamp}.log")
    gdn2_raw, _, gdn2_dropped_parse, gdn2_malformed = parse_log(Path(gdn2_snapshot["snapshot"]), Path(gdn2_snapshot["snapshot"]).name)
    gdn2_points, gdn2_superseded = dedupe(gdn2_raw)
    assert_series(gdn2_points, "GDN2")
    cutoff_step = gdn2_points[-1].step
    cutoff_tokens = gdn2_points[-1].token
    window = min(80, max(5, len(gdn2_points) // 40))
    gdn2_smoothed = moving_average([p.loss for p in gdn2_points], window)
    gdn2_plot = plot_gdn2(gdn2_points, gdn2_smoothed, window, ops_dir / f"gdn2_mlp_diloco_loss_curve_{stamp}.png")

    gdn2_records = series_summary(gdn2_raw, gdn2_points, gdn2_superseded, [gdn2_snapshot["dropped_final_partial_line"], gdn2_dropped_parse], gdn2_malformed)
    latest = gdn2_points[-1]
    gdn2_summary = {
        "snapshot": gdn2_snapshot,
        "records": gdn2_records,
        "smoothing": {"method": "trailing moving average over effective plotted loss records", "window_formula": "min(80,max(5,n//40))", "window_points": window},
        "latest": {"step": latest.step, "tokens": latest.token, "tokens_billions": latest.token / 1e9, "raw_loss": latest.loss, "smoothed_loss": gdn2_smoothed[-1], "time_utc": iso_z(latest.timestamp)},
        "completion_percent_150b": latest.token / TARGET_150B * 100.0,
        "intervals": {"last_100_optimizer_steps": interval_summary(gdn2_points, latest.step, 100), "last_1000_optimizer_steps": interval_summary(gdn2_points, latest.step, 1000)},
        "plot": gdn2_plot,
    }

    e97_sources = sorted(args.e97_root.glob("run*.log"), key=lambda p: (p.stat().st_mtime, p.name))
    if not e97_sources:
        raise SystemExit("no E97 run*.log files found")
    e97_snapshots = [copy_snapshot(path, snapshots_dir / "e97" / path.name) for path in e97_sources]
    e97_raw: list[Point] = []
    e97_dropped: list[bool] = []
    e97_malformed: list[dict] = []
    order = 0
    for meta in e97_snapshots:
        parsed, order, dropped, malformed = parse_log(Path(meta["snapshot"]), Path(meta["snapshot"]).name, order)
        e97_raw.extend(parsed)
        e97_dropped.append(dropped)
        e97_malformed.extend(malformed)
    e97_points, e97_superseded = dedupe(e97_raw, cutoff_step)
    gdn2_cutoff_points, gdn2_cutoff_superseded = dedupe(gdn2_raw, cutoff_step)
    assert_series(e97_points, "E97")
    assert_series(gdn2_cutoff_points, "GDN2 cutoff")
    e97_by_step = {p.step: p for p in e97_points}
    gdn2_by_step = {p.step: p for p in gdn2_cutoff_points}
    common_steps = sorted(set(e97_by_step) & set(gdn2_by_step))
    if common_steps[-1] != cutoff_step:
        raise SystemExit(f"latest common step {common_steps[-1]} does not match cutoff step {cutoff_step}")
    aligned = [
        {
            "step": step,
            "token": step * TOKENS_PER_STEP,
            "e97_loss": e97_by_step[step].loss,
            "gdn2_loss": gdn2_by_step[step].loss,
            "delta_gdn2_minus_e97": gdn2_by_step[step].loss - e97_by_step[step].loss,
        }
        for step in common_steps
    ]
    e97_aligned = [e97_by_step[step] for step in common_steps]
    gdn2_aligned = [gdn2_by_step[step] for step in common_steps]
    e97_smoothed = moving_average([p.loss for p in e97_aligned], window)
    gdn2_aligned_smoothed = moving_average([p.loss for p in gdn2_aligned], window)
    overlay_plot = plot_overlay(aligned, e97_smoothed, gdn2_aligned_smoothed, window, ops_dir / f"gdn2_vs_e97_matched_tokens_{task_date}.png")
    comparison_intervals = {}
    for width in (100, 1000):
        e97_int = interval_summary(e97_aligned, cutoff_step, width)
        gdn2_int = interval_summary(gdn2_aligned, cutoff_step, width)
        comparison_intervals[f"last_{width}_optimizer_steps"] = {
            "bounds": [cutoff_step - width + 1, cutoff_step],
            "e97": e97_int,
            "gdn2": gdn2_int,
            "delta_gdn2_minus_e97": gdn2_int["mean_loss_observed_records"] - e97_int["mean_loss_observed_records"],
        }
    recent_1000 = [row for row in aligned if cutoff_step - 999 <= row["step"] <= cutoff_step]
    comparison = {
        "cutoff": {"step": cutoff_step, "tokens": cutoff_tokens, "tokens_billions": cutoff_tokens / 1e9},
        "series": {
            "e97": series_summary([p for p in e97_raw if p.step <= cutoff_step], e97_points, e97_superseded, e97_dropped, e97_malformed),
            "gdn2": series_summary([p for p in gdn2_raw if p.step <= cutoff_step], gdn2_cutoff_points, gdn2_cutoff_superseded, [gdn2_snapshot["dropped_final_partial_line"], gdn2_dropped_parse], gdn2_malformed),
        },
        "alignment": {
            "aligned_record_count": len(aligned),
            "common_step_range": [common_steps[0], common_steps[-1]],
            "common_token_range": [common_steps[0] * TOKENS_PER_STEP, common_steps[-1] * TOKENS_PER_STEP],
            "missing_or_nonoverlapping": {
                "e97_only_count": len(set(e97_by_step) - set(gdn2_by_step)),
                "gdn2_only_count": len(set(gdn2_by_step) - set(e97_by_step)),
                "e97_only_steps_through_cutoff": sorted(set(e97_by_step) - set(gdn2_by_step)),
                "gdn2_only_steps_through_cutoff": sorted(set(gdn2_by_step) - set(e97_by_step)),
            },
        },
        "smoothing": {"method": "trailing moving average over aligned effective records", "window_aligned_records": window, "window_reason": f"GDN2 effective count {len(gdn2_cutoff_points)} gives min(80,max(5,n//40))={window}; applied identically to E97."},
        "cutoff_raw": {"step": cutoff_step, "token": cutoff_tokens, "e97_loss": e97_by_step[cutoff_step].loss, "gdn2_loss": gdn2_by_step[cutoff_step].loss, "delta_gdn2_minus_e97": gdn2_by_step[cutoff_step].loss - e97_by_step[cutoff_step].loss},
        "cutoff_smoothed": {"step": cutoff_step, "token": cutoff_tokens, "window_aligned_records": window, "e97_loss": e97_smoothed[-1], "gdn2_loss": gdn2_aligned_smoothed[-1], "delta_gdn2_minus_e97": gdn2_aligned_smoothed[-1] - e97_smoothed[-1]},
        "intervals": comparison_intervals,
        "aggregate_delta_metrics": {
            "mean_signed_loss_delta_gdn2_minus_e97_all_aligned_points": sum(row["delta_gdn2_minus_e97"] for row in aligned) / len(aligned),
            "mean_signed_loss_delta_gdn2_minus_e97_recent_1000_step_interval": sum(row["delta_gdn2_minus_e97"] for row in recent_1000) / len(recent_1000),
        },
        "plot": overlay_plot,
        "snapshots": {"e97": e97_snapshots, "gdn2": gdn2_snapshot},
    }

    rates = {
        "recent_1h": throughput_window(gdn2_points, 3600),
        "recent_6h": throughput_window(gdn2_points, 21_600),
        "since_launch": throughput_window(gdn2_points, None),
    }
    throughput_eta = {
        "rates": rates,
        "targets": {
            "target_150b": target_eta(latest, TARGET_150B, rates),
            "target_e97_parity": target_eta(latest, TARGET_E97_PARITY, rates),
        },
        "assumptions": ["no downtime", "unchanged 8-GPU rate", "65536 aggregate tokens per optimizer step"],
    }

    gdn2_pub = publish_overwrite(
        Path(gdn2_plot["path"]),
        args.remote,
        "www/emender/gdn2_mlp_diloco_loss_curve_20260722.png",
        "http://hypervolu.me/~erik/emender/gdn2_mlp_diloco_loss_curve_20260722.png",
        f"refresh-gdn2-ops-{task_date}",
    )
    protected_after_gdn2 = {
        name: {"ssh_sha256": ssh_hash(args.remote, path), "http": http_artifact(url), "path": path, "url": url}
        for name, (path, url) in protected_names.items()
    }
    overlay_pub = publish_collision_safe(
        Path(overlay_plot["path"]),
        args.remote,
        f"www/emender/gdn2_vs_e97_matched_tokens_{task_date}.png",
        f"http://hypervolu.me/~erik/emender/gdn2_vs_e97_matched_tokens_{task_date}.png",
        f"refresh-gdn2-ops-{task_date}",
    )
    protected_after_overlay = {
        name: {"ssh_sha256": ssh_hash(args.remote, path), "http": http_artifact(url), "path": path, "url": url}
        for name, (path, url) in protected_names.items()
    }
    health_after = inspect_health(args.run_root)

    protected = {}
    for name in protected_names:
        protected[name] = {
            "before_refresh_ssh_sha256": protected_before[name]["ssh_sha256"],
            "before_refresh_http_sha256": protected_before[name]["http"]["sha256"],
            "after_gdn2_publish_ssh_sha256": protected_after_gdn2[name]["ssh_sha256"],
            "after_gdn2_publish_http_sha256": protected_after_gdn2[name]["http"]["sha256"],
            "after_overlay_ssh_sha256": protected_after_overlay[name]["ssh_sha256"],
            "after_overlay_http_sha256": protected_after_overlay[name]["http"]["sha256"],
            "unchanged_after_overlay_vs_after_gdn2": protected_after_overlay[name]["ssh_sha256"] == protected_after_gdn2[name]["ssh_sha256"] == protected_after_overlay[name]["http"]["sha256"] == protected_after_gdn2[name]["http"]["sha256"],
        }
    protected["E97_standalone"]["unchanged_from_initial"] = (
        protected["E97_standalone"]["before_refresh_ssh_sha256"]
        == protected["E97_standalone"]["after_gdn2_publish_ssh_sha256"]
        == protected["E97_standalone"]["after_overlay_ssh_sha256"]
        == protected["E97_standalone"]["before_refresh_http_sha256"]
        == protected["E97_standalone"]["after_gdn2_publish_http_sha256"]
        == protected["E97_standalone"]["after_overlay_http_sha256"]
    )

    summary = {
        "snapshot_utc": iso_z(snapshot_time),
        "task_date": task_date,
        "task_date_formatted": task_date_formatted,
        "run_id": run_id,
        "source_policy": "rank-0/main stdout, finite complete records only, dedupe by optimizer step keeping latest timestamp/order, no interpolation",
        "token_semantics": token_semantics,
        "e97_token_semantics": e97_token_semantics,
        "health": health_after,
        "health_before": health_before,
        "gdn2": gdn2_summary,
        "comparison": comparison,
        "throughput_eta": throughput_eta,
        "publication": {"gdn2_stable": gdn2_pub, "comparison_overlay": overlay_pub},
        "protected_artifacts": protected,
        "confirmations": {"no_training_control": True, "no_checkpoint_write_or_modification": True, "no_s3_command": True},
    }
    summary_path = ops_dir / f"refresh_gdn2_ops_{task_date}_summary.json"
    summary["summary_json"] = str(summary_path)
    summary["report"] = str(args.report)
    summary["result_log"] = compact_result(summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(args.report), "result_log": summary["result_log"]}, indent=2))


if __name__ == "__main__":
    main()
