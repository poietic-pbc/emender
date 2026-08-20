#!/usr/bin/env python3
"""Evaluate held-out discoverable CLI tasks through real Pi and Apptainer."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ndm.e97_agent_protocol import DENSE_AGENT_CLI_SYSTEM


def load_tasks(authority: Path) -> list[dict[str, Any]]:
    tasks = []
    for line in (authority / "records.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["split"]:
            tasks.append({"id": row["id"], "kind": row["source"].removeprefix("emender-agent-cli-").removesuffix("-v1"), "user": row["user"], **row["task"]})
    return tasks


def balanced_prefix(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a deterministic round-robin prefix across kind and discovery strata."""
    strata: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for task in tasks:
        strata.setdefault((task["kind"], bool(task["discover"])), []).append(task)
    keys = sorted(strata)
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            rows = strata[key]
            if position < len(rows):
                selected.append(rows[position]); added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        position += 1
    return selected


def make_sandbox(root: Path, task: dict[str, Any]) -> Path:
    sandbox = root / task["id"]
    sandbox.mkdir(parents=True, exist_ok=False)
    for fixture in task["fixtures"]:
        path = sandbox / fixture["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fixture["content"])
    return sandbox


def events(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value, dict): rows.append(value)
    return rows


def score(task: dict[str, Any], rows: list[dict[str, Any]], returncode: int) -> dict[str, Any]:
    starts = [row for row in rows if row.get("type") == "tool_execution_start"]
    ends = [row for row in rows if row.get("type") == "tool_execution_end"]
    sequence = [row.get("toolName") for row in starts]
    cli_calls = [row for row in starts if row.get("toolName") == "cli"]
    expected_cli = {"argv": task["argv"]}
    operation = next((row for row in cli_calls if row.get("args", {}).get("argv") == task["argv"]), None)
    submits = [row for row in starts if row.get("toolName") == "submit_answer"]
    submit_ids = {row.get("toolCallId") for row in submits}
    submit_ends = [row for row in ends if row.get("toolCallId") in submit_ids]
    expected_sequence = (["cli"] if task["discover"] else []) + ["cli", "submit_answer"]
    checks = {
        "pi_exit_zero": returncode == 0,
        "bounded_sequence": sequence == expected_sequence,
        "operation_argv": operation is not None,
        "submitted_value": len(submits) == 1 and submits[0].get("args", {}).get("value") == task["expected"],
        "submitted_evidence": len(submits) == 1 and submits[0].get("args", {}).get("evidence") == task["evidence"],
        "grounded_submit": len(submit_ends) == 1 and submit_ends[0].get("isError") is False,
        "agent_completed": sum(row.get("type") == "agent_end" for row in rows) == 1,
    }
    return {"id": task["id"], "kind": task["kind"], "discover": task["discover"], "expected": task["expected"], "expected_argv": expected_cli, "status": "pass" if all(checks.values()) else "fail", "checks": checks, "tool_calls": [{"name": row.get("toolName"), "args": row.get("args")} for row in starts]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-mode", choices=("all", "direct", "discovery"), default="all")
    parser.add_argument("--pi-config-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    args = parser.parse_args()
    tasks = load_tasks(args.authority_root)
    if args.task_mode == "direct":
        tasks = [task for task in tasks if not task["discover"]]
    elif args.task_mode == "discovery":
        tasks = [task for task in tasks if task["discover"]]
    if args.limit: tasks = balanced_prefix(tasks, args.limit)
    tasks = [task for position, task in enumerate(tasks) if position % args.world_size == args.rank]
    shard = args.output_root / f"rank-{args.rank:02d}"
    for child in ("sandboxes", "traces", "results"): (shard / child).mkdir(parents=True, exist_ok=False)
    results = []
    for task in tasks:
        sandbox = make_sandbox(shard / "sandboxes", task)
        environment = dict(os.environ)
        environment.update({"PI_CODING_AGENT_DIR": str(args.pi_config_dir), "PI_CODING_AGENT_SESSION_DIR": str(shard / "sessions" / task["id"]), "PI_OFFLINE": "1"})
        command = ["pi", "--mode", "json", "--provider", "emender-local", "--model", "e97-dense-agent", "--no-session", "--no-builtin-tools", "--no-skills", "--no-context-files", "-e", str(args.extension), "--system-prompt", DENSE_AGENT_CLI_SYSTEM, "--approve", task["user"]]
        try:
            completed = subprocess.run(command, cwd=sandbox, env=environment, text=True, capture_output=True, timeout=args.timeout_seconds, check=False)
            stdout, stderr, rc = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""; stderr = (exc.stderr if isinstance(exc.stderr, str) else "") + "\nTIMEOUT\n"; rc = 124
        (shard / "traces" / f"{task['id']}.jsonl").write_text(stdout); (shard / "traces" / f"{task['id']}.stderr").write_text(stderr)
        result = score(task, events(stdout), rc); (shard / "results" / f"{task['id']}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); results.append(result)
        print(json.dumps({"id": task["id"], "status": result["status"]}), flush=True)
    summary = {"schema": "emender-e97-pi-cli-shard-v1", "rank": args.rank, "tasks": len(results), "passed": sum(row["status"] == "pass" for row in results)}
    (shard / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
