#!/usr/bin/env python3
"""Run held-out RS-free dense-agent v2 tasks through the real Pi CLI."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

from scripts.build_e97_dense_agent_sft_v2 import SYSTEM, split, trace


def validation_tasks(seed: int) -> list[dict[str, Any]]:
    tasks = []
    for index in range(30_000):
        kind = ("calculator", "lookup", "count")[index % 3]
        identity = f"agent-v2-{kind}-{index:08d}"
        if not split(identity):
            continue
        user, turns = trace(kind, index, random.Random(seed + index))
        first_action = turns[0][1]
        first_name = first_action.splitlines()[0].removeprefix("Action: ")
        first_args = json.loads(first_action.partition("\nArguments: ")[2])
        observation = json.loads(turns[1][1])
        submit_args = json.loads(turns[2][1].partition("\nArguments: ")[2])
        tasks.append({
            "id": identity,
            "index": index,
            "kind": kind,
            "user": user,
            "first_tool": first_name,
            "first_args": first_args,
            "observation": observation,
            "expected": submit_args["value"],
        })
    if len(tasks) != 300:
        raise RuntimeError(f"expected 300 validation tasks, found {len(tasks)}")
    return tasks


def make_sandbox(root: Path, task: dict[str, Any]) -> Path:
    sandbox = root / task["id"]
    sandbox.mkdir(parents=True, exist_ok=False)
    if task["kind"] == "lookup":
        path = sandbox / task["first_args"]["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        field = task["first_args"]["field"]
        path.write_text(f"record has {field} {task['expected']}.\n")
    elif task["kind"] == "count":
        directory = sandbox / task["first_args"]["path"]
        directory.mkdir(parents=True, exist_ok=True)
        suffix = task["first_args"]["suffix"]
        for item in range(int(task["expected"])):
            (directory / f"item-{item:03d}{suffix}").write_text("fixture\n")
        (directory / "distractor.other").write_text("fixture\n")
    return sandbox


def load_events(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def score(task: dict[str, Any], events: list[dict[str, Any]], returncode: int) -> dict[str, Any]:
    starts = [event for event in events if event.get("type") == "tool_execution_start"]
    ends = [event for event in events if event.get("type") == "tool_execution_end"]
    first = starts[0] if starts else {}
    submits = [event for event in starts if event.get("toolName") == "submit_answer"]
    submit_ids = {event.get("toolCallId") for event in submits}
    submit_ends = [event for event in ends if event.get("toolCallId") in submit_ids]
    checks = {
        "pi_exit_zero": returncode == 0,
        "first_tool": first.get("toolName") == task["first_tool"],
        "first_arguments": first.get("args") == task["first_args"],
        "bounded_turns": len(starts) <= 2,
        "one_submit": len(submits) == 1,
        "submitted_value": len(submits) == 1 and submits[0].get("args") == {"value": task["expected"]},
        "grounded_submit": len(submit_ends) == 1 and submit_ends[0].get("isError") is False,
        "agent_completed": sum(event.get("type") == "agent_end" for event in events) == 1,
    }
    return {
        **task,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tool_calls": [{"name": event.get("toolName"), "args": event.get("args")} for event in starts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=9702)
    parser.add_argument("--limit", type=int, default=0, help="Global validation-task limit; zero runs all 300")
    parser.add_argument("--pi-config-dir", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank/world size")
    tasks = validation_tasks(args.seed)
    if args.limit:
        tasks = tasks[: args.limit]
    tasks = [task for position, task in enumerate(tasks) if position % args.world_size == args.rank]
    shard_root = args.output_root / f"rank-{args.rank:02d}"
    sandbox_root = shard_root / "sandboxes"
    trace_root = shard_root / "traces"
    result_root = shard_root / "results"
    for path in (sandbox_root, trace_root, result_root):
        path.mkdir(parents=True, exist_ok=False)
    results = []
    for task in tasks:
        sandbox = make_sandbox(sandbox_root, task)
        env = dict(os.environ)
        env.update({
            "PI_CODING_AGENT_DIR": str(args.pi_config_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(shard_root / "sessions" / task["id"]),
            "PI_OFFLINE": "1",
        })
        command = [
            "pi", "--mode", "json", "--provider", "emender-local", "--model", "e97-dense-agent",
            "--no-session", "--no-builtin-tools", "--no-skills", "--no-context-files",
            "-e", str(args.extension), "--system-prompt", SYSTEM, "--approve", task["user"],
        ]
        try:
            completed = subprocess.run(
                command, cwd=sandbox, env=env, text=True, capture_output=True,
                timeout=args.timeout_seconds, check=False,
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stderr += "\nTIMEOUT\n"
            returncode = 124
        (trace_root / f"{task['id']}.jsonl").write_text(stdout)
        (trace_root / f"{task['id']}.stderr").write_text(stderr)
        result = score(task, load_events(stdout), returncode)
        (result_root / f"{task['id']}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(json.dumps({"id": task["id"], "kind": task["kind"], "status": result["status"]}), flush=True)
    passed = sum(result["status"] == "pass" for result in results)
    summary = {"schema": "emender-e97-dense-agent-pi-v2-shard", "rank": args.rank, "tasks": len(results), "passed": passed, "status": "pass" if passed == len(results) else "fail"}
    (shard_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
