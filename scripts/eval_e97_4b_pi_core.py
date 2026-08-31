#!/usr/bin/env python3
"""Evaluate held-out Pi core-tool tasks through the real Pi agent loop."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from ndm.e97_agent_protocol import E97_PI_CORE_SYSTEM


def load_tasks(authority: Path) -> list[dict[str, Any]]:
    tasks = []
    with (authority / "records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"]:
                tasks.append(row)
    return tasks


def balanced_prefix(tasks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        groups.setdefault(task["kind"], []).append(task)
    selected, position = [], 0
    while len(selected) < limit:
        added = False
        for kind in sorted(groups):
            if position < len(groups[kind]):
                selected.append(groups[kind][position])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            return selected
        position += 1
    return selected


def make_sandbox(root: Path, task: dict[str, Any]) -> Path:
    path = root / task["id"]
    path.mkdir(parents=True, exist_ok=False)
    for fixture in task["task"]["fixtures"]:
        destination = path / fixture["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(fixture["content"])
    return path


def events(text: str) -> list[dict[str, Any]]:
    values = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def expected_calls(task: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    kind = task["kind"]
    fixture = task["task"]["fixtures"][0] if task["task"]["fixtures"] else None
    verifier = task["task"]["verifier"]
    path = fixture["path"] if fixture else verifier.get("path")
    if kind == "read":
        return [("read", {"path": path, "offset": 1, "limit": 80})]
    if kind == "bash":
        return [("bash", {"command": verifier["command"]})]
    if kind == "edit":
        old = fixture["content"].rstrip("\n")
        return [("read", {"path": path, "offset": 1, "limit": 40}),
                ("edit", {"path": path, "oldText": old, "newText": verifier["contains"]}),
                ("bash", {"command": verifier["command"]})]
    if kind == "write":
        return [("write", {"path": verifier["path"], "content": verifier["content"]}),
                ("bash", {"command": verifier["command"]})]
    if kind == "recover-read":
        wrong = path.replace("authority-", "authorities-")
        return [("read", {"path": wrong, "offset": 1, "limit": 40}),
                ("read", {"path": path, "offset": 1, "limit": 40})]
    if kind == "recover-test":
        old = fixture["content"].rstrip("\n")
        return [("bash", {"command": verifier["command"]}),
                ("read", {"path": path, "offset": 1, "limit": 40}),
                ("edit", {"path": path, "oldText": old, "newText": verifier["contains"]}),
                ("bash", {"command": verifier["command"]})]
    raise ValueError(kind)


def verify_sandbox(sandbox: Path, task: dict[str, Any]) -> bool:
    verifier = task["task"]["verifier"]
    kind = verifier["kind"]
    if kind in {"exact_text", "recovery_exact_text"}:
        return True
    path = sandbox / verifier["path"]
    if not path.is_file():
        return False
    if "content" in verifier and path.read_text() != verifier["content"]:
        return False
    if "contains" in verifier and verifier["contains"] not in path.read_text():
        return False
    completed = subprocess.run(
        verifier["command"], cwd=sandbox, shell=True, executable="/bin/bash",
        text=True, capture_output=True, timeout=30, check=False)
    return completed.returncode == 0


def score(task: dict[str, Any], rows: list[dict[str, Any]], returncode: int,
          sandbox: Path) -> dict[str, Any]:
    starts = [row for row in rows if row.get("type") == "tool_execution_start"]
    observed = [(str(row.get("toolName")), row.get("args")) for row in starts]
    expected = expected_calls(task)
    sequence = [name for name, _ in observed]
    repeated = any(observed[index] == observed[index - 1] for index in range(1, len(observed)))
    checks = {
        "pi_exit_zero": returncode == 0,
        "tool_sequence": sequence == [name for name, _ in expected],
        "tool_arguments": observed == expected,
        "sandbox_postcondition": verify_sandbox(sandbox, task),
        "agent_completed": sum(row.get("type") == "agent_end" for row in rows) == 1,
        "no_identical_call_cycle": not repeated,
    }
    return {
        "id": task["id"], "kind": task["kind"],
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "expected_calls": [{"name": name, "args": values} for name, values in expected],
        "observed_calls": [{"name": name, "args": values} for name, values in observed],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--pi-config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--model-id", default="e97-dense-agent")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("invalid rank")
    selected = balanced_prefix(load_tasks(args.authority_root), args.limit)
    tasks = [task for index, task in enumerate(selected) if index % args.world_size == args.rank]
    shard = args.output_root / f"rank-{args.rank:02d}"
    for child in ("sandboxes", "traces", "results", "sessions"):
        (shard / child).mkdir(parents=True, exist_ok=False)
    results = []
    for task in tasks:
        sandbox = make_sandbox(shard / "sandboxes", task)
        environment = dict(os.environ)
        environment.update({
            "PI_CODING_AGENT_DIR": str(args.pi_config_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(shard / "sessions" / task["id"]),
            "PI_OFFLINE": "1",
        })
        command = [
            "pi", "--mode", "json", "--provider", "emender-local", "--model", args.model_id,
            "--no-session", "--tools", "read,bash,edit,write", "--no-skills",
            "--no-context-files", "--system-prompt", E97_PI_CORE_SYSTEM,
            "--approve", task["user"],
        ]
        try:
            completed = subprocess.run(
                command, cwd=sandbox, env=environment, text=True, capture_output=True,
                timeout=args.timeout_seconds, check=False)
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = (error.stderr if isinstance(error.stderr, str) else "") + "\nTIMEOUT\n"
            returncode = 124
        (shard / "traces" / f"{task['id']}.jsonl").write_text(stdout)
        (shard / "traces" / f"{task['id']}.stderr").write_text(stderr)
        result = score(task, events(stdout), returncode, sandbox)
        (shard / "results" / f"{task['id']}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(json.dumps({"id": task["id"], "status": result["status"]}), flush=True)
    summary = {
        "schema": "emender-e97-4b-pi-core-eval-shard-v1",
        "rank": args.rank, "world_size": args.world_size,
        "tasks": len(results), "passed": sum(row["status"] == "pass" for row in results),
        "by_kind": {
            kind: {"tasks": sum(row["kind"] == kind for row in results),
                   "passed": sum(row["kind"] == kind and row["status"] == "pass" for row in results)}
            for kind in sorted({row["kind"] for row in results})
        },
    }
    (shard / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
