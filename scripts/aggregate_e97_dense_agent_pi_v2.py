#!/usr/bin/env python3
"""Aggregate real-Pi dense-agent v2 task results."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_root.glob("rank-*/results/*.json"))
    results = [json.loads(path.read_text()) for path in paths]
    if len(results) != args.expected_tasks:
        raise RuntimeError(f"expected {args.expected_tasks} results, found {len(results)}")
    ids = [result["id"] for result in results]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate task identities")
    checks = sorted(results[0]["checks"]) if results else []
    families = defaultdict(list)
    for result in results:
        families[result["kind"]].append(result)
    summary = {
        "schema": "emender-e97-dense-agent-pi-v2-summary",
        "tasks": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "success_rate": sum(result["status"] == "pass" for result in results) / len(results),
        "checks": {check: sum(result["checks"][check] for result in results) / len(results) for check in checks},
        "families": {
            kind: {
                "tasks": len(items),
                "passed": sum(item["status"] == "pass" for item in items),
                "success_rate": sum(item["status"] == "pass" for item in items) / len(items),
            }
            for kind, items in sorted(families.items())
        },
        "tool_call_sequences": dict(Counter(" -> ".join(call["name"] or "?" for call in result["tool_calls"]) for result in results)),
    }
    summary["status"] = "pass" if summary["passed"] == summary["tasks"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
