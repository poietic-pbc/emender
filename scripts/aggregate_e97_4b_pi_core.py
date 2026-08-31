#!/usr/bin/env python3
"""Aggregate fixed real-Pi core-tool evaluation shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.input_root.glob("rank-*/results/*.json")):
        results.append(json.loads(path.read_text()))
    if len(results) != args.expected_tasks:
        raise SystemExit(f"expected {args.expected_tasks} results, found {len(results)}")
    ids = [row["id"] for row in results]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate evaluation task IDs")
    kinds = sorted({row["kind"] for row in results})
    checks = sorted({key for row in results for key in row["checks"]})
    summary = {
        "schema": "emender-e97-4b-pi-core-eval-v1",
        "tasks": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "pass_rate": sum(row["status"] == "pass" for row in results) / len(results),
        "by_kind": {
            kind: {
                "tasks": sum(row["kind"] == kind for row in results),
                "passed": sum(row["kind"] == kind and row["status"] == "pass" for row in results),
            } for kind in kinds
        },
        "checks": {
            check: {
                "passed": sum(bool(row["checks"].get(check)) for row in results),
                "total": len(results),
            } for check in checks
        },
        "failed_ids": [row["id"] for row in results if row["status"] != "pass"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
