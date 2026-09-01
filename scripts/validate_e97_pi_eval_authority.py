#!/usr/bin/env python3
"""Mechanically execute an immutable synthetic Pi evaluation authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from ndm.data.masked_sft_dataset import sha256
from scripts.eval_e97_4b_pi_core import expected_calls, make_sandbox, verify_sandbox


def apply_call(sandbox: Path, name: str, values: dict) -> int:
    if name == "read":
        path = sandbox / values["path"]
        if not path.is_file():
            raise RuntimeError(f"read path missing: {values['path']}")
        return 0
    if name == "bash":
        completed = subprocess.run(
            values["command"], cwd=sandbox, shell=True, executable="/bin/bash",
            check=False, timeout=30, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True)
        return completed.returncode
    if name == "edit":
        path = sandbox / values["path"]
        text = path.read_text()
        old = values["oldText"]
        if text.count(old) != 1:
            raise RuntimeError(f"edit oldText is not unique: {values['path']}")
        path.write_text(text.replace(old, values["newText"], 1))
        return 0
    if name == "write":
        path = sandbox / values["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(values["content"])
        return 0
    raise RuntimeError(f"unsupported tool: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    args = parser.parse_args()
    manifest_path = args.authority_root / "manifest.json"
    if sha256(manifest_path) != args.authority_sha256:
        raise SystemExit("authority manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    metadata = args.authority_root / Path(manifest["outputs"]["metadata"]["path"]).name
    output = manifest["outputs"]["metadata"]
    if metadata.stat().st_size != output["bytes"] or sha256(metadata) != output["sha256"]:
        raise SystemExit("authority metadata identity mismatch")
    rows = [json.loads(line) for line in metadata.open()]
    calls = 0
    with tempfile.TemporaryDirectory(prefix="e97-pi-eval-validation-") as temporary:
        root = Path(temporary)
        for row in rows:
            sandbox = make_sandbox(root, row)
            declared_calls = expected_calls(row)
            exit_codes = row["task"].get("expected_exit_codes", [0] * len(declared_calls))
            if len(exit_codes) != len(declared_calls):
                raise RuntimeError(f"exit-code accounting mismatch: {row['id']}")
            for (name, values), expected_exit in zip(declared_calls, exit_codes):
                observed_exit = apply_call(sandbox, name, values)
                if observed_exit != expected_exit:
                    raise RuntimeError(
                        f"unexpected exit {observed_exit} != {expected_exit}: {row['id']}")
                calls += 1
            if not verify_sandbox(sandbox, row):
                raise RuntimeError(f"postcondition failed: {row['id']}")
            if not row["task"].get("final_contains"):
                raise RuntimeError(f"missing final evidence: {row['id']}")
    result = {
        "schema": "emender-e97-pi-eval-mechanical-validation-v1",
        "status": "pass", "authority_manifest_sha256": args.authority_sha256,
        "tasks": len(rows), "expected_calls_executed": calls,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
