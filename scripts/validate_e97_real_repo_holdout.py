#!/usr/bin/env python3
"""Validate frozen real-repository mutations fail and expected repairs pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from ndm.data.masked_sft_dataset import sha256


def run(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, shell=True, executable="/bin/bash", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.authority_root / "manifest.json"
    if sha256(manifest_path) != args.authority_sha256:
        raise SystemExit("authority manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    tasks_path = args.authority_root / Path(manifest["outputs"]["tasks"]["path"]).name
    output = manifest["outputs"]["tasks"]
    if tasks_path.stat().st_size != output["bytes"] or sha256(tasks_path) != output["sha256"]:
        raise SystemExit("task payload mismatch")
    rows = [json.loads(line) for line in tasks_path.open()]
    receipts = []
    with tempfile.TemporaryDirectory(prefix="e97-real-repo-holdout-") as temporary:
        root = Path(temporary)
        for row in rows:
            source = args.source_root / row["repository"]
            sandbox = root / row["id"]
            shutil.copytree(source, sandbox, symlinks=True)
            subprocess.run(
                ["git", "-C", str(sandbox), "checkout", "--detach", row["commit"]],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for relative, content in row["setup_files"].items():
                path = sandbox / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            path = sandbox / row["path"]
            clean = row["clean"]
            mutated = row["mutated"]
            text = path.read_text()
            if text.count(clean) != 1:
                raise RuntimeError(f"{row['id']}: clean text is not unique")
            path.write_text(text.replace(clean, mutated, 1))
            changed = subprocess.run(
                ["git", "-C", str(sandbox), "diff", "--name-only"], check=True,
                text=True, stdout=subprocess.PIPE).stdout.splitlines()
            if changed != [row["path"]]:
                raise RuntimeError(f"{row['id']}: mutation changed unexpected tracked files")
            failed = run(row["focused_test"], sandbox)
            if failed.returncode == 0:
                raise RuntimeError(f"{row['id']}: mutation did not fail focused test")
            text = path.read_text()
            if text.count(mutated) != 1:
                raise RuntimeError(f"{row['id']}: expected repair precondition mismatch")
            path.write_text(text.replace(mutated, clean, 1))
            passed = run(row["focused_test"], sandbox)
            if passed.returncode != 0:
                raise RuntimeError(
                    f"{row['id']}: expected repair failed focused test\n{passed.stdout[-2000:]}")
            tracked_diff = subprocess.run(
                ["git", "-C", str(sandbox), "diff", "--exit-code"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT).returncode
            if tracked_diff != 0:
                raise RuntimeError(f"{row['id']}: expected repair did not restore tracked tree")
            receipts.append({"id": row["id"], "failure_exit": failed.returncode,
                             "repair_exit": passed.returncode,
                             "failure_output_sha256": __import__("hashlib").sha256(
                                 failed.stdout.encode()).hexdigest(),
                             "repair_output_sha256": __import__("hashlib").sha256(
                                 passed.stdout.encode()).hexdigest()})
    result = {
        "schema": "emender-e97-real-repo-holdout-validation-v1",
        "status": "pass", "authority_manifest_sha256": args.authority_sha256,
        "tasks": len(rows), "receipts": receipts,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
