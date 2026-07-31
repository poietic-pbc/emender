#!/usr/bin/env python3
"""Regenerate the checked-in canonical native/Lean trace corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = (
    ROOT
    / "formal/resilient/.lake/build/bin/resilient-conformance-corpus"
)
DEFAULT_OUTPUT = ROOT / "formal/resilient/corpus/native-v1"
MANIFEST_SCHEMA = "emender-native-lean-conformance-corpus-v1"


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _run(runner: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [str(runner.resolve()), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"{runner} exited {completed.returncode}"
        )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    scenarios = [
        value
        for value in _run(arguments.runner, "--list").splitlines()
        if value
    ]
    if not scenarios:
        raise RuntimeError("Lean conformance corpus contains no scenarios")
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for scenario in scenarios:
        encoded = _run(arguments.runner, "--trace", scenario)
        decoded = json.loads(encoded)
        canonical = canonical_json(decoded)
        if encoded != canonical:
            raise RuntimeError(f"Lean emitted noncanonical trace: {scenario}")
        target = arguments.output_directory / f"{scenario}.json"
        target.write_text(canonical + "\n", encoding="utf-8")
        entries.append(
            {
                "id": scenario,
                "path": target.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    (canonical + "\n").encode("utf-8")
                ).hexdigest(),
                "events": len(decoded["steps"]),
                "replayCommand": (
                    "scripts/conformance/run_native_lean_conformance.py "
                    f"--trace {target.relative_to(ROOT).as_posix()} "
                    "--build-manifest "
                    "build/native-resilient-dataplane/native-artifacts.json"
                ),
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generator": (
            "formal/resilient/.lake/build/bin/"
            "resilient-conformance-corpus"
        ),
        "traceSchema": "emender-resilient-coordination-trace-v1",
        "traceSchemaDigest": (
            "cf654525395e63b31b2d76e8109ee2bcc"
            "6a652f6273d1c6e4ca5bec9ecb776b4"
        ),
        "toolchain": (
            "leanprover/lean4:v4.26.0"
            "@d8204c9fd894f91bbb2cdfec5912ec8196fd8562"
        ),
        "entries": entries,
    }
    manifest_path = arguments.output_directory / "manifest.json"
    manifest_path.write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
