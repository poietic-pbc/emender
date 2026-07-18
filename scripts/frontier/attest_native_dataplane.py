#!/usr/bin/env python3
"""Record or verify the native resilient data-plane artifact identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.native_artifacts import attest_launch, record_build_manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    actions = value.add_subparsers(dest="action", required=True)
    record = actions.add_parser("record-build")
    record.add_argument("--prefix", required=True)
    record.add_argument("--source-root", default=str(ROOT))
    record.add_argument("--cmake-cache", required=True)
    record.add_argument("--output", default="")
    verify = actions.add_parser("verify")
    verify.add_argument("--backend", required=True)
    verify.add_argument("--build-manifest", default="")
    verify.add_argument("--gate-json", default="")
    verify.add_argument("--source-root", default=str(ROOT))
    verify.add_argument("--production", action="store_true")
    verify.add_argument("--full-layout", action="store_true")
    verify.add_argument("--output", default="")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.action == "record-build":
        result = record_build_manifest(
            prefix=args.prefix, source_root=args.source_root,
            cmake_cache=args.cmake_cache, output=args.output or None)
        value: object = {"status": "recorded", "build_manifest": str(result)}
    else:
        value = attest_launch(
            backend=args.backend, production=args.production,
            full_layout=args.full_layout,
            build_manifest=args.build_manifest or None,
            gate_json=args.gate_json or None, source_root=args.source_root)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
