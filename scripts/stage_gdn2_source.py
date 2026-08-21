#!/usr/bin/env python3
"""Create an immutable extracted GDN2 source tree with a verified receipt."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.models.external_gdn2 import (
    BOUND_SOURCE_RECEIPT,
    gdn2_source_tree_sha256,
    verify_bound_gdn2_source,
)


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    head = git(checkout, "rev-parse", "HEAD")
    if head != args.expected_commit:
        raise RuntimeError(f"checkout HEAD {head} != required {args.expected_commit}")
    if git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("GDN2 checkout has tracked modifications")
    archive = subprocess.check_output([
        "git", "-C", str(checkout), "archive", "--format=tar", head])
    archive_sha256 = hashlib.sha256(archive).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(temporary, filter="data")
        receipt = {
            "schema": "emender-gdn2-source-v1",
            "origin": git(checkout, "remote", "get-url", "origin"),
            "commit": head,
            "tree": git(checkout, "rev-parse", f"{head}^{{tree}}"),
            "git_archive_sha256": archive_sha256,
            "source_tree_sha256": gdn2_source_tree_sha256(temporary),
        }
        (temporary / BOUND_SOURCE_RECEIPT).write_text(
            json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        verify_bound_gdn2_source(temporary, head)
        if args.output.exists():
            existing = verify_bound_gdn2_source(args.output, head)
            if existing != receipt:
                raise RuntimeError(
                    f"existing staged source conflicts with requested receipt: {args.output}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, args.output)
        print(json.dumps(receipt, sort_keys=True))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
