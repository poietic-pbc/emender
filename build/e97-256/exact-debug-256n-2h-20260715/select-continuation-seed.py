#!/usr/bin/env python3
"""Fail-closed selector for an exact-debug immutable continuation handoff."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
args = parser.parse_args()
manifest = args.manifest.resolve(strict=True)
record = json.loads(manifest.read_text())
checkpoint = Path(record["checkpoint_path"]).resolve(strict=True)
h = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
        h.update(block)
digest = h.hexdigest()
if digest != record["checkpoint_sha256"]:
    raise SystemExit(f"checkpoint SHA256 mismatch: {digest}")
if checkpoint.stat().st_size != record["checkpoint_size_bytes"]:
    raise SystemExit("checkpoint size mismatch")
print(checkpoint)
