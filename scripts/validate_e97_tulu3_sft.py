#!/usr/bin/env python3
"""Independently validate an E97 Tulu token-and-mask SFT authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import random
import struct

from build_e97_tulu3_sft import INDEX, SCHEMA, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise SystemExit("manifest is not a complete Tulu SFT authority")
    paths = {name: Path(info["path"]) for name, info in manifest["outputs"].items()}
    for name, path in paths.items():
        info = manifest["outputs"][name]
        if path.resolve() != (args.root / path.name).resolve():
            raise RuntimeError(f"output escaped authority root: {name}")
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise RuntimeError(f"output integrity failure: {name}")
    counts = manifest["counts"]
    records = int(counts["records"])
    tokens = int(counts["tokens"])
    if paths["index"].stat().st_size != records * INDEX.size:
        raise RuntimeError("index size contradicts record count")
    if paths["tokens"].stat().st_size != tokens * 4:
        raise RuntimeError("token size contradicts token count")
    if paths["mask"].stat().st_size != tokens:
        raise RuntimeError("mask size contradicts token count")

    rng = random.Random(970035)
    sample_indices = set(rng.sample(range(records), min(args.samples, records)))
    observed_offset = observed_targets = train_records = validation_records = 0
    with (paths["index"].open("rb") as index_file,
          paths["tokens"].open("rb") as token_file,
          paths["mask"].open("rb") as mask_file):
        token_map = mmap.mmap(token_file.fileno(), 0, access=mmap.ACCESS_READ)
        mask_map = mmap.mmap(mask_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for record_index in range(records):
                raw = index_file.read(INDEX.size)
                if len(raw) != INDEX.size:
                    raise RuntimeError("truncated index")
                offset, length, targets, split = INDEX.unpack(raw)
                if offset != observed_offset or length <= 1 or not 0 < targets <= length:
                    raise RuntimeError(f"invalid index accounting at record {record_index}")
                if split not in (0, 1):
                    raise RuntimeError("invalid split value")
                observed_offset += length
                observed_targets += targets
                train_records += split == 0
                validation_records += split == 1
                if record_index in sample_indices:
                    mask = mask_map[offset:offset + length]
                    if sum(mask) != targets or any(value not in (0, 1) for value in mask):
                        raise RuntimeError(f"invalid sampled mask at record {record_index}")
                    final_token, = struct.unpack_from("<I", token_map, (offset + length - 1) * 4)
                    if final_token != 218 or mask[-1] != 1:
                        raise RuntimeError(f"record {record_index} lacks targeted terminal RS")
                    for position in (0, length // 2, length - 1):
                        token, = struct.unpack_from("<I", token_map, (offset + position) * 4)
                        if token >= 50281:
                            raise RuntimeError(f"token outside p50k vocabulary: {token}")
        finally:
            token_map.close(); mask_map.close()
    if observed_offset != tokens or observed_targets != counts["assistant_target_tokens"]:
        raise RuntimeError("full index accounting contradicts manifest")
    if train_records != counts["train_records"] or validation_records != counts["validation_records"]:
        raise RuntimeError("split accounting contradicts manifest")
    metadata_lines = sum(1 for _ in paths["metadata"].open("rb"))
    if metadata_lines != records:
        raise RuntimeError("metadata count contradicts record count")
    receipt = {
        "schema": "emender-e97-tulu3-masked-sft-validation-v1",
        "status": "pass", "authority_manifest_sha256": sha256(args.root / "manifest.json"),
        "records": records, "tokens": tokens,
        "assistant_target_tokens": observed_targets,
        "sampled_records": len(sample_indices),
        "outputs": {name: info["sha256"] for name, info in manifest["outputs"].items()},
    }
    output = args.root / "validation.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
