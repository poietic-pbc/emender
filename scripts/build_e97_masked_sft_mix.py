#!/usr/bin/env python3
"""Construct an immutable target-token-weighted union of masked-SFT authorities."""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256


@dataclass
class Source:
    name: str
    root: Path
    manifest_sha256: str
    target_assistant_tokens: int
    manifest: dict
    records: np.memmap
    eligible_record_ids: np.ndarray
    token_file: object
    token_map: mmap.mmap
    mask_file: object
    mask_map: mmap.mmap

    def close(self) -> None:
        del self.records
        self.token_map.close()
        self.mask_map.close()
        self.token_file.close()
        self.mask_file.close()


def parse_source(spec: str) -> tuple[str, Path, str, int]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("source must be NAME=ROOT,MANIFEST_SHA256,TARGET_ASSISTANT_TOKENS")
    fields = parts[1].rsplit(",", 2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("invalid source specification")
    name, root, digest, target = parts[0], Path(fields[0]), fields[1], int(fields[2])
    if not name or target <= 0:
        raise argparse.ArgumentTypeError("source name and target must be positive")
    return name, root, digest, target


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def open_source(spec: tuple[str, Path, str, int], max_record_tokens: int) -> Source:
    name, root, digest, target = spec
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != digest:
        raise RuntimeError(f"{name}: manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != AUTHORITY_SCHEMA or manifest.get("status") != "complete":
        raise RuntimeError(f"{name}: unsupported authority")
    outputs = manifest["outputs"]
    paths = {key: root / Path(outputs[key]["path"]).name for key in ("tokens", "mask", "index")}
    for key, path in paths.items():
        if path.stat().st_size != int(outputs[key]["bytes"]) or sha256(path) != outputs[key]["sha256"]:
            raise RuntimeError(f"{name}: {key} payload mismatch")
    records = np.memmap(paths["index"], mode="r", dtype=np.dtype([
        ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
        ("split", "u1"), ("pad", "V7")]))
    eligible_record_ids = np.flatnonzero(
        (records["targets"] > 0) & (records["tokens"] <= max_record_tokens))
    if len(eligible_record_ids) == 0:
        raise RuntimeError(f"{name}: no target-bearing records fit max-record-tokens")
    token_file = paths["tokens"].open("rb")
    mask_file = paths["mask"].open("rb")
    return Source(name, root, digest, target, manifest, records,
                  eligible_record_ids, token_file,
                  mmap.mmap(token_file.fileno(), 0, access=mmap.ACCESS_READ),
                  mask_file, mmap.mmap(mask_file.fileno(), 0, access=mmap.ACCESS_READ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--seed", type=int, default=974002)
    parser.add_argument("--max-record-tokens", type=int, default=4097,
                        help="Sample only whole records fitting this token bound")
    args = parser.parse_args()
    if args.max_record_tokens < 2:
        raise SystemExit("max-record-tokens must be at least 2")
    names = [source[0] for source in args.source]
    if len(names) != len(set(names)):
        raise SystemExit("source names must be unique")
    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    sources = [open_source(spec, args.max_record_tokens) for spec in args.source]
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
              "train_records": 0, "validation_records": 0}
    receipts = {}
    output_offset = 0
    try:
        with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
            for source_index, source in enumerate(sources):
                rng = random.Random(args.seed + source_index)
                observed_targets = observed_tokens = records_written = 0
                while observed_targets < source.target_assistant_tokens:
                    record_id = int(source.eligible_record_ids[
                        rng.randrange(len(source.eligible_record_ids))])
                    record = source.records[record_id]
                    token_count = int(record["tokens"])
                    target_count = int(record["targets"])
                    if token_count < 2 or target_count <= 0:
                        continue
                    source_offset = int(record["offset"])
                    token_left, token_right = source_offset * 4, (source_offset + token_count) * 4
                    token_bytes = source.token_map[token_left:token_right]
                    mask_bytes = source.mask_map[source_offset:source_offset + token_count]
                    if len(token_bytes) != token_count * 4 or len(mask_bytes) != token_count:
                        raise RuntimeError(f"{source.name}: record payload range mismatch")
                    if sum(mask_bytes) != target_count:
                        raise RuntimeError(f"{source.name}: record target count mismatch")
                    split = int(record["split"])
                    token_out.write(token_bytes)
                    mask_out.write(mask_bytes)
                    index_out.write(RECORD_INDEX.pack(output_offset, token_count, target_count, split))
                    metadata_out.write(json.dumps({
                        "id": f"mix-{source.name}-{records_written:09d}",
                        "source": source.name,
                        "source_manifest_sha256": source.manifest_sha256,
                        "source_record_id": record_id,
                        "split": split,
                        "tokens": token_count,
                        "targets": target_count,
                    }, sort_keys=True) + "\n")
                    output_offset += token_count
                    observed_tokens += token_count
                    observed_targets += target_count
                    records_written += 1
                    counts["records"] += 1
                    counts["tokens"] += token_count
                    counts["assistant_target_tokens"] += target_count
                    counts["validation_records" if split else "train_records"] += 1
                receipts[source.name] = {
                    "root": str(source.root.resolve()),
                    "manifest_sha256": source.manifest_sha256,
                    "requested_assistant_target_tokens": source.target_assistant_tokens,
                    "eligible_source_records": len(source.eligible_record_ids),
                    "records": records_written,
                    "tokens": observed_tokens,
                    "assistant_target_tokens": observed_targets,
                }
        for handle in (token_out, mask_out, index_out, metadata_out):
            pass
    finally:
        for source in sources:
            source.close()
    total_targets = counts["assistant_target_tokens"]
    for receipt in receipts.values():
        receipt["assistant_target_fraction"] = receipt["assistant_target_tokens"] / total_targets
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": "target-token-weighted E97 Pi instruction mixture",
        "construction": "deterministic sampling with replacement from immutable source records",
        "seed": args.seed,
        "max_record_tokens": args.max_record_tokens,
        "source_order": names,
        "sources": receipts,
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
