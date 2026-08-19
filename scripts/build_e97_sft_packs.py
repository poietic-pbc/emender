#!/usr/bin/env python3
"""Build deterministic complete-record pack descriptors for masked SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from ndm.data.masked_sft_dataset import (
    AUTHORITY_SCHEMA, PACK_INDEX, PACK_SCHEMA, RECORD_INDEX, sha256,
)


def _atomic(path: Path) -> Path:
    return path.with_name(path.name + ".partial")


def _entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-size", type=int, required=True)
    parser.add_argument("--authority-manifest-sha256", required=True)
    parser.add_argument("--include-source", action="append", default=[],
                        help="Exact metadata source to retain; repeat for a union")
    args = parser.parse_args()
    if args.context_size <= 0:
        raise SystemExit("context-size must be positive")
    authority_manifest_path = args.authority_root / "manifest.json"
    if sha256(authority_manifest_path) != args.authority_manifest_sha256:
        raise SystemExit("authority manifest SHA-256 mismatch")
    authority = json.loads(authority_manifest_path.read_text())
    if authority.get("schema") != AUTHORITY_SCHEMA or authority.get("status") != "complete":
        raise SystemExit("input is not a complete masked-SFT authority")
    record_path = args.authority_root / Path(authority["outputs"]["index"]["path"]).name
    if (record_path.stat().st_size != authority["outputs"]["index"]["bytes"]
            or sha256(record_path) != authority["outputs"]["index"]["sha256"]):
        raise SystemExit("record index integrity mismatch")
    records = np.memmap(
        record_path, mode="r",
        dtype=np.dtype([("offset", "<u8"), ("tokens", "<u8"),
                        ("targets", "<u8"), ("split", "u1"), ("pad", "V7")]))
    include_sources = tuple(sorted(set(args.include_source)))
    included_record_ids = None
    if include_sources:
        metadata_entry = authority["outputs"].get("metadata")
        if metadata_entry is None:
            raise SystemExit("source filtering requires authority metadata")
        metadata_path = args.authority_root / Path(metadata_entry["path"]).name
        if (metadata_path.stat().st_size != metadata_entry["bytes"]
                or sha256(metadata_path) != metadata_entry["sha256"]):
            raise SystemExit("record metadata integrity mismatch")
        wanted = set(include_sources)
        included_record_ids = set()
        metadata_count = 0
        with metadata_path.open() as stream:
            for record_id, line in enumerate(stream):
                metadata_count += 1
                if json.loads(line).get("source") in wanted:
                    included_record_ids.add(record_id)
        if metadata_count != len(records):
            raise SystemExit("record metadata/index count mismatch")
        if not included_record_ids:
            raise SystemExit("source filter selected no records")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pack_records": args.output_root / "pack_records.uint32.bin",
        "train_index": args.output_root / "train_packs.idx",
        "validation_index": args.output_root / "validation_packs.idx",
    }
    if any(path.exists() for path in (*outputs.values(), args.output_root / "manifest.json")):
        raise SystemExit("refusing to overwrite an existing pack authority")
    temporary = {name: _atomic(path) for name, path in outputs.items()}
    for path in temporary.values():
        path.unlink(missing_ok=True)

    sequence_tokens = args.context_size + 1
    split_receipts = {}
    record_cursor = 0
    try:
        with (temporary["pack_records"].open("wb", buffering=4 << 20) as record_out,
              temporary["train_index"].open("wb", buffering=4 << 20) as train_out,
              temporary["validation_index"].open("wb", buffering=1 << 20) as validation_out):
            for split_name, split_value, index_out in (
                    ("train", 0, train_out), ("validation", 1, validation_out)):
                current_ids = []
                current_tokens = 0
                current_targets = 0
                counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
                          "packs": 0, "excluded_oversize_records": 0,
                          "excluded_oversize_tokens": 0,
                          "excluded_oversize_assistant_target_tokens": 0}

                def flush() -> None:
                    nonlocal current_ids, current_tokens, current_targets, record_cursor
                    if not current_ids:
                        return
                    record_out.write(struct.pack(f"<{len(current_ids)}I", *current_ids))
                    index_out.write(PACK_INDEX.pack(
                        record_cursor, len(current_ids), current_tokens, current_targets))
                    record_cursor += len(current_ids)
                    counts["packs"] += 1
                    current_ids, current_tokens, current_targets = [], 0, 0

                for record_id, record in enumerate(records):
                    if (int(record["split"]) != split_value
                            or (included_record_ids is not None
                                and record_id not in included_record_ids)):
                        continue
                    token_count = int(record["tokens"])
                    target_count = int(record["targets"])
                    if token_count > sequence_tokens:
                        counts["excluded_oversize_records"] += 1
                        counts["excluded_oversize_tokens"] += token_count
                        counts["excluded_oversize_assistant_target_tokens"] += target_count
                        continue
                    if current_ids and current_tokens + token_count > sequence_tokens:
                        flush()
                    current_ids.append(record_id)
                    current_tokens += token_count
                    current_targets += target_count
                    counts["records"] += 1
                    counts["tokens"] += token_count
                    counts["assistant_target_tokens"] += target_count
                flush()
                split_receipts[split_name] = counts
        for name, path in outputs.items():
            temporary[name].replace(path)
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    del records

    manifest = {
        "schema": PACK_SCHEMA, "status": "complete",
        "authority_manifest_sha256": args.authority_manifest_sha256,
        "context_size": args.context_size, "sequence_tokens": sequence_tokens,
        "packing": "stable-record-order greedy next-fit; no record splitting",
        "source_filter": ({"include_exact": list(include_sources)}
                          if include_sources else None),
        "sampling": "pack IDs sampled with replacement by emender-record-pack-counter-v1",
        "splits": split_receipts,
        "record_index_bytes": RECORD_INDEX.size, "pack_index_bytes": PACK_INDEX.size,
        "outputs": {name: _entry(path) for name, path in outputs.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
