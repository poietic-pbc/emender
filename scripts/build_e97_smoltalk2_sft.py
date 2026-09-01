#!/usr/bin/env python3
"""Build an immutable masked-SFT authority from admitted SmolTalk2 subsets."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct

import pyarrow.parquet as pq

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from scripts import build_e97_tulu3_sft as codec

DATASET_ID = "HuggingFaceTB/smoltalk2"
DATASET_REVISION = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
DATASET_CARD_SHA256 = "8bdc71dea7688d0b57e7eeb463f468641b460e5e512b2c57f8b45a070588b2c7"
TOKENIZER = "p50k_base"
ALLOWED_PREFIXES = (
    "multi_turn_reasoning_if_think-",
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think-",
    "smoltalk_smollm3_smol_magpie_ultra_no_think-",
    "smoltalk_smollm3_smol_rewrite_no_think-",
    "smoltalk_smollm3_smol_summarize_no_think-",
    "smoltalk_smollm3_systemchats_30k_no_think-",
)
ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}


def split(identity: str) -> int:
    digest = hashlib.sha256(f"smoltalk2-v1\0{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 100 == 0


def worker_init() -> None:
    codec._worker_init()


def serialize_item(item):
    identity, source_file, row_index, row = item
    messages = row.get("messages")
    source = row.get("source")
    if not isinstance(messages, list) or not messages or not isinstance(source, str):
        return {"error": "missing_messages_or_source", "identity": identity}
    pieces = []
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return {"error": "invalid_message", "identity": identity}
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            return {"error": f"unsupported_role:{role}", "identity": identity}
        if not isinstance(content, str) or not content.strip():
            return {"error": "empty_content", "identity": identity}
        normalized = content.replace(codec.RS, " ").replace("\r\n", "\n").replace("\r", "\n").strip()
        if index:
            pieces.append(("\n\n", False))
        pieces.append((f"{ROLES[role]}:\n", False))
        pieces.append((normalized, role == "assistant"))
        roles.append(role)
    if roles[-1] != "assistant":
        return {"error": "final_message_is_not_assistant", "identity": identity}
    pieces.append((codec.RS, True))
    try:
        tokens, masks, complete = codec._encode_pieces(pieces)
    except ValueError as error:
        return {"error": str(error), "identity": identity}
    return {
        "identity": identity, "source": source, "source_file": source_file,
        "source_row": row_index, "split": int(split(identity)),
        "tokens": len(tokens), "targets": sum(masks),
        "token_bytes": struct.pack(f"<{len(tokens)}I", *tokens),
        "mask_bytes": bytes(masks),
        "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
        "has_think": any("<think>" in str(message.get("content", "")) for message in messages),
    }


def rows(paths):
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row_index, row in enumerate(batch.to_pylist(), offset):
                identity = f"smoltalk2:{path.name}:{row_index}"
                yield identity, path.name, row_index, row
            offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified p50k cache")
    cache_object = Path(cache_root) / codec.TOKENIZER_CACHE_KEY
    if not cache_object.is_file() or sha256(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache object is missing or corrupt")
    worker_init()
    if sha256(args.input_root / "README.md") != DATASET_CARD_SHA256:
        raise SystemExit("SmolTalk2 dataset card mismatch")
    paths = sorted(
        path for path in (args.input_root / "SFT").glob("*.parquet")
        if path.name.startswith(ALLOWED_PREFIXES))
    if len(paths) != 11:
        raise SystemExit(f"expected 11 admitted SmolTalk2 parquet files, found {len(paths)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {"tokens": args.output_root / "tokens.uint32.bin",
               "mask": args.output_root / "assistant_mask.uint8.bin",
               "index": args.output_root / "records.idx",
               "metadata": args.output_root / "records.jsonl"}
    counts = Counter()
    errors = Counter()
    sources = Counter()
    offset = 0
    iterator = rows(paths)
    if args.limit:
        import itertools
        iterator = itertools.islice(iterator, args.limit)
    with outputs["tokens"].open("wb") as token_out, outputs["mask"].open("wb") as mask_out, \
         outputs["index"].open("wb") as index_out, outputs["metadata"].open("w") as metadata_out, \
         mp.Pool(args.workers, initializer=worker_init) as pool:
        for result in pool.imap(serialize_item, iterator, chunksize=16):
            counts["input_records"] += 1
            if "error" in result:
                errors[result["error"]] += 1
                continue
            token_out.write(result.pop("token_bytes"))
            mask_out.write(result.pop("mask_bytes"))
            index_out.write(RECORD_INDEX.pack(
                offset, result["tokens"], result["targets"], result["split"]))
            metadata_out.write(json.dumps(result, sort_keys=True) + "\n")
            offset += result["tokens"]
            counts["records"] += 1
            counts["tokens"] += result["tokens"]
            counts["assistant_target_tokens"] += result["targets"]
            counts["think_records"] += int(result["has_think"])
            counts["validation_records" if result["split"] else "train_records"] += 1
            sources[result["source"]] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "purpose": "admitted Apache-2.0 SmolTalk2 broad instruction and reasoning SFT",
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "dataset_card_sha256": DATASET_CARD_SHA256,
        "license_policy": "new SmolTalk2 Apache-2.0 subsets only",
        "excluded_downloaded_subset": "smolagents_toolcalling_traces requires Pi tool normalization",
        "tokenizer": TOKENIZER, "counts": dict(counts), "errors": dict(errors),
        "source_counts": dict(sources),
        "input_files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
        "outputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                             "sha256": sha256(path)} for name, path in outputs.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
