#!/usr/bin/env python3
"""Build an immutable p50k token-plus-assistant-mask authority from Tulu 3."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct
import sys
import time

import pyarrow.parquet as pq
import tiktoken

SCHEMA = "emender-e97-tulu3-masked-sft-v1"
DATASET_ID = "allenai/tulu-3-sft-mixture"
DATASET_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"
TOKENIZER = "p50k_base"
TOKENIZER_SHA256 = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"
EXPECTED_ROWS = 939343
RS = "\x1e"
INDEX = struct.Struct("<QQQB7x")  # token offset, token count, target count, split
ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}
_WORKER_ENCODING = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_init():
    global _WORKER_ENCODING
    _WORKER_ENCODING = tiktoken.get_encoding(TOKENIZER)


def _split(source: str, identity: str) -> int:
    digest = hashlib.sha256(
        f"{SCHEMA}\0{source}\0{identity}".encode()).digest()
    return 1 if int.from_bytes(digest[:8], "little") % 100 == 0 else 0


def _encode_pieces(pieces):
    complete = "".join(text for text, _target in pieces)
    complete_bytes = complete.encode("utf-8")
    target_ranges = []
    offset = 0
    for text, target in pieces:
        stop = offset + len(text.encode("utf-8"))
        if target:
            target_ranges.append((offset, stop))
        offset = stop
    tokens = _WORKER_ENCODING.encode_ordinary(complete)
    masks = []
    decoded = bytearray()
    for token in tokens:
        start = len(decoded)
        token_bytes = _WORKER_ENCODING.decode_single_token_bytes(token)
        decoded.extend(token_bytes)
        stop = len(decoded)
        overlaps = [(left, right) for left, right in target_ranges
                    if start < right and stop > left]
        if not overlaps:
            masks.append(0)
        elif any(start >= left and stop <= right for left, right in overlaps):
            masks.append(1)
        else:
            raise ValueError("token crosses an assistant-mask boundary")
    if bytes(decoded) != complete_bytes:
        raise ValueError("token bytes do not reconstruct serialized record")
    return tokens, masks, complete


def serialize_row(row):
    identity = row.get("id")
    source = row.get("source")
    messages = row.get("messages")
    if not isinstance(identity, str) or not identity or not isinstance(source, str):
        return {"error": "missing_identity_or_source"}
    if not isinstance(messages, list) or not messages:
        return {"error": "missing_messages"}
    pieces = []
    roles = []
    replacements = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return {"error": "invalid_message"}
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            return {"error": f"unsupported_role:{role}"}
        if not isinstance(content, str) or not content.strip():
            return {"error": "empty_content"}
        replacements += content.count(RS)
        content = content.replace(RS, " ").replace("\r\n", "\n").replace("\r", "\n")
        if index:
            pieces.append(("\n\n", False))
        pieces.append((f"{ROLES[role]}:\n", False))
        pieces.append((content, role == "assistant"))
        roles.append(role)
    if roles[-1] != "assistant":
        return {"error": "final_message_is_not_assistant"}
    pieces.append((RS, True))
    try:
        tokens, masks, text = _encode_pieces(pieces)
    except ValueError as error:
        return {"error": str(error)}
    if not tokens or len(tokens) != len(masks) or sum(masks) == 0:
        return {"error": "empty_token_or_target_stream"}
    if max(tokens) >= 2**32:
        return {"error": "token_id_exceeds_uint32"}
    token_bytes = struct.pack(f"<{len(tokens)}I", *tokens)
    mask_bytes = bytes(masks)
    record_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "id": identity, "source": source, "roles": roles,
        "split": _split(source, identity), "tokens": len(tokens),
        "targets": sum(masks), "rs_replacements": replacements,
        "sha256": record_digest, "token_bytes": token_bytes,
        "mask_bytes": mask_bytes,
    }


def iter_rows(paths, batch_size=512):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def atomic_path(path: Path) -> Path:
    return path.with_name(path.name + f".partial-{Path('/proc/self').resolve().name}")


def main():
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    if not cache_root:
        raise SystemExit("TIKTOKEN_CACHE_DIR must bind the verified shared p50k cache")
    cache_files = list(Path(cache_root).glob("*"))
    verified_cache = [path for path in cache_files if path.is_file()
                      and sha256(path) == TOKENIZER_SHA256]
    if len(verified_cache) != 1:
        raise SystemExit("verified p50k cache object is missing or ambiguous")
    # Populate the process-local constructor before creating workers, so a bad
    # cache binding fails once instead of causing a worker respawn loop.
    _worker_init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0,
                        help="Developer fixture limit; zero means complete authority")
    args = parser.parse_args()
    paths = sorted((args.input_root / "data").glob("train-*.parquet"))
    if len(paths) != 6:
        raise SystemExit(f"expected six pinned Tulu parquet shards, found {len(paths)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("refusing to overwrite an existing SFT authority")
    partials = {name: atomic_path(path) for name, path in outputs.items()}
    for path in partials.values():
        path.unlink(missing_ok=True)

    counts = Counter()
    source_counts = Counter()
    role_counts = Counter()
    error_counts = Counter()
    length_counts = Counter()
    token_offset = 0
    started = time.time()
    row_iterator = iter_rows(paths)
    if args.limit:
        import itertools
        row_iterator = itertools.islice(row_iterator, args.limit)
    try:
        with (partials["tokens"].open("wb", buffering=16 << 20) as token_out,
              partials["mask"].open("wb", buffering=16 << 20) as mask_out,
              partials["index"].open("wb", buffering=4 << 20) as index_out,
              partials["metadata"].open("w", encoding="utf-8", buffering=4 << 20) as metadata_out,
              mp.Pool(args.workers, initializer=_worker_init) as pool):
            for result in pool.imap(serialize_row, row_iterator, chunksize=32):
                counts["input_rows"] += 1
                if "error" in result:
                    error_counts[result["error"]] += 1
                    continue
                token_out.write(result.pop("token_bytes"))
                mask_out.write(result.pop("mask_bytes"))
                index_out.write(INDEX.pack(
                    token_offset, result["tokens"], result["targets"], result["split"]))
                metadata_out.write(json.dumps({
                    key: value for key, value in result.items()
                    if key not in {"roles"}}, sort_keys=True) + "\n")
                token_offset += result["tokens"]
                split_name = "validation" if result["split"] else "train"
                counts["records"] += 1
                counts[f"{split_name}_records"] += 1
                counts["tokens"] += result["tokens"]
                counts[f"{split_name}_tokens"] += result["tokens"]
                counts["assistant_target_tokens"] += result["targets"]
                counts[f"{split_name}_assistant_target_tokens"] += result["targets"]
                counts["rs_replacements"] += result["rs_replacements"]
                source_counts[result["source"]] += 1
                role_counts.update(result["roles"])
                for threshold in (4096, 8192, 16384, 32768):
                    if result["tokens"] >= threshold:
                        length_counts[f"records_ge_{threshold}"] += 1
                if counts["input_rows"] % 10000 == 0:
                    print(json.dumps({"event": "progress", "elapsed": time.time() - started,
                                      **counts}), flush=True)
        if not args.limit and counts["input_rows"] != EXPECTED_ROWS:
            raise RuntimeError(
                f"pinned Tulu row count changed: {counts['input_rows']} != {EXPECTED_ROWS}")
        unexpected_errors = set(error_counts) - {"empty_content"}
        if unexpected_errors:
            raise RuntimeError(
                "unexpected Tulu serialization errors: " + ", ".join(sorted(unexpected_errors)))
        expected_token_bytes = counts["tokens"] * 4
        if partials["tokens"].stat().st_size != expected_token_bytes:
            raise RuntimeError("token file byte accounting mismatch")
        if partials["mask"].stat().st_size != counts["tokens"]:
            raise RuntimeError("mask file byte accounting mismatch")
        if partials["index"].stat().st_size != counts["records"] * INDEX.size:
            raise RuntimeError("index file byte accounting mismatch")
        for name, path in outputs.items():
            partials[name].replace(path)
    except BaseException:
        for path in partials.values():
            path.unlink(missing_ok=True)
        raise

    receipt = {
        "schema": SCHEMA, "status": "complete" if not args.limit else "fixture",
        "created_unix": time.time(), "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION, "tokenizer": TOKENIZER,
        "tokenizer_sha256": TOKENIZER_SHA256, "rs_token_id": 218,
        "validation_rule": "sha256(schema\\0source\\0id) little-u64 mod 100 == 0",
        "counts": dict(counts), "errors": dict(error_counts),
        "sources": dict(source_counts), "roles": dict(role_counts),
        "lengths": dict(length_counts),
        "inputs": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in paths],
        "outputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                            "sha256": sha256(path)} for name, path in outputs.items()},
        "index_record_bytes": INDEX.size, "token_dtype": "little-endian uint32",
        "mask_dtype": "uint8", "elapsed_seconds": time.time() - started,
    }
    receipt_path = args.output_root / "manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
