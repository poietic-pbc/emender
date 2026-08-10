#!/usr/bin/env python3
"""Serialize and tokenize one pinned E97 instruction source into an inventory."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Iterator, Mapping

import pyarrow.parquet as pq
import tiktoken

from ndm.data.instruction_corpus import SERIALIZER_SCHEMA, serialize_row

INDEX = struct.Struct("<QQQI")  # payload offset, byte length, tokens, replaced RS


def rows_from_jsonl(path: Path) -> Iterator[Mapping]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rows_from_parquet(path: Path) -> Iterator[Mapping]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=32):
        yield from batch.to_pylist()


def input_files(root: Path, source: str) -> list[Path]:
    if source == "swe_chat":
        files = list(root.rglob("conversations.parquet"))
        if len(files) != 1:
            raise RuntimeError("SWE-chat requires exactly one conversations.parquet")
        return files
    if source == "gair_openswe":
        # openswe_traj is the SFT trajectory artifact; oss/other describe the
        # executable environment collection and are not duplicate transcripts.
        files = list(root.rglob("openswe_traj.jsonl"))
        if len(files) != 1:
            raise RuntimeError("GAIR/OpenSWE requires exactly one openswe_traj.jsonl")
        return files
    files = []
    restored_chat = list(root.rglob("chat.with_prompts.jsonl"))
    for path in root.rglob("*"):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if (source == "nemotron_instruction_chat_v3" and restored_chat
                and path.name == "chat.jsonl"):
            continue
        if path.suffix == ".parquet" or path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
            files.append(path)
    return sorted(files)


def swe_chat_rows(path: Path) -> Iterator[Mapping]:
    columns = ["session_id", "turn_number", "role", "turn_type", "content",
               "tool_name", "tool_call_id", "tool_input_json"]
    table = pq.read_table(path, columns=columns).sort_by(
        [("session_id", "ascending"), ("turn_number", "ascending")])
    current = None
    messages = []
    for item in table.to_pylist():
        session = item["session_id"]
        if current is not None and session != current:
            if messages:
                yield {"messages": messages}
            messages = []
        current = session
        role, turn_type, content = item.get("role"), item.get("turn_type"), item.get("content")
        if turn_type == "assistant_thinking":
            message = {"role": "assistant", "reasoning_content": content}
        elif role == "tool_use" or turn_type == "tool_use":
            call = item.get("tool_input_json") or content
            message = {"role": "assistant", "tool_calls": [{
                "name": item.get("tool_name"), "id": item.get("tool_call_id"),
                "arguments": call}]}
        elif role == "tool_result" or turn_type == "tool_result":
            message = {"role": "tool", "content": content,
                       "tool_call_id": item.get("tool_call_id")}
        elif turn_type in ("system_injected", "system_event"):
            message = {"role": "system", "content": content}
        else:
            message = {"role": role or "unknown", "content": content}
        messages.append(message)
    if messages:
        yield {"messages": messages}


def source_rows(source: str, path: Path) -> Iterator[Mapping]:
    if source == "swe_chat":
        yield from swe_chat_rows(path)
    elif path.suffix == ".parquet":
        yield from rows_from_parquet(path)
    else:
        yield from rows_from_jsonl(path)


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--tokenizer", default="p50k_base")
    p.add_argument("--allow-incomplete-nemotron-chat", action="store_true")
    args = p.parse_args()
    files = input_files(args.input_root, args.source)
    if not files:
        raise SystemExit(f"no parquet/jsonl inputs under {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records_partial = args.output_root / f"{args.source}.records.partial"
    index_partial = args.output_root / f"{args.source}.index.partial"
    stats_path = args.output_root / f"{args.source}.inventory.json"
    enc = tiktoken.get_encoding(args.tokenizer)
    sha = hashlib.sha256()
    stats = {
        "schema": "emender-e97-instruction-inventory-v1",
        "serializer_schema": SERIALIZER_SCHEMA,
        "source": args.source,
        "input_root": str(args.input_root.resolve()),
        "input_files": [str(x.relative_to(args.input_root)) for x in files],
        "tokenizer": args.tokenizer,
        "rows_seen": 0, "records": 0, "rejected": 0,
        "incomplete_nemotron_chat": 0,
        "tokens": 0, "bytes": 0, "embedded_rs_replaced": 0,
        "records_ge_32k": 0, "records_ge_64k": 0, "records_ge_128k": 0,
        "started_unix": time.time(),
    }
    with records_partial.open("wb") as records, index_partial.open("wb") as index:
        for path in files:
            for row in source_rows(args.source, path):
                stats["rows_seen"] += 1
                if args.source == "nemotron_instruction_chat_v3":
                    messages = row.get("messages") or []
                    incomplete = any(
                        isinstance(m, Mapping) and m.get("role") in ("system", "user")
                        and m.get("content") is None for m in messages[:2])
                    if incomplete:
                        stats["incomplete_nemotron_chat"] += 1
                        if not args.allow_incomplete_nemotron_chat:
                            stats["rejected"] += 1
                            continue
                try:
                    text, replaced = serialize_row(args.source, row)
                except (TypeError, ValueError, json.JSONDecodeError):
                    stats["rejected"] += 1
                    continue
                payload = text.encode("utf-8")
                tokens = len(enc.encode(text, disallowed_special=()))
                offset = records.tell()
                records.write(payload)
                index.write(INDEX.pack(offset, len(payload), tokens, replaced))
                sha.update(payload)
                stats["records"] += 1
                stats["tokens"] += tokens
                stats["bytes"] += len(payload)
                stats["embedded_rs_replaced"] += replaced
                stats["records_ge_32k"] += tokens >= 32768
                stats["records_ge_64k"] += tokens >= 65536
                stats["records_ge_128k"] += tokens >= 131072
                if stats["records"] % 10000 == 0:
                    print(json.dumps({k: stats[k] for k in (
                        "source", "rows_seen", "records", "rejected", "tokens", "bytes")}),
                        flush=True)
    if stats["records"] == 0:
        raise SystemExit(f"{args.source} produced zero complete records")
    if (args.source == "nemotron_instruction_chat_v3"
            and stats["incomplete_nemotron_chat"]
            and not args.allow_incomplete_nemotron_chat):
        raise SystemExit(
            "Nemotron chat still has withheld prompts; run the pinned "
            "prepare_chat_prompts.py reconstruction before inventory")
    records_path = args.output_root / f"{args.source}.records"
    index_path = args.output_root / f"{args.source}.index"
    records_partial.replace(records_path)
    index_partial.replace(index_path)
    stats.update(completed_unix=time.time(), elapsed_s=time.time() - stats["started_unix"],
                 records_sha256=sha.hexdigest(), index_entry_bytes=INDEX.size,
                 records_path=str(records_path), index_path=str(index_path))
    write_json(stats_path, stats)
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
