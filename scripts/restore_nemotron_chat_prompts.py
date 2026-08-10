#!/usr/bin/env python3
"""Restore Nemotron protected prompts from pinned local LMSYS/WildChat snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq


def family(row):
    metadata = row.get("metadata") or {}
    seed = str(metadata.get("seed_dataset") or metadata.get("seed_source") or "").lower()
    if seed.startswith("lmsys"): return "lmsys"
    if seed.startswith("allenai/wildchat") or seed.startswith("wildchat"): return "wildchat"
    return None


def source_messages(row):
    conversation = row.get("conversation")
    if not isinstance(conversation, list): return None, None
    system = user = None
    for message in conversation:
        if not isinstance(message, dict): continue
        role, content = message.get("role"), message.get("content")
        if role == "system" and system is None and isinstance(content, str): system = content
        if role == "user" and isinstance(content, str): user = content; break
    if user is None and conversation and isinstance(conversation[0], dict):
        value = conversation[0].get("content")
        if isinstance(value, str): user = value
    return system, user


def parquet_rows(root):
    files = sorted(root.rglob("*.parquet"))
    if not files: raise RuntimeError(f"no parquet files under {root}")
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256): yield from batch.to_pylist()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lmsys-root", type=Path, required=True)
    p.add_argument("--wildchat-root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    needed = {"lmsys": set(), "wildchat": set()}
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line); fam = family(row)
            value = (row.get("metadata") or {}).get("seed_prompt_sha256")
            if fam and value: needed[fam].add(value)
    replacements = {}
    for fam, root in (("lmsys", args.lmsys_root), ("wildchat", args.wildchat_root)):
        remaining = set(needed[fam])
        for row in parquet_rows(root):
            system, user = source_messages(row)
            if user is None: continue
            key = hashlib.sha256(user.encode()).hexdigest()
            if key in remaining:
                replacements[(fam, key)] = (system, user); remaining.remove(key)
                if not remaining: break
        if remaining:
            raise RuntimeError(f"missing {len(remaining)} {fam} prompt hashes")
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    restored = 0
    with args.input.open(encoding="utf-8") as src, partial.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(src, 1):
            row = json.loads(line); fam = family(row)
            key = (row.get("metadata") or {}).get("seed_prompt_sha256")
            if fam and key:
                system, user = replacements[(fam, key)]
                messages = row.get("messages") or []
                if messages and messages[0].get("role") == "system": messages[0]["content"] = system
                user_message = next((x for x in messages if x.get("role") == "user"), None)
                if user_message is None: raise RuntimeError(f"no user message at line {line_number}")
                user_message["content"] = user; restored += 1
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    partial.replace(args.output)
    receipt = {"schema": "emender-nemotron-prompt-restoration-v1",
               "created_unix": time.time(), "input": str(args.input), "output": str(args.output),
               "input_sha256": digest(args.input), "output_sha256": digest(args.output),
               "needed_unique_hashes": {k: len(v) for k,v in needed.items()},
               "restored_rows": restored, "lmsys_root": str(args.lmsys_root),
               "wildchat_root": str(args.wildchat_root)}
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__": main()
