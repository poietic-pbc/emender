#!/usr/bin/env python3
"""Derive a final-only Pi repair authority with exact live-provider tool context."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from scripts.build_e97_pi_instruction_sft import ENCODING, SYSTEM, serialize, split, trace


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def normalize_live_tool_results(messages: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Match Pi's OpenAI-completions representation for successful empty tools."""
    return [
        (role, "(no tool output)" if role == "tool" and text == "" else text)
        for role, text in messages
    ]


def serialize_final_only(
    messages: list[tuple[str, str]], encoding,
) -> tuple[list[int], list[int], str]:
    """Mask only the terminal final and its learned one-line stop newline."""
    if not messages or messages[-1][0] != "assistant" or not messages[-1][1].startswith("Final:"):
        raise ValueError("repair records require one terminal Final assistant turn")
    normalized = normalize_live_tool_results(messages)
    pieces: list[tuple[str, bool]] = []
    for position, (role, text) in enumerate(normalized):
        if position:
            pieces.append(("\n\n", False))
        label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}[role]
        pieces.append((f"{label}:\n", False))
        pieces.append((text, position == len(normalized) - 1))
    # Canonical serving ends a concise final at its first newline. Make that
    # boundary supervised rather than relying on unrelated pretraining records.
    pieces.append(("\n", True))
    complete = "".join(text for text, _ in pieces)
    if "\x1e" in complete:
        raise RuntimeError("repair trajectories must remain RS-free")
    ranges, cursor = [], 0
    for text, target in pieces:
        stop = cursor + len(text.encode())
        if target:
            ranges.append((cursor, stop))
        cursor = stop
    tokens = encoding.encode_ordinary(complete)
    masks, decoded = [], bytearray()
    for token_id in tokens:
        left = len(decoded)
        decoded.extend(encoding.decode_single_token_bytes(token_id))
        right = len(decoded)
        overlaps = [(start, stop) for start, stop in ranges if left < stop and right > start]
        if not overlaps:
            masks.append(0)
        elif any(left >= start and right <= stop for start, stop in overlaps):
            masks.append(1)
        else:
            raise RuntimeError("token crosses the final-only target boundary")
    if not masks or sum(masks) <= 0 or masks[-1] != 1:
        raise RuntimeError("terminal final/newline target was not preserved")
    return tokens, masks, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_manifest_path = args.source_root / "manifest.json"
    if sha256(source_manifest_path) != args.source_sha256:
        raise SystemExit("source manifest SHA-256 mismatch")
    source = json.loads(source_manifest_path.read_text())
    if source.get("schema") != AUTHORITY_SCHEMA or source.get("status") != "complete":
        raise SystemExit("unsupported source authority")
    metadata_path = args.source_root / Path(source["outputs"]["metadata"]["path"]).name
    expected_metadata = source["outputs"]["metadata"]
    if metadata_path.stat().st_size != expected_metadata["bytes"] or sha256(metadata_path) != expected_metadata["sha256"]:
        raise SystemExit("source metadata payload mismatch")
    rows = [json.loads(line) for line in metadata_path.open()]
    if len(rows) != int(source["counts"]["records"]):
        raise SystemExit("source record count mismatch")
    seed = int(source["seed"])
    kinds = tuple(source["kinds"])
    encoding = tiktoken.get_encoding(ENCODING)
    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
              "train_records": 0, "validation_records": 0}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for index, row in enumerate(rows):
            kind = kinds[index % len(kinds)]
            identity = f"pi-native-{kind}-{index:08d}"
            user, turns, task = trace(kind, index, random.Random(seed + index))
            original = [("system", SYSTEM), ("user", user), *turns]
            _, _, original_text = serialize(original, encoding)
            if row != {**row, "id": identity, "kind": kind, "user": user, "task": task}:
                raise RuntimeError(f"source reconstruction mismatch at record {index}")
            if hashlib.sha256(original_text.encode()).hexdigest() != row["serialization_sha256"]:
                raise RuntimeError(f"source serialization mismatch at record {index}")
            validation = int(split(identity))
            if validation != int(row["split"]):
                raise RuntimeError(f"source split mismatch at record {index}")
            tokens, masks, complete = serialize_final_only(original, encoding)
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_out.write(json.dumps({
                "id": f"pi-final-repair-{index:08d}", "source_id": identity,
                "split": validation, "kind": kind, "tokens": len(tokens),
                "targets": sum(masks),
                "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
            }, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += sum(masks)
            counts["validation_records" if validation else "train_records"] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": "E97 Pi finalization repair with live empty-tool normalization",
        "source_root": str(args.source_root.resolve()),
        "source_manifest_sha256": args.source_sha256,
        "construction": "deterministic source reconstruction; prior actions context-only; terminal Final plus newline targeted",
        "empty_tool_result": "(no tool output)",
        "terminal_boundary": "one targeted newline after Final",
        "seed": seed,
        "kinds": list(kinds),
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
