#!/usr/bin/env python3
"""Re-tokenize a Pi SFT authority under a versioned system prompt."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct

import numpy as np
import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import E97_PI_AGENT_SYSTEM_V2, E97_PI_CORE_SYSTEM
from scripts.build_e97_tulu3_sft import TOKENIZER_CACHE_KEY, TOKENIZER_SHA256

RS = "\x1e"


def encode_pieces(encoding, pieces):
    text = "".join(value for value, _ in pieces)
    ranges, cursor = [], 0
    for value, target in pieces:
        stop = cursor + len(value.encode())
        if target:
            ranges.append((cursor, stop))
        cursor = stop
    tokens = encoding.encode_ordinary(text)
    masks, decoded = [], bytearray()
    for token in tokens:
        left = len(decoded); decoded.extend(encoding.decode_single_token_bytes(token)); right = len(decoded)
        overlaps = [(start, stop) for start, stop in ranges if left < stop and right > start]
        if not overlaps:
            masks.append(0)
        elif any(left >= start and right <= stop for start, stop in overlaps):
            masks.append(1)
        else:
            raise ValueError("token crosses target boundary")
    if bytes(decoded) != text.encode():
        raise ValueError("token bytes do not reconstruct record")
    return tokens, masks, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cache = Path(os.environ.get("TIKTOKEN_CACHE_DIR", "")) / TOKENIZER_CACHE_KEY
    if not cache.is_file() or sha256(cache) != TOKENIZER_SHA256:
        raise SystemExit("verified p50k tokenizer cache is required")
    manifest_path = args.input_root / "manifest.json"
    if sha256(manifest_path) != args.input_manifest_sha256:
        raise SystemExit("input manifest mismatch")
    source = json.loads(manifest_path.read_text())
    if source.get("schema") != AUTHORITY_SCHEMA or source.get("status") != "complete":
        raise SystemExit("input is not a complete masked-SFT authority")
    paths = {key: args.input_root / Path(source["outputs"][key]["path"]).name
             for key in ("tokens", "mask", "index")}
    for key, path in paths.items():
        expected = source["outputs"][key]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise SystemExit(f"input {key} integrity mismatch")
    records = np.memmap(paths["index"], mode="r", dtype=np.dtype([
        ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
        ("split", "u1"), ("pad", "V7")]))
    encoding = tiktoken.get_encoding("p50k_base")
    old = f"System:\n{E97_PI_CORE_SYSTEM}\n\n"
    new = f"System:\n{E97_PI_AGENT_SYSTEM_V2}\n\n"
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {"tokens": args.output_root / "tokens.uint32.bin",
               "mask": args.output_root / "assistant_mask.uint8.bin",
               "index": args.output_root / "records.idx",
               "metadata": args.output_root / "records.jsonl"}
    with paths["tokens"].open("rb") as tf, paths["mask"].open("rb") as mf:
        token_map = mmap.mmap(tf.fileno(), 0, access=mmap.ACCESS_READ)
        mask_map = mmap.mmap(mf.fileno(), 0, access=mmap.ACCESS_READ)
        offset = total_tokens = total_targets = rewritten_records = 0
        split_counts = [0, 0]
        with outputs["tokens"].open("wb") as token_out, outputs["mask"].open("wb") as mask_out, \
             outputs["index"].open("wb") as index_out, outputs["metadata"].open("w") as metadata_out:
            for record_id, record in enumerate(records):
                source_offset, count = int(record["offset"]), int(record["tokens"])
                ids = struct.unpack(f"<{count}I", token_map[source_offset * 4:(source_offset + count) * 4])
                masks = mask_map[source_offset:source_offset + count]
                pieces, current, current_target = [], bytearray(), None
                for token_id, target in zip(ids, masks, strict=True):
                    target = bool(target)
                    if current_target is not None and target != current_target:
                        pieces.append((current.decode(), current_target)); current = bytearray()
                    current.extend(encoding.decode_single_token_bytes(token_id)); current_target = target
                if current_target is not None:
                    pieces.append((current.decode(), current_target))
                complete = "".join(value for value, _ in pieces)
                replaced = False
                adjusted = []
                for value, target in pieces:
                    if complete.startswith(old) and not replaced and old in value:
                        if target:
                            raise SystemExit(f"record {record_id} targets its system prompt")
                        value = value.replace(old, new, 1); replaced = True
                    adjusted.append((value, target))
                if complete.startswith(old) and not replaced:
                    raise SystemExit(f"record {record_id} prompt boundary was not replaceable")
                rewritten_records += int(replaced)
                new_tokens, new_masks, text = encode_pieces(encoding, adjusted)
                token_out.write(struct.pack(f"<{len(new_tokens)}I", *new_tokens)); mask_out.write(bytes(new_masks))
                target_count, split = sum(new_masks), int(record["split"])
                index_out.write(RECORD_INDEX.pack(offset, len(new_tokens), target_count, split))
                metadata_out.write(json.dumps({
                    "id": f"pi-v2-replay-{record_id:09d}", "source": "pi-v2-retention",
                    "source_record_id": record_id, "split": split, "tokens": len(new_tokens),
                    "targets": target_count, "serialization_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }, sort_keys=True) + "\n")
                offset += len(new_tokens); total_tokens += len(new_tokens); total_targets += target_count
                split_counts[split] += 1
        token_map.close(); mask_map.close()
    output_entries = {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                             "sha256": sha256(path)} for name, path in outputs.items()}
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "purpose": "Pi cumulative retention replay under grounded system prompt v2",
        "source_manifest_sha256": args.input_manifest_sha256,
        "old_system_prompt": E97_PI_CORE_SYSTEM, "system_prompt": E97_PI_AGENT_SYSTEM_V2,
        "counts": {"records": len(records), "rewritten_pi_records": rewritten_records,
                   "unmodified_non_pi_records": len(records) - rewritten_records,
                   "tokens": total_tokens, "assistant_target_tokens": total_targets,
                   "train_records": split_counts[0], "validation_records": split_counts[1]},
        "outputs": output_entries,
    }
    output_manifest = args.output_root / "manifest.json"
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(output_manifest), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
