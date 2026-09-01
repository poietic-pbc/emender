#!/usr/bin/env python3
"""Build disjoint live-aligned missing-path recovery trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import E97_PI_CORE_SYSTEM
from scripts.build_e97_pi_finalization_repair_sft import serialize_live_aligned
from scripts.build_e97_pi_instruction_sft import ENCODING, split, trace


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=5_701_111)
    args = parser.parse_args()
    if args.records <= 0:
        raise SystemExit("records must be positive")
    args.output_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    encoding = tiktoken.get_encoding(ENCODING)
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0,
              "train_records": 0, "validation_records": 0}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for record_index in range(args.records):
            identity = f"pi-recover-read-live-{record_index:09d}"
            source_index = 2_000_000 + record_index
            user, turns, task = trace(
                "recover-read", source_index, random.Random(args.seed + record_index))
            tokens, masks, complete = serialize_live_aligned(
                [("system", E97_PI_CORE_SYSTEM), ("user", user), *turns],
                encoding, target_mode="all-assistant")
            validation = int(split(identity))
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_out.write(json.dumps({
                "id": identity, "source": "emender-pi-recover-read-live-v1",
                "source_index": source_index, "split": validation,
                "kind": "recover-read", "tokens": len(tokens), "targets": sum(masks),
                "user": user,
                "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
            }, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += sum(masks)
            counts["validation_records" if validation else "train_records"] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "purpose": "live-aligned missing-path read recovery retention repair",
        "serialization": "all assistant actions/final/newline targeted; exact live tool context",
        "evaluation_exclusion": {
            "smoke_authority_manifest_sha256": "48f6b7ecb0083f09402e2f0715b95d7ca71ba45a2711375b811472ccdeb804e1",
            "policy": "disjoint identity, two-million index domain, seed, values, and paths",
        },
        "tokenizer": ENCODING, "seed": args.seed, "source_index_offset": 2_000_000,
        "kinds": ["recover-read"], "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
