#!/usr/bin/env python3
"""Build RS-free, typed, mechanically grounded dense-agent trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import DENSE_AGENT_V2_SYSTEM

ENCODING = "p50k_base"
SYSTEM = DENSE_AGENT_V2_SYSTEM


def split(identity: str) -> int:
    return int(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 100 == 0)


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def action(name: str, arguments: dict[str, object]) -> tuple[str, str]:
    return "assistant", f"Action: {name}\nArguments: {json.dumps(arguments, separators=(',', ':'))}"


def trace(kind: str, index: int, rng: random.Random):
    if kind == "calculator":
        left, right = rng.randint(11, 999), rng.randint(11, 999)
        operator = rng.choice(["+", "-", "*"])
        expression = f"{left} {operator} {right}"
        value = str({"+": left + right, "-": left - right, "*": left * right}[operator])
        return f"Calculate {expression}.", [
            action("calculator", {"expression": expression}),
            ("tool", json.dumps({"expression": expression, "value": value}, separators=(",", ":"))),
            action("submit_answer", {"value": value}),
        ]
    if kind == "lookup":
        project = f"Project-{index:06d}"
        owner = rng.choice(["Amina", "Boris", "Chen", "Devika", "Elena", "Farid"])
        budget = f"${rng.randint(20, 900) * 1000:,}"
        field = rng.choice(["owner", "budget"])
        value = owner if field == "owner" else budget
        path = f"records/{project}.txt"
        return f"What is the {field} of {project}?", [
            action("lookup", {"path": path, "field": field}),
            ("tool", json.dumps({"field": field, "value": value}, separators=(",", ":"))),
            action("submit_answer", {"value": value}),
        ]
    if kind == "count":
        suffix = rng.choice([".md", ".py", ".json", ".txt"])
        count = rng.randint(2, 18)
        return f"How many {suffix} files are in data/?", [
            action("count", {"path": "data", "suffix": suffix}),
            ("tool", json.dumps({"count": count}, separators=(",", ":"))),
            action("submit_answer", {"value": str(count)}),
        ]
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=9702)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    encoding = tiktoken.get_encoding(ENCODING)
    paths = {
        "tokens": args.output_root / "tokens.uint32.bin",
        "mask": args.output_root / "assistant_mask.uint8.bin",
        "index": args.output_root / "records.idx",
        "metadata": args.output_root / "records.jsonl",
    }
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0, "train_records": 0, "validation_records": 0}
    offset = 0
    with paths["tokens"].open("wb") as token_file, paths["mask"].open("wb") as mask_file, paths["index"].open("wb") as index_file, paths["metadata"].open("w") as metadata_file:
        for index in range(args.records):
            kind = ("calculator", "lookup", "count")[index % 3]
            identity = f"agent-v2-{kind}-{index:08d}"
            user, turns = trace(kind, index, random.Random(args.seed + index))
            messages = [("system", SYSTEM), ("user", user), *turns]
            pieces: list[tuple[str, bool]] = []
            for position, (role, text) in enumerate(messages):
                if position:
                    pieces.append(("\n\n", False))
                label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}[role]
                pieces.append((f"{label}:\n", False))
                pieces.append((text, role == "assistant"))
            complete = "".join(text for text, _ in pieces)
            if "\x1e" in complete:
                raise RuntimeError("v2 trajectories must not contain RS")
            target_ranges = []
            cursor = 0
            for text, target in pieces:
                stop = cursor + len(text.encode())
                if target:
                    target_ranges.append((cursor, stop))
                cursor = stop
            tokens = encoding.encode_ordinary(complete)
            masks = []
            decoded = bytearray()
            for token in tokens:
                left = len(decoded)
                token_bytes = encoding.decode_single_token_bytes(token)
                decoded.extend(token_bytes)
                right = len(decoded)
                overlaps = [(start, stop) for start, stop in target_ranges if left < stop and right > start]
                if not overlaps:
                    masks.append(0)
                elif any(left >= start and right <= stop for start, stop in overlaps):
                    masks.append(1)
                else:
                    raise RuntimeError("token crosses target boundary")
            validation = split(identity)
            token_file.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_file.write(bytes(masks))
            index_file.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_file.write(json.dumps({"id": identity, "source": f"emender-agent-{kind}-v2", "split": validation, "tokens": len(tokens), "targets": sum(masks)}, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += sum(masks)
            counts["validation_records" if validation else "train_records"] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": "dense-e97-bounded-grounded-agent-v2",
        "serialization": "RS-free System/User/Assistant/Tool; structured assistant actions only",
        "seed": args.seed,
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
