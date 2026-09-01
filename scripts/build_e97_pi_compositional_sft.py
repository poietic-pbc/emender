#!/usr/bin/env python3
"""Build live-aligned compositional Pi trajectories from the frozen v2 gaps."""
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
from scripts.build_e97_pi_eval_v2 import KINDS, trace
from scripts.build_e97_pi_finalization_repair_sft import serialize_live_aligned
from scripts.build_e97_pi_instruction_sft import ENCODING, action, read_result, split


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def trajectory(kind: str, index: int, rng: random.Random):
    user, task = trace(kind, index, rng)
    state = {fixture["path"]: fixture["content"] for fixture in task["fixtures"]}
    turns: list[tuple[str, str]] = []
    for position, declared in enumerate(task["expected_calls"]):
        name, args = declared["name"], declared["args"]
        turns.append(action(name, args))
        if name == "read":
            text = state[args["path"]].rstrip("\n").splitlines()
            result = read_result(args["path"], text, int(args.get("offset", 1)))
        elif name == "write":
            state[args["path"]] = args["content"]
            result = f"Successfully wrote {len(args['content'].encode())} bytes to {args['path']}"
        elif name == "edit":
            current = state[args["path"]]
            count = current.count(args["oldText"])
            if count != 1:
                result = (
                    f"AssertionError: expected one exact block, found {count}\n"
                    "Command exited with code 1"
                )
            else:
                state[args["path"]] = current.replace(args["oldText"], args["newText"])
                result = f"Successfully replaced 1 block(s) in {args['path']}."
        elif name == "bash":
            if kind == "diagnose-test" and position == 0:
                result = (
                    "Traceback (most recent call last):\n"
                    "  File \"<string>\", line 1, in <module>\n"
                    "AssertionError\nCommand exited with code 1"
                )
            elif kind == "search-edit" and position == 0:
                result = next(path for path in state if path.startswith("services/") and "archive_" not in path) + "\n"
            else:
                result = ""
        else:
            raise ValueError(name)
        turns.append(("tool", result))
    evidence = ", ".join(f"`{value}`" for value in task["final_contains"])
    turns.append(("assistant", f"Final: Completed the requested {kind} workflow and verified {evidence}."))
    return user, turns, task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=3_701_077)
    args = parser.parse_args()
    if args.records <= 0 or args.records % len(KINDS):
        raise SystemExit(f"records must be a positive multiple of {len(KINDS)}")
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
    kind_counts = {kind: 0 for kind in KINDS}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, \
         paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for record_index in range(args.records):
            kind = KINDS[record_index % len(KINDS)]
            identity = f"pi-compositional-{kind}-{record_index:09d}"
            # The training domain is disjoint from frozen v2 identities and seed.
            source_index = 1_000_000 + record_index
            user, turns, task = trajectory(
                kind, source_index, random.Random(args.seed + record_index))
            tokens, masks, complete = serialize_live_aligned(
                [("system", E97_PI_CORE_SYSTEM), ("user", user), *turns],
                encoding,
                target_mode="all-assistant",
            )
            validation = int(split(identity))
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_out.write(json.dumps({
                "id": identity,
                "source": "emender-pi-compositional-live-v1",
                "split": validation,
                "kind": kind,
                "tokens": len(tokens),
                "targets": sum(masks),
                "user": user,
                "task_sha256": hashlib.sha256(
                    json.dumps(task, sort_keys=True).encode()).hexdigest(),
                "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
            }, sort_keys=True) + "\n")
            offset += len(tokens)
            counts["records"] += 1
            counts["tokens"] += len(tokens)
            counts["assistant_target_tokens"] += sum(masks)
            counts["validation_records" if validation else "train_records"] += 1
            kind_counts[kind] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": "live-aligned compositional Pi core-tools instruction tuning",
        "serialization": (
            "RS-free exact live empty-tool context; all assistant actions and final newline targeted"
        ),
        "evaluation_exclusion": {
            "authority": "pi-core-eval-v2-template-heldout",
            "manifest_sha256": "b7d308bbcaaa6526234fadd8f59a77be5f2f4cfac3cd67d38f584863c9444c29",
            "policy": "different identities, index domain, seed, values, paths, and payloads",
        },
        "system_prompt": E97_PI_CORE_SYSTEM,
        "tokenizer": ENCODING,
        "seed": args.seed,
        "source_index_offset": 1_000_000,
        "kinds": list(KINDS),
        "kind_counts": kind_counts,
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
