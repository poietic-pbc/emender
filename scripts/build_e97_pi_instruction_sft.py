#!/usr/bin/env python3
"""Build deterministic, RS-free, Pi-native masked-SFT trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path
from typing import Iterable

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import E97_PI_CORE_SYSTEM

ENCODING = "p50k_base"
SYSTEM = E97_PI_CORE_SYSTEM


def split(identity: str) -> int:
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 100 == 0


def action(name: str, arguments: dict[str, object]) -> tuple[str, str]:
    return "assistant", f"Action: {name}\nArguments: {json.dumps(arguments, separators=(',', ':'))}"


def read_result(path: str, lines: Iterable[str], offset: int = 1) -> str:
    return "\n".join(f"{number}: {line}" for number, line in enumerate(lines, offset))


def trace(kind: str, index: int, rng: random.Random):
    token = f"{rng.choice(('amber', 'cobalt', 'indigo', 'saffron'))}-{rng.randint(1000, 9999)}"
    fixtures: list[dict[str, str]] = []
    if kind == "read":
        path = f"docs/fact-{index:06d}.txt"
        content = f"project fact\nrelease_token={token}\nstatus=approved\n"
        fixtures.append({"path": path, "content": content})
        user = f"Read {path} and report the exact approved release token."
        turns = [
            action("read", {"path": path, "offset": 1, "limit": 80}),
            ("tool", read_result(path, content.rstrip().splitlines())),
            ("assistant", f"Final: The approved release token is `{token}` from `{path}` line 2."),
        ]
        verifier = {"kind": "exact_text", "expected": token}
    elif kind == "bash":
        path = f"configs/service-{index:06d}.json"
        port = rng.randint(2000, 9000)
        content = json.dumps({"service": {"port": port, "release": token}}, separators=(",", ":")) + "\n"
        fixtures.append({"path": path, "content": content})
        command = f"python -c 'import json; print(json.load(open(\"{path}\"))[\"service\"][\"port\"])'"
        user = f"Use the repository to determine the configured service port in {path}."
        turns = [
            action("bash", {"command": command}),
            ("tool", f"{port}\n"),
            ("assistant", f"Final: `{path}` configures service port `{port}`; the Python inspection command printed `{port}`."),
        ]
        verifier = {"kind": "exact_text", "expected": str(port)}
    elif kind == "edit":
        path = f"src/settings_{index:06d}.py"
        old = f'RELEASE = "draft-{token}"'
        new = f'RELEASE = "{token}"'
        content = old + "\n"
        fixtures.append({"path": path, "content": content})
        command = f"python -c 'import runpy; assert runpy.run_path(\"{path}\")[\"RELEASE\"] == \"{token}\"'"
        user = f"Change the RELEASE value in {path} from draft-{token} to {token}, then verify it."
        turns = [
            action("read", {"path": path, "offset": 1, "limit": 40}),
            ("tool", read_result(path, [old])),
            action("edit", {"path": path, "oldText": old, "newText": new}),
            ("tool", f"Successfully replaced 1 block(s) in {path}."),
            action("bash", {"command": command}),
            ("tool", ""),
            ("assistant", f"Final: Updated `{path}` to `RELEASE = \"{token}\"` and verified the value with the Python assertion."),
        ]
        verifier = {"kind": "file_and_command", "path": path, "contains": new, "command": command}
    elif kind == "write":
        path = f"metadata/release-{index:06d}.json"
        content = json.dumps({"release": token, "approved": True}, indent=2, sort_keys=True) + "\n"
        command = f"python -m json.tool {path} >/dev/null"
        user = f"Create {path} as JSON with release {token} and approved true, then validate it."
        turns = [
            action("write", {"path": path, "content": content}),
            ("tool", f"Successfully wrote {len(content.encode())} bytes to {path}"),
            action("bash", {"command": command}),
            ("tool", ""),
            ("assistant", f"Final: Created `{path}` with release `{token}` and `approved: true`; `python -m json.tool` validated it."),
        ]
        verifier = {"kind": "file_and_command", "path": path, "content": content, "command": command}
    elif kind == "recover-read":
        path = f"docs/authority-{index:06d}.txt"
        wrong = f"docs/authorities-{index:06d}.txt"
        content = f"authority={token}\n"
        fixtures.append({"path": path, "content": content})
        user = f"Find the authority value in {path}. Recover if an attempted path is missing."
        turns = [
            action("read", {"path": wrong, "offset": 1, "limit": 40}),
            ("tool", f"File not found: {wrong}"),
            action("read", {"path": path, "offset": 1, "limit": 40}),
            ("tool", read_result(path, [f"authority={token}"])),
            ("assistant", f"Final: The authority value is `{token}` from `{path}`. I corrected the initial path and read the requested file."),
        ]
        verifier = {"kind": "recovery_exact_text", "expected": token}
    elif kind == "recover-test":
        path = f"src/limit_{index:06d}.py"
        old = "LIMIT = 2"
        new_value = rng.randint(3, 12)
        new = f"LIMIT = {new_value}"
        content = old + "\n"
        fixtures.append({"path": path, "content": content})
        command = f"python -c 'import runpy; assert runpy.run_path(\"{path}\")[\"LIMIT\"] == {new_value}'"
        user = f"Set LIMIT in {path} to {new_value}. Run the focused assertion and repair the file if it fails."
        turns = [
            action("bash", {"command": command}),
            ("tool", "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nAssertionError\n\nCommand exited with code 1"),
            action("read", {"path": path, "offset": 1, "limit": 40}),
            ("tool", read_result(path, [old])),
            action("edit", {"path": path, "oldText": old, "newText": new}),
            ("tool", f"Successfully replaced 1 block(s) in {path}."),
            action("bash", {"command": command}),
            ("tool", ""),
            ("assistant", f"Final: Set `LIMIT = {new_value}` in `{path}` and reran the focused assertion successfully after the initial failure."),
        ]
        verifier = {"kind": "recovery_file_and_command", "path": path, "contains": new, "command": command}
    else:
        raise ValueError(kind)
    return user, turns, {"fixtures": fixtures, "verifier": verifier}


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def serialize(messages: list[tuple[str, str]], encoding) -> tuple[list[int], list[int], str]:
    pieces: list[tuple[str, bool]] = []
    for position, (role, text) in enumerate(messages):
        if position:
            pieces.append(("\n\n", False))
        label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}[role]
        pieces.extend(((f"{label}:\n", False), (text, role == "assistant")))
    complete = "".join(text for text, _ in pieces)
    if "\x1e" in complete:
        raise RuntimeError("Pi trajectories must be RS-free")
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
            raise RuntimeError("token crosses an assistant target boundary")
    return tokens, masks, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=974001)
    parser.add_argument("--kinds", default="read,bash,edit,write,recover-read,recover-test")
    args = parser.parse_args()
    if args.records <= 0:
        raise SystemExit("records must be positive")
    kinds = tuple(part.strip() for part in args.kinds.split(",") if part.strip())
    valid = {"read", "bash", "edit", "write", "recover-read", "recover-test"}
    if not kinds or any(kind not in valid for kind in kinds):
        raise SystemExit("unsupported Pi task kind")
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
    kind_counts = {kind: 0 for kind in kinds}
    offset = 0
    with paths["tokens"].open("wb") as token_out, paths["mask"].open("wb") as mask_out, paths["index"].open("wb") as index_out, paths["metadata"].open("w") as metadata_out:
        for index in range(args.records):
            kind = kinds[index % len(kinds)]
            identity = f"pi-native-{kind}-{index:08d}"
            user, turns, task = trace(kind, index, random.Random(args.seed + index))
            tokens, masks, complete = serialize([("system", SYSTEM), ("user", user), *turns], encoding)
            validation = int(split(identity))
            token_out.write(struct.pack(f"<{len(tokens)}I", *tokens))
            mask_out.write(bytes(masks))
            index_out.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_out.write(json.dumps({
                "id": identity, "source": "emender-pi-native-core-tools-v1",
                "split": validation, "kind": kind, "tokens": len(tokens),
                "targets": sum(masks), "user": user, "task": task,
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
        "purpose": "E97 4B exact-Pi core-tool instruction tuning",
        "serialization": "RS-free System/User/Assistant/Tool; assistant action and final targets only",
        "system_prompt": SYSTEM,
        "tool_contract": {
            "read": ["path", "offset", "limit"], "bash": ["command"],
            "edit": ["path", "oldText", "newText"], "write": ["path", "content"],
        },
        "tokenizer": ENCODING,
        "seed": args.seed,
        "kinds": list(kinds),
        "kind_counts": kind_counts,
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
