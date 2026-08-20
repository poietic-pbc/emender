#!/usr/bin/env python3
"""Build deterministic RS-free trajectories for the discoverable sandbox CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path

import tiktoken

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
try:
    from scripts.e97_repo_cli import parser as repo_parser
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from e97_repo_cli import parser as repo_parser

ENCODING = "p50k_base"
SYSTEM = (
    "Work only in the current directory. Use cli with an argv array. Use repo --help when "
    "you need to discover repository commands. Then call submit_answer with an exact value "
    "and exact evidence copied from successful CLI stdout. Respond only with Action and Arguments."
)


def split(identity: str) -> int:
    return int(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 100 == 0)


def action(name: str, arguments: dict[str, object]) -> tuple[str, str]:
    return "assistant", f"Action: {name}\nArguments: {json.dumps(arguments, separators=(',', ':'))}"


def cli_observation(argv: list[str], stdout: str, *, exit_code: int = 0, stderr: str = "") -> str:
    return json.dumps({
        "argv": argv,
        "cwd": ".",
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "timed_out": False,
    }, separators=(",", ":"))


def repo_stdout(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True) + "\n"


def trace(kind: str, index: int, rng: random.Random, discover: bool):
    fixtures: list[dict[str, str]] = []
    if kind == "json":
        tokenizer = rng.choice(["p50k_base", "cl100k_base", "gpt2", "o200k_base"])
        depth = rng.randint(4, 48)
        path = f"configs/model-{index:06d}.json"
        pointer = rng.choice(["/model/tokenizer", "/model/depth"])
        value: object = tokenizer if pointer.endswith("tokenizer") else depth
        fixtures.append({"path": path, "content": json.dumps({"model": {"tokenizer": tokenizer, "depth": depth}}, separators=(",", ":")) + "\n"})
        user = f"What is the {pointer.rsplit('/', 1)[1]} in {path}?"
        argv = ["repo", "json", "--path", path, "--pointer", pointer]
        stdout = repo_stdout({"path": path, "pointer": pointer, "value": value})
        answer, evidence = str(value), f'"value":{json.dumps(value, separators=(",", ":"))}'
    elif kind == "count":
        suffix = rng.choice([".py", ".md", ".json", ".txt"])
        count = rng.randint(1, 24)
        directory = f"data-{index:06d}"
        fixtures.extend({"path": f"{directory}/item-{item:03d}{suffix}", "content": "fixture\n"} for item in range(count))
        fixtures.append({"path": f"{directory}/ignore.other", "content": "fixture\n"})
        pattern = f"*{suffix}"
        user = f"How many {suffix} files are directly in {directory}?"
        argv = ["repo", "count", "--path", directory, "--pattern", pattern]
        stdout = repo_stdout({"count": count, "path": directory, "pattern": pattern})
        answer, evidence = str(count), f'"count":{count}'
    elif kind == "search":
        component = f"component-{index:06d}"
        version = f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}"
        path = f"notes/release-{index:06d}.txt"
        line = f"{component} version {version} is approved"
        fixtures.append({"path": path, "content": f"release notes\n{line}\n"})
        user = f"Find the approved version of {component}."
        argv = ["repo", "search", "--query", component, "--path", "notes", "--pattern", "*.txt"]
        stdout = repo_stdout({"matches": [{"line": 2, "path": path, "text": line}], "truncated": False})
        answer, evidence = version, line
    elif kind == "read":
        path = f"docs/facts-{index:06d}.txt"
        value = rng.choice(["amber", "cobalt", "indigo", "saffron"]) + f"-{rng.randint(1000,9999)}"
        target_line = rng.randint(2, 5)
        lines = [f"filler-{item}" for item in range(1, 7)]
        lines[target_line - 1] = value
        fixtures.append({"path": path, "content": "\n".join(lines) + "\n"})
        user = f"What exact value is on line {target_line} of {path}?"
        argv = ["repo", "read", "--path", path, "--start", str(target_line), "--end", str(target_line)]
        stdout = repo_stdout({"end": target_line, "lines": [{"line": target_line, "text": value}], "path": path, "start": target_line})
        answer, evidence = value, value
    else:
        raise ValueError(kind)

    turns: list[tuple[str, str]] = []
    if discover:
        help_argv = ["repo", "--help"]
        help_stdout = repo_parser().format_help()
        turns.extend([action("cli", {"argv": help_argv}), ("tool", cli_observation(help_argv, help_stdout))])
    turns.extend([
        action("cli", {"argv": argv}),
        ("tool", cli_observation(argv, stdout)),
        action("submit_answer", {"value": answer, "evidence": evidence}),
    ])
    return user, turns, {"fixtures": fixtures, "expected": answer, "evidence": evidence, "argv": argv, "discover": discover}


def entry(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--output-root", type=Path, required=True)
    argument_parser.add_argument("--records", type=int, default=30_000)
    argument_parser.add_argument("--seed", type=int, default=9704)
    args = argument_parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    encoding = tiktoken.get_encoding(ENCODING)
    paths = {name: args.output_root / filename for name, filename in (
        ("tokens", "tokens.uint32.bin"), ("mask", "assistant_mask.uint8.bin"),
        ("index", "records.idx"), ("metadata", "records.jsonl"),
    )}
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0, "train_records": 0, "validation_records": 0, "discovery_records": 0}
    offset = 0
    with paths["tokens"].open("wb") as token_file, paths["mask"].open("wb") as mask_file, paths["index"].open("wb") as index_file, paths["metadata"].open("w") as metadata_file:
        for index in range(args.records):
            kind = ("json", "count", "search", "read")[index % 4]
            identity = f"agent-cli-{kind}-{index:08d}"
            discover = index % 5 == 0
            user, turns, task = trace(kind, index, random.Random(args.seed + index), discover)
            messages = [("system", SYSTEM), ("user", user), *turns]
            pieces: list[tuple[str, bool]] = []
            for position, (role, text) in enumerate(messages):
                if position:
                    pieces.append(("\n\n", False))
                label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}[role]
                pieces.extend([(f"{label}:\n", False), (text, role == "assistant")])
            complete = "".join(text for text, _ in pieces)
            if "\x1e" in complete:
                raise RuntimeError("CLI trajectories must not contain RS")
            ranges = []
            cursor = 0
            for text, target in pieces:
                stop = cursor + len(text.encode())
                if target:
                    ranges.append((cursor, stop))
                cursor = stop
            tokens = encoding.encode_ordinary(complete)
            masks, decoded = [], bytearray()
            for token in tokens:
                left = len(decoded); decoded.extend(encoding.decode_single_token_bytes(token)); right = len(decoded)
                overlaps = [(start, stop) for start, stop in ranges if left < stop and right > start]
                if not overlaps: masks.append(0)
                elif any(left >= start and right <= stop for start, stop in overlaps): masks.append(1)
                else: raise RuntimeError("token crosses target boundary")
            validation = split(identity)
            token_file.write(struct.pack(f"<{len(tokens)}I", *tokens)); mask_file.write(bytes(masks)); index_file.write(RECORD_INDEX.pack(offset, len(tokens), sum(masks), validation))
            metadata_file.write(json.dumps({"id": identity, "source": f"emender-agent-cli-{kind}-v1", "split": validation, "tokens": len(tokens), "targets": sum(masks), "user": user, "task": task}, sort_keys=True) + "\n")
            offset += len(tokens); counts["records"] += 1; counts["tokens"] += len(tokens); counts["assistant_target_tokens"] += sum(masks); counts["validation_records" if validation else "train_records"] += 1; counts["discovery_records"] += int(discover)
    manifest = {"schema": AUTHORITY_SCHEMA, "status": "complete", "purpose": "dense-e97-discoverable-cwd-cli-agent-v1", "serialization": "RS-free CLI argv and grounded submit_answer", "seed": args.seed, "counts": counts, "outputs": {name: entry(path) for name, path in paths.items()}}
    manifest_path = args.output_root / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
