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
from ndm.e97_agent_protocol import DENSE_AGENT_CLI_DIRECT_SYSTEM, DENSE_AGENT_CLI_SYSTEM
try:
    from scripts.e97_repo_cli import parser as repo_parser
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from e97_repo_cli import parser as repo_parser

ENCODING = "p50k_base"
SYSTEM = DENSE_AGENT_CLI_SYSTEM


def split(identity: str) -> int:
    return int(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % 100 == 0)


def action(name: str, arguments: dict[str, object]) -> tuple[str, str]:
    return "assistant", f"Action: {name}\nArguments: {json.dumps(arguments, separators=(',', ':'))}"


def help_summary(stdout: str) -> str:
    usage_lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("usage:") or (usage_lines and line.startswith(" ")):
            usage_lines.append(line.strip())
        elif usage_lines:
            break
    summary = " ".join(" ".join(usage_lines).split())
    import re
    summary = re.sub(r"\{(?:\d+,){8,}\d+\}", "INTEGER", summary)
    return summary + "\n"


def cli_observation(
    argv: list[str], stdout: str, *, exit_code: int = 0, stderr: str = "", compact: bool = False
) -> str:
    if compact:
        if exit_code != 0:
            value = {
                "ok": False,
                "phase": "RECOVER",
                "exit_code": exit_code,
                "stderr": stderr,
                "instruction": "correct argv and do not repeat an identical failed command",
            }
        elif argv[-1:] == ["--help"]:
            top_level = argv == ["repo", "--help"]
            value = {
                "ok": True,
                "phase": "DISCOVER_COMMAND" if top_level else "DISCOVER_OPTIONS",
                "usage": help_summary(stdout).rstrip("\n"),
                "instruction": (
                    "choose the relevant subcommand and inspect its help; do not repeat this command"
                    if top_level else
                    "construct and execute argv using these options; do not repeat help"
                ),
            }
        else:
            value = {
                "ok": True,
                "phase": "INTERPRET",
                "stdout": stdout,
                "instruction": "extract an exact value and evidence, then submit",
            }
        return json.dumps(value, separators=(",", ":"))
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


def trace(
    kind: str,
    index: int,
    rng: random.Random,
    discover: bool,
    *,
    compact: bool = False,
    subcommand_help: bool = False,
):
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
        turns.extend([action("cli", {"argv": help_argv}), ("tool", cli_observation(help_argv, help_stdout, compact=compact))])
        if subcommand_help:
            root_parser = repo_parser()
            subparsers = next(action for action in root_parser._actions if hasattr(action, "choices") and action.choices)
            command_help_argv = ["repo", kind, "--help"]
            command_help_stdout = subparsers.choices[kind].format_help()
            turns.extend([
                action("cli", {"argv": command_help_argv}),
                ("tool", cli_observation(command_help_argv, command_help_stdout, compact=compact)),
            ])
    turns.extend([
        action("cli", {"argv": argv}),
        ("tool", cli_observation(argv, stdout, compact=compact)),
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
    argument_parser.add_argument("--curriculum", choices=("mixed", "direct", "discovery"), default="mixed")
    argument_parser.add_argument("--compact-observations", action="store_true")
    argument_parser.add_argument("--kinds", default="json,count,search,read")
    argument_parser.add_argument("--discovery-period", type=int, default=5)
    argument_parser.add_argument("--target-stage", choices=("all", "discover-command", "discover-options"), default="all")
    args = argument_parser.parse_args()
    if args.target_stage != "all" and args.curriculum != "discovery":
        raise ValueError("transition target stages require --curriculum discovery")
    args.output_root.mkdir(parents=True, exist_ok=False)
    encoding = tiktoken.get_encoding(ENCODING)
    paths = {name: args.output_root / filename for name, filename in (
        ("tokens", "tokens.uint32.bin"), ("mask", "assistant_mask.uint8.bin"),
        ("index", "records.idx"), ("metadata", "records.jsonl"),
    )}
    kinds = tuple(part.strip() for part in args.kinds.split(",") if part.strip())
    if not kinds or any(kind not in {"json", "count", "search", "read"} for kind in kinds):
        raise ValueError("--kinds must contain only json,count,search,read")
    counts = {"records": 0, "tokens": 0, "assistant_target_tokens": 0, "train_records": 0, "validation_records": 0, "discovery_records": 0}
    offset = 0
    with paths["tokens"].open("wb") as token_file, paths["mask"].open("wb") as mask_file, paths["index"].open("wb") as index_file, paths["metadata"].open("w") as metadata_file:
        for index in range(args.records):
            kind = kinds[index % len(kinds)]
            identity = f"agent-cli-{kind}-{index:08d}"
            if args.discovery_period < 1:
                raise ValueError("--discovery-period must be positive")
            discover = args.curriculum == "discovery" or (
                args.curriculum == "mixed" and (index // len(kinds)) % args.discovery_period == 0
            )
            user, turns, task = trace(
                kind,
                index,
                random.Random(args.seed + index),
                discover,
                compact=args.compact_observations,
                subcommand_help=discover,
            )
            if discover:
                user = "Inspect repo --help and the relevant subcommand help before answering. " + user
            system = DENSE_AGENT_CLI_DIRECT_SYSTEM if args.curriculum == "direct" else SYSTEM
            messages = [("system", system), ("user", user), *turns]
            pieces: list[tuple[str, bool]] = []
            assistant_index = 0
            target_index = {"discover-command": 1, "discover-options": 2}.get(args.target_stage)
            for position, (role, text) in enumerate(messages):
                if position:
                    pieces.append(("\n\n", False))
                label = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}[role]
                is_target = role == "assistant" and (target_index is None or assistant_index == target_index)
                pieces.extend([(f"{label}:\n", False), (text, is_target)])
                if role == "assistant":
                    assistant_index += 1
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
    manifest = {
        "schema": AUTHORITY_SCHEMA,
        "status": "complete",
        "purpose": f"dense-e97-cwd-cli-agent-{args.curriculum}-v2",
        "serialization": "RS-free CLI argv, compact typed observations, and grounded submit_answer" if args.compact_observations else "RS-free CLI argv and grounded submit_answer",
        "curriculum": args.curriculum,
        "compact_observations": args.compact_observations,
        "seed": args.seed,
        "kinds": list(kinds),
        "discovery_period": args.discovery_period,
        "target_stage": args.target_stage,
        "counts": counts,
        "outputs": {name: entry(path) for name, path in paths.items()},
    }
    manifest_path = args.output_root / "manifest.json"; manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
