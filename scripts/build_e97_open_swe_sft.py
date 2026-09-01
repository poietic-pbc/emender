#!/usr/bin/env python3
"""Normalize verified OpenHands trajectories into the bounded Pi action protocol."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import struct

import pyarrow.parquet as pq

from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX, sha256
from ndm.e97_agent_protocol import E97_PI_AGENT_SYSTEM_V2
from scripts import build_e97_tulu3_sft as codec

DATASET_ID = "nvidia/Open-SWE-Traces"
DATASET_REVISION = "0b7d2a801a3b91541a48f8bca03e5ea90fd1fa5c"
DATASET_CARD_SHA256 = "ae4642e7b3e312483e9cbb7a19f2212d7a3b5f4c795e927d35e4af9f7dbeffcf"
ALLOWED_LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
EXCLUDED_REPOSITORIES = {
    "pallets/markupsafe", "python-humanize/humanize",
    "more-itertools/more-itertools", "prettytable/prettytable",
}
_WORKSPACE = re.compile(r"/(?:workspace|testbed)(?:/[^/\s'\"]+)?/")


def worker_init() -> None:
    codec._worker_init()


def split(identity: str) -> int:
    value = hashlib.sha256(f"open-swe-pi-v1\0{identity}".encode()).digest()
    return int.from_bytes(value[:8], "little") % 100 == 0


def normalize_text(text: str) -> str:
    text = text.replace(codec.RS, " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _WORKSPACE.sub("", text).replace("/workspace/", "").replace("/testbed/", "")
    return text.strip()


def normalize_path(path: str) -> str:
    path = normalize_text(path).strip()
    if path in {"/workspace", "/testbed", "workspace", "testbed", ""}:
        return "."
    return path.removeprefix("./") or "."


def normalize_command(command: str) -> str:
    command = command.replace(codec.RS, " ").replace("\r\n", "\n").replace("\r", "\n").strip()
    command = re.sub(r"^cd\s+(?:/workspace|/testbed)(?:/[^\s&;]+)?\s*&&\s*", "", command)
    return normalize_text(command)


def action_from_call(call: dict, next_tool: str | None):
    function = call.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ValueError("invalid_tool_call")
    name = function["name"]
    try:
        values = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as error:
        raise ValueError("invalid_tool_arguments") from error
    if not isinstance(values, dict):
        raise ValueError("non_object_tool_arguments")
    if name == "finish":
        message = " ".join(str(values.get("message", "Task completed.")).split())
        return "final", f"Final: {message[:480]}"
    if name == "think":
        return "skip", None
    if name == "execute_bash":
        command = normalize_command(str(values.get("command", "")))
        if not command or command == "C-c":
            return "skip", None
        return "action", f"Action: bash\nArguments: {json.dumps({'command': command}, separators=(',', ':'))}"
    if name != "str_replace_editor":
        raise ValueError(f"unsupported_tool:{name}")
    command = values.get("command")
    path = normalize_path(str(values.get("path", "")))
    if command == "view":
        if next_tool and "files and directories" in next_tool.lower():
            shell = f"find {json.dumps(path)} -maxdepth 2 -mindepth 1 | sort | head -200"
            return "action", f"Action: bash\nArguments: {json.dumps({'command': shell}, separators=(',', ':'))}"
        view_range = values.get("view_range")
        offset, limit = 1, 200
        if isinstance(view_range, list) and len(view_range) == 2:
            try:
                offset = max(1, int(view_range[0]))
                stop = int(view_range[1])
                limit = 200 if stop < 0 else max(1, min(2000, stop - offset + 1))
            except (TypeError, ValueError):
                pass
        args = {"path": path, "offset": offset, "limit": limit}
        return "action", f"Action: read\nArguments: {json.dumps(args, separators=(',', ':'))}"
    if command == "str_replace":
        args = {"path": path, "oldText": str(values.get("old_str", "")),
                "newText": str(values.get("new_str", ""))}
        if not args["oldText"]:
            raise ValueError("empty_old_str")
        return "action", f"Action: edit\nArguments: {json.dumps(args, separators=(',', ':'))}"
    if command == "create":
        args = {"path": path, "content": str(values.get("file_text", ""))}
        return "action", f"Action: write\nArguments: {json.dumps(args, separators=(',', ':'))}"
    raise ValueError(f"unsupported_editor_command:{command}")


def normalize_messages(row: dict):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError("missing_messages")
    user = next((normalize_text(str(m.get("content", ""))) for m in messages if m.get("role") == "user"), "")
    if not user:
        raise ValueError("missing_user")
    normalized = [("system", E97_PI_AGENT_SYSTEM_V2), ("user", user)]
    assistant_actions = reasoning_characters = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        reasoning_characters += len(str(message.get("reasoning_content") or ""))
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            index += 1
            continue
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        next_tool = (normalize_text(str(next_message.get("content", "")))
                     if isinstance(next_message, dict) and next_message.get("role") == "tool" else None)
        kind, content = action_from_call(calls[0], next_tool)
        if kind == "final":
            normalized.append(("assistant", content))
            break
        if kind == "action":
            if next_tool is None:
                raise ValueError("action_without_tool_result")
            normalized.append(("assistant", content))
            normalized.append(("tool", next_tool or "(no tool output)"))
            assistant_actions += 1
            index += 1
        index += 1
    if normalized[-1][0] != "assistant" or not normalized[-1][1].startswith("Final:"):
        raise ValueError("missing_finish")
    if assistant_actions == 0:
        raise ValueError("no_actions")
    return normalized, assistant_actions, reasoning_characters


ROLE_PREFIX = {"system": "System:\n", "user": "User:\n",
               "assistant": "Assistant:\n", "tool": "Tool:\n"}
_MAX_RECORD_TOKENS = 4096


def encode_record(messages):
    pieces = []
    for index, (role, content, targeted) in enumerate(messages):
        if index:
            pieces.append(("\n\n", False))
        pieces.append((ROLE_PREFIX[role], False))
        pieces.append((content, targeted))
    pieces.append((codec.RS, True))
    tokens, masks, complete = codec._encode_pieces(pieces)
    return tokens, masks, complete


def segment_messages(messages, max_tokens):
    base = [(role, content, False) for role, content in messages[:2]]
    if len(encode_record(base)[0]) >= max_tokens:
        raise ValueError("oversize_system_and_user")
    units = []
    index = 2
    while index < len(messages):
        role, content = messages[index]
        if role != "assistant":
            index += 1
            continue
        unit = [(role, content, True)]
        if index + 1 < len(messages) and messages[index + 1][0] == "tool":
            tool_role, tool_content = messages[index + 1]
            unit.append((tool_role, tool_content, False))
            index += 1
        units.append(unit)
        index += 1
    records, current = [], list(base)
    dropped_observations = dropped_actions = 0
    previous_context = None
    for unit in units:
        trial = current + unit
        if len(encode_record(trial)[0]) <= max_tokens:
            current = trial
            previous_context = [(role, content, False) for role, content, _ in unit]
            continue
        if any(targeted for _, _, targeted in current):
            records.append(current)
        current = list(base)
        if previous_context and len(encode_record(current + previous_context + unit)[0]) <= max_tokens:
            current += previous_context
        if len(encode_record(current + unit)[0]) <= max_tokens:
            current += unit
            previous_context = [(role, content, False) for role, content, _ in unit]
            continue
        action_only = [unit[0]]
        if len(encode_record(current + action_only)[0]) <= max_tokens:
            current += action_only
            dropped_observations += int(len(unit) == 2)
            previous_context = None
        else:
            dropped_actions += 1
            previous_context = None
    if any(targeted for _, _, targeted in current):
        records.append(current)
    if not records:
        raise ValueError("no_bounded_records")
    return records, dropped_observations, dropped_actions


def serialize_item(item):
    source_file, row_index, row = item
    identity = f"open-swe:{row.get('trajectory_id', source_file + ':' + str(row_index))}"
    repo = str(row.get("repo", "")).strip()
    license_name = str(row.get("license", "")).strip()
    if row.get("resolved") != 1:
        return {"excluded": "not_resolved", "identity": identity}
    if repo.lower() in EXCLUDED_REPOSITORIES:
        return {"excluded": "holdout_repository", "identity": identity}
    if license_name not in ALLOWED_LICENSES:
        return {"excluded": f"license:{license_name}", "identity": identity}
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("git_hack_attempted"):
        return {"excluded": "git_hack_attempted", "identity": identity}
    try:
        messages, actions, reasoning_characters = normalize_messages(row)
        segments, dropped_observations, dropped_actions = segment_messages(messages, _MAX_RECORD_TOKENS)
    except ValueError as error:
        return {"error": str(error), "identity": identity}
    encoded = []
    trajectory_split = int(split(identity))
    for segment_index, segment in enumerate(segments):
        try:
            tokens, masks, complete = encode_record(segment)
        except ValueError as error:
            return {"error": str(error), "identity": identity}
        encoded.append({
            "identity": f"{identity}:window:{segment_index:04d}",
            "trajectory_identity": identity, "source_file": source_file,
            "source_row": row_index, "repo": repo, "license": license_name,
            "split": trajectory_split, "tokens": len(tokens), "targets": sum(masks),
            "actions": sum(1 for role, _, targeted in segment if role == "assistant" and targeted),
            "token_bytes": struct.pack(f"<{len(tokens)}I", *tokens), "mask_bytes": bytes(masks),
            "serialization_sha256": hashlib.sha256(complete.encode()).hexdigest(),
        })
    return {"records": encoded, "identity": identity, "reasoning_characters_dropped": reasoning_characters,
            "source_actions": actions, "dropped_oversize_observations": dropped_observations,
            "dropped_oversize_actions": dropped_actions}


def rows(paths):
    for path in paths:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row_index, row in enumerate(batch.to_pylist(), offset):
                yield path.name, row_index, row
            offset += batch.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-record-tokens", type=int, default=4096)
    args = parser.parse_args()
    global _MAX_RECORD_TOKENS
    _MAX_RECORD_TOKENS = args.max_record_tokens
    if _MAX_RECORD_TOKENS < 512:
        raise SystemExit("--max-record-tokens must be at least 512")
    cache_root = os.environ.get("TIKTOKEN_CACHE_DIR")
    cache_object = Path(cache_root or "") / codec.TOKENIZER_CACHE_KEY
    if not cache_root or not cache_object.is_file() or sha256(cache_object) != codec.TOKENIZER_SHA256:
        raise SystemExit("verified p50k cache is required")
    worker_init()
    if sha256(args.input_root / "README.md") != DATASET_CARD_SHA256:
        raise SystemExit("Open-SWE dataset card mismatch")
    paths = sorted(args.input_root.rglob("*.parquet"))
    if len(paths) != 18:
        raise SystemExit(f"expected 18 pinned OpenHands shards, found {len(paths)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = {"tokens": args.output_root / "tokens.uint32.bin",
               "mask": args.output_root / "assistant_mask.uint8.bin",
               "index": args.output_root / "records.idx",
               "metadata": args.output_root / "records.jsonl"}
    counts, errors, excluded, repos = Counter(), Counter(), Counter(), Counter()
    offset = 0
    iterator = rows(paths)
    if args.limit:
        import itertools
        iterator = itertools.islice(iterator, args.limit)
    with outputs["tokens"].open("wb") as token_out, outputs["mask"].open("wb") as mask_out, \
         outputs["index"].open("wb") as index_out, outputs["metadata"].open("w") as metadata_out, \
         mp.Pool(args.workers, initializer=worker_init) as pool:
        for result in pool.imap(serialize_item, iterator, chunksize=2):
            counts["input_records"] += 1
            if "excluded" in result:
                excluded[result["excluded"]] += 1
                continue
            if "error" in result:
                errors[result["error"]] += 1
                continue
            counts["source_trajectories"] += 1
            counts["source_actions"] += result["source_actions"]
            counts["reasoning_characters_dropped"] += result["reasoning_characters_dropped"]
            counts["dropped_oversize_observations"] += result["dropped_oversize_observations"]
            counts["dropped_oversize_actions"] += result["dropped_oversize_actions"]
            for record in result["records"]:
                token_out.write(record.pop("token_bytes")); mask_out.write(record.pop("mask_bytes"))
                index_out.write(RECORD_INDEX.pack(offset, record["tokens"], record["targets"], record["split"]))
                metadata_out.write(json.dumps(record, sort_keys=True) + "\n")
                offset += record["tokens"]
                counts["records"] += 1; counts["tokens"] += record["tokens"]
                counts["assistant_target_tokens"] += record["targets"]
                counts["assistant_actions"] += record["actions"]
                counts["validation_records" if record["split"] else "train_records"] += 1
                repos[record["repo"]] += 1
    manifest = {
        "schema": AUTHORITY_SCHEMA, "status": "complete",
        "purpose": "verified OpenHands action-only Pi v2 protocol distillation",
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "dataset_card_sha256": DATASET_CARD_SHA256,
        "filters": {"resolved": 1, "licenses": sorted(ALLOWED_LICENSES),
                    "excluded_repositories": sorted(EXCLUDED_REPOSITORIES),
                    "git_hack_attempted": False},
        "system_prompt": E97_PI_AGENT_SYSTEM_V2,
        "reasoning_policy": "teacher reasoning dropped in action-only branch",
        "window_policy": {"max_record_tokens": _MAX_RECORD_TOKENS,
                          "boundary": "whole assistant-action/tool-observation units",
                          "overlap": "previous complete unit as untargeted context when it fits",
                          "oversize": "omit whole observation or action and count it; never truncate content"},
        "counts": dict(counts), "errors": dict(errors), "excluded": dict(excluded),
        "repository_counts": dict(repos),
        "input_files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths],
        "outputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                             "sha256": sha256(path)} for name, path in outputs.items()},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
