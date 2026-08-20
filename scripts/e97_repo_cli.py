#!/usr/bin/env python3
"""Deterministic, cwd-confined repository inspection CLI for E97 agents."""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

MAX_READ_BYTES = 64 * 1024
MAX_RESULTS = 256
ROOT = Path.cwd().resolve()


def confined(requested: str, *, require_exists: bool = True) -> Path:
    candidate = (ROOT / requested.removeprefix("@")).resolve(strict=require_exists)
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError("path escapes the working directory")
    return candidate


def relative(path: Path) -> str:
    return "." if path == ROOT else path.relative_to(ROOT).as_posix()


def emit(value: Any) -> None:
    print(json.dumps(value, separators=(",", ":"), ensure_ascii=False, sort_keys=True))


def command_list(args: argparse.Namespace) -> None:
    root = confined(args.path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    rows = []
    iterator = root.rglob("*") if args.recursive else root.iterdir()
    for path in sorted(iterator):
        if len(rows) >= args.max_results:
            raise ValueError("result limit exceeded; narrow the pattern")
        rel = relative(path)
        if fnmatch.fnmatch(path.name, args.pattern):
            rows.append({"path": rel, "type": "dir" if path.is_dir() else "file"})
    emit({"entries": rows})


def command_count(args: argparse.Namespace) -> None:
    root = confined(args.path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    iterator = root.rglob("*") if args.recursive else root.iterdir()
    count = sum(path.is_file() and fnmatch.fnmatch(path.name, args.pattern) for path in iterator)
    emit({"count": count, "path": relative(root), "pattern": args.pattern})


def command_read(args: argparse.Namespace) -> None:
    path = confined(args.path)
    if not path.is_file():
        raise ValueError("path is not a file")
    data = path.read_bytes()
    if len(data) > MAX_READ_BYTES:
        raise ValueError(f"file exceeds {MAX_READ_BYTES} byte limit")
    lines = data.decode("utf-8").splitlines()
    start = args.start
    end = min(args.end if args.end is not None else len(lines), len(lines))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    emit({"path": relative(path), "start": start, "end": end, "lines": [
        {"line": number, "text": lines[number - 1]}
        for number in range(start, end + 1)
    ]})


def command_search(args: argparse.Namespace) -> None:
    root = confined(args.path)
    if not root.is_dir():
        raise ValueError("path is not a directory")
    matches = []
    for path in sorted(root.rglob(args.pattern)):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            continue
        for line_number, text in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if args.query in text:
                matches.append({"path": relative(path), "line": line_number, "text": text})
                if len(matches) >= args.max_results:
                    emit({"matches": matches, "truncated": True})
                    return
    emit({"matches": matches, "truncated": False})


def json_pointer(value: Any, pointer: str) -> Any:
    if pointer in ("", "/"):
        return value
    current = value
    for raw in pointer.removeprefix("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise ValueError("JSON pointer traverses a scalar")
    return current


def command_json(args: argparse.Namespace) -> None:
    path = confined(args.path)
    data = path.read_bytes()
    if len(data) > MAX_READ_BYTES:
        raise ValueError(f"file exceeds {MAX_READ_BYTES} byte limit")
    value = json_pointer(json.loads(data), args.pointer)
    emit({"path": relative(path), "pointer": args.pointer, "value": value})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="repo", description="Inspect only the current repository working directory.")
    sub = result.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list bounded directory entries")
    listing.add_argument("--path", default=".")
    listing.add_argument("--pattern", default="*")
    listing.add_argument("--recursive", action="store_true")
    listing.add_argument("--max-results", type=int, default=MAX_RESULTS, choices=range(1, MAX_RESULTS + 1))
    listing.set_defaults(run=command_list)

    count = sub.add_parser("count", help="count matching files without asking the model to count output")
    count.add_argument("--path", default=".")
    count.add_argument("--pattern", default="*")
    count.add_argument("--recursive", action="store_true")
    count.set_defaults(run=command_count)

    read = sub.add_parser("read", help="read a bounded line-numbered file range")
    read.add_argument("--path", required=True)
    read.add_argument("--start", type=int, default=1)
    read.add_argument("--end", type=int)
    read.set_defaults(run=command_read)

    search = sub.add_parser("search", help="search literal text in bounded UTF-8 files")
    search.add_argument("--query", required=True)
    search.add_argument("--path", default=".")
    search.add_argument("--pattern", default="*")
    search.add_argument("--max-results", type=int, default=32, choices=range(1, MAX_RESULTS + 1))
    search.set_defaults(run=command_search)

    inspect_json = sub.add_parser("json", help="extract one RFC 6901 JSON pointer")
    inspect_json.add_argument("--path", required=True)
    inspect_json.add_argument("--pointer", required=True)
    inspect_json.set_defaults(run=command_json)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.run(args)
    except (OSError, UnicodeError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise SystemExit(f"repo: {exc}") from exc


if __name__ == "__main__":
    main()
