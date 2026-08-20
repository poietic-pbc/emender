#!/usr/bin/env python3
"""Execute one argv vector in the cwd-only E97 Apptainer sandbox."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_MAX_OUTPUT = 16 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT)
    parser.add_argument("--max-address-space", default="2G")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv or len(argv) > 64 or any(not value or len(value) > 4096 for value in argv):
        raise SystemExit("invalid command argument vector")
    if args.timeout < 1 or args.timeout > 120:
        raise SystemExit("timeout must be between 1 and 120 seconds")
    if args.max_output_bytes < 256 or args.max_output_bytes > 1024 * 1024:
        raise SystemExit("invalid output bound")
    image = args.image.resolve(strict=True)
    cwd = args.cwd.resolve(strict=True)
    if not cwd.is_dir():
        raise SystemExit("cwd must be a directory")
    if sha256(image) != args.image_sha256:
        raise SystemExit("sandbox image authority mismatch")

    command = [
        "apptainer", "exec", "--containall", "--cleanenv", "--net", "--network", "none",
        "--no-privs", "--drop-caps", "all",
        "--no-mount", "bind-paths,home,cwd,tmp,hostfs,proc,sys",
        "--bind", f"{cwd}:/work:rw", "--cwd", "/work", str(image),
        "/usr/bin/prlimit", "--core=0", "--cpu=60", "--fsize=1048576", "--nofile=256",
        f"--as={args.max_address_space}", "--", *argv,
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            env=environment,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_raw = stdout_file.read(args.max_output_bytes + 1)
        stderr_raw = stderr_file.read(args.max_output_bytes + 1)
    stdout_truncated = len(stdout_raw) > args.max_output_bytes
    stderr_truncated = len(stderr_raw) > args.max_output_bytes
    stdout_raw = stdout_raw[: args.max_output_bytes]
    stderr_raw = stderr_raw[: args.max_output_bytes]
    result = {
        "argv": argv,
        "cwd": ".",
        "exit_code": returncode,
        "stdout": stdout_raw.decode("utf-8", errors="replace"),
        "stderr": stderr_raw.decode("utf-8", errors="replace"),
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timed_out": timed_out,
        "duration_ms": round((time.monotonic() - started) * 1000),
    }
    print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
