#!/usr/bin/env python3
"""Bounded rank lane for the resilient Frontier debug topology.

Local rank zero owns the single model-bearing trainer/manager process.  The
remaining Slurm ranks are independently visible sentinels and never load the
multi-gigabyte checkpoint.  They follow an atomic per-node terminal record so
an srun still accounts for every requested rank without multiplying host RAM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    lanes = Path(args.run_dir) / "rank_lanes"
    terminal = lanes / f"node-{args.node_rank:05d}.terminal.json"
    heartbeat = lanes / f"node-{args.node_rank:05d}-local-{args.local_rank:02d}.json"
    started = time.monotonic()
    _atomic_json(heartbeat, {"role": "trainer_manager" if args.local_rank == 0 else "sentinel",
                             "pid": os.getpid(), "state": "starting", "time": time.time()})
    if args.local_rank == 0:
        if not command:
            raise SystemExit("trainer lane requires a command")
        result = subprocess.run(command, check=False)
        _atomic_json(terminal, {"exit_code": result.returncode, "time": time.time()})
        return int(result.returncode)
    while time.monotonic() - started < args.timeout_s:
        if terminal.exists():
            payload = json.loads(terminal.read_text())
            _atomic_json(heartbeat, {"role": "sentinel", "pid": os.getpid(),
                                     "state": "trainer_terminal", "time": time.time(),
                                     "trainer_exit_code": int(payload["exit_code"])})
            return int(payload["exit_code"])
        _atomic_json(heartbeat, {"role": "sentinel", "pid": os.getpid(),
                                 "state": "healthy", "time": time.time()})
        time.sleep(min(2.0, max(0.05, args.timeout_s / 10.0)))
    _atomic_json(heartbeat, {"role": "sentinel", "pid": os.getpid(),
                             "state": "deadline", "time": time.time()})
    return 124


if __name__ == "__main__":
    sys.exit(main())
