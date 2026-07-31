#!/usr/bin/env python3
"""Launch independently killable one-node E97 manager steps inside an allocation.

Unlike a single multi-node ``srun``, each manager has its own Slurm step and
process group.  A failed step can therefore be terminated without signalling
healthy peers.  This launcher never requeues or mutates its allocation/job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--command", required=True,
                        help="Shell command; {rank}, {node}, and {run_dir} are substituted")
    parser.add_argument("--deadline-s", type=float, required=True)
    parser.add_argument("--inject-kill-rank", type=int, default=-1)
    parser.add_argument("--inject-after-s", type=float, default=0.0)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or "SLURM_JOB_NODELIST" not in os.environ:
        raise RuntimeError("must run inside an existing Slurm allocation")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)  # exactly-once run-root guard
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).splitlines()
    processes: dict[int, subprocess.Popen] = {}
    timeline = []
    started = time.monotonic()
    for rank, node in enumerate(nodes):
        command = args.command.format(rank=rank, node=node, run_dir=run_dir)
        out = (run_dir / f"node-{rank:05d}.out").open("wb")
        err = (run_dir / f"node-{rank:05d}.err").open("wb")
        processes[rank] = subprocess.Popen(
            ["srun", "--nodes=1", "--ntasks=1", "--nodelist", node,
             "--exact", "--no-kill", "bash", "-lc", command],
            stdout=out, stderr=err, start_new_session=True,
        )
        timeline.append({"event": "step_started", "rank": rank, "node": node,
                         "elapsed_s": time.monotonic() - started})
    injected = False
    while processes:
        elapsed = time.monotonic() - started
        if (not injected and args.inject_kill_rank in processes
                and elapsed >= args.inject_after_s):
            os.killpg(processes[args.inject_kill_rank].pid, signal.SIGTERM)
            timeline.append({"event": "injected_step_termination",
                             "rank": args.inject_kill_rank, "elapsed_s": elapsed})
            injected = True
        for rank, process in list(processes.items()):
            code = process.poll()
            if code is not None:
                timeline.append({"event": "step_exited", "rank": rank,
                                 "returncode": code, "elapsed_s": elapsed})
                del processes[rank]
        if elapsed >= args.deadline_s:
            for rank, process in processes.items():
                os.killpg(process.pid, signal.SIGTERM)
                timeline.append({"event": "deadline_termination", "rank": rank,
                                 "elapsed_s": elapsed})
            break
        time.sleep(.2)
    (run_dir / "fault-timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    # The injected rank is allowed to fail; every other node manager must exit cleanly.
    bad = [item for item in timeline if item["event"] == "step_exited"
           and item["returncode"] and item["rank"] != args.inject_kill_rank]
    return 1 if bad or any(item["event"] == "deadline_termination" for item in timeline) else 0


if __name__ == "__main__":
    raise SystemExit(main())
