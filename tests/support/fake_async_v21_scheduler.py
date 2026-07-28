#!/usr/bin/env python3
"""Process-level fake Slurm surface for the durable v2.1 collector test."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Callable


STATE_ENV = "FAKE_ASYNC_V21_SCHEDULER_STATE"


def _state_path() -> Path:
    value = os.environ.get(STATE_ENV)
    if not value:
        raise RuntimeError(f"{STATE_ENV} is required")
    return Path(value).resolve()


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _locked(update: Callable[[dict], object]) -> object:
    path = _state_path()
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {"next_job_id": 81001, "jobs": {}}
        )
        result = update(state)
        _atomic(path, state)
        return result


def _option(arguments: list[str], prefix: str, default: str = "") -> str:
    for argument in arguments:
        if argument.startswith(prefix):
            return argument[len(prefix):]
    return default


def _expand(pattern: str, job_id: str) -> Path:
    return Path(pattern.replace("%j", job_id).replace("%A", job_id)).resolve()


def _sbatch(arguments: list[str]) -> int:
    held = "--hold" in arguments
    kind = "payload" if held else "collector"
    name = _option(arguments, "--job-name=")
    comment = _option(arguments, "--comment=")

    def add(state: dict) -> str:
        job_id = str(state["next_job_id"])
        state["next_job_id"] += 1
        record = {
            "job_id": job_id,
            "kind": kind,
            "name": name,
            "comment": comment,
            "state": "PENDING",
            "partition": _option(arguments, "--partition="),
            "qos": _option(arguments, "--qos="),
            "released": False,
            "output": _option(arguments, "--output="),
            "error": _option(arguments, "--error="),
        }
        if kind == "collector":
            record.update({
                "dependency": _option(arguments, "--dependency="),
                "wrap_argv": shlex.split(_option(arguments, "--wrap=")),
                "launched": False,
            })
        state["jobs"][job_id] = record
        return job_id

    print(_locked(add))
    return 0


def _matching_jobs(arguments: list[str]) -> list[dict]:
    path = _state_path()
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    jobs = list(state["jobs"].values())
    name = _option(arguments, "--name")
    if not name:
        try:
            name = arguments[arguments.index("--name") + 1]
        except (ValueError, IndexError):
            name = ""
    if name:
        jobs = [job for job in jobs if job["name"] == name]
    try:
        job_id = arguments[arguments.index("-j") + 1]
    except (ValueError, IndexError):
        job_id = ""
    if job_id:
        jobs = [job for job in jobs if job["job_id"] == job_id]
    return jobs


def _squeue(arguments: list[str]) -> int:
    jobs = [
        job for job in _matching_jobs(arguments)
        if job["state"] not in {"COMPLETED", "FAILED"}
    ]
    output = _option(arguments, "-o")
    if not output:
        try:
            output = arguments[arguments.index("-o") + 1]
        except (ValueError, IndexError):
            output = "%i"
    for job in jobs:
        rendered = output
        for token, value in (
            ("%i", job["job_id"]),
            ("%j", job["name"]),
            ("%k", job["comment"]),
            ("%T", job["state"]),
            ("%P", job["partition"]),
            ("%q", job["qos"]),
        ):
            rendered = rendered.replace(token, value)
        print(rendered)
    return 0


def _sacct(arguments: list[str]) -> int:
    jobs = _matching_jobs(arguments)
    format_value = next(
        (item.removeprefix("--format=") for item in arguments
         if item.startswith("--format=")),
        "JobIDRaw,JobName,Comment,State",
    )
    fields = format_value.split(",")
    for job in jobs:
        values = {
            "JobIDRaw": job["job_id"],
            "JobName": job["name"],
            "Comment": job["comment"],
            "State": job["state"],
            "ExitCode": "0:0" if job["state"] == "COMPLETED" else "0:0",
            "DerivedExitCode": "0:0",
            "Partition": job["partition"],
            "QOS": job["qos"],
            "NNodes": "2" if job["kind"] == "payload" else "1",
            "NodeList": "fake[01-02]" if job["kind"] == "payload" else "fake01",
            "Submit": "2026-07-27T00:00:00",
            "Eligible": "2026-07-27T00:00:01",
            "Start": "2026-07-27T00:00:02",
            "End": "2026-07-27T00:00:03",
            "ElapsedRaw": "1",
        }
        print("|".join(values.get(field, "") for field in fields) + "|")
    return 0


def _scontrol(arguments: list[str]) -> int:
    if len(arguments) != 2 or arguments[0] != "release":
        raise RuntimeError(f"unsupported fake scontrol command: {arguments}")
    payload_job_id = arguments[1]

    def release(state: dict) -> tuple[dict, dict]:
        payload = state["jobs"][payload_job_id]
        payload["released"] = True
        payload["state"] = "COMPLETED"
        collectors = [
            job for job in state["jobs"].values()
            if job["kind"] == "collector"
            and job["dependency"] == f"afterany:{payload_job_id}"
        ]
        if len(collectors) != 1:
            raise RuntimeError("payload release lacks exactly one collector")
        collector = collectors[0]
        collector["launched"] = True
        collector["state"] = "RUNNING"
        return dict(payload), dict(collector)

    payload, collector = _locked(release)
    stdout = _expand(payload["output"], payload_job_id)
    stderr = _expand(payload["error"], payload_job_id)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("fake model stdout: completed\n")
    stderr.write_text("fake model stderr: no errors\n")
    collector_stdout = _expand(collector["output"], collector["job_id"])
    collector_stderr = _expand(collector["error"], collector["job_id"])
    collector_stdout.parent.mkdir(parents=True, exist_ok=True)
    collector_stderr.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["SLURM_JOB_ID"] = collector["job_id"]
    with collector_stdout.open("w") as out, collector_stderr.open("w") as err:
        subprocess.Popen(
            collector["wrap_argv"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
    # Keep the submit-side worker inside release long enough for the test to
    # kill it.  The collector above is already an independent scheduler child.
    time.sleep(2)
    return 0


def main() -> int:
    command = Path(sys.argv[0]).name
    arguments = sys.argv[1:]
    if command == "sbatch":
        return _sbatch(arguments)
    if command == "squeue":
        return _squeue(arguments)
    if command == "sacct":
        return _sacct(arguments)
    if command == "scontrol":
        return _scontrol(arguments)
    raise RuntimeError(f"unsupported fake scheduler executable: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
