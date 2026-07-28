#!/usr/bin/env python3
"""Scheduler-owned, idempotent async-v2.1 terminal evidence collector.

The qualification controller registers this program as an ``afterany`` Slurm
job before releasing the held model payload.  It intentionally uses only the
Python standard library and Slurm commands; WG, Codex, and a submit-side
monitoring process are not runtime dependencies.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Mapping, Sequence


COLLECTOR_SCHEMA = "emender-async-v21-terminal-collector-v1"
VERDICT_SCHEMA = "emender-async-v21-terminal-verdict-v1"
STATE_SCHEMA = "emender-async-v21-qualification-state-v2"
LEGACY_STATE_SCHEMA = "emender-async-v21-qualification-state-v1"
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "DerivedExitCode",
    "Partition",
    "QOS",
    "NNodes",
    "NodeList",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "ElapsedRaw",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(path, _canonical(value) + b"\n")


def _retained_file(source: Path, destination: Path) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"required terminal evidence is missing: {source}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return {
        "source_path": str(source),
        "retained_path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _expanded(pattern: str, job_id: str) -> Path:
    return Path(
        pattern.replace("%j", job_id).replace("%A", job_id)
    ).resolve()


def _terminal_accounting(job_id: str) -> tuple[str, dict[str, object]]:
    command = [
        "sacct",
        "-n",
        "-X",
        "-j",
        job_id,
        f"--format={','.join(SACCT_FIELDS)}",
        "-P",
    ]
    last_output = ""
    for attempt in range(60):
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        last_output = completed.stdout
        for line in completed.stdout.splitlines():
            fields = line.rstrip("\n").split("|")
            if len(fields) < len(SACCT_FIELDS):
                continue
            values = dict(zip(
                SACCT_FIELDS, fields[:len(SACCT_FIELDS)], strict=True))
            if values["JobIDRaw"].split(".", 1)[0] != job_id:
                continue
            state = values["State"].split("+", 1)[0]
            if state in {
                "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE",
                "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
                "REVOKED", "TIMEOUT",
            }:
                return completed.stdout, {
                    "job_id": job_id,
                    "job_name": values["JobName"],
                    "state": state,
                    "exit_code": values["ExitCode"],
                    "derived_exit_code": values["DerivedExitCode"],
                    "partition": values["Partition"],
                    "qos": values["QOS"],
                    "nodes": int(values["NNodes"] or 0),
                    "node_list": values["NodeList"],
                    "submit": values["Submit"],
                    "eligible": values["Eligible"],
                    "start": values["Start"],
                    "end": values["End"],
                    "elapsed_raw_seconds": int(values["ElapsedRaw"] or 0),
                }
        if attempt != 59:
            time.sleep(2)
    raise TimeoutError(
        f"terminal sacct row for payload job {job_id} did not become visible; "
        f"last output={last_output!r}")


def _load_semantic_verdict(path: Path) -> tuple[dict[str, object], bool]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("semantic validator output must be a JSON object")
    passed = value.get("status") == "passed" or value.get("passed") is True
    return value, passed


def _load_existing(
    path: Path, *, payload_digest: str, payload_job_id: str,
) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items()
                if key != "manifest_digest"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != VERDICT_SCHEMA
        or value.get("payload_digest") != payload_digest
        or value.get("payload_job_id") != payload_job_id
        or value.get("manifest_digest") != _digest(unsigned)
    ):
        raise ValueError("existing terminal verdict identity is invalid")
    return value


def _update_state(
    state_path: Path,
    *,
    payload_digest: str,
    payload_job_id: str,
    verdict_path: Path,
    verdict: Mapping[str, object],
) -> None:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            not isinstance(state, dict)
            or state.get("schema") not in {STATE_SCHEMA, LEGACY_STATE_SCHEMA}
            or not isinstance(state.get("payloads"), dict)
        ):
            raise ValueError("qualification state identity is invalid")
        record = state["payloads"].get(payload_digest)
        if (
            not isinstance(record, dict)
            or str(record.get("job_id")) != payload_job_id
        ):
            raise ValueError("collector payload/job identity is not durable")
        record["status"] = (
            "terminal" if verdict.get("passed") is True else "retired")
        record["verdict"] = verdict["verdict"]
        record["terminal_evidence"] = {
            "path": str(verdict_path.resolve()),
            "sha256": _sha256(verdict_path),
        }
        collector = record.get("collector")
        if not isinstance(collector, dict):
            raise ValueError("durable collector identity disappeared")
        collector["status"] = "completed"
        collector["executed_by_slurm_job_id"] = os.environ.get(
            "SLURM_JOB_ID", collector.get("job_id", ""))
        state["schema"] = STATE_SCHEMA
        active = state.get("active_job")
        if (
            isinstance(active, Mapping)
            and active.get("payload_digest") == payload_digest
        ):
            state["active_job"] = None
        _atomic_json(state_path, state)


def collect_terminal_evidence(
    *,
    state_path: Path,
    payload_digest: str,
    payload_job_id: str,
    evidence_dir: Path,
    payload_input: Path,
    stdout_pattern: str,
    stderr_pattern: str,
    semantic_verdict: Path | None = None,
) -> dict[str, object]:
    if (
        len(payload_digest) != 64
        or not payload_job_id.isdigit()
        or not payload_input.is_file()
    ):
        raise ValueError("collector requires exact payload/job/input identity")
    evidence_dir = evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    lock_path = evidence_dir / ".collector.lock"
    verdict_path = evidence_dir / "terminal-verdict.json"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if verdict_path.exists():
            verdict = _load_existing(
                verdict_path,
                payload_digest=payload_digest,
                payload_job_id=payload_job_id,
            )
            _update_state(
                state_path,
                payload_digest=payload_digest,
                payload_job_id=payload_job_id,
                verdict_path=verdict_path,
                verdict=verdict,
            )
            return verdict

        sacct_text, scheduler = _terminal_accounting(payload_job_id)
        sacct_path = evidence_dir / "payload-sacct.psv"
        _atomic_bytes(sacct_path, sacct_text.encode("utf-8"))
        logs = {
            "stdout": _retained_file(
                _expanded(stdout_pattern, payload_job_id),
                evidence_dir / "payload.stdout.log",
            ),
            "stderr": _retained_file(
                _expanded(stderr_pattern, payload_job_id),
                evidence_dir / "payload.stderr.log",
            ),
        }
        validator_inputs: dict[str, object] = {
            "payload": {
                "path": str(payload_input.resolve()),
                "bytes": payload_input.stat().st_size,
                "sha256": _sha256(payload_input),
            },
            "sacct": {
                "path": str(sacct_path.resolve()),
                "bytes": sacct_path.stat().st_size,
                "sha256": _sha256(sacct_path),
            },
        }
        semantic_passed = True
        if semantic_verdict is not None:
            _semantic_value, semantic_passed = _load_semantic_verdict(
                semantic_verdict.resolve())
            validator_inputs["semantic_verdict"] = _retained_file(
                semantic_verdict.resolve(),
                evidence_dir / "semantic-verdict.json",
            )
            validator_inputs["semantic_verdict"]["passed"] = semantic_passed
            validator_inputs["semantic_verdict"]["required"] = True
        else:
            validator_inputs["semantic_verdict"] = {
                "required": False,
                "present": False,
            }
        scheduler_passed = (
            scheduler["state"] == "COMPLETED"
            and scheduler["exit_code"] == "0:0"
            and scheduler["derived_exit_code"] == "0:0"
            and scheduler["partition"] == "batch"
            and scheduler["qos"] == "debug"
        )
        passed = scheduler_passed and semantic_passed
        verdict: dict[str, object] = {
            "schema": VERDICT_SCHEMA,
            "payload_digest": payload_digest,
            "payload_job_id": payload_job_id,
            "collector_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "scheduler_owned": True,
            "requires_wg_or_codex": False,
            "scheduler": scheduler,
            "logs": logs,
            "validator_inputs": validator_inputs,
            "passed": passed,
            "verdict": "passed" if passed else "failed",
        }
        verdict["manifest_digest"] = _digest(verdict)
        _atomic_json(verdict_path, verdict)
        _update_state(
            state_path,
            payload_digest=payload_digest,
            payload_job_id=payload_job_id,
            verdict_path=verdict_path,
            verdict=verdict,
        )
        return verdict


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--payload-digest", required=True)
    parser.add_argument("--payload-job-id", required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--payload-input", type=Path, required=True)
    parser.add_argument("--stdout-pattern", required=True)
    parser.add_argument("--stderr-pattern", required=True)
    parser.add_argument("--semantic-verdict", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    verdict = collect_terminal_evidence(
        state_path=args.state.resolve(),
        payload_digest=args.payload_digest,
        payload_job_id=args.payload_job_id,
        evidence_dir=args.evidence_dir,
        payload_input=args.payload_input.resolve(),
        stdout_pattern=args.stdout_pattern,
        stderr_pattern=args.stderr_pattern,
        semantic_verdict=(
            args.semantic_verdict.resolve()
            if args.semantic_verdict is not None
            else None
        ),
    )
    print(f"terminal_verdict={verdict['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
