#!/usr/bin/env python3
"""Own and atomically publish the three retained native-G2 namespaces.

The batch job is the sole publisher of ``ARTIFACT_ROOT/<payload-job-id>``.
Submit/controller observations and afterany collector evidence are immutable,
content-addressed records in disjoint reserved subtrees.  Callers never supply
an arbitrary destination path.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Mapping, Sequence


OWNERSHIP_SCHEMA = "emender-native-g2-artifact-ownership-v1"
EVIDENCE_SCHEMA = "emender-native-g2-retained-evidence-v1"
OWNER_MARKER = ".artifact-owner.json"
ROOT_SCHEMA_FILE = "ARTIFACT-OWNERSHIP.json"
EXIT_CONFLICT = 73

_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_KIND = re.compile(r"[a-z][a-z0-9-]*\Z")
class ArtifactNamespaceError(RuntimeError):
    """The retained namespace cannot be used without weakening ownership."""

    exit_code = 74


class ArtifactNamespaceConflict(ArtifactNamespaceError):
    """Another actor or a prior attempt already published the batch root."""

    exit_code = EXIT_CONFLICT


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _validate_job_id(value: str, *, field: str) -> str:
    value = str(value)
    if _JOB_ID.fullmatch(value) is None:
        raise ArtifactNamespaceError(f"{field} must be a positive decimal Slurm job ID")
    return value


def _validate_kind(value: str) -> str:
    value = str(value)
    if _KIND.fullmatch(value) is None:
        raise ArtifactNamespaceError(
            "evidence kind must match [a-z][a-z0-9-]*"
        )
    return value


def _validate_identity(value: str, *, field: str) -> str:
    value = str(value)
    if not value or "\0" in value or len(value.encode("utf-8")) > 4096:
        raise ArtifactNamespaceError(f"{field} is empty or unbounded")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _publish_bytes_noreplace(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(temporary_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-filesystem hard link publishes the already-fsynced inode
            # atomically and fails with EEXIST instead of replacing history.
            # This primitive is supported by Frontier Lustre, whose renameat2
            # implementation rejects RENAME_NOREPLACE with EINVAL.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != payload:
                raise ArtifactNamespaceConflict(
                    f"conflicting immutable evidence already exists: {path}"
                )
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _ownership_record() -> dict[str, object]:
    return {
        "schema": OWNERSHIP_SCHEMA,
        "namespaces": [
            {
                "owner": "controller",
                "path_template": "controller/<payload-job-id>/scheduler-evidence",
                "permissions": "write immutable scheduler observations only",
                "publication": "content-addressed hard-link no-replace",
            },
            {
                "owner": "batch",
                "path_template": "<payload-job-id>",
                "permissions": "create and write its own authoritative job artifacts",
                "publication": "pre-marked directory symlink handoff exactly once",
            },
            {
                "owner": "collector",
                "path_template": (
                    "collectors/<collector-job-id>/payload-<payload-job-id>"
                ),
                "permissions": (
                    "read batch artifacts; write only immutable collector evidence"
                ),
                "publication": "content-addressed hard-link no-replace",
            },
        ],
        "reserved_top_level": ["controller", "collectors", ".batch-storage"],
        "invariant": "no actor may pre-create another actor's authoritative root",
    }


def initialize_artifact_root(artifact_root: str | Path) -> Path:
    """Create/verify the shared container and its immutable ownership schema."""
    root = Path(artifact_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    if not root.is_dir():
        raise ArtifactNamespaceError(f"artifact root is not a directory: {root}")
    schema_path = root / ROOT_SCHEMA_FILE
    _publish_bytes_noreplace(schema_path, _canonical(_ownership_record()))
    return root


def _require_artifact_root(artifact_root: str | Path) -> Path:
    root = Path(artifact_root).expanduser().resolve()
    schema_path = root / ROOT_SCHEMA_FILE
    try:
        record = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactNamespaceError(
            f"native G2 ownership schema is missing or unreadable: {schema_path}"
        ) from error
    if record != _ownership_record():
        raise ArtifactNamespaceConflict(
            f"native G2 ownership schema conflicts with {OWNERSHIP_SCHEMA}: "
            f"{schema_path}"
        )
    return root


def batch_namespace(artifact_root: str | Path, *, job_id: str) -> Path:
    root = _require_artifact_root(artifact_root)
    return root / _validate_job_id(job_id, field="job_id")


def controller_namespace(artifact_root: str | Path, *, job_id: str) -> Path:
    root = _require_artifact_root(artifact_root)
    job_id = _validate_job_id(job_id, field="job_id")
    return root / "controller" / job_id / "scheduler-evidence"


def collector_namespace(
    artifact_root: str | Path, *, collector_job_id: str, payload_job_id: str
) -> Path:
    root = _require_artifact_root(artifact_root)
    collector_job_id = _validate_job_id(
        collector_job_id, field="collector_job_id"
    )
    payload_job_id = _validate_job_id(payload_job_id, field="payload_job_id")
    return root / "collectors" / collector_job_id / f"payload-{payload_job_id}"


def _mkdir_namespace_without_symlinks(directory: Path, *, root: Path) -> None:
    """Create an owner subtree without following a cross-owner symlink."""
    try:
        relative = directory.relative_to(root)
    except ValueError as error:  # pragma: no cover - callers construct paths.
        raise ArtifactNamespaceError("owned namespace escapes artifact root") from error
    current = root
    for component in relative.parts:
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                metadata = current.lstat()
            else:
                metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactNamespaceConflict(
                f"owned namespace component is not a real directory: {current}"
            )


def publish_batch_namespace(
    artifact_root: str | Path,
    *,
    job_id: str,
    run_id: str,
    payload_id: str,
) -> Path:
    """Publish one pre-marked authoritative batch directory, never replacing."""
    root = _require_artifact_root(artifact_root)
    job_id = _validate_job_id(job_id, field="job_id")
    run_id = _validate_identity(run_id, field="run_id")
    payload_id = _validate_identity(payload_id, field="payload_id")
    final = root / job_id
    storage_root = root / ".batch-storage"
    _mkdir_namespace_without_symlinks(storage_root, root=root)
    staging = Path(
        tempfile.mkdtemp(prefix=f"{job_id}.", suffix=".owned", dir=storage_root)
    )
    published = False
    try:
        marker = {
            "schema": OWNERSHIP_SCHEMA,
            "owner": "batch",
            "job_id": job_id,
            "run_id": run_id,
            "payload_id": payload_id,
            "authoritative_root": job_id,
            "storage": staging.relative_to(root).as_posix(),
            "publication": "symlink-no-replace-to-pre-marked-directory",
        }
        _write_file(staging / OWNER_MARKER, _canonical(marker))
        _fsync_directory(staging)
        try:
            # symlink(2) is one atomic no-replace namespace operation on
            # Frontier Lustre. The target directory is already nonempty, so a
            # competing directory rename cannot replace the published link.
            os.symlink(
                staging.relative_to(root),
                final,
                target_is_directory=True,
            )
        except FileExistsError as error:
            raise ArtifactNamespaceConflict(
                f"refusing to overwrite retained job evidence: {final}"
            ) from error
        published = True
        _fsync_directory(root)
        return final
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _record_evidence(
    directory: Path,
    *,
    owner: str,
    kind: str,
    identity: Mapping[str, str],
    evidence: Mapping[str, object],
) -> Path:
    kind = _validate_kind(kind)
    reserved = {"schema", "owner", "kind", *identity}
    overlap = reserved.intersection(evidence)
    if overlap:
        raise ArtifactNamespaceError(
            f"evidence cannot replace reserved identity fields: {sorted(overlap)}"
        )
    record: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "owner": owner,
        "kind": kind,
        **identity,
        **dict(evidence),
    }
    payload = _canonical(record)
    digest = hashlib.sha256(payload).hexdigest()
    return _publish_bytes_noreplace(directory / f"{kind}-{digest}.json", payload)


def record_controller_evidence(
    artifact_root: str | Path,
    *,
    job_id: str,
    kind: str,
    evidence: Mapping[str, object],
) -> Path:
    job_id = _validate_job_id(job_id, field="job_id")
    root = _require_artifact_root(artifact_root)
    directory = controller_namespace(root, job_id=job_id)
    _mkdir_namespace_without_symlinks(directory, root=root)
    return _record_evidence(
        directory,
        owner="controller",
        kind=kind,
        identity={"job_id": job_id},
        evidence=evidence,
    )


def record_collector_evidence(
    artifact_root: str | Path,
    *,
    collector_job_id: str,
    payload_job_id: str,
    kind: str,
    evidence: Mapping[str, object],
) -> Path:
    collector_job_id = _validate_job_id(
        collector_job_id, field="collector_job_id"
    )
    payload_job_id = _validate_job_id(payload_job_id, field="payload_job_id")
    root = _require_artifact_root(artifact_root)
    directory = collector_namespace(
        root,
        collector_job_id=collector_job_id,
        payload_job_id=payload_job_id,
    )
    _mkdir_namespace_without_symlinks(directory, root=root)
    return _record_evidence(
        directory,
        owner="collector",
        kind=kind,
        identity={
            "collector_job_id": collector_job_id,
            "payload_job_id": payload_job_id,
        },
        evidence=evidence,
    )


def _run_scheduler(command: Sequence[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return {
            "argv": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except OSError as error:
        return {
            "argv": list(command),
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}\n",
        }


def observe_scheduler(
    artifact_root: str | Path,
    *,
    job_id: str,
    kind: str,
    squeue: str = "squeue",
    scontrol: str = "scontrol",
) -> Path:
    """Capture immediate/reconciled queue evidence outside the batch root."""
    job_id = _validate_job_id(job_id, field="job_id")
    squeue_command = [
        squeue,
        "-h",
        "-j",
        job_id,
        "-o",
        "%i|%T|%P|%q",
    ]
    scontrol_command = [scontrol, "show", "job", "-dd", job_id]
    return record_controller_evidence(
        artifact_root,
        job_id=job_id,
        kind=kind,
        evidence={
            "commands": {
                "squeue": _run_scheduler(squeue_command),
                "scontrol": _run_scheduler(scontrol_command),
            }
        },
    )


def _load_json_argument(value: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON evidence: {error}") from error
    if not isinstance(decoded, dict):
        raise argparse.ArgumentTypeError("evidence JSON must be an object")
    return decoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser("init-root")
    initialize.add_argument("--artifact-root", required=True, type=Path)

    publish = subparsers.add_parser("publish-batch")
    publish.add_argument("--artifact-root", required=True, type=Path)
    publish.add_argument("--job-id", required=True)
    publish.add_argument("--run-id", required=True)
    publish.add_argument("--payload-id", required=True)

    observe = subparsers.add_parser("observe-scheduler")
    observe.add_argument("--artifact-root", required=True, type=Path)
    observe.add_argument("--job-id", required=True)
    observe.add_argument("--kind", default="monitor")
    observe.add_argument("--squeue", default="squeue")
    observe.add_argument("--scontrol", default="scontrol")

    controller = subparsers.add_parser("record-controller")
    controller.add_argument("--artifact-root", required=True, type=Path)
    controller.add_argument("--job-id", required=True)
    controller.add_argument("--kind", required=True)
    controller.add_argument(
        "--evidence-json", required=True, type=_load_json_argument
    )

    collect = subparsers.add_parser("record-collector")
    collect.add_argument("--artifact-root", required=True, type=Path)
    collect.add_argument("--collector-job-id", required=True)
    collect.add_argument("--payload-job-id", required=True)
    collect.add_argument("--kind", required=True)
    collect.add_argument("--evidence-json", required=True, type=_load_json_argument)

    layout = subparsers.add_parser("show-layout")
    layout.add_argument("--artifact-root", required=True, type=Path)
    layout.add_argument("--job-id", required=True)
    layout.add_argument("--collector-job-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init-root":
            initialize_artifact_root(args.artifact_root)
        elif args.command == "publish-batch":
            publish_batch_namespace(
                args.artifact_root,
                job_id=args.job_id,
                run_id=args.run_id,
                payload_id=args.payload_id,
            )
        elif args.command == "observe-scheduler":
            observe_scheduler(
                args.artifact_root,
                job_id=args.job_id,
                kind=args.kind,
                squeue=args.squeue,
                scontrol=args.scontrol,
            )
        elif args.command == "record-controller":
            record_controller_evidence(
                args.artifact_root,
                job_id=args.job_id,
                kind=args.kind,
                evidence=args.evidence_json,
            )
        elif args.command == "record-collector":
            record_collector_evidence(
                args.artifact_root,
                collector_job_id=args.collector_job_id,
                payload_job_id=args.payload_job_id,
                kind=args.kind,
                evidence=args.evidence_json,
            )
        elif args.command == "show-layout":
            root = _require_artifact_root(args.artifact_root)
            value: dict[str, object] = {
                "schema": OWNERSHIP_SCHEMA,
                "controller": str(
                    controller_namespace(root, job_id=args.job_id)
                ),
                "batch": str(batch_namespace(root, job_id=args.job_id)),
            }
            if args.collector_job_id is not None:
                value["collector"] = str(
                    collector_namespace(
                        root,
                        collector_job_id=args.collector_job_id,
                        payload_job_id=args.job_id,
                    )
                )
            print(json.dumps(value, sort_keys=True, separators=(",", ":")))
        else:  # pragma: no cover - argparse makes this unreachable.
            raise AssertionError(args.command)
    except ArtifactNamespaceError as error:
        print(str(error), file=os.sys.stderr)
        return error.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
