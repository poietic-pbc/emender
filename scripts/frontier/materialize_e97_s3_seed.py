#!/usr/bin/env python3
"""Fail-closed, atomic materialization of the canonical immutable E97 seed."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

FIELDS = {
    "checkpoint_s3_uri": "uri",
    "checkpoint_size_bytes": "size",
    "checkpoint_sha256": "sha256",
    "step": "step",
    "loss": "loss",
}
SEED_FIELDS = {
    "uri", "manifest_uri", "latest_pointer_uri", "size", "sha256", "step",
    "loss", "tokens", "provenance",
}
SHARED_FILESYSTEM_TYPES = {"lustre", "gpfs", "nfs", "nfs4", "cifs"}


def https_url(uri: str) -> str:
    match = re.fullmatch(r"s3://([a-z0-9.-]+)/(.+)", uri)
    if not match:
        raise ValueError(f"invalid S3 URI: {uri!r}")
    bucket, key = match.groups()
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key, safe='/')}"


def fetch_json(uri: str, opener=urllib.request.urlopen) -> tuple[dict, bytes]:
    with opener(https_url(uri), timeout=60) as response:
        payload = response.read()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"S3 JSON is not an object: {uri}")
    return value, payload


def validate_seed(seed: dict) -> None:
    if not isinstance(seed, dict) or set(seed) != SEED_FIELDS:
        raise ValueError(
            "unknown or missing seed fields")
    if not all(
            isinstance(seed[key], str) and seed[key].startswith("s3://")
            for key in ("uri", "manifest_uri", "latest_pointer_uri")):
        raise ValueError("seed authorities must be S3 URIs")
    if "latest" in seed["uri"].lower() or not re.fullmatch(r"[0-9a-f]{64}", seed["sha256"]):
        raise ValueError("immutable checkpoint identity is invalid")
    if not isinstance(seed["step"], int) or seed["step"] <= 0:
        raise ValueError("immutable checkpoint step is invalid")
    if not isinstance(seed["tokens"], int) or seed["tokens"] <= 0:
        raise ValueError("immutable checkpoint token count is invalid")
    if not isinstance(seed["size"], int) or seed["size"] <= 0:
        raise ValueError("immutable checkpoint size is invalid")


def load_seed_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or "seed" not in config:
        raise ValueError("seed configuration has no seed object")
    seed = config["seed"]
    if not isinstance(seed, dict):
        raise ValueError("seed configuration seed is not an object")
    validate_seed(seed)
    return seed


def verify_authorities(seed: dict, opener=urllib.request.urlopen) -> dict:
    validate_seed(seed)
    documents = {}
    raw_digests = {}
    for name, uri in (("step_manifest", seed["manifest_uri"]), ("latest_pointer", seed["latest_pointer_uri"])):
        document, raw = fetch_json(uri, opener)
        for document_key, seed_key in FIELDS.items():
            if document_key not in document or document[document_key] != seed[seed_key]:
                raise ValueError(f"{name} {document_key} disagrees with immutable seed")
        documents[name] = document
        raw_digests[name] = hashlib.sha256(raw).hexdigest()
    return {"documents": documents, "document_sha256": raw_digests}


def _filesystem_type(path: Path) -> str:
    target = path.resolve()
    mounts: list[tuple[int, str]] = []
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mount = Path(fields[1].replace("\\040", " ")).resolve()
            if target == mount or target.is_relative_to(mount):
                mounts.append((len(mount.parts), fields[2].lower()))
    except OSError as error:
        raise RuntimeError(f"cannot verify seed destination mount: {error}") from error
    if not mounts:
        raise ValueError("seed destination has no verifiable filesystem mount")
    return max(mounts)[1]


def _validate_destination(destination: Path) -> Path:
    destination = destination.resolve()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or destination.parent.name != f"emender-e97-seed-{job_id}":
        raise ValueError("destination must be scoped by the current SLURM_JOB_ID")
    filesystem_type = _filesystem_type(destination)
    if filesystem_type in SHARED_FILESYSTEM_TYPES:
        raise ValueError(
            f"seed destination must be node-local, not shared filesystem "
            f"{filesystem_type}")
    if destination.exists():
        raise FileExistsError(f"refusing stale staged seed: {destination}")
    return destination


def materialize(seed: dict, destination: Path, runtime_manifest: Path, opener=urllib.request.urlopen) -> Path:
    validate_seed(seed)
    destination = _validate_destination(destination)
    runtime_manifest = runtime_manifest.resolve()
    authority = verify_authorities(seed, opener)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".tmp.", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as output, opener(https_url(seed["uri"]), timeout=600) as response:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != seed["size"] or digest.hexdigest() != seed["sha256"]:
            raise ValueError(f"download identity mismatch: size={size} sha256={digest.hexdigest()}")
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        evidence = {
            "schema_version": 1,
            "seed": seed,
            "staged_path": str(destination),
            "staged_size": size,
            "staged_sha256": digest.hexdigest(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "hostname": os.uname().nodename,
            **authority,
        }
        runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
        runtime_manifest.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
        return destination
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed-json")
    source.add_argument("--seed-config", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    args = parser.parse_args()
    seed = (
        json.loads(args.seed_json)
        if args.seed_json is not None
        else load_seed_config(args.seed_config)
    )
    print(materialize(seed, args.destination, args.runtime_manifest))


if __name__ == "__main__":
    main()
