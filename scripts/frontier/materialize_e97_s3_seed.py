#!/usr/bin/env python3
"""Fail-closed submit-side acquisition and offline allocation seed verification."""
import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import stat
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
ATTESTATION_SCHEMA = "emender-e97-seed-bootstrap-attestation-v1"


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
    raw_documents = {}
    for name, uri in (("step_manifest", seed["manifest_uri"]), ("latest_pointer", seed["latest_pointer_uri"])):
        document, raw = fetch_json(uri, opener)
        for document_key, seed_key in FIELDS.items():
            if document_key not in document or document[document_key] != seed[seed_key]:
                raise ValueError(f"{name} {document_key} disagrees with immutable seed")
        documents[name] = document
        raw_digests[name] = hashlib.sha256(raw).hexdigest()
        raw_documents[name] = base64.b64encode(raw).decode("ascii")
    return {
        "documents": documents,
        "document_sha256": raw_digests,
        "document_bytes_base64": raw_documents,
    }


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_regular_file(path: Path, seed: dict) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"seed cache entry is not a single regular file: {path}")
    size, digest = _digest_file(path)
    if size != seed["size"] or digest != seed["sha256"]:
        raise ValueError(
            f"seed identity mismatch: size={size} sha256={digest}")


def verify_attestation(attestation: dict, seed: dict) -> None:
    validate_seed(seed)
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema") != ATTESTATION_SCHEMA
        or attestation.get("seed") != seed
    ):
        raise ValueError("seed bootstrap attestation identity is invalid")
    encoded = attestation.get("document_bytes_base64")
    digests = attestation.get("document_sha256")
    documents = attestation.get("documents")
    if not all(isinstance(value, dict) for value in (encoded, digests, documents)):
        raise ValueError("seed bootstrap attestation authority evidence is invalid")
    for name in ("step_manifest", "latest_pointer"):
        try:
            raw = base64.b64decode(encoded[name], validate=True)
            document = json.loads(raw)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{name} authority bytes are invalid") from error
        if hashlib.sha256(raw).hexdigest() != digests.get(name):
            raise ValueError(f"{name} authority digest is invalid")
        if document != documents.get(name):
            raise ValueError(f"{name} authority parsed bytes disagree")
        for document_key, seed_key in FIELDS.items():
            if document.get(document_key) != seed[seed_key]:
                raise ValueError(
                    f"{name} {document_key} disagrees with immutable seed")


def prefetch(
    seed: dict,
    cache_root: Path,
    attestation_path: Path,
    opener=urllib.request.urlopen,
) -> tuple[Path, Path]:
    """Acquire and publish a verified content-addressed cold cache entry."""
    validate_seed(seed)
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"sha256-{seed['sha256']}.pt"
    lock = cache_root / f".sha256-{seed['sha256']}.lock"
    with lock.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        for stale_temporary in cache_root.glob(
                f".{cache.name}.download.*"):
            stale_temporary.unlink(missing_ok=True)
        authority = verify_authorities(seed, opener)
        reuse = False
        if cache.exists() or cache.is_symlink():
            try:
                _verify_regular_file(cache, seed)
                reuse = True
            except (ValueError, OSError):
                cache.unlink(missing_ok=True)
                _fsync_directory(cache_root)
        if not reuse:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{cache.name}.download.", dir=cache_root)
            temporary = Path(temporary_name)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as output, opener(
                        https_url(seed["uri"]), timeout=600) as response:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size != seed["size"] or digest.hexdigest() != seed["sha256"]:
                    raise ValueError(
                        f"download identity mismatch: size={size} "
                        f"sha256={digest.hexdigest()}")
                # A lock-protected, no-overwrite publication makes incomplete
                # downloads invisible and stale temporary files harmless.
                os.replace(temporary, cache)
                _fsync_directory(cache_root)
                _verify_regular_file(cache, seed)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        evidence = {
            "schema": ATTESTATION_SCHEMA,
            "seed": seed,
            "cache_path": str(cache),
            "cache_size": seed["size"],
            "cache_sha256": seed["sha256"],
            "verified_cache_reuse": reuse,
            **authority,
        }
        verify_attestation(evidence, seed)
        _atomic_json(attestation_path.resolve(), evidence)
    return cache, attestation_path.resolve()


def verify_local(
    seed: dict,
    checkpoint: Path,
    attestation_path: Path,
    expected_job_id: str,
    runtime_manifest: Path,
    expected_attestation_sha256: str | None = None,
) -> Path:
    """Verify node-local bytes and pinned authorities without any network I/O."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or job_id != expected_job_id:
        raise ValueError("offline seed verification has wrong SLURM_JOB_ID")
    checkpoint = _validate_destination(checkpoint, require_absent=False)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"sbcast seed copy is missing: {checkpoint}")
    _verify_regular_file(checkpoint, seed)
    attestation_bytes = attestation_path.read_bytes()
    attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
    if (
        expected_attestation_sha256 is not None
        and attestation_sha256 != expected_attestation_sha256
    ):
        raise ValueError("local seed authority attestation digest mismatch")
    attestation = json.loads(attestation_bytes)
    verify_attestation(attestation, seed)
    if (attestation.get("cache_size"), attestation.get("cache_sha256")) != (
            seed["size"], seed["sha256"]):
        raise ValueError("attested cold cache identity is invalid")
    _atomic_json(runtime_manifest.resolve(), {
        "schema": "emender-e97-offline-node-seed-verification-v1",
        "seed": seed,
        "node_checkpoint": str(checkpoint),
        "node_checkpoint_size": seed["size"],
        "node_checkpoint_sha256": seed["sha256"],
        "authority_attestation": str(attestation_path.resolve()),
        "authority_attestation_sha256": attestation_sha256,
        "slurm_job_id": job_id,
        "hostname": os.uname().nodename,
        "network_fetches": 0,
    })
    return checkpoint


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


def _validate_destination(destination: Path, *, require_absent: bool = True) -> Path:
    destination = destination.resolve()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or destination.parent.name != f"emender-e97-seed-{job_id}":
        raise ValueError("destination must be scoped by the current SLURM_JOB_ID")
    filesystem_type = _filesystem_type(destination)
    if filesystem_type in SHARED_FILESYSTEM_TYPES:
        raise ValueError(
            f"seed destination must be node-local, not shared filesystem "
            f"{filesystem_type}")
    if require_absent and destination.exists():
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prefetch", action="store_true")
    mode.add_argument("--verify-local", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--expected-job-id")
    parser.add_argument("--expected-attestation-sha256")
    args = parser.parse_args()
    seed = (
        json.loads(args.seed_json)
        if args.seed_json is not None
        else load_seed_config(args.seed_config)
    )
    if args.prefetch:
        if not args.cache_root or not args.attestation:
            parser.error("--prefetch requires --cache-root and --attestation")
        cache, attestation = prefetch(
            seed, args.cache_root, args.attestation)
        print(json.dumps({
            "cache": str(cache),
            "attestation": str(attestation),
            "attestation_sha256": hashlib.sha256(
                attestation.read_bytes()).hexdigest(),
        }, sort_keys=True))
    elif args.verify_local:
        if not all((args.destination, args.runtime_manifest, args.attestation,
                    args.expected_job_id, args.expected_attestation_sha256)):
            parser.error(
                "--verify-local requires --destination, --runtime-manifest, "
                "--attestation, --expected-job-id, and "
                "--expected-attestation-sha256")
        print(verify_local(
            seed, args.destination, args.attestation, args.expected_job_id,
            args.runtime_manifest, args.expected_attestation_sha256))
    else:
        if not args.destination or not args.runtime_manifest:
            parser.error(
                "legacy materialization requires --destination and "
                "--runtime-manifest")
        print(materialize(seed, args.destination, args.runtime_manifest))


if __name__ == "__main__":
    main()
