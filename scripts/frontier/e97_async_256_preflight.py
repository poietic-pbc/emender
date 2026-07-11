#!/usr/bin/env python3
"""Fail-closed on-node verifier for a rendered E97 launch bundle."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ARTIFACTS = (
    "runtime.json",
    "inputs.json",
    "code.json",
    "runtime-manifest.json",
    "helper-manifest.json",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(path, chunk_size=8 * 1024 * 1024):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            result.update(block)
    return result.hexdigest()


def refuse(kind, detail):
    print(canonical({"ok": False, "kind": kind, "detail": detail}), file=sys.stderr)
    raise SystemExit(72)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        refuse("bundle_json", {"path": str(path), "error": str(exc)})


def verify_file_manifest(repo, records, kind):
    for record in records:
        if set(record) != {"path", "sha256", "size"}:
            refuse(kind, {"invalid_record": record})
        path = repo / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            refuse(kind, record["path"])


def verify_inputs(records):
    for name, record in records.items():
        path = Path(record["realpath"])
        if not path.is_file() or path.stat().st_size != record["size"]:
            refuse("input", name)
        if os.path.realpath(record["path"]) != record["realpath"]:
            refuse("input_realpath", name)
        if record["hash_kind"] == "sha256":
            observed = digest(path)
        elif record["hash_kind"] == "sha256-stat-v1":
            stat = path.stat()
            identity = {
                "realpath": os.path.realpath(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
            }
            observed = hashlib.sha256(canonical(identity).encode()).hexdigest()
        else:
            refuse("input_hash_kind", {name: record["hash_kind"]})
        if observed != record["sha256"]:
            refuse("input_hash", name)


def argv_value(argv, flag):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        refuse("trainer_argv", f"missing {flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--phase", choices=("before-helper", "before-srun"), required=True)
    parser.add_argument("--helper", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    repo = args.repo.resolve()
    fingerprint_path = bundle / "fingerprint.sha256"
    if not fingerprint_path.is_file() or fingerprint_path.read_text().strip() != args.fingerprint:
        refuse("fingerprint", str(fingerprint_path))
    expected_fingerprint_file = (bundle / "fingerprint-file.sha256").read_text().strip()
    if digest(fingerprint_path) != expected_fingerprint_file:
        refuse("fingerprint_sidecar", str(fingerprint_path))

    expected_bundle_files = (bundle / "bundle-files.sha256").read_text().splitlines()
    expected_names = set(ARTIFACTS) | {"fingerprint.sha256", "fingerprint-file.sha256"}
    observed_names = set()
    for line in expected_bundle_files:
        try:
            expected_hash, name = line.split("  ", 1)
        except ValueError:
            refuse("bundle_manifest", line)
        if name not in expected_names or name in observed_names:
            refuse("bundle_manifest", name)
        observed_names.add(name)
        if digest(bundle / name) != expected_hash:
            refuse("bundle_file_hash", name)
    if observed_names != expected_names:
        refuse("bundle_manifest", {"missing": sorted(expected_names - observed_names)})

    runtime = read_json(bundle / "runtime.json")
    code = read_json(bundle / "code.json")
    verify_file_manifest(repo, read_json(bundle / "runtime-manifest.json"), "runtime_file")
    verify_file_manifest(repo, read_json(bundle / "helper-manifest.json"), "helper_file")
    verify_inputs(read_json(bundle / "inputs.json"))

    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != code["git_commit"]:
        refuse("code_commit", code["git_commit"])
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip() != code["git_tree"]:
        refuse("code_tree", code["git_tree"])
    entrypoint = repo / code["entrypoint"]["path"]
    if entrypoint.stat().st_size != code["entrypoint"]["size"] or digest(entrypoint) != code["entrypoint"]["sha256"]:
        refuse("entrypoint", str(entrypoint))

    for name, value in runtime["environment"].items():
        if os.environ.get(name) != value:
            refuse("environment", {name: os.environ.get(name), "expected": value})
    loaded = os.environ.get("LOADEDMODULES", "").split(":")
    for requested in runtime["modules"]:
        base = requested.split("/", 1)[0]
        present = requested in loaded if "/" in requested else any(item.split("/", 1)[0] == base for item in loaded)
        if not present:
            refuse("modules", {"missing": requested, "loaded": loaded})

    launcher = repo / runtime["trainer_argv"][0]
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        refuse("launcher", str(launcher))
    topology = runtime["topology"]
    if [topology[key] for key in ("launched_ranks", "participant_ranks", "worker_ranks", "global_quorum")] != [2048] * 4:
        refuse("topology", topology)
    if topology["rank_ids"] != {"start": 0, "stop": 2047}:
        refuse("topology", topology["rank_ids"])
    expected_srun = ["srun", "-N", "256", "-n", "2048", "--ntasks-per-node", "8", "-c", "7", "--gpus-per-task", "1", "--gpu-bind", "closest"]
    if runtime["srun_argv"] != expected_srun:
        refuse("srun_argv", runtime["srun_argv"])
    trainer_argv = runtime["trainer_argv"]
    for flag, expected in (("--worker-count", "2048"), ("--node-count", "2048"), ("--global-quorum", "2048"), ("--node-rank", "@SLURM_PROCID@")):
        if argv_value(trainer_argv, flag) != expected:
            refuse("trainer_argv", {flag: argv_value(trainer_argv, flag)})

    if args.phase == "before-srun":
        if args.helper is None or not args.helper.is_file() or not os.access(args.helper, os.X_OK):
            refuse("helper", str(args.helper))
        shared = Path(str(args.helper) + ".so")
        if not shared.is_file() or shared.stat().st_size == 0:
            refuse("helper_shared", str(shared))

    print(canonical({"ok": True, "phase": args.phase, "fingerprint": args.fingerprint}))


if __name__ == "__main__":
    main()
