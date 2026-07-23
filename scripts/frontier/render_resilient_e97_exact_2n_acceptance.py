#!/usr/bin/env python3
"""Render the authoritative real-E97 two-node pipelined acceptance sequence.

Rendering is deliberately the default.  Submission requires ``--submit`` and
passes through the same fail-closed source/allocation checks, which makes this
file safe to use in CI without accidentally creating a Frontier allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.frontier.materialize_e97_s3_seed import prefetch

WALLTIME = "02:00:00"
PARTITION = "batch"
QOS = "debug"
APPROVED_ENV = "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312"
SEED_CONFIG = Path("configs/frontier/e97_async_256.yaml")
DATA = "/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt"
TIKTOKEN = "/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15"
TIKTOKEN_SHA256 = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"
SEED_CACHE_ROOT = Path(
    "/lustre/orion/bif148/proj-shared/emender/bootstrap/e97-seeds")
STAGE_DEADLINES = {
    "handoff_s": 180,
    "apply_s": 180,
    "quorum_s": 420,
    "integrity_s": 120,
    "publication_s": 180,
    "progress_s": 420,
}


def canonical_seed(repo: Path) -> dict[str, Any]:
    """Load the one reviewed seed identity from the canonical launch config."""
    config_path = (repo / SEED_CONFIG).resolve()
    config_path.relative_to(repo.resolve())
    config = json.loads(config_path.read_text())
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 3
        or set(config) != {"schema_version", "golden_manifest", "seed", "profiles"}
    ):
        raise ValueError("canonical E97 seed configuration shape is invalid")
    seed = config.get("seed")
    required = {
        "uri", "manifest_uri", "latest_pointer_uri", "step", "loss", "tokens",
        "size", "sha256", "provenance",
    }
    if not isinstance(seed, dict) or set(seed) != required:
        raise ValueError("canonical E97 seed identity shape is invalid")
    if not all(
            isinstance(seed[key], str) and seed[key].startswith("s3://")
            for key in ("uri", "manifest_uri", "latest_pointer_uri")):
        raise ValueError("canonical E97 seed authorities must be S3 URIs")
    if (
        "latest" in seed["uri"].lower()
        or len(seed["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in seed["sha256"])
        or not all(isinstance(seed[key], int) and seed[key] > 0
                   for key in ("step", "tokens", "size"))
    ):
        raise ValueError("canonical E97 seed identity is not immutable")
    return seed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_authoritative_source(repo: Path, *, check_allocation: bool) -> str:
    if git(repo, "branch", "--show-current") != "main":
        raise ValueError("authoritative acceptance must be rendered from main")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("authoritative acceptance requires a clean tracked source tree")
    commit = git(repo, "rev-parse", "HEAD")
    if git(repo, "rev-parse", "origin/main") != commit:
        raise ValueError("authoritative main must be merged and pushed to origin/main")
    if check_allocation:
        queued = subprocess.check_output(
            ["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], text=True
        ).strip()
        if queued:
            raise ValueError("refusing to overlap another user allocation")
    return commit


def source_identity(repo: Path, commit: str) -> dict[str, Any]:
    """Record every tracked source byte, not merely the symbolic Git revision."""
    records = []
    for relative in git(repo, "ls-files", "-z").split("\0"):
        if not relative:
            continue
        path = repo / relative
        if path.is_file():
            records.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"commit": commit, "tree": tree, "files": records,
            "files_sha256": hashlib.sha256(encoded).hexdigest()}


def verify_source_identity(repo: Path, expected: dict[str, Any]) -> None:
    commit = require_authoritative_source(repo, check_allocation=False)
    actual = source_identity(repo, commit)
    if actual != expected:
        raise ValueError("authoritative source changed after native bundle attestation")


def build_authoritative_bundle(repo: Path, commit: str, stage_root: Path) -> tuple[Path, dict[str, Any]]:
    """Clean-build an immutable installation and bind it to the source inventory."""
    source = source_identity(repo, commit)
    stage = (stage_root / commit).resolve()
    if stage.exists():
        shutil.rmtree(stage)
    build, install = stage / "build", stage / "install"
    stage.mkdir(parents=True)
    env = {**os.environ, "REPO": str(repo.resolve()), "SOURCE_DIR": str((repo / "native").resolve()),
           "BUILD_DIR": str(build), "INSTALL_DIR": str(install)}
    subprocess.run([str(repo / "scripts/frontier/build_native_resilient_dataplane.sh")],
                   cwd=repo, env=env, check=True)
    manifest = install / "native-artifacts.json"
    native = native_identity(manifest)
    value = json.loads(manifest.read_text())
    if value.get("source_commit") != commit or value.get("source_tree_dirty") is not False:
        raise ValueError("rebuilt native bundle is not bound to clean authoritative main")
    attestation = {"schema": "emender-authoritative-native-stage-v1", "source": source,
                   "build_manifest": str(manifest), "build_manifest_sha256": sha256(manifest),
                   "native_bundle": native}
    target = stage / "authoritative-stage.json"
    target.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    attestation["attestation"] = str(target)
    attestation["attestation_sha256"] = sha256(target)
    verify_source_identity(repo, source)
    return manifest, attestation


def native_identity(manifest: Path) -> dict[str, Any]:
    value = json.loads(manifest.read_text())
    artifacts: dict[str, str] = {}
    for name in ("service_binary", "local_library", "transport_library"):
        relative = value["artifacts"][name]["path"]
        path = (manifest.parent / relative).resolve()
        path.relative_to(manifest.parent.resolve())
        if not path.is_file() or (name == "service_binary" and not os.access(path, os.X_OK)):
            raise ValueError(f"required native artifact is missing: {name}")
        actual = sha256(path)
        expected = value["artifacts"][name].get("sha256")
        if expected and expected != actual:
            raise ValueError(f"native artifact digest mismatch: {name}")
        artifacts[name] = actual
    return {"manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
            "artifacts": artifacts, "bundle_sha256": value.get("bundle_sha256", "")}


def build_plan(repo: Path, commit: str, manifest: Path, gate: Path, run_root: Path,
               stage: dict[str, Any] | None = None) -> dict[str, Any]:
    native = native_identity(manifest)
    seed = canonical_seed(repo)
    recorded_source = json.loads(manifest.read_text()).get("source_commit")
    if recorded_source != commit:
        raise ValueError("native build does not match the launched source commit")
    if not gate.is_file():
        raise ValueError("retained exact-code G2 full-layout gate is missing")
    common = {
        "nodes": 2, "trainers_per_node": 8, "local_steps": 40,
        "dataplane": "native-cxi", "provider": "cxi", "walltime": WALLTIME,
        "partition": PARTITION, "qos": QOS, "seed": seed,
        "source_commit": commit, "native_bundle": native,
        "authoritative_stage": stage,
        "full_layout_gate": str(gate.resolve()), "full_layout_gate_sha256": sha256(gate),
        "stage_deadlines": STAGE_DEADLINES,
        "launcher": "scripts/frontier/resilient_e97_true_2n.sbatch",
    }
    # Every phase has a distinct allocation/fence.  Restart phases consume only
    # the preceding authoritative handoff; source and bundle remain immutable.
    specs = [
        ("clean-overlap", 5, 0, None, {}),
        ("fault-rejoin", 3, 5, "clean-overlap", {"RESILIENT_E97_INJECT_NATIVE_SERVICE": "1:-1:6"}),
        ("invalid-result-rejection", 2, 8, "fault-rejoin", {"RESILIENT_E97_INJECT_INVALID_RESULT": "1:9"}),
        ("checkpoint-publication-failure", 1, 10, "invalid-result-rejection", {"RESILIENT_E97_INJECT_PUBLICATION_FAILURE": "10"}),
        ("fresh-restart", 2, 10, "checkpoint-publication-failure", {"RESILIENT_E97_FRESH_RESTART": "1"}),
    ]
    phases = []
    for fence, (name, generations, initial, restart_from, injection) in enumerate(specs, 1):
        phase = dict(common)
        phase.update({"name": name, "fence_ordinal": fence, "generations": generations,
                      "initial_generation": initial, "final_generation": initial + generations,
                      "restart_from": restart_from, "run_dir": str((run_root / name).resolve()),
                      "injection": injection})
        if name == "clean-overlap":
            phase["performance_gate"] = {
                "foreground_idle_fraction_strict_max": 0.10,
                "steady_state_cadence_multiple_max": 1.25,
                "requires_background_g_overlap_k40_g_plus_1": True,
            }
        phases.append(phase)
    return {
        "schema": "emender-real-e97-exact-2n-acceptance-v1",
        "source_commit": commit, "node_count": 2, "k_local_steps": 40,
        "seed": seed, "seed_config": str(SEED_CONFIG),
        "queue": {"partition": PARTITION, "qos": QOS},
        "payload_parity": {
            "canonical_profiles": str(SEED_CONFIG),
            "authorized_differences": [
                "nodes", "qos", "walltime", "failure_injection",
            ],
            "seed_and_training_payload_identical": True,
        },
        "authoritative_stage": stage,
        "walltime_per_phase": WALLTIME, "phases": phases,
        "forbidden_node_counts": [4, 8, 32, 64, 256],
        "conformance": {"authority": "RESILIENT_DILOCO_COMPUTE_POOL.md version 1",
                        "requirements": [f"R{i:02d}" for i in range(1, 17)],
                        "native_requirements": [f"NDP{i:02d}" for i in range(1, 18)]},
    }


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
EXPECTED_FAILURES = {"checkpoint-publication-failure"}
SACCT_PROPAGATION_WINDOW_S = 120.0


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def render_batch_script(repo: Path, launcher: str, destination: Path) -> Path:
    """Render the exact submitted script without expanding batch variables."""
    source = (repo / launcher).resolve()
    source.relative_to(repo.resolve())
    payload = source.read_text()
    deferred = "'/tmp/emender-e97-seed-${SLURM_JOB_ID}'"
    if deferred not in payload:
        raise ValueError(
            "exact launcher must preserve the literal SLURM_JOB_ID seed template")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload)
    destination.chmod(source.stat().st_mode)
    if deferred not in destination.read_text():
        raise ValueError("submit-time rendering expanded the batch SLURM_JOB_ID")
    return destination.resolve()


def _scheduler_state(job_id: str) -> dict[str, str]:
    queued = subprocess.check_output(
        ["squeue", "-h", "-j", job_id, "-o", "%T|%P|%q"],
        text=True,
    ).strip()
    if queued:
        fields = queued.splitlines()[0].split("|")
        if len(fields) != 3:
            raise ValueError("squeue did not return explicit State, Partition, and QOS")
        return {
            "state": fields[0],
            "exit_code": "",
            "partition": fields[1],
            "qos": fields[2],
        }
    line = subprocess.check_output(
        [
            "sacct", "-n", "-X", "-j", job_id,
            "--format=State,ExitCode,Partition,QOS", "-P",
        ],
        text=True,
    ).strip().splitlines()
    if not line:
        return {
            "state": "ACCOUNTING_PENDING",
            "exit_code": "",
            "partition": "",
            "qos": "",
        }
    fields = line[0].split("|")
    if len(fields) < 4:
        raise ValueError("sacct did not return explicit State, Partition, and QOS")
    return {
        "state": fields[0].split("+")[0],
        "exit_code": fields[1],
        "partition": fields[2],
        "qos": fields[3],
    }


def _require_exact_debug_queue(result: dict[str, str]) -> None:
    if result["state"] == "ACCOUNTING_PENDING":
        return
    if result.get("partition") != PARTITION or result.get("qos") != QOS:
        raise ValueError(
            "exact two-node acceptance requires scheduler evidence with "
            "Partition=batch and QOS=debug")


def advance(plan: dict[str, Any], output: Path, state_path: Path, repo: Path) -> int:
    """Advance by at most one submission; return 75 while a job must be awaited."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "schema": "emender-exact-2n-serial-state-v1", "next_phase": 0, "active": None, "history": []}
    active = state.get("active")
    if active:
        result = _scheduler_state(active["job_id"])
        _require_exact_debug_queue(result)
        if result["state"] not in TERMINAL_STATES:
            if result["state"] == "ACCOUNTING_PENDING":
                pending_since = float(state.get("accounting_pending_since", time.time()))
                state["accounting_pending_since"] = pending_since
                if time.time() - pending_since > SACCT_PROPAGATION_WINDOW_S:
                    raise TimeoutError(
                        f"job {active['job_id']} remained absent from squeue and sacct "
                        f"for more than {SACCT_PROPAGATION_WINDOW_S:g}s")
            else:
                state.pop("accounting_pending_since", None)
            state["wait"] = {"kind": "slurm-terminal", "job_id": active["job_id"],
                             "observed_state": result["state"], "resumable": True,
                             "scheduler_evidence": result}
            _atomic_json(state_path, state)
            print(f"WAIT phase={active['phase']} job_id={active['job_id']} state={result['state']}")
            return 75
        state.pop("accounting_pending_since", None)
        phase = active["phase"]
        successful = result["state"] == "COMPLETED"
        if successful == (phase in EXPECTED_FAILURES):
            raise ValueError(f"phase {phase} had unexpected terminal state {result['state']}")
        run_dir = Path(active["run_dir"])
        harvest = run_dir / "scheduler-terminal.json"
        _atomic_json(harvest, {**result, "job_id": active["job_id"], "phase": phase,
                               "harvested_unix_seconds": int(time.time())})
        state["history"].append({**active, **result, "terminal_artifact": str(harvest),
                                 "terminal_artifact_sha256": sha256(harvest)})
        state["active"] = None
        state.pop("wait", None)
        state["next_phase"] += 1
        _atomic_json(state_path, state)
    index = state["next_phase"]
    if index >= len(plan["phases"]):
        state["complete"] = True
        _atomic_json(state_path, state)
        print("acceptance phases complete")
        return 0
    # This check is intentionally immediately adjacent to sbatch. It detects
    # tracked or untracked repository mutation after the build attestation.
    verify_source_identity(repo, plan["authoritative_stage"]["source"])
    queued = subprocess.check_output(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], text=True).strip()
    if queued:
        raise ValueError("refusing to overlap another user allocation")
    phase = plan["phases"][index]
    seed = canonical_seed(repo)
    if plan.get("seed") != seed:
        raise ValueError("rendered seed identity drifted from canonical E97 config")
    if plan.get("queue") != {"partition": PARTITION, "qos": QOS}:
        raise ValueError("rendered exact queue is not Partition=batch and QOS=debug")
    run_dir = Path(phase["run_dir"]); run_dir.mkdir(parents=True, exist_ok=True)
    # Submit/login-side only: resolve both S3 authorities and acquire the
    # checkpoint before sbatch. The allocation receives only verified,
    # digest-pinned files and never receives an S3 credential or URI to fetch.
    cache_root = Path(os.environ.get(
        "RESILIENT_E97_SUBMIT_SEED_CACHE_ROOT", str(SEED_CACHE_ROOT)))
    seed_cache, seed_attestation = prefetch(
        seed, cache_root, run_dir / "seed-bootstrap-attestation.json")
    seed_attestation_sha256 = sha256(seed_attestation)
    exports = {
        "REPO": str(repo.resolve()),
        "RESILIENT_E97_ACCEPTANCE_MANIFEST": str(output.resolve()),
        "RESILIENT_E97_ACCEPTANCE_PHASE": phase["name"], "RUN_DIR": str(run_dir),
        "NDP_BUILD_MANIFEST": plan["authoritative_stage"]["build_manifest"],
        "NDP_FULL_LAYOUT_GATE_JSON": phase["full_layout_gate"],
        "EMENDER_CONDA_ENV": APPROVED_ENV,
        "DILOCO_DATAPLANE": "native-cxi", "FI_PROVIDER": "cxi",
        "RESILIENT_E97_RUN_ID": f"exact-2n-{phase['name']}-{plan['source_commit'][:12]}",
        "RESILIENT_E97_SOURCE_ID": (
            f"step-{seed['step']}-tokens-{seed['tokens']}-sha256-{seed['sha256']}"
        ),
        "RESILIENT_E97_PAYLOAD_ID": f"{plan['source_commit'][:12]}-{phase['name']}-e97-k40",
        "RESILIENT_E97_CODE_ID": plan["source_commit"],
        "RESILIENT_E97_SEED_CONFIG": str((repo / SEED_CONFIG).resolve()),
        "RESILIENT_E97_SEED_STEP": str(seed["step"]),
        "RESILIENT_E97_SEED_TOKENS": str(seed["tokens"]),
        "RESILIENT_E97_SEED_SIZE": str(seed["size"]),
        "RESILIENT_E97_SEED_SHA256": seed["sha256"],
        "RESILIENT_E97_SEED_CACHE": str(seed_cache),
        "RESILIENT_E97_SEED_ATTESTATION": str(seed_attestation),
        "RESILIENT_E97_SEED_ATTESTATION_SHA256": seed_attestation_sha256,
        "RESILIENT_E97_TRAIN_ARGS_JSON": str((repo / "configs/frontier/e97_resilient_split_role_flat.json").resolve()),
        "RESILIENT_E97_DATA": DATA, "RESILIENT_E97_TIKTOKEN_CACHE_FILE": TIKTOKEN,
        "RESILIENT_E97_TIKTOKEN_SHA256": TIKTOKEN_SHA256,
        "RESILIENT_E97_NODE_COUNT": "2", "RESILIENT_E97_GENERATIONS": str(phase["generations"]),
        "RESILIENT_E97_INITIAL_GENERATION": str(phase["initial_generation"]),
        "RESILIENT_E97_COORDINATOR_EPOCH": str(phase["fence_ordinal"]),
        "RESILIENT_E97_GLOBAL_QUORUM": "2", "RESILIENT_E97_GLOBAL_TOKEN_MIN": "3934080",
        "RESILIENT_E97_STARTUP_SMOKE": "0", "RESILIENT_E97_REQUESTED_WALLTIME": WALLTIME,
        "RESILIENT_E97_BULK_ROOT": f"/tmp/exact-2n-{plan['source_commit'][:12]}-{phase['name']}",
        "RESILIENT_E97_GENERATION_DEADLINE_S": str(STAGE_DEADLINES["quorum_s"]), **phase["injection"],
    }
    if state["history"]:
        exports["RESILIENT_E97_RESUME_HANDOFF"] = state["history"][-1]["terminal_artifact"]
    launcher = render_batch_script(
        repo, phase["launcher"], run_dir / "rendered.sbatch")
    command = ["sbatch", "--parsable", "-N", "2", "-t", WALLTIME,
               "-p", "batch", "--qos=debug", "--network=job_vni",
               "--chdir", str(repo.resolve()),
               "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()), str(launcher)]
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    state["active"] = {"phase": phase["name"], "job_id": job_id, "run_dir": str(run_dir),
                       "submitted_unix_seconds": int(time.time()),
                       "scheduler_request": {"partition": PARTITION, "qos": QOS}}
    state["wait"] = {"kind": "slurm-terminal", "job_id": job_id, "observed_state": "SUBMITTED", "resumable": True}
    _atomic_json(state_path, state)
    print(f"SUBMITTED phase={phase['name']} job_id={job_id}; rerun to wait/advance")
    return 75


def submit(plan: dict[str, Any], output: Path, state_path: Path, repo: Path) -> int:
    # Kept as a named entry point for callers; unlike the former dependency
    # loop, one invocation can never submit more than one job.
    return advance(plan, output, state_path, repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--native-build-manifest", type=Path, required=True)
    parser.add_argument("--full-layout-gate", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--state", type=Path, help="resumable serial submission state")
    parser.add_argument("--native-stage-root", type=Path,
                        help="clean authoritative build/stage root (required with --submit)")
    parser.add_argument("--allow-non-authoritative-dry-run", action="store_true",
                        help="test/review only; never allowed with --submit")
    args = parser.parse_args(argv)
    if args.submit and args.allow_non_authoritative_dry_run:
        parser.error("--allow-non-authoritative-dry-run cannot be submitted")
    try:
        commit = (git(args.repo, "rev-parse", "HEAD") if args.allow_non_authoritative_dry_run
                  else require_authoritative_source(args.repo, check_allocation=args.submit))
        stage = None
        manifest = args.native_build_manifest
        if args.submit and args.state and args.state.exists():
            if not args.output.is_file():
                raise ValueError("serial state exists but acceptance manifest is missing")
            plan = json.loads(args.output.read_text())
            if plan.get("source_commit") != commit or not plan.get("authoritative_stage"):
                raise ValueError("serial state does not match authoritative main")
            return submit(plan, args.output, args.state, args.repo)
        if args.submit:
            if not args.native_stage_root or not args.state:
                parser.error("--submit requires --native-stage-root and --state")
            manifest, stage = build_authoritative_bundle(args.repo, commit, args.native_stage_root)
            subprocess.run([sys.executable, str(args.repo / "scripts/frontier/attest_native_dataplane.py"),
                            "verify", "--backend", "native-cxi", "--production", "--full-layout",
                            "--build-manifest", str(manifest), "--gate-json", str(args.full_layout_gate),
                            "--source-root", str(args.repo)], check=True)
        plan = build_plan(args.repo, commit, manifest, args.full_layout_gate, args.run_root, stage)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        if args.submit:
            return submit(plan, args.output, args.state, args.repo)
        else:
            print(args.output)
        return 0
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"acceptance launcher refused: {error}", file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
