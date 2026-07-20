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
import subprocess
import sys
from typing import Any


WALLTIME = "02:00:00"
STAGE_DEADLINES = {
    "handoff_s": 180,
    "apply_s": 180,
    "quorum_s": 420,
    "integrity_s": 120,
    "publication_s": 180,
    "progress_s": 420,
}


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
    if subprocess.call(["git", "-C", str(repo), "diff", "--quiet", "--ignore-submodules", "--"]):
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


def build_plan(repo: Path, commit: str, manifest: Path, gate: Path, run_root: Path) -> dict[str, Any]:
    native = native_identity(manifest)
    if not gate.is_file():
        raise ValueError("retained exact-code G2 full-layout gate is missing")
    common = {
        "nodes": 2, "trainers_per_node": 8, "local_steps": 40,
        "dataplane": "native-cxi", "provider": "cxi", "walltime": WALLTIME,
        "source_commit": commit, "native_bundle": native,
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
        phases.append(phase)
    return {
        "schema": "emender-real-e97-exact-2n-acceptance-v1",
        "source_commit": commit, "node_count": 2, "k_local_steps": 40,
        "walltime_per_phase": WALLTIME, "phases": phases,
        "forbidden_node_counts": [4, 8, 32, 64, 256],
        "conformance": {"authority": "RESILIENT_DILOCO_COMPUTE_POOL.md version 1",
                        "requirements": [f"R{i:02d}" for i in range(1, 17)],
                        "native_requirements": [f"NDP{i:02d}" for i in range(1, 18)]},
    }


def submit(plan: dict[str, Any], output: Path) -> None:
    previous = ""
    for phase in plan["phases"]:
        exports = {
            "RESILIENT_E97_ACCEPTANCE_MANIFEST": str(output.resolve()),
            "RESILIENT_E97_ACCEPTANCE_PHASE": phase["name"],
            "RESILIENT_E97_NODE_COUNT": "2", "RESILIENT_E97_GENERATIONS": str(phase["generations"]),
            "RESILIENT_E97_INITIAL_GENERATION": str(phase["initial_generation"]),
            "RESILIENT_E97_GENERATION_DEADLINE_S": str(STAGE_DEADLINES["quorum_s"]),
            **phase["injection"],
        }
        command = ["sbatch", "--parsable", "-N", "2", "-t", WALLTIME,
                   "--qos=debug", "--network=job_vni"]
        if previous:
            # Publication failure is expected to exit non-zero; its restart is
            # consequently chained with afternotok, all other edges afterok.
            dependency = "afternotok" if phase["name"] == "fresh-restart" else "afterok"
            command += [f"--dependency={dependency}:{previous}"]
        command += ["--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()),
                    phase["launcher"]]
        previous = subprocess.check_output(command, text=True).strip()
        print(f"{phase['name']}={previous}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--native-build-manifest", type=Path, required=True)
    parser.add_argument("--full-layout-gate", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--allow-non-authoritative-dry-run", action="store_true",
                        help="test/review only; never allowed with --submit")
    args = parser.parse_args(argv)
    if args.submit and args.allow_non_authoritative_dry_run:
        parser.error("--allow-non-authoritative-dry-run cannot be submitted")
    try:
        commit = (git(args.repo, "rev-parse", "HEAD") if args.allow_non_authoritative_dry_run
                  else require_authoritative_source(args.repo, check_allocation=args.submit))
        plan = build_plan(args.repo, commit, args.native_build_manifest,
                          args.full_layout_gate, args.run_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        if args.submit:
            submit(plan, args.output)
        else:
            print(args.output)
        return 0
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"acceptance launcher refused: {error}", file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
