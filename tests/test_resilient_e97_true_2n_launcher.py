from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from ndm.manifest_peer_control import ManifestPeerAuthority
from scripts.frontier.resilient_e97_allocation_supervisor import AllocationSupervisor, Child
from scripts.frontier.resilient_e97_allocation_supervisor import (
    _node_local_child_cpu_set,
)
from scripts.frontier.check_resilient_e97_parity import compare


ROOT = Path(__file__).parents[1]
APPROVED_ENV = Path("/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312")


def _publish_commit_chain(
        root: Path, *, fence: int, generation: int, run_id: str = "run",
        payload_id: str = "payload", source_id: str = "source",
        code_id: str = "code", native_runtime: dict | None = None):
    authority = ManifestPeerAuthority(root)
    claim = authority.claim(
        run_id=run_id,
        allocation_id=f"job-{fence}",
        incarnation=f"allocation-{fence}",
        fence=fence,
        protocol_id="async-decoupled-v2.1-simple",
        config_id="config",
    )
    prior_root = ""
    receipt = authority.current_commit(claim)
    start = 1 if receipt is None else receipt.generation + 1
    for item in range(start, generation + 1):
        checkpoint = (
            root / "checkpoints"
            / f"generation-{item:08d}-fence-{fence:08d}.pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint:{run_id}:{fence}:{item}".encode())
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result_root = hashlib.sha256(
            f"result:{run_id}:{fence}:{item}".encode()).hexdigest()
        manifest = (
            root / "handoff"
            / f"generation-{item:08d}-fence-{fence:08d}.json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "schema": 2,
            "finalized": True,
            "run_id": run_id,
            "payload_id": payload_id,
            "source_id": source_id,
            "code_id": code_id,
            "generation": item,
            "step": item * 40,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "accepted_tokens": item * 3_934_080,
            "outer_update_state": {
                "mode": "delta_sgd",
                "eta_outer": 1.0,
                "step": item,
                "accepted_tokens": item * 3_934_080,
            },
            "membership": [
                {"worker_id": "node-0", "incarnation": f"node-0-{fence}"},
                {"worker_id": "node-1", "incarnation": f"node-1-{fence}"},
            ],
            "fence": {"coordinator_epoch": fence},
            "digests": {
                "result_root": result_root,
                "previous_result_root": prior_root or "00" * 32,
                **({"native_runtime": native_runtime}
                   if native_runtime is not None else {}),
            },
        }, sort_keys=True) + "\n")
        receipt = authority.publish_checkpoint(claim, manifest)
        prior_root = result_root
    return authority, claim, receipt


def _fake_native_manifest(root: Path) -> Path:
    prefix = root / "native"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    service = prefix / "bin/ndp_cxi_service"
    service.write_text("#!/bin/sh\nexit 0\n"); service.chmod(0o755)
    local = prefix / "lib/libemender_ndp.so.1"; local.write_bytes(b"local")
    transport = prefix / "lib/libemender_ndp_transport.so.1"; transport.write_bytes(b"transport")
    manifest = prefix / "native-artifacts.json"
    manifest.write_text(json.dumps({"artifacts": {
        "service_binary": {"path": "bin/ndp_cxi_service"},
        "local_library": {"path": "lib/libemender_ndp.so.1"},
        "transport_library": {"path": "lib/libemender_ndp_transport.so.1"},
    }}))
    return manifest


def _install_fake_canonical_seed(repo: Path) -> tuple[Path, dict]:
    config = repo / "configs/frontier/e97_async_256.yaml"
    config.write_bytes((ROOT / "configs/frontier/e97_async_256.yaml").read_bytes())
    seed = json.loads(config.read_text())["seed"]
    materializer = repo / "scripts/frontier/materialize_e97_s3_seed.py"
    materializer.write_text(
        "import argparse\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--seed-config')\n"
        "p.add_argument('--destination')\n"
        "p.add_argument('--verify-local', action='store_true')\n"
        "p.add_argument('--attestation')\n"
        "p.add_argument('--expected-job-id')\n"
        "p.add_argument('--expected-attestation-sha256')\n"
        "p.add_argument('--runtime-manifest')\n"
        "p.parse_args()\n"
    )
    return config, seed


def _canonical_seed_env(config: Path, seed: dict, job_id: str) -> dict[str, str]:
    seed_cache = config.parent / f"sha256-{seed['sha256']}.pt"
    seed_cache.write_bytes(b"verified-submit-side-seed-fixture")
    attestation = config.parent / f"{job_id}-seed-attestation.json"
    attestation.write_text('{"status":"verified"}\n')
    return {
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_PARTITION": "batch",
        "RESILIENT_E97_SEED_CONFIG": str(config),
        "RESILIENT_E97_SEED_STEP": str(seed["step"]),
        "RESILIENT_E97_SEED_TOKENS": str(seed["tokens"]),
        "RESILIENT_E97_SEED_SIZE": str(seed["size"]),
        "RESILIENT_E97_SEED_SHA256": seed["sha256"],
        "RESILIENT_E97_SEED_CACHE": str(seed_cache),
        "RESILIENT_E97_SEED_ATTESTATION": str(attestation),
        "RESILIENT_E97_SEED_ATTESTATION_SHA256":
            hashlib.sha256(attestation.read_bytes()).hexdigest(),
    }


def test_true_launcher_is_exact_debug_twenty_minute_topology_without_sentinels():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -q debug" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH --gpus-per-node=8" in text
    assert "#SBATCH -t 00:20:00" in text
    assert "#SBATCH --signal=B:TERM@300" in text
    assert '"RESILIENT_E97_ROLE": child.role' in supervisor
    assert '"CUDA_VISIBLE_DEVICES="' in supervisor
    assert '"--overlap", "--no-kill", "--exact"' in supervisor
    assert '"ASYNC_LOCAL_STEPS=40"' in supervisor
    assert "topology managers=%s real_trainers=%s" in text
    assert '"$((RESILIENT_E97_NODE_COUNT * 8))"' in text
    assert "resilient_e97_rank_lane.py" not in text
    assert "sentinel ranks" in text


def test_exact_launcher_materializes_canonical_seed_node_locally_before_model_load():
    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    materialize = launcher.index("materialize_e97_s3_seed.py")
    trainer_command = launcher.index("RESILIENT_E97_TRAINER_COMMAND=")
    supervisor = launcher.index("resilient_e97_allocation_supervisor.py")
    assert '[[ ${SLURM_JOB_PARTITION:?} == batch ]]' in launcher
    assert '[[ ${SLURM_JOB_QOS:?} == debug ]]' in launcher
    assert 'APPROVED_SEED_CONFIG="$REPO/configs/frontier/e97_async_256.yaml"' in launcher
    assert 'RESILIENT_E97_SEED_CONFIG:?' in launcher
    assert "/tmp/emender-e97-seed-$SLURM_JOB_ID" in launcher
    assert "--seed-config" in launcher and "--runtime-manifest" in launcher
    assert "--nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT" in launcher
    assert materialize < trainer_command < supervisor
    assert ": \"${RESILIENT_E97_SEED:?set verified generation-9 seed checkpoint}\"" not in launcher
    assert "PINNED_STEP_1525000_SHA256" not in role
    load_real = role[role.index("def _load_real(args)"):]
    assert load_real.index("RESILIENT_E97_SEED_SHA256") < load_real.index("torch.load(")
    assert load_real.index("RESILIENT_E97_SEED_STEP") < load_real.index("torch.load(")
    assert "pinned step-1525000" not in load_real


def test_legacy_runner_cannot_misreport_sentinels_as_real_trainers():
    legacy = (ROOT / "scripts/frontier/resilient_e97_rank_lane.py").read_text()
    assert '"role": "sentinel"' in legacy
    true_launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "RESILIENT_E97_MANAGER_COMMAND" in true_launcher
    assert "RESILIENT_E97_TRAINER_COMMAND" in true_launcher


def test_allocation_supervisor_uses_independent_restartable_steps():
    text = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert "TRAINERS_PER_NODE = 8" in text
    assert 'Child("manager"' in text
    assert 'Child("trainer"' in text
    assert '"--overlap", "--no-kill", "--exact"' in text
    assert '"--gpus-per-node=8"' in text
    assert '"--gpus-per-task=1"' not in text
    assert '"ASYNC_LOCAL_STEPS=40"' in text
    assert '"heartbeat_deadline"' in text
    assert '"progress_deadline"' in text
    assert '"allocation_term_handoff"' in text
    assert '"restart_exhausted"' in text
    assert "MPI" not in text


def test_launcher_discovers_coordinator_and_wires_exact_restart_handoff():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert 'scontrol show hostnames "$SLURM_JOB_NODELIST"' in text
    assert 'RESILIENT_E97_COORDINATOR_HOST:-${ALLOCATION_NODES[0]}' in text
    assert "--initial-generation $RESILIENT_E97_INITIAL_GENERATION" in text
    assert '--resume-handoff "$RESILIENT_E97_RESUME_HANDOFF"' in text


def test_v21_launcher_defaults_two_nodes_and_scale_requires_serial_ladder():
    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    supervisor = (
        ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    ).read_text()

    assert "RESILIENT_E97_NODE_COUNT=${RESILIENT_E97_NODE_COUNT:-2}" in launcher
    assert "2|4|8|16|32|64|256" in launcher
    assert '[[ ${ASYNC_V21_GATE:-} == scale ]]' in launcher
    assert "ASYNC_V21_SCALE_AUTHORIZATION" in launcher
    assert "ASYNC_V21_PRIOR_RUNG_PASS" in launcher
    assert '[[ ${SLURM_JOB_NUM_NODES:?} == "$RESILIENT_E97_NODE_COUNT" ]]' in launcher
    assert "--nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT" in launcher
    assert "--node-count $RESILIENT_E97_NODE_COUNT" in launcher
    assert 'RESILIENT_E97_NODE_COUNT", "2"' in supervisor
    assert "node_count not in QUALIFICATION_NODE_LADDER" in supervisor
    assert 'os.environ.get("ASYNC_V21_GATE") != "scale"' in supervisor
    assert "ASYNC_V21_SCALE_AUTHORIZATION" in supervisor
    assert "ASYNC_V21_PRIOR_RUNG_PASS" in supervisor
    assert "ASYNC_V21_SCALE_CLOSURE_DIGEST" in supervisor
    assert 'f"-N{node_count}", f"-n{node_count}"' in supervisor
    assert "requires exactly two physical nodes" not in supervisor


def test_launcher_omits_empty_resume_argument(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/frontier").mkdir(parents=True)
    (repo / "configs/frontier").mkdir(parents=True)
    approved_args = repo / "configs/frontier/e97_resilient_split_role_flat.json"
    approved_args.write_text((ROOT / "configs/frontier/e97_resilient_split_role_flat.json").read_text())
    seed_config, seed = _install_fake_canonical_seed(repo)
    supervisor = repo / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    supervisor.write_text(
        "import os\nprint(os.environ['RESILIENT_E97_TRAINER_COMMAND'])\n"
    )
    (repo / "scripts/frontier/attest_native_dataplane.py").write_text(
        "import json,sys\nprint(json.dumps({'status':'attested'}))\n"
    )
    (repo / "scripts/frontier/frontier_runtime_env.sh").write_text(
        "frontier_load_default_modules() { :; }\n"
        "frontier_activate_emender_conda_env() { :; }\n"
        "frontier_assert_emender_conda_env() { :; }\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scontrol = bindir / "scontrol"
    scontrol.write_text("#!/bin/sh\nprintf 'node0\\nnode1\\n'\n")
    scontrol.chmod(0o755)
    srun = bindir / "srun"
    srun.write_text(
        "#!/bin/sh\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in bash|rm) exec \"$@\" ;; *) shift ;; esac\n"
        "done\n"
        "exit 64\n")
    srun.chmod(0o755)
    sbcast = bindir / "sbcast"
    sbcast.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = -f ] || exit 64\n"
        "cp \"$2\" \"$3\"\n")
    sbcast.chmod(0o755)
    tokenizer_cache = tmp_path / "p50k.cache"
    tokenizer_cache.write_text("offline-tokenizer-cache")
    native_manifest = _fake_native_manifest(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{APPROVED_ENV / 'bin'}:{os.environ['PATH']}",
        "EMENDER_CONDA_ENV": str(APPROVED_ENV),
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "debug",
        "SLURM_TIMELIMIT": "00:20:00",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "REPO": str(repo),
        "RUN_DIR": str(tmp_path / "run"),
        "RESILIENT_E97_RUN_ID": "run",
        "RESILIENT_E97_SOURCE_ID": "source",
        "RESILIENT_E97_PAYLOAD_ID": "payload",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
        "RESILIENT_E97_TIKTOKEN_CACHE_FILE": str(tokenizer_cache),
        "RESILIENT_E97_TIKTOKEN_SHA256": hashlib.sha256(tokenizer_cache.read_bytes()).hexdigest(),
        "NDP_BUILD_MANIFEST": str(native_manifest),
        "NDP_FULL_LAYOUT_GATE_JSON": str(tmp_path / "full-layout-gate.json"),
        **_canonical_seed_env(seed_config, seed, "pytest-empty-resume"),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch")],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    trainer_command = result.stdout.splitlines()[-1]
    assert trainer_command.endswith("--migration-policy initialize-from-approved-config")
    assert "  " not in trainer_command


def test_startup_smoke_is_short_one_generation_and_forbids_injection():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "RESILIENT_E97_STARTUP_SMOKE" in text
    assert "startup smoke requires exactly 00:20:00" in text
    assert "startup smoke requires exactly one finalized generation" in text
    assert "startup smoke forbids failure injection" in text
    assert "RESILIENT_E97_REQUESTED_WALLTIME" in text


def test_full_layout_launcher_requires_native_cxi_and_exact_artifact_gate_before_roles():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    attestation = text.index("attest_native_dataplane.py")
    roles = text.index("RESILIENT_E97_MANAGER_COMMAND=")
    assert "DILOCO_DATAPLANE=${DILOCO_DATAPLANE:-native-cxi}" in text
    assert "full-layout Frontier launcher requires DILOCO_DATAPLANE=native-cxi" in text
    assert "FI_PROVIDER=${FI_PROVIDER:-cxi}" in text
    assert "--production --full-layout" in text
    assert "NDP_BUILD_MANIFEST" in text
    assert "NDP_FULL_LAYOUT_GATE_JSON" in text
    assert attestation < roles
    assert "python-tcp-debug" not in text


def test_node_local_topology_starts_persistent_service_before_model_free_manager():
    source = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert 'Child("native-service"' in source
    assert 'Child("manager"' in source
    assert "for rank in range(TRAINERS_PER_NODE)" in source
    assert source.index('Child("native-service"') < source.index('Child("manager"')
    assert "--admission-token-fd" in source
    assert "EMENDER_NDP_ADMISSION_TOKEN_FD" in source
    assert "NDP_SERVICE_COMMAND" in launcher
    assert 'NDP_SERVICE_BINARY=${NDP_ARTIFACT_PATHS[0]}' in launcher
    assert "--provider cxi --require-provider cxi --production --serve" in launcher
    assert "--domain cxi0" in launcher
    assert "export EMENDER_NDP_MAX_SHARED_BYTES=" in launcher
    assert "--resident-limit ${EMENDER_NDP_MAX_SHARED_BYTES}" in launcher
    assert "--local-quorum 8" in launcher


def test_launcher_activates_approved_frontier_python_before_any_role():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    activation = text.index("frontier_activate_emender_conda_env")
    supervisor = text.index('"$TRAIN_PYTHON_BIN" "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"')
    assert 'source "$REPO/scripts/frontier/frontier_runtime_env.sh"' in text
    assert "frontier_load_default_modules" in text
    assert "frontier_assert_emender_conda_env" in text
    assert activation < supervisor
    assert 'RESILIENT_E97_MANAGER_COMMAND="$TRAIN_PYTHON_BIN ' in text
    assert 'RESILIENT_E97_TRAINER_COMMAND="$TRAIN_PYTHON_BIN ' in text
    assert 'exec python3 "$REPO/scripts/frontier/resilient_e97_allocation_supervisor.py"' not in text


def test_launcher_requires_and_attests_exact_known_good_runtime():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    approved = "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312"
    assert f"APPROVED_EMENDER_CONDA_ENV={approved}" in text
    assert 'EMENDER_CONDA_ENV:?' in text
    assert 'realpath "$EMENDER_CONDA_ENV"' in text
    assert 'realpath "$APPROVED_EMENDER_CONDA_ENV/bin/python"' in text
    assert '"python": "3.12.13"' in text
    assert '"torch": "2.10.0+rocm7.1"' in text
    assert '"torch_hip": "7.1.25424"' in text
    assert '"triton": "3.6.0"' in text
    assert '"$RUN_DIR/runtime-identity.json"' in text
    assert text.index("runtime identity mismatch") < text.index("RESILIENT_E97_MANAGER_COMMAND=")


def test_launcher_rejects_omitted_or_wrong_runtime_before_module_activation(tmp_path):
    script = ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch"
    base = {**os.environ, "REPO": str(ROOT)}
    base.pop("EMENDER_CONDA_ENV", None)
    omitted = subprocess.run(["bash", str(script)], env=base, text=True, capture_output=True)
    assert omitted.returncode != 0
    assert "immutable submit payload must export EMENDER_CONDA_ENV" in omitted.stderr

    wrong = subprocess.run(
        ["bash", str(script)], env={**base, "EMENDER_CONDA_ENV": str(tmp_path)},
        text=True, capture_output=True,
    )
    assert wrong.returncode == 64
    assert "EMENDER_CONDA_ENV must resolve" in wrong.stderr


def test_startup_smoke_accepts_explicit_walltime_when_slurm_omits_environment(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/frontier").mkdir(parents=True)
    (repo / "configs/frontier").mkdir(parents=True)
    approved_args = repo / "configs/frontier/e97_resilient_split_role_flat.json"
    approved_args.write_text((ROOT / "configs/frontier/e97_resilient_split_role_flat.json").read_text())
    seed_config, seed = _install_fake_canonical_seed(repo)
    (repo / "scripts/frontier/resilient_e97_allocation_supervisor.py").write_text("pass\n")
    (repo / "scripts/frontier/attest_native_dataplane.py").write_text(
        "import json,sys\nprint(json.dumps({'status':'attested'}))\n"
    )
    (repo / "scripts/frontier/frontier_runtime_env.sh").write_text(
        "frontier_load_default_modules() { :; }\n"
        "frontier_activate_emender_conda_env() { :; }\n"
        "frontier_assert_emender_conda_env() { :; }\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scontrol = bindir / "scontrol"
    scontrol.write_text("#!/bin/sh\nprintf 'node0\\nnode1\\n'\n")
    scontrol.chmod(0o755)
    srun = bindir / "srun"
    srun.write_text(
        "#!/bin/sh\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  case \"$1\" in bash|rm) exec \"$@\" ;; *) shift ;; esac\n"
        "done\n"
        "exit 64\n")
    srun.chmod(0o755)
    sbcast = bindir / "sbcast"
    sbcast.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = -f ] || exit 64\n"
        "cp \"$2\" \"$3\"\n")
    sbcast.chmod(0o755)
    tokenizer_cache = tmp_path / "p50k.cache"
    tokenizer_cache.write_text("offline-tokenizer-cache")
    native_manifest = _fake_native_manifest(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{APPROVED_ENV / 'bin'}:{os.environ['PATH']}",
        "EMENDER_CONDA_ENV": str(APPROVED_ENV),
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "debug",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "RESILIENT_E97_STARTUP_SMOKE": "1",
        "RESILIENT_E97_REQUESTED_WALLTIME": "00:20:00",
        "SLURM_TIMELIMIT": "20",
        "RESILIENT_E97_GENERATIONS": "1",
        "REPO": str(repo),
        "RUN_DIR": str(tmp_path / "run"),
        "RESILIENT_E97_RUN_ID": "run",
        "RESILIENT_E97_SOURCE_ID": "source",
        "RESILIENT_E97_PAYLOAD_ID": "payload",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
        "RESILIENT_E97_TIKTOKEN_CACHE_FILE": str(tokenizer_cache),
        "RESILIENT_E97_TIKTOKEN_SHA256": hashlib.sha256(tokenizer_cache.read_bytes()).hexdigest(),
        "NDP_BUILD_MANIFEST": str(native_manifest),
        "NDP_FULL_LAYOUT_GATE_JSON": str(tmp_path / "full-layout-gate.json"),
        **_canonical_seed_env(seed_config, seed, "pytest-startup-smoke"),
    }
    subprocess.run(["bash", str(ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch")],
                   env=env, text=True, capture_output=True, check=True)


def test_approved_training_arguments_are_flat_overrides():
    path = ROOT / "configs/frontier/e97_resilient_split_role_flat.json"
    value = json.loads(path.read_text())
    assert value["level"] == "E97" and value["optimizer"] == "schedulefree"
    assert value["dim"] == 1792 and value["depth"] == 11 and value["lr"] == 0.001007
    assert value["n_groups"] == 32 and value["n_slots"] == 64
    assert value["mlp_ratio"] == 2.2623 and value["mlp_multiple"] == 64
    assert value["batch_size"] == 4 and value["chunk_size"] == 2048
    assert value["gradient_checkpointing"] is False
    assert value["use_chunked_e97"] == 0 and value["e97_chunk_size"] == 32
    assert value["use_triton"] == 1 and value["use_split_edit"] == 1
    assert value["linear_state"] == 0 and value["e88_raw_write"] == 0
    assert value["gate_activation"] == "silu" and value["use_permutation"] == 1
    assert not ({"resolved", "export", "source_artifacts"} & value.keys())


def test_launcher_rejects_nonapproved_training_arguments_before_roles_start():
    script = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert 'APPROVED_TRAIN_ARGS="$REPO/configs/frontier/e97_resilient_split_role_flat.json"' in script
    assert 'realpath "$RESILIENT_E97_TRAIN_ARGS_JSON"' in script
    assert "training arguments must be the approved flat E97 split-role configuration" in script


def test_launcher_stages_verified_p50k_cache_to_each_node_before_roles_start():
    script = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "RESILIENT_E97_TIKTOKEN_CACHE_FILE:?" in script
    assert "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069" in script
    assert "ec7223a39ce59f226a68acc30dc1af2788490e15" in script
    assert "--nodes=$RESILIENT_E97_NODE_COUNT --ntasks=$RESILIENT_E97_NODE_COUNT" in script
    assert "export TIKTOKEN_CACHE_DIR=$RESILIENT_E97_NODE_TIKTOKEN_CACHE" in script
    assert script.index("export TIKTOKEN_CACHE_DIR=") < script.index("RESILIENT_E97_TRAINER_COMMAND=")


def test_launch_modes_preserve_identical_local_role_identity_and_environment(tmp_path, monkeypatch):
    launched = []
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "cache-isolation-test")
    monkeypatch.setenv("RESILIENT_E97_KERNEL_CACHE_ROOT", str(tmp_path / "kernel-cache"))

    class Process:
        pid = 123
        returncode = None
        def poll(self): return None

    def fake_popen(argv, **kwargs):
        launched.append((argv, kwargs.get("env")))
        return Process()

    monkeypatch.setattr("scripts.frontier.resilient_e97_allocation_supervisor.subprocess.Popen",
                        fake_popen)
    child = Child("trainer", 7, "node007", 3, "python trainer.py")
    for backend in ("independent-step", "node-local-child"):
        supervisor = AllocationSupervisor(tmp_path / backend, [child], heartbeat_s=2,
                                          progress_s=3, max_restarts=1,
                                          launch_backend=backend)
        supervisor.start(child)
        assert child.identity == "node-7-trainer-3"
    step_argv, _ = launched[0]
    local_argv, local_env = launched[1]
    assert step_argv[0] == "srun" and "--no-kill" in step_argv
    assert local_argv == ["python", "trainer.py"]
    assert local_env["RESILIENT_E97_ROLE"] == "trainer"
    assert local_env["RESILIENT_E97_NODE_RANK"] == "7"
    assert local_env["RESILIENT_E97_LOCAL_RANK"] == "3"
    assert local_env["ROCR_VISIBLE_DEVICES"] == "3"
    assert local_env["TRITON_CACHE_DIR"] == str(
        tmp_path / "kernel-cache/cache-isolation-test-rank-3/triton")
    assert local_env["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "kernel-cache/cache-isolation-test-rank-3/inductor")
    assert Path(local_env["TRITON_CACHE_DIR"]).is_dir()
    assert Path(local_env["TORCHINDUCTOR_CACHE_DIR"]).is_dir()
    assert any(value.startswith("TRITON_CACHE_DIR=") for value in step_argv)
    assert any(value.startswith("TORCHINDUCTOR_CACHE_DIR=") for value in step_argv)


def test_node_local_trainers_partition_the_existing_fifty_six_cpu_step():
    available = tuple(range(100, 156))
    trainer_sets = [
        _node_local_child_cpu_set(
            Child("trainer", 0, "node000", rank, "trainer"),
            available,
        )
        for rank in range(8)
    ]

    assert all(len(cpus) == 7 for cpus in trainer_sets)
    assert set().union(*trainer_sets) == set(available)
    assert sum(len(cpus) for cpus in trainer_sets) == len(available)
    assert _node_local_child_cpu_set(
        Child("manager", 0, "node000", None, "manager"),
        available,
    ) == set(available)


def test_node_local_native_service_uses_pinned_cxi_domain_without_hostname_bind(
        tmp_path, monkeypatch):
    launched = []

    class Process:
        pid = 123
        returncode = None

        def poll(self):
            return None

    def fake_popen(argv, **kwargs):
        launched.append((argv, kwargs))
        return Process()

    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor.subprocess.Popen",
        fake_popen)
    command = (
        "ndp_cxi_service --provider cxi --require-provider cxi "
        "--production --serve --domain cxi0 --socket /tmp/ndp.sock"
    )
    child = Child("native-service", 0, "frontier00001", None, command)
    supervisor = AllocationSupervisor(
        tmp_path, [child], heartbeat_s=2, progress_s=3, max_restarts=0,
        launch_backend="node-local-child")
    os.lseek(supervisor.native_token_fd, 0, os.SEEK_END)
    supervisor.start(child)

    argv, kwargs = launched[0]
    assert argv[argv.index("--domain") + 1] == "cxi0"
    assert "--bind-node" not in argv
    assert argv[argv.index("--admission-token-fd") + 1] == str(
        supervisor.native_token_fd)
    assert supervisor.native_token_fd in kwargs["pass_fds"]
    assert os.lseek(supervisor.native_token_fd, 0, os.SEEK_CUR) == 0


def test_manager_rejoin_restarts_same_node_native_service(tmp_path, monkeypatch):
    service = Child("native-service", 2, "node002", None, "service")
    manager = Child("manager", 2, "node002", None, "manager")
    supervisor = AllocationSupervisor(
        tmp_path, [service, manager], heartbeat_s=2, progress_s=3,
        max_restarts=2, startup_s=1, poll_s=.001,
        launch_backend="node-local-child")
    service.process = type(
        "Process", (), {"poll": lambda self: None, "pid": 123})()
    manager.restarts = 0
    socket_path = tmp_path / "ndp.sock"
    monkeypatch.setenv("EMENDER_NDP_SOCKET", str(socket_path))
    monkeypatch.setattr(Path, "is_socket", lambda self: self == socket_path)
    actions = []

    def fake_stop(child, reason):
        actions.append(("stop", child.identity, reason))

    def fake_start(child):
        actions.append(("start", child.identity, child.restarts))
        child.process = type(
            "Process", (), {"poll": lambda self: None, "pid": 124})()

    monkeypatch.setattr(supervisor, "stop_child", fake_stop)
    monkeypatch.setattr(supervisor, "start", fake_start)
    assert supervisor._restart_native_service_for_manager(
        [service], manager, "injected_generation_gate")
    assert service.restarts == 1
    assert actions == [
        ("stop", "node-2-native-service",
         "manager_rejoin:injected_generation_gate"),
        ("start", "node-2-native-service", 1),
    ]
    event = json.loads(
        (tmp_path / "supervision/events.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "native_service_rejoin"
    assert event["manager_identity"] == manager.identity


def test_job_5068873_partial_apply_restarts_atomic_eight_trainer_cohort(
        tmp_path, monkeypatch):
    """Four missing apply receipts must not leak into generation one.

    This is the retained job-5068873 ordering: trainers 0--3 completed the
    generation-zero result while trainers 4--7 never emitted their durable
    recovery receipt, the manager crossed its metadata deadline, and stale
    generation-one metadata was left behind.  Recovery is a single node
    operation: archive the incomplete transaction, replace the service and
    manager, and reconstruct all eight trainers under one fresh incarnation.
    """
    service = Child("native-service", 1, "node001", None, "service")
    manager = Child("manager", 1, "node001", None, "manager")
    trainers = [
        Child("trainer", 1, "node001", rank, f"trainer-{rank}")
        for rank in range(8)
    ]
    supervisor = AllocationSupervisor(
        tmp_path, [service, manager, *trainers],
        heartbeat_s=2, progress_s=3, max_restarts=2, startup_s=1,
        poll_s=.001, launch_backend="node-local-child")
    old_incarnation = supervisor.node_incarnation(1)

    control = tmp_path / "node-1-control"
    control.mkdir()
    monkeypatch.setattr(supervisor, "_node_control_root", lambda _node: control)
    for rank in range(4):
        (control / f"native-applied-00000000-{rank:02d}.json").write_text(
            json.dumps({
                "run_id": "run", "fence_epoch": 7, "generation": 0,
                "rank": rank, "node_incarnation": old_incarnation,
            }))
    (control / "native-generation-00000001.json").write_text(json.dumps({
        "run_id": "run", "fence_epoch": 7, "generation": 1,
        "worker_incarnation": old_incarnation,
    }))
    (control / "native-submit-00000001-00.json").write_text(json.dumps({
        "run_id": "run", "fence_epoch": 7, "generation": 1,
        "worker_incarnation": old_incarnation,
    }))

    _publish_commit_chain(tmp_path, fence=7, generation=1)

    actions = []

    def fake_stop(children, reason, **_kwargs):
        actions.append(("stop", tuple(child.identity for child in children), reason))

    def fake_start(child):
        actions.append((
            "start", child.identity, supervisor.node_incarnation(child.node_rank)))
        child.process = type(
            "Process", (), {"poll": lambda self: None, "pid": 124})()

    monkeypatch.setattr(supervisor, "stop_children", fake_stop)
    monkeypatch.setattr(supervisor, "start", fake_start)
    monkeypatch.setattr(
        supervisor, "_manager_ready_for_trainers", lambda _manager: True)
    socket_path = tmp_path / "ndp.sock"
    monkeypatch.setenv("EMENDER_NDP_SOCKET", str(socket_path))
    monkeypatch.setattr(Path, "is_socket", lambda self: self == socket_path)

    assert supervisor._restart_native_node_cohort(
        [service], manager, trainers,
        "job_5068873_generation_0_apply_deadline")

    new_incarnation = supervisor.node_incarnation(1)
    assert new_incarnation != old_incarnation
    assert [action[1] for action in actions if action[0] == "start"] == [
        service.identity, manager.identity,
        *(trainer.identity for trainer in trainers),
    ]
    assert {
        action[2] for action in actions if action[0] == "start"
    } == {new_incarnation}
    failed = control / "failed-cohorts" / old_incarnation
    assert sorted(path.name for path in failed.glob("native-applied-*.json")) == [
        f"native-applied-00000000-{rank:02d}.json" for rank in range(4)
    ]
    assert (failed / "native-generation-00000001.json").exists()
    assert (failed / "native-submit-00000001-00.json").exists()
    assert not tuple(control.glob("native-applied-*.json"))
    transaction = json.loads(
        (control / "atomic-cohort-recovery.json").read_text())
    assert transaction["failed_incarnation"] == old_incarnation
    assert transaction["node_incarnation"] == new_incarnation
    assert transaction["authoritative_generation"] == 1
    assert transaction["status"] == "reconstructed"
    assert transaction["required_trainers"] == list(range(8))


def test_node_local_supervisor_waits_for_manager_ready_before_cold_trainers(
        tmp_path, monkeypatch):
    manager = Child("manager", 0, "node000", None, "manager")
    trainers = [Child("trainer", 0, "node000", rank, "trainer") for rank in range(2)]
    supervisor = AllocationSupervisor(
        tmp_path, [manager, *trainers], heartbeat_s=2, progress_s=3,
        max_restarts=0, startup_s=1, poll_s=.001,
        launch_backend="node-local-child")
    state = tmp_path / "supervision/node-0-manager.json"
    shared_events = tmp_path / "supervision/events.jsonl"
    starts = []
    trainer_started_before_ready = []

    def manager_ready_is_shared():
        if not shared_events.exists():
            return False
        return any(json.loads(line).get("event") == "manager_ready"
                   for line in shared_events.read_text().splitlines())

    class Process:
        returncode = None
        pid = 123

        def poll(self):
            if len(starts) == 3:
                self.returncode = 0
            return self.returncode

    def fake_start(child):
        starts.append(child.identity)
        child.started_at = time.time()
        child.process = Process()
        if child.role == "manager":
            def publish_ready():
                time.sleep(.03)
                state.parent.mkdir(parents=True, exist_ok=True)
                state.write_text(json.dumps({
                    "identity": child.identity, "heartbeat_time": time.time(),
                    "progress_time": time.time(), "generation": 0,
                    "stage": "collecting"}))
            threading.Thread(target=publish_ready, daemon=True).start()
        else:
            trainer_started_before_ready.append(not manager_ready_is_shared())

    monkeypatch.setattr(supervisor, "start", fake_start)
    assert supervisor.run() == 0
    assert starts == [manager.identity, *(item.identity for item in trainers)]
    assert trainer_started_before_ready == [False, False]
    ready_events = [json.loads(line) for line in shared_events.read_text().splitlines()
                    if json.loads(line).get("event") == "manager_ready"]
    assert [event["identity"] for event in ready_events] == [manager.identity]
    assert ready_events[0]["stage"] == "collecting"


def test_node_local_supervisor_shares_only_bounded_role_stage_transitions(tmp_path):
    child = Child("trainer", 0, "node000", 0, "trainer")
    supervisor = AllocationSupervisor(
        tmp_path, [child], heartbeat_s=2, progress_s=3,
        max_restarts=0, poll_s=.001, launch_backend="node-local-child")
    state = tmp_path / "supervision/node-0-trainer-0.json"
    state.parent.mkdir(parents=True, exist_ok=True)

    for stage in ("runtime_import", "runtime_import", "training", "training",
                  "streaming_delta", "streaming_delta"):
        state.write_text(json.dumps({
            "identity": child.identity, "heartbeat_time": time.time(),
            "progress_time": time.time(), "generation": 0, "stage": stage,
        }))
        supervisor._share_stage_transition(child)

    events = [json.loads(line) for line in
              (tmp_path / "supervision/events.jsonl").read_text().splitlines()]
    assert [event["stage"] for event in events] == [
        "runtime_import", "training", "streaming_delta"]
    assert all(event["event"] == "role_stage" for event in events)


def test_allocation_term_uses_one_shared_grace_for_all_role_groups(tmp_path, monkeypatch):
    children = [Child("trainer", 0, "node000", rank, "trainer") for rank in range(3)]
    supervisor = AllocationSupervisor(
        tmp_path, children, heartbeat_s=2, progress_s=3,
        max_restarts=0, poll_s=.001)
    processes = {}

    class Process:
        returncode = None

        def __init__(self, pid):
            self.pid = pid
            processes[pid] = self

        def poll(self):
            return self.returncode

    for pid, child in enumerate(children, start=100):
        child.process = Process(pid)
        child.started_at = time.time()
    signals = []

    def fake_killpg(pid, requested):
        signals.append((pid, requested))
        if requested == signal.SIGKILL:
            processes[pid].returncode = -signal.SIGKILL

    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor.os.killpg", fake_killpg)
    started = time.monotonic()
    supervisor.stop_children(
        children, "allocation_term_handoff", grace_s=.02, kill_grace_s=.02)
    elapsed = time.monotonic() - started
    assert signals[:3] == [(100, signal.SIGTERM), (101, signal.SIGTERM),
                           (102, signal.SIGTERM)]
    assert signals[3:] == [(100, signal.SIGKILL), (101, signal.SIGKILL),
                           (102, signal.SIGKILL)]
    assert elapsed < .10


def test_node_supervisor_reuses_allocation_gpus_without_step_gres(tmp_path, monkeypatch):
    launched = []

    class Process:
        pid = 123
        returncode = None
        def poll(self): return None

    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor.subprocess.Popen",
        lambda argv, **kwargs: launched.append(argv) or Process())
    child = Child("node-supervisor", 0, "node000", None, "python supervisor.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=2,
                                      progress_s=3, max_restarts=1)
    supervisor.start(child)
    argv = launched[0]
    # The batch allocation already owns all eight node GPUs.  GRES are not
    # shareable between overlapping steps on Frontier, so requesting them a
    # second time leaves the step pending with "Requested nodes are busy".
    assert not any("gpu" in argument for argument in argv)


def test_node_local_supervisors_inherit_allocation_gpus_without_step_gres(monkeypatch):
    monkeypatch.setenv("RUN_DIR", "/tmp/resilient-test")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[000-001]")
    monkeypatch.setenv("RESILIENT_E97_MANAGER_COMMAND", "manager")
    monkeypatch.setenv("RESILIENT_E97_TRAINER_COMMAND", "trainer")
    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor.subprocess.check_output",
        lambda *args, **kwargs: "node000\nnode001\n",
    )
    calls = []
    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor.subprocess.call",
        lambda argv: calls.append(argv) or 0,
    )
    from scripts.frontier import resilient_e97_allocation_supervisor as module

    assert module.main() == 0
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:7] == ["srun", "--overlap", "--no-kill", "--exact", "-N2", "-n2",
                        "--ntasks-per-node=1"]
    # Live Frontier smoke 5021737 proved a repeated step-level GRES request
    # remains pending while a CPU-only two-node step starts immediately.
    assert not any("gpu" in argument for argument in argv)
    assert "-c56" in argv
    assert "-c64" not in argv
    assert argv[-1] == "--node-local"


def test_node_local_entrypoint_uses_slurm_node_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SLURM_NODEID", "1")
    monkeypatch.delenv("RESILIENT_E97_NODE_RANK", raising=False)
    monkeypatch.setenv("RESILIENT_E97_MANAGER_COMMAND", "manager")
    monkeypatch.setenv("RESILIENT_E97_TRAINER_COMMAND", "trainer")
    captured = {}

    def fake_run(self):
        captured["ranks"] = {child.node_rank for child in self.children}
        return 0

    monkeypatch.setattr(AllocationSupervisor, "run", fake_run)
    from scripts.frontier import resilient_e97_allocation_supervisor as module

    assert module._node_local_main() == 0
    assert captured["ranks"] == {1}


def test_allocation_fence_is_acquired_before_roles_and_loser_is_zero_work(
        monkeypatch, tmp_path):
    from scripts.frontier import resilient_e97_allocation_supervisor as module
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    monkeypatch.setenv("RESILIENT_E97_SOURCE_ID", "seed")
    monkeypatch.setenv("RESILIENT_E97_PAYLOAD_ID", "layout")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ID", "job-a")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_INCARNATION", "inc-a")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_FENCE", "10")
    # Admission publishes these keys directly. Register their original values
    # with monkeypatch so later subprocess integration tests cannot inherit a
    # claim after the authority tree has been restored.
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_CLAIM", "")
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", "0")
    first = module._allocation_admission(tmp_path)
    try:
        assert isinstance(first, module.AllocationFenceGuard)
        assert int(os.environ["RESILIENT_E97_FENCE_EPOCH"]) == first.claim.fence
        telemetry = json.loads((tmp_path / "supervision/allocation-fence.json").read_text())
        assert telemetry["stage"] == "admitted_before_model_load"
        assert telemetry["live_authority"] == "native_peer_memory"
        assert telemetry["restart_authority"] == "immutable_commit_receipt_chain"
        assert telemetry["ready_hard_s"] == 180
        assert telemetry["k40_hard_s"] == 420
        assert telemetry["exchange_commit_hard_s"] == 180
        assert telemetry["first_commit_hard_s"] == 720
        monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ID", "job-b")
        monkeypatch.setenv("RESILIENT_E97_ALLOCATION_INCARNATION", "inc-b")
        monkeypatch.setenv("RESILIENT_E97_ALLOCATION_FENCE", "9")
        assert module._allocation_admission(tmp_path) is False
    finally:
        first.close()


def test_stage_specific_deadlines_replace_broad_900_second_silence(tmp_path, monkeypatch):
    child = Child("trainer", 0, "node000", 0, "trainer")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=700,
                                      progress_s=600, max_restarts=0)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision/node-0-trainer-0.json"
    state.write_text(json.dumps({"heartbeat_time": 500, "progress_time": 0,
                                 "generation": 0, "stage": "training"}))
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ADMITTED_AT", "0")
    monkeypatch.setenv("RESILIENT_E97_INITIAL_GENERATION", "0")
    assert supervisor._deadline_reason(child, 421) == "progress_deadline"
    source = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert 'RESILIENT_E97_PROGRESS_DEADLINE_S", "900"' not in source


def test_rendered_debug_production_contract_has_only_approved_deltas():
    debug = json.loads((ROOT / "configs/frontier/e97_resilient_debug_rendered.json").read_text())
    production = json.loads((ROOT / "configs/frontier/e97_resilient_production_rendered.json").read_text())
    result = compare(debug, production)
    assert result["ok"], result
    assert set(result["allowlisted_diff"]) == {
        "qos", "walltime", "nodes", "failure_injection"}
    assert result["injection_disabled_in_production"]


def test_node_step_injection_uses_manager_generation_and_distinct_control(tmp_path, monkeypatch):
    child = Child("node-supervisor", 1, "node001", None, "python supervisor.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=60, max_restarts=1)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-1-manager.json"
    state.write_text(json.dumps({"heartbeat_time": 10, "progress_time": 10,
                                 "generation": 2, "stage": "published"}))
    monkeypatch.setenv("RESILIENT_E97_INJECT_MANAGER", "1:-1:2")
    assert supervisor._deadline_reason(child, 10) is None
    monkeypatch.setenv("RESILIENT_E97_INJECT_NODE_STEP", "1:-1:2")
    assert supervisor._deadline_reason(child, 10) == "injected_generation_gate"
    assert supervisor._deadline_reason(child, 10) is None


def test_finalized_generation_injection_waits_for_durable_role_stage(tmp_path, monkeypatch):
    child = Child("trainer", 0, "node000", 3, "trainer")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=60, max_restarts=1)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-0-trainer-3.json"
    monkeypatch.setenv("RESILIENT_E97_INJECT_TRAINER", "0:3:1")
    state.write_text(json.dumps({"heartbeat_time": 10, "progress_time": 10,
                                 "generation": 1, "stage": "checkpoint_commit"}))
    assert supervisor._deadline_reason(child, 10) is None
    state.write_text(json.dumps({"heartbeat_time": 10, "progress_time": 10,
                                 "generation": 1, "stage": "applied"}))
    assert supervisor._deadline_reason(child, 10) == "injected_generation_gate"


def test_native_service_loss_is_gated_by_manager_owner_stage(tmp_path, monkeypatch):
    child = Child("native-service", 1, "node001", None, "service")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=60, max_restarts=0)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-1-manager.json"
    state.write_text(json.dumps({"heartbeat_time": 10, "progress_time": 10,
                                 "generation": 1, "stage": "training_wait"}))
    monkeypatch.setenv("RESILIENT_E97_INJECT_NATIVE_SERVICE",
                       "1:-1:1:owner_transport")
    assert supervisor._deadline_reason(child, 10) is None
    state.write_text(json.dumps({"heartbeat_time": 10, "progress_time": 10,
                                 "generation": 1, "stage": "owner_transport"}))
    assert supervisor._deadline_reason(child, 10) == "injected_native_service_stage"


def test_native_manager_rejoin_syncs_authoritative_latest_generation(tmp_path):
    from types import SimpleNamespace
    from scripts.frontier import resilient_e97_role as role

    handoff = tmp_path / "handoff"
    handoff.mkdir()
    manifest = handoff / "generation-00000002-fence-00000001.json"
    manifest.write_text(json.dumps({
        "finalized": True, "run_id": "run", "payload_id": "payload",
        "source_id": "source", "code_id": "code", "generation": 2,
        "fence": {"coordinator_epoch": 1},
    }, sort_keys=True))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (handoff / "latest.json").write_text(json.dumps({
        "generation": 2, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": digest,
    }, sort_keys=True))
    args = SimpleNamespace(initial_generation=0, generations=3, run_id="run",
                           payload_id="payload", source_id="source", code_id="code",
                           coordinator_epoch=1)
    generation, evidence = role._native_manager_resume_point(tmp_path, args, None)
    assert generation == 2
    assert evidence == {
        "status": "synchronized", "generation": 2, "fence": 1,
        "source_fence": 1, "manifest": str(manifest.resolve()),
        "manifest_sha256": digest,
    }

    (handoff / "latest.json").write_text(json.dumps({
        "generation": 2, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": "00" * 32,
    }))
    with pytest.raises(ValueError, match="manifest checksum"):
        role._native_manager_resume_point(tmp_path, args, None)


def test_fresh_allocation_manager_syncs_older_authoritative_handoff(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.frontier import resilient_e97_role as role

    old_runtime = {
        "schema": "emender-native-e97-runtime-digests-v1",
        "source_commit": "old-code", "build_manifest_sha256": "old-manifest",
        "build_bundle_sha256": "bundle", "config_sha256": "config",
        "provider": "cxi", "provider_sha256": "provider",
        "artifacts": {"service_binary": "service", "local_library": "local",
                      "transport_library": "transport",
                      "synthetic_gate_binary": "gate"},
    }
    current_runtime = {
        **old_runtime, "source_commit": "new-code",
        "build_manifest_sha256": "new-manifest",
    }
    authority, old, receipt = _publish_commit_chain(
        tmp_path,
        fence=1,
        generation=3,
        code_id="old-code",
        native_runtime=old_runtime,
    )
    apply_receipts = []
    for node in range(2):
        apply_receipts.append(authority.record_node_apply(
            old,
            receipt,
            node_id=f"node-{node}",
            node_incarnation=f"node-{node}-old",
            trainer_receipts=[
                (
                    rank,
                    f"node-{node}-trainer-{rank}-old",
                    hashlib.sha256(
                        f"node-{node}-recovery-{rank}".encode()).hexdigest(),
                )
                for rank in range(8)
            ],
        ))
    manifest = receipt.manifest_path
    digest = receipt.manifest_sha256
    current = authority.claim(
        run_id="run",
        allocation_id="job-fresh",
        incarnation="fresh",
        fence=2,
        protocol_id="pool-v1",
        config_id="config",
    )
    assert current.fence == 2
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", str(current.fence))
    args = SimpleNamespace(
        initial_generation=3, generations=2, run_id="run",
        payload_id="payload", source_id="source", code_id="new-code",
        coordinator_epoch=current.fence)

    generation, evidence = role._native_manager_resume_point(
        tmp_path, args, (authority, current), native_runtime=current_runtime)

    assert generation == 3
    assert evidence == {
        "status": "synchronized", "generation": 3, "fence": 2,
        "source_fence": 1, "source_code_id": "old-code",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": digest,
        "commit_receipt_digest": receipt.receipt_digest,
        "accepted_tokens": receipt.accepted_tokens,
        "result_root": receipt.result_root,
        "apply_receipts": [
            {"worker_id": item.node_id,
             "receipt_digest": item.receipt_digest}
            for item in apply_receipts
        ],
    }


def test_native_restart_runtime_compatibility_rejects_substantive_digest_change():
    from scripts.frontier import resilient_e97_role as role

    recorded = {
        "schema": "emender-native-e97-runtime-digests-v1",
        "source_commit": "old", "build_manifest_sha256": "old-manifest",
        "build_bundle_sha256": "bundle", "config_sha256": "config",
        "provider": "cxi", "provider_sha256": "provider",
        "artifacts": {"service_binary": "service", "local_library": "local",
                      "transport_library": "transport",
                      "synthetic_gate_binary": "gate"},
    }
    current = {
        **recorded, "source_commit": "new",
        "build_manifest_sha256": "new-manifest",
    }

    assert role._native_runtime_resume_compatible(recorded, current)
    assert not role._native_runtime_resume_compatible(
        recorded, {**current, "build_bundle_sha256": "changed"})
    assert not role._native_runtime_resume_compatible(
        recorded, {**current, "artifacts": {**current["artifacts"],
                                              "service_binary": "changed"}})


def test_delayed_ready_injection_is_exact_and_bounded(monkeypatch):
    from scripts.frontier import resilient_e97_role as role

    monkeypatch.setenv("RESILIENT_E97_DELAY_READY", "1:2:7.5")
    assert role._native_ready_delay(node=1, generation=2) == 7.5
    assert role._native_ready_delay(node=0, generation=2) == 0.0
    monkeypatch.setenv("RESILIENT_E97_DELAY_READY", "1:2:181")
    with pytest.raises(ValueError, match="bounded by 180"):
        role._native_ready_delay(node=1, generation=2)


def test_rejoining_manager_delays_ready_before_advertising(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.frontier import resilient_e97_role as role

    monkeypatch.setenv("RESILIENT_E97_DELAY_READY", "1:2:0.02")
    args = SimpleNamespace(run_id="run-a", coordinator_epoch=4)
    term = {"value": False}
    started = time.monotonic()
    assert role._wait_native_ready_delay(
        tmp_path, args, node=1, generation=2, incarnation="new-inc",
        term_requested=term)
    assert time.monotonic() - started >= 0.015
    marker = json.loads(
        (tmp_path / "native-delayed-ready-00000002-new-inc.json").read_text())
    assert marker["status"] == "completed"
    assert marker["incarnation"] == "new-inc"

    source = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    manager = source[source.index("def _native_manager(args)"):]
    initial_ready = manager.index(
        "pool_client.ready(session.owner_endpoint, start_generation")
    assert manager.rfind("_wait_native_ready_delay(", 0, initial_ready) >= 0


def test_child_without_first_heartbeat_hits_startup_deadline(tmp_path):
    child = Child("manager", 0, "node000", None, "python manager.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=900, max_restarts=1,
                                      startup_s=30)
    child.process = type("Process", (), {"poll": lambda self: None})()
    child.started_at = 100
    assert supervisor._deadline_reason(child, 129) is None
    assert supervisor._deadline_reason(child, 131) == "startup_deadline"


def test_manager_liveness_does_not_disguise_stalled_generation_progress(tmp_path, monkeypatch):
    child = Child("manager", 0, "node000", None, "python manager.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=10,
                                      progress_s=20, max_restarts=1)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-0-manager.json"
    state.write_text(json.dumps({"heartbeat_time": 100, "progress_time": 100,
                                 "generation": 0}))
    liveness = tmp_path / "supervision" / "node-0-manager.liveness.json"
    liveness.write_text(json.dumps({"identity": child.identity,
                                    "heartbeat_time": 125}))
    monkeypatch.delenv("RESILIENT_E97_BULK_ROOT", raising=False)
    monkeypatch.delenv("RESILIENT_E97_RUN_ID", raising=False)
    assert supervisor._deadline_reason(child, 126) == "progress_deadline"


def test_pipelined_trainer_apply_receipts_refresh_manager_progress(tmp_path, monkeypatch):
    child = Child("manager", 0, "node000", None, "python manager.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=20, max_restarts=1)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-0-manager.json"
    state.write_text(json.dumps({"heartbeat_time": 125, "progress_time": 100,
                                 "generation": 2, "stage": "redistribution"}))
    control = tmp_path / "control"
    control.mkdir()
    receipt = control / "native-applied-00000002-03.json"
    receipt.write_text(json.dumps({"run_id": "run", "generation": 2, "rank": 3}))
    os.utime(receipt, (124, 124))
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    monkeypatch.delenv("RESILIENT_E97_BULK_ROOT", raising=False)

    assert supervisor._deadline_reason(child, 126) is None
    assert supervisor._deadline_reason(child, 145) == "progress_deadline"


def test_live_diagnostic_uses_peer_commit_and_restart_checks_immutable_checkpoint(
        tmp_path, monkeypatch):
    child = Child("manager", 0, "node000", None, "python manager.py")
    supervisor = AllocationSupervisor(tmp_path, [child], heartbeat_s=60,
                                      progress_s=60, max_restarts=1)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-0-manager.json"
    state.write_text(json.dumps({"heartbeat_time": 800, "progress_time": 800,
                                 "generation": 0, "stage": "training_wait"}))
    _authority, _claim, receipt = _publish_commit_chain(
        tmp_path, fence=1, generation=1)
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    monkeypatch.setenv("RESILIENT_E97_INITIAL_GENERATION", "0")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ADMITTED_AT", "0")
    monkeypatch.delenv("RESILIENT_E97_BULK_ROOT", raising=False)

    assert supervisor._deadline_reason(
        child, 800) == "first_atomic_generation_deadline"

    # The 720-second authority is specifically the first committed `latest`,
    # not the later all-eight live-model apply.  Once the manager has verified
    # the immutable receipt through peer control, its authenticated progress
    # state must satisfy that deadline while bounded result preparation and
    # atomic apply continue.
    state.write_text(json.dumps({
        "heartbeat_time": 800,
        "progress_time": 800,
        "generation": 0,
        "authoritative_generation": 1,
        "stage": "result_preparation",
        "commit_receipt_digest": receipt.receipt_digest,
    }))
    assert supervisor._deadline_reason(child, 800) is None

    # The later node-apply acknowledgement remains valid evidence too.  The
    # hot supervisor diagnostic consumes manager-published peer evidence and
    # performs no shared-manifest scan in either phase.
    state.write_text(json.dumps({
        "heartbeat_time": 800,
        "progress_time": 800,
        "generation": 1,
        "authoritative_generation": 1,
        "stage": "published_node_applied",
        "commit_receipt_digest": receipt.receipt_digest,
        "node_apply_receipt_digest": "ab" * 32,
    }))
    monkeypatch.setattr(
        "scripts.frontier.resilient_e97_allocation_supervisor."
        "ManifestPeerAuthority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("live diagnostic read immutable filesystem authority")),
    )
    assert supervisor._deadline_reason(child, 800) is None

    # An actual restart still verifies immutable authority and fails closed on
    # corruption; it is deliberately separate from the live diagnostic.
    monkeypatch.undo()
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    receipt.manifest_path.write_text('{"corrupt":true}\n')
    assert supervisor._durable_generation() is None


def test_first_commit_deadline_is_owned_by_manager_not_individual_trainer(
        tmp_path, monkeypatch):
    child = Child("trainer", 0, "node000", 0, "trainer")
    supervisor = AllocationSupervisor(
        tmp_path, [child], heartbeat_s=60, progress_s=60, max_restarts=0)
    child.process = type("Process", (), {"poll": lambda self: None})()
    state = tmp_path / "supervision" / "node-0-trainer-0.json"
    state.write_text(json.dumps({
        "heartbeat_time": 800,
        "progress_time": 800,
        "generation": 0,
        "stage": "training",
    }))
    monkeypatch.setenv("RESILIENT_E97_INITIAL_GENERATION", "0")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ADMITTED_AT", "0")
    monkeypatch.delenv("RESILIENT_E97_BULK_ROOT", raising=False)
    monkeypatch.delenv("RESILIENT_E97_RUN_ID", raising=False)

    # Trainer heartbeat/K40/stage budgets remain independently fail-closed.
    # Allocation-wide durable-commit authority is manager-owned, so a healthy
    # trainer must not be killed merely because it does not duplicate the
    # manager's immutable receipt in its hot progress document.
    assert supervisor._deadline_reason(child, 800) is None


def test_durable_generation_diagnostic_never_opens_sqlite(
        tmp_path, monkeypatch):
    """The job-5072235 diagnostic path uses immutable receipts only.

    Poisoning sqlite3.connect is covered by the dedicated compute-closure
    tripwire; this regression retains the observed generation-eight shape and
    proves the supervisor resolves it without a database symbol or path.
    """
    from scripts.frontier import resilient_e97_allocation_supervisor as module

    supervisor = AllocationSupervisor(
        tmp_path, [], heartbeat_s=60, progress_s=60, max_restarts=0)
    authority, claim, _receipt = _publish_commit_chain(
        tmp_path, fence=5072235, generation=8)
    monkeypatch.setenv("RESILIENT_E97_RUN_ID", "run")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_CLAIM", claim.encode())
    assert authority.current_commit(claim).generation == 8
    assert supervisor._durable_generation() == 8
    assert "sqlite" not in module.__dict__


def test_all_real_roles_publish_import_liveness_without_generation_progress():
    text = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    assert 'sys.argv[1] not in {"manager", "trainer"}' in text
    assert 'f"node-{node_rank}-trainer-{local_rank}"' in text
    assert '"stage": "runtime_import"' in text
    assert '"generation": 0, "step": 0' in text
    assert text.count("if _IMPORT_HEARTBEAT is not None:") == 2


def test_real_trainer_keeps_liveness_during_checkpoint_load_and_training():
    text = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = text[text.index("def trainer(args)"):]
    assert trainer.index("_liveness_heartbeat(bulk, identity)") < trainer.index("_load_real(args)")
    assert 'f"{identity}.liveness.json"' in text


def test_generation_deadline_includes_local_training_and_aggregate_wait():
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    deadline = "generation_deadline = time.monotonic() + min(args.deadline_s, 420.0)"
    training_guard = "if time.monotonic() >= generation_deadline:"
    aggregate_wait = "fence, deadline=exchange_deadline,"

    assert role.count(deadline) == 1
    assert role.index(deadline) < role.index("report = _run_real_worker(")
    assert role.index(training_guard) < role.index("progress_callback=training_progress")
    assert role.index("progress_callback=training_progress") < role.index(aggregate_wait)
    assert "measured baseline 212-215s" in role
    assert "exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)" in role
    assert "aggregation_deadline_s=min(args.deadline_s, 420.0)" in role
    assert "generation_started, pool_config.slo.training_hard_s" in role


def test_local_and_owner_transport_use_separate_bounded_frontier_chunks():
    from ndm.resilient_e97_reducer import TensorLayout

    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()

    # Job 5028225 proved 1 MiB local records too slow. Job 5028347 proved that
    # reusing 1 MiB for the full owner plane creates about 40,000 short-lived
    # request/response connections per E97 generation. Both planes retain hard,
    # independent 64 MiB bounds under the shared 64 GiB byte ledger.
    assert "--local-spool-chunk-bytes" in role
    assert "RESILIENT_E97_LOCAL_SPOOL_CHUNK_BYTES:-67108864" in launcher
    assert "RESILIENT_E97_MAX_SPOOL_BYTES:-68719476736" in launcher
    assert 'value.add_argument("--max-spool-bytes", type=int, default=64 << 30)' in role
    assert "--bulk-chunk-bytes ${RESILIENT_E97_BULK_CHUNK_BYTES:-67108864}" in launcher
    assert 'value.add_argument("--bulk-chunk-bytes", type=int, default=64 << 20)' in role
    assert "chunk_elements = max(1, args.local_spool_chunk_bytes // 8)" in role
    assert '"local_delta_spool"' in role
    assert "local_spool_chunk_bytes=args.local_spool_chunk_bytes" in role
    assert '"local_reduce_wait": K40_HARD_S' in supervisor

    # The measured 5,506,770,496-byte f32 trainer delta becomes this many
    # float64 owner bytes. The larger frame remains bounded and cuts the live
    # request count by 64x without materializing a Python element list.
    layout = TensorLayout.from_flat_stream(
        5_506_770_496 // 4, max_chunk_bytes=64 << 20)
    assert layout.shard_count == 165
    assert layout.chunk_elements * 8 == 64 << 20


def test_native_manager_advances_progress_after_owner_transport():
    """Separately bounded commit/apply waits must own fresh stage clocks."""
    source = (
        ROOT / "scripts/frontier/resilient_e97_role.py"
    ).read_text(encoding="utf-8")
    manager = source[
        source.index("def _native_manager(args) -> int:"):
        source.index("def manager(args) -> int:")
    ]
    owner = manager.index('stage="owner_transport"')
    reduced = manager.index(
        "final_operation, final_result, freeze = "
        "_native_sharded_owner_reduce(",
        owner,
    )
    checkpoint_stage = manager.index(
        'stage="checkpoint_commit"', reduced)
    proposal_wait = manager.index(
        'control / f"trainer-proposal-{generation:08d}.json"',
        reduced,
    )
    apply_stage = manager.index('stage="peer_apply"', proposal_wait)
    apply_receipt_wait = manager.index(
        'control / f"native-applied-{generation:08d}-{rank:02d}.json"',
        proposal_wait,
    )

    assert owner < reduced < checkpoint_stage < proposal_wait
    assert proposal_wait < apply_stage < apply_receipt_wait


def test_native_all_eight_apply_releases_after_bounded_concurrent_preparation():
    """One reader prepares at a time; eight K boundaries precede release."""
    role = (
        ROOT / "scripts/frontier/resilient_e97_role.py"
    ).read_text(encoding="utf-8")
    manager = role[
        role.index("def _native_manager(args) -> int:"):
        role.index("def manager(args) -> int:")
    ]
    committed_evidence = manager.index("committed_evidence =")
    preparation_stage = manager.index('stage="result_preparation"')
    rendezvous_call = manager.index(
        "_coordinate_native_safe_boundary(",
        preparation_stage,
    )
    apply_stage = manager.index('stage="peer_apply"', rendezvous_call)
    receipt_wait = manager.index(
        'control / f"native-applied-{generation:08d}-{rank:02d}.json"',
        apply_stage,
    )
    assert (
        committed_evidence < preparation_stage < rendezvous_call
        < apply_stage < receipt_wait
    )
    assert (
        'stage="result_preparation", **committed_evidence'
        in manager[committed_evidence:rendezvous_call]
    )
    assert (
        'stage="peer_apply", **committed_evidence'
        in manager[rendezvous_call:receipt_wait]
    )
    coordinator = role[
        role.index("def _coordinate_native_safe_boundary("):
        role.index("def _liveness_heartbeat(")
    ]
    prepared_wait = coordinator.index("native-candidate-prepared-")
    rendezvous_open = coordinator.index("native-boundary-rendezvous-")
    boundary_wait = coordinator.index("native-boundary-ready-")
    release = coordinator.index("native-apply-release-")
    assert prepared_wait < rendezvous_open < boundary_wait < release
    assert "transaction.release_apply(" in coordinator[boundary_wait:release]
    assert "abort_before_release(" in coordinator

    trainer = role[role.index("def trainer(args) -> int:"):]
    result_visible = trainer.index(
        "manifest, aggregate = native_context.__enter__()")
    result_apply = trainer.index(
        "apply_delta_with_correction_ledger(", result_visible)
    assert result_visible < result_apply
    assert "_wait_for_native_apply_lane(" not in trainer[
        result_visible:result_apply]
    result_materialized = trainer.index(
        'control / f"native-result-applied-{generation:08d}-{rank:02d}.json"')
    reload_verified = trainer.index(
        "_reload_verified_async_v2_latest(", result_materialized)
    prepared = trainer.index(
        "native-candidate-prepared-",
        reload_verified,
    )
    rendezvous = trainer.index(
        'marker_name="native-boundary-rendezvous"', prepared)
    boundary_stop = trainer.index(
        "async_training_lane.finish_at_boundary(", rendezvous)
    boundary_ready = trainer.index("native-boundary-ready-", boundary_stop)
    wait_release = trainer.index(
        'marker_name="native-apply-release"', boundary_ready)
    live_swap = trainer.index("safe_apply_started = time.monotonic()", wait_release)
    applied = trainer.index(
        'control / f"native-applied-{generation:08d}-{rank:02d}.json"',
        live_swap,
    )
    assert (
        result_materialized < reload_verified < prepared < rendezvous
        < boundary_stop < boundary_ready < wait_release < live_swap < applied
    )

    supervisor = (
        ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    ).read_text(encoding="utf-8")
    assert '"result_preparation": RESULT_PREPARATION_HARD_S' in supervisor
    assert (
        '"boundary_rendezvous": BOUNDARY_RENDEZVOUS_HARD_S'
        in supervisor
    )
    assert '"peer_apply": ALL_EIGHT_APPLY_HARD_S' in supervisor


def test_trainer_exchange_window_waits_for_manager_local_reduce(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    bulk = tmp_path / "bulk"
    supervision = bulk / "supervision"
    supervision.mkdir(parents=True)
    observed = {}

    def wait_for_window():
        observed["stage"] = role._wait_for_manager_exchange_window(
            bulk, node=1, generation=3, deadline=time.monotonic() + 2)

    waiter = threading.Thread(target=wait_for_window)
    waiter.start()
    time.sleep(.05)
    assert waiter.is_alive(), "trainer must not spend its apply window during local reduction"
    (supervision / "node-1-manager.json").write_text(json.dumps({
        "generation": 3, "stage": "owner_transport",
    }))
    waiter.join(2)

    assert not waiter.is_alive()
    assert observed == {"stage": "owner_transport"}
    (supervision / "node-1-manager.json").write_text(json.dumps({
        "generation": 4, "stage": "collecting",
    }))
    with pytest.raises(TimeoutError, match="local reduction deadline expired"):
        role._wait_for_manager_exchange_window(
            bulk, node=1, generation=4, deadline=time.monotonic() + .03)
    source = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = source[source.index("def trainer(args)"):]
    assert trainer.index("_wait_for_manager_exchange_window(") < trainer.index(
        "exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)")


def test_checkpoint_leader_proposes_before_disposable_local_recovery():
    trainer = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = trainer[trainer.index("def trainer(args)"):]

    proposal = trainer.index(
        'atomic_json(bulk / "control" / "trainer-proposal.json"')
    local_recovery = trainer.index("recovery_checkpoint =")
    assert proposal < local_recovery
    assert "completed == target_generation" in trainer[:local_recovery]


def test_multinode_manager_publishes_only_final_global_aggregate_and_leader_applies_first():
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    manager = role[role.index("def manager(args)"):role.index("def _load_real(args)")]
    trainer = role[role.index("def trainer(args)"):]

    # The multi-node branch keeps exact local collection private to the
    # manager; only the single-node fixture uses SplitManagerLoop.generation.
    assert "loop.manager.collect(" in manager
    assert manager.count("loop.generation(fence)") == 1
    assert manager.index("loop.generation(fence)") < manager.index(
        "members, local_weight, local_shards = loop.manager.collect(")
    assert "storage_dtype=torch.float32" in manager

    priority = trainer.index("_wait_for_leader_apply_release(")
    streamed_apply = trainer.index("spool.stream_aggregate(")
    proposal = trainer.index(
        'atomic_json(bulk / "control" / "trainer-proposal.json"')
    release = trainer.index('f"leader-apply-release-{generation:08d}.json"')
    assert priority < streamed_apply < proposal < release
    assert "if args.node_count > 1 and node == 0 and rank != 0:" in trainer
    assert "in_place=True" in trainer[streamed_apply:proposal]
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert '"leader_apply_wait": RESULT_PREPARATION_HARD_S' in supervisor


def test_checkpoint_leader_wait_uses_enclosing_result_preparation_deadline():
    """The peer wait includes readiness, leader apply, and checkpoint.

    Each component retains its stricter inner deadline. Charging the composite
    path to the 180-second result-readiness budget makes a valid 153-second
    result plus a 43-second materialization fail before the release exists.
    """
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = role[role.index("def trainer(args)"):]
    wait = trainer.index("_wait_for_leader_apply_release(")
    result = trainer.index("native_plane.result_shards(", wait)
    assert "leader_release_deadline = (" in trainer[:wait]
    assert (
        "time.monotonic() + min(args.deadline_s, 420.0)"
        in trainer[:wait]
    )
    assert "deadline=leader_release_deadline" in trainer[wait:result]
    assert (
        "exchange_deadline = time.monotonic() + min(args.deadline_s, 180.0)"
        in trainer[wait:result]
    )

    supervisor = (
        ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    ).read_text()
    assert "RESULT_PREPARATION_HARD_S = 420.0" in supervisor
    assert '"leader_apply_wait": RESULT_PREPARATION_HARD_S' in supervisor


def test_all_peers_get_fresh_supervised_apply_window_after_aggregate_visibility():
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = role[role.index("def trainer(args)"):]
    aggregate_visible = trainer.index("manifest, aggregate = spool.stream_aggregate(")
    preparation = trainer.index('stage="result_preparation"', aggregate_visible)
    apply = trainer.index("state = apply_delta(", aggregate_visible)
    release = trainer.index(
        'marker_name="native-apply-release"', apply)
    foreground = trainer.index('stage="peer_apply"', release)
    live_swap = trainer.index("safe_apply_started = time.monotonic()", foreground)
    assert (
        aggregate_visible < preparation < apply < release < foreground
        < live_swap
    )
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert '"result_preparation": RESULT_PREPARATION_HARD_S' in supervisor
    assert (
        '"boundary_rendezvous": BOUNDARY_RENDEZVOUS_HARD_S'
        in supervisor
    )
    assert '"peer_apply": ALL_EIGHT_APPLY_HARD_S' in supervisor


def test_terminal_generation_does_not_rejoin_draining_pool():
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    manager = role[role.index("def _native_manager(args)"):
                   role.index("def manager(args)")]
    assert "target_generation = args.initial_generation + args.generations" in manager
    assert "has_next_generation = generation + 1 < target_generation" in manager
    assert "if pool_client is not None and has_next_generation:" in manager
    assert manager.index('stage="published_node_applied"') < manager.index(
        "has_next_generation =",
        manager.index('stage="published_node_applied"'))
    assert "terminal_published = False" in manager
    assert "terminal_published = True" in manager
    assert "if pool_client is not None and not terminal_published:" in manager


def test_node_local_exit_retains_only_small_control_evidence(tmp_path):
    from scripts.frontier import resilient_e97_allocation_supervisor as supervisor

    run = tmp_path / "run"
    bulk = tmp_path / "bulk" / "run-a" / "node-1"
    (bulk / "supervision").mkdir(parents=True)
    (bulk / "telemetry").mkdir()
    (bulk / "control/recovery").mkdir(parents=True)
    (bulk / "mailbox").mkdir()
    (bulk / "supervision/trainer.json").write_text('{"stage":"streaming_delta"}\n')
    (bulk / "telemetry/trainer.jsonl").write_text('{"phase":"K40"}\n')
    (bulk / "control/recovery/trainer.json").write_text('{"generation":0}\n')
    (bulk / "mailbox/contribution.data").write_bytes(b"tensor bytes")
    (bulk / "control/recovery/generation.pt").write_bytes(b"checkpoint bytes")

    retained = supervisor._retain_node_evidence(
        run, bulk_root=tmp_path / "bulk", run_id="run-a", node_rank=1)

    assert json.loads((retained / "snapshot.json").read_text())["node_rank"] == 1
    assert (retained / "supervision/trainer.json").exists()
    assert (retained / "telemetry/trainer.jsonl").exists()
    assert (retained / "control/recovery/trainer.json").exists()
    assert not (retained / "mailbox").exists()
    assert not (retained / "control/recovery/generation.pt").exists()
    assert all(path.suffix in {".json", ".jsonl"}
               for path in retained.rglob("*") if path.is_file())


def test_clean_overlap_validates_post_supervisor_retained_node_telemetry():
    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    supervisor = launcher.index("resilient_e97_allocation_supervisor.py")
    validator = launcher.index("validate_pipelined_e97_performance.py")
    retained_root = launcher.index(
        '--telemetry-root "$RUN_DIR/retained-evidence"', validator)
    assert supervisor < validator < retained_root
    assert '--telemetry-root "$RESILIENT_E97_BULK_ROOT/telemetry"' not in launcher


def test_real_trainer_streams_model_delta_without_full_cpu_materialization():
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    worker = (ROOT / "ndm/async_diloco_real.py").read_text()

    assert "delta_consumer=publish_trained_delta" in role
    assert "spool.publish(fence, rank, shards(), weight=tokens" in role
    assert "worker_chunk.sub(base_flat[offset:end])" in role
    assert "if delta_consumer is None:" in worker
    assert "delta_consumer(base_state, model, tokens)" in worker


def test_pool_wiring_preserves_exact_e97_trainer_model_data_optimizer_and_k40():
    """Freeze the known-good trainer call; only its post-training consumer may change."""
    import hashlib
    launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    config = ROOT / "configs/frontier/e97_resilient_split_role_flat.json"
    assert hashlib.sha256(config.read_bytes()).hexdigest() \
        == "afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c"
    trainer_command = next(line for line in launcher.splitlines()
                           if line.startswith("export RESILIENT_E97_TRAINER_COMMAND="))
    assert "--local-steps 40" in launcher
    assert "--seed $RESILIENT_E97_SEED" in trainer_command
    assert "--train-args-json $RESILIENT_E97_TRAIN_ARGS_JSON" in trainer_command
    assert "--data $RESILIENT_E97_DATA" in trainer_command
    assert "--migration-policy initialize-from-approved-config" in trainer_command
    assert 'overrides.update({"data": args.data, "optimizer": "schedulefree"})' in role
    call = role[role.index("report = _run_real_worker("):role.index("if report.update is None:")]
    assert "train_args=train_args" in call
    assert "local_steps, rank" in call
    assert "optimizer_state_dict=optimizer_state" in call
    assert "consume_optimizer_state=True" in call
    assert "synthetic_token_stream=False" in call
    assert "delta_consumer=publish_trained_delta" in call
    assert "RESILIENT_E97_GENERATION_DEADLINE_S:-900" not in launcher


def test_frontier_native_trainer_has_one_async_v21_production_authority():
    """The v1 mailbox scheduler must not compete with the real v2 lane."""
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = role[role.index("def trainer(args) -> int:"):]

    assert "PersistentAsyncTrainingLane(" in trainer
    assert "async_training_lane.start(" in trainer
    assert "async_training_lane.finish_at_boundary(" in trainer
    assert '"async_v21_result_readiness"' in trainer
    assert "NativeGenerationPipeline(" not in trainer
    assert "pipeline." not in trainer


def test_production_k_next_starts_after_snapshot_before_publication_and_result():
    """Guard the immutable-snapshot foreground edge in the rendered trainer.

    The sole mutable lane resumes from the coherent endpoint before queue
    admission/OWNED, result readiness, aggregate materialization, and
    checkpoint write.  All work after that boundary consumes only the retained
    immutable snapshot until the verified result is applied at a later K edge.
    """
    role = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = role[role.index("def trainer(args) -> int:"):]
    snapshot = trainer.index(
        "retained_endpoint = persistent_worker.snapshot()")
    next_k = trainer.index("async_training_lane.start(", snapshot)
    publish = trainer.index("marker = native_plane.publish_state_delta(")
    owned = trainer.index("v2_owned_seconds_max = max(", publish)
    result = trainer.index("native_plane.result_shards(", owned)
    apply = trainer.index("state = apply_delta(", result)
    checkpoint = trainer.index("torch.save(", apply)
    verified = trainer.index(
        "_reload_verified_async_v2_latest(", checkpoint)
    boundary = trainer.index(
        "async_training_lane.finish_at_boundary(", verified)

    assert (snapshot < next_k < publish < owned < result < apply
            < checkpoint < verified < boundary)
    between = trainer[next_k:result]
    assert "fabric_receipt_waited=False" in between
    assert "publish_state_delta(" in between
    assert "native_plane.result_shards(" not in between
    assert "state = apply_delta(" not in between
    assert "torch.save(" not in between
    assert "publish_model_delta(" not in between
    assert "PersistentRealWorkerSession(" in trainer[:next_k]
    assert trainer.count("PersistentRealWorkerSession(") == 1
    assert "persistent_worker.bootstrap_counts" in trainer
    lane = (
        ROOT / "ndm/async_diloco_real.py").read_text().split(
            "class PersistentAsyncTrainingLane", 1)[1].split(
                "def _run_real_worker(", 1)[0]
    assert "self.session.run_window(" in lane


def test_frontier_default_keeps_supervision_state_node_local():
    text = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert 'os.environ.get("RESILIENT_E97_LAUNCH_MODE", "node-local")' in text
    assert 'launch_backend="node-local-child"' in text
    assert 'sys.executable, __file__, "--node-local"' in text
    assert 'f"-N{node_count}", f"-n{node_count}"' in text
def test_exact_renderer_binds_production_delayed_scheduler_marker(tmp_path):
    """The reviewed scheduler must be constructed by the rendered role path."""
    from scripts.frontier import render_resilient_e97_exact_2n_acceptance as renderer

    manifest = tmp_path / "native-artifacts.json"
    gate = tmp_path / "gate.json"
    # build_plan validates these through native_identity; retain the exact
    # launcher assertion here without manufacturing a second launch path.
    source = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    batch = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert renderer.build_plan.__module__.endswith(
        "render_resilient_e97_exact_2n_acceptance")
    assert 'launcher": "scripts/frontier/resilient_e97_true_2n.sbatch"' in (
        ROOT / "scripts/frontier/render_resilient_e97_exact_2n_acceptance.py").read_text()
    assert 'ROLE="$REPO/scripts/frontier/resilient_e97_role.py"' in batch
    assert "AsyncV21DescriptorService(" in source
    assert "AsyncV21WorkerLane(" in source
    assert '"schema": "emender-production-async-decoupled-v2.1"' in source
    assert "LiveNativeGenerationScheduler(" not in source
    assert '"ndm.async_diloco_real.PersistentAsyncTrainingLane"' in source
    assert '"ndm.async_diloco_v2.AsyncV21WorkerLane"' in source


def test_exact_rendered_production_role_starts_g1_without_g0_quorum_permission():
    """Regression: production role, not an abstraction-only fixture, overlaps."""
    from scripts.frontier import render_resilient_e97_exact_2n_acceptance as renderer
    from scripts.frontier import resilient_e97_role as role

    batch = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert renderer.build_plan.__module__.endswith(
        "render_resilient_e97_exact_2n_acceptance")
    assert 'ROLE="$REPO/scripts/frontier/resilient_e97_role.py"' in batch

    background_started = threading.Event()
    background_release = threading.Event()
    scheduler, events = role.production_overlap_probe(
        background_release=background_release,
        background_started=background_started)
    try:
        k40 = next(event for event in events if event.phase == "k40_start")
        quorum = next(event for event in events
                      if event.phase == "discovery_membership_quorum_start")
        assert quorum.monotonic_ns < k40.monotonic_ns
        assert not any(event.phase == "checkpoint_publication" for event in events)
        background_release.set()
    finally:
        background_release.set()
        scheduler.close()
    publication = next(event for event in events
                       if event.phase == "checkpoint_publication")
    assert k40.monotonic_ns < publication.monotonic_ns
