from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from scripts.frontier.resilient_e97_allocation_supervisor import AllocationSupervisor, Child
from scripts.frontier.check_resilient_e97_parity import compare


ROOT = Path(__file__).parents[1]
APPROVED_ENV = Path("/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312")


def test_true_launcher_is_exact_debug_twenty_minute_topology_without_sentinels():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert "#SBATCH -q debug" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH --gpus-per-node=8" in text
    assert "#SBATCH -t 00:20:00" in text
    assert "#SBATCH --signal=B:TERM@300" in text
    assert '"RESILIENT_E97_ROLE": child.role' in supervisor
    assert '"CUDA_VISIBLE_DEVICES="' in supervisor
    assert '"--overlap", "--no-kill", "--exact"' in supervisor
    assert '"ASYNC_LOCAL_STEPS=40"' in supervisor
    assert "topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none" in text
    assert "resilient_e97_rank_lane.py" not in text
    assert "sentinel ranks" in text


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


def test_launcher_omits_empty_resume_argument(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts/frontier").mkdir(parents=True)
    (repo / "configs/frontier").mkdir(parents=True)
    approved_args = repo / "configs/frontier/e97_resilient_split_role_flat.json"
    approved_args.write_text((ROOT / "configs/frontier/e97_resilient_split_role_flat.json").read_text())
    supervisor = repo / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    supervisor.write_text(
        "import os\nprint(os.environ['RESILIENT_E97_TRAINER_COMMAND'])\n"
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
    srun.write_text("#!/bin/sh\nwhile [ \"$1\" != bash ]; do shift; done\nexec \"$@\"\n")
    srun.chmod(0o755)
    tokenizer_cache = tmp_path / "p50k.cache"
    tokenizer_cache.write_text("offline-tokenizer-cache")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{APPROVED_ENV / 'bin'}:{os.environ['PATH']}",
        "EMENDER_CONDA_ENV": str(APPROVED_ENV),
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_QOS": "debug",
        "SLURM_TIMELIMIT": "00:20:00",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "REPO": str(repo),
        "RUN_DIR": str(tmp_path / "run"),
        "RESILIENT_E97_RUN_ID": "run",
        "RESILIENT_E97_SOURCE_ID": "source",
        "RESILIENT_E97_PAYLOAD_ID": "payload",
        "RESILIENT_E97_SEED": "/seed.pt",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
        "RESILIENT_E97_TIKTOKEN_CACHE_FILE": str(tokenizer_cache),
        "RESILIENT_E97_TIKTOKEN_SHA256": hashlib.sha256(tokenizer_cache.read_bytes()).hexdigest(),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch")],
        env=env, text=True, capture_output=True, check=True,
    )
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


def test_launcher_activates_approved_frontier_python_before_any_role():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    activation = text.index("frontier_activate_emender_conda_env")
    supervisor = text.index('exec "$TRAIN_PYTHON_BIN"')
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
    (repo / "scripts/frontier/resilient_e97_allocation_supervisor.py").write_text("pass\n")
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
    srun.write_text("#!/bin/sh\nwhile [ \"$1\" != bash ]; do shift; done\nexec \"$@\"\n")
    srun.chmod(0o755)
    tokenizer_cache = tmp_path / "p50k.cache"
    tokenizer_cache.write_text("offline-tokenizer-cache")
    env = {
        **os.environ,
        "PATH": f"{bindir}:{APPROVED_ENV / 'bin'}:{os.environ['PATH']}",
        "EMENDER_CONDA_ENV": str(APPROVED_ENV),
        "SLURM_JOB_NUM_NODES": "2",
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
        "RESILIENT_E97_SEED": "/seed.pt",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
        "RESILIENT_E97_TIKTOKEN_CACHE_FILE": str(tokenizer_cache),
        "RESILIENT_E97_TIKTOKEN_SHA256": hashlib.sha256(tokenizer_cache.read_bytes()).hexdigest(),
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
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1" in script
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
    monkeypatch.setenv("RESILIENT_E97_FENCE_DB", str(tmp_path / "pool.sqlite"))
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ID", "job-a")
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_INCARNATION", "inc-a")
    monkeypatch.setenv("RESILIENT_E97_LEASE_TTL_S", "10")
    monkeypatch.setenv("RESILIENT_E97_LEASE_RENEW_S", "1")
    # Admission publishes these keys directly. Register their original values
    # with monkeypatch so later subprocess integration tests cannot inherit a
    # lease after its database path has been restored.
    monkeypatch.setenv("RESILIENT_E97_ALLOCATION_LEASE", "")
    monkeypatch.setenv("RESILIENT_E97_FENCE_EPOCH", "0")
    first = module._allocation_admission(tmp_path)
    try:
        assert isinstance(first, module.AllocationLeaseGuard)
        assert int(os.environ["RESILIENT_E97_FENCE_EPOCH"]) == first.lease.fence
        telemetry = json.loads((tmp_path / "supervision/allocation-lease.json").read_text())
        assert telemetry["stage"] == "admitted_before_model_load"
        assert telemetry["ready_hard_s"] == 180
        assert telemetry["k40_hard_s"] == 420
        assert telemetry["exchange_commit_hard_s"] == 180
        assert telemetry["first_commit_hard_s"] == 720
        monkeypatch.setenv("RESILIENT_E97_ALLOCATION_ID", "job-b")
        monkeypatch.setenv("RESILIENT_E97_ALLOCATION_INCARNATION", "inc-b")
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
                                 "generation": 2}))
    monkeypatch.setenv("RESILIENT_E97_INJECT_MANAGER", "1:-1:2")
    assert supervisor._deadline_reason(child, 10) is None
    monkeypatch.setenv("RESILIENT_E97_INJECT_NODE_STEP", "1:-1:2")
    assert supervisor._deadline_reason(child, 10) == "injected_generation_gate"
    assert supervisor._deadline_reason(child, 10) is None
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


def test_frontier_default_keeps_supervision_state_node_local():
    text = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert 'os.environ.get("RESILIENT_E97_LAUNCH_MODE", "node-local")' in text
    assert 'launch_backend="node-local-child"' in text
    assert 'sys.executable, __file__, "--node-local"' in text
    assert '"-N2", "-n2"' in text
