from pathlib import Path
import json
import os
import subprocess

from scripts.frontier.resilient_e97_allocation_supervisor import AllocationSupervisor, Child
from scripts.frontier.check_resilient_e97_parity import compare


ROOT = Path(__file__).parents[1]


def test_true_launcher_is_exact_debug_two_hour_topology_without_sentinels():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    supervisor = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert "#SBATCH -q debug" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH --gpus-per-node=8" in text
    assert "#SBATCH -t 02:00:00" in text
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
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_QOS": "debug",
        "SLURM_TIMELIMIT": "02:00:00",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "REPO": str(repo),
        "RUN_DIR": str(tmp_path / "run"),
        "RESILIENT_E97_RUN_ID": "run",
        "RESILIENT_E97_SOURCE_ID": "source",
        "RESILIENT_E97_PAYLOAD_ID": "payload",
        "RESILIENT_E97_SEED": "/seed.pt",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
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
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_QOS": "debug",
        "SLURM_JOB_NODELIST": "node[0-1]",
        "RESILIENT_E97_STARTUP_SMOKE": "1",
        "RESILIENT_E97_REQUESTED_WALLTIME": "00:20:00",
        "RESILIENT_E97_GENERATIONS": "1",
        "REPO": str(repo),
        "RUN_DIR": str(tmp_path / "run"),
        "RESILIENT_E97_RUN_ID": "run",
        "RESILIENT_E97_SOURCE_ID": "source",
        "RESILIENT_E97_PAYLOAD_ID": "payload",
        "RESILIENT_E97_SEED": "/seed.pt",
        "RESILIENT_E97_TRAIN_ARGS_JSON": str(approved_args),
        "RESILIENT_E97_DATA": "/data",
    }
    env.pop("SLURM_TIMELIMIT", None)
    subprocess.run(["bash", str(ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch")],
                   env=env, text=True, capture_output=True, check=True)


def test_approved_training_arguments_are_flat_overrides():
    path = ROOT / "configs/frontier/e97_resilient_split_role_flat.json"
    value = json.loads(path.read_text())
    assert value["level"] == "E97" and value["optimizer"] == "schedulefree"
    assert value["dim"] == 1792 and value["lr"] == 0.001007
    assert not ({"resolved", "export", "source_artifacts"} & value.keys())


def test_launcher_rejects_nonapproved_training_arguments_before_roles_start():
    script = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert 'APPROVED_TRAIN_ARGS="$REPO/configs/frontier/e97_resilient_split_role_flat.json"' in script
    assert 'realpath "$RESILIENT_E97_TRAIN_ARGS_JSON"' in script
    assert "training arguments must be the approved flat E97 split-role configuration" in script


def test_launch_modes_preserve_identical_local_role_identity_and_environment(tmp_path, monkeypatch):
    launched = []

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


def test_real_trainer_keeps_liveness_during_checkpoint_load_and_training():
    text = (ROOT / "scripts/frontier/resilient_e97_role.py").read_text()
    trainer = text[text.index("def trainer(args)"):]
    assert trainer.index("_liveness_heartbeat(bulk, identity)") < trainer.index("_load_real(args)")
    assert 'f"{identity}.liveness.json"' in text
    assert '"stage": "runtime_import"' in text
    assert '"generation": 0, "step": 0' in text
    assert text.count("if _IMPORT_HEARTBEAT is not None:") == 2


def test_frontier_default_keeps_supervision_state_node_local():
    text = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert 'os.environ.get("RESILIENT_E97_LAUNCH_MODE", "node-local")' in text
    assert 'launch_backend="node-local-child"' in text
    assert 'sys.executable, __file__, "--node-local"' in text
    assert '"-N2", "-n2"' in text
