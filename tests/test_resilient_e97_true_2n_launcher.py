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
    supervisor = repo / "scripts/frontier/resilient_e97_allocation_supervisor.py"
    supervisor.write_text(
        "import os\nprint(os.environ['RESILIENT_E97_TRAINER_COMMAND'])\n"
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
        "RESILIENT_E97_TRAIN_ARGS_JSON": "/args.json",
        "RESILIENT_E97_DATA": "/data",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch")],
        env=env, text=True, capture_output=True, check=True,
    )
    trainer_command = result.stdout.splitlines()[-1]
    assert trainer_command.endswith("--migration-policy initialize-from-approved-config")
    assert "  " not in trainer_command


def test_approved_training_arguments_are_flat_overrides():
    path = ROOT / "configs/frontier/e97_resilient_split_role_flat.json"
    value = json.loads(path.read_text())
    assert value["level"] == "E97" and value["optimizer"] == "schedulefree"
    assert value["dim"] == 1792 and value["lr"] == 0.001007
    assert not ({"resolved", "export", "source_artifacts"} & value.keys())


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


def test_frontier_default_avoids_nested_srun_steps():
    text = (ROOT / "scripts/frontier/resilient_e97_allocation_supervisor.py").read_text()
    assert 'os.environ.get("RESILIENT_E97_LAUNCH_MODE", "independent-step")' in text
