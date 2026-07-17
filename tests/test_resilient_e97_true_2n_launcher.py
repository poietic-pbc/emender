from pathlib import Path
import json

from scripts.frontier.resilient_e97_allocation_supervisor import AllocationSupervisor, Child


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
