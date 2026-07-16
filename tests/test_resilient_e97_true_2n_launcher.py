from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_true_launcher_is_exact_debug_two_hour_topology_without_sentinels():
    text = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "#SBATCH -q debug" in text
    assert "#SBATCH -N 2" in text
    assert "#SBATCH -t 02:00:00" in text
    assert "#SBATCH --signal=B:TERM@300" in text
    assert "RESILIENT_E97_ROLE=manager" in text
    assert "RESILIENT_E97_ROLE=trainer" in text
    assert "CUDA_VISIBLE_DEVICES=" in text
    assert "--overlap --no-kill --exact" in text
    assert "ASYNC_LOCAL_STEPS=40" in text
    assert "topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none" in text
    assert "resilient_e97_rank_lane.py" not in text
    assert "sentinel ranks" in text


def test_legacy_runner_cannot_misreport_sentinels_as_real_trainers():
    legacy = (ROOT / "scripts/frontier/resilient_e97_rank_lane.py").read_text()
    assert '"role": "sentinel"' in legacy
    true_launcher = (ROOT / "scripts/frontier/resilient_e97_true_2n.sbatch").read_text()
    assert "RESILIENT_E97_MANAGER_COMMAND" in true_launcher
    assert "RESILIENT_E97_TRAINER_COMMAND" in true_launcher
