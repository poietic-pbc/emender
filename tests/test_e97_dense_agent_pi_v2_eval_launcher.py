from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_dense_agent_pi_v2_eval_1n.sbatch"


def test_v2_pi_eval_launcher_is_fixed_world_fail_closed_and_rs_free():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH -p batch",
        "#SBATCH -q debug",
        "#SBATCH --ntasks-per-node=8",
        "#SBATCH --no-requeue",
        'LOCAL_RANK=0',
        "--v2-canonical-system",
        "configs/pi/e97-v2-tools.ts",
        "eval_e97_dense_agent_pi_v2.py",
        "aggregate_e97_dense_agent_pi_v2.py",
        "completion cache=miss",
        "sha256sum -c -",
    ):
        assert required in text
    assert "--v1-canonical-system" not in text
    assert "e97-v1-tools.ts" not in text


def test_v2_pi_eval_launcher_requires_immutable_inputs():
    text = LAUNCHER.read_text()
    for required in ('${SOURCE_COMMIT:?}', '${RUN_ROOT:?}', '${CHECKPOINT:?}', '${CHECKPOINT_SHA256:?}'):
        assert required in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert 'Partition=batch|QOS=debug' in text
