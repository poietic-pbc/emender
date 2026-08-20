from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_dense_agent_pi_cli_eval_1n.sbatch"


def test_cli_eval_launcher_uses_real_pi_apptainer_and_fixed_frontier_world():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH -p batch", "#SBATCH -q debug", "#SBATCH --ntasks-per-node=8",
        "#SBATCH --no-requeue", "LOCAL_RANK=0", "--cli-canonical-system",
        "configs/pi/e97-cli-tools.ts", "eval_e97_dense_agent_pi_cli.py",
        "EMENDER_CLI_IMAGE", "EMENDER_CLI_IMAGE_SHA256", "sha256sum -c -",
        "completion cache=miss", "Partition=batch|QOS=debug",
    ):
        assert required in text
    assert "e97-v1-tools.ts" not in text
    assert "e97-v2-tools.ts" not in text


def test_cli_eval_launcher_requires_immutable_model_inputs():
    text = LAUNCHER.read_text()
    for required in ('${SOURCE_COMMIT:?}', '${RUN_ROOT:?}', '${CHECKPOINT:?}', '${CHECKPOINT_SHA256:?}'):
        assert required in text
    assert 'git archive "$SOURCE_COMMIT"' in text
