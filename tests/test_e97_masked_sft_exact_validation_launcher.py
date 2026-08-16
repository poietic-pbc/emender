from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_exact_validation_1n.sbatch"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"


def test_exact_validation_launcher_is_read_only_fixed_world():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 1" in text and "#SBATCH --no-requeue" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert "--max-steps 0" in text
    assert "--sft-validation-exhaustive --sft-validation-only" in text
    assert "--checkpoint-root" not in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_validation_only_runner_exits_before_training_loop():
    text = RUNNER.read_text()
    assert '"--sft-validation-only"' in text
    assert '"validation_only_complete"' in text
    validation = text.index('emit(args.log_jsonl, "validation_only_complete"')
    training = text.index("replicated = tuple(node_replicated_parameters(model))")
    assert validation < training
