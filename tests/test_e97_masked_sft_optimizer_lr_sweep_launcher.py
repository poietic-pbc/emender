from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_optimizer_lr_sweep_10n.sbatch"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"


def test_optimizer_lr_sweep_is_ten_isolated_fixed_worlds():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 10" in text
    assert "#SBATCH --no-requeue" in text
    assert '"NumNodes=10"' in text and '"NumTasks=80"' in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --exclusive --exact --nodes=1 --ntasks=8" in text
    assert "WORLD_SIZE=8" in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_optimizer_lr_sweep_crosses_state_and_requested_learning_rates():
    text = LAUNCHER.read_text()
    for lr in ("0.000005", "0.00003", "0.0001", "0.0003", "0.001007"):
        assert text.count(lr) >= 2
    assert text.count("fresh-lr") >= 5
    assert text.count("preserved-lr") >= 5
    assert "--sft-resume-parent-optimizer --resume-lr-override" in text
    assert "--weight-decay 0.01" in text
    assert "--checkpoint-root" not in text


def test_optimizer_lr_sweep_binds_authorities_and_exact_validation():
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    for identity in (
        "SOURCE_COMMIT", "PARENT_MANIFEST_SHA256",
        "AUTHORITY_MANIFEST_SHA256", "PACK_MANIFEST_SHA256",
    ):
        assert identity in launcher
    assert 'git archive "$SOURCE_COMMIT"' in launcher
    assert "--sft-validation-exhaustive" in launcher
    assert "validation=all-1777-packs-exactly-once" in launcher
    assert '"exact-pack-enumeration"' in runner
    assert "exhaustive SFT validation requires one complete eight-rank node" in runner
    assert "np.random.default_rng(970035)" in runner
    assert 'result["routing"] = routing' in runner
    assert '"unused_experts"' in runner
