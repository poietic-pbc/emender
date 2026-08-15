from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_masked_sft.sbatch"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"


def test_masked_sft_launcher_is_fixed_world_fail_stop_and_scheduler_bound():
    text = LAUNCHER.read_text()
    assert "#SBATCH --no-requeue" in text
    assert 'Partition=batch' in text and 'QOS=$EXPECTED_QOS' in text
    assert 'Requeue=0' in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_masked_sft_launcher_binds_all_immutable_authorities():
    text = LAUNCHER.read_text()
    for value in (
        "AUTHORITY_MANIFEST_SHA256", "PACK_MANIFEST_SHA256",
        "PARENT_MANIFEST_SHA256", "SOURCE_COMMIT", "SFT_SAMPLER_KEY",
    ):
        assert value in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert 'sha256sum "$AUTHORITY_ROOT/manifest.json"' in text
    assert 'sha256sum "$PARENT_ROOT/manifest.json"' in text
    assert 'PACK_ROOT/validation.json' in text


def test_masked_sft_recipe_uses_fresh_parent_and_conservative_objective():
    text = LAUNCHER.read_text()
    assert '--sft-parent-root "$PARENT_ROOT"' in text
    assert '--lr "$LR" --weight-decay 0.0' in text
    assert '--batch-size "$BATCH_SIZE" --chunk-size "$CONTEXT_SIZE"' in text
    assert "--offload-schedulefree-state" in text
    assert '--sft-validation-batches "$VALIDATION_BATCHES"' in text


def test_runner_has_target_clocks_and_target_weighted_merge():
    text = RUNNER.read_text()
    assert "sft_target_tokens" in text
    assert "loss_reduction=\"sum\"" in text
    assert "sum_replicated_gradients_" in text
    assert "interval_node_target_tokens" in text
    assert "absolute_rank_sample_index" in text
