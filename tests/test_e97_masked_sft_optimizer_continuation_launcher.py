from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_masked_sft_optimizer_continuation.sbatch"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"
CHECKPOINT = ROOT / "ndm/e97_moe_checkpoint.py"


def test_preserved_optimizer_launcher_is_fixed_world_and_immutable():
    text = LAUNCHER.read_text()
    assert "#SBATCH --no-requeue" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()
    assert 'git archive "$SOURCE_COMMIT"' in text
    for identity in (
        "SOURCE_COMMIT", "PARENT_MANIFEST_SHA256",
        "AUTHORITY_MANIFEST_SHA256", "PACK_MANIFEST_SHA256",
    ):
        assert identity in text


def test_preserved_optimizer_launcher_does_not_silently_override_parent_lr():
    text = LAUNCHER.read_text()
    assert "optimizer_state=preserve-parent-exact" in text
    assert 'LR_OVERRIDE=${LR_OVERRIDE:-}' in text
    assert '--sft-resume-parent-optimizer' in text
    assert 'lr_override=(); [[ -z "$LR_OVERRIDE" ]]' in text
    assert '"${lr_override[@]}"' in text
    assert '--weight-decay 0.01' in text
    assert 'KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}' in text
    assert '--keep-checkpoints "$KEEP_CHECKPOINTS"' in text


def test_split_checkpoint_resume_reconstructs_optimizer_groups_before_restore():
    text = LAUNCHER.read_text()
    assert 'PARENT_OPTIMIZER_SPLIT=${PARENT_OPTIMIZER_SPLIT:-}' in text
    assert 'optimizer_split=${PARENT_OPTIMIZER_SPLIT:-none}' in text
    assert 'optimizer_transition=(--sft-resume-parent-optimizer)' in text
    assert 'optimizer_transition=(--sft-parent-optimizer-split "$PARENT_OPTIMIZER_SPLIT")' in text
    assert '"${optimizer_transition[@]}"' in text


def test_launcher_supports_explicit_fixed_world_sft_scale_transition():
    text = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    checkpoint = CHECKPOINT.read_text()
    assert 'EXPECTED_NODES" =~ ^(1|8|16|32|64)$' in text
    assert "SFT_TRANSITION_DATA_WORLD_SIZE" in text
    assert "--sft-transition-data-world-size" in text
    assert "allow_sft_world_size_transition" in runner
    assert '"sft-data-world-size-transition"' in runner
    assert "validate_sft_sampler_manifest" in checkpoint
    assert "may change only data_world_size" in checkpoint
    assert "K-aligned cursor" in checkpoint


def test_launcher_binds_explicit_record_reset_objective_transition():
    text = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "SFT_RESET_STATE_BETWEEN_RECORDS" in text
    assert "SFT_TRANSITION_RECORD_RESET" in text
    assert "--sft-reset-state-between-records" in text
    assert "--sft-transition-record-reset" in text
    assert "reset-at-record-boundaries-v1" in runner
    assert "sft-record-reset-objective-transition" in runner
    assert "record-reset objective transition requires a legacy K boundary" in runner


def test_launcher_supports_explicit_cross_node_gradient_sync_transition():
    text = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "SFT_CROSS_NODE_GRADIENT_SYNC" in text
    assert "SFT_TRANSITION_CROSS_NODE_GRADIENT_SYNC" in text
    assert "--sft-cross-node-gradient-sync" in text
    assert "--sft-transition-cross-node-gradient-sync" in text
    assert "synchronize_sharded_gradients_" in runner
    assert "corresponding-lane-gradient-sum-v1" in runner


def test_runner_records_explicit_counter_to_sft_optimizer_transition():
    runner = RUNNER.read_text()
    checkpoint = CHECKPOINT.read_text()
    assert '"counter-to-sft-preserve-optimizer"' in runner
    assert 'args.sft_parent_optimizer_split or "preserved"' in runner
    assert "allow_sft_parent_optimizer_transition=parent_optimizer_transition" in runner
    assert 'sampler_status = "sft-parent-optimizer-transition"' in checkpoint
    assert "validate_sft_parent_optimizer_transition" in checkpoint
