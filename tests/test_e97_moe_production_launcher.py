from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_35b_moe_production.sbatch"
SUBMITTER = ROOT / "scripts/frontier/submit_e97_35b_moe_scale.sh"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"


def test_production_launcher_is_fixed_world_fail_stop_and_canonical():
    text = LAUNCHER.read_text()
    assert "#SBATCH --no-requeue" in text
    assert "--kill-on-bad-exit=1" in text
    assert "Requeue=0" in text
    assert "Partition=$EXPECTED_PARTITION" in text
    assert "QOS=$EXPECTED_QOS" in text
    assert "sbcast -f \"$SEED_CHECKPOINT\" \"$JOB_SEED\"" in text
    assert "multinode production requires positive MAX_STEPS and TRAIN_MINUTES=0" in text
    assert '--max-steps "$MAX_STEPS" --minutes "$TRAIN_MINUTES"' in text
    assert "--save-every \"$SAVE_EVERY\"" in text
    assert "--keep-checkpoints \"$KEEP_CHECKPOINTS\"" in text
    assert 'RESUME_ROOT="$CHECKPOINT_ROOT"' in text
    assert "scontrol requeue" not in text


def test_submitter_verifies_partition_and_qos_separately():
    text = SUBMITTER.read_text()
    assert "'%i|%P|%q|%T|%D|%l'" in text
    assert '--nodes="$NODES"' in text
    assert '--qos="$QOS"' in text
    assert "TRAIN_MINUTES=${TRAIN_MINUTES:-0}" in text
    assert "MAX_STEPS=${MAX_STEPS:-200}" in text
    assert "HEAD == origin/main" in text
    assert "scancel" in text


def test_runner_uses_restored_step_for_data_and_one_canonical_island():
    text = RUNNER.read_text()
    assert 'parser.add_argument("--max-steps", type=int, default=0)' in text
    assert 'args.max_steps == 0 and args.minutes == 0' in text
    assert "groups.node_count > 1 and args.minutes > 0" in text
    assert "data_seed_base = 42 + starting_step" in text
    assert 'rank_seed=data_seed_base + dist.get_rank()' in text
    assert "if groups.node_index == 0:" in text
    assert "dist.broadcast(authority, src=0)" in text
    assert 'if dist.get_rank() == 0:' in text
    assert 'args.checkpoint_root / f"node-' not in text
    assert "step % args.save_every == 0" in text
