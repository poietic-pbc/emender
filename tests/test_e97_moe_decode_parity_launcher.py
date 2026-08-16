from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_decode_parity_1n.sbatch"
RUNNER = ROOT / "scripts/eval_e97_moe_decode_parity.py"


def test_decode_parity_launcher_is_read_only_bound_one_node():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 1" in text and "#SBATCH --no-requeue" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert 'sha256sum "$PANEL"' in text
    assert 'sha256sum "$PARENT_ROOT/manifest.json"' in text
    assert "scripts/eval_e97_moe_decode_parity.py" in text
    assert "unaligned_inference_returns_prepadding_state" in text
    assert "--checkpoint-root" not in text
    assert "scontrol requeue" not in text


def test_decode_parity_compares_teacher_forced_and_greedy_paths():
    text = RUNNER.read_text()
    assert "teacher_forced_parity" in text
    assert "greedy_cached" in text and "greedy_recomputed" in text
    assert '"cached_recompute_top1_fraction"' in text
    assert '"cached_oneshot_top1_fraction"' in text
    assert '"greedy_exact_fraction"' in text
    assert 'summary["target_logp_max_abs"] <= 0.02' in text
    assert "[encoding.eot_token] * args.tokens" in text
    assert 'raise RuntimeError("cached recurrent decoding parity failed")' in text
