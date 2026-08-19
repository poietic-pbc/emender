from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_dense_recurrent_cache_1n.sbatch"
QUALIFIER = ROOT / "scripts/qualify_e97_recurrent_cache.py"


def test_cache_qualification_is_fixed_world_immutable_and_fail_closed():
    text = LAUNCHER.read_text()
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -q debug" in text
    assert "#SBATCH -N 1" in text
    assert "#SBATCH --ntasks-per-node=8" in text
    assert "#SBATCH --no-requeue" in text
    assert "Partition=$EXPECTED_PARTITION" in text
    assert "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert "LOCAL_RANK=0" in text
    assert "sha256sum -c" in text
    assert "scontrol requeue" not in text


def test_cache_qualification_checks_incremental_contract():
    text = QUALIFIER.read_text()
    for required in (
        "advance_e97_cache",
        "generate_e97_from_cache",
        "e97_cache_suffix",
        "boundary_greedy_equal",
        "greedy_continuation_equal",
        "post_generation_argmax_equal",
        "post_generation_logits_max_abs_difference",
        "state_fp32",
        "transaction_preserved",
        "stop_token_consumed",
        "branch_rejected",
        "truncation_rejected",
    ):
        assert required in text
