from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_dense_513b_paired_eval_1n.sbatch"
EVAL = ROOT / "scripts/eval_e97_dense_paired.py"


def test_dense_eval_is_read_only_immutable_fixed_world():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 1" in text
    assert "#SBATCH --ntasks-per-node=8" in text
    assert "#SBATCH --no-requeue" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "srun --label --kill-on-bad-exit=1 --wait=120" in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_dense_eval_binds_immutable_authorities_and_no_moe_conversion():
    launcher = LAUNCHER.read_text()
    evaluator = EVAL.read_text()
    for value in (
        "e559df3e8c540aef59ce8c9d73338f255cbe2fb9c7301ab45c7ef36a5b0fb857",
        "fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f",
        "SOURCE_COMMIT",
    ):
        assert value in launcher
    assert "load_e97_checkpoint" in evaluator
    assert "score_mmlu" in evaluator
    assert "score_wikitext" in evaluator
    assert "score_hellaswag" in evaluator
    assert "convert_e97_ffns_to_node_local_moe" not in evaluator
    assert 'weight_mode="train"' in evaluator
    assert 'loaded.step != 2322520' in evaluator
