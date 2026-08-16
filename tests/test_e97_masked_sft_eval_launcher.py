from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_canary_eval_3n.sbatch"
COMPARE = ROOT / "scripts/compare_e97_moe_sft_canary.py"


def test_sft_evaluation_is_three_independent_node_islands():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 3" in text
    assert 'NumNodes=3' in text and 'NumTasks=24' in text
    assert "--nodes=1 --ntasks=8" in text
    assert text.count("run_one \"") == 3
    assert "--kill-on-bad-exit=1 --wait=120" in text
    assert "#SBATCH --no-requeue" in text


def test_sft_evaluation_binds_all_checkpoint_and_panel_hashes():
    text = LAUNCHER.read_text()
    for field in ("PANEL_SHA256", "PARENT_MANIFEST_SHA256",
                  "LR2_MANIFEST_SHA256", "LR5_MANIFEST_SHA256", "SOURCE_COMMIT"):
        assert field in text
    assert '--generation-tokens 256' in text
    assert 'Partition=batch' in text and 'QOS=debug' in text


def test_sft_comparison_includes_parent_and_direct_lr_delta():
    text = COMPARE.read_text()
    assert '"parent-282b"' in text
    assert '"sft-lr2e-6"' in text
    assert '"sft-lr5e-6"' in text
    assert '("sft-lr2e-6", "sft-lr5e-6")' in text
    assert "assistant_response_nll_delta" in text
    assert "wikitext_nll_delta" in text
