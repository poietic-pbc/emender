from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_router_preserved_eval_2n.sbatch"
RUNNER = ROOT / "scripts/eval_e97_moe_paired.py"
COMPARATOR = ROOT / "scripts/compare_e97_moe_sft_router_preserved.py"


def test_router_preserved_eval_is_two_isolated_eight_rank_worlds():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 2" in text and "#SBATCH --no-requeue" in text
    assert "NumNodes=2" in text and "NumTasks=16" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert text.count("--nodes=1 --ntasks=8") == 1
    assert "parent-282b" in text and "router-preserved-1e-4-64u" in text
    assert "--generation-tokens 256" in text
    assert "--expert-backend rocblas" in text
    assert "scontrol requeue" not in text


def test_router_preserved_eval_binds_inputs_and_native_cache_generation():
    launcher = LAUNCHER.read_text()
    runner = RUNNER.read_text()
    assert "PANEL_SHA256" in launcher
    assert "PARENT_MANIFEST_SHA256" in launcher
    assert "CANDIDATE_MANIFEST_SHA256" in launcher
    assert "f190f0bcf735c07dc7f59ec27ac98fc8c0257718d8a331fc62ab6c529314814d" in launcher
    assert "generation=native-fp32-recurrent-cache" in launcher
    assert "prev_hiddens=hiddens" in runner
    assert "stop_tokens = {eos, 218}" in runner


def test_router_preserved_comparison_reports_required_gates():
    text = COMPARATOR.read_text()
    assert '"assistant_response_nll_delta"' in text
    assert '"wikitext_nll_delta"' in text
    assert '"mmlu_accuracy_delta"' in text
    assert '"hellaswag_normalized_accuracy_delta"' in text
    assert '"generations"' in text and '"recurrent_health"' in text
    assert '"routing"' in text
    assert '(row["id"], row["mode"])' in text
