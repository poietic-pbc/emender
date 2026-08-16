from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_router_preserved_k64_8n.sbatch"


def test_k64_launcher_is_fixed_eight_node_world():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 8" in text and "#SBATCH --no-requeue" in text
    assert "NumNodes=8" in text and "NumTasks=64" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text
    assert "WORLD_SIZE=64" in text
    assert "--max-steps 64" in text and "--diloco-k 64" in text
    assert "expected_merges=1" in text
    assert "scontrol requeue" not in text


def test_k64_launcher_uses_selected_full_model_recipe():
    text = LAUNCHER.read_text()
    assert "--chunk-size 4096" in text and "--batch-size 1" in text
    assert "--lr \"$LR\" --weight-decay 0.01" in text
    assert "--expert-backend rocblas" in text
    assert "--sft-parent-optimizer-split router-preserved" in text
    assert "--resume-lr-override \"$LR\"" in text
    assert "--sft-resume-parent-optimizer" not in text
    assert "--offload-schedulefree-state" in text
    assert "--save-every 64" in text


def test_k64_publication_checks_canonical_node_zero_generation():
    text = LAUNCHER.read_text()
    assert ".complete == true" in text and "(.ranks | length) == 8" in text
    assert ".sampler.identity.data_world_size == 64" in text
    assert ".sampler.absolute_rank_sample_index == 64" in text
    assert ".sampler_transition.optimizer_state == \"router-preserved\"" in text
    assert "stat -c %s" in text and "sha256sum \"$manifest\"" in text
