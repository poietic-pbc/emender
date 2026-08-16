from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_router_preserved_canonical_1n.sbatch"


def test_canonical_router_preserved_launcher_is_fixed_world_and_bound():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 1" in text and "#SBATCH --no-requeue" in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "Requeue=0" in text and "NumTasks=8" in text
    assert "--max-steps 64" in text
    assert "--chunk-size 4096" in text and "--batch-size 1" in text
    assert "--diloco-k 1" in text
    assert "--expert-backend rocblas" in text
    assert "--checkpoint-root \"$RUN_ROOT/checkpoints\"" in text
    assert "--save-every 64" in text
    assert "scontrol requeue" not in text


def test_canonical_router_preserved_launcher_encodes_selected_recipe():
    text = LAUNCHER.read_text()
    assert "--lr \"$LR\" --weight-decay 0.01" in text
    assert "--sft-parent-optimizer-split router-preserved" in text
    assert "--resume-lr-override \"$LR\"" in text
    assert "--sft-resume-parent-optimizer" not in text
    assert "--sft-validation-exhaustive" in text
    assert "all-1777-packs-4352510-targets-exactly-once" in text
    assert "emender-record-pack-counter-v1" in text
    assert ".sampler_transition.optimizer_state == \"router-preserved\"" in text
    assert "(.optimizer_groups | length) == 2" in text


def test_canonical_publication_checks_all_eight_shards_and_manifest():
    text = LAUNCHER.read_text()
    assert ".complete == true" in text
    assert "(.ranks | length) == 8" in text
    assert ".step == 2338600" in text
    assert ".sampler.absolute_rank_sample_index == 64" in text
    assert "stat -c %s" in text
    assert "sha256sum \"$manifest\"" in text
