from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_masked_sft_lr_canary_32n.sbatch"


def test_canary_is_exact_32_node_debug_binding_split_into_matched_worlds():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 32" in text
    assert "#SBATCH -q debug" in text
    assert "#SBATCH --no-requeue" in text
    assert 'NumNodes=32' in text and 'NumTasks=256' in text
    assert text.count("--nodes=16 --ntasks=128") == 1
    assert "nodes_a=" in text and "nodes_b=" in text
    assert "WORLD_SIZE=128" in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_canary_arms_differ_only_in_lr_and_checkpoint_delay():
    text = LAUNCHER.read_text()
    assert "launch_arm lr2e-6 0.000002 0" in text
    assert "launch_arm lr5e-6 0.000005 120" in text
    assert '--lr "$ARM_LR"' in text
    assert '--final-checkpoint-delay-seconds "$ARM_DELAY"' in text
    assert '--sft-sampler-key 42' in text
    assert "MAX_STEPS=${MAX_STEPS:-512}" in text
    assert "DILOCO_K=${DILOCO_K:-64}" in text


def test_canary_binds_parent_data_pack_and_source_authorities():
    text = LAUNCHER.read_text()
    for field in (
        "SOURCE_COMMIT", "AUTHORITY_MANIFEST_SHA256", "PACK_MANIFEST_SHA256",
        "PARENT_MANIFEST_SHA256", "PACK_ROOT/validation.json",
    ):
        assert field in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert 'sha256sum "$PARENT_ROOT/manifest.json"' in text


def test_canary_is_fail_stop_and_publishes_both_terminal_results():
    text = LAUNCHER.read_text()
    assert "--kill-on-bad-exit=1 --wait=120" in text
    assert 'wait "$pid_a"' in text and 'wait "$pid_b"' in text
    assert 'lr2e-6-rc.txt' in text and 'lr5e-6-rc.txt' in text
    assert "rc_a == 0 && rc_b == 0" in text
