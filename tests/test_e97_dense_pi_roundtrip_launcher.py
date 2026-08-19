from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_dense_pi_roundtrip_1n.sbatch"
VALIDATOR = ROOT / "scripts/validate_e97_pi_roundtrip.py"


def test_pi_roundtrip_launcher_is_fixed_world_immutable_and_bounded():
    text = LAUNCHER.read_text()
    for required in (
        "#SBATCH -p batch",
        "#SBATCH -q debug",
        "#SBATCH --ntasks-per-node=8",
        "#SBATCH --no-requeue",
        "Partition=$EXPECTED_PARTITION",
        "QOS=$EXPECTED_QOS",
        "Requeue=0",
        "git archive \"$SOURCE_COMMIT\"",
        "sha256sum -c",
        "LOCAL_RANK=0",
        "--max-output-tokens 96",
        "--max-sessions 2",
        "--ingest-mode segment",
        "--v1-canonical-system",
        "--no-builtin-tools",
        "--no-skills",
        "--no-context-files",
        "configs/pi/e97-v1-tools.ts",
        "--system-prompt",
        "PI_CODING_AGENT_DIR",
        "PI_CODING_AGENT_SESSION_DIR",
    ):
        assert required in text
    assert "scontrol requeue" not in text


def test_pi_roundtrip_validator_requires_real_tool_loop_and_cache_hit():
    text = VALIDATOR.read_text()
    for required in (
        'event.get("type") == "tool_execution_start"',
        'event.get("type") == "tool_execution_end"',
        'event.get("type") == "agent_end"',
        '"one_calculator_call"',
        '"exact_expression"',
        '"tool_succeeded"',
        '"final_was_emitted"',
        '"completion cache=miss"',
        '"completion cache=hit"',
    ):
        assert required in text
