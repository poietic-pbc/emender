import json
import random
import subprocess
import sys

from scripts.build_e97_dense_agent_cli_sft import cli_observation, help_summary, trace


def test_cli_trajectories_are_typed_grounded_and_rs_free():
    for index, kind in enumerate(("json", "count", "search", "read")):
        user, turns, task = trace(kind, index, random.Random(9704 + index), discover=False)
        assert user
        assert turns[0][0] == "assistant" and turns[0][1].startswith("Action: cli\nArguments: ")
        observation = json.loads(turns[1][1])
        wrapper = json.loads(observation["stdout"])
        assert observation["exit_code"] == 0
        assert task["expected"] in observation["stdout"]
        assert task["evidence"] in observation["stdout"]
        assert turns[-1][1].startswith("Action: submit_answer")
        assert "\x1e" not in "".join(text for _, text in turns)
        assert wrapper


def test_discovery_trajectory_reads_help_before_using_command():
    _, turns, task = trace("json", 10, random.Random(9714), discover=True)
    assert json.loads(turns[0][1].partition("\nArguments: ")[2]) == {"argv": ["repo", "--help"]}
    help_result = json.loads(turns[1][1])
    assert "Inspect only the current repository" in help_result["stdout"]
    assert task["discover"] is True


def test_cli_observation_has_stable_model_visible_fields_only():
    value = json.loads(cli_observation(["repo", "--help"], "help\n"))
    assert set(value) == {
        "argv", "cwd", "exit_code", "stdout", "stderr",
        "stdout_truncated", "stderr_truncated", "timed_out",
    }
    compact = json.loads(cli_observation(["repo", "--help"], "usage: repo [-h]\n\nhelp\n", compact=True))
    assert compact == {"ok": True, "stdout": "usage: repo [-h]\n"}
    assert "duration_ms" not in value


def test_discovery_curriculum_uses_subcommand_help_before_execution():
    _, turns, _ = trace("count", 12, random.Random(9716), True, compact=True, subcommand_help=True)
    calls = [json.loads(text.partition("\nArguments: ")[2]) for role, text in turns if role == "assistant" and text.startswith("Action: cli")]
    assert calls[0] == {"argv": ["repo", "--help"]}
    assert calls[1] == {"argv": ["repo", "count", "--help"]}
    assert calls[2]["argv"][:2] == ["repo", "count"]


def test_help_summary_keeps_usage_and_removes_enumerated_integer_ranges():
    summary = help_summary("usage: repo search [-h]\n  [--max-results {1,2,3,4,5,6,7,8,9,10}]\n\noptions:\n")
    assert summary == "usage: repo search [-h] [--max-results INTEGER]\n"


def test_mixed_curriculum_makes_discovery_condition_observable(tmp_path):
    output = tmp_path / "authority"
    subprocess.run([
        sys.executable, "scripts/build_e97_dense_agent_cli_sft.py",
        "--output-root", str(output), "--records", "8", "--curriculum", "mixed",
        "--discovery-period", "2", "--compact-observations",
    ], check=True, capture_output=True, text=True)
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert all(row["user"].startswith("Inspect repo --help") for row in rows if row["task"]["discover"])
    assert all(not row["user"].startswith("Inspect repo --help") for row in rows if not row["task"]["discover"])
