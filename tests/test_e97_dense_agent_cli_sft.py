import json
import random

from scripts.build_e97_dense_agent_cli_sft import cli_observation, trace


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
    assert "duration_ms" not in value
