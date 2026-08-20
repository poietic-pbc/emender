import json
import random
from pathlib import Path

from scripts.build_e97_dense_agent_cli_sft import trace
from scripts.eval_e97_dense_agent_pi_cli import load_tasks, make_sandbox, score


def authority(tmp_path: Path) -> Path:
    rows = []
    for index, kind in enumerate(("json", "count", "search", "read")):
        user, _, task = trace(kind, index, random.Random(9704 + index), discover=index == 0)
        rows.append({"id": f"task-{index}", "source": f"emender-agent-cli-{kind}-v1", "split": 1, "user": user, "task": task})
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return tmp_path


def test_cli_validation_authority_loads_mechanical_tasks(tmp_path: Path):
    tasks = load_tasks(authority(tmp_path))
    assert len(tasks) == 4
    assert {task["kind"] for task in tasks} == {"json", "count", "search", "read"}
    assert all(task["fixtures"] and task["argv"] and task["expected"] for task in tasks)


def test_cli_sandbox_materializes_only_task_fixtures(tmp_path: Path):
    root = tmp_path / "authority"; root.mkdir()
    task = load_tasks(authority(root))[0]
    sandbox = make_sandbox(tmp_path / "sandboxes", task)
    for fixture in task["fixtures"]:
        assert (sandbox / fixture["path"]).read_text() == fixture["content"]


def test_cli_score_requires_expected_sequence_argv_value_and_evidence(tmp_path: Path):
    task = next(task for task in load_tasks(authority(tmp_path)) if not task["discover"])
    rows = [
        {"type": "tool_execution_start", "toolCallId": "cli", "toolName": "cli", "args": {"argv": task["argv"]}},
        {"type": "tool_execution_end", "toolCallId": "cli", "isError": False},
        {"type": "tool_execution_start", "toolCallId": "submit", "toolName": "submit_answer", "args": {"value": task["expected"], "evidence": task["evidence"]}},
        {"type": "tool_execution_end", "toolCallId": "submit", "isError": False},
        {"type": "agent_end"},
    ]
    assert score(task, rows, 0)["status"] == "pass"
    rows[2]["args"]["value"] = "fabricated"
    assert score(task, rows, 0)["status"] == "fail"
