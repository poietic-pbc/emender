import json
from pathlib import Path

from scripts.eval_e97_dense_agent_pi_v2 import make_sandbox, score, validation_tasks


def test_validation_panel_is_exact_disjoint_300():
    tasks = validation_tasks(9702)
    assert len(tasks) == 300
    assert len({task["id"] for task in tasks}) == 300
    assert {task["kind"] for task in tasks} == {"calculator", "lookup", "count"}
    assert all(task["expected"] for task in tasks)


def test_simplified_v3_panel_uses_field_specific_lookup_tools():
    tasks = validation_tasks(9702, "v3")
    assert len(tasks) == 291
    lookups = [task for task in tasks if task["kind"] == "lookup"]
    assert {task["first_tool"] for task in lookups} == {"lookup_owner", "lookup_budget"}
    assert all(set(task["first_args"]) == {"project"} for task in lookups)


def test_sandboxes_reproduce_lookup_and_count_observations(tmp_path: Path):
    tasks = validation_tasks(9702)
    lookup = next(task for task in tasks if task["kind"] == "lookup")
    lookup_root = make_sandbox(tmp_path, lookup)
    assert lookup["expected"] in (lookup_root / lookup["first_args"]["path"]).read_text()
    lookup_v3 = next(task for task in validation_tasks(9702, "v3") if task["kind"] == "lookup")
    lookup_v3_root = make_sandbox(tmp_path, lookup_v3)
    project = lookup_v3["first_args"]["project"]
    assert lookup_v3["expected"] in (lookup_v3_root / "records" / f"{project}.txt").read_text()

    count = next(task for task in tasks if task["kind"] == "count")
    count_root = make_sandbox(tmp_path, count)
    directory = count_root / count["first_args"]["path"]
    suffix = count["first_args"]["suffix"]
    assert len(list(directory.glob(f"*{suffix}"))) == int(count["expected"])


def test_score_requires_exact_grounded_terminating_path():
    task = next(task for task in validation_tasks(9702) if task["kind"] == "calculator")
    events = [
        {"type": "tool_execution_start", "toolCallId": "op", "toolName": task["first_tool"], "args": task["first_args"]},
        {"type": "tool_execution_end", "toolCallId": "op", "isError": False},
        {"type": "tool_execution_start", "toolCallId": "answer", "toolName": "submit_answer", "args": {"value": task["expected"]}},
        {"type": "tool_execution_end", "toolCallId": "answer", "isError": False},
        {"type": "agent_end"},
    ]
    assert score(task, events, 0)["status"] == "pass"
    events[2]["args"] = {"value": "fabricated"}
    failed = score(task, events, 0)
    assert failed["status"] == "fail"
    assert failed["checks"]["submitted_value"] is False
