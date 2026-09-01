import json

from scripts import build_e97_open_swe_sft as open_swe
from scripts import build_e97_smoltalk2_sft as smoltalk


def test_open_swe_normalizes_workspace_paths_and_core_tools():
    kind, action = open_swe.action_from_call({
        "function": {"name": "execute_bash", "arguments": json.dumps({
            "command": "cd /workspace/owner__repo__1.0 && pytest -q"
        })}
    }, None)
    assert kind == "action"
    assert json.loads(action.split("Arguments: ", 1)[1]) == {"command": "pytest -q"}

    kind, action = open_swe.action_from_call({
        "function": {"name": "str_replace_editor", "arguments": json.dumps({
            "command": "str_replace",
            "path": "/workspace/owner__repo__1.0/src/core.py",
            "old_str": "OLD", "new_str": "NEW",
        })}
    }, "changed")
    assert kind == "action"
    assert action.startswith("Action: edit\n")
    assert json.loads(action.split("Arguments: ", 1)[1]) == {
        "path": "src/core.py", "oldText": "OLD", "newText": "NEW"
    }


def test_open_swe_finish_becomes_bounded_one_line_final():
    kind, final = open_swe.action_from_call({
        "function": {"name": "finish", "arguments": json.dumps({
            "message": "Changed src/a.py.\n\nTests: pytest -q"
        })}
    }, None)
    assert kind == "final"
    assert final == "Final: Changed src/a.py. Tests: pytest -q"
    assert len(final) <= 487


def test_open_swe_windowing_preserves_boundaries_and_bounds_records(monkeypatch):
    class Encoding:
        def encode_ordinary(self, text):
            return list(text.encode())

        def decode_single_token_bytes(self, token):
            return bytes([token])

    monkeypatch.setattr(open_swe.codec, "_WORKER_ENCODING", Encoding())
    messages = [
        ("system", "short"), ("user", "fix it"),
        ("assistant", "Action: read\nArguments: {}"), ("tool", "x" * 40),
        ("assistant", "Action: bash\nArguments: {}"), ("tool", "y" * 40),
        ("assistant", "Final: done"),
    ]
    records, dropped_observations, dropped_actions = open_swe.segment_messages(messages, 160)
    assert records
    assert dropped_actions == 0
    assert all(len(open_swe.encode_record(record)[0]) <= 160 for record in records)
    assert all(record[0][0] == "system" and record[1][0] == "user" for record in records)
    assert sum(sum(target for _, _, target in record) for record in records) >= 3
    assert dropped_observations >= 0


def test_smoltalk_admits_only_reviewed_non_tool_subsets():
    assert len(smoltalk.ALLOWED_PREFIXES) == 6
    assert not any("toolcalling" in value for value in smoltalk.ALLOWED_PREFIXES)
    assert smoltalk.DATASET_REVISION == "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
