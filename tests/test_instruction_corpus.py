import json

import pytest

from ndm.data.instruction_corpus import RS, serialize_row


def test_serializes_complete_chat_as_plain_utf8():
    text, replaced = serialize_row("chat", {"messages": [
        {"role": "system", "content": "Be useful."},
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "reasoning_content": "Inspect first.",
         "content": "Done.", "tool_calls": [{"name": "bash", "x": 1}]},
        {"role": "tool", "content": "ok"},
    ]})
    assert text.startswith("System:\nBe useful.\n\nUser:\nFix it.")
    assert "Assistant reasoning:\nInspect first." in text
    assert 'Assistant tool call:\n[{"name":"bash","x":1}]' in text
    assert text.endswith("Tool:\nok")
    assert replaced == 0
    assert RS not in text


def test_accepts_json_trajectory_and_sharegpt_roles():
    trajectory = json.dumps([
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": "answer"},
    ])
    text, _ = serialize_row("trace", {"trajectory": trajectory})
    assert text == "User:\nquestion\n\nAssistant:\nanswer"


def test_normalizes_lines_and_replaces_embedded_rs():
    text, replaced = serialize_row("chat", {
        "conversations": [{"role": "user", "content": "a\r\nb\x1ec"}]})
    assert text == "User:\na\nb c"
    assert replaced == 1


def test_rejects_rows_without_complete_sequence():
    with pytest.raises(ValueError, match="no complete message sequence"):
        serialize_row("bad", {"text": "not a structured trajectory"})
