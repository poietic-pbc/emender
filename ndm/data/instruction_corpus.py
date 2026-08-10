"""Canonical plain-text serialization for the E97 instruction corpus."""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

RS = "\x1e"
SERIALIZER_SCHEMA = "emender-e97-instruction-plain-v1"

_ROLE_LABELS = {
    "system": "System",
    "human": "User",
    "user": "User",
    "assistant": "Assistant",
    "gpt": "Assistant",
    "tool": "Tool",
    "function": "Tool",
    "developer": "Developer",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _clean(value: Any) -> tuple[str, int]:
    text = _text(value).replace("\r\n", "\n").replace("\r", "\n")
    replaced = text.count(RS)
    if replaced:
        text = text.replace(RS, " ")
    return text.strip(), replaced


def _message_parts(message: Mapping[str, Any]) -> tuple[list[str], int]:
    role = str(message.get("role", message.get("from", "unknown"))).lower()
    label = _ROLE_LABELS.get(role, role.replace("_", " ").title() or "Message")
    parts: list[str] = []
    replaced = 0

    reasoning, n = _clean(message.get("reasoning_content", message.get("reasoning")))
    replaced += n
    if reasoning:
        parts.append(f"{label} reasoning:\n{reasoning}")

    content, n = _clean(message.get("content", message.get("value")))
    replaced += n
    if content:
        parts.append(f"{label}:\n{content}")

    tool_calls = message.get("tool_calls")
    if tool_calls not in (None, [], ""):
        rendered, n = _clean(tool_calls)
        replaced += n
        if rendered:
            parts.append(f"{label} tool call:\n{rendered}")

    # Some trajectory formats store a single function call separately.
    function_call = message.get("function_call")
    if function_call not in (None, {}, ""):
        rendered, n = _clean(function_call)
        replaced += n
        if rendered:
            parts.append(f"{label} function call:\n{rendered}")
    return parts, replaced


def serialize_messages(messages: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    sections: list[str] = []
    replaced = 0
    for message in messages:
        if not isinstance(message, Mapping):
            rendered, n = _clean(message)
            replaced += n
            if rendered:
                sections.append(f"Message:\n{rendered}")
            continue
        parts, n = _message_parts(message)
        replaced += n
        sections.extend(parts)
    return "\n\n".join(sections).strip(), replaced


def _decode_sequence(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return value


def serialize_row(source: str, row: Mapping[str, Any]) -> tuple[str, int]:
    """Serialize one already-complete conversation/trajectory row.

    Multi-table sources such as SWE-chat must be reconstructed into a row with
    ``messages`` before calling this function.
    """
    sequence = None
    for key in ("messages", "conversations", "trajectory"):
        if row.get(key) not in (None, "", []):
            sequence = _decode_sequence(row[key])
            break
    if not isinstance(sequence, list):
        raise ValueError(f"{source}: row has no complete message sequence")
    text, replaced = serialize_messages(sequence)
    if not text:
        raise ValueError(f"{source}: message sequence serialized to empty text")
    tools = row.get("tools")
    if tools not in (None, "", []):
        tools = _decode_sequence(tools)
        rendered, n = _clean(tools)
        replaced += n
        if rendered:
            text = f"Tools:\n{rendered}\n\n{text}"
    return text, replaced
