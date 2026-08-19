#!/usr/bin/env python3
"""Validate one real Pi -> E97 -> tool -> E97 JSON event trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi-jsonl", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-expression", default="608 * 57")
    args = parser.parse_args()

    events = []
    for line_number, line in enumerate(args.pi_jsonl.read_text().splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Pi JSONL line {line_number}") from exc

    starts = [event for event in events if event.get("type") == "tool_execution_start"]
    ends = [event for event in events if event.get("type") == "tool_execution_end"]
    agent_ends = [event for event in events if event.get("type") == "agent_end"]
    assistant_messages = [
        event["message"]
        for event in events
        if event.get("type") == "message_end"
        and isinstance(event.get("message"), dict)
        and event["message"].get("role") == "assistant"
    ]
    final_texts = []
    for message in assistant_messages:
        content = message.get("content", [])
        if isinstance(content, str):
            final_texts.append(content)
        elif isinstance(content, list):
            final_texts.extend(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )

    server_log = args.server_log.read_text()
    checks = {
        "one_calculator_call": len(starts) == 1 and starts[0].get("toolName") == "calculator",
        "exact_expression": len(starts) == 1 and starts[0].get("args") == {"expression": args.expected_expression},
        "tool_succeeded": len(ends) == 1 and ends[0].get("isError") is False,
        "tool_returned_value": len(ends) == 1 and "34656" in json.dumps(ends[0].get("result")),
        "agent_completed": len(agent_ends) == 1,
        "final_was_emitted": any(text.startswith("Final:") for text in final_texts),
        "server_cache_miss": "completion cache=miss" in server_log,
        "server_cache_hit": "completion cache=hit" in server_log,
    }
    result = {
        "schema": "emender-e97-pi-roundtrip-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "tool_calls": starts,
        "final_texts": final_texts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
