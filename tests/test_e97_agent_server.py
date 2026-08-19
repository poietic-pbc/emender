from dataclasses import dataclass
from http.client import HTTPConnection
from http.server import HTTPServer
from threading import Thread

import pytest

from ndm.e97_agent_protocol import AgentProtocolError, RS
from ndm.e97_agent_server import (
    AgentCompletionService,
    RecurrentSessionStore,
    chat_completion_sse,
    make_openai_handler,
)


@dataclass(frozen=True)
class FakeCache:
    token_ids: tuple[int, ...]

    @property
    def state_bytes(self):
        return 1234


class FakeEngine:
    checkpoint = "/fake/e97.pt"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.advance_calls = []

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8")

    def advance(self, token_ids, cache=None):
        consumed = tuple(token_ids)
        self.advance_calls.append((None if cache is None else cache.token_ids, consumed))
        return FakeCache((() if cache is None else cache.token_ids) + consumed)

    def generate(self, cache, *, max_new_tokens, temperature, top_p):
        text = self.outputs.pop(0)
        tokens = tuple(self.encode(text))[:max_new_tokens]
        return list(tokens), FakeCache(cache.token_ids + tokens)


def tool(name):
    return {
        "type": "function",
        "function": {"name": name, "description": name, "parameters": {"type": "object"}},
    }


def test_session_store_hit_replay_rollback_commit_and_lru():
    store = RecurrentSessionStore(max_sessions=2)

    def advance(tokens, cache=None):
        return FakeCache((() if cache is None else cache.token_ids) + tuple(tokens))

    first = store.prepare("a", [1, 2], advance)
    assert first.cache_event == "miss"
    store.discard(first)
    assert len(store) == 0
    assert store.commit(first, FakeCache((1, 2, 3))) is True

    hit = store.prepare("a", [1, 2, 3, 4], advance)
    assert hit.cache_event == "hit"
    assert hit.suffix_tokens == 1
    assert hit.prompt_cache.token_ids == (1, 2, 3, 4)
    assert store.commit(hit, FakeCache((1, 2, 3, 4, 5))) is True

    replay = store.prepare("a", [1, 9], advance)
    assert replay.cache_event == "replay"
    assert replay.suffix_tokens == 2

    stale = store.prepare("a", [1, 2, 3, 4, 5, 6], advance)
    concurrent = store.prepare("a", [1, 2, 3, 4, 5, 7], advance)
    assert store.commit(concurrent, FakeCache((1, 2, 3, 4, 5, 7, 8))) is True
    assert store.commit(stale, FakeCache((1, 2, 3, 4, 5, 6, 8))) is False

    for name in ("b", "c"):
        prepared = store.prepare(name, [10], advance)
        assert store.commit(prepared, FakeCache((10, 11))) is True
    assert len(store) == 2


def test_system_override_removes_harness_runtime_suffix():
    engine = FakeEngine(["Final: done" + RS])
    service = AgentCompletionService(engine, system_prompt_override="canonical system")
    service.prepare_completion(
        {
            "messages": [
                {"role": "system", "content": "canonical system\nCurrent working directory: /tmp/task"},
                {"role": "user", "content": "Do it."},
            ]
        },
        session_id=None,
    )
    consumed = bytes(engine.advance_calls[0][1]).decode("utf-8")
    assert consumed.startswith("System:\ncanonical system\n\nUser:")
    assert "working directory" not in consumed


def test_completion_round_trip_uses_cached_suffix_and_structured_tool_call():
    engine = FakeEngine([
        'Action: calculator\nArguments: {"expression":"2 + 3"}',
        "Final: 2 + 3 = 5." + RS,
    ])
    service = AgentCompletionService(engine, max_sessions=2)
    tools = [tool("calculator")]
    first_request = {
        "model": "e97-dense-agent",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Calculate 2 + 3."},
        ],
        "tools": tools,
        "temperature": 0,
        "max_tokens": 64,
    }

    first = service.prepare_completion(first_request, session_id="pi-session")
    assert first.diagnostics["x-emender-cache"] == "miss"
    message = first.response["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"] == {
        "name": "calculator",
        "arguments": '{"expression":"2 + 3"}',
    }
    assert first.response["choices"][0]["finish_reason"] == "tool_calls"
    assert len(service.sessions) == 0
    assert service.commit(first) is True

    second_request = {
        "model": "e97-dense-agent",
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Calculate 2 + 3."},
            {"role": "assistant", "content": None, "tool_calls": message["tool_calls"]},
            {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": '{"value":"5"}'},
        ],
        "tools": tools,
    }
    second = service.prepare_completion(second_request, session_id="pi-session")
    assert second.diagnostics["x-emender-cache"] == "hit"
    assert 0 < second.session.suffix_tokens < second.response["usage"]["prompt_tokens"]
    assert second.response["choices"][0]["message"]["content"] == "Final: 2 + 3 = 5."
    assert second.response["choices"][0]["finish_reason"] == "stop"
    assert service.commit(second) is True


def test_generated_error_trace_is_explicit_opt_in():
    request = {"messages": [{"role": "user", "content": "Do it."}]}
    ordinary = AgentCompletionService(FakeEngine(["secret malformed output" + RS]))
    with pytest.raises(AgentProtocolError) as ordinary_error:
        ordinary.prepare_completion(request, session_id=None)
    assert "secret" not in str(ordinary_error.value)

    traced = AgentCompletionService(
        FakeEngine(["secret malformed output" + RS]), trace_generated_errors=True
    )
    with pytest.raises(AgentProtocolError, match="generated_prefix=secret"):
        traced.prepare_completion(request, session_id=None)


def test_repeated_tool_call_cycle_fails_before_generation():
    repeated = {
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression":"5"}'},
        }],
    }
    service = AgentCompletionService(FakeEngine(["Final: unreachable" + RS]))
    with pytest.raises(AgentProtocolError, match="repeated tool call cycle"):
        service.prepare_completion(
            {
                "messages": [
                    {"role": "user", "content": "calculate"},
                    repeated,
                    {"role": "tool", "content": "error"},
                    repeated,
                    {"role": "tool", "content": "error"},
                ]
            },
            session_id="cycle",
        )
    assert len(service.sessions) == 0
    assert service.engine.outputs == ["Final: unreachable" + RS]


def test_unknown_tool_and_malformed_generation_do_not_commit():
    for output, match in (
        ('Action: shell\nArguments: {"command":"id"}' + RS, "unknown tool"),
        ("I refuse." + RS, "must begin"),
    ):
        service = AgentCompletionService(FakeEngine([output]))
        with pytest.raises(AgentProtocolError, match=match):
            service.prepare_completion(
                {"messages": [{"role": "user", "content": "Do it."}], "tools": [tool("read")]},
                session_id="unsafe",
            )
        assert len(service.sessions) == 0


def test_sse_has_role_payload_finish_and_done_events():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS]))
    completion = service.prepare_completion(
        {"messages": [{"role": "user", "content": "Do it."}]},
        session_id=None,
    )
    events = chat_completion_sse(completion.response)
    assert b'"role":"assistant"' in events[0]
    assert b'"content":"Final: done"' in events[1]
    assert b'"finish_reason":"stop"' in events[2]
    assert events[-1] == b"data: [DONE]\n\n"


def run_test_server(service, *, api_key=None):
    server = HTTPServer(("127.0.0.1", 0), make_openai_handler(service, api_key=api_key))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_http_models_nonstream_stream_auth_and_transaction_commit():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS, "Final: again" + RS]))
    server, thread = run_test_server(service, api_key="secret")
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/v1/models")
        assert connection.getresponse().status == 401

        headers = {"Authorization": "Bearer secret"}
        connection.request("GET", "/v1/models", headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        assert b"e97-dense-agent" in response.read()

        body = '{"model":"e97-dense-agent","messages":[{"role":"user","content":"Do it."}]}'
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={**headers, "Content-Type": "application/json", "x-session-id": "http-test"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("x-emender-cache") == "miss"
        assert b'"content":"Final: done"' in response.read()
        assert len(service.sessions) == 1

        stream_body = '{"model":"e97-dense-agent","stream":true,"messages":[{"role":"user","content":"Again."}]}'
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=stream_body,
            headers={**headers, "Content-Type": "application/json", "x-session-id": "http-test"},
        )
        response = connection.getresponse()
        payload = response.read()
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream"
        assert b'"content":"Final: again"' in payload
        assert payload.endswith(b"data: [DONE]\n\n")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_request_limits_fail_closed():
    service = AgentCompletionService(FakeEngine(["Final: done" + RS]), max_output_tokens=16)
    base = {"messages": [{"role": "user", "content": "Do it."}]}
    for update in (
        {"max_tokens": 0},
        {"max_tokens": "10"},
        {"temperature": -1},
        {"top_p": 0},
        {"stream": "yes"},
        {"model": "other"},
    ):
        with pytest.raises(AgentProtocolError):
            service.prepare_completion({**base, **update}, session_id=None)
