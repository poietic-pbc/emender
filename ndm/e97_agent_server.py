"""Bounded OpenAI-compatible serving core for the recurrent E97 agent."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Mapping, Protocol, Sequence

from .e97 import (
    E97RecurrentCache,
    LoadedE97Checkpoint,
    advance_e97_cache,
    advance_e97_cache_segment,
    e97_cache_suffix,
    generate_e97_from_cache,
)
from .e97_agent_protocol import (
    AgentProtocolError,
    ParsedAgentTurn,
    generated_turn_is_complete,
    parse_agent_turn,
    serialize_pi_messages,
    validate_generated_tool,
)


class AgentEngine(Protocol):
    checkpoint: str

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    def advance(self, token_ids: Sequence[int], cache: Any | None = None) -> Any: ...
    def generate(
        self,
        cache: Any,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[list[int], Any]: ...


@dataclass(frozen=True)
class PreparedSession:
    session_id: str | None
    expected_version: int | None
    prompt_cache: Any
    cache_event: str
    suffix_tokens: int


@dataclass
class SessionRecord:
    cache: Any
    version: int
    last_used: float


class RecurrentSessionStore:
    """Bounded LRU store with compare-and-commit transactional semantics."""

    def __init__(self, max_sessions: int = 8):
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self.max_sessions = max_sessions
        self._records: OrderedDict[str, SessionRecord] = OrderedDict()
        self._lock = threading.RLock()

    def prepare(
        self,
        session_id: str | None,
        requested_tokens: Sequence[int],
        advance: Callable[[Sequence[int], Any | None], Any],
    ) -> PreparedSession:
        requested = tuple(int(token) for token in requested_tokens)
        if not requested:
            raise ValueError("requested token prefix cannot be empty")
        if session_id is None:
            return PreparedSession(None, None, advance(requested, None), "uncached", len(requested))
        if not session_id or len(session_id) > 256:
            raise ValueError("session id must contain 1-256 characters")

        with self._lock:
            record = self._records.get(session_id)
            expected_version = None if record is None else record.version
            suffix = None if record is None else e97_cache_suffix(record.cache, requested)
            if suffix is None:
                prompt_cache = advance(requested, None)
                event = "miss" if record is None else "replay"
                suffix_count = len(requested)
            else:
                prompt_cache = advance(suffix, record.cache)
                event = "hit"
                suffix_count = len(suffix)
                record.last_used = time.monotonic()
                self._records.move_to_end(session_id)
            return PreparedSession(
                session_id=session_id,
                expected_version=expected_version,
                prompt_cache=prompt_cache,
                cache_event=event,
                suffix_tokens=suffix_count,
            )

    def commit(self, prepared: PreparedSession, completed_cache: Any) -> bool:
        if prepared.session_id is None:
            return False
        with self._lock:
            current = self._records.get(prepared.session_id)
            current_version = None if current is None else current.version
            if current_version != prepared.expected_version:
                return False
            next_version = 1 if current is None else current.version + 1
            self._records[prepared.session_id] = SessionRecord(
                cache=completed_cache,
                version=next_version,
                last_used=time.monotonic(),
            )
            self._records.move_to_end(prepared.session_id)
            while len(self._records) > self.max_sessions:
                self._records.popitem(last=False)
            return True

    def discard(self, prepared: PreparedSession) -> None:
        """Document rollback; preparation never mutates the committed cache."""

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class TorchE97AgentEngine:
    """Adapter from a loaded dense checkpoint to the serving core protocol."""

    def __init__(
        self,
        loaded: LoadedE97Checkpoint,
        *,
        ingest_mode: str = "tokenwise",
    ):
        import tiktoken

        if ingest_mode not in {"tokenwise", "segment"}:
            raise ValueError("ingest_mode must be tokenwise or segment")
        self.loaded = loaded
        self.ingest_mode = ingest_mode
        self.checkpoint = str(loaded.checkpoint_path)
        tokenizer_name = loaded.config.get("tokenizer")
        if not tokenizer_name:
            raise ValueError("Pi agent serving requires a named tokenizer")
        self.tokenizer = tiktoken.get_encoding(str(tokenizer_name))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def advance(
        self,
        token_ids: Sequence[int],
        cache: E97RecurrentCache | None = None,
    ) -> E97RecurrentCache:
        advance = (
            advance_e97_cache
            if self.ingest_mode == "tokenwise"
            else advance_e97_cache_segment
        )
        return advance(self.loaded, token_ids, cache)

    def generate(
        self,
        cache: E97RecurrentCache,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[list[int], E97RecurrentCache]:
        # Stop a structured action as soon as its JSON object is complete. In
        # particular, do not let the model emit or consume RS: pretraining used
        # RS between unrelated records, while a tool turn is the same task.
        shadow = cache
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_tokens, shadow = generate_e97_from_cache(
                self.loaded,
                shadow,
                max_new_tokens=1,
                temperature=temperature,
                top_k=0,
                top_p=top_p,
                stop_token_ids=(218,),
            )
            generated.extend(next_tokens)
            if generated_turn_is_complete(self.decode(generated)):
                break
            if next_tokens and next_tokens[-1] == 218:
                break
        return generated, shadow


@dataclass(frozen=True)
class PreparedCompletion:
    response: Mapping[str, Any]
    session: PreparedSession
    completed_cache: Any
    diagnostics: Mapping[str, str]


class AgentCompletionService:
    """Validate one Chat Completions request and prepare a transactional reply."""

    def __init__(
        self,
        engine: AgentEngine,
        *,
        model_id: str = "e97-dense-agent",
        max_output_tokens: int = 512,
        max_sessions: int = 8,
        trace_generated_errors: bool = False,
        system_prompt_override: str | None = None,
        require_tool_call: bool = False,
    ):
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.engine = engine
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.trace_generated_errors = bool(trace_generated_errors)
        self.system_prompt_override = system_prompt_override
        self.require_tool_call = bool(require_tool_call)
        self.sessions = RecurrentSessionStore(max_sessions=max_sessions)

    def prepare_completion(
        self,
        request: Mapping[str, Any],
        *,
        session_id: str | None,
    ) -> PreparedCompletion:
        if not isinstance(request, Mapping):
            raise AgentProtocolError("request body must be an object")
        model = request.get("model", self.model_id)
        if model != self.model_id:
            raise AgentProtocolError(f"unknown model: {model}")
        if not isinstance(request.get("stream", False), bool):
            raise AgentProtocolError("stream must be boolean")
        messages = request.get("messages")
        if not isinstance(messages, list):
            raise AgentProtocolError("messages must be an array")
        seen_tool_calls: set[tuple[str, str]] = set()
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                raise AgentProtocolError("E97 supports exactly one tool call per turn")
            function = tool_calls[0].get("function") if isinstance(tool_calls[0], Mapping) else None
            if not isinstance(function, Mapping):
                raise AgentProtocolError("assistant tool call requires function metadata")
            key = (str(function.get("name")), str(function.get("arguments")))
            if key in seen_tool_calls:
                raise AgentProtocolError("repeated tool call cycle detected")
            seen_tool_calls.add(key)
        if self.system_prompt_override is not None:
            messages = [dict(message) for message in messages]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = self.system_prompt_override
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": self.system_prompt_override,
                })
        prompt = serialize_pi_messages(messages)
        prompt_tokens = self.engine.encode(prompt)
        if not prompt_tokens:
            raise AgentProtocolError("serialized prompt has no tokens")

        requested_max = request.get("max_completion_tokens", request.get("max_tokens", 128))
        if isinstance(requested_max, bool) or not isinstance(requested_max, int):
            raise AgentProtocolError("max tokens must be an integer")
        if requested_max <= 0:
            raise AgentProtocolError("max tokens must be positive")
        max_new_tokens = min(requested_max, self.max_output_tokens)
        temperature = request.get("temperature", 0)
        top_p = request.get("top_p", 1.0)
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise AgentProtocolError("temperature must be non-negative")
        if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            raise AgentProtocolError("top_p must be in (0, 1]")

        prepared = self.sessions.prepare(session_id, prompt_tokens, self.engine.advance)
        generated_tokens, completed_cache = self.engine.generate(
            prepared.prompt_cache,
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        generated_text = self.engine.decode(generated_tokens)
        try:
            turn = parse_agent_turn(generated_text)
            if self.require_tool_call and turn.kind != "tool_call":
                raise AgentProtocolError("this agent protocol requires a structured tool call")
            validate_generated_tool(turn, request.get("tools"))
        except AgentProtocolError as exc:
            if not self.trace_generated_errors:
                raise
            escaped = generated_text[:512].encode("unicode_escape").decode("ascii")
            raise AgentProtocolError(f"{exc}; generated_prefix={escaped}") from exc

        created = int(time.time())
        completion_id = "chatcmpl-" + hashlib.sha256(
            (str(session_id) + prompt + turn.raw_text).encode("utf-8")
        ).hexdigest()[:24]
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish_reason = "stop"
        if turn.kind == "tool_call":
            call_id = "call_" + hashlib.sha256(
                (completion_id + turn.raw_text).encode("utf-8")
            ).hexdigest()[:24]
            message["tool_calls"] = [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": turn.tool_name,
                    "arguments": turn.arguments_json,
                },
            }]
            finish_reason = "tool_calls"
        else:
            message["content"] = turn.raw_text

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(prompt_tokens) + len(generated_tokens),
            },
        }
        state_bytes = getattr(completed_cache, "state_bytes", 0)
        diagnostics = {
            "x-emender-cache": prepared.cache_event,
            "x-emender-suffix-tokens": str(prepared.suffix_tokens),
            "x-emender-state-bytes": str(state_bytes),
        }
        return PreparedCompletion(response, prepared, completed_cache, diagnostics)

    def commit(self, completion: PreparedCompletion) -> bool:
        return self.sessions.commit(completion.session, completion.completed_cache)

    def discard(self, completion: PreparedCompletion) -> None:
        self.sessions.discard(completion.session)


def chat_completion_sse(response: Mapping[str, Any]) -> list[bytes]:
    """Convert one prepared response to OpenAI Chat Completions SSE events."""

    choice = response["choices"][0]
    message = choice["message"]
    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }

    def event(delta: Mapping[str, Any], finish_reason: str | None) -> bytes:
        payload = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"

    events = [event({"role": "assistant"}, None)]
    if message.get("tool_calls"):
        call = message["tool_calls"][0]
        events.append(event({"tool_calls": [{
            "index": 0,
            "id": call["id"],
            "type": "function",
            "function": call["function"],
        }]}, None))
    else:
        events.append(event({"content": message.get("content", "")}, None))
    events.append(event({}, choice["finish_reason"]))
    events.append(b"data: [DONE]\n\n")
    return events


def make_openai_handler(
    service: AgentCompletionService,
    *,
    api_key: str | None = None,
    max_body_bytes: int = 4 * 1024 * 1024,
) -> type[BaseHTTPRequestHandler]:
    """Create a single-service OpenAI-compatible HTTP handler."""

    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive")

    class OpenAIHandler(BaseHTTPRequestHandler):
        server_version = "EmenderE97/1"

        def _authorized(self) -> bool:
            return api_key is None or self.headers.get("Authorization") == f"Bearer {api_key}"

        def _json(
            self,
            status: int,
            payload: Mapping[str, Any],
            headers: Mapping[str, str] | None = None,
        ) -> bool:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        def _error(self, status: int, message: str, error_type: str) -> None:
            self._json(status, {"error": {"message": message, "type": error_type}})

        def do_GET(self) -> None:
            if not self._authorized():
                self._error(401, "invalid API key", "authentication_error")
                return
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": service.model_id})
                return
            if self.path == "/v1/models":
                self._json(200, {
                    "object": "list",
                    "data": [{
                        "id": service.model_id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "emender",
                    }],
                })
                return
            self._error(404, "not found", "not_found_error")

        def do_POST(self) -> None:
            if not self._authorized():
                self._error(401, "invalid API key", "authentication_error")
                return
            if self.path != "/v1/chat/completions":
                self._error(404, "not found", "not_found_error")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "invalid Content-Length", "invalid_request_error")
                return
            if length <= 0 or length > max_body_bytes:
                self._error(413 if length > max_body_bytes else 400, "invalid request body size", "invalid_request_error")
                return
            try:
                request = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._error(400, "request body is not valid JSON", "invalid_request_error")
                return

            session_id = next((
                self.headers.get(name)
                for name in (
                    "x-session-id",
                    "session_id",
                    "x-session-affinity",
                    "x-client-request-id",
                )
                if self.headers.get(name)
            ), None)
            try:
                completion = service.prepare_completion(request, session_id=session_id)
            except AgentProtocolError as exc:
                self._error(422, str(exc), "agent_protocol_error")
                return
            except (TypeError, ValueError) as exc:
                self._error(400, str(exc), "invalid_request_error")
                return
            except Exception as exc:
                self.log_error("completion failed: %s", exc)
                self._error(500, "model completion failed", "server_error")
                return

            self.log_message(
                "completion cache=%s suffix_tokens=%s state_bytes=%s",
                completion.diagnostics["x-emender-cache"],
                completion.diagnostics["x-emender-suffix-tokens"],
                completion.diagnostics["x-emender-state-bytes"],
            )
            stream = request.get("stream", False)
            if not isinstance(stream, bool):
                service.discard(completion)
                self._error(400, "stream must be boolean", "invalid_request_error")
                return
            delivered = False
            if stream:
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    for name, value in completion.diagnostics.items():
                        self.send_header(name, value)
                    self.end_headers()
                    for event in chat_completion_sse(completion.response):
                        self.wfile.write(event)
                    self.wfile.flush()
                    delivered = True
                except (BrokenPipeError, ConnectionResetError):
                    delivered = False
            else:
                delivered = self._json(200, completion.response, completion.diagnostics)

            if delivered:
                service.commit(completion)
            else:
                service.discard(completion)

        def log_message(self, format: str, *args: Any) -> None:
            # Never log request bodies or model-generated arguments here.
            super().log_message(format, *args)

    return OpenAIHandler


def run_openai_server(
    service: AgentCompletionService,
    *,
    host: str = "127.0.0.1",
    port: int = 8797,
    api_key: str | None = None,
    max_body_bytes: int = 4 * 1024 * 1024,
) -> None:
    """Run the bounded single-request-at-a-time model server."""

    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("non-loopback serving requires --api-key")
    handler = make_openai_handler(
        service,
        api_key=api_key,
        max_body_bytes=max_body_bytes,
    )
    HTTPServer((host, port), handler).serve_forever()
