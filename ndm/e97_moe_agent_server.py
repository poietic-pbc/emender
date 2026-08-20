"""Synchronized eight-rank serving adapter for node-local E97 MoE.

Rank zero owns the HTTP/Pi service.  Every model operation is broadcast to the
seven worker ranks so all eight ranks enter expert-parallel collectives in the
same order.  This first authority deliberately serves one request at a time.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import torch
import torch.distributed as dist

from .e97 import (
    E97RecurrentCache,
    LoadedE97Checkpoint,
    _sample_token,
    advance_e97_cache,
    advance_e97_cache_segment,
)
from .e97_agent_protocol import AgentProtocolError, parse_agent_turn


class TorchE97MoEAgentEngine:
    """Collective-safe AgentEngine for one eight-GCD MoE island."""

    def __init__(
        self,
        loaded: LoadedE97Checkpoint,
        *,
        node_group,
        ingest_mode: str = "segment",
        worker_cache_limit: int = 32,
    ) -> None:
        import tiktoken

        if ingest_mode not in {"tokenwise", "segment"}:
            raise ValueError("ingest_mode must be tokenwise or segment")
        if dist.get_world_size(node_group) != 8:
            raise RuntimeError("MoE agent serving requires one eight-rank EP group")
        if worker_cache_limit <= 0:
            raise ValueError("worker_cache_limit must be positive")
        tokenizer_name = loaded.config.get("tokenizer")
        if not tokenizer_name:
            raise ValueError("MoE agent serving requires a named tokenizer")
        self.loaded = loaded
        self.node_group = node_group
        self.ingest_mode = ingest_mode
        self.worker_cache_limit = int(worker_cache_limit)
        self.checkpoint = str(loaded.checkpoint_path)
        self.tokenizer = tiktoken.get_encoding(str(tokenizer_name))
        self._worker_caches: OrderedDict[tuple[int, ...], E97RecurrentCache] = OrderedDict()

    @property
    def is_coordinator(self) -> bool:
        return dist.get_rank(self.node_group) == 0

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids))

    def _broadcast_command(self, command: dict[str, Any]) -> None:
        if not self.is_coordinator:
            raise RuntimeError("only rank zero may issue MoE serving commands")
        payload: list[Any] = [command]
        dist.broadcast_object_list(payload, src=0, group=self.node_group)

    def _advance_local(
        self,
        consumed: Sequence[int],
        cache: E97RecurrentCache | None,
    ) -> E97RecurrentCache:
        function = advance_e97_cache if self.ingest_mode == "tokenwise" else advance_e97_cache_segment
        return function(self.loaded, consumed, cache)

    def advance(
        self,
        token_ids: Sequence[int],
        cache: E97RecurrentCache | None = None,
    ) -> E97RecurrentCache:
        consumed = tuple(int(token) for token in token_ids)
        parent = None if cache is None else cache.token_ids
        self._broadcast_command({"op": "advance", "tokens": consumed, "parent": parent})
        return self._advance_local(consumed, cache)

    def _broadcast_token(self, token: int) -> None:
        value = torch.tensor([int(token)], dtype=torch.int64, device=next(self.loaded.model.parameters()).device)
        dist.broadcast(value, src=0, group=self.node_group)

    def _broadcast_finished(self, finished: bool) -> None:
        value = torch.tensor([1 if finished else 0], dtype=torch.int32, device=next(self.loaded.model.parameters()).device)
        dist.broadcast(value, src=0, group=self.node_group)

    def generate(
        self,
        cache: E97RecurrentCache,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[list[int], E97RecurrentCache]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        self._broadcast_command({"op": "generate", "parent": cache.token_ids, "max_new_tokens": max_new_tokens})
        shadow = cache
        generated: list[int] = []
        for _ in range(max_new_tokens):
            token = _sample_token(
                shadow.next_logits,
                temperature=float(temperature),
                top_k=0,
                top_p=float(top_p),
            )
            self._broadcast_token(token)
            generated.append(token)
            shadow = self._advance_local((token,), shadow)
            try:
                turn = parse_agent_turn(self.decode(generated))
            except AgentProtocolError:
                turn = None
            finished = bool(token == 218 or (turn is not None and turn.kind == "tool_call"))
            self._broadcast_finished(finished)
            if finished:
                break
        return generated, shadow

    def _remember_worker_cache(self, cache: E97RecurrentCache) -> None:
        self._worker_caches[cache.token_ids] = cache
        self._worker_caches.move_to_end(cache.token_ids)
        while len(self._worker_caches) > self.worker_cache_limit:
            self._worker_caches.popitem(last=False)

    def _worker_parent(self, token_ids: Sequence[int] | None) -> E97RecurrentCache | None:
        if token_ids is None:
            return None
        key = tuple(int(token) for token in token_ids)
        try:
            cache = self._worker_caches[key]
        except KeyError as error:
            raise RuntimeError("worker lacks the requested recurrent-cache lineage") from error
        self._worker_caches.move_to_end(key)
        return cache

    def worker_loop(self) -> None:
        """Run on ranks 1-7 until rank zero broadcasts ``stop``."""
        if self.is_coordinator:
            raise RuntimeError("rank zero cannot enter the MoE worker loop")
        while True:
            payload: list[Any] = [None]
            dist.broadcast_object_list(payload, src=0, group=self.node_group)
            command = payload[0]
            if not isinstance(command, dict):
                raise RuntimeError("invalid MoE serving command")
            operation = command.get("op")
            if operation == "stop":
                return
            if operation == "advance":
                parent = self._worker_parent(command.get("parent"))
                cache = self._advance_local(command["tokens"], parent)
                self._remember_worker_cache(cache)
                continue
            if operation != "generate":
                raise RuntimeError(f"unknown MoE serving operation: {operation!r}")
            shadow = self._worker_parent(command.get("parent"))
            if shadow is None:
                raise RuntimeError("generation requires an existing worker cache")
            for _ in range(int(command["max_new_tokens"])):
                token_tensor = torch.empty(1, dtype=torch.int64, device=next(self.loaded.model.parameters()).device)
                dist.broadcast(token_tensor, src=0, group=self.node_group)
                shadow = self._advance_local((int(token_tensor.item()),), shadow)
                finished_tensor = torch.empty(1, dtype=torch.int32, device=token_tensor.device)
                dist.broadcast(finished_tensor, src=0, group=self.node_group)
                if bool(finished_tensor.item()):
                    break
            self._remember_worker_cache(shadow)

    def stop_workers(self) -> None:
        if self.is_coordinator:
            self._broadcast_command({"op": "stop"})
