"""Versioned tokenized byte-window datasets.

The scientific sampler is counter based: a sample is a pure function of the
frozen sampler/corpus/tokenizer identity, data world, rank, absolute per-rank
sample index, and bounded retry index.  ``legacy`` mode remains available only
so historical callers can be loaded without being relabelled.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import mmap
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import tiktoken


COUNTER_SAMPLER_SCHEMA = "emender-byte-window-counter-v1"
LEGACY_SAMPLER_SCHEMA = "legacy-mutable-rng-v0"
DEFAULT_MAX_RETRIES = 64


@dataclass(frozen=True)
class CounterSamplerIdentity:
    """Immutable identity shared by every model consuming a token stream."""

    schema: str
    corpus_sha256: str
    tokenizer_sha256: str
    sampler_key: int
    data_world_size: int
    context_size: int

    def __post_init__(self) -> None:
        if self.schema != COUNTER_SAMPLER_SCHEMA:
            raise ValueError(
                f"unsupported sampler schema {self.schema!r}; "
                f"expected {COUNTER_SAMPLER_SCHEMA!r}")
        for name in ("corpus_sha256", "tokenizer_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.sampler_key < 0:
            raise ValueError("sampler_key must be nonnegative")
        if self.data_world_size <= 0:
            raise ValueError("data_world_size must be positive")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "CounterSamplerIdentity":
        required = {field.name for field in cls.__dataclass_fields__.values()}
        observed = set(metadata)
        if observed != required:
            raise ValueError(
                "sampler identity fields mismatch: "
                f"missing={sorted(required - observed)}, "
                f"unexpected={sorted(observed - required)}")
        return cls(**{name: metadata[name] for name in required})

    def assert_matches(self, expected: "CounterSamplerIdentity") -> None:
        if self != expected:
            differences = [
                name for name in self.__dataclass_fields__
                if getattr(self, name) != getattr(expected, name)
            ]
            raise ValueError(
                "sampler identity mismatch in fields: " + ", ".join(differences))


def absolute_rank_sample_index(
    total_accepted_tokens: int,
    *,
    data_world_size: int,
    context_size: int,
) -> int:
    """Convert the accepted-token authority into an exact per-rank cursor."""
    if total_accepted_tokens < 0:
        raise ValueError("total_accepted_tokens must be nonnegative")
    if data_world_size <= 0 or context_size <= 0:
        raise ValueError("data_world_size and context_size must be positive")
    tokens_per_global_sample = data_world_size * context_size
    cursor, remainder = divmod(total_accepted_tokens, tokens_per_global_sample)
    if remainder:
        raise ValueError(
            "accepted-token cursor is not exactly divisible by data world and "
            f"context: {total_accepted_tokens} % {tokens_per_global_sample} "
            f"= {remainder}")
    return cursor


def sampler_checkpoint_metadata(
    identity: CounterSamplerIdentity,
    *,
    total_accepted_tokens: int,
) -> dict[str, Any]:
    """Return JSON-safe, fail-closed checkpoint metadata for one authority."""
    cursor = absolute_rank_sample_index(
        total_accepted_tokens,
        data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    return {
        "identity": identity.to_metadata(),
        "total_accepted_tokens": int(total_accepted_tokens),
        "absolute_rank_sample_index": cursor,
    }


def restore_sampler_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_identity: CounterSamplerIdentity,
) -> tuple[CounterSamplerIdentity, int, int]:
    """Validate persisted metadata before any dataset/model state is mutated."""
    required = {"identity", "total_accepted_tokens", "absolute_rank_sample_index"}
    if set(metadata) != required:
        raise ValueError("sampler checkpoint metadata fields mismatch")
    identity = CounterSamplerIdentity.from_metadata(metadata["identity"])
    identity.assert_matches(expected_identity)
    accepted_tokens = int(metadata["total_accepted_tokens"])
    cursor = absolute_rank_sample_index(
        accepted_tokens,
        data_world_size=identity.data_world_size,
        context_size=identity.context_size,
    )
    if int(metadata["absolute_rank_sample_index"]) != cursor:
        raise ValueError("stored sampler cursor disagrees with accepted-token clock")
    return identity, accepted_tokens, cursor


class TokenizedStreamDataset(Dataset):
    """Mmap a corpus and produce fixed-length token tensors.

    Passing ``sampler_identity`` selects the scientific counter sampler and
    requires an accepted-token authority. Omitting it selects explicitly named
    legacy mutable-RNG behavior for historical compatibility only.
    """

    BYTES_PER_TOKEN_SAFETY = 6

    def __init__(
        self,
        data_path: str,
        chunk_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        tokenizer_name: str = "gpt2",
        *,
        sampler_identity: Optional[CounterSamplerIdentity] = None,
        total_accepted_tokens: Optional[int] = None,
        accepted_tokens_per_sample: Optional[int] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.chunk_size = int(chunk_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        self.max_retries = int(max_retries)
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab

        self.data_path = str(Path(data_path))
        self.data_file = open(self.data_path, "rb")
        try:
            self.mmap = mmap.mmap(
                self.data_file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self.data_file.close()
            raise
        self.file_size = len(self.mmap)

        self.sampler_identity = sampler_identity
        if sampler_identity is None:
            if total_accepted_tokens is not None:
                self.close()
                raise ValueError(
                    "total_accepted_tokens requires a counter sampler identity")
            self.sampler_schema = LEGACY_SAMPLER_SCHEMA
            self.rng = np.random.RandomState(seed + rank)
            self.initial_absolute_rank_sample_index = None
            self.next_absolute_rank_sample_index = None
        else:
            try:
                if sampler_identity.data_world_size != self.world_size:
                    raise ValueError("dataset world_size mismatches sampler identity")
                tokens_per_sample = (
                    self.chunk_size if accepted_tokens_per_sample is None
                    else int(accepted_tokens_per_sample)
                )
                if sampler_identity.context_size != tokens_per_sample:
                    raise ValueError(
                        "accepted_tokens_per_sample mismatches sampler identity context")
                if self.chunk_size not in (tokens_per_sample, tokens_per_sample + 1):
                    raise ValueError(
                        "dataset chunk_size must equal accepted context or context + 1")
                if total_accepted_tokens is None:
                    raise ValueError(
                        "counter sampler requires total_accepted_tokens authority")
                cursor = absolute_rank_sample_index(
                    total_accepted_tokens,
                    data_world_size=self.world_size,
                    context_size=self.chunk_size,
                )
            except Exception:
                self.close()
                raise
            self.sampler_schema = sampler_identity.schema
            self.rng = None
            self.initial_absolute_rank_sample_index = cursor
            self.next_absolute_rank_sample_index = cursor

        self.chunks_served = 0
        self.bytes_processed = 0
        self.tokens_served = 0
        self.last_batch_sample_ids: tuple[str, ...] = ()

    def close(self) -> None:
        mmap_obj = getattr(self, "mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
            self.mmap = None
        data_file = getattr(self, "data_file", None)
        if data_file is not None:
            data_file.close()
            self.data_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self):
        return 1_000_000_000

    @property
    def _need_bytes(self) -> int:
        return self.chunk_size * self.BYTES_PER_TOKEN_SAFETY

    @property
    def _max_start(self) -> int:
        return max(1, self.file_size - self._need_bytes - 1)

    def _counter_digest(self, absolute_index: int, retry_index: int) -> bytes:
        if self.sampler_identity is None:
            raise RuntimeError("counter digest requested for legacy sampler")
        if absolute_index < 0 or not 0 <= retry_index < self.max_retries:
            raise ValueError("invalid absolute sample or retry index")
        payload = {
            **self.sampler_identity.to_metadata(),
            "global_rank": self.rank,
            "absolute_rank_sample_index": int(absolute_index),
            "bounded_retry_index": int(retry_index),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(encoded).digest()

    def sample_id(self, absolute_index: int, retry_index: int = 0) -> str:
        return self._counter_digest(absolute_index, retry_index).hex()

    def candidate_byte_position(self, absolute_index: int, retry_index: int) -> int:
        digest = self._counter_digest(absolute_index, retry_index)
        return int.from_bytes(digest[:8], "little") % self._max_start

    def _tokenize_position(self, pos: int) -> Optional[torch.Tensor]:
        raw = bytes(self.mmap[pos:pos + self._need_bytes])
        try:
            text = raw.decode("utf-8", errors="replace")
            tokens = self.enc.encode(text, disallowed_special=())
        except Exception:
            return None
        if len(tokens) < self.chunk_size + 2:
            return None
        tokens = tokens[1:self.chunk_size + 1]
        if len(tokens) != self.chunk_size:
            return None
        return torch.tensor(tokens, dtype=torch.long)

    def sample_at(self, absolute_index: int) -> tuple[torch.Tensor, str]:
        """Return one counter sample without mutating the accepted cursor."""
        if self.sampler_identity is None:
            raise RuntimeError("sample_at is unavailable for the legacy sampler")
        for retry_index in range(self.max_retries):
            pos = self.candidate_byte_position(absolute_index, retry_index)
            chunk = self._tokenize_position(pos)
            if chunk is not None:
                return chunk, self.sample_id(absolute_index, retry_index)
        raise RuntimeError(
            f"counter sampler exhausted {self.max_retries} retries for rank "
            f"{self.rank}, absolute sample {absolute_index}")

    def __getitem__(self, idx):
        """Return ``(tokens, False, chunk_size)`` for a relative sample index."""
        if self.sampler_identity is not None:
            absolute_index = self.initial_absolute_rank_sample_index + int(idx)
            chunk, _sample_id = self.sample_at(absolute_index)
        else:
            while True:
                pos = self.rng.randint(0, self._max_start)
                chunk = self._tokenize_position(pos)
                if chunk is not None:
                    break
        self.chunks_served += 1
        self.tokens_served += self.chunk_size
        self.bytes_processed += self._need_bytes
        return chunk, False, self.chunk_size

    def get_batch(self, batch_size: int, device=None):
        """Return a batch and advance only the provisional in-process cursor."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (not hasattr(self, "_pinned_chunks")
                or self._pinned_chunks.shape[0] != batch_size):
            pin_memory = torch.cuda.is_available()
            self._pinned_chunks = torch.empty(
                batch_size, self.chunk_size, dtype=torch.long,
                pin_memory=pin_memory)
            self._pinned_doc_ends = torch.empty(
                batch_size, dtype=torch.bool, pin_memory=pin_memory)
            self._pinned_lengths = torch.empty(
                batch_size, dtype=torch.long, pin_memory=pin_memory)

        sample_ids = []
        for i in range(batch_size):
            if self.sampler_identity is None:
                chunk, is_doc_end, actual_length = self[0]
            else:
                absolute_index = self.next_absolute_rank_sample_index + i
                chunk, sample_id = self.sample_at(absolute_index)
                sample_ids.append(sample_id)
                is_doc_end, actual_length = False, self.chunk_size
                self.chunks_served += 1
                self.tokens_served += self.chunk_size
                self.bytes_processed += self._need_bytes
            self._pinned_chunks[i] = chunk
            self._pinned_doc_ends[i] = is_doc_end
            self._pinned_lengths[i] = actual_length

        if self.sampler_identity is not None:
            self.next_absolute_rank_sample_index += batch_size
            self.last_batch_sample_ids = tuple(sample_ids)

        if device is not None:
            return (
                self._pinned_chunks.to(device, non_blocking=True),
                self._pinned_doc_ends.to(device, non_blocking=True),
                self._pinned_lengths.to(device, non_blocking=True),
            )
        return self._pinned_chunks, self._pinned_doc_ends, self._pinned_lengths

    def get_stats(self):
        return {
            "sampler_schema": self.sampler_schema,
            "chunks_served": self.chunks_served,
            "tokens_served": self.tokens_served,
            "bytes_processed": self.bytes_processed,
            "next_absolute_rank_sample_index": self.next_absolute_rank_sample_index,
        }
