"""Immutable record-aware packing and counter sampling for masked SFT."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import mmap
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np
import torch


AUTHORITY_SCHEMA = "emender-e97-tulu3-masked-sft-v1"
PACK_SCHEMA = "emender-e97-sft-complete-record-packs-v1"
SAMPLER_SCHEMA = "emender-record-pack-counter-v1"
RECORD_INDEX = struct.Struct("<QQQB7x")
PACK_INDEX = struct.Struct("<QQQQ")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class SFTSamplerIdentity:
    authority_manifest_sha256: str
    pack_manifest_sha256: str
    sampler_key: int
    data_world_size: int
    context_size: int
    split: str = "train"
    schema: str = SAMPLER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SAMPLER_SCHEMA:
            raise ValueError(f"unsupported SFT sampler schema {self.schema!r}")
        _digest(self.authority_manifest_sha256, "authority_manifest_sha256")
        _digest(self.pack_manifest_sha256, "pack_manifest_sha256")
        if self.sampler_key < 0 or self.data_world_size <= 0 or self.context_size <= 0:
            raise ValueError("SFT sampler key/world/context are invalid")
        if self.split not in {"train", "validation"}:
            raise ValueError("SFT split must be train or validation")

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "SFTSamplerIdentity":
        required = {
            "authority_manifest_sha256", "pack_manifest_sha256", "sampler_key",
            "data_world_size", "context_size", "split", "schema",
        }
        if set(metadata) != required:
            raise ValueError("SFT sampler identity fields mismatch")
        return cls(**{key: metadata[key] for key in required})


def sft_checkpoint_metadata(
    identity: SFTSamplerIdentity,
    *,
    parent: Mapping[str, Any],
    total_tokens: int,
    assistant_target_tokens: int,
    absolute_rank_sample_index: int,
) -> dict[str, Any]:
    required_parent = {"manifest_sha256", "step", "accepted_tokens", "generation"}
    if set(parent) != required_parent:
        raise ValueError("SFT parent metadata fields mismatch")
    _digest(str(parent["manifest_sha256"]), "parent manifest_sha256")
    values = (int(total_tokens), int(assistant_target_tokens),
              int(absolute_rank_sample_index))
    if min(values) < 0 or values[1] > values[0]:
        raise ValueError("invalid SFT token or sampler clock")
    return {
        "identity": identity.to_metadata(),
        "parent": dict(parent),
        "total_tokens": values[0],
        "assistant_target_tokens": values[1],
        "absolute_rank_sample_index": values[2],
    }


def restore_sft_checkpoint_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_identity: SFTSamplerIdentity,
    expected_parent: Mapping[str, Any],
    model_accepted_tokens: int,
) -> tuple[int, int, int]:
    required = {
        "identity", "parent", "total_tokens", "assistant_target_tokens",
        "absolute_rank_sample_index",
    }
    if set(metadata) != required:
        raise ValueError("SFT checkpoint metadata fields mismatch")
    observed = SFTSamplerIdentity.from_metadata(metadata["identity"])
    if observed != expected_identity:
        raise ValueError("SFT sampler identity mismatch")
    if dict(metadata["parent"]) != dict(expected_parent):
        raise ValueError("SFT parent authority mismatch")
    total = int(metadata["total_tokens"])
    targets = int(metadata["assistant_target_tokens"])
    cursor = int(metadata["absolute_rank_sample_index"])
    if min(total, targets, cursor) < 0 or targets > total:
        raise ValueError("invalid persisted SFT clocks")
    expected_model_tokens = int(expected_parent["accepted_tokens"]) + total
    if int(model_accepted_tokens) != expected_model_tokens:
        raise ValueError("SFT total-token clock contradicts model accepted tokens")
    return total, targets, cursor


class MaskedSFTPackedDataset:
    """Mmap immutable complete-record packs and sample pack IDs with replacement."""

    def __init__(
        self,
        authority_root: str | Path,
        pack_root: str | Path,
        *,
        identity: SFTSamplerIdentity,
        rank: int,
        initial_absolute_rank_sample_index: int = 0,
        verify_payload_hashes: bool = False,
        pad_token_id: int = 0,
    ) -> None:
        self.authority_root = Path(authority_root)
        self.pack_root = Path(pack_root)
        self.identity = identity
        self.rank = int(rank)
        self.context_size = int(identity.context_size)
        self.sequence_tokens = self.context_size + 1
        self.pad_token_id = int(pad_token_id)
        if not 0 <= self.rank < identity.data_world_size:
            raise ValueError("rank must be within the SFT data world")
        if initial_absolute_rank_sample_index < 0:
            raise ValueError("initial SFT sampler cursor must be nonnegative")

        authority_manifest_path = self.authority_root / "manifest.json"
        pack_manifest_path = self.pack_root / "manifest.json"
        if sha256(authority_manifest_path) != identity.authority_manifest_sha256:
            raise RuntimeError("SFT authority manifest digest mismatch")
        if sha256(pack_manifest_path) != identity.pack_manifest_sha256:
            raise RuntimeError("SFT pack manifest digest mismatch")
        self.authority_manifest = json.loads(authority_manifest_path.read_text())
        self.pack_manifest = json.loads(pack_manifest_path.read_text())
        if self.authority_manifest.get("schema") != AUTHORITY_SCHEMA:
            raise RuntimeError("unsupported SFT token authority schema")
        if self.pack_manifest.get("schema") != PACK_SCHEMA:
            raise RuntimeError("unsupported SFT pack authority schema")
        if (self.pack_manifest.get("authority_manifest_sha256")
                != identity.authority_manifest_sha256
                or int(self.pack_manifest.get("context_size", -1)) != self.context_size):
            raise RuntimeError("SFT pack authority binding mismatch")

        outputs = self.authority_manifest["outputs"]
        pack_outputs = self.pack_manifest["outputs"]
        paths = {
            "tokens": self.authority_root / Path(outputs["tokens"]["path"]).name,
            "mask": self.authority_root / Path(outputs["mask"]["path"]).name,
            "records": self.authority_root / Path(outputs["index"]["path"]).name,
            "pack_records": self.pack_root / Path(pack_outputs["pack_records"]["path"]).name,
            "packs": self.pack_root / Path(pack_outputs[f"{identity.split}_index"]["path"]).name,
        }
        infos = {
            "tokens": outputs["tokens"], "mask": outputs["mask"],
            "records": outputs["index"], "pack_records": pack_outputs["pack_records"],
            "packs": pack_outputs[f"{identity.split}_index"],
        }
        for name, path in paths.items():
            if not path.is_file() or path.stat().st_size != int(infos[name]["bytes"]):
                raise RuntimeError(f"SFT payload size mismatch: {name}")
            if verify_payload_hashes and sha256(path) != infos[name]["sha256"]:
                raise RuntimeError(f"SFT payload digest mismatch: {name}")

        self._files = {name: path.open("rb") for name, path in paths.items()}
        self._maps = {
            name: mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            for name, handle in self._files.items()
        }
        self.tokens = np.ndarray(
            (infos["tokens"]["bytes"] // 4,), dtype="<u4", buffer=self._maps["tokens"])
        self.masks = np.ndarray(
            (infos["mask"]["bytes"],), dtype="u1", buffer=self._maps["mask"])
        self.records = np.ndarray(
            (infos["records"]["bytes"] // RECORD_INDEX.size,),
            dtype=np.dtype([("offset", "<u8"), ("tokens", "<u8"),
                            ("targets", "<u8"), ("split", "u1"), ("pad", "V7")]),
            buffer=self._maps["records"])
        self.pack_record_ids = np.ndarray(
            (infos["pack_records"]["bytes"] // 4,), dtype="<u4",
            buffer=self._maps["pack_records"])
        self.packs = np.ndarray(
            (infos["packs"]["bytes"] // PACK_INDEX.size,),
            dtype=np.dtype([("record_offset", "<u8"), ("record_count", "<u8"),
                            ("tokens", "<u8"), ("targets", "<u8")]),
            buffer=self._maps["packs"])
        if len(self.packs) != int(self.pack_manifest["splits"][identity.split]["packs"]):
            raise RuntimeError("SFT pack count contradicts manifest")
        if len(self.packs) == 0:
            raise RuntimeError("SFT pack authority contains no packs")
        self.initial_absolute_rank_sample_index = int(initial_absolute_rank_sample_index)
        self.next_absolute_rank_sample_index = int(initial_absolute_rank_sample_index)
        self.last_batch_sample_ids: tuple[str, ...] = ()

    def close(self) -> None:
        for array_name in ("tokens", "masks", "records", "pack_record_ids", "packs"):
            if hasattr(self, array_name):
                delattr(self, array_name)
        for mapping in getattr(self, "_maps", {}).values():
            mapping.close()
        for handle in getattr(self, "_files", {}).values():
            handle.close()
        self._maps = {}
        self._files = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _digest(self, absolute_index: int) -> bytes:
        if absolute_index < 0:
            raise ValueError("absolute SFT sample index must be nonnegative")
        payload = {
            **self.identity.to_metadata(), "global_rank": self.rank,
            "absolute_rank_sample_index": int(absolute_index),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).digest()

    def pack_id_at(self, absolute_index: int) -> int:
        return int.from_bytes(self._digest(absolute_index)[:8], "little") % len(self.packs)

    def sample_id(self, absolute_index: int) -> str:
        return self._digest(absolute_index).hex()

    def pack_at(self, pack_id: int) -> tuple[torch.Tensor, torch.Tensor, int, int, str]:
        """Materialize one authority pack by exact index without counter sampling."""
        if not 0 <= int(pack_id) < len(self.packs):
            raise IndexError("SFT pack index is out of range")
        pack_id = int(pack_id)
        pack = self.packs[pack_id]
        record_start = int(pack["record_offset"])
        record_stop = record_start + int(pack["record_count"])
        record_ids = self.pack_record_ids[record_start:record_stop]
        token_count = int(pack["tokens"])
        target_count = int(pack["targets"])
        if not 0 < token_count <= self.sequence_tokens or len(record_ids) == 0:
            raise RuntimeError("invalid complete-record pack descriptor")
        token_out = torch.full((self.sequence_tokens,), self.pad_token_id, dtype=torch.long)
        mask_out = torch.zeros((self.sequence_tokens,), dtype=torch.bool)
        cursor = 0
        observed_targets = 0
        expected_split = 0 if self.identity.split == "train" else 1
        for record_id_value in record_ids:
            record = self.records[int(record_id_value)]
            count = int(record["tokens"])
            source = int(record["offset"])
            if int(record["split"]) != expected_split or cursor + count > token_count:
                raise RuntimeError("pack record membership contradicts authority")
            token_out[cursor:cursor + count].copy_(
                torch.from_numpy(self.tokens[source:source + count].astype(np.int64)))
            record_mask = torch.from_numpy(
                self.masks[source:source + count].astype(np.bool_))
            mask_out[cursor:cursor + count].copy_(record_mask)
            observed_targets += int(record_mask.sum())
            cursor += count
        if cursor != token_count or observed_targets != target_count:
            raise RuntimeError("materialized pack accounting mismatch")
        return token_out, mask_out, token_count, target_count, f"pack-{pack_id:08d}"

    def record_spans_at(self, pack_id: int) -> tuple[tuple[int, int], ...]:
        """Return exact half-open record spans for one immutable pack."""
        if not 0 <= int(pack_id) < len(self.packs):
            raise IndexError("SFT pack index is out of range")
        pack = self.packs[int(pack_id)]
        first = int(pack["record_offset"])
        record_ids = self.pack_record_ids[first:first + int(pack["record_count"])]
        spans = []
        cursor = 0
        for record_id_value in record_ids:
            count = int(self.records[int(record_id_value)]["tokens"])
            spans.append((cursor, cursor + count))
            cursor += count
        if cursor != int(pack["tokens"]) or not spans:
            raise RuntimeError("pack record spans contradict authority")
        return tuple(spans)

    def sample_at(self, absolute_index: int) -> tuple[torch.Tensor, torch.Tensor, int, int, str]:
        pack_id = self.pack_id_at(absolute_index)
        token, mask, length, targets, _pack_id = self.pack_at(pack_id)
        return token, mask, length, targets, self.sample_id(absolute_index)

    def get_batch_with_record_spans(self, batch_size: int, device=None):
        """Sample packs and retain boundaries needed for clean recurrent resets."""
        if batch_size != 1:
            raise ValueError("record-reset SFT currently requires batch_size=1")
        absolute = self.next_absolute_rank_sample_index
        pack_id = self.pack_id_at(absolute)
        result = self.get_batch(batch_size, device=device)
        return (*result, (self.record_spans_at(pack_id),))

    def get_batch(self, batch_size: int, device=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        tokens = torch.empty((batch_size, self.sequence_tokens), dtype=torch.long)
        masks = torch.empty((batch_size, self.sequence_tokens), dtype=torch.bool)
        lengths = torch.empty(batch_size, dtype=torch.long)
        targets = torch.empty(batch_size, dtype=torch.long)
        sample_ids = []
        for item in range(batch_size):
            absolute = self.next_absolute_rank_sample_index + item
            token, mask, length, target_count, sample_id = self.sample_at(absolute)
            tokens[item], masks[item] = token, mask
            lengths[item], targets[item] = length, target_count
            sample_ids.append(sample_id)
        self.next_absolute_rank_sample_index += batch_size
        self.last_batch_sample_ids = tuple(sample_ids)
        if device is not None:
            return (tokens.to(device, non_blocking=True), masks.to(device, non_blocking=True),
                    lengths.to(device, non_blocking=True), targets.to(device, non_blocking=True))
        return tokens, masks, lengths, targets
