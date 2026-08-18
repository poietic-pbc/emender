import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch

from ndm.data.masked_sft_dataset import (
    AUTHORITY_SCHEMA, RECORD_INDEX, MaskedSFTPackedDataset, SFTSamplerIdentity,
    restore_sft_checkpoint_metadata, sft_checkpoint_metadata, sha256,
)


def _authority(root: Path):
    root.mkdir()
    records = [
        ([10, 11, 218], [0, 1, 1], 0),
        ([20, 21], [0, 1], 0),
        ([30, 31, 32, 33, 34, 35], [0, 0, 1, 1, 1, 1], 0),
        ([40, 218], [0, 1], 1),
    ]
    token_path = root / "tokens.uint32.bin"
    mask_path = root / "assistant_mask.uint8.bin"
    index_path = root / "records.idx"
    metadata_path = root / "records.jsonl"
    offset = 0
    with token_path.open("wb") as tokens, mask_path.open("wb") as masks, index_path.open("wb") as index:
        for token_values, mask_values, split in records:
            tokens.write(struct.pack(f"<{len(token_values)}I", *token_values))
            masks.write(bytes(mask_values))
            index.write(RECORD_INDEX.pack(offset, len(token_values), sum(mask_values), split))
            offset += len(token_values)
    metadata_path.write_text("\n".join("{}" for _ in records) + "\n")
    outputs = {}
    for name, path in (("tokens", token_path), ("mask", mask_path),
                       ("index", index_path), ("metadata", metadata_path)):
        outputs[name] = {"path": str(path.resolve()), "bytes": path.stat().st_size,
                         "sha256": sha256(path)}
    manifest = {"schema": AUTHORITY_SCHEMA, "status": "complete", "outputs": outputs}
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return sha256(root / "manifest.json")


def _fixture(tmp_path: Path):
    authority = tmp_path / "authority"
    authority_sha = _authority(authority)
    packs = tmp_path / "packs"
    subprocess.run([
        sys.executable, "scripts/build_e97_sft_packs.py",
        "--authority-root", str(authority), "--output-root", str(packs),
        "--context-size", "4", "--authority-manifest-sha256", authority_sha,
    ], check=True, capture_output=True, text=True)
    identity = SFTSamplerIdentity(
        authority_manifest_sha256=authority_sha,
        pack_manifest_sha256=sha256(packs / "manifest.json"), sampler_key=42,
        data_world_size=2, context_size=4)
    return authority, packs, identity


def test_complete_record_packing_and_counter_sampling(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    manifest = json.loads((packs / "manifest.json").read_text())
    assert manifest["splits"]["train"] == {
        "records": 2, "tokens": 5, "assistant_target_tokens": 3, "packs": 1,
        "excluded_oversize_records": 1, "excluded_oversize_tokens": 6,
        "excluded_oversize_assistant_target_tokens": 4,
    }
    dataset = MaskedSFTPackedDataset(
        authority, packs, identity=identity, rank=1,
        initial_absolute_rank_sample_index=7, verify_payload_hashes=True)
    tokens, masks, lengths, targets = dataset.get_batch(2)
    assert tokens.tolist() == [[10, 11, 218, 20, 21], [10, 11, 218, 20, 21]]
    assert masks.tolist() == [[False, True, True, False, True]] * 2
    assert lengths.tolist() == [5, 5]
    assert targets.tolist() == [3, 3]
    assert dataset.next_absolute_rank_sample_index == 9
    assert len(dataset.last_batch_sample_ids) == 2

    reset_dataset = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=1)
    reset_batch = reset_dataset.get_batch_with_record_spans(1)
    assert reset_batch[4] == (((0, 3), (3, 5)),)
    assert reset_dataset.next_absolute_rank_sample_index == 1
    with pytest.raises(ValueError, match="batch_size=1"):
        reset_dataset.get_batch_with_record_spans(2)


def test_exact_pack_access_bypasses_replacement_sampler(tmp_path):
    authority, packs, train_identity = _fixture(tmp_path)
    validation_identity = SFTSamplerIdentity(
        authority_manifest_sha256=train_identity.authority_manifest_sha256,
        pack_manifest_sha256=train_identity.pack_manifest_sha256,
        sampler_key=train_identity.sampler_key,
        data_world_size=train_identity.data_world_size,
        context_size=train_identity.context_size,
        split="validation")
    dataset = MaskedSFTPackedDataset(
        authority, packs, identity=validation_identity, rank=0)
    token, mask, length, targets, pack_id = dataset.pack_at(0)
    assert token.tolist() == [40, 218, 0, 0, 0]
    assert mask.tolist() == [False, True, False, False, False]
    assert (length, targets, pack_id) == (2, 1, "pack-00000000")
    with pytest.raises(IndexError, match="out of range"):
        dataset.pack_at(1)


def test_sft_checkpoint_clocks_fail_closed(tmp_path):
    _authority_root, _packs, identity = _fixture(tmp_path)
    parent = {"manifest_sha256": "a" * 64, "step": 10,
              "accepted_tokens": 1000, "generation": "/immutable/parent"}
    metadata = sft_checkpoint_metadata(
        identity, parent=parent, total_tokens=50, assistant_target_tokens=30,
        absolute_rank_sample_index=4)
    assert restore_sft_checkpoint_metadata(
        metadata, expected_identity=identity, expected_parent=parent,
        model_accepted_tokens=1050) == (50, 30, 4)
    with pytest.raises(ValueError, match="accepted tokens"):
        restore_sft_checkpoint_metadata(
            metadata, expected_identity=identity, expected_parent=parent,
            model_accepted_tokens=1051)


def test_counter_samples_depend_on_rank_and_absolute_cursor(tmp_path):
    authority, packs, identity = _fixture(tmp_path)
    rank0 = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=0)
    rank1 = MaskedSFTPackedDataset(authority, packs, identity=identity, rank=1)
    assert rank0.sample_id(0) != rank1.sample_id(0)
    assert rank0.sample_id(0) != rank0.sample_id(1)
