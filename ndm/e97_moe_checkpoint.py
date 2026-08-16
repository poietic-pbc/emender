"""Atomic sharded checkpoints for eight-GCD E97 MoE node islands."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Mapping

import torch
import torch.distributed as dist

from ndm.data.masked_sft_dataset import (
    SFTSamplerIdentity,
    restore_sft_checkpoint_metadata,
)
from ndm.data.tokenized_dataset import (
    BOUNDARY_COUNTER_SAMPLER_SCHEMA,
    LEGACY_SAMPLER_SCHEMA,
    CounterSamplerIdentity,
    restore_sampler_checkpoint_metadata,
)


SCHEMA = "emender-e97-moe-sharded-v1"
_LOCAL_SUFFIXES = ("local_gate_weight", "local_up_weight", "local_down_weight")


def _replicated_owner(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def validate_moe_sampler_manifest(
    manifest: Mapping,
    *,
    expected_identity: CounterSamplerIdentity | None,
    allow_legacy_transition: bool = False,
    allow_counter_transition: bool = False,
    diloco_k: int | None = None,
) -> str:
    """Validate sampler authority before any checkpoint tensor is restored.

    Returns ``counter``, ``legacy``, or ``legacy-transition``. Missing sampler
    fields are historical legacy evidence, never an implicit counter stream.
    """
    accepted_tokens = int(manifest.get("accepted_tokens", -1))
    persisted = manifest.get("sampler")
    if expected_identity is None:
        if persisted is None or persisted == {
                "schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}:
            return "legacy"
        raise RuntimeError(
            "checkpoint contains counter sampler metadata but launch selects legacy mode")
    if persisted is None or persisted.get("schema") == LEGACY_SAMPLER_SCHEMA:
        if not allow_legacy_transition:
            raise RuntimeError(
                "legacy MoE checkpoint cannot be silently relabelled as counter sampled")
        if diloco_k is None or diloco_k <= 0:
            raise RuntimeError("legacy sampler transition requires a positive DiLoCo K")
        step = int(manifest.get("step", -1))
        if step < 0 or step % diloco_k:
            raise RuntimeError(
                "legacy sampler transition requires a complete K-aligned checkpoint")
        if expected_identity.schema != BOUNDARY_COUNTER_SAMPLER_SCHEMA:
            raise RuntimeError(
                "legacy sampler transition requires boundary-relative counter-v2")
        if expected_identity.stream_origin_accepted_tokens != accepted_tokens:
            raise RuntimeError(
                "counter-v2 stream origin must equal the legacy checkpoint "
                f"accepted-token boundary: {expected_identity.stream_origin_accepted_tokens} "
                f"!= {accepted_tokens}")
        return "legacy-transition"
    try:
        _identity, sampler_tokens, _cursor = restore_sampler_checkpoint_metadata(
            persisted, expected_identity=expected_identity)
        status = "counter"
    except (TypeError, ValueError) as error:
        if not allow_counter_transition:
            raise RuntimeError(f"MoE checkpoint sampler metadata mismatch: {error}") from error
        try:
            previous_identity = CounterSamplerIdentity.from_metadata(
                persisted["identity"])
            _identity, sampler_tokens, _cursor = restore_sampler_checkpoint_metadata(
                persisted, expected_identity=previous_identity)
        except (KeyError, TypeError, ValueError) as previous_error:
            raise RuntimeError(
                f"invalid previous counter sampler authority: {previous_error}") from previous_error
        step = int(manifest.get("step", -1))
        if diloco_k is None or diloco_k <= 0 or step < 0 or step % diloco_k:
            raise RuntimeError(
                "counter phase transition requires a complete K-aligned checkpoint")
        if expected_identity.schema != BOUNDARY_COUNTER_SAMPLER_SCHEMA:
            raise RuntimeError("counter phase transition requires boundary-relative counter-v2")
        if expected_identity.stream_origin_accepted_tokens != accepted_tokens:
            raise RuntimeError(
                "new counter stream origin must equal the previous accepted-token boundary")
        status = "counter-transition"
    if sampler_tokens != accepted_tokens:
        raise RuntimeError(
            "MoE sampler accepted-token clock contradicts checkpoint manifest")
    return status


def save_node_sharded_checkpoint(
    root: str | Path,
    model,
    optimizer,
    *,
    step: int,
    accepted_tokens: int,
    source_commit: str,
    sampler: Mapping | None = None,
    sampler_transition: Mapping | None = None,
    node_group=None,
    keep_generations: int = 0,
) -> Path:
    """Synchronously publish a complete atomic eight-shard checkpoint."""
    if dist.get_world_size(node_group) != 8:
        raise RuntimeError("checkpoint publication requires one complete eight-rank node")
    rank = dist.get_rank(node_group)
    root = Path(root)
    generation = root / f"step-{step:08d}-tokens-{accepted_tokens:016d}"
    generation.mkdir(parents=True, exist_ok=True)
    optimizer.eval()
    named = dict(model.named_parameters())

    def entry(name, parameter):
        state = optimizer.state.get(parameter, {})
        if "z" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(f"optimizer state is incomplete for {name}")
        return {
            "parameter": parameter.detach(),
            "z": state["z"].detach(),
            "exp_avg_sq": state["exp_avg_sq"].detach(),
        }

    local_entries = {
        name: entry(name, parameter)
        for name, parameter in named.items() if name.endswith(_LOCAL_SUFFIXES)
    }
    replicated_entries = {
        name: entry(name, parameter)
        for name, parameter in named.items()
        if not name.endswith(_LOCAL_SUFFIXES) and _replicated_owner(name) == rank
    }
    sampler = dict(sampler or {
        "schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"})
    common = {
        "schema": SCHEMA, "rank": rank, "step": int(step),
        "accepted_tokens": int(accepted_tokens), "source_commit": source_commit,
        "sampler": sampler,
    }
    local_path = generation / f"local-rank-{rank}.pt"
    replicated_path = generation / f"replicated-owner-{rank}.pt"
    _atomic_torch_save({**common, "entries": local_entries}, local_path)
    _atomic_torch_save({**common, "entries": replicated_entries}, replicated_path)
    sidecar = {
        **common,
        "local_expert_start": rank * 8,
        "local": {"file": local_path.name, "bytes": local_path.stat().st_size,
                  "sha256": _sha256(local_path), "entries": len(local_entries),
                  "names": sorted(local_entries)},
        "replicated": {"file": replicated_path.name,
                       "bytes": replicated_path.stat().st_size,
                       "sha256": _sha256(replicated_path),
                       "entries": len(replicated_entries),
                       "names": sorted(replicated_entries)},
    }
    _atomic_json(sidecar, generation / f"rank-{rank}.json")
    dist.barrier(group=node_group)
    if rank == 0:
        ranks = [json.loads((generation / f"rank-{lane}.json").read_text())
                 for lane in range(8)]
        all_replicated = sorted(
            name for name in named if not name.endswith(_LOCAL_SUFFIXES))
        replicated_names = [
            name for sidecar in ranks for name in sidecar["replicated"]["names"]]
        if sorted(replicated_names) != all_replicated or len(set(replicated_names)) != len(replicated_names):
            raise RuntimeError("replicated checkpoint ownership is incomplete or duplicated")
        manifest = {
            "schema": SCHEMA, "complete": True, "step": int(step),
            "accepted_tokens": int(accepted_tokens), "source_commit": source_commit,
            "sampler": sampler,
            "sampler_transition": (
                dict(sampler_transition) if sampler_transition is not None else None),
            "ranks": ranks,
            "optimizer_groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in optimizer.param_groups],
        }
        _atomic_json(manifest, generation / "manifest.json")
        temporary_link = root / f".latest.tmp-{os.getpid()}"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(generation.name)
        os.replace(temporary_link, root / "latest")
        if keep_generations > 0:
            complete = sorted(
                path for path in root.glob("step-*-tokens-*")
                if path.is_dir() and (path / "manifest.json").is_file())
            for expired in complete[:-keep_generations]:
                if expired != generation:
                    shutil.rmtree(expired)
    dist.barrier(group=node_group)
    optimizer.train()
    return generation


def _resolve_complete_generation(root_or_generation: str | Path) -> tuple[Path, dict]:
    candidate = Path(root_or_generation)
    generation = candidate if (candidate / "manifest.json").is_file() else (
        candidate / "latest").resolve(strict=True)
    manifest = json.loads((generation / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise RuntimeError("checkpoint manifest is not a complete E97 MoE shard set")
    return generation, manifest


def load_node_sharded_model(
    root_or_generation: str | Path,
    model,
    *,
    node_group=None,
    verify_sha256: bool = True,
):
    """Restore model parameters only from a complete canonical eight-shard set.

    This read-only evaluation path accepts either a checkpoint root containing
    ``latest`` or an immutable generation directory. ScheduleFree state is not
    materialized on GPU. Replicated tensors retain the canonical owner/broadcast
    protocol used by training restores.
    """
    if dist.get_world_size(node_group) != 8:
        raise RuntimeError("checkpoint restore requires one complete eight-rank node")
    rank = dist.get_rank(node_group)
    generation, manifest = _resolve_complete_generation(root_or_generation)
    sidecars = manifest.get("ranks", [])
    if len(sidecars) != 8 or int(sidecars[rank].get("rank", -1)) != rank:
        raise RuntimeError("checkpoint manifest does not contain eight ordered rank sidecars")
    sidecar = sidecars[rank]
    named = dict(model.named_parameters())
    expected_local = sorted(
        name for name in named if name.endswith(_LOCAL_SUFFIXES))
    expected_replicated = sorted(
        name for name in named
        if not name.endswith(_LOCAL_SUFFIXES) and _replicated_owner(name) == rank)
    if (sidecar["local"].get("names") != expected_local
            or sidecar["replicated"].get("names") != expected_replicated):
        raise RuntimeError("checkpoint sidecar parameter ownership does not match model")
    restore_paths = [
        generation / sidecar["local"]["file"],
        generation / sidecar["replicated"]["file"],
    ]
    for kind, path in zip(("local", "replicated"), restore_paths):
        info = sidecar[kind]
        if path.stat().st_size != info["bytes"]:
            raise RuntimeError(f"checkpoint shard size mismatch: {path}")
        if verify_sha256 and _sha256(path) != info["sha256"]:
            raise RuntimeError(f"checkpoint shard failed integrity: {path}")

    with torch.no_grad():
        for kind, restore_path in zip(("local", "replicated"), restore_paths):
            payload = torch.load(
                restore_path, map_location="cpu", weights_only=False, mmap=True)
            if (payload.get("sampler") != manifest.get("sampler")
                    or int(payload.get("step", -1)) != int(manifest["step"])
                    or int(payload.get("accepted_tokens", -1))
                    != int(manifest["accepted_tokens"])):
                raise RuntimeError(
                    f"checkpoint shard sampler/clock authority mismatch: {restore_path}")
            entries = payload.get("entries", {})
            if sorted(entries) != sidecar[kind]["names"]:
                raise RuntimeError(
                    f"checkpoint payload entries contradict sidecar: {restore_path}")
            for name, saved in entries.items():
                named[name].copy_(saved["parameter"])
            del payload

        for name, parameter in named.items():
            if name.endswith(_LOCAL_SUFFIXES):
                continue
            owner = _replicated_owner(name)
            source = dist.get_global_rank(node_group, owner)
            dist.broadcast(parameter.data, src=source, group=node_group)
    return manifest


def validate_sft_parent_optimizer_transition(
    generation: str | Path, manifest: Mapping, expected_parent: Mapping,
) -> None:
    """Fail closed unless an SFT boundary preserves one exact mature optimizer."""
    required_parent = {"manifest_sha256", "step", "accepted_tokens", "generation"}
    if set(expected_parent) != required_parent:
        raise RuntimeError("SFT parent optimizer transition authority fields mismatch")
    if Path(generation).resolve() != Path(str(expected_parent["generation"])).resolve():
        raise RuntimeError("SFT parent optimizer transition generation mismatch")
    if (int(manifest.get("step", -1)) != int(expected_parent["step"])
            or int(manifest.get("accepted_tokens", -1))
            != int(expected_parent["accepted_tokens"])):
        raise RuntimeError("SFT parent optimizer transition clock mismatch")
    groups = manifest.get("optimizer_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise RuntimeError("SFT parent optimizer transition requires one optimizer group")
    group = groups[0]
    if (int(group.get("k", 0)) <= 0 or float(group.get("weight_sum", 0.0)) <= 0
            or float(group.get("lr", 0.0)) <= 0
            or float(group.get("lr_max", 0.0)) <= 0
            or group.get("train_mode") is not False):
        raise RuntimeError("SFT parent optimizer state is not mature eval-state ScheduleFree")


def load_node_sharded_checkpoint(
    root: str | Path,
    model,
    optimizer,
    *,
    node_group=None,
    expected_sampler_identity: CounterSamplerIdentity | None = None,
    expected_sft_identity: SFTSamplerIdentity | None = None,
    expected_sft_parent: Mapping | None = None,
    allow_sft_parent_optimizer_transition: bool = False,
    allow_legacy_sampler_transition: bool = False,
    allow_counter_sampler_transition: bool = False,
    diloco_k: int | None = None,
):
    """Restore this rank's local experts and every replicated parameter/state."""
    if dist.get_world_size(node_group) != 8:
        raise RuntimeError("checkpoint restore requires one complete eight-rank node")
    rank = dist.get_rank(node_group)
    generation, manifest = _resolve_complete_generation(root)
    parent_transition = False
    if expected_sft_identity is not None:
        if expected_sampler_identity is not None or expected_sft_parent is None:
            raise RuntimeError("SFT restore requires one exclusive sampler and parent")
        parent_transition = bool(allow_sft_parent_optimizer_transition)
        if parent_transition:
            validate_sft_parent_optimizer_transition(
                generation, manifest, expected_sft_parent)
            sampler_status = "sft-parent-optimizer-transition"
            manifest["sft_restore_clocks"] = (0, 0, 0)
        else:
            try:
                sft_clocks = restore_sft_checkpoint_metadata(
                    manifest.get("sampler", {}),
                    expected_identity=expected_sft_identity,
                    expected_parent=expected_sft_parent,
                    model_accepted_tokens=int(manifest["accepted_tokens"]))
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"SFT checkpoint metadata mismatch: {error}") from error
            sampler_status = "sft"
            manifest["sft_restore_clocks"] = sft_clocks
    else:
        sampler_status = validate_moe_sampler_manifest(
            manifest, expected_identity=expected_sampler_identity,
            allow_legacy_transition=allow_legacy_sampler_transition,
            allow_counter_transition=allow_counter_sampler_transition,
            diloco_k=diloco_k)
    manifest["sampler_restore_status"] = sampler_status
    for rank_sidecar in manifest.get("ranks", []):
        if (rank_sidecar.get("sampler") != manifest.get("sampler")
                or int(rank_sidecar.get("step", -1)) != int(manifest["step"])
                or int(rank_sidecar.get("accepted_tokens", -1))
                != int(manifest["accepted_tokens"])):
            raise RuntimeError("checkpoint sidecar sampler/clock authority mismatch")
    # Each lane validates and reads only the files it owns. Replicated tensors
    # are then broadcast inside the node island. This preserves rank-0-style
    # canonical checkpoint authority without making every rank reread and hash
    # the complete ~210 GB island checkpoint.
    sidecar = manifest["ranks"][rank]
    restore_paths = [
        generation / sidecar["local"]["file"],
        generation / sidecar["replicated"]["file"],
    ]
    for kind, path in zip(("local", "replicated"), restore_paths):
        info = sidecar[kind]
        if path.stat().st_size != info["bytes"] or _sha256(path) != info["sha256"]:
            raise RuntimeError(f"checkpoint shard failed integrity: {path}")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for restore_path in restore_paths:
            payload = torch.load(restore_path, map_location="cpu", weights_only=False)
            if (payload.get("sampler") != manifest.get("sampler")
                    or int(payload.get("step", -1)) != int(manifest["step"])
                    or int(payload.get("accepted_tokens", -1))
                    != int(manifest["accepted_tokens"])):
                raise RuntimeError(
                    f"checkpoint shard sampler/clock authority mismatch: {restore_path}")
            for name, saved in payload["entries"].items():
                parameter = named[name]
                parameter.copy_(saved["parameter"])
                optimizer.state[parameter]["z"] = saved["z"].to(parameter.device)
                optimizer.state[parameter]["exp_avg_sq"] = saved["exp_avg_sq"].to(parameter.device)
            del payload

        for name, parameter in named.items():
            if name.endswith(_LOCAL_SUFFIXES):
                continue
            owner = _replicated_owner(name)
            state = optimizer.state[parameter]
            if rank != owner:
                state["z"] = torch.empty_like(parameter)
                state["exp_avg_sq"] = torch.empty_like(parameter)
            source = dist.get_global_rank(node_group, owner)
            dist.broadcast(parameter.data, src=source, group=node_group)
            dist.broadcast(state["z"], src=source, group=node_group)
            dist.broadcast(state["exp_avg_sq"], src=source, group=node_group)
    saved_groups = manifest["optimizer_groups"]
    if parent_transition:
        if len(saved_groups) != 1 or len(optimizer.param_groups) not in (1, 2):
            raise RuntimeError("SFT parent optimizer transition group layout mismatch")
        optimizer.param_groups[0].update(saved_groups[0])
    else:
        if len(saved_groups) != len(optimizer.param_groups):
            raise RuntimeError("checkpoint optimizer group count mismatch")
        for group, saved_group in zip(optimizer.param_groups, saved_groups):
            group.update(saved_group)
    optimizer.assert_no_master_weights()
    return manifest
