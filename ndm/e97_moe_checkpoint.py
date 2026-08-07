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


def save_node_sharded_checkpoint(
    root: str | Path,
    model,
    optimizer,
    *,
    step: int,
    accepted_tokens: int,
    source_commit: str,
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
    common = {
        "schema": SCHEMA, "rank": rank, "step": int(step),
        "accepted_tokens": int(accepted_tokens), "source_commit": source_commit,
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


def load_node_sharded_checkpoint(root: str | Path, model, optimizer, *, node_group=None):
    """Restore this rank's local experts and every replicated parameter/state."""
    if dist.get_world_size(node_group) != 8:
        raise RuntimeError("checkpoint restore requires one complete eight-rank node")
    rank = dist.get_rank(node_group)
    root = Path(root)
    generation = (root / "latest").resolve(strict=True)
    manifest = json.loads((generation / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("complete") is not True:
        raise RuntimeError("checkpoint manifest is not a complete E97 MoE shard set")
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
    for group, saved_group in zip(optimizer.param_groups, manifest["optimizer_groups"]):
        group.update(saved_group)
    optimizer.assert_no_master_weights()
    return manifest
