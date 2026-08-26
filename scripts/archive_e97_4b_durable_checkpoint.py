#!/usr/bin/env python3
"""Archive and stage one trusted E97 4B exact-resume checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise RuntimeError(
                f"refusing to reuse a different existing artifact: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        try:
            with source.open("rb") as inp, os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
                out.flush()
                os.fsync(out.fileno())
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        while chunk := inp.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--durable-root", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="spinozans/e97-4b-training-checkpoints")
    parser.add_argument("--args-json", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    source = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    metadata = checkpoint.get("checkpoint_metadata")
    if not isinstance(metadata, dict) or not metadata.get("is_head"):
        raise ValueError("checkpoint is not a head-rank training authority")
    model = metadata.get("model") or {}
    if int(model.get("total_params", -1)) != 4_045_972_080:
        raise ValueError("checkpoint is not the qualified 4.046B E97 shape")
    sampler = metadata.get("sampler") or {}
    identity = sampler.get("identity") or {}
    if identity.get("schema") != "emender-byte-window-counter-v1":
        raise ValueError("checkpoint lacks the versioned counter sampler")
    step = int(checkpoint["step"])
    tokens = int(checkpoint["total_tokens"])
    if tokens != int(sampler.get("total_accepted_tokens", -1)):
        raise ValueError("checkpoint and sampler token clocks disagree")
    loss = float(checkpoint["loss"])
    rel = Path("checkpoints") / f"step_{step:06d}_tokens_{tokens}"
    durable_dir = args.durable_root / rel.name
    durable = durable_dir / source.name
    link_or_copy(source, durable)
    digest = sha256(durable)
    size = durable.stat().st_size
    receipt = {
        "schema": "emender-e97-4b-durable-checkpoint-v1",
        "repository": args.repo_id,
        "checkpoint": source.name,
        "checkpoint_sha256": digest,
        "size_bytes": size,
        "step": step,
        "total_accepted_tokens": tokens,
        "loss": loss,
        "parameters": int(model["total_params"]),
        "source_commit": args.source_commit,
        "checkpoint_kind": metadata.get("kind"),
        "world_size": int(metadata["world_size"]),
        "schedulefree_state_included": "optimizer_state_dict" in checkpoint,
        "sampler": {**identity, "absolute_rank_sample_index": int(sampler["absolute_rank_sample_index"])},
        "frontier_world256_exact_resume_compatible": int(metadata["world_size"]) == 256,
        "security": "Raw torch.save pickle; verify SHA-256 and load only as trusted content.",
        "archived_unix": time.time(),
    }
    atomic_json(durable_dir / "metadata.json", receipt)
    (durable_dir / "SHA256SUMS").write_text(f"{digest}  {source.name}\n", encoding="utf-8")
    if args.args_json:
        shutil.copy2(args.args_json, durable_dir / "args.json")

    staged_dir = args.staging_root / rel
    link_or_copy(durable, staged_dir / source.name)
    shutil.copy2(durable_dir / "metadata.json", staged_dir / "metadata.json")
    shutil.copy2(durable_dir / "SHA256SUMS", staged_dir / "SHA256SUMS")
    if (durable_dir / "args.json").exists():
        shutil.copy2(durable_dir / "args.json", staged_dir / "args.json")
    atomic_json(args.staging_root / "LATEST.json", {**receipt, "repo_path": str(rel / source.name)})

    latest_tmp = args.durable_root / ".latest.tmp"
    try:
        latest_tmp.unlink()
    except FileNotFoundError:
        pass
    latest_tmp.symlink_to(rel.name)
    os.replace(latest_tmp, args.durable_root / "latest")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
