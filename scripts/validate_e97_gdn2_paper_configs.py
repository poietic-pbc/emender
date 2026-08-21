#!/usr/bin/env python3
"""Validate and render immutable manifests for the three paper arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ndm.models.ladder_lm import LadderLM


CONFIG_ROOT = ROOT / "configs/frontier/e97_gdn2_paper"
CONFIGS = {
    "e97-mlp": CONFIG_ROOT / "e97_mlp.json",
    "e97-linear-mlp": CONFIG_ROOT / "e97_linear_mlp.json",
    "gdn2-mlp": CONFIG_ROOT / "gdn2_mlp.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def build(config: dict):
    common = dict(
        vocab_size=50281,
        dim=config["dim"], depth=config["depth"], level=config["level"],
        expansion=config.get("expansion", 1.0),
        n_state=config.get("n_state", 64), n_heads=config["n_heads"],
        mlp_ratio=config.get("mlp_ratio", 0.0),
        mlp_multiple=config.get("mlp_multiple", 64),
        use_conv=bool(config.get("use_conv", 0)), d_conv=config.get("d_conv", 4),
        gdn2_mlp_ratio=config.get("gdn2_mlp_ratio", 6208 / 2304),
    )
    if config["level"] == "E97":
        common.update(
            n_groups=config["n_groups"], n_slots=config["n_slots"],
            state_expansion=config["state_expansion"],
            use_gate=bool(config["use_gate"]),
            gate_activation=config["gate_activation"],
            linear_state=bool(config["linear_state"]),
            e88_raw_write=bool(config["e88_raw_write"]),
            e88_decay_mode=config["e88_decay_mode"],
            use_triton=bool(config["use_triton"]),
            use_chunked_e97=bool(config["use_chunked_e97"]),
            e97_chunk_size=config["e97_chunk_size"],
            layer_kwargs={
                "use_output_norm": bool(config["use_output_norm"]),
                "output_norm_affine": bool(config["output_norm_affine"]),
            },
        )
    with torch.device("meta"):
        return LadderLM(**common)


def schema(model) -> tuple[list[dict], str]:
    entries = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        }
        for name, tensor in model.state_dict(keep_vars=True).items()
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return entries, sha256_bytes(encoded)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def external_receipt(root: Path, expected_commit: str) -> dict:
    head = git_output(root, "rev-parse", "HEAD")
    if head != expected_commit:
        raise RuntimeError(f"GDN2 HEAD {head} != required {expected_commit}")
    tracked = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise RuntimeError("GDN2 checkout has tracked modifications")
    archive = subprocess.check_output(
        ["git", "-C", str(root), "archive", "--format=tar", expected_commit])
    return {
        "origin": git_output(root, "remote", "get-url", "origin"),
        "commit": head,
        "tree": git_output(root, "rev-parse", f"{head}^{{tree}}"),
        "git_archive_sha256": sha256_bytes(archive),
        "tracked_worktree_clean": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gdn2-path", type=Path,
        default=Path(os.environ.get("GDN2_PATH", ROOT / "src/GatedDeltaNet-2")))
    args = parser.parse_args()

    configs = {arm: load(path) for arm, path in CONFIGS.items()}
    nonlinear, linear = configs["e97-mlp"], configs["e97-linear-mlp"]
    permitted_differences = {"arm", "linear_state", "use_output_norm"}
    differences = {
        key for key in set(nonlinear) | set(linear)
        if nonlinear.get(key) != linear.get(key)
    }
    if differences != permitted_differences:
        raise RuntimeError(
            f"matched E97 arms differ in unexpected fields: {sorted(differences)}")
    if nonlinear["linear_state"] != 0 or linear["linear_state"] != 1:
        raise RuntimeError("matched E97 activation flags are not 0 -> 1")
    if nonlinear["use_output_norm"] != 0 or linear["use_output_norm"] != 1:
        raise RuntimeError("stabilized linear arm must alone enable output RMSNorm")
    if nonlinear["output_norm_affine"] != 0 or linear["output_norm_affine"] != 0:
        raise RuntimeError("E97 paper arms require parameter-free output RMSNorm")

    external = external_receipt(
        args.gdn2_path, configs["gdn2-mlp"]["external_gdn2_commit"])
    os.environ["GDN2_PATH"] = str(args.gdn2_path.resolve())
    args.output.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for arm, config in configs.items():
        path = CONFIGS[arm]
        model = build(config)
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != config["exact_parameters"]:
            raise RuntimeError(
                f"{arm} parameter count {count} != frozen {config['exact_parameters']}")
        entries, digest = schema(model)
        manifest = {
            "schema": "emender-e97-gdn2-paper-arm-manifest-v1",
            "arm": arm,
            "config_path": str(path.relative_to(ROOT)),
            "config_sha256": sha256_bytes(path.read_bytes()),
            "config": config,
            "vocab_size": 50281,
            "exact_parameters": count,
            "tensor_count": len(entries),
            "tensor_schema_sha256": digest,
            "tensors": entries,
            "initialization": {
                "seed": config["seed"],
                "policy": "train.py/LadderLM current-source initialization",
                "initialized_state_sha256": None,
                "status": "pending exact-source machine materialization gate",
            },
            "external_gdn2": external if arm == "gdn2-mlp" else None,
        }
        output = args.output / f"{arm}.json"
        output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        manifests[arm] = manifest

    if (manifests["e97-mlp"]["tensor_schema_sha256"]
            != manifests["e97-linear-mlp"]["tensor_schema_sha256"]):
        raise RuntimeError("matched E97 tensor schemas differ")
    print(json.dumps({
        arm: {
            "parameters": value["exact_parameters"],
            "tensor_count": value["tensor_count"],
            "tensor_schema_sha256": value["tensor_schema_sha256"],
        }
        for arm, value in manifests.items()
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
