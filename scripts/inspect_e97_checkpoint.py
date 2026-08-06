#!/usr/bin/env python3
"""Inspect an actual E97 checkpoint and solve the 35B MoE recipe.

The checkpoint is mmap-loaded, while the corresponding module graph is
instantiated on the meta device.  This validates exact names/shapes and permits
an actual 35B MoE graph parameter count without allocating its tensor storage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train
from ndm.async_diloco_real import default_tiny_e97_train_args
from ndm.models.e97_moe import (
    E97MoEConfig,
    calculate_e97_moe_recipe,
    convert_e97_ffns_to_moe,
)
from ndm.models.ladder_lm import MixerMLPWrapper, SwiGLUMLP


def _shape_map(state):
    return {name: list(value.shape) for name, value in state.items()}


def _validate_checkpoint_graph(model, checkpoint_state) -> None:
    graph = _shape_map(model.state_dict())
    checkpoint = _shape_map(checkpoint_state)
    missing = sorted(set(graph) - set(checkpoint))
    unexpected = sorted(set(checkpoint) - set(graph))
    mismatched = {
        name: {"graph": graph[name], "checkpoint": checkpoint[name]}
        for name in sorted(set(graph) & set(checkpoint))
        if graph[name] != checkpoint[name]
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(json.dumps({
            "error": "checkpoint does not match instantiated E97 graph",
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatches": mismatched,
        }, indent=2))


def inspect(checkpoint_path: Path, config_path: Path, *, target: int,
            multiple: int) -> dict:
    config = json.loads(config_path.read_text())
    if str(config.get("level")) != "E97":
        raise ValueError("inspection config must explicitly select level E97")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict")
    embedding = state.get("embedding.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("checkpoint has no valid E97 embedding.weight")
    vocab_size = int(embedding.shape[0])

    args = default_tiny_e97_train_args(**config)
    with torch.device("meta"):
        model = train.build_training_model(args, vocab_size=vocab_size)
    _validate_checkpoint_graph(model, state)

    dense_ffn_ids: set[int] = set()
    layers = []
    signatures = set()
    for index, layer in enumerate(model.layers):
        if not isinstance(layer, MixerMLPWrapper):
            raise RuntimeError(f"layer {index} lacks the expected post-mixer FFN wrapper")
        if not isinstance(layer.mlp, SwiGLUMLP):
            raise RuntimeError(f"layer {index} FFN is not SwiGLUMLP")
        ffn_parameters = []
        for local_name, parameter in layer.mlp.named_parameters():
            dense_ffn_ids.add(id(parameter))
            full_name = f"layers.{index}.mlp.{local_name}"
            ffn_parameters.append({
                "name": full_name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(state[full_name].dtype),
            })
        signature = (
            type(layer.mixer).__module__, type(layer.mixer).__name__,
            layer.dim, layer.mlp_hidden_dim,
            tuple((entry["name"].split(f"layers.{index}.", 1)[1],
                   tuple(entry["shape"])) for entry in ffn_parameters),
            type(layer.norm_2).__name__,
        )
        signatures.add(signature)
        layers.append({
            "index": index,
            "recurrent_mixer_class": (
                f"{type(layer.mixer).__module__}.{type(layer.mixer).__name__}"),
            "ffn_class": f"{type(layer.mlp).__module__}.{type(layer.mlp).__name__}",
            "model_width": layer.dim,
            "ffn_intermediate_width": layer.mlp_hidden_dim,
            "ffn_parameters": ffn_parameters,
            "residual_and_normalization_order": [
                "LadderLM: residual = previous_mixer_output + previous_residual",
                "LadderLM: mixer_input = pre_mixer_RMSNorm(residual)",
                "MixerMLPWrapper: mix_out = recurrent_mixer(mixer_input, state)",
                "MixerMLPWrapper: ffn_input = post_mixer_RMSNorm(mixer_input + mix_out)",
                "MixerMLPWrapper: block_output = mix_out + FFN(ffn_input)",
            ],
        })
    if len(signatures) != 1:
        raise RuntimeError(f"E97 layers differ unexpectedly ({len(signatures)} signatures)")

    dense_total = sum(parameter.numel() for parameter in model.parameters())
    dense_ffn = sum(
        parameter.numel() for parameter in model.parameters()
        if id(parameter) in dense_ffn_ids)
    non_ffn = dense_total - dense_ffn
    recipe = calculate_e97_moe_recipe(
        model, target_parameters=target, multiple=multiple)

    moe_config = E97MoEConfig(hidden_dim=recipe.expert_hidden)
    convert_e97_ffns_to_moe(model, moe_config)
    instantiated_moe_total = sum(parameter.numel() for parameter in model.parameters())
    if instantiated_moe_total != recipe.total_parameters:
        raise RuntimeError(
            "actual meta-device MoE graph count differs from solved recipe: "
            f"{instantiated_moe_total} != {recipe.total_parameters}")

    metadata = checkpoint.get("checkpoint_metadata", {})
    return {
        "schema": "emender-e97-to-35b-moe-inspection-v1",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "size_bytes": checkpoint_path.stat().st_size,
            "step": int(checkpoint.get("step", -1)),
            "total_tokens": int(checkpoint.get("total_tokens", -1)),
            "loss": float(checkpoint.get("loss", float("nan"))),
            "metadata": metadata,
        },
        "config": {"path": str(config_path.resolve()), "values": config},
        "graph_validation": {
            "checkpoint_names_and_shapes_match": True,
            "layers_uniform": True,
            "layer_count": len(layers),
            "vocab_size": vocab_size,
        },
        "dense_parameters": {
            "total": dense_total,
            "non_ffn": non_ffn,
            "ffn": dense_ffn,
        },
        "moe_recipe": {
            **recipe.to_dict(),
            "instantiated_meta_graph_parameters": instantiated_moe_total,
            "target_parameters": target,
            "rounding_multiple": multiple,
        },
        "layers": layers,
        "protected_modules": [
            "embedding", "layer_norms", "layers.*.mixer", "norm", "lm_head"
        ],
        "replacement_scope": "layers.*.mlp only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/frontier/e97_resilient_split_role_flat.json")
    parser.add_argument("--target-parameters", type=int, default=35_000_000_000)
    parser.add_argument("--multiple", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(
        args.checkpoint, args.config, target=args.target_parameters,
        multiple=args.multiple)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
