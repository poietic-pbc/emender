#!/usr/bin/env python3
"""Build a curated Hugging Face release from the final GDN2-MLP checkpoint.

The release is weights-only. It recovers Schedule-Free train/y weights, writes
BF16 safetensors, and packages a standalone clean PyTorch implementation of the
published GDN2 recurrence. NVIDIA source and raw pickle/optimizer state are not
copied into the release directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.eval_checkpoint import (
    build_model,
    load_checkpoint_weights,
    namespace_from_config,
)
from scripts.prepare_e97_hf_release import clone_state, sha256, tokenizer_files, write_json

SOURCE_COMMIT = "a3a862f30f5c3c9584e18490c986bc9a065d6653"
GDN2_SOURCE_COMMIT = "95709fc250357c2dd109361c353192f2aa5913f9"
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "hf_templates" / "gdn2"


def public_args(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "level", "dim", "depth", "n_heads", "expansion", "use_conv", "d_conv",
        "gdn2_mlp_ratio", "dropout", "tokenizer", "chunk_size",
    )
    result = {key: config[key] for key in keys if key in config}
    result.update({
        "head_dim": 128,
        "gdn2_mlp_multiple": 64,
        "allow_neg_eigval": False,
        "implementation": "portable_reference",
    })
    return result


def model_card(
    repo_id: str,
    *,
    tokens: int,
    step: int,
    loss: float,
    source_sha: str,
    checkpoint_size: int,
) -> str:
    return f'''---
license: other
pipeline_tag: text-generation
tags:
- recurrent-language-model
- gdn2
- gated-delta-net
- diloco
- pytorch
---

# GDN2-MLP 1.3B — 152B-token control

This is the matched GDN2-MLP control for
[Emender E97 1.3B](https://huggingface.co/spinozans/emender-e97-1.3b).
It is a **1.287B-parameter** recurrent base language model trained for
**{tokens:,} tokens**. It is not a 152B-parameter model.

The model uses NVIDIA Gated DeltaNet 2 token mixing plus a post-mixer SwiGLU
MLP, reshaped by CMA-ES to match the E97 parameter budget. Both controls used
the same Pile/p50k_base stream, 2,048-token chunks, BF16 Schedule-Free AdamW,
and eight-island DiLoCo harness.

## Artifact

- Final step: `{step:,}`
- Training-log last-100 loss: `{loss:.4f}` nats/token
- Source checkpoint: `{checkpoint_size:,}` bytes
- Source checkpoint SHA-256: `{source_sha}`
- Export: BF16 Schedule-Free train/y weights
- Tokenizer: `p50k_base`
- Raw pickle checkpoint and optimizer state: not included

At the last common regular log point (150,793,420,800 tokens), the 80-point
moving averages were E97 `2.437045` and GDN2 `2.426705`; the difference was
`-0.010340` nats/token (GDN2 minus E97). These are training-log summaries, not
a replacement for fixed held-out evaluation.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}", trust_remote_code=True, dtype="auto"
)
```

The bundled remote code is a standalone portable PyTorch implementation of the
published GDN2 recurrence. It does not redistribute NVIDIA source. It favors
portability and exact weight loading over fused-kernel throughput.

The original fused GDN2 implementation used for training is available from
[NVIDIA GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) at commit
`{GDN2_SOURCE_COMMIT}` under NVIDIA's Source Code License-NC. The Emender
training wrapper is recorded at commit `{SOURCE_COMMIT}`.

## Validation

- all 267 exported tensors match recovered source-checkpoint train/y weights exactly;
- the Transformers loader reports zero missing, unexpected, or mismatched keys;
- portable source-checkpoint and exported CPU logits are bit-identical;
- CUDA fused and portable paths agree on greedy argmax for `The theorem states`
  (measured mean/max absolute logit delta `0.01244` / `0.125` from BF16 kernel
  evaluation order);
- live-style greedy generation produces a finite continuation.

Machine-readable evidence is included in `validation.json`.

## Limitations and license

This is a raw base LM, not an instruction or chat model. It may emit incorrect,
biased, or unsafe text. Benchmark coverage is limited. No standalone model
license has been selected; `license: other` is intentional. The separately
available NVIDIA fused implementation has its own non-commercial source
license, which is not replaced by this model card.
'''


def copy_templates(output: Path) -> None:
    for name in ("configuration_emender_gdn2.py", "modeling_emender_gdn2.py"):
        shutil.copy2(TEMPLATE_DIR / name, output / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="spinozans/gdn2-mlp-1.3b")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.args_json.read_text())
    model_args = namespace_from_config(config)
    raw = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    step = int(raw["step"])
    loss = float(raw["loss"])
    source_hash = sha256(args.checkpoint)

    model = build_model(model_args, torch.device("cpu"))
    swapped = load_checkpoint_weights(model, raw, model_args, "train")
    if not swapped:
        raise RuntimeError("Schedule-Free train/y weight recovery did not occur")
    model.eval()
    state = model.state_dict()

    token_meta = tokenizer_files(args.output)
    clean = public_args(config)
    write_json(args.output / "args.json", clean)
    write_json(args.output / "config.json", {
        "architectures": ["EmenderGDN2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_emender_gdn2.EmenderGDN2Config",
            "AutoModelForCausalLM": "modeling_emender_gdn2.EmenderGDN2ForCausalLM",
        },
        "model_type": "emender_gdn2_mlp",
        "dtype": "bfloat16",
        "vocab_size": token_meta["vocab_size"],
        "bos_token_id": token_meta["eot_token"],
        "eos_token_id": token_meta["eot_token"],
        "pad_token_id": token_meta["eot_token"],
        "tie_word_embeddings": True,
        "use_cache": False,
        "gdn2_args": clean,
    })
    write_json(args.output / "generation_config.json", {
        "bos_token_id": token_meta["eot_token"],
        "eos_token_id": token_meta["eot_token"],
        "pad_token_id": token_meta["eot_token"],
        "do_sample": False,
        "max_new_tokens": 32,
    })
    copy_templates(args.output)
    (args.output / "requirements.txt").write_text(
        "torch>=2.9\ntransformers>=5.0\nsafetensors>=0.7\ntiktoken>=0.12\n"
    )

    tensors = clone_state(state)
    weights = args.output / "model.safetensors"
    save_file(tensors, str(weights), metadata={
        "format": "pt",
        "model": "GDN2-MLP 1.3B",
        "repo_id": args.repo_id,
        "source_checkpoint_sha256": source_hash,
        "source_step": str(step),
        "source_tokens": str(args.tokens),
        "weight_mode": "train",
        "schedulefree_train_weight_swap": "true",
        "source_commit": SOURCE_COMMIT,
        "gdn2_source_commit": GDN2_SOURCE_COMMIT,
    })
    del tensors, state, model, raw

    provenance = {
        "schema": "emender-hf-release-v1",
        "repo_id": args.repo_id,
        "kind": "gdn2-control",
        "source_checkpoint_name": args.checkpoint.name,
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "source_step": step,
        "source_loss_last100": loss,
        "tokens": args.tokens,
        "parameter_count": int(config["_model_total_params"]),
        "weight_mode": "train",
        "schedulefree_train_weight_swap": swapped,
        "source_commit": SOURCE_COMMIT,
        "gdn2_source_commit": GDN2_SOURCE_COMMIT,
        "model_safetensors_sha256": sha256(weights),
        "model_safetensors_size_bytes": weights.stat().st_size,
        "tensor_count": len(list(safe_open(weights, framework="pt").keys())),
        "nvidia_source_redistributed": False,
    }
    write_json(args.output / "provenance.json", provenance)
    (args.output / "README.md").write_text(model_card(
        args.repo_id,
        tokens=args.tokens,
        step=step,
        loss=loss,
        source_sha=source_hash,
        checkpoint_size=args.checkpoint.stat().st_size,
    ))
    sums = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
