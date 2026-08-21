#!/usr/bin/env python3
"""Build curated Hugging Face artifacts from dense E97 checkpoints.

The export is weights-only: raw pickle checkpoints and optimizer state are never
copied into the release directory. ScheduleFree base checkpoints must be
exported with ``--weight-mode train``; agent checkpoints store their evaluated
weights directly and use ``--weight-mode saved``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ndm.e97 import e97_checkpoint_config, load_e97_checkpoint

SOURCE_COMMIT = "2b5d135587cca9d1081dbbdf7b35a8ac2629db5d"

CONFIGURATION = '''\
from transformers import PretrainedConfig


class EmenderConfig(PretrainedConfig):
    model_type = "emender_e97"

    def __init__(self, e97_args=None, vocab_size=50281, **kwargs):
        super().__init__(
            bos_token_id=kwargs.pop("bos_token_id", 50256),
            eos_token_id=kwargs.pop("eos_token_id", 50256),
            pad_token_id=kwargs.pop("pad_token_id", 50256),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", True),
            use_cache=kwargs.pop("use_cache", False),
            is_decoder=True,
            **kwargs,
        )
        self.e97_args = dict(e97_args or {})
        self.vocab_size = int(vocab_size)
        self.num_hidden_layers = int(self.e97_args.get("depth", 0))
'''

MODELING = '''\
from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ndm.e97 import build_e97_model
from .configuration_emender import EmenderConfig


def _replace_cpu_rmsnorms(module):
    for name, child in list(module.named_children()):
        if child.__class__.__module__.startswith("mamba_ssm.") and hasattr(child, "weight"):
            replacement = torch.nn.RMSNorm(
                int(child.weight.numel()), eps=float(getattr(child, "eps", 1e-6))
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
            module._modules[name] = replacement
        else:
            _replace_cpu_rmsnorms(child)


class EmenderForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = EmenderConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _tied_weights_keys = {"model.lm_head.weight": "model.embedding.weight"}

    def __init__(self, config: EmenderConfig):
        super().__init__(config)
        # Portable loading defaults to the exact native CPU path. Users running
        # the qualified ROCm stack may reconstruct with use_triton=True.
        self.model = build_e97_model(
            config.e97_args, vocab_size=config.vocab_size, use_triton=False
        )
        if hasattr(self.model, "fused_add_norm"):
            self.model.fused_add_norm = False
        _replace_cpu_rmsnorms(self.model)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        if hasattr(self, "model"):
            self.model.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del attention_mask, kwargs
        if input_ids is None:
            raise ValueError("input_ids are required")
        logits = self.model(input_ids, return_loss=False)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
'''


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tokenizer_files(out: Path) -> dict[str, int]:
    import tiktoken
    from transformers.convert_slow_tokenizer import TikTokenConverter

    encoding = tiktoken.get_encoding("p50k_base")
    bpe = out / "tokenizer.model"
    with bpe.open("wb") as handle:
        for token, rank in sorted(encoding._mergeable_ranks.items(), key=lambda x: x[1]):
            handle.write(base64.b64encode(token) + b" " + str(rank).encode() + b"\n")
    tokenizer = TikTokenConverter(
        vocab_file=str(bpe),
        pattern=encoding._pat_str,
        additional_special_tokens=encoding._special_tokens,
    ).converted()
    tokenizer.save(str(out / "tokenizer.json"))
    write_json(out / "tokenizer_config.json", {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_input_names": ["input_ids", "attention_mask"],
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
        "name_or_path": "p50k_base",
    })
    write_json(out / "special_tokens_map.json", {
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
    })
    return {"vocab_size": encoding.n_vocab, "eot_token": encoding.eot_token}


def public_args(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "level", "dim", "depth", "n_heads", "n_state", "expansion", "n_groups",
        "n_slots", "use_gate", "gate_activation", "linear_state", "use_write_gate",
        "e88_decay_mode", "e88_value_residual", "e88_raw_write", "use_chunked_e97",
        "e97_chunk_size", "state_expansion", "r_h_mode", "use_conv", "d_conv",
        "gdn2_mlp_ratio", "dropout", "checkpoint_interval", "projection_chunk_size",
        "loss_chunk_size", "mlp_ratio", "mlp_multiple", "state_summary_dim",
        "mlp_hidden", "tokenizer", "use_triton",
    )
    result = {key: config[key] for key in keys if key in config}
    # Remote-code loading chooses its runtime kernel explicitly.
    result["use_triton"] = False
    return result


def clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    seen: set[tuple[int, int, int]] = set()
    for name, tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        key = (value.untyped_storage().data_ptr(), value.storage_offset(), value.numel())
        if key in seen:
            value = value.clone()
        seen.add(key)
        tensors[f"model.{name}"] = value
    return tensors


def base_card(repo_id: str, tokens: int, step: int, loss: float, source_sha: str) -> str:
    return f'''---
license: other
pipeline_tag: text-generation
tags:
- recurrent-language-model
- emender
- e97
- diloco
- pytorch
---

# Emender E97 1.3B

Emender E97 is a 1.287B-parameter recurrent base language model with nonlinear
matrix state and split erase/write control inspired by Gated DeltaNet 2. It is
an **E97 split-edit model**, not the repository's separate GDN2 control.

This revision contains the checkpoint after **{tokens:,} training tokens**
(step `{step}`, recorded loss `{loss:.4f}`). The default repository revision is
the later 513B authority; the 150B authority is retained as a historical
reproducibility revision.

## Artifact

- Source checkpoint SHA-256: `{source_sha}`
- Export: evaluated ScheduleFree train/y weights only
- Tokenizer: `p50k_base`
- Context used in training: 2,048 tokens
- Training approach: local DiLoCo with periodic model averaging
- Raw optimizer state and pickle checkpoint: not included

The model is a raw base LM, not an instruction or chat model. A full HellaSwag
audit of the 513B authority scored `0.3651663` normalized accuracy; GPT-2 XL in
the same harness scored `0.4890460`.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}", trust_remote_code=True, dtype="auto"
)
```

The custom loader requires the matching Emender source package:

```bash
pip install "git+https://github.com/spinozans/emender.git@{SOURCE_COMMIT}"
```

The qualified fused path targets AMD ROCm/Frontier. The bundled Transformers
wrapper defaults to the slower native portable path.

## Limitations and license

This is a research checkpoint with limited benchmark coverage. It can produce
incorrect, biased, or unsafe text. No standalone model license has yet been
selected; `license: other` is intentional and no permissive license should be
inferred from repository visibility.
'''


def agent_card(repo_id: str, source_sha: str) -> str:
    return f'''---
license: other
pipeline_tag: text-generation
tags:
- recurrent-language-model
- emender
- e97
- tool-use
- cli-agent
- pytorch
---

# Emender E97 1.3B CLI Agent

A narrow experimental recurrent CLI policy derived from the 513B-token dense
E97 base. It is trained for one exact RS-free Pi protocol:

`repository objective -> sandboxed direct CLI argv -> typed observation -> grounded submit_answer`

This is **not a general instruction or coding assistant**.

## Verified behavior

The immutable checkpoint passed **40/40** held-out real-Pi direct CLI tasks:

| family | result |
|---|---:|
| count | 10/10 |
| JSON extraction | 10/10 |
| file read | 10/10 |
| search | 10/10 |

All 40 runs used exactly `cli -> submit_answer` and passed argv, value, evidence,
grounding, bounded-sequence, completion, and clean-exit checks.

- Source checkpoint SHA-256: `{source_sha}`
- Base: dense E97 513B authority
- Weight semantics: saved evaluated agent weights
- Tokenizer: `p50k_base`
- Raw pickle checkpoint: not included

## Important protocol dependency

The result depends on the exact system prompt, tool schema, assistant-target
normalization, structured-action stopping, recurrent cache semantics, and
hash-pinned cwd-only Apptainer sandbox in the Emender repository. Ordinary text
generation does not reproduce the agent evaluation.

See `configs/pi/e97-cli-tools.ts`, `scripts/run_e97_sandbox_cli.py`, and
`docs/E97_DENSE_PI_AGENT_EXECUTION_PLAN.md` at commit `{SOURCE_COMMIT}`.

## Loading weights

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}", trust_remote_code=True, dtype="auto"
)
```

Install the matching model implementation:

```bash
pip install "git+https://github.com/spinozans/emender.git@{SOURCE_COMMIT}"
```

## Limitations and license

The policy has not demonstrated autonomous tool discovery or general software
engineering ability. Its sandbox does not claim arbitrary hostile-code
isolation. No standalone model license has yet been selected; `license: other`
is intentional.
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("base", "agent"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--weight-mode", choices=("train", "saved"), required=True)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.force and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(args.checkpoint)

    raw = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = e97_checkpoint_config(args.checkpoint, raw, args.args_json)
    step = int(raw.get("step") or 0)
    loss = float(raw.get("loss") or 0.0)
    if args.kind == "base":
        if args.weight_mode != "train" or not args.tokens:
            raise ValueError("base release requires --weight-mode train and --tokens")
        del raw
        loaded = load_e97_checkpoint(
            args.checkpoint, args_json=args.args_json, device="cpu", dtype=torch.bfloat16,
            weight_mode="train", use_triton=False, mmap=True,
        )
        state = loaded.model.state_dict()
        schedulefree_swap = loaded.schedulefree_train_weight_swap
    else:
        if args.weight_mode != "saved":
            raise ValueError("agent release requires --weight-mode saved")
        state = raw["model_state_dict"]
        schedulefree_swap = False

    token_meta = tokenizer_files(args.output)
    clean = public_args(config)
    write_json(args.output / "args.json", clean)
    write_json(args.output / "config.json", {
        "architectures": ["EmenderForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_emender.EmenderConfig",
            "AutoModelForCausalLM": "modeling_emender.EmenderForCausalLM",
        },
        "model_type": "emender_e97", "dtype": "bfloat16",
        "vocab_size": token_meta["vocab_size"],
        "bos_token_id": token_meta["eot_token"],
        "eos_token_id": token_meta["eot_token"],
        "pad_token_id": token_meta["eot_token"],
        "tie_word_embeddings": True, "use_cache": False, "e97_args": clean,
    })
    write_json(args.output / "generation_config.json", {
        "bos_token_id": token_meta["eot_token"],
        "eos_token_id": token_meta["eot_token"],
        "pad_token_id": token_meta["eot_token"],
        "do_sample": False, "max_new_tokens": 32,
    })
    (args.output / "configuration_emender.py").write_text(CONFIGURATION)
    (args.output / "modeling_emender.py").write_text(MODELING)
    (args.output / "requirements.txt").write_text(
        f"torch>=2.9\ntransformers>=5.0\nsafetensors>=0.7\ntiktoken>=0.12\n"
        f"git+https://github.com/spinozans/emender.git@{SOURCE_COMMIT}\n"
    )

    tensors = clone_state(state)
    weights = args.output / "model.safetensors"
    save_file(tensors, str(weights), metadata={
        "format": "pt", "model": "Emender E97",
        "repo_id": args.repo_id, "source_checkpoint_sha256": source_hash,
        "source_step": str(step), "weight_mode": args.weight_mode,
        "schedulefree_train_weight_swap": str(schedulefree_swap).lower(),
        "source_commit": SOURCE_COMMIT,
    })
    del tensors, state

    weights_hash = sha256(weights)
    provenance = {
        "schema": "emender-hf-release-v1", "repo_id": args.repo_id,
        "kind": args.kind, "source_checkpoint_name": args.checkpoint.name,
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "source_step": step, "source_loss": loss,
        "tokens": args.tokens, "weight_mode": args.weight_mode,
        "schedulefree_train_weight_swap": schedulefree_swap,
        "source_commit": SOURCE_COMMIT,
        "model_safetensors_sha256": weights_hash,
        "model_safetensors_size_bytes": weights.stat().st_size,
        "tensor_count": len(list(safe_open(weights, framework="pt").keys())),
    }
    write_json(args.output / "provenance.json", provenance)
    if args.kind == "base":
        (args.output / "README.md").write_text(base_card(
            args.repo_id, args.tokens, step, loss, source_hash
        ))
    else:
        (args.output / "README.md").write_text(agent_card(args.repo_id, source_hash))
    sums = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
