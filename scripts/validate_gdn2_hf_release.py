#!/usr/bin/env python3
"""Validate a local or downloaded GDN2 Hugging Face release."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_checkpoint import build_model, load_checkpoint_weights, namespace_from_config
from scripts.prepare_e97_hf_release import sha256, write_json


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--args-json", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--fused-cuda-parity", action="store_true")
    args = ap.parse_args()

    config = json.loads(args.args_json.read_text())
    model_args = namespace_from_config(config)
    raw = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    source = build_model(model_args, torch.device("cpu"))
    swapped = load_checkpoint_weights(source, raw, model_args, "train")
    if not swapped:
        raise RuntimeError("Schedule-Free train/y swap was not applied")
    source.eval()

    weights = args.release / "model.safetensors"
    source_state = source.state_dict()
    mismatch = []
    with safe_open(weights, framework="pt", device="cpu") as handle:
        safe_keys = set(handle.keys())
        expected = {f"model.{name}" for name in source_state}
        missing = sorted(expected - safe_keys)
        unexpected = sorted(safe_keys - expected)
        for name, tensor in source_state.items():
            exported = handle.get_tensor(f"model.{name}")
            if not torch.equal(tensor.detach().cpu(), exported):
                mismatch.append(name)
    if missing or unexpected or mismatch:
        raise RuntimeError(
            f"safetensors mismatch missing={missing} unexpected={unexpected} values={mismatch}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.release)
    prompt = "The theorem states"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if ids != [464, 44728, 2585]:
        raise RuntimeError(f"unexpected p50k_base tokenization: {ids}")

    hf_model, loading = AutoModelForCausalLM.from_pretrained(
        args.release,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    load_failures = {
        key: loading.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(load_failures.values()):
        raise RuntimeError(f"strict load failed: {load_failures}")
    hf_model.eval()
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = hf_model(input_ids).logits
        generated = hf_model.generate(input_ids, max_new_tokens=1, do_sample=False)
    if logits.shape != (1, 3, 50281) or not torch.isfinite(logits).all():
        raise RuntimeError(f"bad local logits shape/finite: {logits.shape}")

    # Reconstruct the standalone reference directly from the recovered source
    # checkpoint state, independently of from_pretrained, and require bit-exact
    # CPU logits against the exported Transformers wrapper.
    module = sys.modules[hf_model.__class__.__module__]
    release_args = json.loads((args.release / "args.json").read_text())
    portable_source = module.PortableGDN2LM(release_args, vocab_size=50281).bfloat16().eval()
    portable_source.load_state_dict(source_state, strict=True)
    with torch.no_grad():
        source_portable_logits = portable_source(input_ids)
    portable_logits_bit_exact = bool(torch.equal(source_portable_logits, logits))
    if not portable_logits_bit_exact:
        raise RuntimeError("portable source-checkpoint and exported logits are not bit-exact")
    del portable_source

    fused = None
    if args.fused_cuda_parity:
        if not torch.cuda.is_available():
            raise RuntimeError("--fused-cuda-parity requested without CUDA")
        device = torch.device("cuda")
        source = source.to(device).eval()
        with torch.no_grad():
            fused_logits = source(input_ids.to(device), return_loss=False).float().cpu()
        source = source.cpu()
        torch.cuda.empty_cache()
        hf_model = hf_model.to(device).eval()
        with torch.no_grad():
            portable_logits = hf_model(input_ids.to(device)).logits.float().cpu()
        difference = (portable_logits - fused_logits).abs()
        fused = {
            "max_abs_logit_delta": float(difference.max()),
            "mean_abs_logit_delta": float(difference.mean()),
            "argmax_equal": bool(torch.equal(portable_logits.argmax(-1), fused_logits.argmax(-1))),
            "fused_logits_sha256": tensor_digest(fused_logits),
            "portable_cuda_logits_sha256": tensor_digest(portable_logits),
        }

    result = {
        "schema": "emender-gdn2-hf-validation-v1",
        "repo_id": json.loads((args.release / "provenance.json").read_text())["repo_id"],
        "source_checkpoint_name": args.checkpoint.name,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "model_safetensors_sha256": sha256(weights),
        "tensor_count": len(source_state),
        "tensor_key_and_value_parity": True,
        "schedulefree_train_weight_swap": True,
        "transformers_strict_load": load_failures,
        "tokenizer": "p50k_base",
        "prompt": prompt,
        "prompt_token_ids": ids,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "portable_source_checkpoint_logits_bit_exact": portable_logits_bit_exact,
        "portable_source_logits_sha256": tensor_digest(source_portable_logits.float()),
        "portable_export_logits_sha256": tensor_digest(logits.float()),
        "generated_token_ids": generated[0].tolist(),
        "generated_new_token_id": int(generated[0, -1]),
        "fused_cuda_parity": fused,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
