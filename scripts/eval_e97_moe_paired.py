#!/usr/bin/env python3
"""Read-only paired diagnostics for one canonical eight-shard E97-MoE model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tiktoken

from ndm.e97 import load_e97_checkpoint
from ndm.e97_moe_checkpoint import load_node_sharded_model
from ndm.e97_moe_ep import assert_node_local_ep_group, create_moe_process_groups
from ndm.models.e97_moe import (
    E97MoEConfig,
    convert_e97_ffns_to_node_local_moe,
    iter_e97_moe_layers,
)

SCHEMA = "emender-e97-moe-paired-eval-result-v1"
LETTERS = "ABCD"
POSITION_BUCKETS = (0, 2048, 4096, 8192, 16384, 24576, 32768)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-args-json", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--expert-backend", choices=("rocblas", "triton", "grouped"), default="rocblas")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def event(name: str, **fields) -> None:
    if dist.get_rank() == 0:
        print(json.dumps({"event": name, "time_unix": time.time(), **fields}, sort_keys=True), flush=True)


def gather_records(local):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return [item for rank_items in gathered for item in rank_items]


def local_tokens(tokens: list[int], device: torch.device):
    # Node-local EP is deliberately ragged: each source rank may route a
    # different number of tokens through the same ordered collectives.
    return torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)


def add_routing_counts(accumulator, model):
    for layer_index, layer in enumerate(iter_e97_moe_layers(model)):
        counts = layer.last_metrics.get("expert_token_counts")
        if counts is None:
            continue
        values = counts.detach().to(device="cpu", dtype=torch.float64)
        if accumulator[layer_index] is None:
            accumulator[layer_index] = values
        else:
            accumulator[layer_index] += values


def summarize_routing(accumulator):
    result = []
    for layer, counts in enumerate(accumulator):
        if counts is None:
            continue
        total = float(counts.sum().item())
        mean = float(counts.mean().item())
        probabilities = counts / max(total, 1.0)
        entropy = float((-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).item())
        result.append({
            "layer": layer,
            "total_assignments": int(total),
            "minimum": int(counts.min().item()),
            "maximum": int(counts.max().item()),
            "max_over_mean": float(counts.max().item() / max(mean, 1e-30)),
            "coefficient_of_variation": float(counts.std(unbiased=False).item() / max(mean, 1e-30)),
            "normalized_entropy": entropy / math.log(counts.numel()),
            "dropped_tokens": 0,
        })
    return result


def hidden_health(hiddens):
    finite = total = 0
    maximum = 0.0
    for value in hiddens:
        tensors = value if isinstance(value, (tuple, list)) else (value,)
        for tensor in tensors:
            mask = torch.isfinite(tensor)
            finite += int(mask.sum().item())
            total += tensor.numel()
            if bool(mask.any().item()):
                maximum = max(maximum, float(tensor[mask].float().abs().max().item()))
    return {"finite": finite, "elements": total, "finite_fraction": finite / max(total, 1), "max_abs": maximum}


def score_wikitext(model, panel, device, routing):
    local = []
    rank = dist.get_rank()
    world = dist.get_world_size()
    for context_text, examples in sorted(panel["wikitext"].items(), key=lambda pair: int(pair[0])):
        context = int(context_text)
        if len(examples) % world:
            raise RuntimeError(f"WikiText context {context} count must divide world size")
        for base in range(0, len(examples), world):
            example = examples[base + rank]
            tokens = example["tokens"].to(device=device, dtype=torch.long).unsqueeze(0)
            hiddens = None
            buckets = []
            boundaries = [value for value in POSITION_BUCKETS if value <= context]
            if boundaries[-1] != context:
                boundaries.append(context)
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                segment = tokens[:, start:stop + 1]
                with torch.inference_mode():
                    loss, (hiddens, _conv) = model(
                        segment, return_loss=True, return_prev_hiddens=True,
                        prev_hiddens=hiddens)
                add_routing_counts(routing, model)
                buckets.append({"start": start, "stop": stop, "tokens": stop - start, "nll": float(loss.item())})
            local.append({
                "id": example["id"], "context": context, "offset": int(example["offset"]),
                "tokens": context,
                "nll": sum(row["nll"] * row["tokens"] for row in buckets) / context,
                "position_buckets": buckets, "hidden_health": hidden_health(hiddens),
            })
        event("wikitext_context_complete", context=context, examples=len(examples))
    return gather_records(local)


def prompt_logits(model, encoding, prompt: str, device, routing):
    tokens = encoding.encode_ordinary(prompt)
    inputs = local_tokens(tokens, device)
    with torch.inference_mode():
        logits = model(inputs)
    add_routing_counts(routing, model)
    return logits[0, -1].float()


def score_mmlu(model, panel, encoding, device, routing):
    local = []
    rank, world = dist.get_rank(), dist.get_world_size()
    examples = panel["mmlu"]
    if len(examples) % world:
        raise RuntimeError("MMLU count must divide world size")
    letter_tokens = [encoding.encode_ordinary(" " + letter) for letter in LETTERS]
    if any(len(value) != 1 for value in letter_tokens):
        raise RuntimeError("p50k MMLU answer letters must each be one token")
    candidate_ids = [value[0] for value in letter_tokens]
    for base in range(0, len(examples), world):
        example = examples[base + rank]
        choices = "\n".join(f"{LETTERS[index]}. {choice}" for index, choice in enumerate(example["choices"]))
        prompt = f"Question: {example['question']}\n{choices}\nAnswer:"
        scores = prompt_logits(model, encoding, prompt, device, routing)[candidate_ids].cpu().tolist()
        predicted = max(range(4), key=lambda index: scores[index])
        local.append({"id": example["id"], "subject": example["subject"], "answer": example["answer"],
                      "prediction": predicted, "correct": predicted == example["answer"], "scores": scores})
    event("mmlu_complete", examples=len(examples))
    return gather_records(local)


def continuation_score(model, encoding, prompt: str, continuation: str, device, routing):
    prefix = encoding.encode_ordinary(prompt)
    full = encoding.encode_ordinary(prompt + continuation)
    if full[:len(prefix)] != prefix:
        raise RuntimeError("continuation tokenization changed the prompt prefix")
    inputs = local_tokens(full, device)
    with torch.inference_mode():
        logits = model(inputs)
    add_routing_counts(routing, model)
    start = len(prefix) - 1
    targets = torch.tensor(full[len(prefix):], device=device, dtype=torch.long)
    selected = logits[0, start:len(full) - 1].float()
    losses = F.cross_entropy(selected, targets, reduction="none")
    return float((-losses.sum()).item()), float((-losses.mean()).item()), len(targets)


def score_hellaswag(model, panel, encoding, device, routing):
    local = []
    rank, world = dist.get_rank(), dist.get_world_size()
    examples = panel["hellaswag"]
    if len(examples) % world:
        raise RuntimeError("HellaSwag count must divide world size")
    for base in range(0, len(examples), world):
        example = examples[base + rank]
        raw, normalized, lengths = [], [], []
        for choice in example["choices"]:
            continuation = choice if choice.startswith(" ") else " " + choice
            score, norm, count = continuation_score(
                model, encoding, example["context"], continuation, device, routing)
            raw.append(score); normalized.append(norm); lengths.append(count)
        raw_prediction = max(range(4), key=lambda index: raw[index])
        norm_prediction = max(range(4), key=lambda index: normalized[index])
        local.append({"id": example["id"], "answer": example["answer"],
                      "raw_prediction": raw_prediction, "normalized_prediction": norm_prediction,
                      "raw_correct": raw_prediction == example["answer"],
                      "normalized_correct": norm_prediction == example["answer"],
                      "raw_scores": raw, "normalized_scores": normalized, "choice_tokens": lengths})
    event("hellaswag_complete", examples=len(examples))
    return gather_records(local)


def advance_without_logits(model, tokens, device, routing, segment_size=4096):
    hiddens = None
    processed = 0
    while len(tokens) - processed > segment_size:
        segment = torch.tensor(tokens[processed:processed + segment_size + 1], device=device).unsqueeze(0)
        with torch.inference_mode():
            _loss, (hiddens, _conv) = model(
                segment, return_loss=True, return_prev_hiddens=True, prev_hiddens=hiddens)
        add_routing_counts(routing, model)
        processed += segment_size
    tail = torch.tensor(tokens[processed:], device=device).unsqueeze(0)
    with torch.inference_mode():
        logits, (hiddens, _conv) = model(
            tail, return_prev_hiddens=True, prev_hiddens=hiddens)
    add_routing_counts(routing, model)
    return logits[0, -1].float(), hiddens


def retrieval_example(encoding, distance, seed):
    rng = random.Random(seed)
    values = [" blue", " green", " red", " yellow", " white", " black", " orange", " purple"]
    ids = [encoding.encode_ordinary(value) for value in values]
    if any(len(value) != 1 for value in ids):
        raise RuntimeError("retrieval candidates must be single p50k tokens")
    answer = rng.randrange(len(values))
    declaration = encoding.encode_ordinary(
        f"Memorize this fact exactly: the passkey value is{values[answer]}.\n")
    query = encoding.encode_ordinary("\nQuestion: What is the passkey value? The passkey value is")
    filler_source = encoding.encode_ordinary(
        " This is neutral filler text about rivers, stones, clouds, books, lamps, roads, gardens, and ordinary daily events.")
    filler_count = max(0, distance - len(declaration) - len(query))
    filler = (filler_source * (filler_count // len(filler_source) + 1))[:filler_count]
    return declaration + filler + query, answer, [value[0] for value in ids]


def score_retrieval(model, panel, encoding, device, routing):
    local = []
    rank, world = dist.get_rank(), dist.get_world_size()
    spec = panel["retrieval"]
    count = int(spec["examples_per_distance"])
    if count % world:
        raise RuntimeError("retrieval count must divide world size")
    for distance in spec["distances"]:
        for base in range(0, count, world):
            index = base + rank
            tokens, answer, candidates = retrieval_example(
                encoding, int(distance), int(spec["seed"]) + int(distance) * 1000 + index)
            logits, hiddens = advance_without_logits(model, tokens, device, routing)
            scores = logits[candidates].cpu().tolist()
            predicted = max(range(len(scores)), key=lambda choice: scores[choice])
            distractor = max(score for choice, score in enumerate(scores) if choice != answer)
            local.append({"id": f"retrieval-d{distance}-{index}", "distance": int(distance),
                          "answer": answer, "prediction": predicted, "correct": predicted == answer,
                          "margin": scores[answer] - distractor, "scores": scores,
                          "hidden_health": hidden_health(hiddens)})
        event("retrieval_distance_complete", distance=int(distance), examples=count)
    return gather_records(local)


def generate(model, panel, encoding, device, routing, maximum_tokens):
    local = []
    rank, world = dist.get_rank(), dist.get_world_size()
    prompts = panel["generation_prompts"]
    if len(prompts) % world:
        raise RuntimeError("generation prompt count must divide world size")
    eos = encoding.eot_token
    for base in range(0, len(prompts), world):
        example = prompts[base + rank]
        prompt_tokens = encoding.encode_ordinary(example["prompt"])
        inputs = local_tokens(prompt_tokens, device)
        with torch.inference_mode():
            logits, (hiddens, _conv) = model(inputs, return_prev_hiddens=True)
        add_routing_counts(routing, model)
        generated = []
        token = int(logits[0, -1].argmax().item())
        finished = False
        for _ in range(maximum_tokens):
            if not finished:
                generated.append(token)
                finished = token == eos
            input_token = torch.tensor([[token]], device=device)
            with torch.inference_mode():
                next_logits, (hiddens, _conv) = model(
                    input_token, return_prev_hiddens=True, prev_hiddens=hiddens)
            token = int(next_logits[0, -1].argmax().item()) if not finished else eos
        add_routing_counts(routing, model)
        local.append({"id": example["id"], "prompt": example["prompt"],
                      "generated_tokens": len(generated), "stopped_on_eot": bool(generated and generated[-1] == eos),
                      "response": encoding.decode(generated)})
    event("generation_complete", examples=len(prompts), maximum_tokens=maximum_tokens)
    return gather_records(local)


def accuracy(rows, field="correct"):
    return sum(bool(row[field]) for row in rows) / max(1, len(rows))


def aggregate(result):
    wiki = {}
    for context in (2048, 8192, 32768):
        rows = [row for row in result["wikitext"] if row["context"] == context]
        total = sum(row["tokens"] for row in rows)
        wiki[str(context)] = {
            "examples": len(rows), "tokens": total,
            "nll": sum(row["nll"] * row["tokens"] for row in rows) / total,
        }
        if context == 32768:
            positions = {}
            for row in rows:
                for bucket in row["position_buckets"]:
                    key = f"{bucket['start']}-{bucket['stop']}"
                    positions.setdefault(key, []).append(bucket)
            wiki[str(context)]["position_buckets"] = {
                key: sum(item["nll"] * item["tokens"] for item in values) / sum(item["tokens"] for item in values)
                for key, values in positions.items()
            }
    result["summary"] = {
        "wikitext": wiki,
        "mmlu_accuracy": accuracy(result["mmlu"]),
        "hellaswag_accuracy": accuracy(result["hellaswag"], "raw_correct"),
        "hellaswag_normalized_accuracy": accuracy(result["hellaswag"], "normalized_correct"),
        "retrieval_accuracy_by_distance": {
            str(distance): accuracy([row for row in result["retrieval"] if row["distance"] == distance])
            for distance in sorted({row["distance"] for row in result["retrieval"]})
        },
        "retrieval_margin_by_distance": {
            str(distance): sum(row["margin"] for row in result["retrieval"] if row["distance"] == distance)
            / len([row for row in result["retrieval"] if row["distance"] == distance])
            for distance in sorted({row["distance"] for row in result["retrieval"]})
        },
    }


def main():
    args = parse_args()
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError("paired evaluation runner requires exactly one eight-GCD node")
        groups = create_moe_process_groups()
        if groups.local_rank != local_rank or groups.node_count != 1:
            raise RuntimeError("evaluation process group is not one contiguous eight-rank island")
        topology = assert_node_local_ep_group(groups.node_group)
        device = torch.device("cuda")
        torch.manual_seed(970035)
        panel_sha = sha256(args.panel)
        panel = torch.load(args.panel, map_location="cpu", weights_only=False)
        if panel.get("schema") != "emender-e97-moe-paired-eval-panel-v1":
            raise RuntimeError("unexpected evaluation panel schema")
        event("load_start", label=args.label, generation=str(args.generation), panel_sha256=panel_sha)
        loaded = load_e97_checkpoint(
            args.seed_checkpoint, device="cuda", dtype=torch.bfloat16,
            weight_mode="train", use_triton=True, mmap=True, args_json=args.seed_args_json)
        if loaded.step != 2322520 or int(loaded.config.get("dim", -1)) != 1792:
            raise RuntimeError("dense graph seed is not the bound final E97 authority")
        model = loaded.model
        model.gradient_checkpointing = False
        model.loss_chunk_size = 2048
        for module in model.modules():
            if hasattr(module, "checkpoint_interval"):
                module.checkpoint_interval = 16
                module.projection_chunk_size = 2048
        convert_e97_ffns_to_node_local_moe(
            model,
            E97MoEConfig(hidden_dim=8832, routed_experts=64, shared_experts=1, top_k=3,
                         expert_parallel_size=8, expert_backend=args.expert_backend),
            local_expert_rank=local_rank, expert_group=groups.node_group)
        model.eval()
        manifest = load_node_sharded_model(
            args.generation, model, node_group=groups.node_group,
            verify_sha256=args.verify_sha256)
        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        event("model_ready", checkpoint_step=manifest["step"], accepted_tokens=manifest["accepted_tokens"],
              source_commit=manifest["source_commit"], hostname=topology.hostname)

        encoding = tiktoken.get_encoding(panel["tokenizer"])
        routing = [None] * len(tuple(iter_e97_moe_layers(model)))
        result = {
            "schema": SCHEMA, "label": args.label, "panel_sha256": panel_sha,
            "checkpoint": {"generation": str(args.generation.resolve()),
                           "step": int(manifest["step"]), "accepted_tokens": int(manifest["accepted_tokens"]),
                           "source_commit": manifest["source_commit"]},
            "wikitext": score_wikitext(model, panel, device, routing),
            "mmlu": score_mmlu(model, panel, encoding, device, routing),
            "hellaswag": score_hellaswag(model, panel, encoding, device, routing),
            "retrieval": score_retrieval(model, panel, encoding, device, routing),
            "generations": generate(model, panel, encoding, device, routing, args.generation_tokens),
        }
        if dist.get_rank() == 0:
            result["routing"] = summarize_routing(routing)
            result["peak_hbm_bytes"] = int(torch.cuda.max_memory_allocated())
            aggregate(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            event("evaluation_complete", output=str(args.output), summary=result["summary"],
                  peak_hbm_bytes=result["peak_hbm_bytes"])
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
