#!/usr/bin/env python3
"""Reproduce one fixed-world E97-linear rank stream on one GCD."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

import train
from ndm.data.tokenized_dataset import CounterSamplerIdentity, TokenizedStreamDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-world-size", type=int, required=True)
    parser.add_argument("--data-rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    saved_argv = sys.argv
    try:
        sys.argv = ["train.py"]
        train_args = train.parse_args()
    finally:
        sys.argv = saved_argv
    for key, value in config.items():
        if hasattr(train_args, key):
            setattr(train_args, key, value)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = train.build_training_model(train_args).to("cuda", dtype=torch.bfloat16)
    model.train()
    optimizer = train.build_training_optimizer(model, train_args)
    optimizer.train()
    identity = CounterSamplerIdentity(
        schema=config["sampler_schema"],
        corpus_sha256=config["sampler_corpus_sha256"],
        tokenizer_sha256=config["sampler_tokenizer_sha256"],
        sampler_key=config["sampler_key"],
        data_world_size=args.data_world_size,
        context_size=config["chunk_size"],
    )
    dataset = TokenizedStreamDataset(
        args.data, config["chunk_size"] + 1,
        rank=args.data_rank, world_size=args.data_world_size,
        tokenizer_name=config["tokenizer"], sampler_identity=identity,
        total_accepted_tokens=0,
        accepted_tokens_per_sample=config["chunk_size"],
    )
    for step in range(args.steps):
        chunks, _, _ = dataset.get_batch(config["batch_size"], device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(chunks, return_loss=True)
        finite_loss = bool(torch.isfinite(loss).item())
        if not finite_loss:
            print(json.dumps({"step": step, "loss": str(loss.item()),
                              "sample_ids": dataset.last_batch_sample_ids,
                              "finite": False}, sort_keys=True), flush=True)
            return 2
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        finite_grad = bool(torch.isfinite(grad_norm).item())
        print(json.dumps({"step": step, "loss": float(loss.item()),
                          "grad_norm": float(grad_norm.item()),
                          "sample_ids": dataset.last_batch_sample_ids,
                          "finite": finite_grad}, sort_keys=True), flush=True)
        if not finite_grad:
            return 3
        optimizer.step()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
