#!/usr/bin/env python3
"""Build the frozen, tokenizer-bound panel for paired E97-MoE evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
import tiktoken
import torch

SCHEMA = "emender-e97-moe-paired-eval-panel-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evenly_spaced_windows(tokens: list[int], context: int, count: int):
    maximum = len(tokens) - context - 1
    if maximum < 0:
        raise RuntimeError(f"WikiText test has only {len(tokens)} tokens for context {context}")
    offsets = [round(index * maximum / max(1, count - 1)) for index in range(count)]
    return [
        {"id": f"wikitext-103-test-c{context}-o{offset}",
         "offset": offset,
         "tokens": torch.tensor(tokens[offset:offset + context + 1], dtype=torch.int32)}
        for offset in offsets
    ]


def stratified_mmlu(rows: list[dict], count: int, seed: int):
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    rng = random.Random(seed)
    for values in by_subject.values():
        rng.shuffle(values)
    subjects = sorted(by_subject)
    selected = []
    cursor = {subject: 0 for subject in subjects}
    while len(selected) < count:
        progressed = False
        for subject in subjects:
            position = cursor[subject]
            if position < len(by_subject[subject]):
                row = by_subject[subject][position]
                cursor[subject] += 1
                selected.append({
                    "id": f"mmlu-{subject}-{position}",
                    "subject": subject,
                    "question": row["question"],
                    "choices": list(row["choices"]),
                    "answer": int(row["answer"]),
                })
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise RuntimeError("not enough MMLU rows")
    rng.shuffle(selected)
    return selected


def random_rows(rows: list[dict], count: int, seed: int):
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), count)
    return [{
        "id": f"hellaswag-{rows[index]['ind']}",
        "context": rows[index]["ctx"],
        "choices": list(rows[index]["endings"]),
        "answer": int(rows[index]["label"]),
    } for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=970035)
    args = parser.parse_args()

    inputs = {
        "wikitext": args.raw_root / "wikitext/wikitext-103-v1/test-00000-of-00001.parquet",
        "mmlu": args.raw_root / "mmlu/all/test-00000-of-00001.parquet",
        "hellaswag": args.raw_root / "hellaswag/data/validation-00000-of-00001.parquet",
    }
    for path in inputs.values():
        path.resolve(strict=True)

    encoding = tiktoken.get_encoding("p50k_base")
    wiki_rows = pq.read_table(inputs["wikitext"]).to_pylist()
    wiki_text = "\n".join(row["text"] for row in wiki_rows)
    wiki_tokens = encoding.encode_ordinary(wiki_text)
    contexts = {2048: 64, 8192: 32, 32768: 8}
    wikitext = {
        str(context): evenly_spaced_windows(wiki_tokens, context, count)
        for context, count in contexts.items()
    }

    mmlu_rows = pq.read_table(inputs["mmlu"]).to_pylist()
    hellaswag_rows = pq.read_table(inputs["hellaswag"]).to_pylist()
    prompts = [
        {"id": "concise-explanation", "prompt": "User: Explain why the sky is blue in three concise sentences.\nAssistant:"},
        {"id": "constraint-list", "prompt": "User: Give exactly four ways to reduce memory use in a Python data pipeline. Use a numbered list and no introduction.\nAssistant:"},
        {"id": "arithmetic", "prompt": "User: A warehouse has 17 boxes with 24 items each and ships 89 items. How many remain? Show the calculation.\nAssistant:"},
        {"id": "counterfactual", "prompt": "User: If Earth had no axial tilt, name two major seasonal consequences and explain each briefly.\nAssistant:"},
        {"id": "python", "prompt": "User: Write a Python function that returns the first duplicate in an iterable, or None. Preserve encounter order and include a short example.\nAssistant:"},
        {"id": "shell", "prompt": "User: Write a safe bash command that finds .json files under a directory and prints their SHA-256 hashes, handling spaces in names.\nAssistant:"},
        {"id": "json-only", "prompt": "User: Return only valid JSON with keys answer and confidence. What is the capital of Japan?\nAssistant:"},
        {"id": "tool-call", "prompt": "User: The available tool is weather(city: string). Find the weather in Reykjavík. Respond with one compact JSON tool call.\nAssistant:"},
        {"id": "refusal", "prompt": "User: Invent a quotation and attribute it to a real scientist as if it were authentic.\nAssistant:"},
        {"id": "editing", "prompt": "User: Rewrite this sentence to remove ambiguity: 'Alex told Jordan that they had won after the interview.'\nAssistant:"},
        {"id": "summarize", "prompt": "User: Summarize in one sentence: Photosynthesis converts light energy into chemical energy, using carbon dioxide and water to produce sugars and oxygen.\nAssistant:"},
        {"id": "sql", "prompt": "User: Write SQL to return each customer's most recent order from orders(customer_id, order_id, created_at).\nAssistant:"},
        {"id": "debug", "prompt": "User: This Python loop skips elements when removing negatives: `for x in xs: if x < 0: xs.remove(x)`. Explain the bug and fix it.\nAssistant:"},
        {"id": "uncertainty", "prompt": "User: Without looking anything up, state whether you are certain who won the 2032 Olympic 100m final and explain why.\nAssistant:"},
        {"id": "translation", "prompt": "User: Translate 'The experiment was repeated three times' into Spanish, then give a literal English back-translation.\nAssistant:"},
        {"id": "planning", "prompt": "User: Make a five-step checklist for migrating a small database with minimal downtime.\nAssistant:"},
    ]

    panel = {
        "schema": SCHEMA,
        "seed": args.seed,
        "tokenizer": "p50k_base",
        "sources": {
            "wikitext": {"revision": WIKITEXT_REVISION, "sha256": sha256(inputs["wikitext"])},
            "mmlu": {"revision": MMLU_REVISION, "sha256": sha256(inputs["mmlu"])},
            "hellaswag": {"revision": HELLASWAG_REVISION, "sha256": sha256(inputs["hellaswag"])},
        },
        "wikitext": wikitext,
        "mmlu": stratified_mmlu(mmlu_rows, 256, args.seed),
        "hellaswag": random_rows(hellaswag_rows, 256, args.seed + 1),
        "generation_prompts": prompts,
        "retrieval": {"distances": [2048, 8192, 16384, 24576, 30720], "examples_per_distance": 32, "seed": args.seed + 2},
        "notes": {
            "wikitext": "External WikiText-103 test split; clean relative to post-training corpora, but pretraining contamination is not ruled out.",
            "mmlu": "External zero-shot multiple-choice diagnostic; benchmark contamination is not ruled out.",
            "hellaswag": "External validation continuation diagnostic; benchmark contamination is not ruled out.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(panel, args.output)
    receipt = {
        "schema": SCHEMA,
        "panel": str(args.output.resolve()),
        "panel_sha256": sha256(args.output),
        "bytes": args.output.stat().st_size,
        "counts": {"wikitext": contexts, "mmlu": 256, "hellaswag": 256, "generation": len(prompts)},
        "sources": panel["sources"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
