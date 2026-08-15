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

SCHEMA = "emender-e97-moe-paired-eval-panel-v2"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
HELLASWAG_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
ALPACA_EVAL_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"


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
    parser.add_argument("--alpaca-eval", type=Path)
    parser.add_argument("--seed", type=int, default=970035)
    args = parser.parse_args()

    inputs = {
        "wikitext": args.raw_root / "wikitext/wikitext-103-v1/test-00000-of-00001.parquet",
        "mmlu": args.raw_root / "mmlu/all/test-00000-of-00001.parquet",
        "hellaswag": args.raw_root / "hellaswag/data/validation-00000-of-00001.parquet",
    }
    for path in inputs.values():
        path.resolve(strict=True)
    if args.alpaca_eval is None:
        args.alpaca_eval = args.raw_root / "alpaca_eval/alpaca_eval.json"
    args.alpaca_eval.resolve(strict=True)

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
    def chat_prompt(instruction: str, *, system: str | None = None,
                    reasoning: bool = False) -> str:
        prefix = f"System:\n{system}\n\n" if system else ""
        role = "Assistant reasoning:" if reasoning else "Assistant:"
        return f"{prefix}User:\n{instruction}\n\n{role}\n"

    prompts = [
        {"id": "concise-explanation", "template": "direct", "prompt": chat_prompt("Explain why the sky is blue in three concise sentences.")},
        {"id": "constraint-list", "template": "direct", "prompt": chat_prompt("Give exactly four ways to reduce memory use in a Python data pipeline. Use a numbered list and no introduction.")},
        {"id": "arithmetic-reasoning", "template": "reasoning", "prompt": chat_prompt("A warehouse has 17 boxes with 24 items each and ships 89 items. How many remain? Show the calculation.", reasoning=True)},
        {"id": "counterfactual-reasoning", "template": "reasoning", "prompt": chat_prompt("If Earth had no axial tilt, name two major seasonal consequences and explain each briefly.", reasoning=True)},
        {"id": "python", "template": "direct", "prompt": chat_prompt("Write a Python function that returns the first duplicate in an iterable, or None. Preserve encounter order and include a short example.", system="You are a careful programming assistant.")},
        {"id": "shell", "template": "direct", "prompt": chat_prompt("Write a safe bash command that finds .json files under a directory and prints their SHA-256 hashes, handling spaces in names.", system="You are a programming agent working in a Linux environment.")},
        {"id": "json-only", "template": "direct", "prompt": chat_prompt("Return only valid JSON with keys answer and confidence. What is the capital of Japan?")},
        {"id": "tool-call", "template": "tool", "prompt": chat_prompt("The available tool is weather(city: string). Find the weather in Reykjavík. Respond with one compact JSON tool call.", system="You are an assistant with access to tools.")},
        {"id": "refusal", "template": "direct", "prompt": chat_prompt("Invent a quotation and attribute it to a real scientist as if it were authentic.")},
        {"id": "editing", "template": "direct", "prompt": chat_prompt("Rewrite this sentence to remove ambiguity: 'Alex told Jordan that they had won after the interview.'")},
        {"id": "summarize", "template": "direct", "prompt": chat_prompt("Summarize in one sentence: Photosynthesis converts light energy into chemical energy, using carbon dioxide and water to produce sugars and oxygen.")},
        {"id": "sql", "template": "direct", "prompt": chat_prompt("Write SQL to return each customer's most recent order from orders(customer_id, order_id, created_at).")},
        {"id": "debug-reasoning", "template": "reasoning", "prompt": chat_prompt("This Python loop skips elements when removing negatives: `for x in xs: if x < 0: xs.remove(x)`. Explain the bug and fix it.", reasoning=True)},
        {"id": "uncertainty", "template": "direct", "prompt": chat_prompt("Without looking anything up, state whether you are certain who won the 2032 Olympic 100m final and explain why.")},
        {"id": "translation", "template": "direct", "prompt": chat_prompt("Translate 'The experiment was repeated three times' into Spanish, then give a literal English back-translation.")},
        {"id": "planning", "template": "direct", "prompt": chat_prompt("Make a five-step checklist for migrating a small database with minimal downtime.")},
    ]
    alpaca_rows = json.loads(args.alpaca_eval.read_text())
    rng = random.Random(args.seed + 3)
    response_rows = rng.sample(alpaca_rows, 128)
    response_likelihood = [
        {"id": f"alpaca-eval-{index}",
         "prompt": chat_prompt(row["instruction"]),
         "response": row["output"] + "\x1e"}
        for index, row in enumerate(response_rows)
    ]

    panel = {
        "schema": SCHEMA,
        "seed": args.seed,
        "tokenizer": "p50k_base",
        "sources": {
            "wikitext": {"revision": WIKITEXT_REVISION, "sha256": sha256(inputs["wikitext"])},
            "mmlu": {"revision": MMLU_REVISION, "sha256": sha256(inputs["mmlu"])},
            "hellaswag": {"revision": HELLASWAG_REVISION, "sha256": sha256(inputs["hellaswag"])},
            "alpaca_eval": {"revision": ALPACA_EVAL_REVISION, "sha256": sha256(args.alpaca_eval)},
        },
        "wikitext": wikitext,
        "mmlu": stratified_mmlu(mmlu_rows, 256, args.seed),
        "hellaswag": random_rows(hellaswag_rows, 256, args.seed + 1),
        "assistant_response_likelihood": response_likelihood,
        "generation_prompts": prompts,
        "retrieval_filler_tokens": torch.tensor(wiki_tokens, dtype=torch.int32),
        "retrieval": {"distances": [2048, 4096, 8192, 16384, 24576, 30720], "examples_per_distance": 32, "seed": args.seed + 2,
                      "variants": ["natural-single", "natural-rs", "natural-multikey"]},
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
        "counts": {"wikitext": contexts, "mmlu": 256, "hellaswag": 256,
                   "assistant_response_likelihood": len(response_likelihood),
                   "generation": len(prompts)},
        "sources": panel["sources"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
