#!/usr/bin/env python3
"""Compare two matched E97-MoE paired-evaluation result files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


def paired_delta(left, right, key, value):
    a = {row[key]: row[value] for row in left}
    b = {row[key]: row[value] for row in right}
    if set(a) != set(b):
        raise RuntimeError(f"paired IDs differ for {value}")
    return [float(b[name]) - float(a[name]) for name in sorted(a)]


def interval(values, seed=970035, draws=10000):
    if not values:
        return {"mean": None, "p025": None, "p975": None, "count": 0}
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws)]
    means.sort()
    return {"mean": statistics.fmean(values), "p025": means[int(.025 * draws)],
            "p975": means[int(.975 * draws)], "count": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text()); after = json.loads(args.after.read_text())
    if before["panel_sha256"] != after["panel_sha256"]:
        raise RuntimeError("evaluation panel hashes differ")

    wiki = {}
    for context in (2048, 8192, 32768):
        old = [row for row in before["wikitext"] if row["context"] == context]
        new = [row for row in after["wikitext"] if row["context"] == context]
        wiki[str(context)] = interval(paired_delta(old, new, "id", "nll"), seed=970035 + context)
    retrieval = {}
    for distance in sorted({row["distance"] for row in before["retrieval"]}):
        old = [row for row in before["retrieval"] if row["distance"] == distance]
        new = [row for row in after["retrieval"] if row["distance"] == distance]
        retrieval[str(distance)] = {
            "accuracy_delta": interval(paired_delta(old, new, "id", "correct"), seed=distance),
            "margin_delta": interval(paired_delta(old, new, "id", "margin"), seed=distance + 1),
        }
    comparison = {
        "schema": "emender-e97-moe-paired-eval-comparison-v1",
        "panel_sha256": before["panel_sha256"],
        "before": before["checkpoint"], "after": after["checkpoint"],
        "delta_definition": "after minus before; negative WikiText NLL is improvement",
        "wikitext_nll_delta": wiki,
        "mmlu_accuracy_delta": interval(paired_delta(before["mmlu"], after["mmlu"], "id", "correct")),
        "hellaswag_accuracy_delta": interval(paired_delta(before["hellaswag"], after["hellaswag"], "id", "raw_correct")),
        "hellaswag_normalized_accuracy_delta": interval(paired_delta(before["hellaswag"], after["hellaswag"], "id", "normalized_correct")),
        "retrieval": retrieval,
        "before_summary": before["summary"], "after_summary": after["summary"],
    }
    args.output_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    lines = ["# E97-MoE paired evaluation", "", f"Panel SHA-256: `{before['panel_sha256']}`", "",
             "All deltas are final instruction checkpoint minus pre-instruction checkpoint.", "",
             "## Aggregate metrics", "",
             "| Metric | Before | After | Delta |", "|---|---:|---:|---:|"]
    for context in (2048, 8192, 32768):
        b = before["summary"]["wikitext"][str(context)]["nll"]
        a = after["summary"]["wikitext"][str(context)]["nll"]
        lines.append(f"| WikiText NLL {context // 1024}K | {b:.6f} | {a:.6f} | {a-b:+.6f} |")
    for label, key in (("MMLU accuracy", "mmlu_accuracy"),
                       ("HellaSwag accuracy", "hellaswag_accuracy"),
                       ("HellaSwag normalized", "hellaswag_normalized_accuracy")):
        b=before["summary"][key];a=after["summary"][key]
        lines.append(f"| {label} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")
    lines += ["", "## Retrieval", "", "| Distance | Before accuracy | After accuracy | Before margin | After margin |",
              "|---:|---:|---:|---:|---:|"]
    for distance in before["summary"]["retrieval_accuracy_by_distance"]:
        ba=before["summary"]["retrieval_accuracy_by_distance"][distance]
        aa=after["summary"]["retrieval_accuracy_by_distance"][distance]
        bm=before["summary"]["retrieval_margin_by_distance"][distance]
        am=after["summary"]["retrieval_margin_by_distance"][distance]
        lines.append(f"| {distance} | {ba:.4f} | {aa:.4f} | {bm:.4f} | {am:.4f} |")
    lines += ["", "## Deterministic generations", ""]
    new_generations = {row["id"]: row for row in after["generations"]}
    for old in before["generations"]:
        new = new_generations[old["id"]]
        lines += [f"### {old['id']}", "", "**Prompt**", "", old["prompt"], "",
                  "**Before**", "", "```text", old["response"], "```", "",
                  "**After**", "", "```text", new["response"], "```", ""]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
