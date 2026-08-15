#!/usr/bin/env python3
"""Compare the frozen 250B/282B/300B/304B E97-MoE evaluation trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_e97_moe_paired_eval import interval, paired_delta

LABELS = ("base-250b", "long-282b", "instruction-300b", "instruction-304b")
PAIRS = (("base-250b", "long-282b"), ("long-282b", "instruction-300b"),
         ("instruction-300b", "instruction-304b"))


def subset(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def main():
    parser = argparse.ArgumentParser()
    for label in LABELS:
        parser.add_argument("--" + label, type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    results = {label: json.loads(getattr(args, label.replace("-", "_")).read_text())
               for label in LABELS}
    panel_hashes = {result["panel_sha256"] for result in results.values()}
    if len(panel_hashes) != 1:
        raise RuntimeError("trajectory results do not share one panel")

    comparisons = {}
    for before_label, after_label in PAIRS:
        before, after = results[before_label], results[after_label]
        key = f"{before_label}-to-{after_label}"
        wiki = {}
        for context in (2048, 8192, 32768):
            wiki[str(context)] = interval(paired_delta(
                subset(before["wikitext"], context=context),
                subset(after["wikitext"], context=context), "id", "nll"),
                seed=context)
        retrieval = {}
        for distance in sorted({row["distance"] for row in before["retrieval"]}):
            old = subset(before["retrieval"], distance=distance)
            new = subset(after["retrieval"], distance=distance)
            retrieval[str(distance)] = {
                "accuracy_delta": interval(paired_delta(old, new, "id", "correct"), seed=distance),
                "margin_delta": interval(paired_delta(old, new, "id", "margin"), seed=distance + 1),
            }
        comparisons[key] = {
            "wikitext_nll_delta": wiki,
            "assistant_response_nll_delta": interval(paired_delta(
                before["assistant_responses"], after["assistant_responses"], "id", "nll")),
            "mmlu_accuracy_delta": interval(paired_delta(before["mmlu"], after["mmlu"], "id", "correct")),
            "hellaswag_normalized_accuracy_delta": interval(paired_delta(
                before["hellaswag"], after["hellaswag"], "id", "normalized_correct")),
            "retrieval": retrieval,
        }
    output = {
        "schema": "emender-e97-moe-trajectory-comparison-v1",
        "panel_sha256": next(iter(panel_hashes)),
        "delta_definition": "later checkpoint minus earlier checkpoint",
        "summaries": {label: result["summary"] for label, result in results.items()},
        "checkpoints": {label: result["checkpoint"] for label, result in results.items()},
        "comparisons": comparisons,
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = ["# E97-MoE checkpoint trajectory", "", f"Panel SHA-256: `{output['panel_sha256']}`", "",
             "## Aggregate results", "",
             "| Checkpoint | Wiki 2K NLL | Wiki 8K NLL | Wiki 32K NLL | Assistant NLL | MMLU | HellaSwag norm |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        summary = results[label]["summary"]
        lines.append(
            f"| {label} | {summary['wikitext']['2048']['nll']:.6f} | "
            f"{summary['wikitext']['8192']['nll']:.6f} | "
            f"{summary['wikitext']['32768']['nll']:.6f} | "
            f"{summary['assistant_response_nll']:.6f} | "
            f"{summary['mmlu_accuracy']:.4f} | "
            f"{summary['hellaswag_normalized_accuracy']:.4f} |")
    lines += ["", "## Retrieval accuracy", "", "| Checkpoint | 2K | 4K | 8K | 16K | 24K | 30K |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label in LABELS:
        values = results[label]["summary"]["retrieval_accuracy_by_distance"]
        lines.append("| " + label + " | " + " | ".join(
            f"{values[str(distance)]:.4f}" for distance in (2048, 4096, 8192, 16384, 24576, 30720)) + " |")
    lines += ["", "## Native-template generations", ""]
    generation_maps = {
        label: {(row["id"], row["mode"]): row for row in result["generations"]}
        for label, result in results.items()}
    first = results[LABELS[0]]["generations"]
    for reference in first:
        identity = (reference["id"], reference["mode"])
        lines += [f"### {reference['id']} — {reference['mode']}", "", "**Prompt**", "", reference["prompt"], ""]
        for label in LABELS:
            row = generation_maps[label][identity]
            lines += [f"**{label}** (stopped={row['stopped']}, stop_token={row['stop_token']})", "",
                      "```text", row["response"], "```", ""]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
