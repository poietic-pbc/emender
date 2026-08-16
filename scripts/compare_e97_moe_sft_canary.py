#!/usr/bin/env python3
"""Compare the matched 282B parent and two masked-SFT LR canary arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_e97_moe_paired_eval import interval, paired_delta

LABELS = ("parent-282b", "sft-lr2e-6", "sft-lr5e-6")
PAIRS = (("parent-282b", "sft-lr2e-6"), ("parent-282b", "sft-lr5e-6"),
         ("sft-lr2e-6", "sft-lr5e-6"))


def subset(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def main() -> None:
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
        raise RuntimeError("SFT canary results do not share one panel")
    comparisons = {}
    for before_label, after_label in PAIRS:
        before, after = results[before_label], results[after_label]
        wiki = {}
        for context in (2048, 8192, 32768):
            wiki[str(context)] = interval(paired_delta(
                subset(before["wikitext"], context=context),
                subset(after["wikitext"], context=context), "id", "nll"), seed=context)
        comparisons[f"{before_label}-to-{after_label}"] = {
            "wikitext_nll_delta": wiki,
            "assistant_response_nll_delta": interval(paired_delta(
                before["assistant_responses"], after["assistant_responses"], "id", "nll")),
            "mmlu_accuracy_delta": interval(paired_delta(
                before["mmlu"], after["mmlu"], "id", "correct")),
            "hellaswag_normalized_accuracy_delta": interval(paired_delta(
                before["hellaswag"], after["hellaswag"], "id", "normalized_correct")),
        }
    output = {
        "schema": "emender-e97-moe-masked-sft-canary-comparison-v1",
        "panel_sha256": next(iter(panel_hashes)),
        "delta_definition": "after minus before; negative NLL is improvement",
        "summaries": {label: result["summary"] for label, result in results.items()},
        "checkpoints": {label: result["checkpoint"] for label, result in results.items()},
        "comparisons": comparisons,
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = ["# E97-MoE masked-SFT canary", "", f"Panel SHA-256: `{output['panel_sha256']}`", "",
             "| Checkpoint | Wiki 2K | Wiki 8K | Wiki 32K | Assistant NLL | MMLU | HellaSwag norm |",
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
    lines += ["", "## Native-template generations", ""]
    maps = {label: {(row["id"], row["mode"]): row for row in result["generations"]}
            for label, result in results.items()}
    for reference in results[LABELS[0]]["generations"]:
        identity = (reference["id"], reference["mode"])
        lines += [f"### {reference['id']} — {reference['mode']}", "", "**Prompt**", "",
                  reference["prompt"], ""]
        for label in LABELS:
            row = maps[label][identity]
            lines += [f"**{label}** (stopped={row['stopped']}, token={row['stop_token']})", "",
                      "```text", row["response"], "```", ""]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
