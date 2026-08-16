#!/usr/bin/env python3
"""Compare the 282B parent with the canonical router-preserved SFT checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_e97_moe_paired_eval import interval, paired_delta


def subset(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def generation_summary(rows):
    result = {}
    for mode in ("greedy", "sample"):
        selected = subset(rows, mode=mode)
        result[mode] = {
            "examples": len(selected),
            "stopped": sum(row["stopped"] for row in selected),
            "rs_stops": sum(row["stop_token"] == 218 for row in selected),
            "eot_stops": sum(row["stop_token"] == 50256 for row in selected),
            "mean_generated_tokens": (
                sum(row["generated_tokens"] for row in selected) / max(len(selected), 1)),
        }
    return result


def recurrent_health(result):
    values = [row["hidden_health"] for row in result["wikitext"]]
    values += [row["hidden_health"] for row in result["retrieval"]]
    return {
        "observations": len(values),
        "minimum_finite_fraction": min(row["finite_fraction"] for row in values),
        "maximum_abs": max(row["max_abs"] for row in values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    parent = json.loads(args.parent.read_text())
    candidate = json.loads(args.candidate.read_text())
    if parent["panel_sha256"] != candidate["panel_sha256"]:
        raise RuntimeError("evaluation panel hashes differ")

    wiki = {}
    for context in (2048, 8192, 32768):
        wiki[str(context)] = interval(paired_delta(
            subset(parent["wikitext"], context=context),
            subset(candidate["wikitext"], context=context), "id", "nll"),
            seed=970035 + context)
    output = {
        "schema": "emender-e97-moe-router-preserved-sft-eval-comparison-v1",
        "panel_sha256": parent["panel_sha256"],
        "delta_definition": "candidate minus parent; negative NLL is improvement",
        "checkpoints": {"parent": parent["checkpoint"], "candidate": candidate["checkpoint"]},
        "summaries": {"parent": parent["summary"], "candidate": candidate["summary"]},
        "wikitext_nll_delta": wiki,
        "assistant_response_nll_delta": interval(paired_delta(
            parent["assistant_responses"], candidate["assistant_responses"], "id", "nll")),
        "mmlu_accuracy_delta": interval(paired_delta(
            parent["mmlu"], candidate["mmlu"], "id", "correct")),
        "hellaswag_normalized_accuracy_delta": interval(paired_delta(
            parent["hellaswag"], candidate["hellaswag"], "id", "normalized_correct")),
        "generations": {
            "parent": generation_summary(parent["generations"]),
            "candidate": generation_summary(candidate["generations"]),
        },
        "recurrent_health": {
            "parent": recurrent_health(parent),
            "candidate": recurrent_health(candidate),
        },
        "routing": {"parent": parent["routing"], "candidate": candidate["routing"]},
    }
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# E97-MoE router-preserved masked-SFT evaluation", "",
        f"Panel SHA-256: `{output['panel_sha256']}`", "",
        "| Metric | Parent | Candidate | Delta |", "|---|---:|---:|---:|",
    ]
    for context in (2048, 8192, 32768):
        before = parent["summary"]["wikitext"][str(context)]["nll"]
        after = candidate["summary"]["wikitext"][str(context)]["nll"]
        lines.append(f"| WikiText NLL {context // 1024}K | {before:.6f} | {after:.6f} | {after-before:+.6f} |")
    for label, key in (("Assistant-response NLL", "assistant_response_nll"),
                       ("MMLU accuracy", "mmlu_accuracy"),
                       ("HellaSwag normalized", "hellaswag_normalized_accuracy")):
        before = parent["summary"][key]; after = candidate["summary"][key]
        lines.append(f"| {label} | {before:.6f} | {after:.6f} | {after-before:+.6f} |")
    lines += ["", "## Native cached generations", ""]
    maps = {
        label: {(row["id"], row["mode"]): row for row in result["generations"]}
        for label, result in (("parent", parent), ("candidate", candidate))
    }
    for identity in sorted(maps["parent"]):
        reference = maps["parent"][identity]
        lines += [f"### {identity[0]} — {identity[1]}", "", "**Prompt**", "",
                  reference["prompt"], ""]
        for label in ("parent", "candidate"):
            row = maps[label][identity]
            lines += [f"**{label}** (stopped={row['stopped']}, token={row['stop_token']}, generated={row['generated_tokens']})",
                      "", "```text", row["response"], "```", ""]
    args.output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
