#!/usr/bin/env python3
"""Build a blind family-held-out Pi compositional evaluation authority."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ndm.data.masked_sft_dataset import sha256
from scripts.build_e97_pi_eval_v2 import call

SCHEMA = "emender-e97-pi-core-eval-authority-v3"
KINDS = (
    "compare-configs",
    "pointer-edit",
    "search-write",
    "recover-nonunique",
    "command-recovery",
    "aggregate-write",
)


def trace(kind: str, index: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    token = f"{rng.choice(('silver', 'coral', 'navy', 'gold'))}-{rng.randint(10000, 99999)}"
    fixtures: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []
    postconditions: list[dict[str, str]] = []
    final_contains: list[str] = []

    if kind == "compare-configs":
        left = f"environments/east-{index:06d}.json"
        right = f"environments/west-{index:06d}.json"
        left_build, right_build = rng.sample(range(1000, 9000), 2)
        fixtures.extend([
            {"path": left, "content": json.dumps({"release": token, "build": left_build}) + "\n"},
            {"path": right, "content": json.dumps({"release": token, "build": right_build}) + "\n"},
        ])
        winner = left if left_build > right_build else right
        winning_build = max(left_build, right_build)
        user = f"Read {left} and {right}; report which file has the larger build and its exact value."
        calls = [call("read", {"path": left, "offset": 1, "limit": 40}),
                 call("read", {"path": right, "offset": 1, "limit": 40})]
        final_contains = [winner, str(winning_build)]

    elif kind == "pointer-edit":
        pointer = f"routing/active-{index:06d}.txt"
        target = f"components/unit-{index:06d}/settings.py"
        fixtures.extend([
            {"path": pointer, "content": f"settings={target}\n"},
            {"path": target, "content": f'RELEASE = "old"\nKEY = "{token}"\n'},
        ])
        verify = (f"python -c 'import runpy; assert runpy.run_path(\"{target}\")"
                  f"[\"RELEASE\"] == \"{token}\"'")
        user = f"Follow the settings pointer in {pointer}, update that referenced component to release {token}, and verify it."
        calls = [
            call("read", {"path": pointer, "offset": 1, "limit": 40}),
            call("read", {"path": target, "offset": 1, "limit": 40}),
            call("edit", {"path": target, "oldText": 'RELEASE = "old"', "newText": f'RELEASE = "{token}"'}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": target, "contains": f'RELEASE = "{token}"'},
                          {"path": target, "contains": f'KEY = "{token}"'}]
        final_contains = [pointer, target, token]

    elif kind == "search-write":
        target = f"specifications/team_{index:06d}/artifact.txt"
        distractor = f"specifications/archive_{index:06d}/artifact.txt"
        output = f"artifacts/result-{index:06d}.json"
        fixtures.extend([
            {"path": target, "content": f"artifact_key={token}\ntier=primary\n"},
            {"path": distractor, "content": "artifact_key=retired\ntier=archive\n"},
        ])
        search = ("python -c 'import pathlib; print(next(str(p) for p in "
                  "pathlib.Path(\"specifications\").glob(\"**/artifact.txt\") if "
                  f"\"artifact_key={token}\" in p.read_text()))'")
        content = json.dumps({"artifact_key": token, "tier": "primary"}, indent=2, sort_keys=True) + "\n"
        verify = f"python -m json.tool {output} >/dev/null"
        user = f"Locate the artifact specification containing key {token}, read it, create {output} as typed JSON, and validate it."
        calls = [call("bash", {"command": search}),
                 call("read", {"path": target, "offset": 1, "limit": 40}),
                 call("write", {"path": output, "content": content}),
                 call("bash", {"command": verify})]
        postconditions = [{"path": output, "exact": content}]
        final_contains = [target, output, token]

    elif kind == "recover-nonunique":
        path = f"src/features_{index:06d}.txt"
        target_block = f"NAME={token}\nENABLED=False"
        other = "NAME=other\nENABLED=False"
        fixtures.append({"path": path, "content": target_block + "\n\n" + other + "\n"})
        corrected = f"NAME={token}\nENABLED=True"
        expected_lines = [f"NAME={token}", "ENABLED=True", "", "NAME=other", "ENABLED=False"]
        verify = (f"python -c 'from pathlib import Path; assert Path(\"{path}\").read_text()"
                  f".splitlines() == {json.dumps(expected_lines)}'")
        user = (f"In {path}, first try the simple exact ENABLED=False replacement for {token}. "
                "If it is non-unique, inspect and repair only that named block, then verify the other block stayed disabled.")
        calls = [
            call("edit", {"path": path, "oldText": "ENABLED=False", "newText": "ENABLED=True"}),
            call("read", {"path": path, "offset": 1, "limit": 80}),
            call("edit", {"path": path, "oldText": target_block, "newText": corrected}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": path, "contains": corrected}, {"path": path, "contains": other}]
        final_contains = [path, token, "other", "disabled"]

    elif kind == "command-recovery":
        path = f"data/current-{index:06d}.json"
        wrong = f"data/legacy-{index:06d}.json"
        value = rng.randint(100, 999)
        fixtures.append({"path": path, "content": json.dumps({"key": token, "value": value}) + "\n"})
        bad = f"python -c 'import json; print(json.load(open(\"{wrong}\"))[\"value\"])'"
        good = f"python -c 'import json; print(json.load(open(\"{path}\"))[\"value\"])'"
        user = (f"Run `{bad}` first. Recover from its stale path using {path}, then rerun the corrected inspection and report the value.")
        calls = [call("bash", {"command": bad}),
                 call("read", {"path": path, "offset": 1, "limit": 40}),
                 call("bash", {"command": good})]
        final_contains = [path, str(value)]

    elif kind == "aggregate-write":
        first = f"inputs/alpha-{index:06d}.txt"
        second = f"inputs/beta-{index:06d}.txt"
        output = f"combined/manifest-{index:06d}.json"
        alpha, beta = rng.randint(10, 99), rng.randint(10, 99)
        fixtures.extend([{ "path": first, "content": f"alpha={alpha}\n"},
                         {"path": second, "content": f"beta={beta}\n"}])
        content = json.dumps({"alpha": alpha, "beta": beta, "release": token}, indent=2, sort_keys=True) + "\n"
        verify = f"python -m json.tool {output} >/dev/null"
        user = f"Read {first} and {second}; create {output} combining both typed values with release {token}, then validate it."
        calls = [call("read", {"path": first, "offset": 1, "limit": 40}),
                 call("read", {"path": second, "offset": 1, "limit": 40}),
                 call("write", {"path": output, "content": content}),
                 call("bash", {"command": verify})]
        postconditions = [{"path": output, "exact": content}]
        final_contains = [first, second, output, token]
    else:
        raise ValueError(kind)
    return user, {"fixtures": fixtures, "expected_calls": calls,
                  "postconditions": postconditions, "final_contains": final_contains}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--seed", type=int, default=4_901_093)
    args = parser.parse_args()
    if args.records <= 0 or args.records % len(KINDS):
        raise SystemExit(f"records must be a positive multiple of {len(KINDS)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    metadata = args.output_root / "records.jsonl"
    counts = {kind: 0 for kind in KINDS}
    with metadata.open("w") as stream:
        for index in range(args.records):
            kind = KINDS[index % len(KINDS)]
            user, task = trace(kind, index, random.Random(args.seed + index))
            stream.write(json.dumps({"id": f"pi-eval-v3-{kind}-{index:08d}",
                                     "source": "emender-pi-core-eval-v3", "split": 1,
                                     "kind": kind, "user": user, "task": task}, sort_keys=True) + "\n")
            counts[kind] += 1
    manifest = {"schema": SCHEMA, "status": "complete",
                "purpose": "blind family-held-out real-Pi compositional evaluation",
                "records": args.records, "seed": args.seed, "kinds": list(KINDS),
                "kind_counts": counts,
                "training_exclusion": "frozen before compositional v3 candidate training",
                "outputs": {"metadata": {"path": str(metadata.resolve()),
                                            "bytes": metadata.stat().st_size,
                                            "sha256": sha256(metadata)}}}
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
