#!/usr/bin/env python3
"""Build a frozen post-broad-SFT structural Pi evaluation authority."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ndm.data.masked_sft_dataset import sha256
from scripts.build_e97_pi_eval_v2 import call

SCHEMA = "emender-e97-pi-core-eval-authority-v4"
KINDS = (
    "three-way-compare",
    "pointer-conditional-edit",
    "csv-aggregate",
    "test-guided-rename",
    "search-link-update",
    "checksum-manifest",
)


def trace(kind: str, index: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    token = f"{rng.choice(('plum', 'mint', 'copper', 'azure'))}-{rng.randint(10000, 99999)}"
    fixtures: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []
    postconditions: list[dict[str, str]] = []
    final_contains: list[str] = []
    expected_exit_codes: list[int] | None = None

    if kind == "three-way-compare":
        paths = [f"releases/{region}-{index:06d}.json" for region in ("north", "south", "central")]
        builds = rng.sample(range(1000, 9999), 3)
        for path, build in zip(paths, builds):
            fixtures.append({"path": path, "content": json.dumps({"release": token, "build": build}) + "\n"})
        winner_index = max(range(3), key=builds.__getitem__)
        user = f"Read {', '.join(paths)} and report the file with the largest build, its value, and release."
        calls = [call("read", {"path": path, "offset": 1, "limit": 40}) for path in paths]
        final_contains = [paths[winner_index], str(builds[winner_index]), token]

    elif kind == "pointer-conditional-edit":
        selector = f"deploy/selector-{index:06d}.json"
        blue = f"deploy/blue-{index:06d}.env"
        green = f"deploy/green-{index:06d}.env"
        selected = rng.choice(("blue", "green"))
        target = blue if selected == "blue" else green
        other = green if selected == "blue" else blue
        fixtures.extend([
            {"path": selector, "content": json.dumps({"active": selected, "release": token}) + "\n"},
            {"path": blue, "content": "RELEASE=old\nCOLOR=blue\n"},
            {"path": green, "content": "RELEASE=old\nCOLOR=green\n"},
        ])
        verify = ("python -c 'from pathlib import Path; "
                  f"assert \"RELEASE={token}\" in Path(\"{target}\").read_text(); "
                  f"assert \"RELEASE=old\" in Path(\"{other}\").read_text()'")
        user = f"Read {selector}, update only its active deployment file to release {token}, preserve the inactive file, and verify both."
        calls = [
            call("read", {"path": selector, "offset": 1, "limit": 40}),
            call("read", {"path": target, "offset": 1, "limit": 40}),
            call("edit", {"path": target, "oldText": "RELEASE=old", "newText": f"RELEASE={token}"}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": target, "contains": f"RELEASE={token}"},
                          {"path": other, "contains": "RELEASE=old"}]
        final_contains = [selector, target, other, token]

    elif kind == "csv-aggregate":
        first = f"metrics/primary-{index:06d}.csv"
        second = f"metrics/secondary-{index:06d}.csv"
        output = f"reports/total-{index:06d}.json"
        left, right = rng.randint(10, 500), rng.randint(10, 500)
        fixtures.extend([{"path": first, "content": f"name,value\nprimary,{left}\n"},
                         {"path": second, "content": f"name,value\nsecondary,{right}\n"}])
        content = json.dumps({"release": token, "total": left + right}, indent=2, sort_keys=True) + "\n"
        verify = (f"python -c 'import json; x=json.load(open(\"{output}\")); "
                  f"assert x == {{\"release\":\"{token}\",\"total\":{left + right}}}'")
        user = f"Read {first} and {second}, sum their typed values, write {output} with release {token}, and verify it."
        calls = [call("read", {"path": first, "offset": 1, "limit": 40}),
                 call("read", {"path": second, "offset": 1, "limit": 40}),
                 call("write", {"path": output, "content": content}),
                 call("bash", {"command": verify})]
        postconditions = [{"path": output, "exact": content}]
        final_contains = [first, second, output, token, str(left + right)]

    elif kind == "test-guided-rename":
        source = f"package/feature_{index:06d}.py"
        test = f"checks/check_{index:06d}.py"
        old = f"legacy_{index:06d}"
        new = f"current_{index:06d}"
        fixtures.extend([
            {"path": source, "content": f"def {old}():\n    return \"{token}\"\n"},
            {"path": test, "content": (
                "import runpy\n"
                f"scope = runpy.run_path(\"{source}\")\n"
                f"assert scope[\"{new}\"]() == \"{token}\"\n")},
        ])
        command = f"python {test}"
        user = f"Run {command}, use its failure and {test} to rename the implementation in {source} as required, then rerun the check."
        calls = [call("bash", {"command": command}),
                 call("read", {"path": test, "offset": 1, "limit": 80}),
                 call("read", {"path": source, "offset": 1, "limit": 80}),
                 call("edit", {"path": source, "oldText": f"def {old}():", "newText": f"def {new}():"}),
                 call("bash", {"command": command})]
        expected_exit_codes = [1, 0, 0, 0, 0]
        postconditions = [{"path": source, "contains": f"def {new}():"}]
        final_contains = [test, source, new]

    elif kind == "search-link-update":
        target = f"sites/team_{index:06d}/links.toml"
        distractor = f"sites/archive_{index:06d}/links.toml"
        old_url = f"https://old.invalid/{index}"
        new_url = f"https://example.invalid/{token}"
        fixtures.extend([
            {"path": target, "content": f'KEY = "{token}"\nURL = "{old_url}"\n'},
            {"path": distractor, "content": 'KEY = "archive"\nURL = "https://archive.invalid"\n'},
        ])
        search = ("python -c 'import pathlib; print(next(str(p) for p in pathlib.Path(\"sites\")"
                  f".glob(\"**/links.toml\") if \"KEY = \\\"{token}\\\"\" in p.read_text()))'")
        verify = (f"python -c 'from pathlib import Path; assert \"{new_url}\" in "
                  f"Path(\"{target}\").read_text()'")
        user = f"Find the links.toml with key {token}, change only its URL to {new_url}, and verify the selected file."
        calls = [call("bash", {"command": search}),
                 call("read", {"path": target, "offset": 1, "limit": 40}),
                 call("edit", {"path": target, "oldText": f'URL = "{old_url}"', "newText": f'URL = "{new_url}"'}),
                 call("bash", {"command": verify})]
        postconditions = [{"path": target, "contains": new_url},
                          {"path": distractor, "contains": "https://archive.invalid"}]
        final_contains = [target, token, new_url]

    elif kind == "checksum-manifest":
        source = f"payloads/item-{index:06d}.txt"
        output = f"manifests/item-{index:06d}.sha256"
        content = f"release={token}\nvalue={rng.randint(1000, 9999)}\n"
        fixtures.append({"path": source, "content": content})
        inspect = f"sha256sum {source}"
        import hashlib
        checksum = hashlib.sha256(content.encode()).hexdigest()
        rendered = f"{checksum}  {source}\n"
        verify = f"sha256sum -c {output}"
        user = f"Inspect {source}, compute its SHA-256, write a standard sha256sum line to {output}, and verify the manifest."
        calls = [call("read", {"path": source, "offset": 1, "limit": 40}),
                 call("bash", {"command": inspect}),
                 call("write", {"path": output, "content": rendered}),
                 call("bash", {"command": verify})]
        postconditions = [{"path": output, "exact": rendered}]
        final_contains = [source, output, checksum]
    else:
        raise ValueError(kind)

    task = {"fixtures": fixtures, "expected_calls": calls,
            "postconditions": postconditions, "final_contains": final_contains}
    if expected_exit_codes is not None:
        task["expected_exit_codes"] = expected_exit_codes
    return user, task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--seed", type=int, default=7_301_117)
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
            row = {"id": f"pi-eval-v4-{kind}-{index:08d}",
                   "source": "emender-pi-core-eval-v4", "split": 1,
                   "kind": kind, "user": user, "task": task}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            counts[kind] += 1
    manifest = {
        "schema": SCHEMA, "status": "complete",
        "purpose": "post-broad-SFT structural family-held-out Pi evaluation",
        "records": args.records, "seed": args.seed, "kinds": list(KINDS),
        "kind_counts": counts,
        "training_exclusion": (
            "frozen before public post-training payload download; evaluation-only; "
            "whole generators, identities, paths, values, and payloads excluded"),
        "outputs": {"metadata": {"path": str(metadata.resolve()),
                                    "bytes": metadata.stat().st_size,
                                    "sha256": sha256(metadata)}},
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
