#!/usr/bin/env python3
"""Build a frozen template-held-out Pi core-tools evaluation authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ndm.data.masked_sft_dataset import sha256

SCHEMA = "emender-e97-pi-core-eval-authority-v2"
KINDS = (
    "inspect-chain",
    "search-edit",
    "multi-edit",
    "recover-edit",
    "diagnose-test",
    "write-from-spec",
)


def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args}


def trace(kind: str, index: int, rng: random.Random) -> tuple[str, dict[str, Any]]:
    token = f"{rng.choice(('ochre', 'teal', 'violet', 'crimson'))}-{rng.randint(10000, 99999)}"
    fixtures: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []
    postconditions: list[dict[str, str]] = []
    final_contains: list[str] = []

    if kind == "inspect-chain":
        catalog = f"catalog/release-{index:06d}.txt"
        detail = f"configs/releases/detail-{index:06d}.json"
        channel = rng.choice(("stable", "candidate", "preview"))
        build = rng.randint(100, 999)
        fixtures.extend([
            {"path": catalog, "content": f"active_detail={detail}\n"},
            {"path": detail, "content": json.dumps(
                {"release": token, "channel": channel, "build": build},
                sort_keys=True,
            ) + "\n"},
        ])
        user = (
            f"Read {catalog}, follow its active_detail reference, and report the "
            "release channel and build with file evidence."
        )
        calls = [
            call("read", {"path": catalog, "offset": 1, "limit": 80}),
            call("read", {"path": detail, "offset": 1, "limit": 80}),
        ]
        final_contains = [detail, channel, str(build)]

    elif kind == "search-edit":
        slug = token.replace("-", "_")
        target = f"services/{slug}/release.py"
        distractor = f"services/archive_{index:06d}/release.py"
        marker = f'TARGET_KEY = "{token}"'
        fixtures.extend([
            {"path": target, "content": f"{marker}\nENABLED = False\n"},
            {"path": distractor, "content": 'TARGET_KEY = "archived"\nENABLED = False\n'},
        ])
        search = (
            "python -c 'import pathlib; print(next(str(p) for p in "
            "pathlib.Path(\"services\").glob(\"*/release.py\") if "
            f"\"TARGET_KEY = \\\"{token}\\\"\" in p.read_text()))'"
        )
        verify = (
            f"python -c 'import runpy; assert runpy.run_path(\"{target}\")"
            "[\"ENABLED\"] is True'"
        )
        user = (
            f"Find the services release.py containing target key {token}, enable it, "
            "and run a focused verification. Do not modify the archived release."
        )
        calls = [
            call("bash", {"command": search}),
            call("read", {"path": target, "offset": 1, "limit": 40}),
            call("edit", {"path": target, "oldText": "ENABLED = False", "newText": "ENABLED = True"}),
            call("bash", {"command": verify}),
        ]
        postconditions = [
            {"path": target, "contains": "ENABLED = True"},
            {"path": distractor, "contains": "ENABLED = False"},
        ]
        final_contains = [target, "ENABLED", token]

    elif kind == "multi-edit":
        first = f"deploy/api-{index:06d}.env"
        second = f"deploy/worker-{index:06d}.env"
        old = f"RELEASE=old-{index:06d}"
        new = f"RELEASE={token}"
        fixtures.extend([
            {"path": first, "content": old + "\nROLE=api\n"},
            {"path": second, "content": old + "\nROLE=worker\n"},
        ])
        verify = (
            "python -c 'from pathlib import Path; assert all("
            f"\"{new}\" in Path(p).read_text() for p in [\"{first}\",\"{second}\"])'"
        )
        user = (
            f"Set release {token} in both {first} and {second}, preserving their roles, "
            "then verify both files together."
        )
        calls = [
            call("read", {"path": first, "offset": 1, "limit": 40}),
            call("edit", {"path": first, "oldText": old, "newText": new}),
            call("read", {"path": second, "offset": 1, "limit": 40}),
            call("edit", {"path": second, "oldText": old, "newText": new}),
            call("bash", {"command": verify}),
        ]
        postconditions = [
            {"path": first, "contains": new}, {"path": first, "contains": "ROLE=api"},
            {"path": second, "contains": new}, {"path": second, "contains": "ROLE=worker"},
        ]
        final_contains = [first, second, token]

    elif kind == "recover-edit":
        path = f"src/mode_{index:06d}.py"
        fixtures.append({"path": path, "content": 'MODE = "draft"\n'})
        verify = (
            f"python -c 'import runpy; assert runpy.run_path(\"{path}\")"
            "[\"MODE\"] == \"production\"'"
        )
        user = (
            f"In {path}, first attempt the requested exact replacement from staging to "
            "production. If that block is unavailable, inspect the file, recover using "
            "its actual value, and verify production mode."
        )
        calls = [
            call("edit", {"path": path, "oldText": 'MODE = "staging"', "newText": 'MODE = "production"'}),
            call("read", {"path": path, "offset": 1, "limit": 40}),
            call("edit", {"path": path, "oldText": 'MODE = "draft"', "newText": 'MODE = "production"'}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": path, "exact": 'MODE = "production"\n'}]
        final_contains = [path, "production"]

    elif kind == "diagnose-test":
        config = f"config/limit-{index:06d}.json"
        source = f"src/limit_{index:06d}.py"
        desired = rng.randint(16, 64)
        fixtures.extend([
            {"path": config, "content": json.dumps({"limit": desired}) + "\n"},
            {"path": source, "content": "LIMIT = 8\n"},
        ])
        verify = (
            "python -c 'import json,runpy; assert runpy.run_path("
            f"\"{source}\")[\"LIMIT\"] == json.load(open(\"{config}\"))[\"limit\"]'"
        )
        user = (
            f"Run the focused consistency assertion for {source} and {config} first. "
            "Diagnose the failure, make the source match the config authority, and rerun it."
        )
        calls = [
            call("bash", {"command": verify}),
            call("read", {"path": config, "offset": 1, "limit": 40}),
            call("read", {"path": source, "offset": 1, "limit": 40}),
            call("edit", {"path": source, "oldText": "LIMIT = 8", "newText": f"LIMIT = {desired}"}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": source, "exact": f"LIMIT = {desired}\n"}]
        final_contains = [source, config, str(desired)]

    elif kind == "write-from-spec":
        spec = f"specs/release-{index:06d}.txt"
        output = f"generated/release-{index:06d}.json"
        fixtures.append({"path": spec, "content": f"release={token}\napproved=true\n"})
        content = json.dumps({"approved": True, "release": token}, indent=2, sort_keys=True) + "\n"
        verify = f"python -m json.tool {output} >/dev/null"
        user = (
            f"Read {spec}, create {output} with the specified typed JSON values, "
            "and validate the generated JSON."
        )
        calls = [
            call("read", {"path": spec, "offset": 1, "limit": 40}),
            call("write", {"path": output, "content": content}),
            call("bash", {"command": verify}),
        ]
        postconditions = [{"path": output, "exact": content}]
        final_contains = [spec, output, token, "approved"]
    else:
        raise ValueError(kind)

    return user, {
        "fixtures": fixtures,
        "expected_calls": calls,
        "postconditions": postconditions,
        "final_contains": final_contains,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2_601_041)
    args = parser.parse_args()
    if args.records <= 0 or args.records % len(KINDS):
        raise SystemExit(f"records must be a positive multiple of {len(KINDS)}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    metadata = args.output_root / "records.jsonl"
    counts = {kind: 0 for kind in KINDS}
    with metadata.open("w") as stream:
        for index in range(args.records):
            kind = KINDS[index % len(KINDS)]
            identity = f"pi-eval-v2-{kind}-{index:08d}"
            user, task = trace(kind, index, random.Random(args.seed + index))
            row = {
                "id": identity,
                "source": "emender-pi-core-eval-v2",
                "split": 1,
                "kind": kind,
                "user": user,
                "task": task,
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            counts[kind] += 1
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "purpose": "template-held-out compositional real-Pi core-tools evaluation",
        "records": args.records,
        "seed": args.seed,
        "kinds": list(KINDS),
        "kind_counts": counts,
        "training_exclusion": (
            "new task families and seed; frozen before candidate training; evaluation-only"
        ),
        "tool_contract": {
            "read": ["path", "offset", "limit"],
            "bash": ["command"],
            "edit": ["path", "oldText", "newText"],
            "write": ["path", "content"],
        },
        "outputs": {
            "metadata": {
                "path": str(metadata.resolve()),
                "bytes": metadata.stat().st_size,
                "sha256": sha256(metadata),
            }
        },
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest_sha256": sha256(manifest_path), **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
