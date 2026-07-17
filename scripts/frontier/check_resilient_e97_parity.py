#!/usr/bin/env python3
"""Fail closed unless debug and production use one resilient E97 payload.

The profiles are rendered launch contracts, not alternate launchers.  Every
non-scheduler field is required to compare byte-for-byte after canonical JSON
encoding.  The deliberate fault schedule is the sole runtime delta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEDULER_DELTAS = {"partition", "qos", "walltime", "nodes"}
ALLOWED_DELTAS = SCHEDULER_DELTAS | {"failure_injection"}
REQUIRED_IDENTICAL = {
    "launcher", "model", "dataset", "optimizer", "seed", "local_steps",
    "trainers_per_node", "managers_per_node", "manager_model_free",
    "quorum_policy", "control_transport", "bulk_transport", "checkpoint_contract",
    "deadlines", "code_id", "payload_id",
}


def compare(debug: dict, production: dict) -> dict:
    missing = sorted(REQUIRED_IDENTICAL - debug.keys())
    missing += sorted(REQUIRED_IDENTICAL - production.keys())
    keys = sorted(set(debug) | set(production))
    differences = {key: {"debug": debug.get(key), "production": production.get(key)}
                   for key in keys if debug.get(key) != production.get(key)}
    forbidden = sorted(set(differences) - ALLOWED_DELTAS)
    injection_ok = (debug.get("failure_injection") not in (None, [], {}) and
                    production.get("failure_injection") in (None, [], {}))
    return {
        "schema": 1,
        "ok": not missing and not forbidden and injection_ok,
        "identical_fields": sorted(key for key in REQUIRED_IDENTICAL
                                   if debug.get(key) == production.get(key)),
        "allowlisted_diff": differences,
        "allowlist": sorted(ALLOWED_DELTAS),
        "missing_required": missing,
        "forbidden_diff": forbidden,
        "injection_disabled_in_production": injection_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.debug.read_text()),
                     json.loads(args.production.read_text()))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
