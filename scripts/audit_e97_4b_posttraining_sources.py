#!/usr/bin/env python3
"""Verify immutable metadata/card identities for candidate broad SFT sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

SCHEMA = "emender-e97-4b-posttraining-source-audit-v1"
SOURCES = {
    "tulu3": {
        "dataset_id": "allenai/tulu-3-sft-mixture",
        "revision": "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
        "card_sha256": "1b6fff37d3f206b70086d9d65015a75dfb98b603a99af7330b0d44cd198464d6",
        "repository_bytes": 1_412_964_994,
        "license_policy": "ODC-BY dataset card; preserve per-source restrictions",
        "decision": "admitted-existing",
    },
    "smoltalk2": {
        "dataset_id": "HuggingFaceTB/smoltalk2",
        "revision": "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
        "card_sha256": "8bdc71dea7688d0b57e7eeb463f468641b460e5e512b2c57f8b45a070588b2c7",
        "repository_bytes": 87_741_812_980,
        "license_policy": "admit Apache-2.0 new subsets only; review inherited subsets individually",
        "decision": "provisional-selective",
    },
    "openthoughts3": {
        "dataset_id": "open-thoughts/OpenThoughts3-1.2M",
        "revision": "61bcf9d4eb38b30295efc2021227a63cc5bb34c8",
        "card_sha256": "8bc5c1c91e5fce307d33b049701606f2a3fb73521c4bae89e16cdacf1aa9b2e1",
        "repository_bytes": 28_188_906_161,
        "license_policy": "Apache-2.0 dataset card",
        "decision": "provisional-deduplicate",
    },
    "opencodereasoning2": {
        "dataset_id": "nvidia/OpenCodeReasoning-2",
        "revision": "eadf535931451525f3e5621d0f960c240bc62fd9",
        "card_sha256": "41342603f3f2562dcb9355a59d7bb809c16097fc926e7befc7f05f5d6d9df809",
        "repository_bytes": 49_411_866_253,
        "license_policy": "CC-BY-4.0 dataset card",
        "decision": "provisional",
    },
    "open_swe_traces": {
        "dataset_id": "nvidia/Open-SWE-Traces",
        "revision": "0b7d2a801a3b91541a48f8bca03e5ea90fd1fa5c",
        "card_sha256": "ae4642e7b3e312483e9cbb7a19f2212d7a3b5f4c795e927d35e4af9f7dbeffcf",
        "repository_bytes": 51_978_939_098,
        "license_policy": "CC-BY-4.0 dataset plus recorded source-repository licenses",
        "decision": "provisional-contamination-audit",
    },
    "swe_next": {
        "dataset_id": "TIGER-Lab/SWE-Next-SFT-Trajectories",
        "revision": "e378a60ddd7050fe9519a31a4d41d4872eeec6ac",
        "card_sha256": "0ff9ca92be883e10c6d3436acf30d526e91282dda8bd49dd192a0940d8db4951",
        "repository_bytes": 215_820_383,
        "license_policy": "MIT dataset card; source repositories still require review",
        "decision": "provisional-contamination-audit",
    },
    "mixture_of_thoughts": {
        "dataset_id": "open-r1/Mixture-of-Thoughts",
        "revision": "e55fa28006c0d0ec60fb3547520f775dd42d02cd",
        "card_sha256": "f69d7fff62298c19e813fa5e2d32e4cfdcd8f0aa6a387492c355bb210dc93c52",
        "repository_bytes": 6_070_569_169,
        "license_policy": "no repository-level license declared in pinned card",
        "decision": "excluded-pending-license",
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi(token=False)
    receipts = {}
    for name, expected in SOURCES.items():
        info = api.dataset_info(
            expected["dataset_id"], revision=expected["revision"],
            files_metadata=True)
        if info.private or info.gated or info.sha != expected["revision"]:
            raise RuntimeError(f"{name}: repository visibility or revision mismatch")
        repository_bytes = sum((item.size or 0) for item in info.siblings)
        if repository_bytes != expected["repository_bytes"]:
            raise RuntimeError(f"{name}: repository byte count changed")
        card = Path(hf_hub_download(
            expected["dataset_id"], "README.md", repo_type="dataset",
            revision=expected["revision"], token=False))
        if digest(card) != expected["card_sha256"]:
            raise RuntimeError(f"{name}: dataset card digest mismatch")
        receipts[name] = {
            **expected,
            "public": True,
            "gated": False,
            "files": len(info.siblings),
            "verified": True,
        }
    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "scope": "metadata and dataset cards only; no training payload admitted by this receipt",
        "sources": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
