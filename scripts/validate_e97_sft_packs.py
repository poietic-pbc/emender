#!/usr/bin/env python3
"""Independently validate a complete-record masked-SFT pack authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ndm.data.masked_sft_dataset import PACK_INDEX, PACK_SCHEMA, RECORD_INDEX, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--authority-manifest-sha256", required=True)
    parser.add_argument("--pack-manifest-sha256", required=True)
    args = parser.parse_args()
    authority_path = args.authority_root / "manifest.json"
    pack_path = args.pack_root / "manifest.json"
    if sha256(authority_path) != args.authority_manifest_sha256:
        raise SystemExit("authority manifest digest mismatch")
    if sha256(pack_path) != args.pack_manifest_sha256:
        raise SystemExit("pack manifest digest mismatch")
    authority = json.loads(authority_path.read_text())
    packs = json.loads(pack_path.read_text())
    if packs.get("schema") != PACK_SCHEMA or packs.get("status") != "complete":
        raise SystemExit("pack manifest is not complete")
    if packs.get("authority_manifest_sha256") != args.authority_manifest_sha256:
        raise SystemExit("pack manifest does not bind the token authority")
    for info in packs["outputs"].values():
        path = args.pack_root / Path(info["path"]).name
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise SystemExit(f"pack output integrity mismatch: {path.name}")

    record_info = authority["outputs"]["index"]
    record_path = args.authority_root / Path(record_info["path"]).name
    if record_path.stat().st_size != record_info["bytes"] or sha256(record_path) != record_info["sha256"]:
        raise SystemExit("record authority integrity mismatch")
    records = np.memmap(
        record_path, mode="r", dtype=np.dtype([
            ("offset", "<u8"), ("tokens", "<u8"), ("targets", "<u8"),
            ("split", "u1"), ("pad", "V7")]))
    source_filter = packs.get("source_filter")
    source_selected = np.ones(len(records), dtype=np.bool_)
    if source_filter is not None:
        included = source_filter.get("include_exact")
        if not isinstance(included, list) or not included or any(
                not isinstance(value, str) for value in included):
            raise SystemExit("invalid pack source filter")
        metadata_info = authority["outputs"].get("metadata")
        if metadata_info is None:
            raise SystemExit("source-filtered packs require authority metadata")
        metadata_path = args.authority_root / Path(metadata_info["path"]).name
        if (metadata_path.stat().st_size != metadata_info["bytes"]
                or sha256(metadata_path) != metadata_info["sha256"]):
            raise SystemExit("record metadata integrity mismatch")
        wanted = set(included)
        selected_values = []
        with metadata_path.open() as stream:
            for line in stream:
                selected_values.append(json.loads(line).get("source") in wanted)
        if len(selected_values) != len(records):
            raise SystemExit("record metadata/index count mismatch")
        source_selected = np.asarray(selected_values, dtype=np.bool_)
    ids_info = packs["outputs"]["pack_records"]
    record_ids = np.memmap(
        args.pack_root / Path(ids_info["path"]).name, mode="r", dtype="<u4")
    sequence_tokens = int(packs["sequence_tokens"])
    observed_ids = []
    for split_name, split_value in (("train", 0), ("validation", 1)):
        info = packs["outputs"][f"{split_name}_index"]
        index = np.memmap(
            args.pack_root / Path(info["path"]).name, mode="r",
            dtype=np.dtype([("record_offset", "<u8"), ("record_count", "<u8"),
                            ("tokens", "<u8"), ("targets", "<u8")]))
        counts = {"packs": len(index), "records": 0, "tokens": 0,
                  "assistant_target_tokens": 0}
        for pack in index:
            start = int(pack["record_offset"])
            stop = start + int(pack["record_count"])
            ids = record_ids[start:stop]
            selected = records[ids]
            if (len(ids) == 0 or np.any(selected["split"] != split_value)
                    or int(selected["tokens"].sum()) != int(pack["tokens"])
                    or int(selected["targets"].sum()) != int(pack["targets"])
                    or int(pack["tokens"]) > sequence_tokens):
                raise SystemExit(f"invalid {split_name} pack descriptor")
            observed_ids.extend(int(value) for value in ids)
            counts["records"] += len(ids)
            counts["tokens"] += int(pack["tokens"])
            counts["assistant_target_tokens"] += int(pack["targets"])
        expected = packs["splits"][split_name]
        for field, value in counts.items():
            if value != expected[field]:
                raise SystemExit(f"{split_name} {field} accounting mismatch")
        eligible = np.flatnonzero(
            (records["split"] == split_value) & (records["tokens"] <= sequence_tokens)
            & source_selected)
        selected_ids = np.asarray(
            [value for value in observed_ids if int(records[value]["split"]) == split_value],
            dtype=np.int64)
        if not np.array_equal(np.sort(selected_ids), eligible):
            raise SystemExit(f"{split_name} eligible records are missing or duplicated")
    if len(set(observed_ids)) != len(observed_ids):
        raise SystemExit("record appears in more than one pack")
    receipt = {
        "schema": "emender-e97-sft-pack-validation-v1", "status": "pass",
        "authority_manifest_sha256": args.authority_manifest_sha256,
        "pack_manifest_sha256": args.pack_manifest_sha256,
        "records_validated": len(observed_ids),
        "packs_validated": sum(value["packs"] for value in packs["splits"].values()),
        "output_sha256": {name: value["sha256"] for name, value in packs["outputs"].items()},
    }
    destination = args.pack_root / "validation.json"
    destination.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
