#!/usr/bin/env python3
"""Select, externally shuffle, and write the two E97 instruction text files."""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import time

import numpy as np

RS = b"\x1e"
INDEX = struct.Struct("<QQQI")
LENGTH = struct.Struct("<Q")
INDEX_DTYPE = np.dtype([
    ("offset", "<u8"), ("bytes", "<u8"), ("tokens", "<u8"), ("replaced", "<u4")])


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def choose_epoch_indices(rng: np.random.Generator, n: int):
    while True:
        yield from rng.permutation(n)


def spool_selection(*, source_name: str, target: int, index: np.ndarray,
                    records: mmap.mmap, rng: np.random.Generator,
                    bucket_handles: list, eligible: np.ndarray | None = None,
                    mirror_handles: list | None = None) -> dict:
    candidates = np.arange(len(index), dtype=np.int64) if eligible is None else eligible
    if len(candidates) == 0:
        raise RuntimeError(f"{source_name} has no eligible records")
    selected_tokens = selected_bytes = selected_records = epochs = 0
    while selected_tokens < target:
        epochs += 1
        permutation = rng.permutation(len(candidates))
        contribution = index["tokens"][candidates[permutation]].astype(np.uint64) + 1
        remaining = target - selected_tokens
        cumulative = np.cumsum(contribution, dtype=np.uint64)
        take = min(len(permutation), int(np.searchsorted(cumulative, remaining)) + 1)
        chosen = candidates[permutation[:take]]
        selected_tokens += int(contribution[:take].sum(dtype=np.uint64))
        # Selection is random, but physical reads are sorted. The later bucket
        # shuffle removes this temporary order and avoids millions of random
        # small reads from the Lustre inventory stream.
        for i in np.sort(chosen):
            row = index[int(i)]
            start, size = int(row["offset"]), int(row["bytes"])
            payload = records[start:start + size]
            bucket = int(rng.integers(len(bucket_handles)))
            bucket_handles[bucket].write(LENGTH.pack(size))
            bucket_handles[bucket].write(payload)
            if mirror_handles is not None:
                mirror = int(rng.integers(len(mirror_handles)))
                mirror_handles[mirror].write(LENGTH.pack(size))
                mirror_handles[mirror].write(payload)
            selected_bytes += size
            selected_records += 1
    return {
        "source": source_name, "target_tokens": target,
        "actual_contribution_tokens": selected_tokens,
        "payload_tokens": selected_tokens - selected_records,
        "rs_tokens": selected_records,
        "overshoot_tokens": selected_tokens - target,
        "selected_records": selected_records, "selected_payload_bytes": selected_bytes,
        "unique_candidates": len(candidates), "epochs_started": epochs,
    }


def bucket_offsets(path: Path) -> np.ndarray:
    values = []
    with path.open("rb") as handle:
        while True:
            pos = handle.tell()
            raw = handle.read(LENGTH.size)
            if not raw:
                break
            if len(raw) != LENGTH.size:
                raise RuntimeError(f"truncated length in {path}")
            size, = LENGTH.unpack(raw)
            values.append((handle.tell(), size))
            handle.seek(size, 1)
    return np.asarray(values, dtype=np.uint64)


def emit(buckets: list[Path], output: Path, rng: np.random.Generator) -> dict:
    partial = output.with_suffix(output.suffix + ".partial")
    sha = hashlib.sha256()
    records = payload_bytes = 0
    order = rng.permutation(len(buckets))
    with partial.open("wb", buffering=8 * 1024 * 1024) as out:
        for bucket_id in order:
            path = buckets[int(bucket_id)]
            offsets = bucket_offsets(path)
            if not len(offsets):
                continue
            # Buckets are deliberately memory-sized. Read each sequentially
            # once, then shuffle record slices in RAM rather than issuing
            # random small reads to Lustre.
            data = path.read_bytes()
            for j in rng.permutation(len(offsets)):
                offset, size = map(int, offsets[int(j)])
                payload = data[offset:offset + size]
                if records:
                    out.write(RS); sha.update(RS)
                out.write(payload); sha.update(payload)
                records += 1; payload_bytes += size
    partial.replace(output)
    return {"path": str(output), "records": records, "payload_bytes": payload_bytes,
            "file_bytes": output.stat().st_size, "rs_count": max(0, records - 1),
            "sha256": sha.hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--inventory-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--readme", type=Path)
    p.add_argument("--buckets", type=int, default=256)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    main_output = args.output_root / "e97_instruction_50b_v1.txt"
    long_output = args.output_root / "e97_instruction_50b_v1_long32k.txt"
    for output in (main_output, long_output):
        if output.exists() and not args.force:
            raise SystemExit(f"{output} exists; pass --force")
    spool = args.output_root / "shuffle-spool"
    spool.mkdir(exist_ok=True)
    main_paths = [spool / f"main-{i:04d}.bin" for i in range(args.buckets)]
    long_paths = [spool / f"long-{i:04d}.bin" for i in range(args.buckets)]
    for path in main_paths + long_paths:
        if path.exists(): path.unlink()
    main_handles = [x.open("wb", buffering=4 * 1024 * 1024) for x in main_paths]
    long_handles = [x.open("wb", buffering=4 * 1024 * 1024) for x in long_paths]
    rng = np.random.default_rng(int(spec["seed"]))
    opened = []
    selections = []
    long_candidates = []
    try:
        for source_id, source in enumerate(spec["sources"]):
            name = source["name"]
            index_path = args.inventory_root / f"{name}.index"
            records_path = args.inventory_root / f"{name}.records"
            index = np.fromfile(index_path, dtype=INDEX_DTYPE)
            if index_path.stat().st_size != len(index) * INDEX.size:
                raise RuntimeError(f"invalid index size for {name}")
            handle = records_path.open("rb")
            records = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            opened.append((handle, records))
            receipt = spool_selection(
                source_name=name, target=int(source["target_tokens"]), index=index,
                records=records, rng=rng, bucket_handles=main_handles)
            selections.append(receipt)
            eligible = np.flatnonzero(index["tokens"] >= int(spec["long_context_min_tokens"]))
            if len(eligible):
                long_candidates.append((name, index, records, eligible))
            print(json.dumps(receipt, sort_keys=True), flush=True)
        if not long_candidates:
            raise RuntimeError("no >=32K records for long-context tranche")
        # Draw uniformly from the eligible record union. Record lengths then
        # determine their contribution naturally; no record is split.
        union = []
        for pool_id, (_name, index, _records, eligible) in enumerate(long_candidates):
            for i in eligible:
                union.append((pool_id, int(i), int(index[int(i)]["tokens"])))
        target = int(spec["long_context"]["target_tokens"])
        actual = payload_tokens = payload_bytes = count = 0
        selected_long = []
        while actual < target:
            item = union[int(rng.integers(len(union)))]
            selected_long.append(item)
            actual += item[2] + 1
        # As above, sort physical reads while preserving randomized bucket
        # assignment and final output order.
        for pool_id, row_id, tokens in sorted(selected_long):
            name, index, records, _eligible = long_candidates[pool_id]
            row = index[row_id]; start, size = int(row["offset"]), int(row["bytes"])
            payload = records[start:start + size]
            for handles in (main_handles, long_handles):
                b = int(rng.integers(len(handles)))
                handles[b].write(LENGTH.pack(size)); handles[b].write(payload)
            payload_tokens += tokens; payload_bytes += size; count += 1
        selections.append({"source": "long32k", "target_tokens": target,
                           "actual_contribution_tokens": actual,
                           "payload_tokens": payload_tokens, "rs_tokens": count,
                           "overshoot_tokens": actual - target,
                           "selected_records": count, "selected_payload_bytes": payload_bytes,
                           "unique_candidates": len(union)})
    finally:
        for x in main_handles + long_handles: x.close()
        for handle, records in opened:
            records.close(); handle.close()
    main_receipt = emit(main_paths, main_output, rng)
    long_receipt = emit(long_paths, long_output, rng)
    main_accounted_tokens = sum(
        int(row["actual_contribution_tokens"]) for row in selections) - 1
    long_row = next(row for row in selections if row["source"] == "long32k")
    manifest = {
        "schema": "emender-e97-instruction-corpus-v1",
        "created_unix": time.time(), "seed": spec["seed"], "delimiter_hex": "1e",
        "target_tokens": int(spec["target_tokens"]),
        "main_accounted_tokens": main_accounted_tokens,
        "main_overshoot_tokens": main_accounted_tokens - int(spec["target_tokens"]),
        "long32k_accounted_tokens": int(long_row["actual_contribution_tokens"]) - 1,
        "tokenizer": spec["tokenizer"],
        "tokenizer_sha256": spec["tokenizer_sha256"],
        "token_accounting": "p50k payload tokens plus one RS token per selected occurrence; final stream has one fewer RS",
        "selections": selections, "main": main_receipt, "long32k": long_receipt,
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
    }
    write_json(args.output_root / "e97_instruction_50b_v1.manifest.json", manifest)
    (args.output_root / "e97_instruction_50b_v1.sources.json").write_bytes(
        args.spec.read_bytes())
    if args.readme is not None:
        (args.output_root / "README.md").write_bytes(args.readme.read_bytes())
    (args.output_root / "e97_instruction_50b_v1.sha256").write_text(
        f"{main_receipt['sha256']}  {main_output.name}\n"
        f"{long_receipt['sha256']}  {long_output.name}\n")
    for path in main_paths + long_paths: path.unlink()
    spool.rmdir()
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
