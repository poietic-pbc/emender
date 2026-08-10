#!/usr/bin/env python3
"""Download pinned E97 instruction-corpus source snapshots, with receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def tree_receipt(root: Path) -> tuple[list[dict], int]:
    rows, total = [], 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total += size
        rows.append({"path": str(path.relative_to(root)), "size": size})
    return rows, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=True)
    results = []
    failures = []
    all_sources = spec["sources"] + spec.get("prompt_restoration_sources", [])
    for source in all_sources:
        name, repo, revision = source["name"], source["repo"], source["revision"]
        local = args.output_root / name
        started = time.time()
        row = {"name": name, "repo": repo, "revision": revision,
               "local_dir": str(local), "started_unix": started}
        print(f"DOWNLOAD {repo}@{revision} -> {local}", flush=True)
        try:
            info = api.dataset_info(repo, revision=revision, files_metadata=True)
            if info.sha != revision:
                raise RuntimeError(f"resolved revision {info.sha} != pinned {revision}")
            snapshot_download(
                repo_id=repo, repo_type="dataset", revision=revision,
                local_dir=local, token=True, max_workers=args.max_workers)
            files, total = tree_receipt(local)
            row.update(status="complete", completed_unix=time.time(),
                       elapsed_s=time.time() - started, files=files,
                       total_bytes=total, resolved_revision=info.sha,
                       gated=info.gated, licenses=[
                           x.removeprefix("license:") for x in info.tags
                           if x.startswith("license:")])
        except Exception as exc:
            row.update(status="failed", completed_unix=time.time(),
                       elapsed_s=time.time() - started,
                       error_type=type(exc).__name__, error=str(exc))
            failures.append(name)
            print(f"FAILED {repo}: {type(exc).__name__}: {exc}", flush=True)
        results.append(row)
        write_json(args.output_root.parent / "download-receipt.partial.json", {
            "schema": "emender-e97-instruction-download-v1",
            "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            "sources": results,
            "failures": failures,
        })
    if failures:
        raise SystemExit("source downloads failed: " + ", ".join(failures))
    receipt = args.output_root.parent / "download-receipt.json"
    (args.output_root.parent / "download-receipt.partial.json").replace(receipt)
    print(f"COMPLETE receipt={receipt}")


if __name__ == "__main__":
    main()
