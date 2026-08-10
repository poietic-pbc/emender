import json
from pathlib import Path
import struct
import subprocess
import sys


INDEX = struct.Struct("<QQQI")


def _inventory(root: Path, name: str, records):
    offset = 0
    with (root / f"{name}.records").open("wb") as data, (root / f"{name}.index").open("wb") as index:
        for payload, tokens in records:
            raw = payload.encode()
            data.write(raw)
            index.write(INDEX.pack(offset, len(raw), tokens, 0))
            offset += len(raw)


def test_builder_writes_main_and_long_rs_streams(tmp_path):
    inventory = tmp_path / "inventory"
    output = tmp_path / "output"
    inventory.mkdir()
    _inventory(inventory, "a", [("User:\na", 3), ("User:\nlong-a", 8)])
    _inventory(inventory, "b", [("Assistant:\nb", 4), ("Assistant:\nlong-b", 9)])
    spec = {
        "seed": 7, "tokenizer": "p50k_base", "tokenizer_sha256": "a" * 64,
        "long_context_min_tokens": 8,
        "sources": [
            {"name": "a", "target_tokens": 10},
            {"name": "b", "target_tokens": 11},
        ],
        "long_context": {"target_tokens": 12},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    subprocess.run([
        sys.executable, "scripts/build_e97_instruction_corpus.py",
        "--spec", str(spec_path), "--inventory-root", str(inventory),
        "--output-root", str(output), "--buckets", "4",
    ], check=True, capture_output=True, text=True)
    main = (output / "e97_instruction_50b_v1.txt").read_bytes()
    long = (output / "e97_instruction_50b_v1_long32k.txt").read_bytes()
    assert b"\x1e" in main
    assert all(len(record) for record in main.split(b"\x1e"))
    assert all(b"long" in record for record in long.split(b"\x1e"))
    manifest = json.loads((output / "e97_instruction_50b_v1.manifest.json").read_text())
    assert manifest["main"]["rs_count"] == manifest["main"]["records"] - 1
    assert manifest["long32k"]["records"] >= 1
