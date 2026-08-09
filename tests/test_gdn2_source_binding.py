import json
from pathlib import Path
import subprocess
import sys

import pytest

from ndm.models.external_gdn2 import (
    BOUND_SOURCE_RECEIPT,
    verify_bound_gdn2_source,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / "src/GatedDeltaNet-2"
COMMIT = "95709fc250357c2dd109361c353192f2aa5913f9"


def test_stage_and_verify_exact_gdn2_source(tmp_path):
    output = tmp_path / "gdn2"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/stage_gdn2_source.py"),
        "--checkout", str(CHECKOUT),
        "--expected-commit", COMMIT,
        "--output", str(output),
    ], cwd=ROOT, check=True)
    receipt = verify_bound_gdn2_source(output, COMMIT)
    assert receipt["commit"] == COMMIT
    assert (output / "lit_gpt/gdn2.py").is_file()


def test_wrong_commit_and_modified_stage_fail_closed(tmp_path):
    output = tmp_path / "gdn2"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/stage_gdn2_source.py"),
        "--checkout", str(CHECKOUT),
        "--expected-commit", COMMIT,
        "--output", str(output),
    ], cwd=ROOT, check=True)
    with pytest.raises(RuntimeError, match="!= required"):
        verify_bound_gdn2_source(output, "0" * 40)
    with (output / "lit_gpt/gdn2.py").open("a") as handle:
        handle.write("\n# modified\n")
    with pytest.raises(RuntimeError, match="tree digest mismatch"):
        verify_bound_gdn2_source(output, COMMIT)


def test_missing_or_forged_receipt_fails_closed(tmp_path):
    output = tmp_path / "gdn2"
    output.mkdir()
    with pytest.raises(RuntimeError, match="receipt missing"):
        verify_bound_gdn2_source(output, COMMIT)
    (output / BOUND_SOURCE_RECEIPT).write_text(json.dumps({
        "schema": "emender-gdn2-source-v1",
        "commit": COMMIT,
        "source_tree_sha256": "0" * 64,
    }))
    with pytest.raises(RuntimeError, match="tree digest mismatch"):
        verify_bound_gdn2_source(output, COMMIT)
