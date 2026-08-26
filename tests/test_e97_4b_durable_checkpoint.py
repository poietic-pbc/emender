import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "scripts/archive_e97_4b_durable_checkpoint.py"
PUBLISH = ROOT / "scripts/publish_e97_4b_durable_checkpoint.sh"
DOWNLOAD = ROOT / "scripts/frontier/download_e97_4b_durable_checkpoint.sh"


def test_archive_stages_checksum_and_fixed_world_receipt(tmp_path):
    source_dir = tmp_path / "run"
    source_dir.mkdir()
    checkpoint = source_dir / "checkpoint_step_000012_loss_2.5000.pt"
    torch.save(
        {
            "step": 12,
            "total_tokens": 6_291_456,
            "loss": 2.5,
            "model_state_dict": {"x": torch.ones(2)},
            "optimizer_state_dict": {},
            "checkpoint_metadata": {
                "kind": "periodic",
                "is_head": True,
                "world_size": 8,
                "model": {"total_params": 4_045_972_080},
                "sampler": {
                    "identity": {
                        "schema": "emender-byte-window-counter-v1",
                        "corpus_sha256": "a" * 64,
                        "tokenizer_sha256": "b" * 64,
                        "sampler_key": 42,
                        "data_world_size": 8,
                        "context_size": 2048,
                    },
                    "total_accepted_tokens": 6_291_456,
                    "absolute_rank_sample_index": 384,
                },
            },
        },
        checkpoint,
    )
    args_json = source_dir / "args.json"
    args_json.write_text("{}\n")
    durable = tmp_path / "durable"
    staging = tmp_path / "staging"
    result = subprocess.run(
        [
            sys.executable,
            str(ARCHIVE),
            "--checkpoint", str(checkpoint),
            "--args-json", str(args_json),
            "--durable-root", str(durable),
            "--staging-root", str(staging),
            "--source-commit", "c" * 40,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["step"] == 12
    assert receipt["world_size"] == 8
    assert receipt["frontier_world256_exact_resume_compatible"] is False
    archived = durable / "step_000012_tokens_6291456" / checkpoint.name
    staged = staging / "checkpoints/step_000012_tokens_6291456" / checkpoint.name
    assert os.path.samefile(checkpoint, archived)
    assert os.path.samefile(checkpoint, staged)
    assert (durable / "latest").resolve() == archived.parent
    assert len(receipt["checkpoint_sha256"]) == 64
    assert json.loads((staging / "LATEST.json").read_text())["checkpoint"] == checkpoint.name


def test_publish_and_frontier_download_fail_closed_contracts():
    publish = PUBLISH.read_text()
    assert "PUBLISH_CONFIRM" in publish
    assert "SOURCE_COMMIT" in publish
    assert "hf upload-large-folder" in publish
    download = DOWNLOAD.read_text()
    assert "source scripts/frontier/activate_emender_frontier.sh" in download
    assert "REVISION must be a full Hub commit" in download
    assert "sha256sum -c SHA256SUMS" in download
    assert "exact resume world mismatch" in download
    assert "CONFIRM_MODEL_ONLY" in download
