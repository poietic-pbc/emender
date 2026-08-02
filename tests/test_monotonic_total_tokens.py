import json
from pathlib import Path

import pytest
import torch

import train
from scripts.frontier.validate_total_token_migration_receipt import (
    validate_migration_receipt,
)


REPO = Path(__file__).resolve().parents[1]
RECEIPT = REPO / "docs/validation/e97-total-token-migration-step2303840.json"
CANONICAL_RUN_ID = "e97-final-seed-production-256n"
CANONICAL_RUN_DIR = Path(
    "/lustre/orion/bif148/proj-shared/emender/frontier_runs/"
    "final-seed-production-256n/runs/e97-final-seed-production-256n"
)
CANONICAL_CHECKPOINT = CANONICAL_RUN_DIR / "train/checkpoint_step_2303840_loss_2.3178.pt"


def _legacy_checkpoint(step: int = 2300930) -> dict:
    return {
        "step": step,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "checkpoint_metadata": {"kind": "legacy"},
    }


def test_initial_seed_legacy_bootstrap_requires_explicit_trusted_count():
    checkpoint = _legacy_checkpoint()

    with pytest.raises(ValueError, match="legacy checkpoint.*--total_tokens"):
        train.resolve_checkpoint_total_tokens(checkpoint, None, "seed.pt")

    assert train.resolve_checkpoint_total_tokens(
        checkpoint, 150_793_748_480, "seed.pt"
    ) == 150_793_748_480


def test_periodic_and_final_checkpoint_roundtrip_embed_total_tokens(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    periodic = train.save_checkpoint(
        model,
        optimizer,
        step=2301130,
        loss=2.4,
        output_dir=tmp_path,
        keep_n=2,
        total_tokens=154_149_191_680,
        metadata={"kind": "periodic"},
    )
    final = train.save_checkpoint(
        model,
        optimizer,
        step=2301330,
        loss=2.3,
        output_dir=tmp_path,
        keep_n=2,
        total_tokens=157_504_634_880,
        metadata={"kind": "final"},
    )

    for path, expected, kind in (
        (periodic, 154_149_191_680, "periodic"),
        (final, 157_504_634_880, "final"),
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert type(checkpoint["total_tokens"]) is int
        assert checkpoint["total_tokens"] == expected
        assert checkpoint["checkpoint_metadata"]["total_tokens"] == expected
        assert checkpoint["checkpoint_metadata"]["kind"] == kind
        assert train.resolve_checkpoint_total_tokens(checkpoint, None, path) == expected

    latest = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["total_tokens"] == 157_504_634_880


def test_exact_per_optimizer_step_increment_and_grad_accumulation():
    per_step = train.tokens_per_global_optimizer_step(
        world_size=2048, batch_size=4, chunk_size=2048, grad_accum=1
    )
    assert per_step == 16_777_216
    assert train.advance_total_tokens(150_793_748_480, 2048, 4, 2048, 1) == (
        150_810_525_696
    )

    assert train.tokens_per_global_optimizer_step(32, 3, 1024, 4) == 393_216
    assert train.advance_total_tokens(10, 32, 3, 1024, 4) == 393_226


def test_changed_world_resume_preserves_clock_then_uses_new_world_increment():
    checkpoint = {
        "step": 12,
        "total_tokens": 1_000_000,
        "checkpoint_metadata": {"total_tokens": 1_000_000},
    }
    resumed = train.resolve_checkpoint_total_tokens(checkpoint, None, "latest.pt")

    assert resumed == 1_000_000
    assert train.advance_total_tokens(resumed, 8, 4, 2048, 1) == 1_065_536


def test_failed_epoch_rolls_back_to_last_committed_checkpoint_count(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    committed = train.save_checkpoint(
        model,
        optimizer,
        step=200,
        loss=2.0,
        output_dir=tmp_path,
        total_tokens=900_000,
    )

    execution_epoch_only = train.advance_total_tokens(900_000, 16, 4, 2048, 2)
    assert execution_epoch_only > 900_000
    reloaded = torch.load(committed, map_location="cpu", weights_only=False)
    assert train.resolve_checkpoint_total_tokens(reloaded, None, committed) == 900_000


def test_explicit_and_embedded_or_metadata_mismatch_fail_closed():
    checkpoint = {
        "total_tokens": 100,
        "checkpoint_metadata": {"total_tokens": 100},
    }
    with pytest.raises(ValueError, match="explicit total_tokens.*embedded"):
        train.resolve_checkpoint_total_tokens(checkpoint, 101, "latest.pt")

    checkpoint["checkpoint_metadata"]["total_tokens"] = 99
    with pytest.raises(ValueError, match="metadata total_tokens.*top-level"):
        train.resolve_checkpoint_total_tokens(checkpoint, None, "latest.pt")


@pytest.mark.parametrize("value", [-1, 1.5, True, "10"])
def test_invalid_token_counts_fail_closed(value):
    checkpoint = {
        "total_tokens": value,
        "checkpoint_metadata": {"total_tokens": value},
    }
    with pytest.raises((TypeError, ValueError), match="total_tokens"):
        train.resolve_checkpoint_total_tokens(checkpoint, None, "latest.pt")


def _write_bound_receipt(run_dir: Path, checkpoint: Path, *, total_tokens: int) -> Path:
    receipt = {
        "schema": "emender-total-token-legacy-migration-v1",
        "run_identity": {
            "run_id": "test-run",
            "run_dir": str(run_dir.resolve()),
            "latest_path": str(run_dir.resolve() / "train/latest.pt"),
        },
        "checkpoint": {
            "resolved_path": str(checkpoint.resolve()),
            "basename": checkpoint.name,
            "step": 2303840,
            "size_bytes": checkpoint.stat().st_size,
            "source_job_id": 5134243,
        },
        "token_accounting": {
            "seed_step": 2300930,
            "seed_total_tokens": 150793748480,
            "completed_steps": 2910,
            "world_size": 2048,
            "batch_size": 4,
            "chunk_size": 2048,
            "grad_accum": 1,
            "tokens_per_step": 16777216,
            "total_tokens": total_tokens,
        },
    }
    path = run_dir / "receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return path


def test_legacy_migration_receipt_is_bound_to_path_size_step_job_and_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True)
    checkpoint = train_dir / "checkpoint_step_2303840_loss_2.3178.pt"
    checkpoint.write_bytes(b"legacy-checkpoint")
    latest = train_dir / "latest.pt"
    latest.symlink_to(checkpoint.name)
    receipt = _write_bound_receipt(run_dir, checkpoint, total_tokens=199_615_447_040)

    result = validate_migration_receipt(
        receipt,
        checkpoint=latest,
        run_id="test-run",
        run_dir=run_dir,
        latest=latest,
    )
    assert result == 199_615_447_040

    other = train_dir / "checkpoint_step_2303840_loss_2.3178-copy.pt"
    other.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match="latest target|resolved checkpoint path"):
        validate_migration_receipt(
            receipt,
            checkpoint=other,
            run_id="test-run",
            run_dir=run_dir,
            latest=latest,
        )
    with pytest.raises(ValueError, match="run_id"):
        validate_migration_receipt(
            receipt,
            checkpoint=latest,
            run_id="other-run",
            run_dir=run_dir,
            latest=latest,
        )


def test_canonical_migration_receipt_records_exact_one_time_authority():
    receipt = json.loads(RECEIPT.read_text())

    assert receipt["schema"] == "emender-total-token-legacy-migration-v1"
    assert receipt["run_identity"] == {
        "run_id": CANONICAL_RUN_ID,
        "run_dir": str(CANONICAL_RUN_DIR),
        "latest_path": str(CANONICAL_RUN_DIR / "train/latest.pt"),
    }
    assert receipt["checkpoint"] == {
        "resolved_path": str(CANONICAL_CHECKPOINT),
        "basename": CANONICAL_CHECKPOINT.name,
        "step": 2303840,
        "size_bytes": 7719680116,
        "source_job_id": 5134243,
    }
    assert receipt["token_accounting"] == {
        "seed_step": 2300930,
        "seed_total_tokens": 150793748480,
        "completed_steps": 2910,
        "world_size": 2048,
        "batch_size": 4,
        "chunk_size": 2048,
        "grad_accum": 1,
        "tokens_per_step": 16777216,
        "total_tokens": 199615447040,
    }
