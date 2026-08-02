#!/usr/bin/env python3
"""Validate the one-time legacy total-token migration receipt.

This deliberately reads checkpoint filesystem identity, not checkpoint tensor
contents.  ``train.py`` remains responsible for deciding whether the checkpoint
has an embedded clock.  A receipt can bootstrap only the exact legacy artifact
and stable run identity to which it was issued.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA = "emender-total-token-legacy-migration-v1"


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:  # bool is intentionally rejected
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"migration receipt {name} mismatch: {actual!r} != {expected!r}")


def validate_migration_receipt(
    receipt_path: str | Path,
    *,
    checkpoint: str | Path,
    run_id: str,
    run_dir: str | Path,
    latest: str | Path,
    expected_step: int | None = None,
    expected_total_tokens: int | None = None,
    expected_source_job_id: int | None = None,
    expected_size_bytes: int | None = None,
) -> int:
    """Return the trusted bootstrap count after exact binding validation."""
    receipt_path = Path(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read migration receipt {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise TypeError("migration receipt must be a JSON object")
    _require_equal(receipt.get("schema"), SCHEMA, "schema")

    identity = receipt.get("run_identity")
    checkpoint_record = receipt.get("checkpoint")
    accounting = receipt.get("token_accounting")
    if not all(isinstance(value, dict) for value in (identity, checkpoint_record, accounting)):
        raise TypeError("migration receipt identity/checkpoint/token_accounting must be objects")

    run_dir_actual = _lexical_absolute(run_dir)
    latest_actual = _lexical_absolute(latest)
    checkpoint_input = Path(checkpoint)
    try:
        checkpoint_actual = checkpoint_input.resolve(strict=True)
        latest_resolved = latest_actual.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"migration checkpoint/latest is not readable: {exc}") from exc
    if not checkpoint_actual.is_file():
        raise ValueError(f"migration checkpoint is not a regular file: {checkpoint_actual}")
    if not latest_actual.is_symlink():
        raise ValueError(f"migration latest path is not an atomic symlink: {latest_actual}")
    _require_equal(latest_resolved, checkpoint_actual, "latest target")

    _require_equal(identity.get("run_id"), run_id, "run_id")
    _require_equal(identity.get("run_dir"), str(run_dir_actual), "run_dir")
    _require_equal(identity.get("latest_path"), str(latest_actual), "latest_path")
    _require_equal(latest_actual, run_dir_actual / "train/latest.pt", "stable latest path")

    _require_equal(
        checkpoint_record.get("resolved_path"),
        str(checkpoint_actual),
        "resolved checkpoint path",
    )
    _require_equal(checkpoint_record.get("basename"), checkpoint_actual.name, "checkpoint basename")
    size_bytes = _integer(checkpoint_record.get("size_bytes"), "checkpoint.size_bytes", positive=True)
    _require_equal(size_bytes, checkpoint_actual.stat().st_size, "checkpoint size")
    step = _integer(checkpoint_record.get("step"), "checkpoint.step", positive=True)
    source_job_id = _integer(
        checkpoint_record.get("source_job_id"), "checkpoint.source_job_id", positive=True
    )
    match = re.fullmatch(r"checkpoint_step_([0-9]+)_loss_[^/]+\.pt", checkpoint_actual.name)
    if match is None or int(match.group(1)) != step:
        raise ValueError("migration receipt checkpoint filename/step mismatch")

    seed_step = _integer(accounting.get("seed_step"), "token_accounting.seed_step")
    seed_tokens = _integer(
        accounting.get("seed_total_tokens"), "token_accounting.seed_total_tokens"
    )
    completed_steps = _integer(
        accounting.get("completed_steps"), "token_accounting.completed_steps"
    )
    world_size = _integer(accounting.get("world_size"), "token_accounting.world_size", positive=True)
    batch_size = _integer(accounting.get("batch_size"), "token_accounting.batch_size", positive=True)
    chunk_size = _integer(accounting.get("chunk_size"), "token_accounting.chunk_size", positive=True)
    grad_accum = _integer(accounting.get("grad_accum"), "token_accounting.grad_accum", positive=True)
    tokens_per_step = _integer(
        accounting.get("tokens_per_step"), "token_accounting.tokens_per_step", positive=True
    )
    total_tokens = _integer(accounting.get("total_tokens"), "token_accounting.total_tokens")
    _require_equal(seed_step + completed_steps, step, "step arithmetic")
    _require_equal(
        world_size * batch_size * chunk_size * grad_accum,
        tokens_per_step,
        "tokens_per_step arithmetic",
    )
    _require_equal(
        seed_tokens + completed_steps * tokens_per_step,
        total_tokens,
        "total_tokens arithmetic",
    )

    for actual, expected, name in (
        (step, expected_step, "expected step"),
        (total_tokens, expected_total_tokens, "expected total_tokens"),
        (source_job_id, expected_source_job_id, "expected source job"),
        (size_bytes, expected_size_bytes, "expected size"),
    ):
        if expected is not None:
            _require_equal(actual, expected, name)
    return total_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--latest", required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-total-tokens", type=int)
    parser.add_argument("--expected-source-job-id", type=int)
    parser.add_argument("--expected-size-bytes", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    total_tokens = validate_migration_receipt(
        args.receipt,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        run_dir=args.run_dir,
        latest=args.latest,
        expected_step=args.expected_step,
        expected_total_tokens=args.expected_total_tokens,
        expected_source_job_id=args.expected_source_job_id,
        expected_size_bytes=args.expected_size_bytes,
    )
    print(total_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
