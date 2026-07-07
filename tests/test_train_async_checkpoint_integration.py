import json
from argparse import Namespace

import train
from ndm.async_diloco import (
    GLOBAL_MERGER_ROLE,
    AsyncDiLoCoCheckpointCadence,
    AsyncDiLoCoCheckpointManager,
    AsyncDiLoCoGenerationMetrics,
)


def _metric(generation):
    return AsyncDiLoCoGenerationMetrics(
        run_id="train-run",
        generation=generation,
        requested_workers=2,
        participating_workers=2,
        quorum_threshold=1,
        quorum_size=2,
        accepted_updates=2,
        stale_updates=0,
        timed_out_updates=0,
        failed_updates=0,
        invalid_updates=0,
        generation_duration_s=1.0,
        merge_duration_s=0.1,
        rebase_duration_s=0.1,
        checkpoint_duration_s=0.0,
        tokens_per_sec=100.0,
        tokens_per_generation=200,
    )


def _manager(path, role=GLOBAL_MERGER_ROLE):
    return AsyncDiLoCoCheckpointManager(
        path,
        run_id="train-run",
        role=role,
        cadence=AsyncDiLoCoCheckpointCadence(
            recovery_every_generations=None,
            recovery_every_seconds=None,
            export_every_generations=None,
            export_every_seconds=None,
        ),
    )


def test_train_async_resume_prefers_finalized_global_latest_pt(tmp_path):
    checkpoint = tmp_path / "global_gen_000003.pt"
    checkpoint.write_bytes(b"checkpoint")
    _manager(tmp_path).publish_global_generation(_metric(3), checkpoint_path=checkpoint)
    _manager(tmp_path, role="worker").emit_cached_manifest(
        generation=9,
        payload={"checkpoint_path": str(tmp_path / "cache" / "update.pt")},
    )

    selected = train.resolve_async_quorum_resume_checkpoint(
        Namespace(
            resume=None,
            async_diloco_checkpoint_dir=str(tmp_path),
            async_diloco_run_id="train-run",
            async_diloco_debug_resume_from_cache=None,
        )
    )

    assert selected == str(checkpoint)


def test_train_async_resume_ignores_worker_cache_latest_manifest(tmp_path):
    checkpoint = tmp_path / "global_gen_000001.pt"
    checkpoint.write_bytes(b"checkpoint")
    _manager(tmp_path).publish_global_generation(_metric(1), checkpoint_path=checkpoint)
    cache_checkpoint = tmp_path / "cache" / "worker" / "update.pt"
    cache_checkpoint.parent.mkdir(parents=True)
    cache_checkpoint.write_bytes(b"cache")
    latest_pt = tmp_path / "latest.pt"
    latest_pt.unlink()
    latest_pt.symlink_to(cache_checkpoint)
    cache_latest = {
        "run_id": "train-run",
        "generation": 99,
        "checkpoint_path": str(cache_checkpoint),
        "published_by": "worker",
        "finalized": False,
    }
    (tmp_path / "latest.json").write_text(json.dumps(cache_latest), encoding="utf-8")

    selected = train.resolve_async_quorum_resume_checkpoint(
        Namespace(
            resume=None,
            async_diloco_checkpoint_dir=str(tmp_path),
            async_diloco_run_id="train-run",
            async_diloco_debug_resume_from_cache=None,
        )
    )

    assert selected == str(checkpoint)


def test_train_debug_flag_can_explicitly_select_cache_resume(tmp_path):
    cache_checkpoint = tmp_path / "cache" / "worker" / "update.pt"
    cache_checkpoint.parent.mkdir(parents=True)
    cache_checkpoint.write_bytes(b"debug")

    selected = train.resolve_async_quorum_resume_checkpoint(
        Namespace(
            resume=None,
            async_diloco_checkpoint_dir=str(tmp_path),
            async_diloco_run_id="train-run",
            async_diloco_debug_resume_from_cache=str(cache_checkpoint),
        )
    )

    assert selected == str(cache_checkpoint)
