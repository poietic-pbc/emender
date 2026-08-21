from argparse import Namespace
from pathlib import Path

import torch

import train


def test_slurm_env_fallback_derives_rank_and_rank_local_device(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "3")

    status = train.resolve_distributed_env_from_slurm(device_count=1)

    assert status == "derived-from-slurm"
    assert train.os.environ["WORLD_SIZE"] == "8"
    assert train.os.environ["RANK"] == "3"
    assert train.os.environ["LOCAL_RANK"] == "0"


def test_slurm_env_fallback_preserves_exported_world_size(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "3")

    status = train.resolve_distributed_env_from_slurm(device_count=1)

    assert status == "world-size-present"
    assert train.os.environ["RANK"] == "5"
    assert train.os.environ["LOCAL_RANK"] == "0"


def test_exact_output_dir_is_stable_across_supervised_execution_epochs(tmp_path):
    exact = tmp_path / "stable-run" / "train"
    args = Namespace(
        output=str(tmp_path / "unused"),
        exact_output_dir=str(exact),
        level="E97",
        params="100m",
        resume=str(exact / "latest.pt"),
        diloco_bootstrap_outer_state="none",
    )

    first_epoch = train.setup_output_dir(args)
    second_epoch = train.setup_output_dir(args)

    assert first_epoch == second_epoch == exact
    assert (exact / "args.json").is_file()
    assert (exact / "run_manifest.json").is_file()


def test_save_checkpoint_atomically_updates_latest_and_keeps_newest(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = train.save_checkpoint(
        model, optimizer, 1, 3.0, tmp_path, keep_n=2, total_tokens=10)
    second = train.save_checkpoint(
        model, optimizer, 2, 2.0, tmp_path, keep_n=2, total_tokens=20)
    third = train.save_checkpoint(
        model, optimizer, 3, 1.0, tmp_path, keep_n=2, total_tokens=30)

    latest = tmp_path / "latest.pt"
    assert latest.is_symlink()
    assert latest.readlink() == Path(third.name)
    assert not first.exists()
    assert second.exists()
    assert third.exists()
    assert sorted(path.name for path in tmp_path.glob("checkpoint_step_*.pt")) == [
        second.name,
        third.name,
    ]


def test_same_step_final_save_keeps_latest_target_with_retention_one(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    periodic = train.save_checkpoint(
        model, optimizer, 2, 5.5010, tmp_path, keep_n=1, total_tokens=20)
    final = train.save_checkpoint(
        model, optimizer, 2, 5.5308, tmp_path, keep_n=1, total_tokens=20)

    latest = tmp_path / "latest.pt"
    assert not periodic.exists()
    assert final.exists()
    assert latest.is_symlink()
    assert latest.resolve(strict=True) == final
