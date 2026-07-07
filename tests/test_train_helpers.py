import math
import sys
from argparse import Namespace

import pytest
import torch

import train


def _tiny_e97_args(**overrides):
    values = {
        "level": "E97",
        "params": "100m",
        "dim": 8,
        "depth": 1,
        "tokenizer": None,
        "use_triton": 0,
        "bf16": False,
        "e88_raw_write": 0,
        "expansion": 1.0,
        "n_groups": 2,
        "n_state": 4,
        "n_slots": 4,
        "n_heads": 2,
        "top_k": None,
        "k_fast": None,
        "k_slow": None,
        "use_gate": 1,
        "gate_activation": "sigmoid",
        "linear_state": 1,
        "use_write_gate": 0,
        "e88_decay_mode": "mamba",
        "e88_value_residual": 0,
        "use_chunked_e97": 0,
        "e97_chunk_size": 4,
        "state_expansion": 2,
        "r_h_mode": "none",
        "use_conv": 0,
        "d_conv": 4,
        "gdn2_mlp_ratio": 0.0,
        "dropout": 0.0,
        "checkpoint_interval": 16,
        "gradient_checkpointing": False,
        "projection_chunk_size": 0,
        "loss_chunk_size": 0,
        "mlp_ratio": 0.0,
        "mlp_multiple": 8,
        "head_type_logits": None,
        "corner_mixture": None,
        "lam_max": None,
        "beta_max": None,
        "igain_max": None,
        "layer_kwargs": None,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "optimizer": "adamw",
        "knob_lr_mult": 1.0,
        "grad_accum": 1,
        "grad_clip": 1.0,
        "steps": 4,
        "min_lr_frac": 0.1,
        "tbptt": False,
        "batch_size": 2,
        "chunk_size": 8,
    }
    values.update(overrides)
    return Namespace(**values)


def _parallel_args(**overrides):
    values = {
        "diloco": False,
        "async_quorum_diloco": False,
        "diloco_k": 250,
        "diloco_outer_lr": 1.0,
        "diloco_outer_beta": 0.0,
        "diloco_outer_optimizer": "avg",
        "diloco_island_size": 0,
        "async_quorum_min_workers": 0,
        "async_quorum_fraction": 1.0,
        "async_quorum_timeout_seconds": 300.0,
        "async_quorum_staleness_policy": "reject-stale",
        "async_quorum_max_staleness": 0,
        "async_quorum_update_representation": "delta",
        "async_quorum_metrics_path": None,
        "async_quorum_run_id": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _fixed_batches(batch):
    lengths = torch.full((batch.shape[0],), batch.shape[1], dtype=torch.long)
    doc_end = torch.zeros(batch.shape[0], dtype=torch.bool)
    while True:
        yield batch.clone(), doc_end.clone(), lengths.clone()


def test_training_helpers_build_tiny_e97_without_cli_side_effects():
    args = _tiny_e97_args()

    train.normalize_training_args(args)
    model = train.build_training_model(args)
    optimizer = train.build_training_optimizer(model, args)

    assert args.level == "E97"
    assert args.use_triton == 0
    assert model.get_num_params() > 0
    assert optimizer.param_groups[0]["base_lr"] == args.lr


def test_train_one_optimizer_step_runs_real_tiny_e97_path():
    torch.manual_seed(1234)
    args = _tiny_e97_args()
    model = train.build_training_model(args)
    optimizer = train.build_training_optimizer(model, args)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    batch = torch.randint(0, 256, (args.batch_size, args.chunk_size + 1), dtype=torch.long)

    metrics = train.train_one_optimizer_step(
        model,
        optimizer,
        args,
        batch_iter=_fixed_batches(batch),
        step=0,
    )

    after = [p.detach() for p in model.parameters() if p.requires_grad]
    assert metrics["step"] == 1
    assert math.isfinite(metrics["loss"])
    assert metrics["tokens_processed"] == args.batch_size * (args.chunk_size + 1)
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_async_quorum_mode_selection_disables_ddp_but_keeps_distributed_rank_mapping():
    args = _parallel_args(
        async_quorum_diloco=True,
        async_quorum_min_workers=6,
        async_quorum_fraction=0.75,
        async_quorum_timeout_seconds=45.0,
        async_quorum_metrics_path="metrics.jsonl",
        async_quorum_run_id="run-123",
    )

    mode = train.resolve_training_parallel_mode(
        args,
        dist_enabled=True,
        rank=3,
        local_rank=3,
        world_size=8,
    )

    assert mode.name == "async_quorum_diloco"
    assert mode.use_ddp is False
    assert mode.use_diloco is False
    assert mode.use_async_quorum_diloco is True
    assert mode.rank == 3
    assert mode.local_rank == 3
    assert mode.world_size == 8
    assert mode.is_main is False
    assert mode.one_rank_per_gpu is True
    assert mode.quorum_threshold == 6
    assert args._ddp_enabled is False
    assert args._dist_enabled is True
    assert args._use_diloco is False
    assert args._use_async_quorum_diloco is True
    assert args._async_quorum_threshold == 6


def test_async_quorum_cli_flags_parse(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            "train.bin",
            "--async_quorum_diloco",
            "--diloco_k",
            "40",
            "--async_quorum_min_workers",
            "7",
            "--async_quorum_fraction",
            "0.875",
            "--async_quorum_timeout_seconds",
            "90",
            "--async_quorum_staleness_policy",
            "reject-stale",
            "--async_quorum_update_representation",
            "delta",
            "--async_quorum_metrics_path",
            "metrics.jsonl",
            "--async_quorum_run_id",
            "e97-async",
        ],
    )

    args = train.parse_args()

    assert args.async_quorum_diloco is True
    assert args.diloco_k == 40
    assert args.async_quorum_min_workers == 7
    assert args.async_quorum_fraction == 0.875
    assert args.async_quorum_timeout_seconds == 90
    assert args.async_quorum_staleness_policy == "reject-stale"
    assert args.async_quorum_update_representation == "delta"
    assert args.async_quorum_metrics_path == "metrics.jsonl"
    assert args.async_quorum_run_id == "e97-async"


def test_async_quorum_rejects_diloco_hybrid_island_ddp():
    args = _parallel_args(async_quorum_diloco=True, diloco_island_size=2)

    with pytest.raises(ValueError, match="diloco_island_size"):
        train.resolve_training_parallel_mode(
            args,
            dist_enabled=True,
            rank=0,
            local_rank=0,
            world_size=8,
        )


def test_async_quorum_rejects_sync_diloco_and_non_avg_outer_state():
    with pytest.raises(ValueError, match="mutually exclusive"):
        train.resolve_training_parallel_mode(
            _parallel_args(async_quorum_diloco=True, diloco=True),
            dist_enabled=True,
            rank=0,
            local_rank=0,
            world_size=8,
        )

    with pytest.raises(ValueError, match="diloco_outer_optimizer"):
        train.resolve_training_parallel_mode(
            _parallel_args(
                async_quorum_diloco=True,
                diloco_outer_optimizer="momentum",
            ),
            dist_enabled=True,
            rank=0,
            local_rank=0,
            world_size=8,
        )


def test_existing_ddp_and_sync_diloco_mode_selection_unchanged():
    ddp_args = _parallel_args()
    ddp_mode = train.resolve_training_parallel_mode(
        ddp_args,
        dist_enabled=True,
        rank=0,
        local_rank=0,
        world_size=4,
    )
    assert ddp_mode.name == "ddp"
    assert ddp_mode.use_ddp is True
    assert ddp_mode.use_diloco is False
    assert ddp_mode.use_async_quorum_diloco is False

    diloco_args = _parallel_args(diloco=True)
    diloco_mode = train.resolve_training_parallel_mode(
        diloco_args,
        dist_enabled=True,
        rank=0,
        local_rank=0,
        world_size=4,
    )
    assert diloco_mode.name == "diloco"
    assert diloco_mode.use_ddp is False
    assert diloco_mode.use_diloco is True
    assert diloco_mode.use_async_quorum_diloco is False
