import math
from argparse import Namespace

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
