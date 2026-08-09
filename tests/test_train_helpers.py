import importlib.util
import math
from argparse import Namespace
from pathlib import Path

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


def test_train_one_optimizer_step_reports_phase_timestamps_in_execution_order():
    torch.manual_seed(1234)
    args = _tiny_e97_args()
    model = train.build_training_model(args)
    optimizer = train.build_training_optimizer(model, args)
    batch = torch.randint(0, 256, (args.batch_size, args.chunk_size + 1), dtype=torch.long)
    phases = []

    train.train_one_optimizer_step(
        model, optimizer, args, batch_iter=_fixed_batches(batch), step=7,
        phase_callback=lambda name, details: phases.append((name, details)),
    )

    assert [name for name, _ in phases] == [
        "optimizer_step_start", "data_load_start", "data_load_end",
        "forward_start", "forward_end", "backward_start", "backward_end",
        "loss_sync_start", "loss_sync_end", "optimizer_update_start",
        "optimizer_update_end", "optimizer_step_end",
    ]
    assert all(details["step"] == 7 for _, details in phases)
    assert phases[-1][1]["tokens"] == args.batch_size * (args.chunk_size + 1)


def test_training_dataset_seed_includes_resume_step_and_global_rank(monkeypatch):
    captured = {}

    class Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train, "TokenizedStreamDataset", Dataset)
    args = Namespace(
        seed=42, tbptt=False, tokenizer="p50k_base", data="corpus.txt",
        chunk_size=2048, batch_size=4)
    train.build_training_dataset(
        args, rank=17, dist_enabled=True, start_step=2_322_520)
    assert captured["seed"] == 2_322_579
    assert captured["sampler_identity"] is None


def _counter_sampler_args(**overrides):
    values = dict(
        seed=42, tbptt=False, tokenizer="p50k_base", data="corpus.txt",
        chunk_size=2048, batch_size=2,
        sampler_schema=train.COUNTER_SAMPLER_SCHEMA,
        sampler_corpus_sha256="1" * 64,
        sampler_tokenizer_sha256="2" * 64,
        sampler_key=42,
        sampler_data_world_size=4,
        total_tokens=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_training_dataset_uses_accepted_token_cursor_and_fixed_identity(monkeypatch):
    captured = {}

    class Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train, "TokenizedStreamDataset", Dataset)
    args = _counter_sampler_args()
    train.build_training_dataset(
        args, rank=2, dist_enabled=True, world_size=4,
        total_accepted_tokens=4 * 2048 * 6)

    identity = captured["sampler_identity"]
    assert identity.data_world_size == 4
    assert identity.context_size == 2048
    assert captured["rank"] == 2
    assert captured["chunk_size"] == 2049
    assert captured["accepted_tokens_per_sample"] == 2048
    assert captured["total_accepted_tokens"] == 4 * 2048 * 6


def test_dense_counter_v2_uses_boundary_relative_origin(monkeypatch):
    captured = {}

    class Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(train, "TokenizedStreamDataset", Dataset)
    origin = 150_134_063_104
    args = _counter_sampler_args(
        sampler_schema=train.BOUNDARY_COUNTER_SAMPLER_SCHEMA,
        sampler_stream_origin_accepted_tokens=origin)
    train.build_training_dataset(
        args, rank=2, dist_enabled=True, world_size=4,
        total_accepted_tokens=origin)
    identity = captured["sampler_identity"]
    assert identity.schema == train.BOUNDARY_COUNTER_SAMPLER_SCHEMA
    assert identity.stream_origin_accepted_tokens == origin
    assert captured["total_accepted_tokens"] == origin


def test_dense_e97_gdn2_and_moe_resolve_identical_samples(tmp_path, monkeypatch):
    import ndm.data.tokenized_dataset as td

    class Encoding:
        n_vocab = 512

        def encode(self, text, disallowed_special=()):
            return [ord(char) % self.n_vocab for char in text]

    monkeypatch.setattr(td.tiktoken, "get_encoding", lambda _name: Encoding())
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(32 + (index * 37) % 95 for index in range(32768)))
    dense_args = _counter_sampler_args(data=str(corpus))
    dense_identity = train.counter_sampler_identity_from_args(
        dense_args, world_size=4)

    runner_path = (Path(__file__).resolve().parents[1]
                   / "scripts/frontier/e97_35b_moe_train.py")
    spec = importlib.util.spec_from_file_location("e97_moe_sampler_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    moe_args = Namespace(
        sampler_schema=dense_args.sampler_schema,
        sampler_corpus_sha256=dense_args.sampler_corpus_sha256,
        sampler_tokenizer_sha256=dense_args.sampler_tokenizer_sha256,
        sampler_key=dense_args.sampler_key,
        sampler_data_world_size=dense_args.sampler_data_world_size,
        sampler_transition_from_legacy=False,
        chunk_size=dense_args.chunk_size,
    )
    moe_identity = module._sampler_identity(moe_args, world_size=4)
    assert moe_identity == dense_identity

    # E97 and GDN2 both use build_training_dataset; MoE uses the same dataset
    # identity/API. Model selection therefore cannot alter IDs or tensors.
    streams = [
        td.TokenizedStreamDataset(
            str(corpus), 2049, rank=2, world_size=4,
            tokenizer_name="p50k_base", sampler_identity=resolved,
            total_accepted_tokens=4 * 2048 * 3,
            accepted_tokens_per_sample=2048)
        for resolved in (dense_identity, dense_identity, moe_identity)
    ]
    batches = [stream.get_batch(2)[0].clone() for stream in streams]
    ids = [stream.last_batch_sample_ids for stream in streams]
    assert ids[0] == ids[1] == ids[2]
    assert torch.equal(batches[0], batches[1])
    assert torch.equal(batches[0], batches[2])


def test_dense_checkpoint_sampler_metadata_survives_atomic_latest(tmp_path):
    args = _counter_sampler_args()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    total_tokens = 4 * 2048 * 6
    sampler = train.dense_sampler_checkpoint_metadata(
        args, world_size=4, total_tokens=total_tokens)

    path = train.save_checkpoint(
        model, optimizer, 3, 1.0, tmp_path, total_tokens=total_tokens,
        metadata={"sampler": sampler})
    assert (tmp_path / "latest.pt").resolve() == path.resolve()
    checkpoint = torch.load(
        tmp_path / "latest.pt", map_location="cpu", mmap=True,
        weights_only=False)
    assert train.validate_dense_checkpoint_sampler(
        checkpoint, args, world_size=4, checkpoint_path=path) == total_tokens
    assert checkpoint["checkpoint_metadata"]["sampler"] == sampler


def test_dense_sampler_mismatch_fails_before_model_mutation(tmp_path):
    args = _counter_sampler_args()
    source = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(source.parameters())
    total_tokens = 4 * 2048 * 2
    sampler = train.dense_sampler_checkpoint_metadata(
        args, world_size=4, total_tokens=total_tokens)
    path = train.save_checkpoint(
        source, optimizer, 1, 1.0, tmp_path, total_tokens=total_tokens,
        metadata={"sampler": sampler})

    target = torch.nn.Linear(2, 2)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    mismatched = _counter_sampler_args(sampler_tokenizer_sha256="3" * 64)
    with pytest.raises(ValueError, match="sampler identity mismatch"):
        train.load_checkpoint(
            path, target,
            preflight=lambda checkpoint: train.validate_dense_checkpoint_sampler(
                checkpoint, mismatched, world_size=4,
                checkpoint_path=path))
    assert all(torch.equal(before[name], value)
               for name, value in target.state_dict().items())


def test_counter_sampler_refuses_legacy_checkpoint_metadata():
    args = _counter_sampler_args()
    checkpoint = {"total_tokens": 0, "checkpoint_metadata": {"total_tokens": 0}}
    with pytest.raises(ValueError, match="cannot be silently relabelled"):
        train.validate_dense_checkpoint_sampler(
            checkpoint, args, world_size=4, checkpoint_path="legacy.pt")


def test_real_worker_reports_bootstrap_phases_before_training(monkeypatch):
    import ndm.async_diloco_real as real

    phases = []
    monkeypatch.setattr(real.train, "build_training_model", lambda args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(real.train, "build_training_optimizer",
                        lambda model, args: torch.optim.SGD(model.parameters(), lr=0.1))
    monkeypatch.setattr(real, "_build_batch_iter", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(real.train, "train_one_optimizer_step", lambda *args, **kwargs: {
        "loss": 1.0, "tokens_processed": 1, "hidden_state": None})
    args = Namespace(seed=1, bf16=False, lr=0.1)
    model = torch.nn.Linear(1, 1)
    report = real._run_real_worker(
        run_id="run", generation=0, base_state=model.state_dict(), train_args=args,
        spec=real.RealAsyncWorkerSpec("trainer", "node-0", "cpu", 1, 0),
        synthetic_token_stream=False, synthetic_vocab_size=16,
        phase_callback=lambda name, details: phases.append((name, details)))
    assert report.error is None
    assert [name for name, _ in phases[:7]] == [
        "model_build_start", "model_device_ready", "model_state_loaded",
        "model_dtype_ready", "optimizer_built", "optimizer_state_loaded",
        "data_iterator_ready",
    ]
