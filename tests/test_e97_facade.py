from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ndm.e97 import (
    build_e97_model,
    e97_model_kwargs_from_config,
    generate_e97,
    load_e97_checkpoint,
)
from ndm.models import E97SplitEditLayer, LadderLM
from ndm.models.e88_fla_hybrid import E88FLAHybrid
from ndm.triton.e97_sequential import e97_split_edit_triton_apply
from scripts.generate_racer_samples import build_model as build_generation_model


def tiny_e97_config(**overrides):
    config = {
        "level": "E97",
        "dim": 32,
        "depth": 1,
        "n_heads": 2,
        "n_state": 8,
        "expansion": 1.0,
        "n_groups": 4,
        "n_slots": 8,
        "use_gate": 1,
        "gate_activation": "silu",
        "linear_state": 0,
        "use_write_gate": 0,
        "e88_decay_mode": "mamba",
        "e88_value_residual": 0,
        "e88_raw_write": 0,
        "use_chunked_e97": 0,
        "e97_chunk_size": 32,
        "use_triton": 0,
        "use_conv": 0,
        "mlp_ratio": 1.0,
        "mlp_multiple": 16,
        "tokenizer": None,
        "optimizer": "adamw",
    }
    config.update(overrides)
    return config


def e97_layers(model):
    return [module for module in model.modules() if isinstance(module, E97SplitEditLayer)]


def test_e97_level_has_distinct_type_without_changing_state_dict_schema():
    e97 = E97SplitEditLayer(dim=32, n_heads=2, n_state=8, use_gate=True)
    legacy = E88FLAHybrid(
        dim=32,
        n_heads=2,
        n_state=8,
        use_gate=True,
        use_split_edit=True,
    )

    assert e97.use_split_edit is True
    assert e97.architecture_name == "emender/nonlin"
    assert e97.historical_level == "E97"
    assert set(e97.state_dict()) == set(legacy.state_dict())
    assert {
        key: tuple(value.shape) for key, value in e97.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in legacy.state_dict().items()
    }

    model = LadderLM(
        vocab_size=64,
        dim=32,
        depth=1,
        level="E97",
        n_heads=2,
        n_state=8,
        use_gate=True,
        gate_activation="silu",
    )
    assert len(e97_layers(model)) == 1


def test_e97_layer_cannot_disable_split_edit():
    with pytest.raises(ValueError, match="requires use_split_edit=True"):
        E97SplitEditLayer(
            dim=32,
            n_heads=2,
            n_state=8,
            use_split_edit=False,
        )


def test_e97_sequential_facade_requires_both_split_edit_gates():
    tensor = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="requires both erase_gate and value_write_gate"):
        e97_split_edit_triton_apply(
            False,
            tensor,
            tensor,
            tensor,
            tensor[..., 0],
            erase_gate=None,
            value_write_gate=tensor,
        )


def test_e97_sequential_facade_delegates_to_shared_core(monkeypatch):
    shared = importlib.import_module("ndm.triton.e88_triton_optimized")
    sentinel = (object(), object())
    received = {}

    def fake_apply(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(shared, "e88_triton_optimized_apply", fake_apply)
    tensor = torch.zeros(1, 1, 1, 1)
    result = e97_split_edit_triton_apply(
        False,
        tensor,
        tensor,
        tensor,
        tensor[..., 0],
        erase_gate=tensor,
        value_write_gate=tensor,
    )

    assert result is sentinel
    assert received["kwargs"]["erase_gate"] is tensor
    assert received["kwargs"]["value_write_gate"] is tensor
    assert received["kwargs"]["linear_state"] is False


def test_150b_shape_config_maps_to_complete_e97_model_arguments():
    config = tiny_e97_config(
        dim=1792,
        depth=11,
        n_heads=216,
        n_state=32,
        use_triton=1,
        mlp_ratio=2.2623,
        mlp_multiple=64,
        state_summary_dim=None,
    )
    kwargs = e97_model_kwargs_from_config(config, vocab_size=50281)

    assert kwargs["level"] == "E97"
    assert kwargs["dim"] == 1792
    assert kwargs["depth"] == 11
    assert kwargs["n_heads"] == 216
    assert kwargs["n_state"] == 32
    assert kwargs["use_triton"] is True
    assert kwargs["use_chunked_e97"] is False
    assert kwargs["linear_state"] is False
    assert kwargs["e88_raw_write"] is False
    assert kwargs["mlp_ratio"] == 2.2623
    assert kwargs["mlp_multiple"] == 64
    assert kwargs["state_summary_dim"] == 0


def test_existing_generation_builder_routes_e97_through_public_model():
    model = build_generation_model(tiny_e97_config(), vocab_size=256)
    assert len(e97_layers(model)) == 1
    assert model.mlp_ratio == 1.0


def test_e97_checkpoint_roundtrip_and_cpu_generation(tmp_path):
    config = tiny_e97_config()
    original = build_e97_model(config, vocab_size=256, use_triton=False)
    run_dir = tmp_path / "emender_E97_tiny"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(json.dumps(config))
    checkpoint_path = run_dir / "checkpoint_step_000123_loss_2.5000.pt"
    torch.save(
        {
            "step": 123,
            "loss": 2.5,
            "model_state_dict": original.state_dict(),
        },
        checkpoint_path,
    )
    (run_dir / "latest.pt").symlink_to(checkpoint_path.name)

    loaded = load_e97_checkpoint(
        run_dir,
        device="cpu",
        dtype="float32",
        use_triton=False,
    )
    assert loaded.step == 123
    assert loaded.loss == 2.5
    assert loaded.checkpoint_path == checkpoint_path.resolve()
    assert loaded.schedulefree_train_weight_swap is False
    assert len(e97_layers(loaded.model)) == 1
    for key, value in original.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key]), key

    result = generate_e97(
        loaded,
        "hello",
        max_new_tokens=2,
        temperature=0,
        mode="stateful",
        seed=7,
    )
    assert result["mode"] == "stateful"
    assert result["model"] == "emender/nonlin"
    assert result["historical_level"] == "E97"
    assert result["kernel_api"] == "e97-split-edit-eager"
    assert result["kernel_core"] == "python-reference"
    assert len(result["new_token_ids"]) == 2


def test_fused_e97_generation_rejects_stateful_carry(monkeypatch):
    e97_module = importlib.import_module("ndm.e97")
    config = tiny_e97_config(use_triton=1, mlp_ratio=0.0)
    loaded_model = build_e97_model(config, vocab_size=256, use_triton=True)
    loaded = SimpleNamespace(
        model=loaded_model,
        config=config,
        checkpoint_path=None,
        step=0,
        weight_mode="saved",
    )
    monkeypatch.setattr(e97_module, "_uses_triton", lambda model: True)

    with pytest.raises(ValueError, match="requires load_e97_checkpoint"):
        generate_e97(loaded, "x", max_new_tokens=1, mode="stateful")


def test_schedulefree_checkpoint_requires_optimizer_for_train_weights(tmp_path):
    config = tiny_e97_config(optimizer="schedulefree", use_triton=1)
    model = build_e97_model(config, vocab_size=256, use_triton=False)
    run_dir = tmp_path / "schedulefree_without_optimizer"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(json.dumps(config))
    checkpoint_path = run_dir / "checkpoint_step_000001_loss_3.0000.pt"
    torch.save(
        {"step": 1, "loss": 3.0, "model_state_dict": model.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="has no optimizer_state_dict"):
        load_e97_checkpoint(checkpoint_path, weight_mode="train", mmap=True)

    loaded = load_e97_checkpoint(checkpoint_path, weight_mode="saved", mmap=True)
    assert loaded.weight_mode == "saved"
    assert loaded.schedulefree_train_weight_swap is False
    assert all(layer.use_triton is False for layer in e97_layers(loaded.model))


def test_e97_runtime_banner_separates_model_and_shared_core(capsys):
    layer = E97SplitEditLayer(
        dim=32,
        n_heads=2,
        n_state=8,
        use_gate=True,
        use_triton=True,
    )
    layer._maybe_log_runtime_path(
        torch.zeros(1, 1, 32),
        use_optimized=True,
        use_chunked=False,
        log_decay=True,
    )
    output = capsys.readouterr().out
    assert "model=emender/nonlin" in output
    assert "historical_level=E97" in output
    assert "path=e97-sequential-split-edit-triton" in output
    assert "kernel_core=e88-shared-triton" in output


def test_generate_e97_cli_human_flow(tmp_path):
    config = tiny_e97_config()
    model = build_e97_model(config, vocab_size=256, use_triton=False)
    run_dir = tmp_path / "cli_e97_run"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(json.dumps(config))
    torch.save(
        {
            "step": 9,
            "loss": 2.9,
            "model_state_dict": model.state_dict(),
        },
        run_dir / "checkpoint_step_000009_loss_2.9000.pt",
    )

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/generate_e97.py",
            "--checkpoint",
            str(run_dir),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--temperature",
            "0",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--mode",
            "stateful",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "loaded E97 step=9" in completed.stderr
    assert "hello" in completed.stdout
