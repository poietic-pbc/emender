from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from scripts.prepare_gdn2_hf_release import GDN2_SOURCE_COMMIT, model_card, public_args


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "hf_templates" / "gdn2" / "modeling_emender_gdn2.py"


def _load_modeling():
    # The remote module uses a relative config import; provide a minimal package.
    package = "_test_gdn2_hf"
    config_path = MODEL_PATH.with_name("configuration_emender_gdn2.py")
    config_spec = importlib.util.spec_from_file_location(f"{package}.configuration_emender_gdn2", config_path)
    config_mod = importlib.util.module_from_spec(config_spec)
    config_mod.__package__ = package
    import sys, types
    pkg = types.ModuleType(package); pkg.__path__ = [str(MODEL_PATH.parent)]
    sys.modules[package] = pkg
    sys.modules[config_spec.name] = config_mod
    assert config_spec.loader is not None; config_spec.loader.exec_module(config_mod)
    spec = importlib.util.spec_from_file_location(f"{package}.modeling_emender_gdn2", MODEL_PATH)
    mod = importlib.util.module_from_spec(spec); mod.__package__ = package
    sys.modules[spec.name] = mod
    assert spec.loader is not None; spec.loader.exec_module(mod)
    return mod


def test_public_args_is_sanitized_and_explicit() -> None:
    result = public_args({
        "level": "gdn2-mlp", "dim": 2176, "depth": 12, "n_heads": 30,
        "expansion": 1.0, "use_conv": 1, "d_conv": 4,
        "gdn2_mlp_ratio": 3.25, "tokenizer": "p50k_base",
        "data": "/private/pile.txt", "output": "/private/run",
    })
    assert result["implementation"] == "portable_reference"
    assert result["head_dim"] == 128
    assert result["allow_neg_eigval"] is False
    assert "data" not in result and "output" not in result


def test_portable_model_has_checkpoint_compatible_names_and_finite_logits() -> None:
    mod = _load_modeling()
    args = {
        "dim": 16, "depth": 2, "n_heads": 2, "head_dim": 4,
        "d_conv": 3, "gdn2_mlp_ratio": 2.0, "gdn2_mlp_multiple": 4,
    }
    model = mod.PortableGDN2LM(args, vocab_size=31).bfloat16().eval()
    keys = set(model.state_dict())
    assert "layers.0.gdn2.gdn2.q_proj.weight" in keys
    assert "layers.0.gdn2.gdn2.q_conv1d.weight" in keys
    assert "layers.0.norm_2.weight" in keys
    assert "layers.0.mlp.w3.weight" in keys
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert logits.shape == (1, 3, 31)
    assert torch.isfinite(logits).all()


def test_card_is_bounded_and_records_external_source_policy() -> None:
    card = model_card(
        "spinozans/gdn2-mlp-1.3b", tokens=152_280_498_176,
        step=2_323_616, loss=2.4034, source_sha="a" * 64,
        checkpoint_size=7_720_577_595,
    )
    assert "1.287B-parameter" in card
    assert "not a 152B-parameter model" in card
    assert "does not redistribute NVIDIA source" in card
    assert GDN2_SOURCE_COMMIT in card
    assert "license: other" in card
