from __future__ import annotations

import torch

from scripts.prepare_e97_hf_release import (
    CONFIGURATION,
    MODELING,
    agent_card,
    base_card,
    clone_state,
    public_args,
)


def test_public_args_removes_training_paths_and_disables_triton() -> None:
    result = public_args(
        {
            "level": "E97",
            "dim": 1792,
            "depth": 11,
            "mlp_ratio": 2.2623,
            "use_triton": 1,
            "data": "/private/corpus",
            "output": "/private/run",
        }
    )
    assert result == {
        "level": "E97",
        "dim": 1792,
        "depth": 11,
        "mlp_ratio": 2.2623,
        "use_triton": False,
    }


def test_clone_state_adds_wrapper_prefix_and_breaks_shared_storage() -> None:
    tensor = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    result = clone_state({"embedding.weight": tensor, "lm_head.weight": tensor})
    assert set(result) == {"model.embedding.weight", "model.lm_head.weight"}
    assert torch.equal(result["model.embedding.weight"], result["model.lm_head.weight"])
    assert (
        result["model.embedding.weight"].untyped_storage().data_ptr()
        != result["model.lm_head.weight"].untyped_storage().data_ptr()
    )


def test_release_templates_encode_recurrent_transformers_contract() -> None:
    assert 'use_cache=kwargs.pop("use_cache", False)' in CONFIGURATION
    assert "self.num_hidden_layers" in CONFIGURATION
    assert "self.post_init()" in MODELING
    assert '"model.lm_head.weight": "model.embedding.weight"' in MODELING
    assert "use_triton=False" in MODELING


def test_cards_make_bounded_claims_and_preserve_license_uncertainty() -> None:
    base = base_card("spinozans/base", 513_013_841_920, 2_322_520, 2.2798, "a" * 64)
    agent = agent_card("spinozans/agent", "b" * 64)
    assert "raw base LM, not an instruction or chat model" in base
    assert "not the repository's separate GDN2 control" in base
    assert "No standalone model license" in base
    assert "40/40" in agent
    assert "not a general instruction or coding assistant" in agent
    assert "No standalone model license" in agent
