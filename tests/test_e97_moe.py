import copy

import pytest
import torch

from ndm.models.e97_moe import (
    E97MoEConfig,
    NodeLocalSharedRoutedMoE,
    SharedRoutedMoE,
    calculate_e97_moe_recipe,
    convert_e97_ffns_to_moe,
    convert_e97_ffns_to_node_local_moe,
    expert_owner,
    experts_for_rank,
    e97_moe_auxiliary_loss,
    widen_swiglu_function_preserving,
)
from ndm.models.ladder_lm import LadderLM, MixerMLPWrapper, SwiGLUMLP


def _seed_ffn(dim=8, hidden=12):
    torch.manual_seed(17)
    return SwiGLUMLP(dim, hidden)


def test_widened_ffn_is_function_preserving_and_new_down_columns_are_zero():
    seed = _seed_ffn()
    x = torch.randn(3, 5, 8)
    widened = widen_swiglu_function_preserving(seed, 20)

    # GEMM accumulation order changes when the down projection widens even
    # though the added columns are exactly zero.
    torch.testing.assert_close(widened(x), seed(x), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(widened.w1.weight[:12], seed.w1.weight)
    torch.testing.assert_close(widened.w2.weight[:12], seed.w2.weight)
    torch.testing.assert_close(widened.w3.weight[:, :12], seed.w3.weight)
    assert torch.count_nonzero(widened.w3.weight[:, 12:]) == 0
    assert torch.count_nonzero(widened.w1.weight[12:]) > 0
    assert torch.count_nonzero(widened.w2.weight[12:]) > 0


def test_exact_clone_moe_matches_dense_for_different_top3_selections():
    seed = _seed_ffn()
    config = E97MoEConfig(
        hidden_dim=20, routed_experts=8, top_k=3,
        expert_parallel_size=4, router_init_std=1e-3,
    )
    moe = SharedRoutedMoE.from_dense(seed, config)
    x = torch.randn(2, 7, 8)
    selections_a = torch.tensor([0, 1, 2]).expand(2, 7, 3)
    selections_b = torch.tensor([5, 3, 7]).expand(2, 7, 3)

    dense = seed(x)
    output_a = moe(x, forced_topk_indices=selections_a)
    output_b = moe(x, forced_topk_indices=selections_b)

    torch.testing.assert_close(output_a, dense, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(output_b, dense, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(output_a, output_b, rtol=2e-6, atol=2e-7)
    assert moe.router.weight.dtype == torch.float32
    assert moe.last_metrics["router_logits_dtype"] == torch.float32
    assert moe.last_metrics["dropped_tokens"] == 0
    assert int(moe.last_metrics["expert_token_counts"].sum()) == x.numel() // x.shape[-1] * 3
    assert tuple(moe.last_metrics["expert_co_selection"].shape) == (8, 8)
    assert moe.load_balance_loss.isfinite()
    assert moe.z_loss.isfinite()


def test_router_auxiliary_losses_are_differentiable():
    moe = SharedRoutedMoE.from_dense(
        _seed_ffn(),
        E97MoEConfig(hidden_dim=16, routed_experts=8, top_k=3,
                     expert_parallel_size=4),
    )
    moe(torch.randn(2, 4, 8))
    moe.auxiliary_loss.backward()
    assert moe.router.weight.grad is not None
    assert torch.isfinite(moe.router.weight.grad).all()


def test_gradient_checkpointed_ladder_returns_router_auxiliary_graph():
    torch.manual_seed(19)
    model = LadderLM(
        vocab_size=32, dim=8, depth=2, level="E97", expansion=1.0,
        n_state=4, n_heads=2, use_gate=True, gate_activation="sigmoid",
        linear_state=True, use_triton=False, mlp_ratio=2.0, mlp_multiple=4,
        gradient_checkpointing=True,
    )
    convert_e97_ffns_to_moe(
        model, E97MoEConfig(hidden_dim=16, routed_experts=8, top_k=3,
                            expert_parallel_size=4))
    model.gradient_checkpoint_group_size = 2
    model.train()
    loss = model(torch.randint(0, 32, (1, 9)), return_loss=True)
    auxiliary = e97_moe_auxiliary_loss(model)
    (loss + auxiliary).backward()
    assert model._checkpointed_moe_auxiliary_losses is not None
    assert len(model._checkpointed_moe_auxiliary_losses) == model.depth
    for layer in model.layers:
        assert layer.mlp.router.weight.grad is not None
        assert torch.isfinite(layer.mlp.router.weight.grad).all()


def test_checkpointed_auxiliary_outputs_remain_differentiable_authority():
    model = torch.nn.Linear(2, 2)
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)
    model._checkpointed_moe_auxiliary_losses = (first, second)
    auxiliary = e97_moe_auxiliary_loss(model)
    assert auxiliary.item() == 3.0
    auxiliary.backward()
    assert first.grad.item() == 1.0
    assert second.grad.item() == 1.0


def test_forced_routing_rejects_duplicate_or_out_of_range_experts():
    moe = SharedRoutedMoE.from_dense(
        _seed_ffn(),
        E97MoEConfig(hidden_dim=12, routed_experts=8, top_k=3,
                     expert_parallel_size=4),
    )
    x = torch.randn(1, 1, 8)
    with pytest.raises(ValueError, match="distinct"):
        moe(x, forced_topk_indices=torch.tensor([[[1, 1, 2]]]))
    with pytest.raises(ValueError, match="invalid"):
        moe(x, forced_topk_indices=torch.tensor([[[0, 1, 8]]]))


def test_frontier_expert_ownership_is_contiguous_and_complete():
    expected = []
    for rank in range(8):
        owned = experts_for_rank(rank)
        assert owned == tuple(range(rank * 8, rank * 8 + 8))
        assert all(expert_owner(expert) == rank for expert in owned)
        expected.extend(owned)
    assert expected == list(range(64))


def test_tiny_full_e97_conversion_preserves_loss_logits_layers_and_recurrent_states():
    torch.manual_seed(31)
    dense = LadderLM(
        vocab_size=32, dim=8, depth=2, level="E97", expansion=1.0,
        n_state=4, n_heads=2, use_gate=True, gate_activation="sigmoid",
        linear_state=True, use_triton=False, mlp_ratio=2.0, mlp_multiple=4,
    ).eval()
    moe = copy.deepcopy(dense)
    convert_e97_ffns_to_moe(
        moe,
        E97MoEConfig(hidden_dim=20, routed_experts=8, top_k=3,
                     expert_parallel_size=4),
    )
    tokens = torch.randint(0, 32, (2, 6))
    dense_layers, moe_layers, dense_ffns, moe_ffns = [], [], [], []
    handles = []
    for layer in dense.layers:
        handles.append(layer.register_forward_hook(
            lambda _m, _a, out: dense_layers.append(out[0].detach().clone())))
        handles.append(layer.mlp.register_forward_hook(
            lambda _m, _a, out: dense_ffns.append(out.detach().clone())))
    for layer in moe.layers:
        handles.append(layer.register_forward_hook(
            lambda _m, _a, out: moe_layers.append(out[0].detach().clone())))
        handles.append(layer.mlp.register_forward_hook(
            lambda _m, _a, out: moe_ffns.append(out.detach().clone())))
    try:
        dense_logits, (dense_states, _) = dense(tokens, return_prev_hiddens=True)
        moe_logits, (moe_states, _) = moe(tokens, return_prev_hiddens=True)
    finally:
        for handle in handles:
            handle.remove()

    dense_loss = dense(tokens, return_loss=True)
    moe_loss = moe(tokens, return_loss=True)
    torch.testing.assert_close(moe_loss, dense_loss, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(moe_logits, dense_logits, rtol=3e-6, atol=3e-7)
    assert len(dense_layers) == len(moe_layers) == 2
    assert len(dense_ffns) == len(moe_ffns) == 2
    for actual, expected in zip(moe_layers, dense_layers):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    for actual, expected in zip(moe_ffns, dense_ffns):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)
    for actual, expected in zip(moe_states, dense_states):
        torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)


def test_packed_node_local_conversion_materializes_only_eight_experts_per_rank():
    seed = _seed_ffn()
    config = E97MoEConfig(hidden_dim=20, routed_experts=64, top_k=3,
                          expert_parallel_size=8)
    shard = NodeLocalSharedRoutedMoE.from_dense(
        seed, config, local_expert_rank=5)
    assert shard.local_gate_weight.shape == (8, 20, 8)
    assert shard.local_up_weight.shape == (8, 20, 8)
    assert shard.local_down_weight.shape == (8, 8, 20)
    assert shard.router.weight.shape == (64, 8)
    assert shard.local_expert_rank == 5
    x = torch.randn(7, 8)
    shared = shard.shared_expert(x)
    gate = torch.nn.functional.silu(x @ shard.local_gate_weight[0].T)
    up = x @ shard.local_up_weight[0].T
    routed_clone = (gate * up) @ shard.local_down_weight[0].T
    torch.testing.assert_close(shared + routed_clone, seed(x), rtol=3e-6, atol=3e-7)

    model = LadderLM(
        vocab_size=32, dim=8, depth=2, level="E97", expansion=1.0,
        n_state=4, n_heads=2, use_gate=True, gate_activation="sigmoid",
        linear_state=True, use_triton=False, mlp_ratio=2.0, mlp_multiple=4)
    protected = {name: parameter for name, parameter in model.named_parameters()
                 if ".mlp." not in name}
    convert_e97_ffns_to_node_local_moe(
        model, E97MoEConfig(hidden_dim=20), local_expert_rank=2)
    assert all(isinstance(layer.mlp, NodeLocalSharedRoutedMoE) for layer in model.layers)
    for name, parameter in protected.items():
        assert dict(model.named_parameters())[name] is parameter


def test_recipe_uses_instantiated_graph_and_conversion_touches_only_ffns():
    torch.manual_seed(2)
    model = LadderLM(
        vocab_size=32, dim=8, depth=2, level="E97", expansion=1.0,
        n_state=4, n_heads=2, use_gate=True, gate_activation="sigmoid",
        linear_state=True, use_triton=False,
        mlp_ratio=2.0, mlp_multiple=4,
    )
    assert all(isinstance(layer, MixerMLPWrapper) for layer in model.layers)
    protected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".mlp." not in name
    }
    recipe = calculate_e97_moe_recipe(
        model, target_parameters=200_000, routed_experts=8,
        shared_experts=1, top_k=3, multiple=4,
    )
    assert recipe.layers == 2
    assert recipe.model_width == 8
    assert recipe.seed_expert_hidden == 16
    assert recipe.expert_hidden % 4 == 0

    config = E97MoEConfig(
        hidden_dim=recipe.expert_hidden, routed_experts=8,
        shared_experts=1, top_k=3, expert_parallel_size=4,
    )
    convert_e97_ffns_to_moe(model, config)
    assert sum(parameter.numel() for parameter in model.parameters()) == recipe.total_parameters
    for name, parameter in protected.items():
        assert dict(model.named_parameters())[name] is parameter
    assert all(isinstance(layer.mlp, SharedRoutedMoE) for layer in model.layers)
