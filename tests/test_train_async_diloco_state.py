from types import SimpleNamespace

import pytest
import torch

import train


def _args(optimizer):
    return SimpleNamespace(optimizer=optimizer)


def _build_linear():
    torch.manual_seed(11)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    )


def _build_schedulefree_scalar():
    schedulefree = pytest.importorskip("schedulefree")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=1e-2, warmup_steps=0)
    optimizer.train()
    x = torch.ones(4, 1)
    optimizer.zero_grad()
    (model(x) ** 2).mean().backward()
    optimizer.step()
    return model, optimizer


def _set_schedulefree_xz(model, optimizer, *, x, z):
    optimizer.eval()
    p = next(model.parameters())
    with torch.no_grad():
        p.fill_(float(x))
        optimizer.state[p]["z"].fill_(float(z))
    optimizer.train()


def _only_param(state, basis="x"):
    return next(iter(state[basis].values()))


def test_async_diloco_dense_avg_matches_direct_weight_average():
    base_model = _build_linear()
    base = train.extract_diloco_tensor_state(base_model, None, _args("adamw"))
    locals_ = []
    direct = {}

    for offset in (1.25, -0.5, 3.0):
        model = _build_linear()
        train.apply_diloco_tensor_state(model, None, _args("adamw"), base)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(offset)
        local_state = train.extract_diloco_tensor_state(model, None, _args("adamw"))
        locals_.append(local_state)
        for name, tensor in local_state["params"].items():
            direct.setdefault(name, torch.zeros_like(tensor)).add_(tensor / 3.0)

    deltas = [train.compute_diloco_dense_delta(base, state) for state in locals_]
    merged = train.merge_diloco_dense_deltas(base, deltas, weights=[1, 1, 1])

    assert set(merged["params"]) == set(direct)
    for name, expected in direct.items():
        assert torch.allclose(merged["params"][name], expected)


def test_schedulefree_apply_translates_x_and_z_without_mode_change():
    model, optimizer = _build_schedulefree_scalar()
    _set_schedulefree_xz(model, optimizer, x=10.0, z=4.0)
    assert optimizer.param_groups[0]["train_mode"] is True
    before = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))
    before_gap = _only_param(before, "z") - _only_param(before, "x")

    shifted = {
        "kind": "schedulefree",
        "x": {name: tensor + 7.0 for name, tensor in before["x"].items()},
        "z": {name: tensor + 7.0 for name, tensor in before["z"].items()},
    }
    train.apply_diloco_tensor_state(model, optimizer, _args("schedulefree"), shifted)

    assert optimizer.param_groups[0]["train_mode"] is True
    after = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))
    after_gap = _only_param(after, "z") - _only_param(after, "x")
    assert torch.allclose(after_gap, before_gap)
    assert torch.allclose(_only_param(after, "x"), _only_param(before, "x") + 7.0)
    assert torch.allclose(_only_param(after, "z"), _only_param(before, "z") + 7.0)


def test_schedulefree_stale_base_rebase_preserves_local_displacement():
    model, optimizer = _build_schedulefree_scalar()
    _set_schedulefree_xz(model, optimizer, x=10.0, z=4.0)
    old_base = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))

    _set_schedulefree_xz(model, optimizer, x=13.0, z=8.0)
    local = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))

    new_base = {
        "kind": "schedulefree",
        "x": {name: tensor + 10.0 for name, tensor in old_base["x"].items()},
        "z": {name: tensor + 11.0 for name, tensor in old_base["z"].items()},
    }
    rebased = train.rebase_diloco_tensor_state(local, old_base, new_base)

    assert torch.allclose(_only_param(rebased, "x"), _only_param(new_base, "x") + 3.0)
    assert torch.allclose(_only_param(rebased, "z"), _only_param(new_base, "z") + 4.0)


def test_delta_roundtrip_restores_schedulefree_z_and_preserves_eval_mode():
    model, optimizer = _build_schedulefree_scalar()
    _set_schedulefree_xz(model, optimizer, x=2.0, z=5.0)
    optimizer.eval()
    assert optimizer.param_groups[0]["train_mode"] is False
    base = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))

    _set_schedulefree_xz(model, optimizer, x=4.5, z=8.5)
    optimizer.eval()
    local = train.extract_diloco_tensor_state(model, optimizer, _args("schedulefree"))
    delta = train.compute_diloco_dense_delta(base, local)
    roundtrip = train.apply_diloco_dense_delta(base, delta)

    target_model, target_optimizer = _build_schedulefree_scalar()
    train.apply_diloco_tensor_state(
        target_model, target_optimizer, _args("schedulefree"), base)
    target_optimizer.eval()
    train.apply_diloco_tensor_state(
        target_model, target_optimizer, _args("schedulefree"), roundtrip)

    assert target_optimizer.param_groups[0]["train_mode"] is False
    restored = train.extract_diloco_tensor_state(
        target_model, target_optimizer, _args("schedulefree"))
    assert torch.allclose(_only_param(restored, "x"), _only_param(local, "x"))
    assert torch.allclose(_only_param(restored, "z"), _only_param(local, "z"))
