#!/usr/bin/env python3
"""Unit tests for pure async quorum DiLoCo x/z state and delta math."""

import json
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from ndm.async_diloco import (  # noqa: E402
    AsyncDilocoState,
    apply_dense_delta,
    compute_dense_delta,
    extract_schedulefree_xz,
    install_schedulefree_xz_state,
    quorum_merge,
    rebase_local_state,
)


def _state(generation, x_rows, z_rows=None):
    if z_rows is None:
        z_rows = x_rows
    return AsyncDilocoState(
        generation=generation,
        x=tuple(torch.tensor(row, dtype=torch.float32) for row in x_rows),
        z=tuple(torch.tensor(row, dtype=torch.float32) for row in z_rows),
    )


def _worker_delta(base, worker_id, x_rows, z_rows, *, tokens=None, local_steps=None):
    worker = _state(base.generation, x_rows, z_rows)
    return compute_dense_delta(
        base,
        worker,
        worker_id=worker_id,
        tokens=tokens,
        local_steps=local_steps,
    )


def _assert_state_close(actual, expected_x, expected_z, tol=1e-6):
    for got, exp in zip(actual.x, expected_x):
        torch.testing.assert_close(got, exp, atol=tol, rtol=0.0)
    for got, exp in zip(actual.z, expected_z):
        torch.testing.assert_close(got, exp, atol=tol, rtol=0.0)


def test_full_cohort_zero_staleness_eta_one_matches_synchronous_average():
    base = _state(
        7,
        x_rows=[[[1.0, 2.0], [3.0, 4.0]], [10.0, -2.0]],
        z_rows=[[[0.5, 1.5], [2.5, 3.5]], [9.0, -3.0]],
    )
    worker_x = [
        [base.x[0] + 1.0, base.x[1] - 2.0],
        [base.x[0] - 3.0, base.x[1] + 4.0],
        [base.x[0] + 2.0, base.x[1] + 8.0],
    ]
    worker_z = [
        [base.z[0] + 0.5, base.z[1] + 1.0],
        [base.z[0] - 1.5, base.z[1] + 5.0],
        [base.z[0] + 4.0, base.z[1] - 3.0],
    ]
    updates = [
        compute_dense_delta(
            base,
            AsyncDilocoState(base.generation, x=tuple(xs), z=tuple(zs)),
            worker_id=f"w{i}",
            tokens=100,
        )
        for i, (xs, zs) in enumerate(zip(worker_x, worker_z))
    ]

    merged = quorum_merge(base, updates, quorum=3, eta_outer=1.0, weight_mode="equal")

    expected_x = [sum(xs[i] for xs in worker_x) / len(worker_x) for i in range(len(base.x))]
    expected_z = [sum(zs[i] for zs in worker_z) / len(worker_z) for i in range(len(base.z))]
    assert merged.advanced
    assert merged.metrics.accepted_count == 3
    assert merged.metrics.rejected_count == 0
    _assert_state_close(merged.state, expected_x, expected_z)


def test_partial_equal_weight_quorum_averages_only_accepted_workers():
    base = _state(2, x_rows=[[0.0, 10.0]], z_rows=[[1.0, 11.0]])
    updates = [
        _worker_delta(base, "w0", [[2.0, 12.0]], [[3.0, 13.0]]),
        _worker_delta(base, "w1", [[6.0, 16.0]], [[9.0, 19.0]]),
    ]

    merged = quorum_merge(
        base,
        updates,
        quorum=2,
        eta_outer=1.0,
        weight_mode="equal",
        expected_worker_ids={"w0", "w1", "w2"},
    )

    _assert_state_close(
        merged.state,
        expected_x=[torch.tensor([4.0, 14.0])],
        expected_z=[torch.tensor([6.0, 16.0])],
    )
    assert merged.metrics.configured_quorum == 2
    assert merged.metrics.effective_quorum == 2
    assert merged.metrics.missing_worker_ids == ("w2",)


def test_token_weighted_quorum_uses_token_counts():
    base = _state(0, x_rows=[[10.0, 20.0]], z_rows=[[30.0, 40.0]])
    updates = [
        _worker_delta(base, "small", [[20.0, 20.0]], [[30.0, 50.0]], tokens=10),
        _worker_delta(base, "large", [[50.0, 60.0]], [[90.0, 100.0]], tokens=30),
    ]

    merged = quorum_merge(base, updates, quorum=2, eta_outer=1.0, weight_mode="tokens")

    expected_x = (updates[0].tokens * (base.x[0] + updates[0].dx[0])
                  + updates[1].tokens * (base.x[0] + updates[1].dx[0])) / 40.0
    expected_z = (updates[0].tokens * (base.z[0] + updates[0].dz[0])
                  + updates[1].tokens * (base.z[0] + updates[1].dz[0])) / 40.0
    _assert_state_close(merged.state, [expected_x], [expected_z])


def test_stale_update_rejection_keeps_only_current_generation():
    current = _state(4, x_rows=[[0.0, 0.0]], z_rows=[[1.0, 1.0]])
    stale_base = current.clone(generation=3)
    stale = _worker_delta(stale_base, "late", [[100.0, 100.0]], [[100.0, 100.0]], tokens=100)
    fresh = _worker_delta(current, "fresh", [[2.0, 4.0]], [[3.0, 5.0]], tokens=1)

    merged = quorum_merge(current, [stale, fresh], quorum=1, eta_outer=1.0, weight_mode="tokens")

    assert merged.advanced
    assert [u.worker_id for u in merged.accepted] == ["fresh"]
    assert len(merged.rejected) == 1
    assert merged.rejected[0].worker_id == "late"
    assert merged.rejected[0].reason == "stale_generation"
    assert merged.metrics.accepted_count == 1
    assert merged.metrics.rejected_count == 1
    _assert_state_close(merged.state, [torch.tensor([2.0, 4.0])], [torch.tensor([3.0, 5.0])])


def test_xz_rebase_preserves_local_displacement_geometry():
    old_base = _state(5, x_rows=[[10.0, 20.0]], z_rows=[[30.0, 40.0]])
    local = _state(5, x_rows=[[13.0, 15.0]], z_rows=[[37.0, 35.0]])
    new_base = _state(6, x_rows=[[100.0, 80.0]], z_rows=[[10.0, 70.0]])

    rebased = rebase_local_state(local, old_base, new_base)

    torch.testing.assert_close(rebased.x[0] - new_base.x[0], local.x[0] - old_base.x[0])
    torch.testing.assert_close(rebased.z[0] - new_base.z[0], local.z[0] - old_base.z[0])
    assert rebased.generation == new_base.generation


def test_missing_worker_does_not_block_advancement_once_quorum_is_satisfied(tmp_path):
    base = _state(11, x_rows=[[1.0, 1.0]], z_rows=[[2.0, 2.0]])
    updates = [
        _worker_delta(base, "node0", [[3.0, 1.0]], [[4.0, 2.0]], local_steps=10),
        _worker_delta(base, "node2", [[1.0, 5.0]], [[2.0, 8.0]], local_steps=10),
    ]
    metrics_path = tmp_path / "async_diloco_core_metrics.json"

    merged = quorum_merge(
        base,
        updates,
        quorum=2,
        eta_outer=1.0,
        weight_mode="equal",
        expected_worker_ids={"node0", "node1", "node2"},
        metrics_path=metrics_path,
    )

    assert merged.advanced
    assert merged.state.generation == 12
    _assert_state_close(merged.state, [torch.tensor([2.0, 3.0])], [torch.tensor([3.0, 5.0])])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["configured_quorum"] == 2
    assert metrics["effective_quorum"] == 2
    assert metrics["accepted_count"] == 2
    assert metrics["rejected_count"] == 0
    assert metrics["merge_time_s"] >= 0.0
    assert metrics["missing_worker_ids"] == ["node1"]


def test_apply_dense_delta_respects_eta_outer():
    base = _state(0, x_rows=[[1.0, 2.0]], z_rows=[[3.0, 4.0]])
    worker = _state(0, x_rows=[[5.0, 10.0]], z_rows=[[7.0, 12.0]])
    delta = compute_dense_delta(base, worker, worker_id="w0")

    applied = apply_dense_delta(base, delta, eta_outer=0.25)

    _assert_state_close(applied, [torch.tensor([2.0, 4.0])], [torch.tensor([4.0, 6.0])])


def test_schedulefree_extract_and_install_hooks_roundtrip():
    schedulefree = pytest.importorskip("schedulefree")
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2, warmup_steps=0)
    opt.train()
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)
    ((model(x) - y) ** 2).mean().backward()
    opt.step()
    before_train_weight = model.weight.detach().clone()

    state = extract_schedulefree_xz(model, opt, generation=9)

    assert opt.param_groups[0]["train_mode"] is True
    torch.testing.assert_close(model.weight, before_train_weight)
    assert state.generation == 9
    assert len(state.x) == 1
    assert len(state.z) == 1
    assert state.x[0].shape == model.weight.shape
    assert state.z[0].shape == model.weight.shape

    shifted = AsyncDilocoState(
        generation=10,
        x=(state.x[0] + 1.25,),
        z=(state.z[0] - 0.75,),
    )
    install_schedulefree_xz_state(model, opt, shifted)
    reloaded = extract_schedulefree_xz(model, opt, generation=10)

    torch.testing.assert_close(reloaded.x[0], shifted.x[0])
    torch.testing.assert_close(reloaded.z[0], shifted.z[0])
