#!/usr/bin/env python3
"""implement-diloco-periodic: REAL correctness test for the DiLoCo periodic merge.

Spawns 2 gloo CPU processes, each holding a real nn model + a REAL ScheduleFree
AdamW optimizer, feeds each rank DIFFERENT gradients so their weights genuinely
diverge over K local steps, then calls the ACTUAL train.diloco_merge() and
asserts the y-mode-swap + full ScheduleFree-state averaging semantics:

  1. CONSENSUS: after the merge, every rank's parameters are byte-identical
     (max cross-rank diff ~0) -> no inter-round drift.
  2. AVERAGING: the merged eval (x) weights equal the mean of the per-rank
     pre-merge eval weights and z equals the mean pre-merge z.
  3. SCHEDULEFREE INVARIANT: post-merge, weight_sum/k/lr_max are preserved, and
     train-mode y is rebuilt from the merged x/z pair. The SF averaging
     denominator is never reset.
  4. OUTER MOMENTUM: with outer_beta>0 / outer_lr!=1 the general DiLoCo update
     W_{r+1} = W_r + outer_lr*(beta*mom + (mean_x - W_r)) is applied exactly.

No GPUs, no fabrication: real optimizer, real distributed collectives.
"""
import os, sys, tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _ns(**kw):
    from types import SimpleNamespace
    base = dict(
        optimizer='schedulefree',
        diloco_outer_lr=1.0,
        diloco_outer_beta=0.0,
        diloco_outer_optimizer='avg',
        diloco_export_basis='x',
        diloco_merge_bucket_numel=0,
        diloco_merge_debug=0,
        diloco_merge_debug_ranks='0',
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_diloco_fault_injection_exits_only_exact_collective(monkeypatch, capsys):
    import train

    monkeypatch.setenv('EMENDER_DILOCO_EXIT_RANK', '3')
    monkeypatch.setenv('EMENDER_DILOCO_EXIT_MERGE', '2')
    monkeypatch.setenv('EMENDER_DILOCO_EXIT_BUCKET', '1')
    monkeypatch.setenv('EMENDER_DILOCO_EXIT_LABEL', 'model')
    monkeypatch.setenv('EMENDER_DILOCO_EXIT_CODE', '86')
    monkeypatch.setenv('EMENDER_DILOCO_EXIT_DELAY_SECONDS', '0')
    monkeypatch.setattr(train.dist, 'get_rank', lambda: 3)
    exits = []
    monkeypatch.setattr(train.os, '_exit', lambda code: exits.append(code))

    train._maybe_inject_diloco_collective_rank_exit(
        label='model', bucket_index=0, merge_index=2)
    assert exits == []
    train._maybe_inject_diloco_collective_rank_exit(
        label='model', bucket_index=1, merge_index=2)
    assert exits == [86]
    assert 'DILOCO_FAULT_INJECTION' in capsys.readouterr().out


def _build():
    """Tiny but real model + real ScheduleFree optimizer (seed-identical across ranks)."""
    import schedulefree
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16),
    )
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2, warmup_steps=0)
    return model, opt


def _local_train(model, opt, rank, k_steps):
    """Run k real fwd/bwd/step iterations with rank-specific data (weights diverge)."""
    opt.train()
    g = torch.Generator().manual_seed(100 + rank)  # distinct data per rank
    for _ in range(k_steps):
        x = torch.randn(8, 16, generator=g)
        y = torch.randn(8, 16, generator=g) + rank  # rank-dependent target
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def _loss(model, x, y):
    with torch.no_grad():
        return ((model(x) - y) ** 2).mean().item()


def _local_train_identical(model, opt, k_steps, seed=42):
    """Run k real fwd/bwd/step iterations with IDENTICAL data on every rank, so
    all ranks end with byte-identical model + SF state (used by the no-op and
    basis-consistency tests where cross-rank divergence must be exactly zero)."""
    opt.train()
    g = torch.Generator().manual_seed(seed)  # SAME seed on all ranks
    for _ in range(k_steps):
        x = torch.randn(8, 16, generator=g)
        y = torch.randn(8, 16, generator=g)
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()


def _build_scalar():
    """A genuine 1-parameter model + real ScheduleFree optimizer (for the scalar
    translation-invariance test)."""
    import schedulefree
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1, bias=False)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=1e-2, warmup_steps=0)
    return model, opt


def _set_scalar_state(model, opt, y_val, z_val):
    """Drive ONE real SF step to instantiate the real 'z' state entry, then
    overwrite to a controlled (train-mode y, base z) pair. The SF state machine
    itself is genuine -- only the values are set, so eval()/train() conversions
    (x = y/beta1 + (1-1/beta1)*z) are the library's real arithmetic."""
    opt.train()
    g = torch.Generator().manual_seed(7)
    xb = torch.randn(4, 1, generator=g)
    opt.zero_grad()
    (model(xb) ** 2).mean().backward()
    opt.step()
    p = model.weight
    with torch.no_grad():
        p.data.fill_(y_val)               # train-mode y
        opt.state[p]['z'].fill_(z_val)    # base iterate z
    return p


def _worker(rank, world_size, init_file, outer_lr, outer_beta, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=outer_lr, diloco_outer_beta=outer_beta)
    outer_state = None
    if not (outer_lr == 1.0 and outer_beta == 0.0):
        outer_state = {
            'anchor': [p.data.detach().clone() for p in model.parameters()],
            'moment': [torch.zeros_like(p.data) for p in model.parameters()],
        }

    _local_train(model, opt, rank, k_steps=15)

    # Capture each rank's pre-merge EVAL (x) and base (z) weights to validate
    # full-state averaging.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    pre_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]
    group_pre = {
        'k': opt.param_groups[0]['k'],
        'weight_sum': opt.param_groups[0]['weight_sum'],
        'lr_max': opt.param_groups[0]['lr_max'],
    }

    # Gather all ranks' pre-merge x/z to rank 0 (to compute expected means).
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        'x': [t.tolist() for t in pre_x],
        'z': [t.tolist() for t in pre_z],
        'group': group_pre,
    })

    # Capture the anchor (W_r) BEFORE the merge advances it to W_{r+1}.
    anchor_pre = None
    if outer_state is not None:
        anchor_pre = [t.detach().clone() for t in outer_state['anchor']]

    # --- THE FUNCTION UNDER TEST ---
    train.diloco_merge(model, opt, args, world_size, outer_state)

    # Post-merge: train-mode y weights, eval-mode x weights, and z state.
    y_now = [p.data.detach().clone() for p in model.parameters()]
    opt.eval()
    x_now = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    z_now = [opt.state[p]['z'].detach().clone() for p in model.parameters()]

    if rank == 0:
        # Expected means of pre-merge x and z across ranks.
        mean_x = []
        mean_z = []
        for j in range(len(pre_x)):
            acc = torch.zeros_like(pre_x[j])
            z_acc = torch.zeros_like(pre_z[j])
            for r in range(world_size):
                acc += torch.tensor(gathered[r]['x'][j])
                z_acc += torch.tensor(gathered[r]['z'][j])
            mean_x.append(acc / world_size)
            mean_z.append(z_acc / world_size)
        ret['mean_x'] = [t.tolist() for t in mean_x]
        ret['mean_z'] = [t.tolist() for t in mean_z]
        ret['x_now'] = [t.tolist() for t in x_now]
        ret['y_now'] = [t.tolist() for t in y_now]
        ret['z_now'] = [t.tolist() for t in z_now]
        ret['group_pre_merge'] = gathered[0]['group']
        if anchor_pre is not None:
            ret['anchor'] = [t.tolist() for t in anchor_pre]
            ret['last_metrics'] = dict(outer_state.get('last_metrics', {}))
        ret['group_after_merge'] = {
            'k': opt.param_groups[0]['k'],
            'weight_sum': opt.param_groups[0]['weight_sum'],
            'lr_max': opt.param_groups[0]['lr_max'],
        }

    # Consensus check: all ranks must hold identical params after merge.
    flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    allp = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(allp, flat)
    max_diff = max((allp[r] - allp[0]).abs().max().item() for r in range(world_size))
    if rank == 0:
        ret['max_cross_rank_diff'] = max_diff

    dist.barrier()
    dist.destroy_process_group()


def _run(outer_lr=1.0, outer_beta=0.0):
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker, args=(world_size, init_file, outer_lr, outer_beta, ret),
             nprocs=world_size, join=True)
    return dict(ret)


def _worker_dynamic(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)

    # Real local training with rank-specific data creates different SF x/z
    # histories and a nonzero pre-merge averaging denominator.
    _local_train(model, opt, rank, k_steps=25)
    pre_weight_sum = opt.param_groups[0]['weight_sum']
    pre_k = opt.param_groups[0]['k']

    g = torch.Generator().manual_seed(999)
    probe_x = torch.randn(16, 16, generator=g)
    probe_y = torch.randn(16, 16, generator=g)
    pre_loss = _loss(model, probe_x, probe_y)

    train.diloco_merge(model, opt, args, world_size, None)
    merge_weight_sum = opt.param_groups[0]['weight_sum']
    merge_k = opt.param_groups[0]['k']
    merged_loss = _loss(model, probe_x, probe_y)
    opt.eval()
    merged_xz_diff = max(
        (p.data - opt.state[p]['z']).abs().max().item() for p in model.parameters()
    )
    opt.train()

    # First post-merge step. The old reset band-aid made the next ckp1 equal 1,
    # collapsing eval x onto the recent z. The principled merge preserves
    # weight_sum, so the next update keeps x as a long-run average.
    step_x = torch.randn(16, 16, generator=g)
    step_y = torch.randn(16, 16, generator=g)
    opt.zero_grad()
    step_loss = ((model(step_x) - step_y) ** 2).mean()
    step_loss.backward()
    opt.step()
    post_loss = _loss(model, probe_x, probe_y)

    opt.eval()
    post_xz_diff = 0.0
    for p in model.parameters():
        post_xz_diff = max(post_xz_diff, (p.data - opt.state[p]['z']).abs().max().item())
    opt.train()

    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        'pre_weight_sum': pre_weight_sum,
        'merge_weight_sum': merge_weight_sum,
        'post_weight_sum': opt.param_groups[0]['weight_sum'],
        'pre_k': pre_k,
        'merge_k': merge_k,
        'pre_loss': pre_loss,
        'merged_loss': merged_loss,
        'post_loss': post_loss,
        'merged_xz_diff': merged_xz_diff,
        'post_xz_diff': post_xz_diff,
    })
    if rank == 0:
        ret['dynamic'] = gathered

    dist.barrier()
    dist.destroy_process_group()


def _run_dynamic():
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker_dynamic, args=(world_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)['dynamic']


def _worker_full_average(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)

    rows = []
    for round_idx in range(4):
        _local_train(model, opt, rank, k_steps=10)
        pre_weight_sum = opt.param_groups[0]['weight_sum']
        train.diloco_merge(model, opt, args, world_size, None)
        opt.eval()
        xz_diff = max(
            (p.data - opt.state[p]['z']).abs().max().item() for p in model.parameters()
        )
        x_norm = sum(p.data.float().norm().item() for p in model.parameters())
        z_norm = sum(opt.state[p]['z'].float().norm().item() for p in model.parameters())
        opt.train()
        rows.append({
            'round': round_idx,
            'pre_weight_sum': pre_weight_sum,
            'merge_weight_sum': opt.param_groups[0]['weight_sum'],
            'k': opt.param_groups[0]['k'],
            'xz_diff': xz_diff,
            'x_norm': x_norm,
            'z_norm': z_norm,
        })

    gathered = [None] * world_size
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        ret['full_average'] = gathered

    dist.barrier()
    dist.destroy_process_group()


def _run_full_average():
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker_full_average, args=(world_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)['full_average']


def _close(a, b, tol):
    ta, tb = torch.tensor(a), torch.tensor(b)
    return (ta - tb).abs().max().item() <= tol


def test_local_sgd_averaging():
    """outer_lr=1, outer_beta=0 -> plain averaging + schedulefree consensus."""
    r = _run(outer_lr=1.0, outer_beta=0.0)
    tol = 1e-5
    # 1. consensus across ranks
    assert r['max_cross_rank_diff'] <= tol, f"ranks diverge: {r['max_cross_rank_diff']}"
    # 2. merged eval weights == mean of pre-merge eval weights
    for xn, mx in zip(r['x_now'], r['mean_x']):
        assert _close(xn, mx, tol), "merged x != mean of pre-merge x"
    # 3a. merged z == mean pre-merge z, not a reset to x/y.
    for zn, mz in zip(r['z_now'], r['mean_z']):
        assert _close(zn, mz, tol), "merged z != mean of pre-merge z"
    # 3b. train-mode y is rebuilt from merged x/z.
    beta1 = 0.9
    for yn, xn, zn in zip(r['y_now'], r['x_now'], r['z_now']):
        ty, tx, tz = torch.tensor(yn), torch.tensor(xn), torch.tensor(zn)
        expected_y = tx * beta1 + tz * (1.0 - beta1)
        assert (ty - expected_y).abs().max().item() <= tol, "post-merge y != SF train interpolation"
    assert r['group_after_merge']['weight_sum'] == r['group_pre_merge']['weight_sum'], (
        "SF weight_sum should be preserved across merge"
    )
    assert r['group_after_merge']['weight_sum'] > 0.0, "SF weight_sum was reset"
    assert r['group_after_merge']['k'] == 15, "Adam/SF step clock should be preserved"
    assert r['group_after_merge']['k'] == r['group_pre_merge']['k'], "SF k changed at merge"
    assert r['group_after_merge']['lr_max'] == r['group_pre_merge']['lr_max'], "SF lr_max changed at merge"
    print("PASS test_local_sgd_averaging: consensus + full SF-state averaging verified")


def test_outer_momentum():
    """outer_beta=0.9, outer_lr=0.7 -> general DiLoCo outer step, exact + consensus."""
    lr, beta = 0.7, 0.9
    r = _run(outer_lr=lr, outer_beta=beta)
    tol = 1e-5
    assert r['max_cross_rank_diff'] <= tol, f"ranks diverge: {r['max_cross_rank_diff']}"
    # First round: mom = beta*0 + (mean_x - W_0) = (mean_x - anchor);
    # W_1 = anchor + lr*mom. The post-merge eval x must equal W_1.
    for xn, mx, anc in zip(r['x_now'], r['mean_x'], r['anchor']):
        tx, tm, ta = torch.tensor(xn), torch.tensor(mx), torch.tensor(anc)
        expected = ta + lr * (tm - ta)
        d = (tx - expected).abs().max().item()
        assert d <= tol, f"outer-momentum update mismatch (max diff {d})"
    print("PASS test_outer_momentum: general DiLoCo outer step verified")


def test_outer_momentum_geometry_metrics_reported():
    """P2 instrumentation: nontrivial outer updates must expose per-merge
    whole-model geometry ratios for log/report analysis."""
    lr, beta = 0.5, 0.5
    r = _run(outer_lr=lr, outer_beta=beta)
    metrics = r.get('last_metrics')
    assert metrics is not None, "outer_state['last_metrics'] was not populated"
    for key in ('land_frac', 'disp_mag', 'gap_health'):
        assert key in metrics, f"missing DiLoCo geometry metric {key}"
        assert torch.isfinite(torch.tensor(metrics[key])), f"non-finite {key}: {metrics[key]}"
        assert metrics[key] >= 0.0, f"negative {key}: {metrics[key]}"
    assert abs(metrics['land_frac'] - lr) <= 5e-4, (
        f"first-round land_frac should equal outer_lr when momentum starts at zero; "
        f"got {metrics['land_frac']} vs {lr}"
    )
    assert metrics['disp_mag'] > 0.0, "rank-divergent training should create nonzero displacement"
    assert metrics['gap_health'] > 0.0, "ScheduleFree x-z gap health should be measurable"
    print("PASS test_outer_momentum_geometry_metrics_reported: DiLoCo geometry ratios populated")


def test_schedulefree_dynamic_loss_continuity():
    """Real train -> merge -> train regression for ScheduleFree-DiLoCo dynamics."""
    rows = _run_dynamic()
    for i, row in enumerate(rows):
        assert row['pre_weight_sum'] > 0.0, f"rank {i} never built SF average"
        assert row['merge_weight_sum'] == row['pre_weight_sum'], (
            f"rank {i} merge did not preserve SF averaging denominator"
        )
        assert row['post_weight_sum'] > row['merge_weight_sum'], (
            f"rank {i} SF averaging mass was not monotonic after merge"
        )
        assert row['merge_k'] == row['pre_k'], f"rank {i} merge changed SF k"
        assert torch.isfinite(torch.tensor(row['pre_loss']))
        assert torch.isfinite(torch.tensor(row['merged_loss']))
        assert torch.isfinite(torch.tensor(row['post_loss']))
        baseline = max(row['pre_loss'], row['merged_loss'], 1e-12)
        assert row['post_loss'] <= baseline * 1.5 + 1e-5, (
            f"rank {i} post-merge loss spike: pre={row['pre_loss']:.6g} "
            f"merged={row['merged_loss']:.6g} post={row['post_loss']:.6g}"
        )
        assert row['merged_xz_diff'] > 1e-8, (
            f"rank {i} merge collapsed full-history x onto recent z"
        )
        assert row['post_xz_diff'] > 1e-8, (
            f"rank {i} first post-merge step re-anchored x to z"
        )
    print("PASS test_schedulefree_dynamic_loss_continuity: no post-merge spike")


def test_schedulefree_full_trajectory_average_survives_merges():
    """Multiple real-rank merges keep x as a long-run average instead of z reset."""
    rank_rows = _run_full_average()
    for rank, rows in enumerate(rank_rows):
        last_weight_sum = 0.0
        for row in rows:
            assert row['merge_weight_sum'] == row['pre_weight_sum'], (
                f"rank {rank} round {row['round']} reset weight_sum"
            )
            assert row['merge_weight_sum'] > last_weight_sum, (
                f"rank {rank} round {row['round']} weight_sum not monotonic"
            )
            assert row['xz_diff'] > 1e-8, (
                f"rank {rank} round {row['round']} x collapsed to z after merge"
            )
            assert torch.isfinite(torch.tensor(row['x_norm']))
            assert torch.isfinite(torch.tensor(row['z_norm']))
            last_weight_sum = row['merge_weight_sum']
    print("PASS test_schedulefree_full_trajectory_average_survives_merges")


def test_single_rank_diloco_merge_is_byte_identical_noop():
    """world_size=1 returns before SF swaps or state mutation."""
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=3)
    before_params = [p.detach().clone() for p in model.parameters()]
    before_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]
    before_group = dict(opt.param_groups[0])

    sync_s = train.diloco_merge(model, opt, _ns(), world_size=1, outer_state=None)

    assert sync_s == 0.0
    for p, before in zip(model.parameters(), before_params):
        assert torch.equal(p, before), "single-rank merge changed live parameter bytes"
    for p, before in zip(model.parameters(), before_z):
        assert torch.equal(opt.state[p]['z'], before), "single-rank merge changed z bytes"
    for key in ('weight_sum', 'k', 'lr_max', 'scheduled_lr', 'train_mode'):
        assert opt.param_groups[0][key] == before_group[key], f"single-rank merge changed {key}"
    print("PASS test_single_rank_diloco_merge_is_byte_identical_noop")


def test_schedulefree_avg_merge_avoids_scalar_collectives():
    """The Frontier DiLoCo scaleout path must not rely on 1-element NCCL
    collectives for ScheduleFree scalar clocks. Tensor state is averaged; scalar
    clocks remain local."""
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=3)

    class NoScalarCollective:
        class ReduceOp:
            SUM = object()

        calls = []

        @staticmethod
        def all_reduce(tensor, op=None):
            assert op is NoScalarCollective.ReduceOp.SUM
            NoScalarCollective.calls.append(int(tensor.numel()))
            assert tensor.numel() > 1, (
                f"DiLoCo merge attempted scalar all_reduce with numel={tensor.numel()}"
            )
            tensor.mul_(2)

    old_dist = train.dist
    try:
        train.dist = NoScalarCollective
        before_group = {
            key: opt.param_groups[0][key]
            for key in ('weight_sum', 'k', 'lr_max')
        }
        train.diloco_merge(model, opt, _ns(), world_size=2, outer_state=None)
        after_group = {
            key: opt.param_groups[0][key]
            for key in ('weight_sum', 'k', 'lr_max')
        }
    finally:
        train.dist = old_dist

    assert NoScalarCollective.calls, "test did not exercise any tensor all_reduce"
    assert before_group == after_group, "ScheduleFree scalar clocks changed at merge"
    print("PASS test_schedulefree_avg_merge_avoids_scalar_collectives")


def test_bucketed_non_schedulefree_merge_preserves_sum_then_divide_semantics():
    """Bucketed non-SF merge must be exactly sum then divide per flat slice."""
    import train
    torch.manual_seed(123)
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Linear(5, 2))
    before = [p.detach().clone() for p in model.parameters()]

    class AddPeerCollective:
        class ReduceOp:
            SUM = object()

        calls = []

        @staticmethod
        def all_reduce(tensor, op=None):
            assert op is AddPeerCollective.ReduceOp.SUM
            AddPeerCollective.calls.append(int(tensor.numel()))
            tensor.add_(1.0)

    args = _ns(
        optimizer='adamw',
        diloco_merge_bucket_numel=4,
    )
    old_dist = train.dist
    try:
        train.dist = AddPeerCollective
        train.diloco_merge(model, optimizer=None, args=args, world_size=2, outer_state=None)
    finally:
        train.dist = old_dist

    assert len(AddPeerCollective.calls) > 1, "bucketed merge did not split the flat tensor"
    assert max(AddPeerCollective.calls) <= 4, AddPeerCollective.calls
    for p, old in zip(model.parameters(), before):
        expected = (old + 1.0) / 2.0
        assert torch.allclose(p, expected), "bucketed non-SF merge changed averaging math"
    print("PASS test_bucketed_non_schedulefree_merge_preserves_sum_then_divide_semantics")


def test_bucketed_schedulefree_merge_avoids_scalar_collectives():
    """ScheduleFree x/z bucketed merge still reduces tensor state only, never clocks."""
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=3)

    class BucketCollective:
        class ReduceOp:
            SUM = object()

        calls = []

        @staticmethod
        def all_reduce(tensor, op=None):
            assert op is BucketCollective.ReduceOp.SUM
            BucketCollective.calls.append(int(tensor.numel()))
            assert 1 <= tensor.numel() <= 7, (
                f"unexpected bucket size for ScheduleFree merge: {tensor.numel()}"
            )
            tensor.mul_(2)

    old_dist = train.dist
    try:
        train.dist = BucketCollective
        before_group = {
            key: opt.param_groups[0][key]
            for key in ('weight_sum', 'k', 'lr_max')
        }
        train.diloco_merge(
            model, opt, _ns(diloco_merge_bucket_numel=7),
            world_size=2, outer_state=None)
        after_group = {
            key: opt.param_groups[0][key]
            for key in ('weight_sum', 'k', 'lr_max')
        }
    finally:
        train.dist = old_dist

    assert len(BucketCollective.calls) > 1, "ScheduleFree bucketed merge did not split"
    assert before_group == after_group, "ScheduleFree scalar clocks changed at bucketed merge"
    print("PASS test_bucketed_schedulefree_merge_avoids_scalar_collectives")


def test_hierarchical_warmup_avoids_default_barrier_between_subgroups(monkeypatch):
    """Frontier NCCL smoke regression: warm only this rank's subgroup, then roots.

    The failed 4908293 smoke died during the old warm-up sequence, where ranks
    outside a subgroup entered a default-group barrier while subgroup members
    were still in that subgroup's NCCL all_reduce. The warm-up must not emit a
    default barrier until after local and root communicators have been warmed.
    """
    import train

    calls = []

    class FakeDist:
        @staticmethod
        def all_reduce(tensor, group=None):
            calls.append(("all_reduce", group))

        @staticmethod
        def barrier():
            calls.append(("barrier", None))

    monkeypatch.setattr(train, "dist", FakeDist)
    monkeypatch.setattr(train.torch, "zeros", lambda *args, **kwargs: object())

    args = _ns()
    args._diloco_merge_groups = {
        "topology": "hierarchical",
        "local_group": "local-ranks-4-7",
        "root_group": "root-ranks-0-4",
        "is_root": True,
    }

    train._warm_diloco_hierarchical_merge_groups(args, device="cuda:0")

    assert calls == [
        ("all_reduce", "local-ranks-4-7"),
        ("all_reduce", "root-ranks-0-4"),
        ("barrier", None),
    ]


def test_hierarchical_group_builder_creates_root_group_before_local_groups(monkeypatch):
    """Root communicator is created before overlapping local communicators."""
    import train

    calls = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def new_group(ranks):
            calls.append(("new_group", tuple(ranks)))
            return f"group-{tuple(ranks)}"

        @staticmethod
        def barrier():
            calls.append(("barrier", None))

    monkeypatch.setattr(train, "dist", FakeDist)

    meta = train._build_diloco_hierarchical_merge_groups(
        world_size=8, rank=4, group_size=4, create_barrier_every=0)

    assert calls == [
        ("new_group", (0, 4)),
        ("new_group", (0, 1, 2, 3)),
        ("new_group", (4, 5, 6, 7)),
    ]
    assert meta["root_ranks"] == [0, 4]
    assert meta["root_group"] == "group-(0, 4)"
    assert meta["local_group"] == "group-(4, 5, 6, 7)"
    assert meta["local_root"] == 4


def test_hierarchical_group_builder_paces_32n_g4_construction(monkeypatch):
    """The 32-node E97 g4 shape must not burst-create all RCCL groups at once."""
    import train

    calls = []

    class FakeDist:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def new_group(ranks):
            calls.append(("new_group", tuple(ranks)))
            return f"group-{tuple(ranks)}"

        @staticmethod
        def barrier():
            calls.append(("barrier", None))

    monkeypatch.setattr(train, "dist", FakeDist)

    meta = train._build_diloco_hierarchical_merge_groups(
        world_size=256, rank=252, group_size=4, create_barrier_every=8)

    root_ranks = tuple(range(0, 256, 4))
    assert calls[0] == ("new_group", root_ranks)
    assert calls[1] == ("barrier", None)
    assert meta["root_ranks"] == list(root_ranks)
    assert meta["local_group_ranks"] == [252, 253, 254, 255]
    assert meta["local_root"] == 252

    barrier_indexes = [idx for idx, call in enumerate(calls) if call == ("barrier", None)]
    assert barrier_indexes == [1, 10, 19, 28, 37, 46, 55, 64, 73]
    assert calls[-1] == ("barrier", None)


def test_hierarchical_merge_syncs_dependent_stages(monkeypatch):
    """Root ranks must not enqueue root all-reduce or local broadcast before the
    prior communicator has completed the same buffer."""
    import train

    calls = []

    class FakeDist:
        class ReduceOp:
            SUM = object()

        @staticmethod
        def reduce(tensor, dst=None, op=None, group=None):
            calls.append(("reduce", dst, group))

        @staticmethod
        def all_reduce(tensor, op=None, group=None):
            calls.append(("all_reduce", group))

        @staticmethod
        def broadcast(tensor, src=None, group=None):
            calls.append(("broadcast", src, group))

    monkeypatch.setattr(train, "dist", FakeDist)
    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train.torch.cuda, "synchronize",
                        lambda *args, **kwargs: calls.append(("cuda_sync",)))

    args = _ns()
    args._diloco_merge_groups = {
        "topology": "hierarchical",
        "local_group": "local-ranks-0-3",
        "local_root": 0,
        "is_root": True,
        "root_group": "root-ranks",
    }

    train._diloco_hierarchical_sum_average_flat(torch.ones(4), world_size=8, args=args)

    assert calls == [
        ("reduce", 0, "local-ranks-0-3"),
        ("cuda_sync",),
        ("all_reduce", "root-ranks"),
        ("cuda_sync",),
        ("broadcast", 0, "local-ranks-0-3"),
        ("cuda_sync",),
    ]


def _worker_hierarchical_exact(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train

    args = _ns(
        optimizer='adamw',
        diloco_merge_topology='hierarchical',
        diloco_merge_group_size=2,
        diloco_merge_bucket_numel=4,
    )
    args._rank = rank
    args._diloco_merge_groups = train._build_diloco_hierarchical_merge_groups(
        world_size=world_size, rank=rank, group_size=2)

    # Non-ScheduleFree path: exact weighted SUM/world_size over unequal groups
    # [0, 1] and [2]. A naive average of local averages would overweight rank 2.
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    with torch.no_grad():
        for idx, p in enumerate(model.parameters()):
            p.fill_(10.0 * rank + idx)
    pre_flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    gathered_pre = [torch.zeros_like(pre_flat) for _ in range(world_size)]
    dist.all_gather(gathered_pre, pre_flat)
    train.diloco_merge(model, optimizer=None, args=args, world_size=world_size, outer_state=None)
    post_flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    gathered_post = [torch.zeros_like(post_flat) for _ in range(world_size)]
    dist.all_gather(gathered_post, post_flat)

    # ScheduleFree path: x and z tensor state use the same hierarchical average.
    sf_model, opt = _build()
    _local_train(sf_model, opt, rank, k_steps=5)
    sf_args = _ns(
        diloco_merge_topology='hierarchical',
        diloco_merge_group_size=2,
        diloco_merge_bucket_numel=7,
    )
    sf_args._rank = rank
    sf_args._diloco_merge_groups = args._diloco_merge_groups
    opt.eval()
    pre_x = torch._utils._flatten_dense_tensors([p.data for p in sf_model.parameters()]).clone()
    opt.train()
    pre_z = torch._utils._flatten_dense_tensors(
        [opt.state[p]['z'] for p in sf_model.parameters()]).clone()
    gathered_sf = [None] * world_size
    dist.all_gather_object(gathered_sf, {
        'x': pre_x.tolist(),
        'z': pre_z.tolist(),
    })
    train.diloco_merge(sf_model, opt, sf_args, world_size=world_size, outer_state=None)
    opt.eval()
    post_x = torch._utils._flatten_dense_tensors([p.data for p in sf_model.parameters()]).clone()
    opt.train()
    post_z = torch._utils._flatten_dense_tensors(
        [opt.state[p]['z'] for p in sf_model.parameters()]).clone()
    gathered_post_x = [torch.zeros_like(post_x) for _ in range(world_size)]
    gathered_post_z = [torch.zeros_like(post_z) for _ in range(world_size)]
    dist.all_gather(gathered_post_x, post_x)
    dist.all_gather(gathered_post_z, post_z)

    if rank == 0:
        expected = torch.stack(gathered_pre).mean(dim=0)
        expected_sf_x = torch.stack([
            torch.tensor(row['x'], dtype=post_x.dtype)
            for row in gathered_sf
        ]).mean(dim=0)
        expected_sf_z = torch.stack([
            torch.tensor(row['z'], dtype=post_z.dtype)
            for row in gathered_sf
        ]).mean(dim=0)
        ret['hierarchical'] = {
            'non_sf_expected': expected.tolist(),
            'non_sf_post': [t.tolist() for t in gathered_post],
            'sf_x_expected': expected_sf_x.tolist(),
            'sf_x_post': [t.tolist() for t in gathered_post_x],
            'sf_z_expected': expected_sf_z.tolist(),
            'sf_z_post': [t.tolist() for t in gathered_post_z],
            'group_counts': [
                len(group_ranks)
                for group_ranks, _ in args._diloco_merge_groups['groups']
            ],
            'root_ranks': list(args._diloco_merge_groups['root_ranks']),
        }

    dist.barrier()
    dist.destroy_process_group()


def _run_hierarchical_exact():
    world_size = 3
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(_worker_hierarchical_exact, args=(world_size, init_file, ret),
             nprocs=world_size, join=True)
    return dict(ret)['hierarchical']


def test_hierarchical_merge_matches_global_average_for_unequal_groups():
    """Opt-in hierarchical merge must match the current global average exactly
    enough for dtype expectations, including unequal first-level groups."""
    r = _run_hierarchical_exact()
    assert r['group_counts'] == [2, 1], r['group_counts']
    assert r['root_ranks'] == [0, 2], r['root_ranks']
    tol = 1e-5
    expected = torch.tensor(r['non_sf_expected'])
    for post in r['non_sf_post']:
        assert torch.allclose(torch.tensor(post), expected, atol=tol, rtol=0), (
            "hierarchical non-ScheduleFree params != global mean"
        )
    expected_x = torch.tensor(r['sf_x_expected'])
    for post_x in r['sf_x_post']:
        assert torch.allclose(torch.tensor(post_x), expected_x, atol=tol, rtol=0), (
            "hierarchical ScheduleFree x != global mean"
        )
    expected_z = torch.tensor(r['sf_z_expected'])
    for post_z in r['sf_z_post']:
        assert torch.allclose(torch.tensor(post_z), expected_z, atol=tol, rtol=0), (
            "hierarchical ScheduleFree z != global mean"
        )
    print("PASS test_hierarchical_merge_matches_global_average_for_unequal_groups")


# ===========================================================================
# sf-diloco-p1 regression tests: lock the SF x/z/y geometry on the outer update.
#
# The bug: a nontrivial outer update displaced the SF EVAL weight x by s but did
# NOT displace z, so optimizer.train() rebuilt y = ybar + beta1*s instead of
# ybar + s -- only a beta1 fraction of the server update reached the next train
# point, and the x-z gap was stretched by -s. These four tests lock the fix
# (x<-x+s, z<-z+s => y<-y+s) forever. Test 1 FAILS on pre-fix code.
# ===========================================================================


def _worker_translation(rank, world_size, init_file, ret):
    """Scalar translation-invariance: engineer a known server displacement s=+3
    on x and assert the WHOLE geometry (x, z, y) shifts by s.

    Setup (per task spec): x=10, z=4, beta1=0.9 -> y=9.4. Outer update with
    anchor=4, outer_lr=1.5, outer_beta=0 -> delta=x_bar-anchor=6, mom=6,
    x_new=4+1.5*6=13 => s=+3. CORRECT: x=13, z=7, y=12.4. Buggy (z not shifted):
    y=0.9*13+0.1*4=12.1, z=4 -> FAILS here."""
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build_scalar()
    p = _set_scalar_state(model, opt, y_val=9.4, z_val=4.0)
    args = _ns(diloco_outer_lr=1.5, diloco_outer_beta=0.0)
    # Anchor lives in the SF EVAL (x) basis; engineered to 4.0 so s=+3.
    outer_state = {
        'anchor': [torch.full_like(p.data, 4.0)],
        'moment': [torch.zeros_like(p.data)],
    }
    train.diloco_merge(model, opt, args, world_size, outer_state)
    y_now = float(model.weight.data.reshape(-1)[0])
    opt.eval()
    x_now = float(model.weight.data.reshape(-1)[0])
    opt.train()
    z_now = float(opt.state[p]['z'].reshape(-1)[0])
    if rank == 0:
        ret['x'] = x_now
        ret['y'] = y_now
        ret['z'] = z_now
    dist.barrier()
    dist.destroy_process_group()


def _run_worker(worker, *extra):
    world_size = 2
    mgr = mp.Manager()
    ret = mgr.dict()
    with tempfile.NamedTemporaryFile() as f:
        init_file = f.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(worker, args=(world_size, init_file, *extra, ret),
             nprocs=world_size, join=True)
    return dict(ret)


def test_translation_invariance_scalar():
    """TEST 1 (FAILS pre-fix): a server displacement s applied to x must apply to
    z (hence y) too. x=10,z=4,beta1=0.9->y=9.4; s=+3 -> x=13,z=7,y=12.4."""
    r = _run_worker(_worker_translation)
    tol = 1e-4
    assert abs(r['x'] - 13.0) <= tol, f"x should shift 10->13, got {r['x']}"
    assert abs(r['z'] - 7.0) <= tol, (
        f"z should shift 4->7 (same displacement as x); got {r['z']} "
        f"(pre-fix bug leaves z=4)"
    )
    assert abs(r['y'] - 12.4) <= tol, (
        f"y should shift 9.4->12.4 (whole geometry translated by s=3); got "
        f"{r['y']} (pre-fix bug gives 12.1 because only beta1*s reaches y)"
    )
    print("PASS test_translation_invariance_scalar: x/z/y all shift by s=3")


def _worker_noop(rank, world_size, init_file, mode, ret):
    """Identical ranks (byte-identical model + SF state) -> diloco_merge is a
    no-op for every outer mode. Modes: 'avg', 'mom_lr1_b0', 'mom_b9_lr7' (the
    latter two with anchor=current x => zero delta)."""
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    _local_train_identical(model, opt, k_steps=12)  # identical across ranks

    # Snapshot pre-merge y (train), x (eval), z.
    pre_y = [p.data.detach().clone() for p in model.parameters()]
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    pre_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]

    if mode == 'avg':
        args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)
        outer_state = None
    elif mode == 'mom_lr1_b0':
        args = _ns(diloco_outer_lr=1.0, diloco_outer_beta=0.0)
        outer_state = {'anchor': [t.clone() for t in pre_x],   # anchor = current x => zero delta
                       'moment': [torch.zeros_like(t) for t in pre_x]}
    elif mode == 'mom_b9_lr7':
        args = _ns(diloco_outer_lr=0.7, diloco_outer_beta=0.9)
        outer_state = {'anchor': [t.clone() for t in pre_x],   # anchor = current x => zero delta
                       'moment': [torch.zeros_like(t) for t in pre_x]}
    else:
        raise ValueError(mode)

    os.environ['NDM_DILOCO_DEBUG_ASSERT'] = '1'  # exercise the runtime gap assert
    train.diloco_merge(model, opt, args, world_size, outer_state)

    post_y = [p.data.detach().clone() for p in model.parameters()]
    opt.eval()
    post_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    post_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]

    dx = max((a - b).abs().max().item() for a, b in zip(pre_x, post_x))
    dy = max((a - b).abs().max().item() for a, b in zip(pre_y, post_y))
    dz = max((a - b).abs().max().item() for a, b in zip(pre_z, post_z))
    if rank == 0:
        ret['dx'] = dx
        ret['dy'] = dy
        ret['dz'] = dz
    dist.barrier()
    dist.destroy_process_group()


def test_identical_rank_noop_all_modes():
    """TEST 2: identical ranks -> merge is a no-op for avg + momentum modes."""
    tol = 1e-4
    for mode in ('avg', 'mom_lr1_b0', 'mom_b9_lr7'):
        r = _run_worker(_worker_noop, mode)
        assert r['dx'] <= tol, f"[{mode}] x moved on identical-rank merge: {r['dx']}"
        assert r['dy'] <= tol, f"[{mode}] y moved on identical-rank merge: {r['dy']}"
        assert r['dz'] <= tol, f"[{mode}] z moved on identical-rank merge: {r['dz']}"
        print(f"PASS test_identical_rank_noop_all_modes[{mode}]: no-op confirmed")


def _worker_sfsgd_noop(rank, world_size, init_file, ret):
    """sfsgd no-op at a coherent boundary: outer_y equals the current inner train
    point and the exported endpoint, so D=0 and the rebase shift is zero."""
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    opt.train()

    pre_y = [p.data.detach().clone() for p in model.parameters()]
    # No inner steps yet, so x == y == z and export x is a strict no-op.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    outer_state = {
        'mode': 'sfsgd',
        'x': [t.clone() for t in pre_y],
        'z': [t.clone() for t in pre_y],
        'y': [t.clone() for t in pre_y],
        'k': 0,
        'weight_sum': 0.0,
        'lr_max': 1.0,
    }
    args = _ns(
        diloco_outer_optimizer='sfsgd',
        diloco_outer_lr=1.0,
        diloco_outer_beta=0.1,
        diloco_export_basis='x',
    )

    train.diloco_merge(model, opt, args, world_size, outer_state)

    post_y = [p.data.detach().clone() for p in model.parameters()]
    opt.eval()
    post_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()

    dy = max((a - b).abs().max().item() for a, b in zip(pre_y, post_y))
    dx = max((a - b).abs().max().item() for a, b in zip(pre_x, post_x))
    outer_dy = max((a - b).abs().max().item()
                   for a, b in zip(pre_y, outer_state['y']))
    if rank == 0:
        ret['dy'] = dy
        ret['dx'] = dx
        ret['outer_dy'] = outer_dy
        ret['k'] = outer_state['k']
        ret['weight_sum'] = outer_state['weight_sum']
    dist.barrier()
    dist.destroy_process_group()


def test_sfsgd_identical_rank_noop_at_coherent_boundary():
    r = _run_worker(_worker_sfsgd_noop)
    assert r['dx'] <= 1e-6, f"sfsgd coherent no-op moved x: {r['dx']}"
    assert r['dy'] <= 1e-6, f"sfsgd coherent no-op moved y: {r['dy']}"
    assert r['outer_dy'] <= 1e-6, f"sfsgd coherent no-op moved outer_y: {r['outer_dy']}"
    assert r['k'] == 1, "sfsgd should still advance its outer clock at a boundary"
    assert r['weight_sum'] > 0.0, "sfsgd should accumulate outer averaging weight"
    print("PASS test_sfsgd_identical_rank_noop_at_coherent_boundary")


def test_diloco_checkpoint_roundtrip_preserves_outer_and_inner_sf_state(monkeypatch):
    """Checkpoint both systems: inner ScheduleFree optimizer state and the
    separate outer sfsgd state must reload onto a fresh model coherently."""
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=4)
    args = _ns(diloco_outer_optimizer='sfsgd', diloco_outer_beta=0.1)
    outer_state = train.initialize_diloco_outer_state(model, opt, args)
    with torch.no_grad():
        for t in outer_state['x']:
            t.add_(0.125)
        for t in outer_state['z']:
            t.sub_(0.25)
        for t in outer_state['y']:
            t.add_(0.5)
        outer_state['k'] = 7
        outer_state['weight_sum'] = 3.5
        outer_state['lr_max'] = 2.0

    with tempfile.TemporaryDirectory() as d:
        path = train.save_checkpoint(
            model, opt, step=123, loss=4.5, output_dir=Path(d),
            keep_n=2, outer_state=outer_state, total_tokens=123000)
        model2, opt2 = _build()
        real_torch_load = torch.load
        load_calls = []

        def recording_load(*args, **kwargs):
            load_calls.append(dict(kwargs))
            return real_torch_load(*args, **kwargs)

        monkeypatch.setattr(torch, 'load', recording_load)
        step, loss, ckpt = train.load_checkpoint(
            path, model2, opt2, return_checkpoint=True)
        restored = train.initialize_diloco_outer_state(
            model2, opt2, args, loaded_state=ckpt['diloco_outer_state'])

    assert step == 123
    assert loss == 4.5
    for key in ('k', 'weight_sum', 'lr_max'):
        assert restored[key] == outer_state[key], f"outer {key} did not roundtrip"
    for key in ('x', 'z', 'y'):
        for a, b in zip(restored[key], outer_state[key]):
            assert torch.equal(a, b), f"outer {key} tensor did not roundtrip"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "model parameter did not roundtrip"
        assert torch.equal(opt.state[p1]['z'], opt2.state[p2]['z']), "inner z did not roundtrip"
    assert opt.param_groups[0]['k'] == opt2.param_groups[0]['k']
    assert opt.param_groups[0]['weight_sum'] == opt2.param_groups[0]['weight_sum']
    assert load_calls == [{
        'map_location': 'cpu',
        'mmap': True,
        'weights_only': False,
    }], "restart checkpoint tensors must remain file-backed to bound per-node RAM"
    print("PASS test_diloco_checkpoint_roundtrip_preserves_outer_and_inner_sf_state")


def test_nonavg_resume_missing_outer_state_fails_closed():
    import train
    model, opt = _build()
    args = _ns(
        resume='missing_outer_state.pt',
        diloco_outer_optimizer='momentum',
        diloco_bootstrap_outer_state='none',
    )
    try:
        train.resolve_diloco_outer_state_for_resume(
            model, opt, args, loaded_state=None, ckpt={'step': 12},
            inner_optimizer_state_loaded=True)
    except ValueError as exc:
        assert 'from-loaded-model' in str(exc)
        assert 'missing_diloco_outer_state' in str(exc)
    else:
        raise AssertionError("non-avg resume without outer state did not fail closed")
    print("PASS test_nonavg_resume_missing_outer_state_fails_closed")


def test_bootstrap_momentum_preserves_loaded_model_tensors():
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=4)
    opt.eval()
    expected_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    pre = [p.data.detach().clone() for p in model.parameters()]

    args = _ns(
        resume='loaded.pt',
        diloco_outer_optimizer='momentum',
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    outer_state, metadata = train.resolve_diloco_outer_state_for_resume(
        model, opt, args, loaded_state=None, ckpt={'step': 4},
        inner_optimizer_state_loaded=True)

    assert outer_state['mode'] == 'momentum'
    assert opt.param_groups[0]['train_mode'] is True
    for p, before in zip(model.parameters(), pre):
        assert torch.equal(p.data, before), "momentum bootstrap changed live model tensor bytes"
    for anchor, expected in zip(outer_state['anchor'], expected_x):
        assert torch.allclose(anchor, expected, rtol=0.0, atol=1e-7), (
            "momentum anchor did not use loaded eval/x basis"
        )
    for moment in outer_state['moment']:
        assert torch.equal(moment, torch.zeros_like(moment)), "momentum buffer is not exact zero"
    assert metadata['inner_schedulefree_basis_used_for_anchor'] == 'x'
    assert metadata['model_tensors_equal_after_restore'] is True
    print("PASS test_bootstrap_momentum_preserves_loaded_model_tensors")


def _worker_bootstrap_partial_average(rank, world_size, init_file, ret):
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    args = _ns(
        resume='loaded.pt',
        diloco_outer_optimizer='momentum',
        diloco_outer_lr=0.5,
        diloco_outer_beta=0.0,
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    outer_state, metadata = train.resolve_diloco_outer_state_for_resume(
        model, opt, args, loaded_state=None, ckpt={'step': 0},
        inner_optimizer_state_loaded=False)
    anchor = [t.detach().clone() for t in outer_state['anchor']]
    assert metadata['inner_schedulefree_basis_used_for_anchor'] == 'loaded'

    _local_train(model, opt, rank, k_steps=8)
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {'x': [t.tolist() for t in pre_x]})

    train.diloco_merge(model, opt, args, world_size, outer_state)
    opt.eval()
    post_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()

    if rank == 0:
        expected = []
        for j, a in enumerate(anchor):
            mean_x = torch.zeros_like(a)
            for rr in range(world_size):
                mean_x += torch.tensor(gathered[rr]['x'][j])
            mean_x /= world_size
            expected.append(a + 0.5 * (mean_x - a))
        ret['max_diff'] = max((a - b).abs().max().item() for a, b in zip(post_x, expected))
        ret['anchor_advanced'] = max(
            (a - b).abs().max().item()
            for a, b in zip(outer_state['anchor'], expected)
        )
    flat = torch._utils._flatten_dense_tensors([p.data for p in model.parameters()]).clone()
    allp = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(allp, flat)
    if rank == 0:
        ret['consensus_diff'] = max((allp[r] - allp[0]).abs().max().item()
                                    for r in range(world_size))
    dist.barrier()
    dist.destroy_process_group()


def test_bootstrap_partial_average_first_merge_math():
    r = _run_worker(_worker_bootstrap_partial_average)
    assert r['max_diff'] <= 1e-5, f"partial-average bootstrap first merge mismatch: {r}"
    assert r['anchor_advanced'] <= 1e-5, f"partial-average anchor did not advance: {r}"
    assert r['consensus_diff'] <= 1e-6, f"partial-average merge lost consensus: {r}"
    print("PASS test_bootstrap_partial_average_first_merge_math")


def test_bootstrap_sfsgd_y_preserves_loaded_model_tensors():
    import train
    model, opt = _build()
    _local_train(model, opt, rank=0, k_steps=4)
    opt.train()
    expected_y = [p.data.detach().clone() for p in model.parameters()]
    pre = [p.data.detach().clone() for p in model.parameters()]
    args = _ns(
        resume='loaded.pt',
        diloco_outer_optimizer='sfsgd',
        diloco_outer_lr=1.25,
        diloco_outer_beta=0.1,
        diloco_export_basis='y',
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    outer_state, metadata = train.resolve_diloco_outer_state_for_resume(
        model, opt, args, loaded_state=None, ckpt={'step': 4},
        inner_optimizer_state_loaded=True)
    assert opt.param_groups[0]['train_mode'] is True
    for p, before in zip(model.parameters(), pre):
        assert torch.equal(p.data, before), "sfsgd_y bootstrap changed live model tensor bytes"
    for key in ('x', 'z', 'y'):
        for got, expected in zip(outer_state[key], expected_y):
            assert torch.equal(got, expected), f"sfsgd_y outer {key} did not start at loaded y"
    assert outer_state['k'] == 0
    assert outer_state['weight_sum'] == 0.0
    assert outer_state['lr_max'] == 1.25
    assert metadata['inner_schedulefree_basis_used_for_anchor'] == 'y'
    print("PASS test_bootstrap_sfsgd_y_preserves_loaded_model_tensors")


def test_bootstrap_sfsgd_y_pretrained_no_optimizer_state():
    import train
    model, opt = _build()
    pre = [p.data.detach().clone() for p in model.parameters()]
    args = _ns(
        resume='pretrained.pt',
        diloco_outer_optimizer='sfsgd',
        diloco_outer_lr=0.75,
        diloco_outer_beta=0.1,
        diloco_export_basis='y',
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    outer_state, metadata = train.resolve_diloco_outer_state_for_resume(
        model, opt, args, loaded_state=None, ckpt={'step': 0},
        inner_optimizer_state_loaded=False)
    for p, before in zip(model.parameters(), pre):
        assert torch.equal(p.data, before), "pretrained sfsgd_y bootstrap changed model bytes"
    for key in ('x', 'z', 'y'):
        for got, expected in zip(outer_state[key], pre):
            assert torch.equal(got, expected), f"pretrained sfsgd_y outer {key} != loaded weights"
    assert metadata['inner_optimizer_state_loaded'] is False
    assert metadata['inner_schedulefree_basis_used_for_anchor'] == 'loaded'
    print("PASS test_bootstrap_sfsgd_y_pretrained_no_optimizer_state")


def test_checkpoint_metadata_records_outer_bootstrap():
    import train
    model, opt = _build()
    args = _ns(
        resume='source.pt',
        diloco_outer_optimizer='momentum',
        diloco_outer_lr=0.5,
        diloco_outer_beta=0.0,
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    outer_state, bootstrap_metadata = train.resolve_diloco_outer_state_for_resume(
        model, opt, args, loaded_state=None, ckpt={'step': 99},
        inner_optimizer_state_loaded=False)
    metadata = train.checkpoint_metadata_with_diloco_bootstrap(
        {'kind': 'periodic'}, bootstrap_metadata)
    with tempfile.TemporaryDirectory() as d:
        path = train.save_checkpoint(
            model, opt, step=100, loss=1.25, output_dir=Path(d), keep_n=2,
            outer_state=outer_state, metadata=metadata, total_tokens=100000)
        ckpt = torch.load(path, map_location='cpu')
    block = ckpt['checkpoint_metadata']['diloco_outer_state_bootstrap']
    assert block['performed'] is True
    assert block['guard'] == 'from-loaded-model'
    assert block['source_checkpoint'] == 'source.pt'
    assert block['source_checkpoint_step'] == 99
    assert block['missing_or_incompatible_reason'] == 'missing_diloco_outer_state'
    assert block['target_outer_optimizer'] == 'momentum'
    assert block['target_outer_lr'] == 0.5
    assert block['target_outer_beta'] == 0.0
    assert block['bootstrap_source'] == 'loaded_model_weights'
    assert block['model_tensors_equal_after_restore'] is True
    assert 'restored byte-identical' in block['model_weight_mutation']
    assert ckpt['diloco_outer_state']['bootstrap_metadata']['guard'] == 'from-loaded-model'
    print("PASS test_checkpoint_metadata_records_outer_bootstrap")


def test_compatible_outer_state_not_overwritten_by_bootstrap_flag():
    import train
    model, opt = _build()
    args = _ns(
        resume='source.pt',
        diloco_outer_optimizer='sfsgd',
        diloco_bootstrap_outer_state='from-loaded-model',
    )
    loaded_state = {
        'mode': 'sfsgd',
        'x': [p.data.detach().clone() for p in model.parameters()],
        'z': [p.data.detach().clone() for p in model.parameters()],
        'y': [p.data.detach().clone() for p in model.parameters()],
        'k': 3,
        'weight_sum': 2.0,
        'lr_max': 1.0,
    }
    try:
        train.resolve_diloco_outer_state_for_resume(
            model, opt, args, loaded_state=loaded_state,
            ckpt={'step': 3, 'diloco_outer_state': loaded_state},
            inner_optimizer_state_loaded=True)
    except ValueError as exc:
        assert 'compatible diloco_outer_state' in str(exc)
    else:
        raise AssertionError("bootstrap flag silently overwrote compatible outer state")
    print("PASS test_compatible_outer_state_not_overwritten_by_bootstrap_flag")


def _worker_basis(rank, world_size, init_file, ret):
    """Basis consistency: after real training x != y. Capture the anchor the way
    train.py does (SF EVAL basis), then merge with NO cross-rank divergence (the
    ranks trained identically). The first outer delta = x_bar - anchor must be 0.
    A train(y)-basis anchor would give delta = x - y != 0 (since x != y)."""
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    _local_train_identical(model, opt, k_steps=15)  # x != y now, identical across ranks

    # Anchor in the EVAL (x) basis -- exactly train.py's fixed capture.
    opt.eval()
    anchor = [p.data.detach().clone() for p in model.parameters()]
    x_at_capture = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    y_at_capture = [p.data.detach().clone() for p in model.parameters()]

    # Non-vacuity: x must actually differ from y (else any basis would pass).
    max_xy_gap = max((xa - ya).abs().max().item()
                     for xa, ya in zip(x_at_capture, y_at_capture))
    # Anchor must equal eval-x (x-basis), and differ from y (would-be wrong basis).
    anchor_minus_x = max((a - xa).abs().max().item()
                         for a, xa in zip(anchor, x_at_capture))
    anchor_minus_y = max((a - ya).abs().max().item()
                         for a, ya in zip(anchor, y_at_capture))

    outer_state = {'anchor': anchor, 'moment': [torch.zeros_like(p.data) for p in model.parameters()]}
    args = _ns(diloco_outer_lr=0.7, diloco_outer_beta=0.9)

    # NO further training between capture and merge -> first delta must be ~0.
    opt.eval()
    x_bar = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    first_delta = max((xb - a).abs().max().item() for xb, a in zip(x_bar, anchor))

    train.diloco_merge(model, opt, args, world_size, outer_state)

    opt.eval()
    post_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    x_moved = max((px - xb).abs().max().item() for px, xb in zip(post_x, x_bar))

    if rank == 0:
        ret['max_xy_gap'] = max_xy_gap
        ret['anchor_minus_x'] = anchor_minus_x
        ret['anchor_minus_y'] = anchor_minus_y
        ret['first_delta'] = first_delta
        ret['x_moved'] = x_moved
    dist.barrier()
    dist.destroy_process_group()


def test_anchor_is_x_basis_first_delta_zero():
    """TEST 3: the DiLoCo anchor is in the SF EVAL (x) basis, so with no local
    training between merges the first outer delta is x_bar - x == 0, NOT
    x_bar - y. Guards the anchor-basis fix."""
    r = _run_worker(_worker_basis)
    assert r['max_xy_gap'] > 1e-3, (
        f"test is vacuous: x==y at capture (gap {r['max_xy_gap']}); need trained state"
    )
    assert r['anchor_minus_x'] <= 1e-6, (
        f"anchor is NOT in x-basis: anchor-x = {r['anchor_minus_x']}"
    )
    assert r['anchor_minus_y'] > 1e-3, (
        f"anchor coincides with y (wrong basis): anchor-y = {r['anchor_minus_y']}"
    )
    assert r['first_delta'] <= 1e-6, (
        f"first outer delta (x_bar - anchor) should be 0 with no training; "
        f"got {r['first_delta']} -> anchor is in the wrong basis"
    )
    assert r['x_moved'] <= 1e-4, (
        f"zero-delta merge moved x by {r['x_moved']} (should be a no-op)"
    )
    print("PASS test_anchor_is_x_basis_first_delta_zero: anchor confirmed in x-basis")


def _worker_gap(rank, world_size, init_file, ret):
    """State-gap preservation under a NONTRIVIAL outer update (nonzero s). Ranks
    train divergently so the merged x/z and the outer momentum both move; assert
    (z_after - x_after) == (z_before - x_before), i.e. the x-z gap (and y's offset
    from x) is invariant to the server rebase."""
    dist.init_process_group(backend='gloo', init_method=f'file://{init_file}',
                            rank=rank, world_size=world_size)
    import train
    model, opt = _build()
    # Anchor at init (x==y here), in x-basis.
    opt.eval()
    anchor = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    outer_state = {'anchor': anchor, 'moment': [torch.zeros_like(p.data) for p in model.parameters()]}
    args = _ns(diloco_outer_lr=0.7, diloco_outer_beta=0.9)

    _local_train(model, opt, rank, k_steps=15)  # rank-specific -> divergence + nonzero delta

    # Pre-merge per-rank x and z; gather to form the post-AVERAGING gap.
    opt.eval()
    pre_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    pre_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, {
        'x': [t.tolist() for t in pre_x],
        'z': [t.tolist() for t in pre_z],
    })

    os.environ['NDM_DILOCO_DEBUG_ASSERT'] = '1'  # exercise the runtime gap assert
    train.diloco_merge(model, opt, args, world_size, outer_state)

    opt.eval()
    post_x = [p.data.detach().clone() for p in model.parameters()]
    opt.train()
    post_z = [opt.state[p]['z'].detach().clone() for p in model.parameters()]

    if rank == 0:
        # gap_before = mean_z - mean_x (state right after averaging, before outer step)
        gap_drift = 0.0
        s_norm = 0.0
        for j in range(len(pre_x)):
            mean_x = torch.zeros_like(pre_x[j])
            mean_z = torch.zeros_like(pre_z[j])
            for rr in range(world_size):
                mean_x += torch.tensor(gathered[rr]['x'][j])
                mean_z += torch.tensor(gathered[rr]['z'][j])
            mean_x /= world_size
            mean_z /= world_size
            gap_before = mean_z - mean_x
            gap_after = post_z[j] - post_x[j]
            gap_drift = max(gap_drift, (gap_after - gap_before).abs().max().item())
            s_norm = max(s_norm, (post_x[j] - mean_x).abs().max().item())  # |s| = |x_new - x_bar|
        ret['gap_drift'] = gap_drift
        ret['s_norm'] = s_norm
    dist.barrier()
    dist.destroy_process_group()


def test_state_gap_preserved_under_outer_rebase():
    """TEST 4: for any SF param, (z+s)-(x+s) == z-x after the outer rebase. The
    x-z gap drift must be ~0 even though the outer update moves x by a nonzero s
    (pre-fix bug drives the gap drift to -s)."""
    r = _run_worker(_worker_gap)
    assert r['s_norm'] > 1e-4, (
        f"test is vacuous: outer update did not move x (|s|={r['s_norm']})"
    )
    print(f"  [test4] |s| (server displacement) = {r['s_norm']:.6g}; "
          f"x-z gap drift = {r['gap_drift']:.3e}")
    assert r['gap_drift'] <= 1e-4, (
        f"x-z gap drifted by {r['gap_drift']} under outer rebase "
        f"(pre-fix bug stretches the gap by -s={-r['s_norm']:.4g})"
    )
    print("PASS test_state_gap_preserved_under_outer_rebase: SF geometry invariant")


if __name__ == '__main__':
    test_nonavg_resume_missing_outer_state_fails_closed()
    test_bootstrap_momentum_preserves_loaded_model_tensors()
    test_bootstrap_partial_average_first_merge_math()
    test_bootstrap_sfsgd_y_preserves_loaded_model_tensors()
    test_bootstrap_sfsgd_y_pretrained_no_optimizer_state()
    test_checkpoint_metadata_records_outer_bootstrap()
    test_compatible_outer_state_not_overwritten_by_bootstrap_flag()
    test_local_sgd_averaging()
    test_outer_momentum()
    test_schedulefree_dynamic_loss_continuity()
    test_schedulefree_full_trajectory_average_survives_merges()
    test_single_rank_diloco_merge_is_byte_identical_noop()
    test_schedulefree_avg_merge_avoids_scalar_collectives()
    test_bucketed_non_schedulefree_merge_preserves_sum_then_divide_semantics()
    test_bucketed_schedulefree_merge_avoids_scalar_collectives()
    # sf-diloco-p1 regression suite (x/z/y geometry on outer update):
    test_translation_invariance_scalar()
    test_identical_rank_noop_all_modes()
    test_anchor_is_x_basis_first_delta_zero()
    test_state_gap_preserved_under_outer_rebase()
    print("ALL DILOCO MERGE TESTS PASSED")
