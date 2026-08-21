import os
import tempfile
from types import SimpleNamespace

import pytest
import schedulefree
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree


def _make_pair(device="cpu", dtype=torch.float32, *, pin_memory=False):
    initial = torch.linspace(-1.0, 1.0, 257, device=device, dtype=dtype)
    reference_parameter = torch.nn.Parameter(initial.clone())
    offload_parameter = torch.nn.Parameter(initial.clone())
    reference = schedulefree.AdamWScheduleFree(
        [reference_parameter], lr=2.5e-3, betas=(0.9, 0.95),
        weight_decay=0.1, warmup_steps=2, foreach=False)
    offload = CPUOffloadAdamWScheduleFree(
        [offload_parameter], lr=2.5e-3, betas=(0.9, 0.95),
        weight_decay=0.1, warmup_steps=2, pin_memory=pin_memory,
        bucket_numel=31)
    return reference_parameter, offload_parameter, reference, offload


def _advance_pair(reference_parameter, offload_parameter, reference, offload,
                  start, stop):
    reference.train()
    offload.train()
    for step in range(start, stop):
        grad = torch.linspace(-0.2, 0.3, reference_parameter.numel(),
                              device=reference_parameter.device,
                              dtype=reference_parameter.dtype) * (step + 1)
        reference_parameter.grad = grad.clone()
        offload_parameter.grad = grad.clone()
        reference.step()
        offload.step()


def test_cpu_offload_matches_schedulefree_and_restores_without_device_cast():
    rp, op, reference, offload = _make_pair(pin_memory=False)
    before = op.detach().clone()
    init_stats = offload.initialize_state_()
    assert init_stats["state_bytes"] == 2 * op.numel() * op.element_size()
    assert torch.equal(op, before)
    _advance_pair(rp, op, reference, offload, 0, 4)
    torch.testing.assert_close(op, rp, rtol=0, atol=0)
    torch.testing.assert_close(offload.state[op]["z"], reference.state[rp]["z"],
                               rtol=0, atol=0)
    torch.testing.assert_close(
        offload.state[op]["exp_avg_sq"], reference.state[rp]["exp_avg_sq"],
        rtol=0, atol=0)

    reference.eval()
    offload.eval()
    torch.testing.assert_close(op, rp, rtol=0, atol=0)
    reference.train()
    offload.train()
    torch.testing.assert_close(op, rp, rtol=0, atol=0)

    saved = offload.state_dict()
    resumed_parameter = torch.nn.Parameter(op.detach().clone())
    resumed = CPUOffloadAdamWScheduleFree(
        [resumed_parameter], lr=9.9, pin_memory=False)
    resumed.load_state_dict(saved)
    assert resumed.state[resumed_parameter]["z"].device.type == "cpu"
    assert resumed.param_groups[0]["k"] == offload.param_groups[0]["k"]

    grad = torch.linspace(-0.4, 0.1, op.numel())
    op.grad = grad.clone()
    resumed_parameter.grad = grad.clone()
    offload.step()
    resumed.step()
    torch.testing.assert_close(resumed_parameter, op, rtol=0, atol=0)
    torch.testing.assert_close(
        resumed.state[resumed_parameter]["z"], offload.state[op]["z"],
        rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_first_step_initializes_pinned_cpu_state_and_matches_reference():
    rp, op, reference, offload = _make_pair(
        device="cuda", dtype=torch.bfloat16, pin_memory=True)
    _advance_pair(rp, op, reference, offload, 0, 3)
    torch.testing.assert_close(op, rp, rtol=0, atol=0)
    state = offload.state[op]
    assert state["z"].device.type == "cpu" and state["z"].is_pinned()
    assert (state["exp_avg_sq"].device.type == "cpu"
            and state["exp_avg_sq"].is_pinned())
    assert offload.last_step_stats["h2d_bytes"] == 4 * op.numel()
    assert offload.last_step_stats["d2h_bytes"] == 4 * op.numel()
    assert op.grad is None


def _diloco_offload_worker(rank, world_size, init_file, result):
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}",
        rank=rank, world_size=world_size)
    import train

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))
    optimizer = CPUOffloadAdamWScheduleFree(
        model.parameters(), lr=1.0e-2, pin_memory=False)
    optimizer.train()
    generator = torch.Generator().manual_seed(100 + rank)
    for _ in range(3):
        x = torch.randn(4, 8, generator=generator)
        target = torch.randn(4, 8, generator=generator) + rank
        ((model(x) - target) ** 2).mean().backward()
        optimizer.step()

    args = SimpleNamespace(
        optimizer="schedulefree", diloco_outer_optimizer="avg",
        diloco_export_basis="x", diloco_outer_lr=1.0,
        diloco_outer_beta=0.0, diloco_merge_bucket_numel=31,
        diloco_merge_topology="global", diloco_merge_debug=0,
        diloco_merge_debug_ranks="0", diloco_merge_completion_barrier=0)
    train.diloco_merge(model, optimizer, args, world_size, None)
    optimizer.assert_state_offloaded()

    values = [p.detach().clone() for p in model.parameters()]
    values.extend(optimizer.state[p]["z"].clone() for p in model.parameters())
    for tensor in values:
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        if rank == 0 and any(not torch.equal(gathered[0], other)
                             for other in gathered[1:]):
            result["consensus"] = False
    if rank == 0 and "consensus" not in result:
        result["consensus"] = True
    dist.destroy_process_group()


def test_diloco_avg_streams_offloaded_z_through_bounded_buckets():
    manager = mp.Manager()
    result = manager.dict()
    with tempfile.NamedTemporaryFile() as handle:
        init_file = handle.name
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(
        _diloco_offload_worker,
        args=(2, init_file, result), nprocs=2, join=True)
    assert result["consensus"] is True
