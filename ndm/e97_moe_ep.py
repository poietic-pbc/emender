"""Eight-GCD node-local RCCL transport for E97 expert assignments.

This module fails closed unless the process group is exactly the eight ranks on
one physical node. It never accepts the global DiLoCo process group: expert
assignments are exchanged only through the separately constructed node group.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import socket

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from .triton.e97_moe_ep import (
    EP_SIZE,
    EPSendPlan,
    build_ep_send_plan,
    build_local_expert_plan,
    unpack_local_expert_rows,
)
from .triton.e97_moe_fused import (
    checkpointed_packed_local_experts_grouped,
    checkpointed_packed_local_experts_rocblas,
    checkpointed_shared_expert_rocblas,
    fused_packed_local_experts_autograd,
    fused_shared_expert_autograd,
    fused_shared_top3_combine_autograd,
    fused_top3_router_autograd,
)


@dataclass(frozen=True)
class NodeLocalEPTopology:
    hostname: str
    hostname_fingerprint: int
    global_ranks: tuple[int, ...]
    local_ranks: tuple[int, ...]
    backend: str


@dataclass(frozen=True)
class EPExchange:
    send_plan: EPSendPlan
    received_x: torch.Tensor
    received_local_expert: torch.Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]


@dataclass(frozen=True)
class NodeLocalMoEResult:
    output: torch.Tensor
    top_indices: torch.Tensor
    expert_counts: torch.Tensor
    load_balance_loss: torch.Tensor
    z_loss: torch.Tensor
    topology: NodeLocalEPTopology


@dataclass(frozen=True)
class MoEProcessGroups:
    node_group: object
    diloco_lane_group: object
    node_index: int
    local_rank: int
    node_count: int


def create_moe_process_groups() -> MoEProcessGroups:
    """Create node EP groups and corresponding-rank cross-node DiLoCo lanes."""
    if not dist.is_initialized():
        raise RuntimeError("distributed process group is not initialized")
    world = dist.get_world_size()
    if world % EP_SIZE:
        raise RuntimeError("global world size must be a multiple of eight GCDs")
    node_count = world // EP_SIZE
    rank = dist.get_rank()
    node_index, local_rank = divmod(rank, EP_SIZE)
    node_group = None
    for node in range(node_count):
        group = dist.new_group(tuple(range(node * EP_SIZE, (node + 1) * EP_SIZE)), backend="nccl")
        if node == node_index:
            node_group = group
    lane_group = None
    for lane in range(EP_SIZE):
        ranks = tuple(node * EP_SIZE + lane for node in range(node_count))
        group = dist.new_group(ranks, backend="nccl")
        if lane == local_rank:
            lane_group = group
    if node_group is None or lane_group is None:
        raise RuntimeError("failed to construct MoE process-group hierarchy")
    return MoEProcessGroups(
        node_group=node_group, diloco_lane_group=lane_group,
        node_index=node_index, local_rank=local_rank, node_count=node_count)


def diloco_average_schedulefree_(model, optimizer, *, lane_group) -> None:
    """Average ScheduleFree x/z between corresponding node shards.

    Only model and optimizer tensors cross nodes. Expert-token dispatch is not
    accepted by this API and remains confined to each node group.
    """
    lane_world = dist.get_world_size(lane_group)
    if lane_world <= 1:
        return
    optimizer.eval()
    for parameter in model.parameters():
        dist.all_reduce(parameter.data, op=dist.ReduceOp.AVG, group=lane_group)
        state = optimizer.state.get(parameter, {})
        z = state.get("z")
        if z is not None:
            if z.is_cuda:
                dist.all_reduce(z, op=dist.ReduceOp.AVG, group=lane_group)
            else:
                z_work = z.to(parameter.device, non_blocking=False)
                dist.all_reduce(z_work, op=dist.ReduceOp.AVG, group=lane_group)
                z.copy_(z_work, non_blocking=False)
    optimizer.train()
    optimizer.assert_no_master_weights()


def _hostname_fingerprint(hostname: str) -> int:
    digest = hashlib.sha256(hostname.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def assert_node_local_ep_group(group=None) -> NodeLocalEPTopology:
    """Collectively prove that ``group`` is one complete eight-GCD node."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("E97 expert parallelism requires initialized torch.distributed")
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("E97 expert parallelism requires Frontier ROCm/HIP")
    world = dist.get_world_size(group)
    if world != EP_SIZE:
        raise RuntimeError(f"expert process group must contain exactly {EP_SIZE} ranks, got {world}")
    backend = str(dist.get_backend(group)).lower()
    if backend != "nccl":
        raise RuntimeError(f"expert process group must use RCCL/NCCL, got {backend}")
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "-1")))
    if not 0 <= local_rank < EP_SIZE:
        raise RuntimeError(f"invalid node-local expert rank {local_rank}")
    hostname = socket.gethostname()
    fingerprint = _hostname_fingerprint(hostname)
    global_rank = dist.get_rank()
    local = torch.tensor([fingerprint, global_rank, local_rank], device="cuda", dtype=torch.int64)
    gathered = torch.empty((EP_SIZE, 3), device="cuda", dtype=torch.int64)
    dist.all_gather_into_tensor(gathered, local, group=group)
    evidence = gathered.cpu().tolist()
    fingerprints = {row[0] for row in evidence}
    local_ranks = tuple(sorted(row[2] for row in evidence))
    if fingerprints != {fingerprint}:
        raise RuntimeError("expert process group crosses a physical node boundary")
    if local_ranks != tuple(range(EP_SIZE)):
        raise RuntimeError(f"expert process group local ranks are not 0..7: {local_ranks}")
    try:
        group_ranks = tuple(dist.get_process_group_ranks(group))
    except AttributeError:
        group_ranks = tuple(sorted(row[1] for row in evidence))
    if tuple(sorted(row[1] for row in evidence)) != tuple(sorted(group_ranks)):
        raise RuntimeError("expert process-group rank identity mismatch")
    return NodeLocalEPTopology(
        hostname=hostname,
        hostname_fingerprint=fingerprint,
        global_ranks=group_ranks,
        local_ranks=local_ranks,
        backend=backend,
    )


def exchange_expert_assignments(
    x: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    group=None,
    topology: NodeLocalEPTopology | None = None,
) -> EPExchange:
    """Pack and exchange assignments through one proven node-local RCCL group."""
    if topology is None:
        topology = assert_node_local_ep_group(group)
    if len(topology.global_ranks) != EP_SIZE:
        raise RuntimeError("invalid cached expert topology evidence")
    plan = build_ep_send_plan(x, top_indices)
    receive_counts = torch.empty_like(plan.send_counts)
    dist.all_to_all_single(receive_counts, plan.send_counts, group=group)
    send_splits = tuple(int(value) for value in plan.send_counts.cpu().tolist())
    receive_splits = tuple(int(value) for value in receive_counts.cpu().tolist())
    if sum(send_splits) != x.shape[0] * 3:
        raise RuntimeError("expert send accounting lost assignments")
    receive_rows = sum(receive_splits)
    received_x = torch.empty((receive_rows, x.shape[1]), device=x.device, dtype=x.dtype)
    received_local_expert = torch.empty(receive_rows, device=x.device, dtype=torch.int32)
    received_x = dist_nn.all_to_all_single(
        received_x, plan.send_x,
        output_split_sizes=list(receive_splits), input_split_sizes=list(send_splits),
        group=group,
    )
    dist.all_to_all_single(
        received_local_expert, plan.send_local_expert,
        output_split_sizes=list(receive_splits), input_split_sizes=list(send_splits),
        group=group,
    )
    return EPExchange(
        send_plan=plan,
        received_x=received_x,
        received_local_expert=received_local_expert,
        send_splits=send_splits,
        receive_splits=receive_splits,
    )


def average_replicated_gradients_(
    parameters,
    *,
    group=None,
    topology: NodeLocalEPTopology | None = None,
) -> None:
    """RCCL-average replicated gradients over exactly one proven node group."""
    if topology is None:
        topology = assert_node_local_ep_group(group)
    if len(topology.global_ranks) != EP_SIZE:
        raise RuntimeError("invalid cached expert topology evidence")
    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("replicated parameter is missing a gradient")
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=group)


def node_replicated_parameters(model):
    """Yield replicated parameters, excluding packed rank-owned expert shards."""
    local_ids = set()
    for module in model.modules():
        for attribute in ("local_gate_weight", "local_up_weight", "local_down_weight"):
            parameter = getattr(module, attribute, None)
            if isinstance(parameter, torch.nn.Parameter):
                local_ids.add(id(parameter))
    for parameter in model.parameters():
        if parameter.requires_grad and id(parameter) not in local_ids:
            yield parameter


def node_local_fused_moe_autograd(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    local_gate_weight: torch.Tensor,
    local_up_weight: torch.Tensor,
    local_down_weight: torch.Tensor,
    shared_gate_weight: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    *,
    group=None,
    topology: NodeLocalEPTopology | None = None,
    expert_backend: str = "triton",
    forced_top_indices: torch.Tensor | None = None,
) -> NodeLocalMoEResult:
    """Execute one complete shared+routed MoE layer across one eight-GCD node."""
    if topology is None:
        topology = assert_node_local_ep_group(group)
    top_indices, top_weights, counts, load_balance, z_loss = (
        fused_top3_router_autograd(
            x, router_weight, forced_indices=forced_top_indices))
    exchange = exchange_expert_assignments(
        x, top_indices, group=group, topology=topology)
    local_plan = build_local_expert_plan(
        exchange.received_x, exchange.received_local_expert)
    if expert_backend == "triton":
        local_packed_output = fused_packed_local_experts_autograd(
            local_plan.packed_x, local_plan.expert_offsets,
            local_gate_weight, local_up_weight, local_down_weight)
    elif expert_backend == "rocblas":
        local_packed_output = checkpointed_packed_local_experts_rocblas(
            local_plan.packed_x, local_plan.expert_offsets,
            local_gate_weight, local_up_weight, local_down_weight)
    elif expert_backend == "grouped":
        local_packed_output = checkpointed_packed_local_experts_grouped(
            local_plan.packed_x, local_plan.expert_offsets,
            local_gate_weight, local_up_weight, local_down_weight)
    else:
        raise ValueError(f"unknown expert backend: {expert_backend}")
    received_output = unpack_local_expert_rows(local_packed_output, local_plan)
    returned_output = return_expert_outputs(exchange, received_output, group=group)
    if expert_backend in {"rocblas", "grouped"}:
        shared_output = checkpointed_shared_expert_rocblas(
            x, shared_gate_weight, shared_up_weight, shared_down_weight)
    else:
        shared_output = fused_shared_expert_autograd(
            x, shared_gate_weight, shared_up_weight, shared_down_weight)
    output = fused_shared_top3_combine_autograd(
        returned_output, exchange.send_plan.assignment_to_send_row,
        top_weights, shared_output)
    return NodeLocalMoEResult(
        output=output,
        top_indices=top_indices,
        expert_counts=counts,
        load_balance_loss=load_balance,
        z_loss=z_loss,
        topology=topology,
    )


def return_expert_outputs(exchange: EPExchange, received_output: torch.Tensor,
                          *, group=None) -> torch.Tensor:
    """Return expert outputs to source ranks in original send-row order."""
    if (received_output.shape != exchange.received_x.shape or
            received_output.dtype != exchange.received_x.dtype or
            not received_output.is_contiguous()):
        raise ValueError("received expert output must match received assignment rows")
    returned = torch.empty_like(exchange.send_plan.send_x)
    return dist_nn.all_to_all_single(
        returned, received_output,
        output_split_sizes=list(exchange.send_splits),
        input_split_sizes=list(exchange.receive_splits),
        group=group,
    )
