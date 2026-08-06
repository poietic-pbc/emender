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

from .triton.e97_moe_ep import EP_SIZE, EPSendPlan, build_ep_send_plan


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
    dist.all_to_all_single(
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


def return_expert_outputs(exchange: EPExchange, received_output: torch.Tensor,
                          *, group=None) -> torch.Tensor:
    """Return expert outputs to source ranks in original send-row order."""
    if (received_output.shape != exchange.received_x.shape or
            received_output.dtype != exchange.received_x.dtype or
            not received_output.is_contiguous()):
        raise ValueError("received expert output must match received assignment rows")
    returned = torch.empty_like(exchange.send_plan.send_x)
    dist.all_to_all_single(
        returned, received_output,
        output_split_sizes=list(exchange.send_splits),
        input_split_sizes=list(exchange.receive_splits),
        group=group,
    )
    return returned
