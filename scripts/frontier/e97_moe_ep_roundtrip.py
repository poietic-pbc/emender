#!/usr/bin/env python3
"""One-node/eight-rank RCCL smoke for E97 expert assignment transport."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from ndm.e97_moe_ep import (
    assert_node_local_ep_group,
    exchange_expert_assignments,
    return_expert_outputs,
)


def main() -> None:
    local_rank = int(os.environ["SLURM_LOCALID"])
    # Slurm --gpus-per-task=1 exposes exactly one process-local device. The
    # physical lane remains SLURM_LOCALID, but its visible torch ordinal is 0.
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    try:
        topology = assert_node_local_ep_group()
        rank = dist.get_rank()
        tokens, dim = 17, 64
        x = (torch.arange(tokens * dim, device="cuda", dtype=torch.float32)
             .reshape(tokens, dim).add(rank * 10000).to(torch.bfloat16).contiguous())
        top = torch.empty((tokens, 3), device="cuda", dtype=torch.int32)
        for token in range(tokens):
            destinations = ((rank + token) % 8, (rank + token + 3) % 8,
                            (rank + token + 6) % 8)
            experts = [destination * 8 + ((token * 5 + slot) % 8)
                       for slot, destination in enumerate(destinations)]
            top[token] = torch.tensor(experts, device="cuda", dtype=torch.int32)
        exchange = exchange_expert_assignments(x, top, topology=topology)
        if exchange.received_local_expert.numel():
            lo = int(exchange.received_local_expert.min().item())
            hi = int(exchange.received_local_expert.max().item())
            if lo < 0 or hi >= 8:
                raise RuntimeError(f"received invalid local expert range {lo}..{hi}")
        returned = return_expert_outputs(exchange, exchange.received_x)
        torch.testing.assert_close(returned, exchange.send_plan.send_x, rtol=0, atol=0)
        if sum(exchange.send_splits) != tokens * 3:
            raise RuntimeError("send assignment total mismatch")
        stats = torch.tensor(
            [sum(exchange.send_splits), sum(exchange.receive_splits)],
            device="cuda", dtype=torch.int64)
        gathered = torch.empty((8, 2), device="cuda", dtype=torch.int64)
        dist.all_gather_into_tensor(gathered, stats)
        if rank == 0:
            print(json.dumps({
                "status": "pass",
                "backend": topology.backend,
                "hostname": topology.hostname,
                "global_ranks": topology.global_ranks,
                "local_ranks": topology.local_ranks,
                "per_rank_send_receive": gathered.cpu().tolist(),
                "expert_traffic_scope": "one-node-eight-rank-group-only",
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
