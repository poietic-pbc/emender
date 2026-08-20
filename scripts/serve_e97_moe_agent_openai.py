#!/usr/bin/env python3
"""Serve one node-local E97 MoE through a synchronized eight-rank Pi endpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from ndm.e97 import load_e97_checkpoint
from ndm.e97_agent_protocol import DENSE_AGENT_CLI_DIRECT_SYSTEM, DENSE_AGENT_CLI_SYSTEM
from ndm.e97_agent_server import AgentCompletionService, run_openai_server
from ndm.e97_moe_agent_server import TorchE97MoEAgentEngine
from ndm.e97_moe_checkpoint import load_node_sharded_model
from ndm.e97_moe_ep import assert_node_local_ep_group, create_moe_process_groups
from ndm.models.e97_moe import E97MoEConfig, convert_e97_ffns_to_node_local_moe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--seed-args-json", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--model-id", default="e97-dense-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--api-key", default=os.environ.get("EMENDER_AGENT_API_KEY"))
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--ingest-mode", choices=("tokenwise", "segment"), default="segment")
    parser.add_argument("--expert-backend", choices=("rocblas", "triton", "grouped"), default="rocblas")
    parser.add_argument("--verify-sha256", action=argparse.BooleanOptionalAction, default=True)
    systems = parser.add_mutually_exclusive_group(required=True)
    systems.add_argument("--cli-canonical-system", action="store_true")
    systems.add_argument("--cli-direct-canonical-system", action="store_true")
    parser.add_argument("--trace-generated-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://")
    engine = None
    try:
        if dist.get_world_size() != 8:
            raise RuntimeError("MoE agent server requires exactly eight ranks")
        groups = create_moe_process_groups()
        if groups.local_rank != local_rank or groups.node_count != 1:
            raise RuntimeError("MoE agent server requires one contiguous eight-rank node")
        assert_node_local_ep_group(groups.node_group)
        loaded = load_e97_checkpoint(
            args.seed_checkpoint,
            args_json=args.seed_args_json,
            device="cuda",
            dtype=torch.bfloat16,
            weight_mode="train",
            use_triton=True,
            mmap=True,
        )
        model = loaded.model
        model.gradient_checkpointing = False
        for module in model.modules():
            if hasattr(module, "checkpoint_interval"):
                module.checkpoint_interval = 16
                module.projection_chunk_size = 2048
        convert_e97_ffns_to_node_local_moe(
            model,
            E97MoEConfig(
                hidden_dim=8832,
                routed_experts=64,
                shared_experts=1,
                top_k=3,
                expert_parallel_size=8,
                expert_backend=args.expert_backend,
            ),
            local_expert_rank=local_rank,
            expert_group=groups.node_group,
        )
        model.eval()
        manifest = load_node_sharded_model(
            args.generation,
            model,
            node_group=groups.node_group,
            verify_sha256=args.verify_sha256,
        )
        # Bind cache identity and diagnostics to the actual sharded generation.
        loaded.checkpoint_path = args.generation.resolve()
        dist.barrier(group=groups.node_group)
        engine = TorchE97MoEAgentEngine(
            loaded,
            node_group=groups.node_group,
            ingest_mode=args.ingest_mode,
        )
        if not engine.is_coordinator:
            engine.worker_loop()
            return
        system = DENSE_AGENT_CLI_DIRECT_SYSTEM if args.cli_direct_canonical_system else DENSE_AGENT_CLI_SYSTEM
        service = AgentCompletionService(
            engine,
            model_id=args.model_id,
            max_output_tokens=args.max_output_tokens,
            max_sessions=args.max_sessions,
            trace_generated_errors=args.trace_generated_errors,
            system_prompt_override=system,
            require_tool_call=True,
        )
        print(
            f"serving synchronized MoE model={args.model_id} generation={args.generation} "
            f"step={manifest['step']} address={args.host}:{args.port} ingest_mode={args.ingest_mode}",
            flush=True,
        )
        run_openai_server(
            service,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            max_body_bytes=args.max_body_bytes,
        )
    finally:
        # Normal HTTP shutdown releases workers. Slurm cancellation kills the
        # entire step, so no rank is allowed to continue independently.
        if engine is not None and engine.is_coordinator:
            try:
                engine.stop_workers()
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
