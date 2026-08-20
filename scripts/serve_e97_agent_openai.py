#!/usr/bin/env python3
"""Serve dense recurrent E97 through bounded OpenAI Chat Completions."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ndm.e97 import load_e97_checkpoint
from ndm.e97_agent_protocol import DENSE_AGENT_CLI_SYSTEM, DENSE_AGENT_V1_SYSTEM, DENSE_AGENT_V2_SYSTEM
from ndm.e97_agent_server import (
    AgentCompletionService,
    TorchE97AgentEngine,
    run_openai_server,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--model-id", default="e97-dense-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--api-key", default=os.environ.get("EMENDER_AGENT_API_KEY"))
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--max-body-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--ingest-mode", choices=("tokenwise", "segment"), default="tokenwise")
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument("--v1-canonical-system", action="store_true")
    system_group.add_argument("--v2-canonical-system", action="store_true")
    system_group.add_argument("--cli-canonical-system", action="store_true")
    parser.add_argument("--trace-generated-errors", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda":
        torch.cuda.set_device(0)
    loaded = load_e97_checkpoint(
        args.checkpoint,
        args_json=args.args_json,
        device=args.device,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        weight_mode="saved",
        use_triton=args.device == "cuda",
        mmap=True,
    )
    engine = TorchE97AgentEngine(loaded, ingest_mode=args.ingest_mode)
    service = AgentCompletionService(
        engine,
        model_id=args.model_id,
        max_output_tokens=args.max_output_tokens,
        max_sessions=args.max_sessions,
        trace_generated_errors=args.trace_generated_errors,
        system_prompt_override=(
            DENSE_AGENT_V1_SYSTEM if args.v1_canonical_system else
            DENSE_AGENT_V2_SYSTEM if args.v2_canonical_system else
            DENSE_AGENT_CLI_SYSTEM if args.cli_canonical_system else None
        ),
        require_tool_call=args.v2_canonical_system or args.cli_canonical_system,
    )
    print(
        f"serving model={args.model_id} checkpoint={loaded.checkpoint_path} "
        f"address={args.host}:{args.port} max_sessions={args.max_sessions} "
        f"ingest_mode={args.ingest_mode}",
        flush=True,
    )
    run_openai_server(
        service,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        max_body_bytes=args.max_body_bytes,
    )


if __name__ == "__main__":
    main()
