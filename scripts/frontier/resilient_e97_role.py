#!/usr/bin/env python3
"""Real manager/trainer entrypoints for the split resilient E97 launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _role_import_heartbeat() -> tuple[threading.Event, threading.Thread] | None:
    """Publish liveness, but no generation progress, during heavy imports."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"manager", "trainer"}:
        return None
    run_id = os.environ.get("RESILIENT_E97_RUN_ID")
    bulk_root = os.environ.get("RESILIENT_E97_BULK_ROOT")
    node_rank = os.environ.get("RESILIENT_E97_NODE_RANK", "0")
    if not run_id or not bulk_root:
        return None
    role = sys.argv[1]
    local_rank = os.environ.get("RESILIENT_E97_LOCAL_RANK")
    if role == "trainer" and local_rank is None:
        return None
    identity = (f"node-{node_rank}-manager" if role == "manager"
                else f"node-{node_rank}-trainer-{local_rank}")
    state = (Path(bulk_root) / run_id / f"node-{node_rank}" / "supervision" /
             f"{identity}.json")
    stop = threading.Event()

    def publish() -> None:
        state.parent.mkdir(parents=True, exist_ok=True)
        while not stop.is_set():
            now = time.time()
            temporary = state.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps({
                "identity": identity, "heartbeat_time": now, "progress_time": now,
                "generation": 0, "step": 0, "loss": None,
                "stage": "runtime_import", "bootstrap_pid": os.getpid(),
            }, sort_keys=True))
            os.replace(temporary, state)
            stop.wait(5)

    thread = threading.Thread(target=publish, name=f"{role}-import-heartbeat", daemon=True)
    thread.start()
    return stop, thread


_IMPORT_HEARTBEAT = _role_import_heartbeat()

import torch

from ndm.resilient_e97_roles import LocalFence, LocalTrainerSpool
from ndm.resilient_e97_runtime import (SplitManagerLoop, apply_delta, atomic_json,
                                       PINNED_STEP_1525000_SHA256, assert_node_local_path,
                                       finalize_checkpoint, flatten_tensors, heartbeat,
                                       outer_state_migration)
from ndm.resilient_node_quorum import GenerationFence
from ndm.resilient_node_transport import (DiskBucketSpool, NodeManagerClient,
                                           BoundedNodeManagerBulkStream,
                                           QuorumTransportServer, TransportConfig,
                                           decode_f64, encode_f64)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("role", choices=("manager", "trainer"))
    value.add_argument("--run-dir", required=True); value.add_argument("--run-id", required=True)
    value.add_argument("--generations", type=int, default=1)
    value.add_argument("--local-steps", type=int, default=40)
    value.add_argument("--local-quorum", type=int, default=6)
    value.add_argument("--node-count", type=int, default=1)
    value.add_argument("--global-quorum", type=int, default=1)
    value.add_argument("--coordinator-host", default="127.0.0.1")
    value.add_argument("--coordinator-port", type=int, default=29571)
    value.add_argument("--deadline-s", type=float, default=120.0)
    value.add_argument("--source-id", required=True); value.add_argument("--payload-id", required=True)
    value.add_argument("--code-id", default="unknown")
    value.add_argument("--coordinator-epoch", type=int, default=1)
    value.add_argument("--seed", default=""); value.add_argument("--train-args-json", default="")
    value.add_argument("--data", default=""); value.add_argument("--device", default="cuda:0")
    value.add_argument("--control", action="store_true")
    value.add_argument("--eta-outer", type=float, default=1.0)
    value.add_argument("--migration-policy", default="")
    value.add_argument("--bulk-root", default=os.environ.get("RESILIENT_E97_BULK_ROOT", "/tmp/resilient-e97"))
    value.add_argument("--max-spool-bytes", type=int, default=32 << 30)
    value.add_argument("--initial-generation", type=int, default=0)
    value.add_argument("--resume-handoff", default="")
    value.add_argument("--bulk-chunk-bytes", type=int, default=1 << 20)
    return value


def _fence(args, generation: int) -> LocalFence:
    return LocalFence(args.run_id, generation, 0, args.coordinator_epoch, args.payload_id)


def _latest_role_generation(control: Path, identity: str, args) -> int:
    path = control / "recovery" / f"{identity}.json"
    if not path.exists():
        return args.initial_generation
    value = json.loads(path.read_text())
    if (value.get("identity") != identity or value.get("run_id") != args.run_id
            or value.get("payload_id") != args.payload_id
            or value.get("source_id") != args.source_id
            or int(value.get("coordinator_epoch", -1)) != args.coordinator_epoch):
        raise ValueError("role recovery identity/fence mismatch")
    return max(args.initial_generation, int(value["generation"]))


def _publish_role_recovery(control: Path, identity: str, args, generation: int,
                           **extra: object) -> None:
    atomic_json(control / "recovery" / f"{identity}.json", {
        "schema": 1, "identity": identity, "run_id": args.run_id,
        "payload_id": args.payload_id, "source_id": args.source_id,
        "coordinator_epoch": args.coordinator_epoch, "generation": generation, **extra})


def _liveness_heartbeat(bulk: Path, identity: str, interval_s: float = 5.0):
    """Refresh liveness without disguising stalled generation progress."""
    state = bulk / "supervision" / f"{identity}.liveness.json"
    stop = threading.Event()

    def publish() -> None:
        while not stop.wait(interval_s):
            atomic_json(state, {"identity": identity, "heartbeat_time": time.time()})

    thread = threading.Thread(target=publish, name=f"{identity}-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def manager(args) -> int:
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    identity = f"node-{node}-manager"
    bulk = Path(args.bulk_root) / args.run_id / f"node-{node}"
    bulk = assert_node_local_path(bulk, run)
    if _IMPORT_HEARTBEAT is not None:
        stop, thread = _IMPORT_HEARTBEAT
        stop.set(); thread.join(10)
    spool = LocalTrainerSpool(bulk / "mailbox", args.max_spool_bytes)
    loop = SplitManagerLoop(spool, quorum=args.local_quorum, source_id=args.source_id,
                            aggregation_deadline_s=args.deadline_s)
    server = thread = None
    if args.node_count > 1 and node == 0:
        server = QuorumTransportServer(
            ("0.0.0.0", args.coordinator_port),
            TransportConfig(args.run_id, args.global_quorum, 1, args.bulk_chunk_bytes,
                            args.deadline_s,
                            args.deadline_s, args.deadline_s),
            run / "manager-network", args.coordinator_epoch)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    control = bulk / "control"
    target_generation = args.initial_generation + args.generations
    start_generation = _latest_role_generation(control, identity, args)
    liveness_stop, liveness_thread = _liveness_heartbeat(bulk, identity)
    try:
        for generation in range(start_generation, target_generation):
            fence = _fence(args, generation)
            heartbeat(bulk, identity, generation=generation, step=generation * args.local_steps,
                      loss=None, stage="collecting")
            if args.node_count == 1:
                result = loop.generation(fence)
            else:
                members, local_weight, local_shards = loop.manager.collect(
                    fence, deadline=time.monotonic() + args.deadline_s,
                    expected_source_id=args.source_id)
                client = NodeManagerClient(args.coordinator_host, args.coordinator_port,
                                           f"node-{node}", DiskBucketSpool(
                                               bulk / "network-spool", args.max_spool_bytes),
                                           timeout_s=args.deadline_s,
                                           max_bucket_bytes=args.bulk_chunk_bytes)
                stream = BoundedNodeManagerBulkStream(
                    client, max_chunk_bytes=args.bulk_chunk_bytes)
                header, aggregate = stream.exchange_chunks(
                    GenerationFence(args.run_id, generation, 0, args.coordinator_epoch),
                    tuple(encode_f64(shard.tolist()) for shard in local_shards),
                    weight=local_weight)
                global_shards = [torch.tensor(decode_f64(chunk), dtype=torch.float64)
                                 for chunk in aggregate]
                spool.publish_aggregate(fence, members, global_shards,
                                        weight=local_weight,
                                        source_id=args.source_id)
                for trainer_id in members:
                    spool.release_trainer(fence, trainer_id)
                result = {"members": members, "weight": local_weight,
                          "accepted_nodes": header["accepted_nodes"],
                          "network_high_water_bytes": stream.high_water_bytes}
            atomic_json(bulk / "control" / f"node-{node}-bulk-ownership.json", {
                "backend": "bounded-node-local-filesystem", "bulk_root": str(bulk),
                "shared_run_dir_is_bulk_path": bulk.is_relative_to(run),
                "max_bytes": args.max_spool_bytes,
                "high_water_bytes": spool.high_water_bytes,
                "post_release_bytes": spool.bytes_used})
            atomic_json(bulk / "control" / f"node-{node}-generation-{generation:08d}.json", {
                "generation": generation, "members": list(result["members"]),
                "weight": int(result["weight"]), "source_id": args.source_id,
                "payload_id": args.payload_id})
            _publish_role_recovery(control, identity, args, generation + 1,
                                   step=(generation + 1) * args.local_steps,
                                   membership=list(result["members"]))
            spool.prune_aggregates(keep_generations=2)
            heartbeat(bulk, identity, generation=generation + 1,
                      step=(generation + 1) * args.local_steps, loss=None, stage="published")
        if node == 0:
            proposal = bulk / "control" / "trainer-proposal.json"
            deadline = time.monotonic() + args.deadline_s
            while time.monotonic() < deadline and not proposal.exists():
                time.sleep(.02)
            if not proposal.exists():
                raise TimeoutError("checkpoint proposal deadline expired")
            value = json.loads(proposal.read_text())
            proposal_fence = LocalFence(**value["fence"])
            finalize_checkpoint(
                run, value["checkpoint"], run_id=args.run_id,
                generation=int(value["generation"]), step=int(value["step"]),
                async_chain=value["async_chain"], membership=value["membership"],
                fence=proposal_fence, source_id=args.source_id, code_id=args.code_id,
                outer_update_state=value["outer_update_state"], migration=value["migration"])
            proposal.unlink()
    finally:
        liveness_stop.set(); liveness_thread.join(10)
        if server is not None:
            server.shutdown(); server.server_close()
        if thread is not None: thread.join(2)
    return 0


def _load_real(args):
    from ndm.async_diloco_real import default_tiny_e97_train_args
    if not args.seed or not args.train_args_json:
        raise ValueError("real E97 trainer requires --seed and --train-args-json")
    overrides = json.loads(Path(args.train_args_json).read_text())
    overrides.update({"data": args.data, "optimizer": "schedulefree"})
    train_args = default_tiny_e97_train_args(**overrides)
    seed_sha = __import__("hashlib").sha256(Path(args.seed).read_bytes()).hexdigest()
    if seed_sha != PINNED_STEP_1525000_SHA256:
        raise ValueError("seed SHA256 does not match pinned step-1525000 checkpoint")
    payload = torch.load(args.seed, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError("verified seed lacks model or ScheduleFree inner optimizer state")
    state = {name: value.clone() for name, value in payload["model_state_dict"].items()
             if value.is_floating_point()}
    seed_meta = {"step": int(payload.get("step", -1)), "sha256": seed_sha,
                 "outer_update_state": payload.get("outer_update_state")}
    migration = outer_state_migration(
        seed_meta, policy=args.migration_policy,
        approved_config={"algorithm": "weighted-mean", "eta_outer": args.eta_outer})
    return train_args, state, payload["optimizer_state_dict"], int(payload.get("step", 0)), migration


def trainer(args) -> int:
    if args.local_steps != 40 and not args.control:
        raise ValueError("approved E97 runtime requires local_steps=40")
    run = Path(args.run_dir); node = int(os.environ.get("RESILIENT_E97_NODE_RANK", "0"))
    rank = int(os.environ.get("RESILIENT_E97_LOCAL_RANK", "0")); identity = f"node-{node}-trainer-{rank}"
    bulk = Path(args.bulk_root) / args.run_id / f"node-{node}"
    bulk = assert_node_local_path(bulk, run)
    # Loading and cloning the real E97 checkpoint can exceed the steady-state
    # heartbeat deadline when all eight local trainers start together. Keep
    # liveness independent from generation progress throughout bootstrap.
    _liveness_heartbeat(bulk, identity)
    if _IMPORT_HEARTBEAT is not None:
        stop, thread = _IMPORT_HEARTBEAT
        stop.set(); thread.join(10)
    spool = LocalTrainerSpool(bulk / "mailbox", args.max_spool_bytes)
    control = bulk / "control"
    target_generation = args.initial_generation + args.generations
    if args.control:
        state, optimizer_state, step, migration = {"weight": torch.tensor([0.0])}, {}, 0, {
            "status": "control_initialized", "policy": "control"}
        train_args = None
    else:
        train_args, state, optimizer_state, step, migration = _load_real(args)
    async_chain = [args.seed] if args.seed else []
    if args.resume_handoff:
        handoff = json.loads(Path(args.resume_handoff).read_text())
        checkpoint_path = Path(handoff["checkpoint"])
        if __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest() != handoff["checkpoint_sha256"]:
            raise ValueError("resume checkpoint checksum mismatch")
        resumed = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if (int(resumed["generation"]) != args.initial_generation
                or resumed["outer_update_state"] != handoff["outer_update_state"]):
            raise ValueError("resume generation/outer state does not match handoff")
        state = {name: value.clone() for name, value in resumed["model_state_dict"].items()}
        optimizer_state, step = resumed["optimizer_state_dict"], int(resumed["step"])
        migration = {"status": "restored", "state": resumed["outer_update_state"],
                     "policy": "new-harness-handoff"}
        async_chain = list(handoff.get("async_chain", ())) + [str(Path(args.resume_handoff).resolve())]
        if (handoff.get("run_id") != args.run_id or handoff.get("payload_id") != args.payload_id
                or handoff.get("source_id") != args.source_id
                or int(handoff["fence"]["coordinator_epoch"]) != args.coordinator_epoch
                or not handoff.get("finalized")):
            raise ValueError("resume handoff membership/identity/fence mismatch")
    recovery_manifest = control / "recovery" / f"{identity}.json"
    if recovery_manifest.exists():
        recovery = json.loads(recovery_manifest.read_text())
        start_generation = _latest_role_generation(control, identity, args)
        checkpoint_path = Path(recovery["checkpoint"])
        if __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest() != recovery["checkpoint_sha256"]:
            raise ValueError("role recovery checkpoint checksum mismatch")
        saved = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        if (saved["identity"] != identity or saved["run_id"] != args.run_id
                or saved["payload_id"] != args.payload_id
                or int(saved["coordinator_epoch"]) != args.coordinator_epoch
                or int(saved["generation"]) != start_generation):
            raise ValueError("role recovery checkpoint fence mismatch")
        state = {name: value.clone() for name, value in saved["model_state_dict"].items()}
        optimizer_state, migration = saved["optimizer_state_dict"], saved["migration"]
        step, async_chain = int(saved["step"]), list(saved["async_chain"])
    else:
        start_generation = args.initial_generation
    losses = []
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("requested", True))
    completed = start_generation
    for generation in range(start_generation, target_generation):
        if stop["requested"]:
            break
        # One deadline covers the complete generation, including real local
        # training, publication, quorum aggregation, and apply.  Starting a
        # fresh timeout only after the expensive 40-step train can let a live
        # but too-slow generation run until Slurm TERM without ever failing the
        # configured generation bound.
        generation_deadline = time.monotonic() + args.deadline_s
        if args.control:
            loss = 1.0 / (step + args.local_steps + rank + 1)
            delta = {"weight": torch.full_like(state["weight"], float(rank + 1))}
            tokens = rank + 1
        else:
            from ndm.async_diloco_real import RealAsyncWorkerSpec, _run_real_worker

            fence = _fence(args, generation)

            phase_log = bulk / "telemetry" / f"{identity}.jsonl"
            phase_log.parent.mkdir(parents=True, exist_ok=True)

            def training_phase(phase, details):
                record = {
                    "timestamp": time.time(), "monotonic_s": time.monotonic(),
                    "identity": identity, "generation": generation,
                    "phase": phase, **details,
                }
                with phase_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                heartbeat(
                    bulk, identity, generation=generation,
                    step=int(details.get("step", step)),
                    loss=details.get("loss"), stage=phase)

            def publish_trained_delta(base_state, model, tokens):
                heartbeat(bulk, identity, generation=generation, step=step, loss=None,
                          stage="streaming_delta")
                worker_state = model.state_dict()

                def shards():
                    chunk_elements = max(1, args.bulk_chunk_bytes // 8)
                    for name, base_tensor in sorted(base_state.items()):
                        worker_tensor = worker_state[name].detach().reshape(-1)
                        base_flat = base_tensor.detach().reshape(-1)
                        if worker_tensor.numel() != base_flat.numel():
                            raise ValueError(f"trainer state layout changed for {name}")
                        for offset in range(0, worker_tensor.numel(), chunk_elements):
                            end = min(offset + chunk_elements, worker_tensor.numel())
                            worker_chunk = worker_tensor[offset:end].to(
                                device="cpu", dtype=base_tensor.dtype)
                            yield worker_chunk.sub(base_flat[offset:end])

                spool.publish(fence, rank, shards(), weight=tokens,
                              source_id=args.source_id)

            def training_progress(local_step, metrics):
                if time.monotonic() >= generation_deadline:
                    raise TimeoutError(
                        f"generation {generation} deadline exceeded during local training "
                        f"at step {local_step}/{args.local_steps}")
                heartbeat(
                    bulk, identity, generation=generation,
                    step=step + local_step, loss=float(metrics["loss"]),
                    stage="training")

            report = _run_real_worker(
                run_id=args.run_id, generation=generation, base_state=state,
                train_args=train_args,
                spec=RealAsyncWorkerSpec(identity, f"node-{node}", args.device,
                                         args.local_steps, rank),
                synthetic_token_stream=False, synthetic_vocab_size=256,
                optimizer_state_dict=optimizer_state, consume_optimizer_state=True,
                progress_callback=training_progress,
                delta_consumer=publish_trained_delta,
                phase_callback=training_phase)
            if report.update is None:
                raise RuntimeError(report.error or "real E97 trainer produced no update")
            delta = report.update.delta
            optimizer_state = report.optimizer_state_dict or {}
            tokens, loss = report.tokens, float(report.losses[-1])
        fence = _fence(args, generation)
        if args.control:
            heartbeat(bulk, identity, generation=generation, step=step, loss=loss,
                      stage="streaming_delta")
            spool.publish(fence, rank, flatten_tensors(
                delta, chunk_elements=max(1, args.bulk_chunk_bytes // 8)), weight=tokens,
                          source_id=args.source_id)
        del delta
        heartbeat(bulk, identity, generation=generation, step=step, loss=loss, stage="submitted")
        manifest, aggregate = spool.wait_aggregate(
            fence, deadline=generation_deadline,
            expected_source_id=args.source_id)
        spool.release_trainer(fence, rank)
        state = apply_delta(state, aggregate, eta_outer=args.eta_outer)
        step += args.local_steps; losses.append(loss)
        completed = generation + 1
        recovery_checkpoint = bulk / "recovery" / identity / f"generation-{completed:08d}.pt"
        recovery_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = recovery_checkpoint.with_suffix(".tmp")
        torch.save({"identity": identity, "model_state_dict": state,
                    "optimizer_state_dict": optimizer_state, "migration": migration,
                    "outer_update_state": migration.get("state", {}), "step": step,
                    "generation": completed, "run_id": args.run_id,
                    "source_id": args.source_id, "payload_id": args.payload_id,
                    "coordinator_epoch": args.coordinator_epoch,
                    "membership": manifest["members"], "fence": fence.__dict__,
                    "async_chain": async_chain}, temporary)
        os.replace(temporary, recovery_checkpoint)
        _publish_role_recovery(
            control, identity, args, completed, step=step,
            checkpoint=str(recovery_checkpoint),
            checkpoint_sha256=__import__("hashlib").sha256(
                recovery_checkpoint.read_bytes()).hexdigest(),
            membership=manifest["members"], fence=fence.__dict__)
        heartbeat(bulk, identity, generation=generation + 1, step=step, loss=loss, stage="applied")
    if node == 0 and rank == 0 and completed > start_generation:
        checkpoint = run / "checkpoints" / f"generation-{completed:08d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model_state_dict": state, "optimizer_state_dict": optimizer_state,
                    "outer_update_state": migration.get("state", {}), "step": step,
                    "generation": completed, "run_id": args.run_id,
                    "source_id": args.source_id, "payload_id": args.payload_id,
                    "coordinator_epoch": args.coordinator_epoch,
                    "loss": losses[-1]}, temporary); os.replace(temporary, checkpoint)
        fence = _fence(args, completed - 1)
        aggregate_manifest, _ = spool.wait_aggregate(
            fence, deadline=time.monotonic() + args.deadline_s, expected_source_id=args.source_id)
        atomic_json(bulk / "control" / "trainer-proposal.json", {
            "checkpoint": str(checkpoint.resolve()), "generation": completed, "step": step,
            "async_chain": async_chain,
            "membership": aggregate_manifest["members"], "fence": fence.__dict__,
            "outer_update_state": migration.get("state", {}), "migration": migration})
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.generations <= 0 or args.deadline_s <= 0:
        raise ValueError("generations and deadlines must be positive")
    return manager(args) if args.role == "manager" else trainer(args)


if __name__ == "__main__":
    raise SystemExit(main())
