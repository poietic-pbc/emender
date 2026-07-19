#!/usr/bin/env python3
"""Qualify the native Compute Pool v1 hyperscale-local adapter without Slurm.

The gate runs one exclusive allocation, dynamic late/disappear/rejoin flows,
and repeated owner failure/restart attempts against the same native v1 ABI and
pool metadata protocol.  It intentionally never invokes a scheduler command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.hyperscale_local_adapter import (  # noqa: E402
    HyperscaleLocalAdapter,
    HyperscaleLocalConfig,
    LocalContribution,
    OwnerUnavailable,
)


MIN_FAILURE_RESTART_CYCLES = 4
RSS_PLATEAU_TOLERANCE_BYTES = 256 << 20


def _resources(adapter: HyperscaleLocalAdapter | None = None) -> dict[str, int]:
    pids = (os.getpid(),) + (() if adapter is None else adapter.native_process_ids)
    rss_kib = fds = threads = 0
    observed = 0
    for pid in pids:
        root = Path(f"/proc/{pid}")
        try:
            fds += len(tuple((root / "fd").iterdir()))
            threads += len(tuple((root / "task").iterdir()))
            for line in (root / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    rss_kib += int(line.split()[1])
                    break
            observed += 1
        except FileNotFoundError:
            raise RuntimeError(f"native host-agent disappeared during resource sample: {pid}")
    return {
        "processes": observed,
        "fds": fds,
        "threads": threads,
        "rss_bytes": rss_kib * 1024,
    }


def _contribution(adapter: HyperscaleLocalAdapter, worker_id: str,
                  generation: int, *, elements: int, seq: int
                  ) -> LocalContribution:
    incarnation, _ = adapter.workers[worker_id]
    worker_number = int(worker_id.rsplit("-", 1)[1])
    values = tuple(float(generation * 16 + worker_number * 2 + offset)
                   for offset in range(elements))
    return LocalContribution.create(
        worker_id, incarnation, seq, worker_number + 1, values)


def _sync_all(adapter: HyperscaleLocalAdapter, generation: int) -> None:
    for worker_id in tuple(sorted(adapter.workers)):
        adapter.sync_worker(worker_id, generation=generation)


def _assert_terminal_bounds(records: tuple[object, ...]) -> None:
    for record in records:
        local = record.local
        transport = record.transport
        forbidden = {
            "local_shared": int(local["shared_bytes_current"]),
            "local_mapped": int(local["mapped_bytes_current"]),
            "disk_replay_bytes": int(local["disk_replay_bytes"]),
            "trainer_spool_bytes": int(local["trainer_spool_bytes"]),
            "trainer_spool_files": int(local["trainer_spool_files"]),
            "python_dense_socket_bytes": int(local["python_dense_socket_bytes"]),
            "handoff_full_copy_bytes": int(local["handoff_full_copy_bytes"]),
            "transport_in_flight": int(transport["in_flight_bytes"]),
            "transport_retained": int(transport["retained_bytes"]),
        }
        if any(forbidden.values()):
            raise RuntimeError(f"native departure retained forbidden resources: {forbidden}")


def _result_record(result: object) -> dict[str, object]:
    return {
        "generation": result.generation,
        "attempt": result.attempt,
        "fence": result.fence,
        "ready_snapshot": result.ready_snapshot,
        "accepted_identities": result.accepted_identities,
        "accepted_tokens": result.accepted_tokens,
        "receipt_statuses": [value["status"] for value in result.receipts],
        "owner_by_shard": dict(result.owner_by_shard),
        "checkpoint_sha256": result.checkpoint_sha256,
        "manifest_sha256": result.manifest_sha256,
    }


def run_gate(*, build_manifest: str | Path, output_root: str | Path,
             failure_restart_cycles: int = 12, elements: int = 12
             ) -> Mapping[str, object]:
    if failure_restart_cycles < MIN_FAILURE_RESTART_CYCLES:
        raise ValueError(
            f"repeated failure qualification requires at least "
            f"{MIN_FAILURE_RESTART_CYCLES} restart cycles")
    if elements < 6:
        raise ValueError("adapter gate needs enough elements for three shard owners")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    before = _resources()
    config = HyperscaleLocalConfig(
        run_id="hyperscale-local-v1", allocation_id="local-allocation-a",
        allocation_incarnation="allocation-a-boot-1",
        protocol_id="compute-pool-v1-native-abi-v1",
        config_id="hyperscale-local-qualified-v1",
        control_db=output_root / "control" / "pool.sqlite3",
        evidence_root=output_root / "winner",
        build_manifest=Path(build_manifest).resolve(), source_root=ROOT,
        lease_ttl_s=300.0, session_deadline_s=240.0,
        q_min=2, t_min=3, payload_max=1 << 20,
        resident_limit_bytes=64 << 20,
    )
    adapter = HyperscaleLocalAdapter.try_start(config)
    if adapter is None:
        raise RuntimeError("first local allocation unexpectedly lost its lease")
    committed = []
    failure_attempts = []
    samples = []
    winner_fence = adapter.lease.fence
    loser_native_starts = 0

    def losing_factory(**_kwargs: object):
        nonlocal loser_native_starts
        loser_native_starts += 1
        raise AssertionError("exclusive lease loser entered the native data plane")

    loser = HyperscaleLocalAdapter.try_start(
        replace(
            config, allocation_id="local-allocation-b",
            allocation_incarnation="allocation-b-boot-1",
            # Admission must reject before inspecting this deliberately absent
            # artifact or invoking the native session factory.
            build_manifest=output_root / "loser-must-not-read-manifest.json",
            evidence_root=output_root / "loser-must-remain-empty"),
        session_factory=losing_factory)
    if loser is not None or loser_native_starts != 0:
        raise RuntimeError("exclusive lease loser performed model/data-plane work")
    if (output_root / "loser-must-remain-empty").exists():
        raise RuntimeError("exclusive lease loser mutated run/data-plane evidence")

    try:
        adapter.join_worker("node-0", "node-0-boot-0", generation=0)
        adapter.join_worker("node-1", "node-1-boot-0", generation=0)
        opened = adapter.open_generation(0, 1)
        adapter.join_worker("node-2", "node-2-late-boot-0", generation=0)
        late = adapter.commit_generation(opened, tuple(
            _contribution(adapter, worker_id, 0, elements=elements, seq=1)
            for worker_id in ("node-0", "node-1", "node-2")))
        if late.ready_snapshot != (
                ("node-0", "node-0-boot-0"),
                ("node-1", "node-1-boot-0")):
            raise RuntimeError("late worker changed the already-open READY snapshot")
        if [value["status"] for value in late.receipts] != [
                "accepted", "accepted", "rejected_not_ready"]:
            raise RuntimeError("late contribution did not receive the normative receipt")
        committed.append(_result_record(late))

        _sync_all(adapter, 1)
        departed = adapter.remove_worker("node-1", reason="simulated_host_loss")
        _assert_terminal_bounds((departed,))
        continued = adapter.commit_generation(
            adapter.open_generation(1, 1), tuple(
                _contribution(adapter, worker_id, 1, elements=elements, seq=2)
                for worker_id in ("node-0", "node-2")))
        if {worker for worker, _ in continued.ready_snapshot} != {"node-0", "node-2"}:
            raise RuntimeError("disappeared worker remained in the active world")
        committed.append(_result_record(continued))

        adapter.join_worker("node-1", "node-1-rejoin-1", generation=2)
        _sync_all(adapter, 2)
        rejoined = adapter.commit_generation(
            adapter.open_generation(2, 1), tuple(
                _contribution(adapter, worker_id, 2, elements=elements, seq=3)
                for worker_id in sorted(adapter.workers)))
        if ("node-1", "node-1-rejoin-1") not in rejoined.ready_snapshot:
            raise RuntimeError("new-incarnation rejoin was not admitted")
        committed.append(_result_record(rejoined))
        baseline = _resources(adapter)

        for cycle in range(failure_restart_cycles):
            generation = 3 + cycle
            _sync_all(adapter, generation)
            failed_open = adapter.open_generation(generation, 1)
            victim = "node-1" if cycle % 2 == 0 else "node-2"
            old_incarnation = adapter.workers[victim][0]
            departure = adapter.remove_worker(
                victim, reason=f"simulated_owner_loss_cycle_{cycle}")
            _assert_terminal_bounds((departure,))
            try:
                adapter.commit_generation(failed_open, tuple(
                    _contribution(adapter, worker_id, generation,
                                  elements=elements, seq=10 + cycle)
                    for worker_id in sorted(adapter.workers)))
            except OwnerUnavailable:
                pass
            else:
                raise RuntimeError("missing frozen owner did not fail closed")
            if adapter.store.read_publication(
                    config.run_id, "commit", f"generation-{generation:08d}") is not None:
                raise RuntimeError("failed owner attempt published a partial commit")
            new_incarnation = f"{victim}-restart-{cycle + 2}"
            adapter.join_worker(victim, new_incarnation, generation=generation)
            if new_incarnation == old_incarnation:
                raise RuntimeError("restart reused a superseded incarnation")
            retried = adapter.commit_generation(
                adapter.open_generation(generation, 2), tuple(
                    _contribution(adapter, worker_id, generation,
                                  elements=elements, seq=100 + cycle)
                    for worker_id in sorted(adapter.workers)))
            committed.append(_result_record(retried))
            failure_attempts.append({
                "generation": generation, "failed_attempt": 1,
                "committed_attempt": retried.attempt, "victim": victim,
                "old_incarnation": old_incarnation,
                "new_incarnation": new_incarnation,
            })
            samples.append(_resources(adapter))
            adapter.renew_lease()

        final_records = adapter.close()
        _assert_terminal_bounds(final_records)
    except BaseException:
        try:
            adapter.close()
        except Exception:
            pass
        raise

    after = _resources()
    window = max(1, len(samples) // 3)
    first_rss = max(value["rss_bytes"] for value in samples[:window])
    last_rss = max(value["rss_bytes"] for value in samples[-window:])
    fd_values = [baseline["fds"], *(value["fds"] for value in samples)]
    thread_values = [baseline["threads"], *(value["threads"] for value in samples)]
    resource_checks = {
        "fixed_live_process_bound": all(value["processes"] == 4 for value in samples),
        "fd_plateau_bounded": max(fd_values) - min(fd_values) <= 12,
        "thread_plateau_bounded": max(thread_values) - min(thread_values) <= 3,
        "rss_plateau_bounded": max(0, last_rss - first_rss)
        <= RSS_PLATEAU_TOLERANCE_BYTES,
        "close_releases_fds": after["fds"] <= before["fds"] + 2,
        "close_releases_threads": after["threads"] <= before["threads"],
    }
    if not all(resource_checks.values()):
        raise RuntimeError(
            "native restart resources were not bounded: "
            f"checks={resource_checks} before={before} baseline={baseline} "
            f"samples={samples} after={after}")

    successor_config = replace(
        config, allocation_id="local-allocation-successor",
        allocation_incarnation="allocation-successor-boot-1",
        evidence_root=output_root / "successor")
    successor = HyperscaleLocalAdapter.try_start(successor_config)
    if successor is None:
        raise RuntimeError("released run lease did not admit a successor")
    successor_fence = successor.lease.fence
    successor.close()
    if successor_fence != winner_fence + 1:
        raise RuntimeError("successor allocation did not receive a strictly newer fence")

    manifest = json.loads(Path(build_manifest).read_text(encoding="utf-8"))
    return {
        "schema": "emender-native-pool-hyperscale-local-gate-v1",
        "status": "passed",
        "backend": "native-test",
        "provider": os.environ.get("NDP_TEST_PROVIDER", "tcp;ofi_rxm"),
        "production_provider": False,
        "frontier_native_cxi_gate_unchanged": True,
        "source_commit": manifest["source_commit"],
        "artifact_bundle_sha256": manifest["bundle_sha256"],
        "exclusive_lease": {
            "winner_fence": winner_fence, "loser_exit_status": 0,
            "loser_native_starts": loser_native_starts,
            "loser_model_loads": 0, "loser_data_plane_bytes": 0,
            "successor_fence": successor_fence,
        },
        "membership": {
            "late_join_excluded_from_open_generation": True,
            "disappearance_continued_without_fixed_world": True,
            "rejoin_required_new_incarnation": True,
            "launched_world_size": None,
        },
        "protocol_contracts": {
            "fence": "SQLiteFencedControlStore/AllocationLease",
            "membership": "PoolControlServer/PeerMembership",
            "contribution": "ContributionIdentity/GenerationAdmission",
            "owner": "TensorLayout.owner/NativeManagerSession",
            "receipt": "ContributionReceipt plus native result roots",
            "checkpoint": "native proposal plus atomic commit/checkpoint/latest bundle",
        },
        "failure_restart_cycles": failure_restart_cycles,
        "failed_attempts_without_publication": failure_attempts,
        "committed_generations": committed,
        "minimum_progress_floor": {"q_min": config.q_min, "t_min": config.t_min},
        "resource_bounds": {
            "before": before, "baseline": baseline, "after": after,
            "fd_min": min(fd_values), "fd_max": max(fd_values),
            "thread_min": min(thread_values), "thread_max": max(thread_values),
            "rss_first_window_max_bytes": first_rss,
            "rss_last_window_max_bytes": last_rss,
            "rss_plateau_tolerance_bytes": RSS_PLATEAU_TOLERANCE_BYTES,
            "checks": resource_checks,
        },
        "native_departures": [asdict(value) for value in final_records],
        "forbidden_path_counters": {
            "python_dense_socket_bytes": 0, "trainer_spool_bytes": 0,
            "handoff_full_copy_bytes": 0, "slurm_jobs_submitted": 0,
        },
        "requirements": {
            "compute_pool": [
                "R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08",
                "R09", "R10", "R11", "R13", "R14", "R15"],
            "native_dataplane": [
                "NDP01", "NDP02", "NDP03", "NDP04", "NDP05", "NDP06",
                "NDP07", "NDP08", "NDP09", "NDP10", "NDP11", "NDP12",
                "NDP13", "NDP14", "NDP15", "NDP16"],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-restart-cycles", type=int, default=12)
    parser.add_argument("--elements", type=int, default=12)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_gate(
        build_manifest=args.build_manifest, output_root=args.output_root,
        failure_restart_cycles=args.failure_restart_cycles,
        elements=args.elements)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
