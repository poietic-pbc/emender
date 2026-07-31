#!/usr/bin/env python3
"""Run the bounded, non-Slurm native data-plane local qualification.

The gate deliberately uses a small resident payload for repeated native
generations, but computes the E97/Frontier layout and admission bounds with
the exact v1 integers.  Each stress worker is a fresh spawned process so the
result also exercises library startup and process-local handle isolation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.native_dataplane import Client, Command, DType, NativeLibrary


SCHEMA = "emender-native-dataplane-local-validation-v1"
E97_ELEMENTS = 688_346_312
WIRE_ELEMENT_BYTES = 8
E97_LAYOUT_BYTES = 5_506_770_496
PAYLOAD_MAX_BYTES = 64 * 1024 * 1024
FRAME_HEADER_BYTES = 320
E97_SHARD_COUNT = 83
FRONTIER_NODES = 256
TRAINERS_PER_NODE = 8
REGISTERED_SLOTS_PER_DIRECTION = 4
RECEIPT_BYTES = 128
FIXED_SERVICE_ALLOWANCE_BYTES = 64 * 1024 * 1024
NATIVE_LOCAL_DEFAULT_LIMIT_BYTES = 16 * 1024**3
RSS_PLATEAU_TOLERANCE_BYTES = 4 * 1024 * 1024


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil division requires a nonnegative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


def full_layout_accounting() -> dict[str, Any]:
    """Return and internally verify the exact NDP v1 E97/Frontier bounds."""
    layout_bytes = E97_ELEMENTS * WIRE_ELEMENT_BYTES
    shard_count = _ceil_div(layout_bytes, PAYLOAD_MAX_BYTES)
    full_shards = layout_bytes // PAYLOAD_MAX_BYTES
    last_shard_bytes = layout_bytes - (shard_count - 1) * PAYLOAD_MAX_BYTES
    data_frames_per_direction = FRONTIER_NODES * shard_count
    logical_bytes_per_direction = FRONTIER_NODES * layout_bytes
    header_bytes_per_direction = data_frames_per_direction * FRAME_HEADER_BYTES
    receipt_ledger_bytes = FRONTIER_NODES * shard_count * RECEIPT_BYTES
    registered_slot_pool_bytes = (
        2 * REGISTERED_SLOTS_PER_DIRECTION
        * (PAYLOAD_MAX_BYTES + FRAME_HEADER_BYTES)
    )

    def owner_and_resident(owner_count: int) -> tuple[int, int]:
        # The formula is intentionally the normative conservative bound, not
        # an estimate of a hoped-for allocation/reuse pattern.
        owner_bytes = _ceil_div(layout_bytes, owner_count) + PAYLOAD_MAX_BYTES
        resident = (
            2 * layout_bytes
            + owner_bytes
            + registered_slot_pool_bytes
            + receipt_ledger_bytes
            + FIXED_SERVICE_ALLOWANCE_BYTES
        )
        return owner_bytes, resident

    two_owner_bytes, two_owner_resident = owner_and_resident(2)
    all_owner_bytes, all_owner_resident = owner_and_resident(FRONTIER_NODES)
    trainer_f32_lane_bytes = E97_ELEMENTS * 4
    eight_trainer_lane_bytes = trainer_f32_lane_bytes * TRAINERS_PER_NODE

    accounting = {
        "total_elements": E97_ELEMENTS,
        "wire_element_bytes": WIRE_ELEMENT_BYTES,
        "layout_bytes": layout_bytes,
        "payload_max_bytes": PAYLOAD_MAX_BYTES,
        "shard_count": shard_count,
        "full_shards": full_shards,
        "last_shard_bytes": last_shard_bytes,
        "frontier_nodes": FRONTIER_NODES,
        "trainers_per_node": TRAINERS_PER_NODE,
        "trainer_lanes_cluster": FRONTIER_NODES * TRAINERS_PER_NODE,
        "logical_contribution_bytes": logical_bytes_per_direction,
        "logical_redistribution_bytes": logical_bytes_per_direction,
        "total_logical_dense_bytes": 2 * logical_bytes_per_direction,
        "data_frames_per_direction": data_frames_per_direction,
        "total_data_frames": 2 * data_frames_per_direction,
        "header_bytes_per_direction": header_bytes_per_direction,
        "total_header_bytes": 2 * header_bytes_per_direction,
        "routes_per_service": FRONTIER_NODES - 1,
        "directed_routes_cluster": FRONTIER_NODES * (FRONTIER_NODES - 1),
        "receipt_ledger_bytes": receipt_ledger_bytes,
        "registered_slot_pool_bytes": registered_slot_pool_bytes,
        "owner_assignment_bound_two_owners_bytes": two_owner_bytes,
        "resident_bound_two_owners_bytes": two_owner_resident,
        "owner_assignment_bound_256_owners_bytes": all_owner_bytes,
        "resident_bound_256_owners_bytes": all_owner_resident,
        "trainer_f32_lane_bytes": trainer_f32_lane_bytes,
        "eight_trainer_lane_bytes": eight_trainer_lane_bytes,
        "native_local_default_limit_bytes": NATIVE_LOCAL_DEFAULT_LIMIT_BYTES,
        "eight_lanes_fit_local_default": (
            eight_trainer_lane_bytes <= NATIVE_LOCAL_DEFAULT_LIMIT_BYTES
        ),
    }
    expected = {
        "layout_bytes": 5_506_770_496,
        "shard_count": 83,
        "full_shards": 82,
        "last_shard_bytes": 3_843_648,
        "trainer_lanes_cluster": 2_048,
        "logical_contribution_bytes": 1_409_733_246_976,
        "logical_redistribution_bytes": 1_409_733_246_976,
        "total_logical_dense_bytes": 2_819_466_493_952,
        "data_frames_per_direction": 21_248,
        "total_data_frames": 42_496,
        "header_bytes_per_direction": 6_799_360,
        "total_header_bytes": 13_598_720,
        "routes_per_service": 255,
        "directed_routes_cluster": 65_280,
        "receipt_ledger_bytes": 2_719_744,
        "registered_slot_pool_bytes": 536_873_472,
        "owner_assignment_bound_two_owners_bytes": 2_820_494_112,
        "resident_bound_two_owners_bytes": 14_440_737_184,
        "owner_assignment_bound_256_owners_bytes": 88_619_687,
        "resident_bound_256_owners_bytes": 11_708_862_759,
        "trainer_f32_lane_bytes": 2_753_385_248,
        "eight_trainer_lane_bytes": 22_027_081_984,
        "eight_lanes_fit_local_default": False,
    }
    mismatches = {
        key: (accounting[key], value)
        for key, value in expected.items()
        if accounting[key] != value
    }
    if mismatches:
        raise AssertionError(f"exact E97/Frontier accounting mismatch: {mismatches}")
    if E97_LAYOUT_BYTES != layout_bytes:
        raise AssertionError("declared E97 layout byte constant is inconsistent")
    return accounting


def _key(worker: int, lane: int) -> bytes:
    return hashlib.sha256(f"native-local-v1:{worker}:{lane}".encode()).digest()[:16]


def _resource_sample() -> dict[str, int]:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    rss_kib = None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
            break
    if rss_kib is None:
        raise RuntimeError("/proc/self/status did not report VmRSS")
    return {
        "fds": len(tuple(Path("/proc/self/fd").iterdir())),
        "threads": len(tuple(Path("/proc/self/task").iterdir())),
        "rss_bytes": rss_kib * 1024,
    }


def _run_generation(
    client: Client,
    *,
    generation: int,
    worker: int,
    inputs: tuple[np.ndarray, np.ndarray],
    expected: np.ndarray,
) -> str:
    client.install_generation(generation, attempt=1, owner_epoch=1, deadline_s=30).close()
    submissions = []
    for lane, (values, weight) in enumerate(zip(inputs, (3, 1_000_003), strict=True)):
        with client.allocate(dtype=DType.F32) as source:
            with source.mapped(DType.F32, write=True) as target:
                target[:] = values
            source.seal()
            submissions.append(client.submit(
                source,
                trainer_key=_key(worker, lane),
                trainer_incarnation=_key(worker, lane + 16),
                submission_seq=generation + 1,
                weight=weight,
            ))
    client.control(Command.FREEZE).close()
    result_op = client.control(Command.FINALIZE_OWNERS)
    try:
        with client.result_view(result_op) as view:
            with view.mapped(DType.F32) as result:
                if not np.array_equal(result, expected):
                    differing = int(np.count_nonzero(result != expected))
                    raise AssertionError(
                        f"generation {generation} differs from exact reference at "
                        f"{differing} elements"
                    )
                result_sha256 = hashlib.sha256(result.tobytes(order="C")).hexdigest()
        for submission in submissions:
            submission.close()
        client.control(Command.COMMIT).close()
    finally:
        for submission in submissions:
            submission.close()
        result_op.close()

    while client.poll(capacity=64, timeout_ms=0):
        pass
    metrics = client.metrics
    retained = {
        "shared_bytes_current": metrics.shared_bytes_current,
        "mapped_bytes_current": metrics.mapped_bytes_current,
        "released_minus_admitted": (
            metrics.released_shared_bytes - metrics.admitted_shared_bytes
        ),
        "trainer_spool_bytes": metrics.trainer_spool_bytes,
        "python_dense_socket_bytes": metrics.python_dense_socket_bytes,
        "handoff_full_copy_bytes": metrics.handoff_full_copy_bytes,
        "disk_replay_bytes": metrics.disk_replay_bytes,
    }
    if any(retained.values()):
        raise AssertionError(
            f"generation {generation} retained native or forbidden-path bytes: {retained}"
        )
    return result_sha256


def _stress_worker(
    worker: int,
    library_path: str,
    generations: int,
    warmup_generations: int,
    elements: int,
) -> dict[str, Any]:
    native = NativeLibrary(library_path)
    base = np.linspace(-3.0, 5.0, elements, dtype=np.float32)
    inputs = (base, np.flip(base).copy())
    # The native v1 order is trainer-key order, not submission order.
    weights_by_key = sorted(
        ((_key(worker, lane), values, weight)
         for lane, (values, weight) in enumerate(
             zip(inputs, (3, 1_000_003), strict=True))),
        key=lambda item: item[0],
    )
    numerator = np.zeros(elements, dtype=np.float64)
    for _trainer_key, values, weight in weights_by_key:
        numerator = numerator + values.astype(np.float64) * np.float64(weight)
    expected = (numerator / np.float64(1_000_006)).astype("<f4")
    expected_sha256 = hashlib.sha256(expected.tobytes(order="C")).hexdigest()

    run_key = _key(worker, 64)
    with Client.open(
        library=native,
        run_key=run_key,
        fence_epoch=worker + 1,
        worker_key=_key(worker, 65),
        incarnation=_key(worker, 66),
        socket_path=f"/tmp/emender-ndp-local-v1-{os.getpid()}.sock",
    ) as client:
        client.install_flat_layout(elements, source_dtype=DType.F32, payload_max=4096)
        for generation in range(warmup_generations):
            digest = _run_generation(
                client, generation=generation, worker=worker,
                inputs=inputs, expected=expected,
            )
            if digest != expected_sha256:
                raise AssertionError("warm-up result digest disagrees with reference")

        gc.collect()
        baseline = _resource_sample()
        samples: list[dict[str, int]] = []
        first_result_sha256 = ""
        last_result_sha256 = ""
        for offset in range(generations):
            generation = warmup_generations + offset
            digest = _run_generation(
                client, generation=generation, worker=worker,
                inputs=inputs, expected=expected,
            )
            if not first_result_sha256:
                first_result_sha256 = digest
            last_result_sha256 = digest
            gc.collect()
            samples.append(_resource_sample())

        metrics = client.metrics
        terminal = _resource_sample()

    final_after_close = _resource_sample()
    window = min(16, len(samples))
    first_window_rss_max = max(item["rss_bytes"] for item in samples[:window])
    last_window_rss_max = max(item["rss_bytes"] for item in samples[-window:])
    rss_plateau_growth = max(0, last_window_rss_max - first_window_rss_max)
    rss_final_growth = max(0, terminal["rss_bytes"] - baseline["rss_bytes"])
    fd_values = [baseline["fds"], *(item["fds"] for item in samples), terminal["fds"]]
    thread_values = [
        baseline["threads"], *(item["threads"] for item in samples), terminal["threads"]
    ]
    invariants = {
        "exact_reference_every_generation": (
            first_result_sha256 == last_result_sha256 == expected_sha256
        ),
        "descriptor_count_stable": min(fd_values) == max(fd_values),
        "thread_count_stable": min(thread_values) == max(thread_values),
        "rss_plateau_bounded": rss_plateau_growth <= RSS_PLATEAU_TOLERANCE_BYTES,
        "rss_final_bounded": rss_final_growth <= RSS_PLATEAU_TOLERANCE_BYTES,
        "native_current_bytes_zero": (
            metrics.shared_bytes_current == 0 and metrics.mapped_bytes_current == 0
        ),
        "native_admitted_bytes_released": (
            metrics.released_shared_bytes == metrics.admitted_shared_bytes
        ),
        "no_forbidden_dense_path": (
            metrics.trainer_spool_bytes == 0
            and metrics.trainer_spool_files == 0
            and metrics.python_dense_socket_bytes == 0
            and metrics.handoff_full_copy_bytes == 0
            and metrics.disk_replay_bytes == 0
            and metrics.disk_replay_files == 0
        ),
        "projection_count_exact": (
            metrics.projection_count == generations + warmup_generations
        ),
        "client_close_releases_descriptors": final_after_close["fds"] < terminal["fds"],
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise AssertionError(f"stress worker {worker} invariant failures: {failed}")
    return {
        "worker": worker,
        "pid": os.getpid(),
        "generations": generations,
        "warmup_generations": warmup_generations,
        "elements": elements,
        "scaled_layout_bytes": elements * WIRE_ELEMENT_BYTES,
        "expected_result_sha256": expected_sha256,
        "baseline": baseline,
        "terminal_before_client_close": terminal,
        "final_after_client_close": final_after_close,
        "fd_min": min(fd_values),
        "fd_max": max(fd_values),
        "thread_min": min(thread_values),
        "thread_max": max(thread_values),
        "rss_first_window_max_bytes": first_window_rss_max,
        "rss_last_window_max_bytes": last_window_rss_max,
        "rss_plateau_growth_bytes": rss_plateau_growth,
        "rss_final_growth_bytes": rss_final_growth,
        "rss_plateau_tolerance_bytes": RSS_PLATEAU_TOLERANCE_BYTES,
        "native_metrics": metrics.__dict__,
        "invariants": invariants,
    }


def run_gate(
    *, library_path: str, generations: int, warmup_generations: int,
    elements: int, workers: int,
) -> dict[str, Any]:
    if generations < 100:
        raise ValueError("the qualification requires at least 100 measured generations")
    if warmup_generations < 1 or elements < 8 or workers < 2:
        raise ValueError("the qualification requires warm-up, >=8 elements, and >=2 processes")
    layout = full_layout_accounting()
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [
            pool.submit(
                _stress_worker, worker, library_path, generations,
                warmup_generations, elements,
            )
            for worker in range(workers)
        ]
        stress = [future.result() for future in futures]
    # A third, later spawn is a clean process/library restart rather than pool
    # reuse.  It repeats the full leak threshold because this gate does not
    # treat a one-generation startup probe as restart evidence.
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        restart = pool.submit(
            _stress_worker, workers, library_path, generations,
            warmup_generations, elements,
        ).result()
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema": SCHEMA,
        "status": "passed",
        "task_id": "validate-native-dataplane-local-v1",
        "source_commit": source_commit,
        "slurm_submitted": False,
        "provider_scope": "local component G0; tcp;ofi_rxm provider is exercised by CTest",
        "production_promoted": False,
        "duration_seconds": time.monotonic() - started,
        "stress": {
            "workers": stress,
            "clean_restart_worker": restart,
            "measured_generations_per_process": generations,
            "measured_generations_total": generations * (workers + 1),
            "warmup_generations_per_process": warmup_generations,
        },
        "exact_e97_frontier_accounting": layout,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="path to libemender_ndp.so.1")
    parser.add_argument("--output", required=True, help="machine-readable JSON output")
    parser.add_argument("--generations", type=int, default=128)
    parser.add_argument("--warmup-generations", type=int, default=8)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_gate(
        library_path=str(Path(args.library).resolve()),
        generations=args.generations,
        warmup_generations=args.warmup_generations,
        elements=args.elements,
        workers=args.workers,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
