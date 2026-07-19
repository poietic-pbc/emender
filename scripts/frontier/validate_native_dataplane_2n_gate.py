#!/usr/bin/env python3
"""Fail-closed validator and independent exact reference for Frontier G2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ndm.native_artifacts import GATE_SCHEMA, sha256_file, validate_build_manifest


LAYOUT_BYTES = 5_506_770_496
TOTAL_ELEMENTS = 688_346_312
PAYLOAD_MAX = 67_108_864
SHARD_COUNT = 83
NODE_WEIGHTS = (1_966_080, 1_968_000)
GLOBAL_WEIGHT = 3_934_080
LOGICAL_BYTES = 11_013_540_992
PYTHON_BASELINE_SECONDS = 98.961446568
NOISE_CEILING_SECONDS = 118.753735882
PYTHON_BASELINE_LOGICAL_BYTES_PER_SECOND = 222_582_457.59
REQUIRED_SPEEDUP_OVER_PYTHON = 4.0
MIN_LOGICAL_BYTES_PER_SECOND = (
    REQUIRED_SPEEDUP_OVER_PYTHON * PYTHON_BASELINE_LOGICAL_BYTES_PER_SECOND
)
NATIVE_TARGET_SECONDS = PYTHON_BASELINE_SECONDS / REQUIRED_SPEEDUP_OVER_PYTHON
RESIDENT_BOUND_TWO_OWNERS = 14_440_737_184
POST_RELEASE_RSS_TOLERANCE = 256 * 1024 * 1024
PROCESS_RSS_BOUND = RESIDENT_BOUND_TWO_OWNERS + LAYOUT_BYTES + POST_RELEASE_RSS_TOLERANCE
NODE_SCHEMA = "emender-native-dataplane-node-gate-v1"
MEMBERSHIP_SCHEMA = "emender-native-dataplane-membership-v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _digest_text(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _reference_values() -> tuple[float, float]:
    results = []
    for parity in (0.0, 0.5):
        node_numerators = []
        for rank, lane_weight in ((0, 245_760), (1, 246_000)):
            numerator = 0.0
            for lane in range(8):
                source = float(rank * 16 + lane) + 0.25 + parity
                numerator = numerator + source * float(lane_weight)
            node_numerators.append(numerator)
        results.append((node_numerators[0] + node_numerators[1]) / float(GLOBAL_WEIGHT))
    return results[0], results[1]


def exact_reference(layout_bytes: int, payload_max: int) -> dict[str, Any]:
    if layout_bytes <= 0 or layout_bytes % 16 or payload_max <= 0 or payload_max % 16:
        raise ValueError("reference layout must contain complete alternating f64 pairs")
    even, odd = _reference_values()
    pattern = struct.pack("<dd", even, odd)
    full = hashlib.sha256()
    shard_sha256: list[str] = []
    shard_count = (layout_bytes + payload_max - 1) // payload_max
    for shard in range(shard_count):
        size = min(payload_max, layout_bytes - shard * payload_max)
        payload = pattern * (size // len(pattern))
        if len(payload) != size:
            raise AssertionError("reference payload construction lost bytes")
        shard_sha256.append(hashlib.sha256(payload).hexdigest())
        full.update(payload)
    return {
        "even_value": even,
        "odd_value": odd,
        "payload_sha256": full.hexdigest(),
        "shard_sha256": shard_sha256,
    }


def expected_result_root(
    *, run_id: str, payload_id: str, generation: int, owner_epoch: int,
    layout_bytes: int, payload_max: int, shard_sha256: list[str],
) -> str:
    run_key = _digest_text("run:" + run_id)[:16]
    layout_digest = _digest_text(f"e97-layout-v1:{layout_bytes}:{payload_max}")
    base_digest = _digest_text(f"synthetic-base:{payload_id}:{generation}")
    encoded = bytearray()
    encoded += run_key
    encoded += struct.pack("<QQIQ", 1, generation, 1, owner_epoch)
    encoded += layout_digest + base_digest + struct.pack("<Q", GLOBAL_WEIGHT)
    for shard, digest in enumerate(shard_sha256):
        size = min(payload_max, layout_bytes - shard * payload_max)
        encoded += struct.pack("<IQ", shard, size) + bytes.fromhex(digest)
    return hashlib.sha256(b"emender-ndp-result-v1\0" + encoded).hexdigest()


def _validate_common_node(node: dict[str, Any], *, rank: int, mode: str,
                          provider: str, exact: bool) -> None:
    required = {
        "schema": NODE_SCHEMA,
        "status": "passed",
        "rank": rank,
        "mode": mode,
        "provider": provider,
        "endpoint_type": "FI_EP_RDM",
        "production_provider": provider == "cxi",
        "trainers_per_node": 8,
        "python_dense_socket_bytes": 0,
        "trainer_spool_bytes": 0,
        "trainer_spool_files": 0,
        "disk_replay_bytes": 0,
        "handoff_full_copy_bytes": 0,
        "central_full_model_broker": False,
        "mpi_collectives": 0,
        "all_rank_barriers": 0,
        "partial_commit": False,
    }
    if exact:
        required.update(
            layout_bytes=LAYOUT_BYTES,
            total_elements=TOTAL_ELEMENTS,
            payload_max=PAYLOAD_MAX,
            shard_count=SHARD_COUNT,
            global_weight=GLOBAL_WEIGHT,
            node_weight=NODE_WEIGHTS[rank],
            local_reduction_input_bytes=22_027_081_984,
        )
    mismatches = {
        key: (node.get(key), expected)
        for key, expected in required.items()
        if node.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"node {rank} identity/bound mismatch: {mismatches}")
    admitted_resident = int(node.get("admitted_resident_bytes", 0))
    if exact and not 0 < admitted_resident <= RESIDENT_BOUND_TWO_OWNERS:
        raise ValueError(
            f"node {rank} resident admission {admitted_resident} exceeds "
            f"the two-owner bound {RESIDENT_BOUND_TWO_OWNERS}"
        )
    transport = node.get("transport")
    if not isinstance(transport, dict):
        raise ValueError(f"node {rank} transport metrics missing")
    for key in ("cq_errors", "route_errors", "in_flight_bytes", "retained_bytes"):
        if transport.get(key) != 0:
            raise ValueError(f"node {rank} terminal {key} is not zero")
    if not 0 < transport.get("tx_slot_high_water", 0) <= 4:
        raise ValueError(f"node {rank} TX slot high-water invalid")
    if not 0 < transport.get("rx_slot_high_water", 0) <= 4:
        raise ValueError(f"node {rank} RX slot high-water invalid")
    if node.get("post_release_rss_bytes", 1 << 63) > (
        node.get("baseline_rss_bytes", 0) + POST_RELEASE_RSS_TOLERANCE
    ):
        raise ValueError(f"node {rank} did not return to its post-release RSS floor")
    rss_high_water = int(node.get("rss_high_water_bytes", -1))
    if not max(int(node.get("baseline_rss_bytes", 0)), 1) <= rss_high_water <= (
        PROCESS_RSS_BOUND if exact else int(node["admitted_resident_bytes"]) +
        int(node["layout_bytes"]) + POST_RELEASE_RSS_TOLERANCE
    ):
        raise ValueError(f"node {rank} process RSS high-water exceeds its admitted bound")
    transport_byte_bound = 4 * (int(node["payload_max"]) + 4096)
    for key in ("in_flight_high_water", "retained_high_water"):
        if not 0 < int(transport.get(key, 0)) <= transport_byte_bound:
            raise ValueError(f"node {rank} transport {key} exceeds four registered slots")
    if int(transport.get("released_bytes", 0)) <= 0:
        raise ValueError(f"node {rank} transport did not report released bytes")


def validate_gate(
    *, mode: str, manifest_path: str | Path, membership_path: str | Path,
    node_paths: tuple[str | Path, str | Path], submission_path: str | Path,
    source_root: str | Path | None, clean_gate_path: str | Path | None,
    exact: bool,
) -> dict[str, Any]:
    build = validate_build_manifest(
        manifest_path, source_root=source_root, require_clean=True,
    )
    manifest = _load(manifest_path)
    if "synthetic_gate_binary" not in build.artifacts:
        raise ValueError("G2 build does not hash the launched synthetic gate executable")
    membership = _load(membership_path)
    nodes = [_load(path) for path in node_paths]
    submission = _load(submission_path)
    provider = "cxi" if exact else str(nodes[0].get("provider", ""))
    if membership.get("schema") != MEMBERSHIP_SCHEMA or membership.get("status") != "passed":
        raise ValueError("two-endpoint membership attestation failed")
    if membership.get("mode") != mode or membership.get("provider") != provider:
        raise ValueError("membership payload/provider mismatch")
    if (
        membership.get("two_endpoints") is not True
        or membership.get("clock_attestation") != "client_minus_controller_offset_delta"
        or membership.get("max_clock_skew_ms", 1e9) > 250
    ):
        raise ValueError("membership did not attest two current, clock-aligned endpoints")
    expected_phases = 2 if mode == "fault" else 1
    if membership.get("phase_count") != expected_phases:
        raise ValueError("membership endpoint phase count mismatch")
    if exact:
        for phase in membership.get("phases", []):
            for rank in ("0", "1"):
                record = phase.get(rank, {})
                provider_facts = {
                    "provider": record.get("provider"),
                    "fabric": record.get("fabric"),
                    "domain": record.get("domain"),
                }
                if provider_facts != {
                    "provider": "cxi", "fabric": "cxi", "domain": "cxi0"
                }:
                    raise ValueError(
                        f"endpoint {rank} selected unexpected CXI facts: {provider_facts}"
                    )

    for rank, node in enumerate(nodes):
        _validate_common_node(node, rank=rank, mode=mode, provider=provider, exact=exact)
    if nodes[0].get("run_id") != nodes[1].get("run_id"):
        raise ValueError("nodes disagree on run ID")
    if nodes[0].get("payload_id") != nodes[1].get("payload_id"):
        raise ValueError("nodes disagree on changed payload ID")
    if nodes[0].get("initial_endpoint_incarnation") == nodes[1].get("initial_endpoint_incarnation"):
        raise ValueError("native endpoint incarnations are not distinct")

    layout_bytes = int(nodes[0]["layout_bytes"])
    payload_max = int(nodes[0]["payload_max"])
    reference = exact_reference(layout_bytes, payload_max)
    warmup_count = 1 if exact else len(nodes[0].get("warmups", []))
    sample_count = (3 if mode == "clean" else 1) if exact else len(nodes[0].get("samples", []))
    for node in nodes:
        if len(node.get("warmups", [])) != warmup_count or len(node.get("samples", [])) != sample_count:
            raise ValueError("warm-up/measured generation count mismatch")

    all_samples = [*zip(nodes[0]["warmups"], nodes[1]["warmups"]),
                   *zip(nodes[0]["samples"], nodes[1]["samples"])]
    owner_epoch = 2 if mode == "fault" else 1
    for generation, pair in enumerate(all_samples):
        epoch = owner_epoch if mode == "fault" and generation >= warmup_count else 1
        expected_root = expected_result_root(
            run_id=str(nodes[0]["run_id"]),
            payload_id=str(nodes[0]["payload_id"]),
            generation=generation,
            owner_epoch=epoch,
            layout_bytes=layout_bytes,
            payload_max=payload_max,
            shard_sha256=reference["shard_sha256"],
        )
        roots = {sample.get("result_root") for sample in pair}
        payloads = {sample.get("result_payload_sha256") for sample in pair}
        if roots != {expected_root} or payloads != {reference["payload_sha256"]}:
            raise ValueError(f"generation {generation} differs from independent exact reference")

    measured_durations = [
        max(float(nodes[0]["samples"][index]["transfer_redistribution_seconds"]),
            float(nodes[1]["samples"][index]["transfer_redistribution_seconds"]))
        for index in range(sample_count)
    ]
    physical_contribution = [
        sum(int(nodes[rank]["samples"][index]["contribution_tx_bytes"]) for rank in (0, 1))
        for index in range(sample_count)
    ]
    physical_redistribution = [
        sum(int(nodes[rank]["samples"][index]["redistribution_tx_bytes"]) for rank in (0, 1))
        for index in range(sample_count)
    ]
    if any(value != layout_bytes for value in physical_contribution):
        raise ValueError("clean physical contribution bytes do not cover distributed owners")
    if any(value != layout_bytes for value in physical_redistribution):
        raise ValueError("physical redistribution bytes do not cover both nodes")

    stale_rejects = sum(
        int(sample["stale_rejects"])
        for node in nodes for sample in [*node["warmups"], *node["samples"]]
    )
    checksum_rejects = sum(
        int(sample["checksum_rejects"])
        for node in nodes for sample in [*node["warmups"], *node["samples"]]
    )
    if mode == "clean":
        if stale_rejects != 2 or checksum_rejects != 2:
            raise ValueError("preflight did not reject one stale and corrupt frame at both endpoints")
        if any(
            int(sample["stale_rejects"]) or int(sample["checksum_rejects"])
            for node in nodes for sample in node["samples"]
        ):
            raise ValueError("timed clean generation contained a rejection")
        median_seconds = statistics.median(measured_durations)
        maximum_seconds = max(measured_durations)
        logical_bytes_per_second = (2 * LOGICAL_BYTES) / median_seconds
        if exact and (
            median_seconds > NATIVE_TARGET_SECONDS
            or maximum_seconds > NOISE_CEILING_SECONDS
            or logical_bytes_per_second < MIN_LOGICAL_BYTES_PER_SECOND
        ):
            raise ValueError("native clean throughput did not reach 4x the retained Python gate")
        gate_name = "G2"
        clean_dependency = None
    else:
        if clean_gate_path is None:
            raise ValueError("fault payload requires the already-passed clean gate")
        clean_dependency = _load(clean_gate_path)
        if clean_dependency.get("gate") != "G2" or clean_dependency.get("status") != "passed":
            raise ValueError("fault payload dependency is not a passing G2 gate")
        if clean_dependency.get("source_commit") != build.source_commit:
            raise ValueError("fault payload changed the native build after the clean gate")
        if clean_dependency.get("payload_id") == nodes[0].get("payload_id"):
            raise ValueError("failure injection did not use a changed payload ID")
        if any(node.get("fault_reassignment_count") != 1 for node in nodes):
            raise ValueError("failure injection did not perform exactly one reassignment")
        if any(node.get("fault_replay_bytes") != 2 * payload_max for node in nodes):
            raise ValueError("failure injection replay byte count is not exact")
        if nodes[1].get("initial_endpoint_incarnation") == nodes[1].get("final_endpoint_incarnation"):
            raise ValueError("lost owner did not rejoin with a new incarnation")
        if sum(int(node.get("old_epoch_rejects", 0)) for node in nodes) != 1:
            raise ValueError("rejoined owner did not reject the old owner epoch")
        if any(node.get("partial_commit") is not False for node in nodes):
            raise ValueError("failure path exposed a partial commit")
        median_seconds = measured_durations[0]
        maximum_seconds = measured_durations[0]
        logical_bytes_per_second = (2 * LOGICAL_BYTES) / median_seconds if exact else 0.0
        gate_name = "G2-fault-rejoin-replay"

    artifact_digests = {
        name: str(record["sha256"])
        for name, record in manifest["artifacts"].items()
    }
    value: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "gate": gate_name,
        "status": "passed",
        "source_commit": build.source_commit,
        "bundle_sha256": build.bundle_sha256,
        "provider": provider,
        "endpoint_type": "FI_EP_RDM",
        "production_provider": exact,
        "nodes": 2,
        "endpoints": 2,
        "owners": 2,
        "layout_bytes": layout_bytes,
        "total_elements": layout_bytes // 8,
        "shard_count": (layout_bytes + payload_max - 1) // payload_max,
        "payload_max": payload_max,
        "trainers_per_node": 8,
        "node_weights": list(NODE_WEIGHTS),
        "global_weight": GLOBAL_WEIGHT,
        "logical_contribution_bytes": 2 * layout_bytes,
        "logical_redistribution_bytes": 2 * layout_bytes,
        "python_dense_socket_bytes": 0,
        "trainer_spool_bytes": 0,
        "disk_replay_bytes": 0,
        "handoff_full_copy_bytes": 0,
        "mpi_collectives": 0,
        "all_rank_barriers": 0,
        "run_id": nodes[0]["run_id"],
        "payload_id": nodes[0]["payload_id"],
        "warmup_generations": warmup_count,
        "timed_generations": sample_count,
        "result_payload_sha256": reference["payload_sha256"],
        "exact_reference": {
            "implementation": "independent Python analytical f64 lane reference",
            "even_value": reference["even_value"],
            "odd_value": reference["odd_value"],
            "all_generations_match": True,
            "stale_rejects": stale_rejects,
            "checksum_rejects": checksum_rejects,
        },
        "performance": {
            "samples_seconds": measured_durations,
            "median_seconds": median_seconds,
            "maximum_seconds": maximum_seconds,
            "python_baseline_seconds": PYTHON_BASELINE_SECONDS,
            "native_target_seconds": NATIVE_TARGET_SECONDS,
            "required_speedup_over_python": REQUIRED_SPEEDUP_OVER_PYTHON,
            "python_baseline_logical_bytes_per_second": PYTHON_BASELINE_LOGICAL_BYTES_PER_SECOND,
            "minimum_native_logical_bytes_per_second": MIN_LOGICAL_BYTES_PER_SECOND,
            "logical_bytes_per_second": logical_bytes_per_second,
            "speedup_over_python": (
                logical_bytes_per_second / PYTHON_BASELINE_LOGICAL_BYTES_PER_SECOND
                if exact else None
            ),
            "noise_ceiling_seconds": NOISE_CEILING_SECONDS,
        },
        "bounds": {
            "owner_resident_admission_bytes": max(
                int(node["admitted_resident_bytes"]) for node in nodes
            ),
            "owner_resident_bound_bytes": RESIDENT_BOUND_TWO_OWNERS if exact else max(
                int(node["admitted_resident_bytes"]) for node in nodes
            ),
            "process_rss_high_water_bytes": max(
                int(node["rss_high_water_bytes"]) for node in nodes
            ),
            "process_rss_bound_bytes": PROCESS_RSS_BOUND if exact else max(
                int(node["admitted_resident_bytes"]) + int(node["layout_bytes"])
                + POST_RELEASE_RSS_TOLERANCE for node in nodes
            ),
            "transport_in_flight_high_water_bytes": max(
                int(node["transport"]["in_flight_high_water"]) for node in nodes
            ),
            "transport_retained_high_water_bytes": max(
                int(node["transport"]["retained_high_water"]) for node in nodes
            ),
            "post_release_transport_bytes": 0,
            "post_release_rss_within_floor": True,
        },
        "transport": {
            "wire_tx_bytes": sum(int(node["transport"]["wire_tx_bytes"]) for node in nodes),
            "wire_rx_bytes": sum(int(node["transport"]["wire_rx_bytes"]) for node in nodes),
            "useful_tx_bytes": sum(int(node["transport"]["useful_tx_bytes"]) for node in nodes),
            "useful_rx_bytes": sum(int(node["transport"]["useful_rx_bytes"]) for node in nodes),
            "retries": sum(int(node["transport"]["retries"]) for node in nodes),
            "cq_errors": 0,
            "route_errors": 0,
            "released_bytes": sum(int(node["transport"]["released_bytes"]) for node in nodes),
        },
        "fault": None if mode == "clean" else {
            "peer_loss": True,
            "new_incarnation": True,
            "owner_epoch_before": 1,
            "owner_epoch_after": 2,
            "reassignment_count": 1,
            "replay_bytes": 2 * payload_max,
            "old_epoch_rejected": True,
            "partial_commit": False,
            "clean_gate_sha256": sha256_file(clean_gate_path),
        },
        "artifacts": artifact_digests,
        "evidence_sha256": {
            "build_manifest": sha256_file(manifest_path),
            "membership": sha256_file(membership_path),
            "node_0": sha256_file(node_paths[0]),
            "node_1": sha256_file(node_paths[1]),
            "submission": sha256_file(submission_path),
        },
        "slurm": submission,
        "conformance": {
            "compute_pool": "RESILIENT_DILOCO_COMPUTE_POOL.md version 1",
            "requirements": [f"R{index:02d}" for index in range(1, 17)],
            "native_requirements": [f"NDP{index:02d}" for index in range(1, 18)],
            "q_min": 2,
            "t_min": GLOBAL_WEIGHT,
            "ready_membership_not_launched_rank": True,
            "bounded_non_lustre_hot_path": True,
            "central_full_model_broker": False,
            "atomic_result_or_no_commit": True,
        },
    }
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clean", "fault"), required=True)
    parser.add_argument("--build-manifest", required=True)
    parser.add_argument("--membership", required=True)
    parser.add_argument("--node-0", required=True)
    parser.add_argument("--node-1", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--clean-gate", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-scaled-test", action="store_true")
    args = parser.parse_args()
    result = validate_gate(
        mode=args.mode,
        manifest_path=args.build_manifest,
        membership_path=args.membership,
        node_paths=(args.node_0, args.node_1),
        submission_path=args.submission,
        source_root=args.source_root,
        clean_gate_path=args.clean_gate or None,
        exact=not args.allow_scaled_test,
    )
    _atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
