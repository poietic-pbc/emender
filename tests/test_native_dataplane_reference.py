import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports/frontier/native-dataplane-reference-v1.json"
REPORT = ROOT / "reports/frontier/native-dataplane-reference-v1.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_dataplane_reference_is_checksum_linked_and_machine_readable():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert baseline["schema_version"] == 1
    assert baseline["baseline_id"] == "frontier-native-dataplane-reference-v1"
    assert baseline["qualification_status"] == "qualified-reference-only"
    assert baseline["slurm_diagnostic"]["submitted"] is False
    assert _sha256(BASELINE) in REPORT.read_text(encoding="utf-8")
    assert baseline["authority"]["applicable_native_requirement_ids"] == [
        "NDP02",
        "NDP03",
        "NDP05",
        "NDP16",
        "NDP17",
    ]

    repo_evidence = baseline["evidence"]["repository"]
    for item in repo_evidence:
        path = ROOT / item["path"]
        assert path.is_file()
        assert item["sha256"] == _sha256(path)

    retained = baseline["evidence"]["retained_job_4974616"]
    assert retained["job_id"] == 4974616
    assert retained["git_commit"] == "40eb8d48e6dfe414aae3bfccf056904433aecdcb"
    assert retained["source_sha256"] != repo_evidence[0]["sha256"]
    assert retained["binary_sha256"] != baseline["current_build"]["binary_sha256"]
    assert retained["shared_library_sha256"] != baseline["current_build"]["shared_library_sha256"]


def test_native_dataplane_reference_defines_topology_latency_throughput_and_targets():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    reference = baseline["fixed_world_reference"]

    assert reference["topology"] == {
        "nodes": 256,
        "ranks": 2048,
        "ranks_per_node": 8,
        "unique_rank_start_hosts": 256,
        "membership_model": "all-launched-ranks",
        "fixed_world": True,
    }
    assert reference["payload"]["aggregate_update_bytes"] == 5_506_770_496
    assert reference["payload"]["logical_ingress_bytes"] == 11_277_865_975_808
    assert reference["latency_seconds"]["reported_merge"] == 5.304643992334604
    assert reference["latency_seconds"]["collective_reduce"] == 73.9047

    throughput = reference["throughput_bytes_per_second"]
    assert math.isclose(
        throughput["reported_merge_payload"],
        reference["payload"]["aggregate_update_bytes"]
        / reference["latency_seconds"]["reported_merge"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        throughput["collective_logical_ingress"],
        reference["payload"]["logical_ingress_bytes"]
        / reference["latency_seconds"]["collective_reduce"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        throughput["collective_payload_per_contribution"],
        reference["payload"]["aggregate_update_bytes"]
        / reference["latency_seconds"]["collective_reduce"],
        rel_tol=1e-12,
    )

    targets = baseline["elastic_backend_acceptance_targets"]
    assert targets["layout"] == {
        "total_elements": 688_346_312,
        "layout_bytes": 5_506_770_496,
        "payload_max_bytes": 67_108_864,
        "shard_count": 83,
        "wire_dtype": "IEEE-754 little-endian binary64",
    }
    assert targets["stage_hard_max_seconds"]["frozen_transfer_including_replay"] == 180
    assert targets["stage_hard_max_seconds"]["redistribution_root_validation"] == 180

    g2 = targets["g2_full_layout_two_node"]
    assert g2["nodes"] == 2
    assert g2["logical_contribution_bytes"] == 11_013_540_992
    assert g2["logical_redistribution_bytes"] == 11_013_540_992
    assert g2["p50_transfer_plus_redistribution_max_seconds"] == 98.961446568
    assert math.isclose(
        g2["minimum_logical_goodput_bytes_per_second"],
        g2["throughput_numerator_bytes"]
        / g2["p50_transfer_plus_redistribution_max_seconds"],
        rel_tol=1e-10,
    )
    assert g2["maximum_timed_iteration_seconds"] == 118.753735882

    g6 = targets["g6_256_node"]
    assert g6["nodes"] == 256
    assert g6["trainer_lanes"] == 2048
    assert g6["acceptance_median_max_seconds"] == 10.609287984669208
    assert g6["performance_target_median_seconds"] == 5.304643992334604
    assert math.isclose(
        g6["acceptance_minimum_logical_goodput_bytes_per_second"],
        g6["throughput_numerator_bytes"] / g6["acceptance_median_max_seconds"],
        rel_tol=1e-12,
    )
    assert targets["synthetic_reference"]["bitwise_exact"] is True
    assert targets["synthetic_reference"]["absolute_tolerance"] == 0
    assert targets["must_tolerate_missing_launched_ranks"] is True
    assert targets["all_rank_collective_allowed"] is False
    assert targets["mpi_allowed_in_elastic_binary"] is False
    assert targets["production_provider_required"] == "cxi"


def test_native_dataplane_reference_records_required_active_providers_and_limitations():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    providers = {provider["name"]: provider for provider in baseline["active_toolchain"]["libfabric"]["providers"]}

    assert providers["cxi"]["protocol"] == "FI_PROTO_CXI"
    assert providers["cxi"]["query_exit_code"] == 0
    assert providers["shm"]["protocol"] == "FI_PROTO_SHM"
    assert providers["sm2"]["protocol"] == "FI_PROTO_SM2"
    assert providers["tcp"]["protocol"] == "FI_PROTO_XNET"
    assert baseline["fixed_world_reference"]["elastic_backend_compatible"] is False

    report = REPORT.read_text(encoding="utf-8")
    for required in [
        "Resilient DiLoCo Compute Pool, version 1",
        "R03",
        "R05",
        "R06",
        "R08",
        "R10",
        "R14",
        "R15",
        "R16",
        "NDP02",
        "NDP03",
        "NDP05",
        "NDP16",
        "NDP17",
        "MPI_Reduce is not the elastic solution",
        "No Slurm command was submitted",
        "5.304643992334604",
        "10.609287984669208",
        "strict_collective_all_launched_ranks=true",
    ]:
        assert required in report
