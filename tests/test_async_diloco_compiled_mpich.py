import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoUpdate
from ndm.async_diloco import RESILIENT_QUORUM_DILOCO_MODE
from ndm.async_diloco import STRICT_COLLECTIVE_DILOCO_MODE
from ndm.async_diloco import stable_json_dumps
import ndm.async_diloco_compiled_mpich as compiled_mpich
from ndm.async_diloco_compiled_mpich import (
    COMPILED_MPICH_TRANSPORT,
    CompiledMpichHelperConfig,
    collect_compiled_mpich_aggregate_result,
    load_aggregate_payload,
    load_received_payloads,
    resolve_compiled_mpich_helper_library,
    run_compiled_mpich_dense_quorum,
    write_compiled_mpich_request,
)
from ndm.async_diloco_mpi import (
    DenseTransportQuorumConfig,
    collect_dense_quorum_from_envelopes,
    pack_dense_update,
)


def _update(value=1.0):
    return AsyncDiLoCoUpdate(
        worker_id="node-00000",
        base_generation=0,
        delta={"w": torch.tensor([float(value), float(value) + 1.0])},
        tokens=8,
        local_steps=1,
        loss_moving_average={"loss": 1.25, "loss_100": 1.25},
    )


def test_compiled_mpich_cpp_uses_bucketed_collective_reduce_not_root_bucket_fanin():
    source = (Path(__file__).resolve().parents[1] / "scripts/frontier/compiled_mpich_dense_helper.cpp").read_text(
        encoding="utf-8"
    )

    assert "MPI_Reduce(" in source
    assert "MPI_Allreduce(" in source
    assert "MPI_Iprobe" not in source
    assert "TAG_BUCKET" not in source
    assert "root_file_gathered_peer" not in source
    assert "MPI_Send(preamble" not in source
    assert "MPI_Recv(&bucket_len" not in source
    assert "MPI ranks have different aggregate bucket counts" in source
    assert "MPI ranks have different aggregate bucket layouts" in source
    reduce_pos = source.index("MPI_Reduce(local_values.data()")
    size_bcast_pos = source.index("MPI_Bcast(&aggregate_size", reduce_pos)
    payload_bcast_pos = source.index("MPI_Bcast(aggregate.data()", size_bcast_pos)
    write_pos = source.index("write_bytes_atomic(ipc_dir / rel, aggregate)", payload_bcast_pos)
    assert reduce_pos < size_bcast_pos < payload_bcast_pos < write_pos


def test_compiled_mpich_request_contract_uses_file_ipc_and_checksums(tmp_path):
    envelope = pack_dense_update(
        _update(),
        run_id="contract",
        rank=0,
        generation=0,
        bucket_bytes=8,
    )
    request_path = write_compiled_mpich_request(
        envelope,
        ipc_dir=tmp_path / "ipc",
        run_id="contract",
        rank=0,
        world_size=1,
        generation=0,
        base_generation=0,
        quorum=1,
        timeout_s=3.0,
        bucket_bytes_target=8,
        base_checkpoint="seed/latest.pt",
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["schema_version"] == 1
    assert request["transport"] == COMPILED_MPICH_TRANSPORT
    assert request["header_path"] == "rank_00000/gen000000/header.json"
    assert request["bucket_descriptors"]
    assert request["bucket_descriptors"][0]["ipc"]["kind"] == "file"
    assert request["bucket_descriptors"][0]["checksum_sha256"]
    assert (tmp_path / "ipc" / request["header_path"]).is_file()
    assert (tmp_path / "ipc" / request["bucket_descriptors"][0]["ipc"]["path"]).is_file()


def test_compiled_mpich_request_replaces_rank_local_generation_workspace(tmp_path):
    ipc_dir = tmp_path / "ipc"
    first = write_compiled_mpich_request(
        pack_dense_update(_update(), run_id="bounded-ipc", rank=3, generation=0, bucket_bytes=8),
        ipc_dir=ipc_dir,
        run_id="bounded-ipc",
        rank=3,
        world_size=4,
        generation=0,
        base_generation=0,
        quorum=4,
        timeout_s=3.0,
        bucket_bytes_target=8,
        base_checkpoint=None,
    )
    old_aggregate = first.parent / "gen000000" / "aggregate.bucket00000.bin"
    old_aggregate.write_bytes(b"old aggregate")

    second = write_compiled_mpich_request(
        pack_dense_update(_update(), run_id="bounded-ipc", rank=3, generation=1, bucket_bytes=8),
        ipc_dir=ipc_dir,
        run_id="bounded-ipc",
        rank=3,
        world_size=4,
        generation=1,
        base_generation=0,
        quorum=4,
        timeout_s=3.0,
        bucket_bytes_target=8,
        base_checkpoint=None,
    )

    assert not first.exists()
    assert not old_aggregate.exists()
    assert second.is_file()
    assert [path.name for path in second.parent.glob("gen*")] == ["gen000001"]


def test_compiled_mpich_load_received_payloads_rejects_corrupt_checksum(tmp_path):
    envelope = pack_dense_update(_update(), run_id="corrupt", rank=0, generation=0, bucket_bytes=64)
    request_path = write_compiled_mpich_request(
        envelope,
        ipc_dir=tmp_path / "ipc",
        run_id="corrupt",
        rank=0,
        world_size=1,
        generation=0,
        base_generation=0,
        quorum=1,
        timeout_s=3.0,
        bucket_bytes_target=64,
        base_checkpoint=None,
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    received = [{
        "rank": 0,
        "header_path": request["header_path"],
        "bucket_paths": [descriptor["ipc"]["path"] for descriptor in request["bucket_descriptors"]],
    }]
    bucket_path = tmp_path / "ipc" / received[0]["bucket_paths"][0]
    bucket_path.write_bytes(b"X" + bucket_path.read_bytes()[1:])
    loaded = load_received_payloads(tmp_path / "ipc", received)
    with pytest.raises(ValueError, match="checksum mismatch"):
        from ndm.async_diloco_mpi import unpack_dense_update

        unpack_dense_update(loaded[0])


def test_compiled_mpich_helper_invocation_uses_shared_library_not_subprocess(tmp_path, monkeypatch):
    helper = tmp_path / "compiled_mpich_dense_helper"
    helper.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    helper.chmod(0o755)
    helper_lib = tmp_path / "compiled_mpich_dense_helper.so"
    helper_lib.write_bytes(b"fake shared library")

    def reject_subprocess(*args, **kwargs):
        raise AssertionError("compiled helper must be called in-process, not as a per-rank subprocess")

    def fake_shared_library_call(lib_path, ipc_dir, request_path):
        assert lib_path == helper_lib
        assert ipc_dir == tmp_path / "ipc"
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        result = {
            "schema_version": 1,
            "transport": COMPILED_MPICH_TRANSPORT,
            "reducer": "mpi_reduce_bucketed_weighted_sum",
            "strict_collective_all_launched_ranks": True,
            "status": "advanced",
            "rank": payload["rank"],
            "generation": payload["generation"],
            "base_generation": payload["base_generation"],
            "accepted_ranks": [payload["rank"]],
            "accepted_count": 1,
            "stale_count": 0,
            "failed_count": 0,
            "timed_out_count": 0,
            "invalid_count": 0,
            "accepted_tokens": 8,
            "accepted_local_steps": 1,
            "timed_out_ranks": [],
            "failed_ranks": [],
            "stale_ranks": [],
            "invalid_ranks": [],
            "bytes_sent": payload["payload_bytes"],
            "bytes_received": payload["payload_bytes"],
            "aggregate_update_bytes": payload["payload_bytes"],
            "helper_exit_code": 0,
            "mpi": {
                "provided_thread_level": "MPI_THREAD_SERIALIZED",
                "world_size": payload["world_size"],
                "root_rank": 0,
                "collective": "MPI_Reduce",
            },
            "aggregate_payload": {
                "rank": payload["rank"],
                "source_header_path": payload["header_path"],
                "bucket_paths": [d["ipc"]["path"] for d in payload["bucket_descriptors"]],
            },
            "received_payloads": [],
            "reduce_metrics": {
                "bucket_count": len(payload["bucket_descriptors"]),
                "aggregate_bucket_count": len(payload["bucket_descriptors"]),
                "aggregate_update_bytes": payload["payload_bytes"],
                "reduce_duration_s": 0.001,
                "per_bucket": [
                    {"index": d["index"], "bytes": d["nbytes"], "reduce_latency_s": 0.001}
                    for d in payload["bucket_descriptors"]
                ],
            },
            "aggregate_loss_window": {"loss": 1.25, "loss_100": 1.25},
        }
        request_path.with_name(f"result.gen{payload['generation']:06d}.json").write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(subprocess, "run", reject_subprocess)
    monkeypatch.setattr(compiled_mpich, "_call_compiled_mpich_shared_library", fake_shared_library_call)

    payload = run_compiled_mpich_dense_quorum(
        base_state={"w": torch.zeros(2)},
        local_update=_update(2.0),
        run_id="invoke",
        generation=0,
        requested_ranks=1,
        quorum=1,
        rank=0,
        helper=CompiledMpichHelperConfig(
            helper_bin=helper,
            ipc_dir=tmp_path / "ipc",
            bucket_bytes=64,
            timeout_s=3.0,
        ),
    )

    assert payload is not None
    assert payload["transport"]["name"] == COMPILED_MPICH_TRANSPORT
    assert payload["transport"]["helper_result"]["mpi"]["provided_thread_level"] == "MPI_THREAD_SERIALIZED"
    assert payload["transport"]["helper_result"]["received_payloads"] == []
    assert payload["transport"]["metrics"]["quorum_size"] == 1
    assert payload["transport"]["metrics"]["bucket_timings_s"]
    assert payload["global_generations"][0]["metrics"]["mode"] == RESILIENT_QUORUM_DILOCO_MODE
    assert payload["global_generations"][0]["metrics"]["accepted_updates"] == 1
    assert payload["latest_generation"] == -1


def test_compiled_mpich_aggregate_matches_dense_quorum_merge_with_stale_failed_metadata(tmp_path):
    base = {"w": torch.zeros(2)}
    accepted0 = pack_dense_update(_update(1.0), run_id="aggregate", rank=0, generation=0, bucket_bytes=64)
    accepted1 = pack_dense_update(_update(3.0), run_id="aggregate", rank=1, generation=0, bucket_bytes=64)
    failed = pack_dense_update(
        AsyncDiLoCoUpdate(
            worker_id="failed",
            base_generation=0,
            delta={"w": torch.tensor([99.0, 100.0])},
            tokens=8,
            local_steps=1,
            loss_moving_average={"loss": 9.0, "loss_100": 9.0},
            failed=True,
        ),
        run_id="aggregate",
        rank=2,
        generation=0,
        bucket_bytes=64,
    )
    stale = pack_dense_update(
        AsyncDiLoCoUpdate(
            worker_id="stale",
            base_generation=-1,
            delta={"w": torch.tensor([42.0, 43.0])},
            tokens=8,
            local_steps=1,
            loss_moving_average={"loss": 7.0, "loss_100": 7.0},
        ),
        run_id="aggregate",
        rank=3,
        generation=0,
        bucket_bytes=64,
    )
    config = DenseTransportQuorumConfig(
        run_id="aggregate",
        generation=0,
        base_generation=0,
        requested_ranks=4,
        quorum=2,
    )
    expected = collect_dense_quorum_from_envelopes(
        base,
        (accepted0, accepted1, failed, stale),
        config=config,
    )

    request_path = write_compiled_mpich_request(
        accepted0,
        ipc_dir=tmp_path / "ipc",
        run_id="aggregate",
        rank=0,
        world_size=4,
        generation=0,
        base_generation=0,
        quorum=2,
        timeout_s=3.0,
        bucket_bytes_target=64,
        base_checkpoint=None,
    )
    aggregate_delta = expected.state["w"] - base["w"]
    aggregate_envelope = pack_dense_update(
        AsyncDiLoCoUpdate(
            worker_id="compiled_mpich_aggregate",
            base_generation=0,
            delta={"w": aggregate_delta},
            tokens=16,
            local_steps=2,
            loss_moving_average=expected.metrics.loss_moving_average,
        ),
        run_id="aggregate",
        rank=0,
        generation=0,
        bucket_bytes=64,
    )
    aggregate_paths = []
    for bucket in aggregate_envelope.buckets:
        rel = f"rank_00000/gen000000/aggregate.bucket{bucket.index:05d}.bin"
        path = tmp_path / "ipc" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bucket.payload)
        aggregate_paths.append(rel)

    helper_result = {
        "schema_version": 1,
        "transport": COMPILED_MPICH_TRANSPORT,
        "reducer": "mpi_reduce_bucketed_weighted_sum",
        "strict_collective_all_launched_ranks": True,
        "status": "advanced",
        "generation": 0,
        "base_generation": 0,
        "accepted_ranks": [0, 1],
        "stale_ranks": [3],
        "failed_ranks": [2],
        "timed_out_ranks": [],
        "invalid_ranks": [],
        "accepted_count": 2,
        "stale_count": 1,
        "failed_count": 1,
        "timed_out_count": 0,
        "invalid_count": 0,
        "accepted_tokens": 16,
        "accepted_local_steps": 2,
        "bytes_sent": sum(e.payload_bytes for e in (accepted0, accepted1, failed, stale)),
        "bytes_received": aggregate_envelope.payload_bytes,
        "aggregate_update_bytes": aggregate_envelope.payload_bytes,
        "aggregate_payload": {
            "rank": 0,
            "source_header_path": json.loads(request_path.read_text(encoding="utf-8"))["header_path"],
            "bucket_paths": aggregate_paths,
        },
        "received_payloads": [],
        "reduce_metrics": {
            "bucket_count": len(aggregate_paths),
            "aggregate_bucket_count": len(aggregate_paths),
            "aggregate_update_bytes": aggregate_envelope.payload_bytes,
            "reduce_duration_s": 0.25,
            "per_bucket": [
                {"index": idx, "bytes": len(bucket.payload), "reduce_latency_s": 0.01}
                for idx, bucket in enumerate(aggregate_envelope.buckets)
            ],
        },
        "aggregate_loss_window": expected.metrics.loss_moving_average,
    }

    loaded = load_aggregate_payload(tmp_path / "ipc", helper_result["aggregate_payload"], helper_result)
    assert len(loaded.buckets) == 1
    result = collect_compiled_mpich_aggregate_result(
        base,
        helper_result=helper_result,
        ipc_dir=tmp_path / "ipc",
        config=config,
    )

    assert torch.allclose(result.state["w"], expected.state["w"])
    assert result.metrics.accepted_updates == 2
    assert result.metrics.stale_updates == 1
    assert result.metrics.failed_updates == 1
    assert result.transport_metrics.bytes_received == aggregate_envelope.payload_bytes
    assert result.transport_metrics.bytes_sent > result.transport_metrics.bytes_received


def test_compiled_mpich_aggregate_uses_rank_local_metadata_across_distinct_node_ipc_roots(tmp_path):
    """A non-root node must not dereference rank 0's node-local /tmp tree."""

    root_ipc = tmp_path / "node0" / "ipc"
    nonroot_ipc = tmp_path / "node1" / "ipc"
    root_envelope = pack_dense_update(
        _update(1.0), run_id="node-local", rank=0, generation=0, bucket_bytes=64
    )
    nonroot_envelope = pack_dense_update(
        _update(3.0), run_id="node-local", rank=8, generation=0, bucket_bytes=64
    )
    root_request = write_compiled_mpich_request(
        root_envelope,
        ipc_dir=root_ipc,
        run_id="node-local",
        rank=0,
        world_size=16,
        generation=0,
        base_generation=0,
        quorum=16,
        timeout_s=3.0,
        bucket_bytes_target=64,
        base_checkpoint=None,
    )
    nonroot_request = write_compiled_mpich_request(
        nonroot_envelope,
        ipc_dir=nonroot_ipc,
        run_id="node-local",
        rank=8,
        world_size=16,
        generation=0,
        base_generation=0,
        quorum=16,
        timeout_s=3.0,
        bucket_bytes_target=64,
        base_checkpoint=None,
    )
    root_header_path = json.loads(root_request.read_text(encoding="utf-8"))["header_path"]
    nonroot_header_path = json.loads(nonroot_request.read_text(encoding="utf-8"))["header_path"]

    # Reproduce job 4979251: rank 0's relative header does not exist under a
    # different node's local IPC root.
    old_payload = {
        "rank": 0,
        "source_header_path": root_header_path,
        "bucket_paths": ["rank_00000/gen000000/aggregate.bucket00000.bin"],
    }
    with pytest.raises(FileNotFoundError, match="header.json"):
        load_aggregate_payload(nonroot_ipc, old_payload, {})

    # The helper now broadcasts reduced bytes through MPI and materializes
    # rank-owned aggregate files beside that rank's local request metadata.
    local_bucket_paths = []
    for bucket in root_envelope.buckets:
        rel = f"rank_00008/gen000000/aggregate.bucket{bucket.index:05d}.bin"
        path = nonroot_ipc / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bucket.payload)
        local_bucket_paths.append(rel)
    helper_result = {
        "accepted_tokens": 8,
        "accepted_local_steps": 1,
        "aggregate_loss_window": {"loss": 1.25, "loss_100": 1.25},
    }
    local_payload = {
        "rank": 8,
        "source_header_path": nonroot_header_path,
        "bucket_paths": local_bucket_paths,
    }
    loaded = load_aggregate_payload(nonroot_ipc, local_payload, helper_result)
    assert loaded.header["rank"] == 8
    assert loaded.buckets[0].payload == root_envelope.buckets[0].payload

    source = (Path(__file__).resolve().parents[1] / "scripts/frontier/compiled_mpich_dense_helper.cpp").read_text(
        encoding="utf-8"
    )
    assert "rel_aggregate_bucket(rank, req.generation, bucket_desc.index)" in source
    assert "MPI_Bcast(aggregate.data()" in source
    assert '"aggregate_payload":{"rank":0' not in source


def test_compiled_mpich_helper_invocation_errors_on_nonzero(tmp_path, monkeypatch):
    helper = tmp_path / "compiled_mpich_dense_helper"
    helper_lib = tmp_path / "compiled_mpich_dense_helper.so"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper_lib.write_bytes(b"fake shared library")
    helper.chmod(0o755)

    monkeypatch.setattr(compiled_mpich, "_call_compiled_mpich_shared_library", lambda *args: 7)

    with pytest.raises(RuntimeError, match="compiled MPICH helper shared library failed"):
        run_compiled_mpich_dense_quorum(
            base_state={"w": torch.zeros(2)},
            local_update=_update(),
            run_id="fail",
            generation=0,
            requested_ranks=1,
            quorum=1,
            rank=0,
            helper=CompiledMpichHelperConfig(helper_bin=helper, ipc_dir=tmp_path / "ipc"),
            quorum_mode=STRICT_COLLECTIVE_DILOCO_MODE,
        )


def test_compiled_mpich_helper_library_path_defaults_to_binary_sibling():
    helper = CompiledMpichHelperConfig(
        helper_bin=Path("/frontier/run/artifacts/compiled_mpich_dense_helper"),
        ipc_dir=Path("/frontier/run/ipc"),
    )
    assert resolve_compiled_mpich_helper_library(helper) == Path(
        "/frontier/run/artifacts/compiled_mpich_dense_helper.so"
    )


def test_compiled_mpich_cpp_request_parser_preserves_all_bucket_paths(tmp_path):
    if shutil.which("CC") is None:
        pytest.skip("Frontier CC compiler wrapper is not available")

    descriptors = []
    for index in range(80):
        descriptors.append({
            "index": index,
            "nbytes": 1024 + index,
            "checksum_sha256": f"{index:064x}",
            "ipc": {
                "kind": "file",
                "path": f"rank_00000/gen000000/bucket{index:05d}.bin",
                "offset": 0,
            },
        })
    request = {
        "schema_version": 1,
        "command": "dense_quorum_generation",
        "transport": COMPILED_MPICH_TRANSPORT,
        "run_id": "parser-contract",
        "rank": 0,
        "world_size": 1,
        "generation": 0,
        "base_generation": 0,
        "base_checkpoint": None,
        "quorum": 1,
        "timeout_s": 3.0,
        "bucket_bytes_target": 64,
        "header_bytes": 2,
        "payload_bytes": sum(item["nbytes"] for item in descriptors),
        "header_path": "rank_00000/gen000000/header.json",
        "bucket_descriptors": descriptors,
    }
    request_path = tmp_path / "request.gen000000.json"
    request_path.write_text(stable_json_dumps(request) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update({
        "ARTIFACT_DIR": str(tmp_path / "build"),
        "OUT": str(tmp_path / "build" / "compiled_mpich_dense_helper"),
    })
    subprocess.run(
        ["bash", "scripts/frontier/build_compiled_mpich_dense_helper.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    completed = subprocess.run(
        [str(tmp_path / "build" / "compiled_mpich_dense_helper"), "--request", str(request_path), "--validate-request"],
        check=True,
        text=True,
        capture_output=True,
    )
    parsed = json.loads(completed.stdout)
    assert parsed["bucket_count"] == 80
    assert parsed["bucket_paths"] == [
        f"rank_00000/gen000000/bucket{index:05d}.bin"
        for index in range(80)
    ]
