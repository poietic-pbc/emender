import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoUpdate
from ndm.async_diloco import stable_json_dumps
import ndm.async_diloco_compiled_mpich as compiled_mpich
from ndm.async_diloco_compiled_mpich import (
    COMPILED_MPICH_TRANSPORT,
    CompiledMpichHelperConfig,
    load_received_payloads,
    resolve_compiled_mpich_helper_library,
    run_compiled_mpich_dense_quorum,
    write_compiled_mpich_request,
)
from ndm.async_diloco_mpi import pack_dense_update


def _update(value=1.0):
    return AsyncDiLoCoUpdate(
        worker_id="node-00000",
        base_generation=0,
        delta={"w": torch.tensor([float(value), float(value) + 1.0])},
        tokens=8,
        local_steps=1,
        loss_moving_average={"loss": 1.25, "loss_100": 1.25},
    )


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
            "transport": "compiled-cray-mpich-helper-p2p",
            "status": "advanced",
            "rank": payload["rank"],
            "generation": payload["generation"],
            "base_generation": payload["base_generation"],
            "accepted_ranks": [payload["rank"]],
            "timed_out_ranks": [],
            "failed_ranks": [],
            "stale_ranks": [],
            "bytes_sent": payload["payload_bytes"],
            "bytes_received": payload["payload_bytes"],
            "helper_exit_code": 0,
            "mpi": {
                "provided_thread_level": "MPI_THREAD_SERIALIZED",
                "world_size": payload["world_size"],
                "root_rank": 0,
            },
            "received_payloads": [{
                "rank": payload["rank"],
                "header_path": payload["header_path"],
                "bucket_paths": [d["ipc"]["path"] for d in payload["bucket_descriptors"]],
            }],
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
    assert payload["global_generations"][0]["metrics"]["accepted_updates"] == 1
    assert payload["latest_generation"] == -1


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
