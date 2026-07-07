import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoUpdate
from ndm.async_diloco_compiled_mpich import (
    COMPILED_MPICH_TRANSPORT,
    CompiledMpichHelperConfig,
    load_received_payloads,
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


def test_compiled_mpich_helper_invocation_contract_and_root_merge(tmp_path):
    helper = tmp_path / "mock_helper.py"
    helper.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

request = Path(sys.argv[sys.argv.index('--request') + 1])
payload = json.loads(request.read_text())
result = {
    'schema_version': 1,
    'transport': 'compiled-cray-mpich-helper-p2p',
    'status': 'advanced',
    'rank': payload['rank'],
    'generation': payload['generation'],
    'base_generation': payload['base_generation'],
    'accepted_ranks': [payload['rank']],
    'timed_out_ranks': [],
    'failed_ranks': [],
    'stale_ranks': [],
    'bytes_sent': payload['payload_bytes'],
    'bytes_received': payload['payload_bytes'],
    'helper_exit_code': 0,
    'mpi': {'provided_thread_level': 'MPI_THREAD_SERIALIZED', 'world_size': payload['world_size'], 'root_rank': 0},
    'received_payloads': [{
        'rank': payload['rank'],
        'header_path': payload['header_path'],
        'bucket_paths': [d['ipc']['path'] for d in payload['bucket_descriptors']],
    }],
}
request.with_name(f"result.gen{payload['generation']:06d}.json").write_text(json.dumps(result, sort_keys=True) + "\\n")
""",
        encoding="utf-8",
    )
    helper.chmod(0o755)

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


def test_compiled_mpich_helper_invocation_errors_on_nonzero(tmp_path):
    helper = tmp_path / "bad_helper.py"
    helper.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(7)\n", encoding="utf-8")
    helper.chmod(0o755)

    with pytest.raises(RuntimeError, match="compiled MPICH helper failed"):
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
