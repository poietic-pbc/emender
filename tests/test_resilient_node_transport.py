import json
import hashlib
import socket
import sys
import threading
import time

import pytest
import torch

from ndm.resilient_node_quorum import GenerationFence
from ndm.resilient_node_transport import (
    BoundedNodeManagerBulkStream,
    DiskBucketSpool,
    NodeManagerClient,
    NodeStepSupervisor,
    QuorumTransportServer,
    TransportConfig,
    decode_f64,
    encode_f64,
    exchange_dense_delta,
)


def test_bulk_stream_separates_control_from_bounded_chunk_payloads(tmp_path):
    server, thread = _server(tmp_path, quorum=1, buckets=1)
    client = NodeManagerClient(
        "127.0.0.1", server.server_address[1], "n0",
        DiskBucketSpool(tmp_path / "stream-spool", 1024), timeout_s=2,
        max_bucket_bytes=8)
    stream = BoundedNodeManagerBulkStream(client, max_chunk_bytes=8)
    header, chunks = stream.exchange_chunks(
        GenerationFence("run", 4, 0, 1), (encode_f64((1,)), encode_f64((2,))), weight=7)
    assert tuple(decode_f64(item) for item in chunks) == ((1.0,), (2.0,))
    assert sorted(attempt for generation, attempt in server._generations if generation == 4) == [0, 1]
    assert all(not state.updates and not state.aggregates
               for key, state in server._generations.items() if key[0] == 4)
    assert stream.high_water_bytes == 8
    assert "payload" not in header
    with pytest.raises(BufferError, match="bounded"):
        stream.exchange_chunks(GenerationFence("run", 5, 0, 2), (b"x" * 9,), weight=1)
    server.shutdown(); server.server_close(); thread.join(2)


def _server(tmp_path, *, quorum=2, buckets=2, deadline=2.0):
    server = QuorumTransportServer(
        ("127.0.0.1", 0),
        TransportConfig("run", quorum, buckets, 1024, deadline),
        tmp_path / "metadata",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _exchange(server, tmp_path, node, values, result, *, weight=1, fence=None):
    fence = fence or GenerationFence("run", 0, 0, 1)
    client = NodeManagerClient(
        "127.0.0.1", server.server_address[1], node,
        DiskBucketSpool(tmp_path / node, 4096), timeout_s=4,
    )
    try:
        result[node] = client.exchange(
            fence, tuple(encode_f64(bucket) for bucket in values), weight=weight
        )
    except Exception as error:  # assertions inspect the concrete exception
        result[node] = error


def test_real_tcp_missing_node_advances_and_redistributes_exact_weighted_mean(tmp_path):
    server, thread = _server(tmp_path, quorum=2)
    result = {}
    workers = [
        threading.Thread(target=_exchange, args=(server, tmp_path, "n0", ((1, 3), (5,)), result),
                         kwargs={"weight": 1}),
        threading.Thread(target=_exchange, args=(server, tmp_path, "n1", ((5, 7), (9,)), result),
                         kwargs={"weight": 3}),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    server.shutdown(); server.server_close(); thread.join(2)

    # n2 is completely missing; two independent TCP peers still commit.
    assert set(result) == {"n0", "n1"}
    for header, buckets in result.values():
        assert header["accepted_nodes"] == ["n0", "n1"]
        assert decode_f64(buckets[0]) == pytest.approx((4.0, 6.0))
        assert decode_f64(buckets[1]) == pytest.approx((8.0,))
    manifest = json.loads(next((tmp_path / "metadata/network-generations").iterdir()).read_text())
    assert manifest["accepted_nodes"] == ["n0", "n1"]
    assert "payload" not in json.dumps(manifest)
    assert not list((tmp_path / "n0").glob("*.bucket"))


def test_stale_epoch_rejected_and_quorum_deadline_fails_closed(tmp_path):
    server, thread = _server(tmp_path, quorum=2, buckets=1, deadline=.2)
    result = {}
    stale = GenerationFence("run", 0, 0, 0)
    _exchange(server, tmp_path, "stale", ((1,),), result, fence=stale)
    assert "epoch fence" in str(result["stale"])
    _exchange(server, tmp_path, "alone", ((2,),), result)
    assert "quorum lost" in str(result["alone"])
    # Failed generations remain retained for replay/reassignment.
    assert list((tmp_path / "alone").glob("*.bucket"))
    server.shutdown(); server.server_close(); thread.join(2)


def test_restart_replays_retained_bucket_and_catches_up(tmp_path):
    server, thread = _server(tmp_path, quorum=2, buckets=1, deadline=2)
    fence = GenerationFence("run", 0, 0, 1)
    spool = DiskBucketSpool(tmp_path / "restart-node", 1024)
    spool.retain(fence, 0, encode_f64((2,)))
    result = {}
    threads = [
        threading.Thread(target=_exchange, args=(server, tmp_path, "healthy", ((4,),), result)),
        threading.Thread(target=_exchange, args=(server, tmp_path, "restart-node", ((2,),), result)),
    ]
    for item in threads: item.start()
    for item in threads: item.join(5)
    assert decode_f64(result["restart-node"][1][0]) == pytest.approx((3,))
    server.shutdown(); server.server_close(); thread.join(2)


def test_disk_spool_is_bounded_and_checksum_protocol_rejects_oversize(tmp_path):
    spool = DiskBucketSpool(tmp_path / "spool", 8)
    fence = GenerationFence("run", 0, 0, 1)
    spool.retain(fence, 0, b"12345678")
    with pytest.raises(BufferError, match="retention limit"):
        spool.retain(fence, 1, b"x")
    assert spool.bytes_used == 8


def test_replay_spool_prunes_old_generations(tmp_path):
    spool = DiskBucketSpool(tmp_path / "spool", 1024)
    for generation in range(4):
        spool.retain(GenerationFence("run", generation, 0, 1), 0, b"x")
    spool.prune(keep_generations=2)
    assert sorted(path.name for path in spool.root.glob("*.bucket")) == [
        "g2-a0-e1-b0.bucket", "g3-a0-e1-b0.bucket",
    ]


def test_nonfinite_and_conflicting_duplicate_payloads_are_rejected(tmp_path):
    server, thread = _server(tmp_path, quorum=2, buckets=1, deadline=1)
    base = {"op": "submit", "run_id": "run", "node_id": "n0", "generation": 0,
            "attempt": 0, "coordinator_epoch": 1, "bucket": 0, "weight": 1}
    with pytest.raises(ValueError, match="nonfinite"):
        server.submit(base, encode_f64((float("nan"),)))
    server.submit(base, encode_f64((1.0,)))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        server.submit(base, encode_f64((2.0,)))
    server.shutdown(); server.server_close(); thread.join(2)


def test_heartbeat_eviction_and_fenced_apply_acknowledgement(tmp_path):
    server = QuorumTransportServer(
        ("127.0.0.1", 0), TransportConfig("run", 1, 1, 1024, 2, .01, .05),
        tmp_path / "metadata",
    )
    incomplete = {"op": "submit", "run_id": "run", "node_id": "old",
                  "generation": 0, "attempt": 0, "coordinator_epoch": 1,
                  "bucket": 0, "weight": 1}
    # Quorum one freezes immediately, so exercise heartbeat eviction on another attempt.
    state = server.generation(1, 0)
    state.last_heartbeat["old"] = time.monotonic() - 1
    state.updates["old"] = {}
    assert server.evict_expired(1, 0) == ("old",)
    committed = server.submit(incomplete, encode_f64((3.0,)))
    fence = GenerationFence("run", 0, 0, 1)
    identity = hashlib.sha256(b"".join(committed.aggregates.values())).hexdigest()
    with pytest.raises(ValueError, match="identity"):
        server.acknowledge_apply(fence, "old", "bad")
    server.acknowledge_apply(fence, "old", identity)
    assert server.wait_for_apply(fence) == ("old",)
    server.server_close()


def test_supervisor_kills_only_stuck_step_and_healthy_step_survives():
    supervisor = NodeStepSupervisor(2)
    healthy = supervisor.run([sys.executable, "-c", "print('healthy')"])
    assert healthy.returncode == 0 and healthy.stdout.strip() == b"healthy"
    with pytest.raises(TimeoutError, match="deadline"):
        NodeStepSupervisor(.2).run([sys.executable, "-c", "import time; time.sleep(30)"])
    # The supervisor itself remains usable after terminating the bad child.
    assert supervisor.run([sys.executable, "-c", "pass"]).returncode == 0


def test_e97_dense_delta_is_weighted_redistributed_and_applied(tmp_path):
    server, thread = _server(tmp_path, quorum=2, buckets=2)
    fence = GenerationFence("run", 0, 0, 1)
    base = {"a": torch.tensor([10.0, 20.0]), "z": torch.tensor([30.0])}
    results = {}

    def worker(node, delta, weight):
        client = NodeManagerClient(
            "127.0.0.1", server.server_address[1], node,
            DiskBucketSpool(tmp_path / node, 4096), timeout_s=4,
        )
        results[node] = exchange_dense_delta(
            client, fence, base, delta, weight=weight, bucket_bytes=16, eta_outer=.5
        )

    threads = [
        threading.Thread(target=worker, args=("n0", {"a": torch.tensor([1., 3.]), "z": torch.tensor([5.])}, 1)),
        threading.Thread(target=worker, args=("n1", {"a": torch.tensor([5., 7.]), "z": torch.tensor([9.])}, 3)),
    ]
    for item in threads: item.start()
    for item in threads: item.join(5)
    server.shutdown(); server.server_close(); thread.join(2)

    for header, state in results.values():
        assert header["accepted_nodes"] == ["n0", "n1"]
        assert torch.equal(state["a"], torch.tensor([12., 23.]))
        assert torch.equal(state["z"], torch.tensor([34.]))
