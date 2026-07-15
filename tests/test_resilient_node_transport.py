import json
import socket
import sys
import threading
import time

import pytest

from ndm.resilient_node_quorum import GenerationFence
from ndm.resilient_node_transport import (
    DiskBucketSpool,
    NodeManagerClient,
    NodeStepSupervisor,
    QuorumTransportServer,
    TransportConfig,
    decode_f64,
    encode_f64,
)


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


def test_supervisor_kills_only_stuck_step_and_healthy_step_survives():
    supervisor = NodeStepSupervisor(2)
    healthy = supervisor.run([sys.executable, "-c", "print('healthy')"])
    assert healthy.returncode == 0 and healthy.stdout.strip() == b"healthy"
    with pytest.raises(TimeoutError, match="deadline"):
        NodeStepSupervisor(.2).run([sys.executable, "-c", "import time; time.sleep(30)"])
    # The supervisor itself remains usable after terminating the bad child.
    assert supervisor.run([sys.executable, "-c", "pass"]).returncode == 0
