from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import socket
import struct
import sys
import threading
import time

from ndm.native_transport import decode_endpoint_record


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/frontier/native_dataplane_2n_controller.py"


def _module():
    spec = importlib.util.spec_from_file_location("native_dataplane_2n_controller", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _endpoint(rank: int, *, epoch: int = 1) -> bytes:
    body = bytearray()
    body += bytes([1]) * 16
    body += (1).to_bytes(8, "little")
    body += bytes([rank + 2]) * 16
    body += bytes([rank + epoch + 5]) * 16
    body += epoch.to_bytes(8, "little")
    body += (time.time_ns() + 60_000_000_000).to_bytes(8, "little")
    for text in (b"cxi", b"cxi-fabric", b"cxi-domain"):
        body += len(text).to_bytes(2, "little") + text
    body += (7).to_bytes(4, "little")
    name = b"opaque-" + bytes([rank, epoch])
    body += len(name).to_bytes(2, "little") + name
    return bytes(body) + hashlib.sha256(b"emender-ndp-endpoint-v1\0" + body).digest()


def test_metadata_exchange_returns_only_the_other_validated_endpoint():
    module = _module()
    records = {rank: decode_endpoint_record(_endpoint(rank)) for rank in (0, 1)}
    result: dict[str, object] = {}

    pairs = [socket.socketpair(), socket.socketpair()]
    thread = threading.Thread(
        target=lambda: result.update(
            parsed=module._phase(
                {0: pairs[0][0], 1: pairs[1][0]}, phase=0,
                expected_provider="cxi", expected_run_key=None,
                expected_fence=None,
            )
        )
    )
    thread.start()
    for rank in (0, 1):
        request = module.REQUEST.pack(
            module.MAGIC, 1, 0, rank, len(records[rank].encoded), time.time_ns()
        ) + records[rank].encoded
        pairs[rank][1].sendall(request)
    replies = []
    for rank in (0, 1):
        raw = module._recv_exact(pairs[rank][1], module.REPLY_PREFIX.size)
        reply = module.REPLY_PREFIX.unpack(raw)
        encoded = module._recv_exact(pairs[rank][1], reply[-1])
        replies.append((reply, decode_endpoint_record(encoded)))
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert replies[0][0][2] == 1 and replies[0][1].worker_key == records[1].worker_key
    assert replies[1][0][2] == 0 and replies[1][1].worker_key == records[0].worker_key
    for pair in pairs:
        pair[0].close()
        pair[1].close()


def test_clock_attestation_measures_offset_not_staggered_launch_time():
    module = _module()
    records = {rank: decode_endpoint_record(_endpoint(rank)) for rank in (0, 1)}
    result: dict[str, object] = {}
    pairs = [socket.socketpair(), socket.socketpair()]
    thread = threading.Thread(
        target=lambda: result.update(
            parsed=module._phase(
                {0: pairs[0][0], 1: pairs[1][0]}, phase=0,
                expected_provider="cxi", expected_run_key=None,
                expected_fence=None,
            )
        )
    )
    thread.start()
    for rank in (0, 1):
        if rank == 1:
            time.sleep(0.3)
        request = module.REQUEST.pack(
            module.MAGIC, 1, 0, rank, len(records[rank].encoded), time.time_ns()
        ) + records[rank].encoded
        pairs[rank][1].sendall(request)
    for rank in (0, 1):
        raw = module._recv_exact(pairs[rank][1], module.REPLY_PREFIX.size)
        reply = module.REPLY_PREFIX.unpack(raw)
        module._recv_exact(pairs[rank][1], reply[-1])
    thread.join(timeout=2)
    assert not thread.is_alive()
    _, clock_offset_delta_ns = result["parsed"]
    assert clock_offset_delta_ns < 50_000_000
    for pair in pairs:
        pair[0].close()
        pair[1].close()


def test_clock_attestation_uses_first_observed_hello_not_delayed_phase_read():
    module = _module()
    records = {rank: decode_endpoint_record(_endpoint(rank)) for rank in (0, 1)}
    pairs = [socket.socketpair(), socket.socketpair()]
    observed_controller_clocks: dict[int, int] = {}
    for rank in (0, 1):
        client_clock = time.time_ns()
        request = module.REQUEST.pack(
            module.MAGIC, 1, 0, rank, len(records[rank].encoded), client_clock
        ) + records[rank].encoded
        pairs[rank][1].sendall(request)
        observed_controller_clocks[rank] = time.time_ns()
        if rank == 0:
            time.sleep(0.3)

    records_by_rank, clock_offset_delta_ns = module._phase(
        {0: pairs[0][0], 1: pairs[1][0]}, phase=0,
        expected_provider="cxi", expected_run_key=None, expected_fence=None,
        observed_controller_clocks=observed_controller_clocks,
    )
    assert set(records_by_rank) == {0, 1}
    assert clock_offset_delta_ns < 50_000_000
    for rank in (0, 1):
        raw = module._recv_exact(pairs[rank][1], module.REPLY_PREFIX.size)
        reply = module.REPLY_PREFIX.unpack(raw)
        module._recv_exact(pairs[rank][1], reply[-1])
    for pair in pairs:
        pair[0].close()
        pair[1].close()


def test_clock_attestation_samples_persistent_clients_independently():
    module = _module()
    records = {rank: decode_endpoint_record(_endpoint(rank)) for rank in (0, 1)}
    result: dict[str, object] = {}
    pairs = [socket.socketpair(), socket.socketpair()]
    thread = threading.Thread(
        target=lambda: result.update(
            parsed=module._phase(
                {0: pairs[0][0], 1: pairs[1][0]}, phase=1,
                expected_provider="cxi", expected_run_key=records[0].run_key,
                expected_fence=records[0].fence_epoch,
            )
        )
    )
    thread.start()
    for rank in (1, 0):
        client_clock = time.time_ns()
        request = module.REQUEST.pack(
            module.MAGIC, 1, 1, rank, len(records[rank].encoded), client_clock
        ) + records[rank].encoded
        pairs[rank][1].sendall(request)
        if rank == 1:
            time.sleep(0.3)
    for rank in (0, 1):
        raw = module._recv_exact(pairs[rank][1], module.REPLY_PREFIX.size)
        reply = module.REPLY_PREFIX.unpack(raw)
        module._recv_exact(pairs[rank][1], reply[-1])
    thread.join(timeout=2)
    assert not thread.is_alive()
    _, clock_offset_delta_ns = result["parsed"]
    assert clock_offset_delta_ns < 50_000_000
    for pair in pairs:
        pair[0].close()
        pair[1].close()
