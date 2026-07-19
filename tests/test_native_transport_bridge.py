from __future__ import annotations

import hashlib
import os
import time

import pytest

from ndm.native_transport import (
    NativeTransport,
    NativeTransportLibrary,
    TransportError,
    decode_endpoint_record,
)
from ndm.native_e97_runtime import (
    decode_credit_frame_fd, decode_owner_frame_fd, encode_credit_frame_fd,
    encode_owner_frame_fd,
)
from ndm.native_dataplane import create_memfd


def _open(tag: int) -> NativeTransport:
    return NativeTransport.open(
        provider="tcp;ofi_rxm", production=False, bind_node="127.0.0.1",
        payload_max=4096, tx_slots=1, rx_slots=1,
        resident_limit_bytes=1 << 20, deadline_s=10,
    )


def test_fenced_endpoint_exchange_installs_only_current_native_routes():
    expiry = time.time_ns() + 8_000_000_000
    with _open(1) as first, _open(2) as second:
        first_record = first.bind(
            run_key="run", fence_epoch=7, worker_key="node-0",
            incarnation="node-0-boot", endpoint_epoch=1,
            expires_unix_ns=expiry,
        )
        second_record = second.bind(
            run_key="run", fence_epoch=7, worker_key="node-1",
            incarnation="node-1-boot", endpoint_epoch=1,
            expires_unix_ns=expiry,
        )

        assert first_record.provider == second_record.provider == "tcp;ofi_rxm"
        assert first_record.fence_epoch == second_record.fence_epoch == 7
        assert first_record.run_key == second_record.run_key
        assert first_record.endpoint_name != second_record.endpoint_name
        assert decode_endpoint_record(first_record.encoded) == first_record

        second_peer = first.upsert(second_record)
        first_peer = second.upsert(first_record)
        assert second_peer > 0 and first_peer > 0
        assert first.metrics.live_peers == second.metrics.live_peers == 1
        assert first.metrics.in_flight_bytes == second.metrics.retained_bytes == 0

        first.remove(second_peer)
        second.remove(first_peer)
        assert first.metrics.live_peers == second.metrics.live_peers == 0


def test_endpoint_checksum_and_fence_conflict_fail_closed():
    expiry = time.time_ns() + 8_000_000_000
    with _open(3) as first, _open(4) as second:
        first.bind(
            run_key="run", fence_epoch=11, worker_key="node-0",
            incarnation="node-0-boot", endpoint_epoch=1,
            expires_unix_ns=expiry,
        )
        stale = second.bind(
            run_key="run", fence_epoch=10, worker_key="node-1",
            incarnation="node-1-boot", endpoint_epoch=1,
            expires_unix_ns=expiry,
        )
        with pytest.raises(TransportError) as rejected:
            first.upsert(stale)
        assert rejected.value.code == -6

        corrupt = bytearray(stale.encoded)
        corrupt[-1] ^= 1
        with pytest.raises(ValueError, match="checksum"):
            decode_endpoint_record(corrupt)


def test_frozen_owner_frame_moves_memfd_to_memfd_over_native_provider():
    """The Python bridge supplies descriptors, never a dense socket payload."""
    expiry = time.time_ns() + 8_000_000_000
    with _open(6) as first, _open(7) as second:
        first_record = first.bind(
            run_key="run", fence_epoch=13, worker_key="node-0",
            incarnation="node-0-boot", endpoint_epoch=1,
            expires_unix_ns=expiry)
        second_record = second.bind(
            run_key="run", fence_epoch=13, worker_key="node-1",
            incarnation="node-1-boot", endpoint_epoch=1,
            expires_unix_ns=expiry)
        second_peer = first.upsert(second_record)
        first_peer = second.upsert(first_record)
        payload = bytes((index * 29 + 7) & 0xFF for index in range(1024))
        source_fd = create_memfd("native-owner-source")
        destination_fd = create_memfd("native-owner-destination")
        credit_fd = received_credit_fd = frame_fd = -1
        try:
            os.ftruncate(source_fd, len(payload)); os.pwrite(source_fd, payload, 0)
            deadline = time.time_ns() + 5_000_000_000
            credit_fd = encode_credit_frame_fd(
                payload_offset=128, payload_bytes=512, payload_max=4096,
                run_id="run", fence_epoch=13, generation=9, attempt=2,
                owner_epoch=4, worker_id="node-1", incarnation="node-1-boot",
                layout_digest=bytes.fromhex("11" * 32),
                base_digest=bytes.fromhex("22" * 32),
                permitted_root=bytes.fromhex("33" * 32), weight=91,
                chunk_index=0, chunk_count=1, deadline_unix_ns=deadline,
                message_seq=(10 << 32) + 1)
            second.send_fd(
                first_peer, credit_fd, frame_bytes=320,
                deadline_unix_ns=deadline)
            received_credit_fd = create_memfd("native-owner-credit-rx")
            os.ftruncate(received_credit_fd, 320)
            credit = None
            while credit is None and time.time_ns() < deadline:
                credit = first.receive_into_fd(received_credit_fd, capacity=320)
                if credit is None:
                    first.poll(timeout_ms=10)
            assert credit == (second_peer, 320)
            grant = decode_credit_frame_fd(
                received_credit_fd, payload_max=4096,
                expected={
                    "run_key": hashlib.sha256(b"run").digest()[:16],
                    "fence_epoch": 13, "generation": 9, "attempt": 2,
                    "owner_epoch": 4,
                    "worker_key": hashlib.sha256(b"node-1").digest()[:16],
                    "incarnation": hashlib.sha256(b"node-1-boot").digest()[:16],
                    "layout_digest": bytes.fromhex("11" * 32),
                    "base_digest": bytes.fromhex("22" * 32),
                    "result_root": bytes.fromhex("33" * 32),
                    "weight": 91, "payload_offset": 128,
                    "payload_bytes": 512, "credit": 512,
                    "chunk_index": 0, "chunk_count": 1,
                })
            assert grant["message_seq"] == (10 << 32) + 1
            frame_fd, frame_bytes = encode_owner_frame_fd(
                source_fd=source_fd, payload_offset=128, payload_bytes=512,
                payload_max=4096, run_id="run", fence_epoch=13, generation=9,
                attempt=2, owner_epoch=4, worker_id="node-0",
                incarnation="node-0-boot", layout_digest=bytes.fromhex("11" * 32),
                base_digest=bytes.fromhex("22" * 32),
                result_root=bytes.fromhex("33" * 32), weight=91,
                chunk_index=0, chunk_count=1, deadline_unix_ns=deadline)
            first.send_fd(
                second_peer, frame_fd, frame_bytes=frame_bytes,
                deadline_unix_ns=deadline)
            capacity = 4096 + 320
            os.ftruncate(destination_fd, capacity)
            received = None
            while received is None and time.time_ns() < deadline:
                received = second.receive_into_fd(destination_fd, capacity=capacity)
                if received is None:
                    second.poll(timeout_ms=10)
            assert received == (first_peer, frame_bytes)
            actual = decode_owner_frame_fd(
                destination_fd, frame_bytes=frame_bytes, payload_max=4096,
                expected={
                    "run_key": hashlib.sha256(b"run").digest()[:16],
                    "fence_epoch": 13, "generation": 9, "attempt": 2,
                    "owner_epoch": 4,
                    "worker_key": hashlib.sha256(b"node-0").digest()[:16],
                    "incarnation": hashlib.sha256(b"node-0-boot").digest()[:16],
                    "layout_digest": bytes.fromhex("11" * 32),
                    "base_digest": bytes.fromhex("22" * 32),
                    "result_root": bytes.fromhex("33" * 32),
                    "weight": 91, "chunk_index": 0, "chunk_count": 1,
                })
            assert actual["payload_bytes"] == 512
            assert os.pread(destination_fd, 512, 320) == payload[128:640]
            assert first.metrics.useful_tx_bytes == second.metrics.useful_rx_bytes
            assert first.metrics.useful_tx_bytes == 512
            assert first.metrics.wire_tx_bytes == second.metrics.wire_rx_bytes == frame_bytes
        finally:
            for fd in (credit_fd, received_credit_fd, frame_fd):
                if fd >= 0:
                    os.close(fd)
            os.close(source_fd); os.close(destination_fd)


def test_production_provider_and_poll_bounds_cannot_be_weakened():
    with pytest.raises(ValueError, match="exact provider cxi"):
        NativeTransport.open(
            provider="tcp;ofi_rxm", production=True, bind_node="127.0.0.1",
            payload_max=4096, tx_slots=1, rx_slots=1,
            resident_limit_bytes=1 << 20,
        )
    with _open(5) as transport:
        with pytest.raises(ValueError, match="bounded"):
            transport.poll(timeout_ms=-1)


def test_production_transport_pins_cxi0_and_omits_hostname_source_bind():
    captured = {}

    class Library:
        @staticmethod
        def ndp_transport_open_v1(config_pointer, handle_pointer):
            config = config_pointer._obj
            captured["domain"] = bytes(config.domain[:config.domain_len])
            captured["bind_node"] = bytes(
                config.bind_node[:config.bind_node_len])
            handle_pointer._obj.value = 41
            return 0

    native = object.__new__(NativeTransportLibrary)
    native.lib = Library()
    transport = NativeTransport.open(
        library=native, provider="cxi", production=True,
        bind_node="frontier00001", payload_max=4096, tx_slots=1, rx_slots=1,
        resident_limit_bytes=1 << 20, deadline_s=10)
    transport.closed = True

    assert captured == {"domain": b"cxi0", "bind_node": b""}
