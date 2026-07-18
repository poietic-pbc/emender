from __future__ import annotations

import time

import pytest

from ndm.native_transport import (
    NativeTransport,
    TransportError,
    decode_endpoint_record,
)


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
