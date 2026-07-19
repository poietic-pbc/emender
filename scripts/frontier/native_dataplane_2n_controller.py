#!/usr/bin/env python3
"""Bounded metadata-only membership exchange for the native two-node gate.

The controller never receives a tensor frame.  It validates the two leased
libfabric endpoint records, then returns each opaque record to the other
native service.  Fault mode repeats the exchange after rank 1 opens a new
endpoint/incarnation; rank 0 keeps its original persistent endpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import time

MAGIC = b"NDPCTL1\0"
REQUEST = struct.Struct("<8sIIIIQ")
REPLY_PREFIX = struct.Struct("<8sII16s16sQQI")
SCHEMA = "emender-native-dataplane-membership-v1"


@dataclass(frozen=True)
class EndpointRecord:
    encoded: bytes
    run_key: bytes
    fence_epoch: int
    worker_key: bytes
    incarnation: bytes
    endpoint_epoch: int
    expires_unix_ns: int
    provider: str
    fabric: str
    domain: str
    addr_format: int
    endpoint_name: bytes


def decode_endpoint_record(encoded: bytes) -> EndpointRecord:
    """Decode the bounded endpoint record without importing model code."""
    encoded = bytes(encoded)
    minimum = 16 + 8 + 16 + 16 + 8 + 8 + 2 + 2 + 2 + 4 + 2 + 32
    if not minimum <= len(encoded) <= 4096:
        raise ValueError("native endpoint record byte bound violated")
    body, digest = encoded[:-32], encoded[-32:]
    if digest != hashlib.sha256(b"emender-ndp-endpoint-v1\0" + body).digest():
        raise ValueError("native endpoint record checksum mismatch")
    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(body):
            raise ValueError("truncated native endpoint record")
        value = body[offset:offset + size]
        offset += size
        return value

    def integer(size: int) -> int:
        return int.from_bytes(take(size), "little")

    run_key = take(16)
    fence_epoch = integer(8)
    worker_key, incarnation = take(16), take(16)
    endpoint_epoch, expires_unix_ns = integer(8), integer(8)

    def text() -> str:
        value = take(integer(2)).decode("utf-8")
        if not value or "\0" in value:
            raise ValueError("invalid native endpoint provider fact")
        return value

    provider, fabric, domain = text(), text(), text()
    addr_format = integer(4)
    endpoint_name = take(integer(2))
    if not endpoint_name or offset != len(body):
        raise ValueError("invalid native endpoint address")
    return EndpointRecord(
        encoded, run_key, fence_epoch, worker_key, incarnation,
        endpoint_epoch, expires_unix_ns, provider, fabric, domain,
        addr_format, endpoint_name,
    )


def _recv_exact(stream: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RuntimeError("native endpoint client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_request(
    stream: socket.socket, *, expected_phase: int
) -> tuple[int, int, int, EndpointRecord]:
    header = _recv_exact(stream, REQUEST.size)
    controller_unix_ns = time.time_ns()
    magic, version, phase, rank, record_bytes, client_unix_ns = REQUEST.unpack(header)
    if magic != MAGIC or version != 1 or phase != expected_phase:
        raise RuntimeError("native endpoint exchange protocol mismatch")
    if rank not in (0, 1) or not 0 < record_bytes <= 4096:
        raise RuntimeError("native endpoint exchange bounds violation")
    record = decode_endpoint_record(_recv_exact(stream, record_bytes))
    return rank, client_unix_ns, controller_unix_ns, record


def _send_peer(stream: socket.socket, rank: int, record: EndpointRecord) -> None:
    stream.sendall(
        REPLY_PREFIX.pack(
            MAGIC,
            0,
            rank,
            record.worker_key,
            record.incarnation,
            record.endpoint_epoch,
            record.expires_unix_ns,
            len(record.encoded),
        )
        + record.encoded
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _phase(
    clients: dict[int, socket.socket],
    *,
    phase: int,
    expected_provider: str,
    expected_run_key: bytes | None,
    expected_fence: int | None,
) -> tuple[dict[int, EndpointRecord], int]:
    records: dict[int, EndpointRecord] = {}
    clock_offsets: dict[int, int] = {}
    for expected_rank in (0, 1):
        rank, client_clock, controller_clock, record = _read_request(
            clients[expected_rank], expected_phase=phase
        )
        if rank != expected_rank or rank in records:
            raise RuntimeError("duplicate or misrouted native endpoint rank")
        if record.provider != expected_provider:
            raise RuntimeError(
                f"native endpoint selected {record.provider!r}, expected {expected_provider!r}"
            )
        if record.expires_unix_ns <= time.time_ns():
            raise RuntimeError("native endpoint was already expired")
        if expected_run_key is not None and record.run_key != expected_run_key:
            raise RuntimeError("native endpoints disagree on run identity")
        if expected_fence is not None and record.fence_epoch != expected_fence:
            raise RuntimeError("native endpoint fence changed")
        records[rank] = record
        clock_offsets[rank] = client_clock - controller_clock
    if records[0].run_key != records[1].run_key or records[0].fence_epoch != records[1].fence_epoch:
        raise RuntimeError("native endpoints are not in one fenced run")
    if records[0].worker_key == records[1].worker_key:
        raise RuntimeError("native endpoint worker identities are not distinct")
    _send_peer(clients[0], 1, records[1])
    _send_peer(clients[1], 0, records[0])
    return records, abs(clock_offsets[0] - clock_offsets[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--mode", choices=("clean", "fault"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accept-deadline-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.accept_deadline_seconds <= 120:
        raise SystemExit("invalid bounded controller configuration")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.host, args.port))
    listener.listen(2)
    listener.settimeout(args.accept_deadline_seconds)
    clients: dict[int, socket.socket] = {}
    try:
        while len(clients) != 2:
            stream, _address = listener.accept()
            stream.settimeout(60.0)
            # The rank is part of the request, so peek just the fixed header and
            # leave it queued for the phase decoder.
            header = stream.recv(REQUEST.size, socket.MSG_PEEK | socket.MSG_WAITALL)
            if len(header) != REQUEST.size:
                raise RuntimeError("truncated native endpoint hello")
            magic, version, phase, rank, _bytes, _clock = REQUEST.unpack(header)
            if magic != MAGIC or version != 1 or phase != 0 or rank not in (0, 1):
                raise RuntimeError("invalid native endpoint hello")
            if rank in clients:
                raise RuntimeError("duplicate native endpoint connection")
            clients[rank] = stream

        initial, initial_skew_ns = _phase(
            clients,
            phase=0,
            expected_provider=args.provider,
            expected_run_key=None,
            expected_fence=None,
        )
        phases = [initial]
        skews = [initial_skew_ns]
        if args.mode == "fault":
            replacement, replacement_skew_ns = _phase(
                clients,
                phase=1,
                expected_provider=args.provider,
                expected_run_key=initial[0].run_key,
                expected_fence=initial[0].fence_epoch,
            )
            if replacement[0].encoded != initial[0].encoded:
                raise RuntimeError("fault payload unexpectedly replaced rank-0 endpoint")
            if replacement[1].incarnation == initial[1].incarnation:
                raise RuntimeError("fault payload did not create a new rank-1 incarnation")
            if replacement[1].endpoint_epoch <= initial[1].endpoint_epoch:
                raise RuntimeError("fault payload did not advance rank-1 endpoint epoch")
            phases.append(replacement)
            skews.append(replacement_skew_ns)

        value = {
            "schema": SCHEMA,
            "status": "passed",
            "mode": args.mode,
            "provider": args.provider,
            "endpoint_type": "FI_EP_RDM",
            "fence_epoch": initial[0].fence_epoch,
            "run_key": initial[0].run_key.hex(),
            "phase_count": len(phases),
            "two_endpoints": True,
            "clock_attestation": "client_minus_controller_offset_delta",
            "max_clock_skew_ms": max(skews) / 1_000_000,
            "phases": [
                {
                    str(rank): {
                        "provider": record.provider,
                        "fabric": record.fabric,
                        "domain": record.domain,
                        "addr_format": record.addr_format,
                        "worker_key": record.worker_key.hex(),
                        "incarnation": record.incarnation.hex(),
                        "endpoint_epoch": record.endpoint_epoch,
                        "expires_unix_ns": record.expires_unix_ns,
                        "endpoint_record_sha256": hashlib.sha256(record.encoded).hexdigest(),
                        "endpoint_name_sha256": hashlib.sha256(record.endpoint_name).hexdigest(),
                    }
                    for rank, record in sorted(phase.items())
                }
                for phase in phases
            ],
        }
        _atomic_json(Path(args.output), value)
        print(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        for stream in clients.values():
            stream.close()
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
