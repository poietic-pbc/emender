"""Typed Python owner for the additive native libfabric transport ABI.

Python uses this module only for fenced lifecycle and endpoint metadata.  The
compiled library owns provider selection, registered buffers, progress, frame
validation, point-to-point movement, and bounded cleanup.  In particular, an
endpoint cannot be published until the post-lease controller binds its
``(run, fence, worker, incarnation, endpoint epoch, expiry)`` identity.
"""

from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
from dataclasses import dataclass
from pathlib import Path
import struct
import time


NDP_TRANSPORT_ABI_V1 = 0x00010000
NDP_TRANSPORT_ENDPOINT_MAX = 4096
NDP_TRANSPORT_PROVIDER_MAX = 64
NDP_TRANSPORT_FABRIC_MAX = 128
NDP_TRANSPORT_DOMAIN_MAX = 128
NDP_TRANSPORT_ECREDIT = -10


class TransportError(RuntimeError):
    """Stable native transport error with its ABI result code."""

    def __init__(self, code: int, operation: str, detail: str):
        self.code, self.operation = int(code), operation
        super().__init__(f"{operation}: {detail} ({code})")


class _OpenV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("mode", ctypes.c_uint32), ("tx_slots", ctypes.c_uint32),
        ("rx_slots", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
        ("payload_max", ctypes.c_uint64),
        ("resident_limit_bytes", ctypes.c_uint64),
        ("operation_deadline_unix_ns", ctypes.c_uint64),
        ("telemetry_fd", ctypes.c_int32), ("reserved1", ctypes.c_uint32),
        ("provider_len", ctypes.c_uint32),
        ("provider", ctypes.c_uint8 * NDP_TRANSPORT_PROVIDER_MAX),
        ("require_provider_len", ctypes.c_uint32),
        ("require_provider", ctypes.c_uint8 * NDP_TRANSPORT_PROVIDER_MAX),
        ("fabric_len", ctypes.c_uint32),
        ("fabric", ctypes.c_uint8 * NDP_TRANSPORT_FABRIC_MAX),
        ("domain_len", ctypes.c_uint32),
        ("domain", ctypes.c_uint8 * NDP_TRANSPORT_DOMAIN_MAX),
        ("bind_node_len", ctypes.c_uint32),
        ("bind_node", ctypes.c_uint8 * NDP_TRANSPORT_DOMAIN_MAX),
    ]


class _IdentityV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("run_key", ctypes.c_uint8 * 16), ("fence_epoch", ctypes.c_uint64),
        ("worker_key", ctypes.c_uint8 * 16),
        ("incarnation", ctypes.c_uint8 * 16),
        ("endpoint_epoch", ctypes.c_uint64),
        ("expires_unix_ns", ctypes.c_uint64),
    ]


class _EndpointV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("record_bytes", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
        ("record", ctypes.c_uint8 * NDP_TRANSPORT_ENDPOINT_MAX),
    ]


class _PeerV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("worker_key", ctypes.c_uint8 * 16),
        ("incarnation", ctypes.c_uint8 * 16),
        ("endpoint_epoch", ctypes.c_uint64),
        ("expires_unix_ns", ctypes.c_uint64),
        ("endpoint_name_bytes", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("endpoint_name", ctypes.c_uint8 * NDP_TRANSPORT_ENDPOINT_MAX),
    ]


class _EventV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("event", ctypes.c_uint32), ("status", ctypes.c_int32),
        ("peer_id", ctypes.c_uint64), ("message_seq", ctypes.c_uint64),
        ("useful_bytes", ctypes.c_uint64), ("wire_bytes", ctypes.c_uint64),
        ("provider_errno", ctypes.c_int32), ("reason", ctypes.c_uint32),
        ("detail", ctypes.c_uint8 * 32), ("reserved0", ctypes.c_uint64),
    ]


class _MetricsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("useful_tx_bytes", ctypes.c_uint64), ("useful_rx_bytes", ctypes.c_uint64),
        ("wire_tx_bytes", ctypes.c_uint64), ("wire_rx_bytes", ctypes.c_uint64),
        ("retries", ctypes.c_uint64), ("replay_bytes", ctypes.c_uint64),
        ("duplicate_frames", ctypes.c_uint64),
        ("checksum_rejects", ctypes.c_uint64), ("stale_rejects", ctypes.c_uint64),
        ("cq_errors", ctypes.c_uint64), ("route_errors", ctypes.c_uint64),
        ("in_flight_bytes", ctypes.c_uint64),
        ("in_flight_high_water", ctypes.c_uint64),
        ("retained_bytes", ctypes.c_uint64),
        ("retained_high_water", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("tx_slot_high_water", ctypes.c_uint64),
        ("rx_slot_high_water", ctypes.c_uint64),
        ("latency_count", ctypes.c_uint64), ("latency_total_ns", ctypes.c_uint64),
        ("service_started_unix_ns", ctypes.c_uint64),
        ("last_progress_unix_ns", ctypes.c_uint64),
        ("owner_state", ctypes.c_uint32), ("live_peers", ctypes.c_uint32),
        ("provider_name_len", ctypes.c_uint32),
        ("provider_name", ctypes.c_uint8 * NDP_TRANSPORT_PROVIDER_MAX),
    ]


for _type, _size in (
    (_OpenV1, 592), (_IdentityV1, 80), (_EndpointV1, 4112),
    (_PeerV1, 4160), (_EventV1, 96), (_MetricsV1, 264),
):
    if ctypes.sizeof(_type) != _size:
        raise RuntimeError(f"native transport ABI layout mismatch for {_type.__name__}")


def _versioned(value: ctypes.Structure) -> ctypes.Structure:
    value.struct_size = ctypes.sizeof(value)
    value.abi_version = NDP_TRANSPORT_ABI_V1
    return value


def _key16(value: bytes | str, *, field: str) -> bytes:
    if isinstance(value, str):
        value = hashlib.sha256(value.encode()).digest()[:16]
    value = bytes(value)
    if len(value) != 16 or value == bytes(16):
        raise ValueError(f"{field} must be a nonzero 16-byte key")
    return value


def _write_bytes(target: ctypes.Array, value: bytes) -> None:
    if len(value) > len(target):
        raise ValueError("native transport ABI span exceeds its fixed capacity")
    target[:len(value)] = value


def _write_text(value: ctypes.Structure, field: str, text: str) -> None:
    raw = text.encode("utf-8")
    if not raw or b"\0" in raw:
        raise ValueError(f"{field} must be nonempty UTF-8 without NUL")
    target = getattr(value, field)
    _write_bytes(target, raw)
    setattr(value, f"{field}_len", len(raw))


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
    """Decode and checksum the normative no-padding endpoint record."""
    encoded = bytes(encoded)
    minimum = 16 + 8 + 16 + 16 + 8 + 8 + 2 + 2 + 2 + 4 + 2 + 32
    if not minimum <= len(encoded) <= NDP_TRANSPORT_ENDPOINT_MAX:
        raise ValueError("native endpoint record byte bound violated")
    body, digest = encoded[:-32], encoded[-32:]
    expected = hashlib.sha256(b"emender-ndp-endpoint-v1\0" + body).digest()
    if digest != expected:
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
        raw = take(integer(2))
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 in native endpoint record") from error
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


@dataclass(frozen=True)
class TransportMetrics:
    provider: str
    useful_tx_bytes: int
    useful_rx_bytes: int
    wire_tx_bytes: int
    wire_rx_bytes: int
    retries: int
    replay_bytes: int
    cq_errors: int
    route_errors: int
    in_flight_bytes: int
    in_flight_high_water: int
    retained_bytes: int
    retained_high_water: int
    released_bytes: int
    tx_slot_high_water: int
    rx_slot_high_water: int
    live_peers: int


class NativeTransportLibrary:
    """Loaded, ABI-checked ``libemender_ndp_transport.so.1``."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = self._resolve(path)
        self.lib = ctypes.CDLL(str(self.path), use_errno=True)
        self._declare()
        if int(self.lib.ndp_transport_abi_version()) != NDP_TRANSPORT_ABI_V1:
            raise TransportError(-2, "load", "native transport ABI major mismatch")

    @staticmethod
    def _resolve(path: str | os.PathLike[str] | None) -> Path:
        repository = Path(__file__).resolve().parents[1]
        candidates = [
            Path(path) if path else None,
            Path(os.environ["EMENDER_NDP_TRANSPORT_LIBRARY"])
            if os.environ.get("EMENDER_NDP_TRANSPORT_LIBRARY") else None,
            repository / "build/native-resilient-dataplane/lib/libemender_ndp_transport.so.1",
            repository / "build/native-resilient-dataplane/lib64/libemender_ndp_transport.so.1",
            repository / "build/native-resilient-dataplane/libemender_ndp_transport.so",
            repository / "build/integrate-transport/libemender_ndp_transport.so",
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "libemender_ndp_transport.so.1 was not found; set "
            "EMENDER_NDP_TRANSPORT_LIBRARY or run "
            "scripts/frontier/build_native_resilient_dataplane.sh")

    def _declare(self) -> None:
        lib = self.lib
        lib.ndp_transport_abi_version.restype = ctypes.c_uint32
        lib.ndp_transport_error_string.argtypes = [ctypes.c_int]
        lib.ndp_transport_error_string.restype = ctypes.c_char_p
        lib.ndp_transport_open_v1.argtypes = [
            ctypes.POINTER(_OpenV1), ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_transport_bind_identity_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(_IdentityV1)]
        lib.ndp_transport_endpoint_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(_EndpointV1)]
        lib.ndp_transport_peer_upsert_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(_PeerV1), ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_transport_peer_remove_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_transport_send_v1.argtypes = [
            ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_transport_receive_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_transport_poll_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(_EventV1), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int]
        lib.ndp_transport_metrics_v1.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(_MetricsV1)]
        lib.ndp_transport_cancel_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_transport_close_v1.argtypes = [ctypes.c_uint64]
        for name in (
            "ndp_transport_open_v1", "ndp_transport_bind_identity_v1",
            "ndp_transport_endpoint_v1", "ndp_transport_peer_upsert_v1",
            "ndp_transport_peer_remove_v1", "ndp_transport_send_v1",
            "ndp_transport_receive_v1", "ndp_transport_poll_v1",
            "ndp_transport_metrics_v1", "ndp_transport_cancel_v1",
            "ndp_transport_close_v1",
        ):
            getattr(lib, name).restype = ctypes.c_int

    def check(self, code: int, operation: str) -> None:
        if int(code) not in (0, 1):
            detail = self.lib.ndp_transport_error_string(int(code))
            raise TransportError(int(code), operation,
                                 detail.decode("utf-8", "replace"))


class NativeTransport:
    """Lifecycle-safe native endpoint and current-fence route table."""

    def __init__(self, library: NativeTransportLibrary, handle: int,
                 deadline_unix_ns: int, payload_max: int):
        self.library, self.handle = library, int(handle)
        self.deadline_unix_ns = int(deadline_unix_ns)
        self.payload_max = int(payload_max)
        self.closed = False

    @classmethod
    def open(cls, *,
             library: NativeTransportLibrary | str | os.PathLike[str] | None = None,
             provider: str, production: bool, bind_node: str,
             deadline_s: float = 30.0, payload_max: int = 64 << 20,
             tx_slots: int = 4, rx_slots: int = 4,
             resident_limit_bytes: int = 16 << 30,
             telemetry_fd: int = -1) -> "NativeTransport":
        native = library if isinstance(library, NativeTransportLibrary) \
            else NativeTransportLibrary(library)
        if deadline_s <= 0:
            raise ValueError("native transport deadline must be positive")
        if production and provider != "cxi":
            raise ValueError("production native transport requires exact provider cxi")
        config = _versioned(_OpenV1())
        config.mode = 1 if production else 2
        config.tx_slots, config.rx_slots = int(tx_slots), int(rx_slots)
        config.payload_max = int(payload_max)
        config.resident_limit_bytes = int(resident_limit_bytes)
        config.operation_deadline_unix_ns = time.time_ns() + int(deadline_s * 1e9)
        config.telemetry_fd = int(telemetry_fd)
        _write_text(config, "provider", provider)
        if production:
            _write_text(config, "require_provider", "cxi")
        _write_text(config, "bind_node", bind_node)
        handle = ctypes.c_uint64()
        native.check(native.lib.ndp_transport_open_v1(
            ctypes.byref(config), ctypes.byref(handle)), "transport_open")
        return cls(native, handle.value, config.operation_deadline_unix_ns,
                   config.payload_max)

    def bind(self, *, run_key: bytes | str, fence_epoch: int,
             worker_key: bytes | str, incarnation: bytes | str,
             endpoint_epoch: int, expires_unix_ns: int) -> EndpointRecord:
        identity = _versioned(_IdentityV1())
        _write_bytes(identity.run_key, _key16(run_key, field="run_key"))
        _write_bytes(identity.worker_key, _key16(worker_key, field="worker_key"))
        _write_bytes(identity.incarnation, _key16(incarnation, field="incarnation"))
        identity.fence_epoch = int(fence_epoch)
        identity.endpoint_epoch = int(endpoint_epoch)
        identity.expires_unix_ns = int(expires_unix_ns)
        self.library.check(self.library.lib.ndp_transport_bind_identity_v1(
            self.handle, ctypes.byref(identity)), "transport_bind_identity")
        return self.endpoint()

    def endpoint(self) -> EndpointRecord:
        value = _versioned(_EndpointV1())
        self.library.check(self.library.lib.ndp_transport_endpoint_v1(
            self.handle, ctypes.byref(value)), "transport_endpoint")
        return decode_endpoint_record(bytes(value.record[:value.record_bytes]))

    def upsert(self, endpoint: EndpointRecord) -> int:
        peer = _versioned(_PeerV1())
        _write_bytes(peer.worker_key, endpoint.worker_key)
        _write_bytes(peer.incarnation, endpoint.incarnation)
        peer.endpoint_epoch = endpoint.endpoint_epoch
        peer.expires_unix_ns = endpoint.expires_unix_ns
        peer.endpoint_name_bytes = len(endpoint.encoded)
        _write_bytes(peer.endpoint_name, endpoint.encoded)
        peer_id = ctypes.c_uint64()
        self.library.check(self.library.lib.ndp_transport_peer_upsert_v1(
            self.handle, ctypes.byref(peer), ctypes.byref(peer_id)),
            "transport_peer_upsert")
        return int(peer_id.value)

    def remove(self, peer_id: int) -> None:
        self.library.check(self.library.lib.ndp_transport_peer_remove_v1(
            self.handle, int(peer_id)), "transport_peer_remove")

    def send(self, peer_id: int, frame: bytes, *, deadline_unix_ns: int) -> None:
        """Send one already encoded native frame without a Python socket copy."""
        payload = bytes(frame)
        if not payload or len(payload) > NDP_TRANSPORT_ENDPOINT_MAX + self.payload_max:
            raise ValueError("native transport frame byte bound violated")
        if not 0 < deadline_unix_ns <= self.deadline_unix_ns:
            raise ValueError("native transport send deadline exceeds service deadline")
        storage = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        self.library.check(self.library.lib.ndp_transport_send_v1(
            self.handle, int(peer_id), storage, len(payload), int(deadline_unix_ns)),
            "transport_send")

    def send_fd(self, peer_id: int, fd: int, *, frame_bytes: int,
                deadline_unix_ns: int) -> None:
        """Submit an encoded frame directly from a bounded memfd mapping."""
        if fd < 0 or not 320 <= frame_bytes <= self.payload_max + 320:
            raise ValueError("native transport frame memfd extent is invalid")
        if not 0 < deadline_unix_ns <= self.deadline_unix_ns:
            raise ValueError("native transport send deadline exceeds service deadline")
        if os.fstat(fd).st_size != frame_bytes:
            raise ValueError("native transport frame memfd size mismatch")
        with mmap.mmap(fd, frame_bytes, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapping:
            storage = (ctypes.c_uint8 * frame_bytes).from_buffer(mapping)
            try:
                self.library.check(self.library.lib.ndp_transport_send_v1(
                    self.handle, int(peer_id), storage, frame_bytes,
                    int(deadline_unix_ns)), "transport_send_fd")
            finally:
                del storage

    def receive(self, *, capacity: int | None = None) -> tuple[int, bytes] | None:
        """Take one authenticated native frame; ``None`` means no completed RX."""
        bound = self.payload_max + 320 if capacity is None else int(capacity)
        if not 320 <= bound <= self.payload_max + 320:
            raise ValueError("native receive capacity exceeds the registered frame bound")
        storage = (ctypes.c_uint8 * bound)()
        frame_bytes, peer_id = ctypes.c_uint64(), ctypes.c_uint64()
        code = int(self.library.lib.ndp_transport_receive_v1(
            self.handle, storage, bound, ctypes.byref(frame_bytes), ctypes.byref(peer_id)))
        if code in (1, NDP_TRANSPORT_ECREDIT):
            return None
        self.library.check(code, "transport_receive")
        if not 320 <= frame_bytes.value <= bound or peer_id.value == 0:
            raise TransportError(-7, "transport_receive", "native returned invalid frame extent")
        return int(peer_id.value), bytes(storage[:frame_bytes.value])

    def receive_into_fd(self, fd: int, *, capacity: int) -> tuple[int, int] | None:
        """Receive one authenticated frame directly into a bounded memfd."""
        bound = int(capacity)
        if fd < 0 or not 320 <= bound <= self.payload_max + 320:
            raise ValueError("native receive memfd capacity exceeds the registered bound")
        if os.fstat(fd).st_size != bound:
            raise ValueError("native receive memfd size mismatch")
        with mmap.mmap(fd, bound, access=mmap.ACCESS_WRITE) as mapping:
            storage = (ctypes.c_uint8 * bound).from_buffer(mapping)
            frame_bytes, peer_id = ctypes.c_uint64(), ctypes.c_uint64()
            try:
                code = int(self.library.lib.ndp_transport_receive_v1(
                    self.handle, storage, bound, ctypes.byref(frame_bytes),
                    ctypes.byref(peer_id)))
            finally:
                del storage
        if code in (1, NDP_TRANSPORT_ECREDIT):
            return None
        self.library.check(code, "transport_receive_fd")
        if not 320 <= frame_bytes.value <= bound or peer_id.value == 0:
            raise TransportError(
                -7, "transport_receive_fd", "native returned invalid frame extent")
        return int(peer_id.value), int(frame_bytes.value)

    def poll(self, timeout_ms: int = 0, capacity: int = 24) -> tuple[dict[str, object], ...]:
        if not 0 <= timeout_ms <= 30_000 or not 1 <= capacity <= 64:
            raise ValueError("native transport poll must be bounded")
        values = (_EventV1 * capacity)()
        for value in values:
            _versioned(value)
        count = ctypes.c_uint32()
        self.library.check(self.library.lib.ndp_transport_poll_v1(
            self.handle, values, capacity, ctypes.byref(count), timeout_ms),
            "transport_poll")
        return tuple({
            "event": int(value.event), "status": int(value.status),
            "peer_id": int(value.peer_id), "message_seq": int(value.message_seq),
            "useful_bytes": int(value.useful_bytes), "wire_bytes": int(value.wire_bytes),
            "provider_errno": int(value.provider_errno), "reason": int(value.reason),
            "detail": bytes(value.detail).hex(),
        } for value in values[:count.value])

    @property
    def metrics(self) -> TransportMetrics:
        value = _versioned(_MetricsV1())
        self.library.check(self.library.lib.ndp_transport_metrics_v1(
            self.handle, ctypes.byref(value)), "transport_metrics")
        provider = bytes(value.provider_name[:value.provider_name_len]).decode("utf-8")
        return TransportMetrics(provider, *(
            int(getattr(value, name)) for name in (
                "useful_tx_bytes", "useful_rx_bytes", "wire_tx_bytes", "wire_rx_bytes",
                "retries", "replay_bytes", "cq_errors", "route_errors",
                "in_flight_bytes", "in_flight_high_water", "retained_bytes",
                "retained_high_water", "released_bytes", "tx_slot_high_water",
                "rx_slot_high_water", "live_peers",
            )))

    def close(self) -> None:
        if not self.closed:
            self.library.check(self.library.lib.ndp_transport_close_v1(
                self.handle), "transport_close")
            self.closed = True

    def __enter__(self) -> "NativeTransport":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
