"""Thin lifecycle-safe bridge for the native resilient data-plane v1 ABI.

Dense values live in service-allocated memfd mappings.  Python passes only
fixed-size metadata through ``ctypes``: it never serializes tensor elements,
packs per-shard scalars, or carries dense bytes over a socket.  The native
library validates, reduces, projects, and owns every retained dense buffer.

This module implements the local half of R04/R05/R08-R10/R14/R15 and
NDP01/NDP04-NDP06/NDP08-NDP10/NDP12/NDP14-NDP16.  Lease acquisition,
membership, generation closure, and durable commit policy remain Python
control-plane responsibilities outside this bridge.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import ctypes.util
from dataclasses import dataclass
from enum import IntEnum
import fcntl
import hashlib
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Iterator, Mapping, Sequence

import numpy as np


ABI_V1 = 0x00010000
ABI_V21 = 0x00020001


class ResultCode(IntEnum):
    OK = 0
    IN_PROGRESS = 1
    EINVAL = -1
    EVERSION = -2
    ESTATE = -3
    EFENCE = -4
    ESTALE = -5
    ECONFLICT = -6
    ECHECKSUM = -7
    ENONFINITE = -8
    EBOUNDS = -9
    ECREDIT = -10
    EDEADLINE = -11
    EROUTE = -12
    EPROVIDER = -13
    ENOMEM = -14
    EIO = -15
    ESHUTDOWN = -16


class Role(IntEnum):
    TRAINER = 1
    CONTROLLER = 2


class DType(IntEnum):
    F32 = 1
    BF16 = 2
    F64 = 3

    @property
    def numpy_dtype(self) -> np.dtype:
        if self is DType.F32:
            return np.dtype("<f4")
        if self is DType.F64:
            return np.dtype("<f8")
        # NumPy's bfloat16 availability varies. Producers may use a uint16 view
        # containing canonical little-endian bfloat16 bits.
        return np.dtype("<u2")


class Command(IntEnum):
    BIND_FENCE = 1
    INSTALL_GENERATION = 2
    FREEZE = 3
    REASSIGN = 4
    FINALIZE_OWNERS = 5
    COMMIT = 6
    ABORT = 7
    DRAIN = 8


class State(IntEnum):
    STARTING = 1
    CONTROL_BOUND = 2
    FABRIC_READY = 3
    IDLE = 4
    LOCAL_COLLECT = 5
    PREPARED = 6
    FROZEN = 7
    TRANSFERRING = 8
    OWNED_READY = 9
    REDISTRIBUTING = 10
    RESULT_READY = 11
    COMMITTED = 12
    ABORTING = 13
    ABORTED = 14
    DRAINING = 15
    STOPPED = 16
    FAULT = 17


class EventKind(IntEnum):
    STATE = 1
    BUFFER_RELEASED = 2
    LOCAL_ACCEPTED = 3
    LOCAL_DUPLICATE = 4
    PREPARED = 5
    RESULT_READY = 6
    COMMITTED = 7
    ABORTED = 8
    DRAINED = 9


class NativeDataplaneError(RuntimeError):
    def __init__(self, code: int, operation: str, detail: str):
        self.code = int(code)
        self.operation = operation
        super().__init__(f"{operation}: {detail} ({self.code})")


class StaleFenceError(NativeDataplaneError):
    pass


class ChecksumError(NativeDataplaneError):
    pass


class NonfiniteError(NativeDataplaneError):
    pass


class BoundsError(NativeDataplaneError):
    pass


class ConflictError(NativeDataplaneError):
    pass


_ERROR_TYPES = {
    ResultCode.EFENCE: StaleFenceError,
    ResultCode.ECHECKSUM: ChecksumError,
    ResultCode.ENONFINITE: NonfiniteError,
    ResultCode.EBOUNDS: BoundsError,
    ResultCode.ENOMEM: BoundsError,
    ResultCode.ECONFLICT: ConflictError,
}


class OpenV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("role", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("socket_path_len", ctypes.c_uint32), ("socket_path", ctypes.c_uint8 * 108),
        ("run_key", ctypes.c_uint8 * 16), ("fence_epoch", ctypes.c_uint64),
        ("worker_key", ctypes.c_uint8 * 16), ("incarnation", ctypes.c_uint8 * 16),
        ("admission_token", ctypes.c_uint8 * 32), ("deadline_unix_ns", ctypes.c_uint64),
    ]


class LayoutV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("descriptor_fd", ctypes.c_int32), ("reserved0", ctypes.c_uint32),
        ("descriptor_bytes", ctypes.c_uint64), ("layout_digest", ctypes.c_uint8 * 32),
    ]


class BufferV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("kind", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("address_or_segid", ctypes.c_uint64), ("offset", ctypes.c_uint64),
        ("length", ctypes.c_uint64), ("handle_generation", ctypes.c_uint64),
        ("fd", ctypes.c_int32), ("reserved0", ctypes.c_uint32),
        ("layout_digest", ctypes.c_uint8 * 32),
    ]


class AllocV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("reserved0", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64), ("deadline_unix_ns", ctypes.c_uint64),
    ]


class SubmitV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("buffer", ctypes.c_uint64), ("trainer_key", ctypes.c_uint8 * 16),
        ("trainer_incarnation", ctypes.c_uint8 * 16),
        ("submission_seq", ctypes.c_uint64), ("weight", ctypes.c_uint64),
        ("element_offset", ctypes.c_uint64), ("element_count", ctypes.c_uint64),
        ("source_dtype", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("deadline_unix_ns", ctypes.c_uint64),
        ("source_buffer_sha256", ctypes.c_uint8 * 32),
    ]


class SubmitV21(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("buffer", ctypes.c_uint64), ("trainer_key", ctypes.c_uint8 * 16),
        ("trainer_incarnation", ctypes.c_uint8 * 16),
        ("submission_seq", ctypes.c_uint64),
        ("exact_tokens", ctypes.c_uint64),
        ("element_offset", ctypes.c_uint64),
        ("element_count", ctypes.c_uint64),
        ("source_dtype", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("deadline_unix_ns", ctypes.c_uint64),
        ("source_buffer_sha256", ctypes.c_uint8 * 32),
        ("stable_worker_key", ctypes.c_uint8 * 16),
        ("worker_incarnation", ctypes.c_uint8 * 16),
        ("contribution_sequence", ctypes.c_uint64),
        ("local_window_start", ctypes.c_uint64),
        ("local_window_end", ctypes.c_uint64),
        ("base_global_version", ctypes.c_uint64),
        ("commit_lag", ctypes.c_uint32),
        ("anchor_lag", ctypes.c_uint32),
        ("result_lag", ctypes.c_uint32),
        ("speculative_window_lag", ctypes.c_uint32),
        ("policy_digest", ctypes.c_uint8 * 32),
        ("code_digest", ctypes.c_uint8 * 32),
        ("base_digest", ctypes.c_uint8 * 32),
        ("payload_digest", ctypes.c_uint8 * 32),
        ("local_trainer_set_digest", ctypes.c_uint8 * 32),
        ("endpoint_digest", ctypes.c_uint8 * 32),
        ("policy_id_len", ctypes.c_uint32),
        ("policy_schema_len", ctypes.c_uint32),
        ("contribution_schema_len", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("policy_id", ctypes.c_uint8 * 32),
        ("policy_schema", ctypes.c_uint8 * 32),
        ("contribution_schema", ctypes.c_uint8 * 48),
    ]


class ControlV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("command", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("run_key", ctypes.c_uint8 * 16), ("fence_epoch", ctypes.c_uint64),
        ("generation", ctypes.c_uint64), ("attempt", ctypes.c_uint32),
        ("metadata_kind", ctypes.c_uint32), ("owner_epoch", ctypes.c_uint64),
        ("deadline_unix_ns", ctypes.c_uint64), ("metadata_fd", ctypes.c_int32),
        ("reserved0", ctypes.c_uint32), ("metadata_bytes", ctypes.c_uint64),
        ("layout_digest", ctypes.c_uint8 * 32), ("base_digest", ctypes.c_uint8 * 32),
        ("plan_digest", ctypes.c_uint8 * 32), ("metadata_sha256", ctypes.c_uint8 * 32),
    ]


class EventV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("event", ctypes.c_uint32), ("status", ctypes.c_uint32),
        ("reason", ctypes.c_uint32), ("state", ctypes.c_uint32),
        ("op", ctypes.c_uint64), ("generation", ctypes.c_uint64),
        ("attempt", ctypes.c_uint32), ("shard_id", ctypes.c_uint32),
        ("owner_epoch", ctypes.c_uint64), ("logical_bytes", ctypes.c_uint64),
        ("detail_digest", ctypes.c_uint8 * 32),
    ]


class ResultV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("dtype", ctypes.c_uint32),
        ("run_key", ctypes.c_uint8 * 16), ("fence_epoch", ctypes.c_uint64),
        ("generation", ctypes.c_uint64), ("attempt", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32), ("layout_digest", ctypes.c_uint8 * 32),
        ("base_digest", ctypes.c_uint8 * 32), ("result_root", ctypes.c_uint8 * 32),
        ("global_weight", ctypes.c_uint64), ("result_bytes", ctypes.c_uint64),
    ]


class MetricsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
        ("shared_bytes_current", ctypes.c_uint64),
        ("shared_bytes_high_water", ctypes.c_uint64),
        ("admitted_shared_bytes", ctypes.c_uint64),
        ("released_shared_bytes", ctypes.c_uint64),
        ("mapped_bytes_current", ctypes.c_uint64),
        ("mapped_bytes_high_water", ctypes.c_uint64),
        ("prompt_source_released_bytes", ctypes.c_uint64),
        ("result_bytes", ctypes.c_uint64),
        ("disk_replay_bytes", ctypes.c_uint64),
        ("disk_replay_files", ctypes.c_uint64),
        ("trainer_spool_bytes", ctypes.c_uint64),
        ("trainer_spool_files", ctypes.c_uint64),
        ("python_dense_socket_bytes", ctypes.c_uint64),
        ("handoff_full_copy_bytes", ctypes.c_uint64),
        ("projection_count", ctypes.c_uint64),
        ("duplicate_count", ctypes.c_uint64),
        ("conflict_count", ctypes.c_uint64),
        ("checksum_rejects", ctypes.c_uint64),
        ("nonfinite_rejects", ctypes.c_uint64),
        ("stale_rejects", ctypes.c_uint64),
        ("cancelled_ops", ctypes.c_uint64),
        ("buffer_exhaustions", ctypes.c_uint64),
    ]


_EXPECTED_SIZES = {
    OpenV1: 224, LayoutV1: 56, BufferV1: 88, AllocV1: 32,
    SubmitV1: 128, SubmitV21: 528, ControlV1: 216,
    EventV1: 96, ResultV1: 168,
    MetricsV1: 184,
}
for _structure, _expected in _EXPECTED_SIZES.items():
    if ctypes.sizeof(_structure) != _expected:  # pragma: no cover - platform gate
        raise RuntimeError(
            f"native data-plane ABI layout mismatch for {_structure.__name__}: "
            f"{ctypes.sizeof(_structure)} != {_expected}"
        )


def _versioned(structure: ctypes.Structure) -> ctypes.Structure:
    structure.struct_size = ctypes.sizeof(type(structure))
    structure.abi_version = ABI_V1
    return structure


def _versioned_v21(structure: ctypes.Structure) -> ctypes.Structure:
    structure.struct_size = ctypes.sizeof(type(structure))
    structure.abi_version = ABI_V21
    return structure


def _key16(value: bytes | str, *, field: str) -> bytes:
    if isinstance(value, str):
        value = hashlib.sha256(value.encode()).digest()[:16]
    raw = bytes(value)
    if len(raw) != 16:
        raise ValueError(f"{field} must be exactly 16 bytes")
    return raw


def _digest32(value: bytes, *, field: str) -> bytes:
    raw = bytes(value)
    if len(raw) != 32:
        raise ValueError(f"{field} must be exactly 32 bytes")
    return raw


def _fixed_text(target, value: object, *, field: str) -> int:
    raw = str(value).encode("utf-8")
    if not raw or len(raw) > len(target):
        raise ValueError(f"{field} does not fit the v2.1 ABI record")
    target[:len(raw)] = raw
    return len(raw)


def _deadline(seconds: float) -> int:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("deadline duration must be finite and positive")
    return time.time_ns() + int(seconds * 1_000_000_000)


def encode_flat_layout(total_elements: int, *, source_dtype: DType = DType.F32,
                       payload_max: int = 64 * 1024 * 1024) -> tuple[bytes, bytes]:
    """Encode the immutable v1 flat-layout descriptor and its domain digest."""
    total_elements = int(total_elements)
    payload_max = int(payload_max)
    if total_elements <= 0 or payload_max <= 0 or payload_max % 8:
        raise ValueError("flat layout requires positive elements and aligned payload_max")
    layout_bytes = total_elements * 8
    shard_count = (layout_bytes + payload_max - 1) // payload_max
    if layout_bytes > 16 * 1024**3 or payload_max > 64 * 1024**2 or shard_count > 256:
        raise ValueError("flat layout exceeds native v1 hard bounds")
    name = b"flat"
    header = b"NDPLAY1\0" + struct.pack(
        "<IIQQII", 1, 1 << (int(source_dtype) - 1), total_elements,
        payload_max, shard_count, 0,
    )
    record = (
        struct.pack("<H", len(name)) + name
        + struct.pack("<HHQ", int(source_dtype), 1, total_elements)
        + struct.pack("<QQ", 0, total_elements)
    )
    descriptor = header + record
    digest = hashlib.sha256(b"emender-ndp-layout-v1\0" + descriptor).digest()
    return descriptor, digest


def _sealed_metadata_memfd(payload: bytes) -> int:
    flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    fd = _memfd_create("emender-ndp-layout-v1", flags)
    try:
        os.ftruncate(fd, len(payload))
        view = memoryview(payload)
        offset = 0
        while offset != len(view):
            offset += os.pwrite(fd, view[offset:], offset)
        add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
        seal_grow = getattr(fcntl, "F_SEAL_GROW", 0x0004)
        seal_shrink = getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
        seal_write = getattr(fcntl, "F_SEAL_WRITE", 0x0008)
        seal_seal = getattr(fcntl, "F_SEAL_SEAL", 0x0001)
        fcntl.fcntl(fd, add_seals, seal_grow | seal_shrink | seal_write | seal_seal)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _memfd_create(name: str, flags: int) -> int:
    """Call memfd_create even when the platform Python omitted its wrapper."""
    wrapper = getattr(os, "memfd_create", None)
    if wrapper is not None:
        return int(wrapper(name, flags))
    syscall_numbers = {"x86_64": 319, "aarch64": 279, "ppc64le": 360}
    machine = os.uname().machine
    number = syscall_numbers.get(machine)
    if number is None:  # pragma: no cover - supported targets are enumerated
        raise OSError(f"memfd_create syscall number is unknown for {machine}")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    fd = int(libc.syscall(ctypes.c_long(number), ctypes.c_char_p(name.encode()),
                          ctypes.c_uint(flags)))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def create_memfd(name: str, *, allow_sealing: bool = False) -> int:
    """Create a CLOEXEC memfd on every supported production architecture.

    Frontier's approved Python omits ``os.memfd_create`` even though the Linux
    syscall is available, so live role code must use this portable binding.
    """
    flags = getattr(os, "MFD_CLOEXEC", 0x0001)
    if allow_sealing:
        flags |= getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    return _memfd_create(name, flags)


def seal_memfd(fd: int) -> None:
    """Make one memfd immutable using Linux constants absent from some Pythons."""
    if fd < 0:
        raise ValueError("sealed memfd descriptor is invalid")
    add_seals = getattr(fcntl, "F_ADD_SEALS", 1033)
    seals = (getattr(fcntl, "F_SEAL_GROW", 0x0004)
             | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
             | getattr(fcntl, "F_SEAL_WRITE", 0x0008)
             | getattr(fcntl, "F_SEAL_SEAL", 0x0001))
    fcntl.fcntl(fd, add_seals, seals)


def copy_fd_range(source_fd: int, destination_fd: int, length: int, *,
                  source_offset: int, destination_offset: int) -> None:
    """Copy an exact kernel-side fd range despite a reduced Python ``os`` API."""
    if (source_fd < 0 or destination_fd < 0 or length < 0
            or source_offset < 0 or destination_offset < 0):
        raise ValueError("fd range copy bounds are invalid")
    copied = 0
    wrapper = getattr(os, "copy_file_range", None)
    while copied < length:
        if wrapper is not None:
            count = int(wrapper(
                source_fd, destination_fd, length - copied,
                offset_src=source_offset + copied,
                offset_dst=destination_offset + copied))
        else:
            libc = ctypes.CDLL(None, use_errno=True)
            native = libc.copy_file_range
            native.argtypes = [
                ctypes.c_int, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int,
                ctypes.POINTER(ctypes.c_longlong), ctypes.c_size_t, ctypes.c_uint,
            ]
            native.restype = ctypes.c_ssize_t
            in_offset = ctypes.c_longlong(source_offset + copied)
            out_offset = ctypes.c_longlong(destination_offset + copied)
            count = int(native(
                source_fd, ctypes.byref(in_offset), destination_fd,
                ctypes.byref(out_offset), length - copied, 0))
            if count < 0:
                error = ctypes.get_errno()
                if error == 4:  # EINTR
                    continue
                raise OSError(error, os.strerror(error))
        if count <= 0:
            raise OSError("copy_file_range made no memfd progress")
        copied += count


class NativeLibrary:
    """Typed owner of one loaded ``libemender_ndp.so.1`` instance."""

    def __init__(self, path: str | os.PathLike[str] | None = None):
        selected = self._resolve(path)
        self.path = selected
        self.library = ctypes.CDLL(str(selected), use_errno=True)
        lib = self.library
        lib.ndp_abi_version.argtypes = []
        lib.ndp_abi_version.restype = ctypes.c_uint32
        lib.ndp_abi_version_v21.argtypes = []
        lib.ndp_abi_version_v21.restype = ctypes.c_uint32
        lib.ndp_error_string.argtypes = [ctypes.c_int]
        lib.ndp_error_string.restype = ctypes.c_char_p
        lib.ndp_client_open_v1.argtypes = [ctypes.POINTER(OpenV1), ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_client_open_v1.restype = ctypes.c_int
        lib.ndp_client_poll_fd_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_int)]
        lib.ndp_client_poll_fd_v1.restype = ctypes.c_int
        lib.ndp_client_close_v1.argtypes = [ctypes.c_uint64]
        lib.ndp_client_close_v1.restype = ctypes.c_int
        lib.ndp_layout_install_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(LayoutV1)]
        lib.ndp_layout_install_v1.restype = ctypes.c_int
        lib.ndp_buffer_allocate_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(AllocV1),
                                               ctypes.POINTER(ctypes.c_uint64),
                                               ctypes.POINTER(ctypes.c_int)]
        lib.ndp_buffer_allocate_v1.restype = ctypes.c_int
        lib.ndp_buffer_register_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(BufferV1),
                                               ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_buffer_register_v1.restype = ctypes.c_int
        lib.ndp_buffer_seal_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_buffer_seal_v1.restype = ctypes.c_int
        lib.ndp_buffer_release_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_buffer_release_v1.restype = ctypes.c_int
        lib.ndp_submit_local_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(SubmitV1),
                                            ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_submit_local_v1.restype = ctypes.c_int
        lib.ndp_submit_local_v21.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(SubmitV21),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.ndp_submit_local_v21.restype = ctypes.c_int
        lib.ndp_control_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(ControlV1),
                                       ctypes.POINTER(ctypes.c_uint64)]
        lib.ndp_control_v1.restype = ctypes.c_int
        lib.ndp_poll_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(EventV1),
                                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                    ctypes.c_int]
        lib.ndp_poll_v1.restype = ctypes.c_int
        lib.ndp_result_view_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                           ctypes.POINTER(ResultV1),
                                           ctypes.POINTER(ctypes.c_uint64),
                                           ctypes.POINTER(ctypes.c_int)]
        lib.ndp_result_view_v1.restype = ctypes.c_int
        lib.ndp_op_release_v1.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
        lib.ndp_op_release_v1.restype = ctypes.c_int
        lib.ndp_client_metrics_v1.argtypes = [ctypes.c_uint64, ctypes.POINTER(MetricsV1)]
        lib.ndp_client_metrics_v1.restype = ctypes.c_int
        if lib.ndp_abi_version() != ABI_V1:
            raise RuntimeError(f"native data-plane ABI mismatch in {selected}")
        if lib.ndp_abi_version_v21() != ABI_V21:
            raise RuntimeError(f"native v2.1 data-plane ABI mismatch in {selected}")

    @staticmethod
    def _resolve(path: str | os.PathLike[str] | None) -> Path:
        candidates: list[Path] = []
        if path is not None:
            candidates.append(Path(path))
        configured = os.environ.get("EMENDER_NDP_LIBRARY")
        if configured:
            candidates.append(Path(configured))
        repository = Path(__file__).resolve().parents[1]
        candidates.extend([
            repository / "build/native-dataplane/local-service/libemender_ndp.so",
            repository / "build/native-dataplane/libemender_ndp.so",
            repository / "build/native-dataplane-portable/libemender_ndp.so",
            repository / "build/native-resilient-dataplane/lib/libemender_ndp.so.1",
            repository / "build/native-resilient-dataplane/lib64/libemender_ndp.so.1",
        ])
        discovered = ctypes.util.find_library("emender_ndp")
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            "libemender_ndp.so.1 was not found; set EMENDER_NDP_LIBRARY or run "
            "scripts/frontier/build_native_resilient_dataplane.sh"
        )

    def check(self, code: int, operation: str) -> None:
        if code >= 0:
            return
        try:
            enum_code = ResultCode(code)
        except ValueError:
            enum_code = code
        detail = self.library.ndp_error_string(code).decode("utf-8", errors="replace")
        error_type = _ERROR_TYPES.get(enum_code, NativeDataplaneError)
        raise error_type(code, operation, detail)


@dataclass(frozen=True)
class Event:
    event: EventKind
    status: int
    reason: int
    state: State
    op: int
    generation: int
    attempt: int
    owner_epoch: int
    logical_bytes: int
    detail_digest: bytes


@dataclass(frozen=True)
class Metrics:
    shared_bytes_current: int
    shared_bytes_high_water: int
    admitted_shared_bytes: int
    released_shared_bytes: int
    mapped_bytes_current: int
    mapped_bytes_high_water: int
    prompt_source_released_bytes: int
    result_bytes: int
    disk_replay_bytes: int
    disk_replay_files: int
    trainer_spool_bytes: int
    trainer_spool_files: int
    python_dense_socket_bytes: int
    handoff_full_copy_bytes: int
    projection_count: int
    duplicate_count: int
    conflict_count: int
    checksum_rejects: int
    nonfinite_rejects: int
    stale_rejects: int
    cancelled_ops: int
    buffer_exhaustions: int


class Operation:
    def __init__(self, client: "Client", handle: int):
        self.client = client
        self.handle = int(handle)
        self.closed = False
        client._operations.add(self)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.client._operations.discard(self)
        if not self.client.closed:
            code = self.client.native.library.ndp_op_release_v1(
                self.client.handle, self.handle
            )
            # Release is idempotent cleanup after the process has finished
            # consuming the mapping.  A disappeared service route cannot
            # retain this process's closed fd and must not replace the primary
            # generation outcome with a restart-inducing teardown failure.
            if code not in (ResultCode.OK, ResultCode.EINVAL, ResultCode.EFENCE,
                            ResultCode.EROUTE, ResultCode.ESHUTDOWN):
                self.client.native.check(code, "ndp_op_release_v1")

    def __enter__(self) -> "Operation":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class Buffer:
    """Explicit owner of one native handle and one caller-owned CLOEXEC fd."""

    def __init__(self, client: "Client", handle: int, fd: int, length: int,
                 *, writable: bool):
        self.client = client
        self.handle = int(handle)
        self.fd = int(fd)
        self.length = int(length)
        self.writable = bool(writable)
        self.closed = False
        self._mappings: set[mmap.mmap] = set()
        client._buffers.add(self)

    @contextmanager
    def mapped(self, dtype: DType | np.dtype | str, *, write: bool = False,
               shape: Sequence[int] | None = None) -> Iterator[np.ndarray]:
        if self.closed:
            raise RuntimeError("native buffer is closed")
        if write and not self.writable:
            raise RuntimeError("native buffer is read-only")
        numpy_dtype = dtype.numpy_dtype if isinstance(dtype, DType) else np.dtype(dtype)
        if self.length % numpy_dtype.itemsize:
            raise ValueError("buffer length is not divisible by requested dtype")
        if write:
            mapping = mmap.mmap(self.fd, self.length, access=mmap.ACCESS_WRITE)
        else:
            mapping = mmap.mmap(self.fd, self.length, flags=mmap.MAP_PRIVATE,
                                prot=mmap.PROT_READ)
        self._mappings.add(mapping)
        try:
            elements = self.length // numpy_dtype.itemsize
            array = np.ndarray((elements,), dtype=numpy_dtype, buffer=mapping)
            if shape is not None:
                array = array.reshape(tuple(shape))
            yield array
        finally:
            del array
            self._mappings.discard(mapping)
            mapping.close()

    def sha256(self) -> bytes:
        if self.closed:
            raise RuntimeError("native buffer is closed")
        with mmap.mmap(self.fd, self.length, flags=mmap.MAP_PRIVATE,
                       prot=mmap.PROT_READ) as mapping:
            return hashlib.sha256(mapping).digest()

    def seal(self) -> None:
        if self.closed:
            raise RuntimeError("native buffer is closed")
        if self._mappings:
            raise RuntimeError("close writable mappings before sealing the buffer")
        self.client.native.check(
            self.client.native.library.ndp_buffer_seal_v1(
                self.client.handle, self.handle
            ),
            "ndp_buffer_seal_v1",
        )
        self.writable = False

    def close(self) -> None:
        if self.closed:
            return
        if self._mappings:
            raise RuntimeError("cannot release a native buffer with live mappings")
        self.closed = True
        self.client._buffers.discard(self)
        try:
            os.close(self.fd)
        finally:
            self.fd = -1
        if not self.client.closed:
            code = self.client.native.library.ndp_buffer_release_v1(
                self.client.handle, self.handle
            )
            # The local fd is already closed.  Route loss means the remote
            # service disappeared and therefore cannot retain this process's
            # reference; release remains complete and bounded.
            if code not in (ResultCode.OK, ResultCode.EINVAL, ResultCode.EFENCE,
                            ResultCode.EROUTE, ResultCode.ESHUTDOWN):
                self.client.native.check(code, "ndp_buffer_release_v1")

    def __enter__(self) -> "Buffer":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


class ResultView(Buffer):
    def __init__(self, client: "Client", handle: int, fd: int, metadata: ResultV1):
        super().__init__(client, handle, fd, int(metadata.result_bytes), writable=False)
        self.run_key = bytes(metadata.run_key)
        self.fence_epoch = int(metadata.fence_epoch)
        self.generation = int(metadata.generation)
        self.attempt = int(metadata.attempt)
        self.layout_digest = bytes(metadata.layout_digest)
        self.base_digest = bytes(metadata.base_digest)
        self.result_root = bytes(metadata.result_root)
        self.global_weight = int(metadata.global_weight)
        self.dtype = DType(int(metadata.dtype))


class Client:
    """Context-managed v1 controller/trainer client and all child handles."""

    def __init__(self, native: NativeLibrary, handle: int, *, role: Role,
                 run_key: bytes, fence_epoch: int, worker_key: bytes,
                 incarnation: bytes):
        self.native = native
        self.handle = int(handle)
        self.role = role
        self.run_key = run_key
        self.fence_epoch = int(fence_epoch)
        self.worker_key = worker_key
        self.incarnation = incarnation
        self.closed = False
        self.layout_digest = bytes(32)
        self.total_elements = 0
        self.source_dtype = DType.F32
        self.generation = 0
        self.attempt = 0
        self.owner_epoch = 0
        self.base_digest = bytes(32)
        self.plan_digest = bytes(32)
        self.generation_deadline_ns = 0
        self._buffers: set[Buffer] = set()
        self._operations: set[Operation] = set()

    @classmethod
    def open(cls, *, library: str | os.PathLike[str] | NativeLibrary | None = None,
             role: Role = Role.CONTROLLER, run_key: bytes | str,
             fence_epoch: int, worker_key: bytes | str,
             incarnation: bytes | str, admission_token: bytes | None = None,
             socket_path: str | None = None,
             deadline_s: float = 10.0) -> "Client":
        native = library if isinstance(library, NativeLibrary) else NativeLibrary(library)
        request = _versioned(OpenV1())
        request.role = int(role)
        if socket_path is None:
            socket_path = os.environ.get("EMENDER_NDP_SOCKET", "/tmp/emender-ndp.sock")
        encoded_path = os.fsencode(socket_path)
        if not 0 < len(encoded_path) <= 108:
            raise ValueError("native control socket path must contain 1..108 bytes")
        request.socket_path_len = len(encoded_path)
        request.socket_path[:len(encoded_path)] = encoded_path
        run = _key16(run_key, field="run_key")
        worker = _key16(worker_key, field="worker_key")
        boot = _key16(incarnation, field="incarnation")
        configured_token = os.environ.get("EMENDER_NDP_ADMISSION_TOKEN_HEX")
        configured_token_fd = os.environ.get("EMENDER_NDP_ADMISSION_TOKEN_FD")
        if admission_token is not None:
            token = admission_token
        elif configured_token_fd:
            try:
                descriptor = int(configured_token_fd)
                token = os.pread(descriptor, 32, 0)
            except (OSError, ValueError) as error:
                raise ValueError(
                    "EMENDER_NDP_ADMISSION_TOKEN_FD must name a readable protected fd"
                ) from error
            if len(token) != 32:
                raise ValueError("native admission token fd must contain exactly 32 bytes")
        elif configured_token:
            try:
                token = bytes.fromhex(configured_token)
            except ValueError as error:
                raise ValueError("EMENDER_NDP_ADMISSION_TOKEN_HEX must be hexadecimal") from error
        else:
            token = hashlib.sha256(run + int(fence_epoch).to_bytes(8, "little")).digest()
        request.run_key[:] = run
        request.fence_epoch = int(fence_epoch)
        request.worker_key[:] = worker
        request.incarnation[:] = boot
        request.admission_token[:] = _digest32(token, field="admission_token")
        request.deadline_unix_ns = _deadline(deadline_s)
        handle = ctypes.c_uint64()
        native.check(native.library.ndp_client_open_v1(ctypes.byref(request), ctypes.byref(handle)),
                     "ndp_client_open_v1")
        return cls(native, handle.value, role=role, run_key=run,
                   fence_epoch=fence_epoch, worker_key=worker, incarnation=boot)

    def attach_generation(self, *, total_elements: int, layout_digest: bytes,
                          generation: int, attempt: int, owner_epoch: int,
                          source_dtype: DType, deadline_s: float,
                          deadline_unix_ns: int | None = None,
                          base_digest: bytes | None = None,
                          plan_digest: bytes | None = None) -> None:
        """Adopt controller-published metadata without mutating service state.

        Trainer RPC clients open after the model-free controller installs a
        generation.  Their dense buffer is allocated directly by the persistent
        service, while this local metadata only populates the typed submit ABI.
        """
        if self.role is not Role.TRAINER:
            raise RuntimeError("only a trainer may attach controller metadata")
        if total_elements <= 0 or generation < 0 or attempt <= 0 or owner_epoch <= 0:
            raise ValueError("native generation metadata is invalid")
        self.total_elements = int(total_elements)
        self.layout_digest = _digest32(layout_digest, field="layout_digest")
        self.generation, self.attempt = int(generation), int(attempt)
        self.owner_epoch, self.source_dtype = int(owner_epoch), source_dtype
        self.base_digest = _digest32(base_digest or hashlib.sha256(
            b"native-local-base" + self.generation.to_bytes(8, "little")
        ).digest(), field="base_digest")
        self.plan_digest = _digest32(plan_digest or hashlib.sha256(
            b"native-local-plan" + self.generation.to_bytes(8, "little")
        ).digest(), field="plan_digest")
        self.generation_deadline_ns = (int(deadline_unix_ns)
                                       if deadline_unix_ns is not None
                                       else _deadline(deadline_s))
        if self.generation_deadline_ns <= time.time_ns():
            raise ValueError("native generation deadline has expired")

    def refresh_generation(self, *, total_elements: int, layout_digest: bytes,
                           generation: int, attempt: int, owner_epoch: int,
                           source_dtype: DType, deadline_s: float,
                           deadline_unix_ns: int, base_digest: bytes,
                           plan_digest: bytes) -> None:
        """Adopt a controller-published retry without mutating service state.

        A trainer can remain connected while the controller replaces its
        node-local attempt with the globally redistributed attempt.  The
        metadata-only metrics RPC first refreshes the native connection's
        fenced request header from the service snapshot; ``attach_generation``
        then keeps the Python-side typed metadata identical to that snapshot.
        """
        if self.role is not Role.TRAINER:
            raise RuntimeError("only a trainer may refresh controller metadata")
        # Metrics is accepted independent of the old attempt header.  Every
        # successful RPC response carries the service's current generation,
        # attempt, and layout, so this does not install or mutate a generation.
        self.metrics
        self.attach_generation(
            total_elements=total_elements, layout_digest=layout_digest,
            generation=generation, attempt=attempt, owner_epoch=owner_epoch,
            source_dtype=source_dtype, deadline_s=deadline_s,
            deadline_unix_ns=deadline_unix_ns, base_digest=base_digest,
            plan_digest=plan_digest)

    @property
    def poll_fd(self) -> int:
        result = ctypes.c_int(-1)
        self.native.check(self.native.library.ndp_client_poll_fd_v1(
            self.handle, ctypes.byref(result)), "ndp_client_poll_fd_v1")
        return result.value

    def install_flat_layout(self, total_elements: int, *,
                            source_dtype: DType = DType.F32,
                            payload_max: int = 64 * 1024 * 1024) -> bytes:
        descriptor, digest = encode_flat_layout(
            total_elements, source_dtype=source_dtype, payload_max=payload_max
        )
        fd = _sealed_metadata_memfd(descriptor)
        try:
            request = _versioned(LayoutV1())
            request.descriptor_fd = fd
            request.descriptor_bytes = len(descriptor)
            request.layout_digest[:] = digest
            self.native.check(self.native.library.ndp_layout_install_v1(
                self.handle, ctypes.byref(request)), "ndp_layout_install_v1")
        finally:
            os.close(fd)
        self.layout_digest = digest
        self.total_elements = int(total_elements)
        self.source_dtype = source_dtype
        return digest

    def _control_request(self, command: Command, *, deadline_s: float = 30.0,
                         owner_epoch: int | None = None) -> ControlV1:
        request = _versioned(ControlV1())
        request.command = int(command)
        request.run_key[:] = self.run_key
        request.fence_epoch = self.fence_epoch
        request.generation = self.generation
        request.attempt = self.attempt
        request.owner_epoch = self.owner_epoch if owner_epoch is None else int(owner_epoch)
        requested_deadline = _deadline(deadline_s)
        if self.generation_deadline_ns and command is not Command.INSTALL_GENERATION:
            requested_deadline = min(requested_deadline, self.generation_deadline_ns)
        request.deadline_unix_ns = requested_deadline
        request.metadata_fd = -1
        request.layout_digest[:] = self.layout_digest
        request.base_digest[:] = self.base_digest
        request.plan_digest[:] = self.plan_digest
        return request

    def control(self, command: Command, *, deadline_s: float = 30.0,
                owner_epoch: int | None = None) -> Operation:
        request = self._control_request(command, deadline_s=deadline_s,
                                        owner_epoch=owner_epoch)
        handle = ctypes.c_uint64()
        self.native.check(self.native.library.ndp_control_v1(
            self.handle, ctypes.byref(request), ctypes.byref(handle)),
            f"ndp_control_v1({command.name})")
        if command is Command.REASSIGN and owner_epoch is not None:
            self.owner_epoch = int(owner_epoch)
        return Operation(self, handle.value)

    def install_generation(self, generation: int, *, attempt: int = 1,
                           owner_epoch: int = 1, base_digest: bytes | None = None,
                           plan_digest: bytes | None = None,
                           deadline_s: float = 30.0,
                           generation_deadline_s: float | None = None
                           ) -> Operation:
        self.generation = int(generation)
        self.attempt = int(attempt)
        self.owner_epoch = int(owner_epoch)
        self.base_digest = _digest32(base_digest or hashlib.sha256(
            b"native-local-base" + self.generation.to_bytes(8, "little")
        ).digest(), field="base_digest")
        self.plan_digest = _digest32(plan_digest or hashlib.sha256(
            b"native-local-plan" + self.generation.to_bytes(8, "little")
        ).digest(), field="plan_digest")
        self.generation_deadline_ns = _deadline(
            deadline_s if generation_deadline_s is None
            else generation_deadline_s)
        request = self._control_request(Command.INSTALL_GENERATION, deadline_s=deadline_s)
        request.deadline_unix_ns = self.generation_deadline_ns
        handle = ctypes.c_uint64()
        self.native.check(self.native.library.ndp_control_v1(
            self.handle, ctypes.byref(request), ctypes.byref(handle)),
            "ndp_control_v1(INSTALL_GENERATION)")
        return Operation(self, handle.value)

    def allocate(self, elements: int | None = None, *, dtype: DType | None = None,
                 bytes_count: int | None = None, deadline_s: float = 30.0) -> Buffer:
        if bytes_count is None:
            dtype = dtype or self.source_dtype
            if elements is None:
                elements = self.total_elements
            bytes_count = int(elements) * dtype.numpy_dtype.itemsize
        request = _versioned(AllocV1())
        request.flags = 3  # READ | WRITE
        request.bytes = int(bytes_count)
        request.deadline_unix_ns = _deadline(deadline_s)
        handle = ctypes.c_uint64()
        fd = ctypes.c_int(-1)
        self.native.check(self.native.library.ndp_buffer_allocate_v1(
            self.handle, ctypes.byref(request), ctypes.byref(handle), ctypes.byref(fd)),
            "ndp_buffer_allocate_v1")
        return Buffer(self, handle.value, fd.value, bytes_count, writable=True)

    def register_memfd(self, fd: int, *, length: int,
                       handle_generation: int | None = None) -> Buffer:
        """Register one immutable memfd for native replay/reduction.

        Only the descriptor crosses the metadata-only AF_UNIX protocol; the
        service duplicates and maps the producer-owned extent directly.
        """
        if fd < 0 or length <= 0:
            raise ValueError("registered native memfd extent is invalid")
        request = _versioned(BufferV1())
        request.kind = 2  # NDP_BUFFER_MEMFD
        request.flags = 1  # NDP_BUFFER_READ
        request.length = int(length)
        request.handle_generation = int(
            self.generation if handle_generation is None else handle_generation)
        request.fd = int(fd)
        request.layout_digest[:] = self.layout_digest
        handle = ctypes.c_uint64()
        self.native.check(self.native.library.ndp_buffer_register_v1(
            self.handle, ctypes.byref(request), ctypes.byref(handle)),
            "ndp_buffer_register_v1")
        return Buffer(self, handle.value, os.dup(fd), length, writable=False)

    def submit(self, buffer: Buffer, *, trainer_key: bytes | str,
               trainer_incarnation: bytes | str, submission_seq: int,
               weight: int, source_dtype: DType | None = None,
               source_sha256: bytes | None = None,
               v21_identity: Mapping[str, object] | None = None,
               deadline_s: float = 30.0) -> Operation:
        dtype = source_dtype or self.source_dtype
        request = (
            _versioned_v21(SubmitV21())
            if v21_identity is not None else _versioned(SubmitV1())
        )
        request.buffer = buffer.handle
        request.trainer_key[:] = _key16(trainer_key, field="trainer_key")
        request.trainer_incarnation[:] = _key16(
            trainer_incarnation, field="trainer_incarnation"
        )
        request.submission_seq = int(submission_seq)
        if v21_identity is None:
            request.weight = int(weight)
        else:
            request.exact_tokens = int(weight)
        request.element_count = self.total_elements
        request.source_dtype = int(dtype)
        request.deadline_unix_ns = min(_deadline(deadline_s), self.generation_deadline_ns)
        digest = source_sha256 if source_sha256 is not None else buffer.sha256()
        request.source_buffer_sha256[:] = _digest32(digest, field="source_sha256")
        operation = "ndp_submit_local_v1"
        submit = self.native.library.ndp_submit_local_v1
        if v21_identity is not None:
            identity = dict(v21_identity)
            request.stable_worker_key[:] = _key16(
                identity["worker_id"], field="worker_id")
            request.worker_incarnation[:] = _key16(
                identity["worker_incarnation"], field="worker_incarnation")
            request.contribution_sequence = int(
                identity["contribution_sequence"])
            request.local_window_start = int(identity["local_window_start"])
            request.local_window_end = int(identity["local_window_end"])
            request.base_global_version = int(identity["base_global_version"])
            request.commit_lag = int(identity["commit_lag"])
            request.anchor_lag = int(identity["anchor_lag"])
            request.result_lag = int(identity["result_lag"])
            request.speculative_window_lag = int(
                identity["speculative_window_lag"])
            for name in (
                "policy_digest", "code_digest", "base_digest",
                "payload_digest", "local_trainer_set_digest",
                "endpoint_digest",
            ):
                raw = identity[name]
                if isinstance(raw, str):
                    raw = bytes.fromhex(raw)
                getattr(request, name)[:] = _digest32(raw, field=name)
            request.policy_id_len = _fixed_text(
                request.policy_id, identity["policy_id"], field="policy_id")
            request.policy_schema_len = _fixed_text(
                request.policy_schema, identity["policy_schema"],
                field="policy_schema")
            request.contribution_schema_len = _fixed_text(
                request.contribution_schema, identity["contribution_schema"],
                field="contribution_schema")
            operation = "ndp_submit_local_v21"
            submit = self.native.library.ndp_submit_local_v21
        handle = ctypes.c_uint64()
        self.native.check(submit(
            self.handle, ctypes.byref(request), ctypes.byref(handle)),
            operation)
        return Operation(self, handle.value)

    def result_view(self, operation: Operation) -> ResultView:
        return self.result_view_handle(operation.handle)

    def result_view_handle(self, operation_handle: int) -> ResultView:
        """Map the service's one shared result from an independent trainer."""
        result = _versioned(ResultV1())
        handle = ctypes.c_uint64()
        fd = ctypes.c_int(-1)
        self.native.check(self.native.library.ndp_result_view_v1(
            self.handle, int(operation_handle), ctypes.byref(result),
            ctypes.byref(handle), ctypes.byref(fd)), "ndp_result_view_v1")
        return ResultView(self, handle.value, fd.value, result)

    def poll(self, *, capacity: int = 64, timeout_ms: int = 0) -> tuple[Event, ...]:
        if capacity <= 0:
            raise ValueError("event capacity must be positive")
        storage = (EventV1 * capacity)()
        count = ctypes.c_uint32()
        self.native.check(self.native.library.ndp_poll_v1(
            self.handle, storage, capacity, ctypes.byref(count), timeout_ms),
            "ndp_poll_v1")
        return tuple(Event(
            EventKind(item.event), item.status, item.reason, State(item.state),
            int(item.op), int(item.generation), int(item.attempt),
            int(item.owner_epoch), int(item.logical_bytes), bytes(item.detail_digest),
        ) for item in storage[:count.value])

    @property
    def metrics(self) -> Metrics:
        wire = _versioned(MetricsV1())
        self.native.check(self.native.library.ndp_client_metrics_v1(
            self.handle, ctypes.byref(wire)), "ndp_client_metrics_v1")
        return Metrics(**{
            name: int(getattr(wire, name))
            for name, _ctype in MetricsV1._fields_
            if name not in {"struct_size", "abi_version"}
        })

    def close(self) -> None:
        if self.closed:
            return
        # Close views before operations so the native result buffer can reach
        # zero references. Child close methods tolerate fence supersession.
        for buffer in tuple(self._buffers):
            buffer.close()
        for operation in tuple(self._operations):
            operation.close()
        code = self.native.library.ndp_client_close_v1(self.handle)
        self.closed = True
        if code not in (ResultCode.OK, ResultCode.EINVAL, ResultCode.EFENCE,
                        ResultCode.EROUTE, ResultCode.ESHUTDOWN):
            self.native.check(code, "ndp_client_close_v1")

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = [
    "ABI_V1", "BoundsError", "Buffer", "ChecksumError", "Client", "Command",
    "ConflictError", "DType", "Event", "EventKind", "Metrics", "NativeDataplaneError",
    "NativeLibrary", "NonfiniteError", "Operation", "ResultCode", "ResultView",
    "Role", "StaleFenceError", "State", "copy_fd_range", "create_memfd",
    "encode_flat_layout", "seal_memfd",
]
