from __future__ import annotations

import ctypes
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from ndm.native_dataplane import (
    ABI_V1, AllocV1, BufferV1, Client, Command, ControlV1, DType, EventV1,
    LayoutV1, MetricsV1, NativeLibrary, OpenV1, ResultCode, ResultV1, SubmitV1,
)
from tests.native_dataplane_test_support import (
    compiled_service, key, library_path, open_client,
)


def test_v1_struct_sizes_soname_and_elastic_symbol_boundary():
    assert {
        OpenV1: 224, LayoutV1: 56, BufferV1: 88, AllocV1: 32,
        SubmitV1: 128, ControlV1: 216, EventV1: 96, ResultV1: 168,
        MetricsV1: 184,
    } == {kind: ctypes.sizeof(kind) for kind in (
        OpenV1, LayoutV1, BufferV1, AllocV1, SubmitV1, ControlV1, EventV1,
        ResultV1, MetricsV1,
    )}
    native = NativeLibrary()
    assert native.library.ndp_abi_version() == ABI_V1
    soname = subprocess.run(
        ["readelf", "-d", native.path], check=True, text=True, capture_output=True
    ).stdout
    assert "libemender_ndp.so.1" in soname
    symbols = subprocess.run(
        ["nm", "-D", native.path], check=True, text=True, capture_output=True
    ).stdout
    assert "ndp_buffer_allocate_v1" in symbols
    assert "ndp_result_view_v1" in symbols
    assert "MPI_" not in symbols and "PMPI_" not in symbols


def test_context_managers_release_fd_buffer_and_operation_on_exception():
    client = open_client(11)
    client.install_flat_layout(8, source_dtype=DType.F32, payload_max=64)
    client.install_generation(1, deadline_s=20).close()
    buffer = client.allocate(dtype=DType.F32)
    fd = buffer.fd
    with pytest.raises(RuntimeError, match="injected trainer failure"):
        with client:
            with buffer:
                with buffer.mapped(DType.F32, write=True) as target:
                    target[:] = np.arange(8, dtype=np.float32)
                buffer.seal()
                client.submit(
                    buffer, trainer_key=key(21), trainer_incarnation=key(22),
                    submission_seq=1, weight=7,
                )
                raise RuntimeError("injected trainer failure")
    assert client.closed and buffer.closed
    with pytest.raises(OSError):
        os.fstat(fd)


def test_poll_fd_is_cloexec_and_result_views_share_one_read_only_memfd():
    with open_client(31) as client:
        poll_fd = client.poll_fd
        try:
            assert fcntl.fcntl(poll_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        finally:
            os.close(poll_fd)
        client.install_flat_layout(16, source_dtype=DType.F32, payload_max=64)
        client.install_generation(3, deadline_s=20).close()
        with client.allocate(dtype=DType.F32) as source:
            with source.mapped(DType.F32, write=True) as target:
                target[:] = np.arange(16, dtype=np.float32)
            source.seal()
            client.submit(source, trainer_key=key(32), trainer_incarnation=key(33),
                          submission_seq=1, weight=13).close()
        client.control(Command.FREEZE).close()
        result_op = client.control(Command.FINALIZE_OWNERS)
        first = client.result_view(result_op)
        second = client.result_view(result_op)
        try:
            assert os.fstat(first.fd).st_ino == os.fstat(second.fd).st_ino
            assert client.metrics.shared_bytes_current == 16 * 4
            with pytest.raises(RuntimeError, match="read-only"):
                with first.mapped(DType.F32, write=True):
                    pass
        finally:
            first.close()
            second.close()
        client.control(Command.COMMIT).close()
        result_op.close()
        assert client.metrics.shared_bytes_current == 0


def test_python_bridge_restarts_with_process_unique_handles(tmp_path):
    native_path = str(library_path())
    service = compiled_service(key(71), 1)
    first_script = """
import json
from ndm.native_dataplane import Client
k=lambda x: bytes([x])*16
c=Client.open(library=LIB,run_key=k(71),fence_epoch=1,worker_key=k(72),incarnation=k(73),admission_token=TOKEN,socket_path=SOCKET)
b=c.allocate(bytes_count=16)
print(json.dumps({'handle': b.handle}))
""".replace("LIB", repr(native_path)).replace("TOKEN", repr(service.token)).replace("SOCKET", repr(str(service.socket_path)))
    environment = dict(os.environ, EMENDER_NDP_LIBRARY=native_path)
    first = subprocess.run([sys.executable, "-c", first_script], check=True, text=True,
                           capture_output=True, env=environment)
    stale_handle = json.loads(first.stdout.strip().splitlines()[-1])["handle"]
    second_script = """
import json
from ndm.native_dataplane import Client
k=lambda x: bytes([x])*16
c=Client.open(library=LIB,run_key=k(71),fence_epoch=2,worker_key=k(72),incarnation=k(74),admission_token=TOKEN,socket_path=SOCKET)
code=c.native.library.ndp_buffer_release_v1(c.handle, STALE)
print(json.dumps({'code': code}))
c.close()
""".replace("LIB", repr(native_path)).replace("STALE", str(stale_handle)).replace("TOKEN", repr(service.token)).replace("SOCKET", repr(str(service.socket_path)))
    second = subprocess.run([sys.executable, "-c", second_script], check=True, text=True,
                            capture_output=True, env=environment)
    assert json.loads(second.stdout.strip().splitlines()[-1])["code"] == ResultCode.EINVAL
