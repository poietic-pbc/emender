from __future__ import annotations

import hashlib
import atexit
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Sequence

import numpy as np

from ndm.native_dataplane import Client, Command, DType, Metrics, NativeLibrary


class _CompiledService:
    def __init__(self, run_key: bytes, minimum_fence: int):
        self.run_key = bytes(run_key)
        self.token = hashlib.sha256(b"emender-native-service-test\0" + self.run_key).digest()
        self.directory = Path(tempfile.mkdtemp(prefix="emender-ndp-pytest-"))
        self.socket_path = self.directory / "service.sock"
        executable = _service_binary()
        environment = dict(os.environ)
        self.process = subprocess.Popen([
            str(executable), "--provider", "tcp;ofi_rxm", "--test-only", "--serve",
            "--bind-node", "127.0.0.1", "--payload-max", "4096",
            "--tx-slots", "1", "--rx-slots", "1",
            "--socket", str(self.socket_path), "--run-key", self.run_key.hex(),
            "--admission-token", self.token.hex(),
            "--initial-fence", str(int(minimum_fence)),
            "--deadline-seconds", "1800",
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
           stderr=subprocess.PIPE, text=True, env=environment)
        for _ in range(1000):
            if self.socket_path.exists():
                if self.socket_path.stat().st_mode & 0o777 != 0o600:
                    self.close()
                    raise RuntimeError("compiled native service socket is not mode 0600")
                break
            if self.process.poll() is not None:
                detail = self.process.stderr.read() if self.process.stderr else ""
                self.close()
                raise RuntimeError(f"compiled native service failed to start: {detail}")
            time.sleep(0.01)
        else:
            self.close()
            raise TimeoutError("compiled native service did not bind its socket")

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process is not None and process.stderr is not None:
            process.stderr.close()
        directory = getattr(self, "directory", None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


def _service_binary() -> Path:
    configured = os.environ.get("EMENDER_NDP_SERVICE")
    candidates = [Path(configured)] if configured else []
    try:
        local_library = NativeLibrary().path
    except FileNotFoundError:
        local_library = Path("/nonexistent")
    candidates.extend([
        local_library.parent.parent / "bin/ndp_cxi_service",
        Path(__file__).resolve().parents[1]
            / "build/compiled-native-service-rpc-v1/transport/ndp_cxi_service",
        Path(__file__).resolve().parents[1]
            / "build/native-resilient-dataplane/bin/ndp_cxi_service",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError("ndp_cxi_service was not built; set EMENDER_NDP_SERVICE")


_SERVICES: dict[bytes, _CompiledService] = {}
_SERVICES_LOCK = threading.Lock()


def compiled_service(run_key: bytes, minimum_fence: int = 1) -> _CompiledService:
    run_key = bytes(run_key)
    with _SERVICES_LOCK:
        service = _SERVICES.get(run_key)
        if service is None or service.process.poll() is not None:
            service = _CompiledService(run_key, minimum_fence)
            _SERVICES[run_key] = service
        return service


def _close_services() -> None:
    with _SERVICES_LOCK:
        services = tuple(_SERVICES.values())
        _SERVICES.clear()
    for service in services:
        service.close()


atexit.register(_close_services)


def key(value: int) -> bytes:
    return bytes([value & 0xFF]) * 16


def library() -> NativeLibrary:
    return NativeLibrary()


def open_client(tag: int, *, fence: int = 1, native: NativeLibrary | None = None) -> Client:
    run_key = key(tag)
    service = compiled_service(run_key, fence)
    return Client.open(
        library=native or library(),
        run_key=run_key,
        fence_epoch=fence,
        worker_key=key(tag + 1),
        incarnation=key(tag + 2),
        admission_token=service.token,
        socket_path=str(service.socket_path),
    )


def source_bits(values: np.ndarray, dtype: DType) -> np.ndarray:
    values = np.asarray(values)
    if dtype is DType.F32:
        return np.ascontiguousarray(values, dtype="<f4")
    if dtype is DType.F64:
        return np.ascontiguousarray(values, dtype="<f8")
    f32 = np.ascontiguousarray(values, dtype="<f4")
    return np.ascontiguousarray(f32.view("<u4") >> np.uint32(16), dtype="<u2")


def decoded_source(values: np.ndarray, dtype: DType) -> np.ndarray:
    bits = source_bits(values, dtype)
    if dtype is not DType.BF16:
        return bits.astype(np.float64)
    expanded = (bits.astype("<u4") << np.uint32(16)).view("<f4")
    return expanded.astype(np.float64)


def python_v1_reference(arrays: Sequence[np.ndarray], weights: Sequence[int],
                        trainer_keys: Sequence[bytes], dtype: DType) -> np.ndarray:
    accumulator = np.zeros(np.asarray(arrays[0]).size, dtype=np.float64)
    for index in sorted(range(len(arrays)), key=lambda item: trainer_keys[item]):
        source = decoded_source(np.asarray(arrays[index]).reshape(-1), dtype)
        term = source * np.float64(weights[index])
        accumulator = accumulator + term
    result = accumulator / np.float64(sum(weights))
    return result.astype("<f4")


def native_reduce(arrays: Sequence[np.ndarray], weights: Sequence[int], order: Sequence[int],
                  *, trainer_keys: Sequence[bytes], tag: int,
                  dtype: DType = DType.F32,
                  native: NativeLibrary | None = None) -> tuple[np.ndarray, bytes, Metrics]:
    elements = int(np.asarray(arrays[0]).size)
    if any(np.asarray(array).size != elements for array in arrays):
        raise ValueError("native reduction fixture arrays differ in size")
    with open_client(tag, native=native) as client:
        client.install_flat_layout(elements, source_dtype=dtype, payload_max=4096)
        client.install_generation(9, attempt=2, owner_epoch=4, deadline_s=30).close()
        submissions = []
        for index in order:
            encoded = source_bits(np.asarray(arrays[index]).reshape(-1), dtype)
            with client.allocate(elements, dtype=dtype) as buffer:
                with buffer.mapped(dtype, write=True) as target:
                    target[:] = encoded
                buffer.seal()
                submissions.append(client.submit(
                    buffer,
                    trainer_key=trainer_keys[index],
                    trainer_incarnation=key(100 + index),
                    submission_seq=1,
                    weight=weights[index],
                    source_dtype=dtype,
                ))
        client.control(Command.FREEZE).close()
        result_op = client.control(Command.FINALIZE_OWNERS)
        with client.result_view(result_op) as view:
            with view.mapped(DType.F32) as result:
                copied = result.copy()
            root = view.result_root
        for submission in submissions:
            submission.close()
        client.control(Command.COMMIT).close()
        result_op.close()
        metrics = client.metrics
        return copied, root, metrics


def library_path() -> Path:
    return library().path
