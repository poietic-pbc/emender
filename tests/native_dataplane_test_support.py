from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from ndm.native_dataplane import Client, Command, DType, Metrics, NativeLibrary


def key(value: int) -> bytes:
    return bytes([value & 0xFF]) * 16


def library() -> NativeLibrary:
    return NativeLibrary()


def open_client(tag: int, *, fence: int = 1, native: NativeLibrary | None = None) -> Client:
    return Client.open(
        library=native or library(),
        run_key=key(tag),
        fence_epoch=fence,
        worker_key=key(tag + 1),
        incarnation=key(tag + 2),
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
