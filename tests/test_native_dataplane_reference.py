from __future__ import annotations

import numpy as np
import pytest

from ndm.native_dataplane import Client, Command, ConflictError, DType
from tests.native_dataplane_test_support import (
    key, native_reduce, open_client, python_v1_reference, source_bits,
)


@pytest.mark.parametrize("dtype", [DType.F32, DType.F64, DType.BF16])
def test_native_exact_weighted_reference_is_arrival_independent(dtype):
    generator = np.random.default_rng(194)
    arrays = [
        generator.normal(0, 1e-4, 257),
        generator.normal(0, 1e4, 257),
        generator.normal(0, 1, 257),
    ]
    weights = [3, 1_000_003, 29]
    trainer_keys = [key(90), key(2), key(41)]
    expected = python_v1_reference(arrays, weights, trainer_keys, dtype)
    forward, forward_root, forward_metrics = native_reduce(
        arrays, weights, [2, 0, 1], trainer_keys=trainer_keys, tag=81, dtype=dtype
    )
    reverse, reverse_root, reverse_metrics = native_reduce(
        arrays, weights, [1, 0, 2], trainer_keys=trainer_keys, tag=81, dtype=dtype
    )
    assert np.array_equal(forward, expected)
    assert np.array_equal(reverse, expected)
    assert forward_root == reverse_root
    for metrics in (forward_metrics, reverse_metrics):
        assert metrics.projection_count == 1
        assert metrics.shared_bytes_current == 0
        assert metrics.released_shared_bytes == metrics.admitted_shared_bytes
        assert metrics.mapped_bytes_current == 0
        assert metrics.prompt_source_released_bytes == sum(
            source_bits(array, dtype).nbytes for array in arrays
        )
        assert metrics.trainer_spool_files == metrics.trainer_spool_bytes == 0
        assert metrics.python_dense_socket_bytes == metrics.handoff_full_copy_bytes == 0
        assert metrics.disk_replay_bytes == 0


def test_duplicate_replay_is_idempotent_conflict_rejects_and_projection_runs_once():
    values = np.linspace(-4, 9, 33, dtype=np.float32)
    with open_client(101) as client:
        client.install_flat_layout(values.size, source_dtype=DType.F32, payload_max=128)
        client.install_generation(5, deadline_s=20).close()
        operations = []
        for weight in (17, 17):
            with client.allocate(dtype=DType.F32) as buffer:
                with buffer.mapped(DType.F32, write=True) as target:
                    target[:] = values
                buffer.seal()
                operations.append(client.submit(
                    buffer, trainer_key=key(102), trainer_incarnation=key(103),
                    submission_seq=4, weight=weight,
                ))
        with client.allocate(dtype=DType.F32) as conflict:
            with conflict.mapped(DType.F32, write=True) as target:
                target[:] = values
            conflict.seal()
            with pytest.raises(ConflictError):
                client.submit(
                    conflict, trainer_key=key(102), trainer_incarnation=key(103),
                    submission_seq=4, weight=18,
                )
            with pytest.raises(ConflictError):
                client.submit(
                    conflict, trainer_key=key(102), trainer_incarnation=key(103),
                    submission_seq=5, weight=17,
                )
        client.control(Command.FREEZE).close()
        first = client.control(Command.FINALIZE_OWNERS)
        second = client.control(Command.FINALIZE_OWNERS)
        assert first.handle == second.handle
        with client.result_view(first) as result:
            assert result.global_weight == 17
            with result.mapped(DType.F32) as actual:
                assert np.array_equal(actual, values)
        assert client.metrics.duplicate_count == 1
        assert client.metrics.conflict_count == 2
        assert client.metrics.projection_count == 1
        client.control(Command.COMMIT).close()
        first.close()
        second.close()
        for operation in operations:
            operation.close()
