from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from ndm.native_dataplane import (
    BoundsError, ChecksumError, Client, Command, DType, NativeDataplaneError,
    NonfiniteError, StaleFenceError,
)
from tests.native_dataplane_test_support import key, library, open_client


def _sealed_source(client: Client, values: np.ndarray):
    buffer = client.allocate(dtype=DType.F32)
    with buffer.mapped(DType.F32, write=True) as target:
        target[:] = values
    buffer.seal()
    return buffer


def test_newer_fence_cancels_older_generation_and_preserves_incarnation_boundary():
    native = library()
    old = open_client(121, fence=8, native=native)
    old.install_flat_layout(4, source_dtype=DType.F32, payload_max=64)
    old.install_generation(6, deadline_s=20).close()
    source = _sealed_source(old, np.arange(4, dtype=np.float32))
    old.submit(source, trainer_key=key(122), trainer_incarnation=key(123),
               submission_seq=1, weight=9).close()
    newer = open_client(121, fence=9, native=native)
    try:
        with pytest.raises(StaleFenceError):
            old.control(Command.FREEZE)
        assert any(event.event.name == "ABORTED" and event.reason == 1
                   for event in old.poll())
        newer.install_flat_layout(4, source_dtype=DType.F32, payload_max=64)
        newer.install_generation(6, attempt=2, deadline_s=20).close()
    finally:
        source.close()
        old.close()
        newer.close()


def test_corruption_and_nonfinite_are_rejected_before_accumulator_mutation():
    with open_client(131) as client:
        client.install_flat_layout(8, source_dtype=DType.F32, payload_max=64)
        client.install_generation(1, deadline_s=20).close()
        finite = _sealed_source(client, np.arange(8, dtype=np.float32))
        with pytest.raises(ChecksumError):
            client.submit(
                finite, trainer_key=key(132), trainer_incarnation=key(133),
                submission_seq=1, weight=1, source_sha256=bytes(32),
            )
        finite.close()
        nonfinite = _sealed_source(
            client, np.array([0, 1, 2, np.nan, 4, 5, 6, 7], dtype=np.float32)
        )
        with pytest.raises(NonfiniteError):
            client.submit(
                nonfinite, trainer_key=key(134), trainer_incarnation=key(135),
                submission_seq=1, weight=1,
            )
        nonfinite.close()
        assert client.metrics.checksum_rejects == 1
        assert client.metrics.nonfinite_rejects == 1
        client.control(Command.ABORT).close()


def test_cancellation_releases_retained_source_and_invalidates_finalize():
    with open_client(141) as client:
        client.install_flat_layout(32, source_dtype=DType.F32, payload_max=128)
        client.install_generation(2, deadline_s=20).close()
        source = _sealed_source(client, np.arange(32, dtype=np.float32))
        client.submit(source, trainer_key=key(142), trainer_incarnation=key(143),
                      submission_seq=1, weight=5).close()
        source.close()
        client.control(Command.ABORT).close()
        with pytest.raises(NativeDataplaneError) as failure:
            client.control(Command.FINALIZE_OWNERS)
        assert failure.value.code in (-3, -5)
        assert client.metrics.cancelled_ops >= 1
        assert client.metrics.shared_bytes_current == 0


def test_buffer_slot_and_shared_byte_exhaustion_are_bounded(monkeypatch):
    with open_client(151) as client:
        buffers = [client.allocate(bytes_count=1) for _ in range(64)]
        try:
            with pytest.raises(BoundsError):
                client.allocate(bytes_count=1)
            assert client.metrics.buffer_exhaustions == 1
            assert client.metrics.shared_bytes_high_water == 64
        finally:
            for buffer in buffers:
                buffer.close()
        assert client.metrics.shared_bytes_current == 0

    monkeypatch.setenv("EMENDER_NDP_MAX_SHARED_BYTES", "31")
    with open_client(152) as client:
        with pytest.raises(BoundsError):
            client.allocate(bytes_count=32)
        assert client.metrics.buffer_exhaustions == 1


def test_optional_fallback_materializes_only_one_reduced_numerator(monkeypatch, tmp_path):
    monkeypatch.setenv("EMENDER_NDP_FALLBACK_SPOOL_DIR", str(tmp_path))
    elements = 128
    with open_client(161) as client:
        client.install_flat_layout(elements, source_dtype=DType.F32, payload_max=256)
        client.install_generation(4, deadline_s=20).close()
        submissions = []
        for index in range(8):
            source = _sealed_source(
                client, np.arange(elements, dtype=np.float32) + np.float32(index)
            )
            submissions.append(client.submit(
                source, trainer_key=key(170 + index),
                trainer_incarnation=key(180 + index), submission_seq=1,
                weight=index + 1,
            ))
            source.close()
        client.control(Command.FREEZE).close()
        journals = list(tmp_path.glob("*.replay"))
        assert len(journals) == 1
        journal_bytes = journals[0].stat().st_size
        assert journal_bytes == elements * 8 + 96
        assert journal_bytes <= elements * 8 + 1024 * 1024
        assert client.metrics.disk_replay_files == 1
        assert client.metrics.disk_replay_bytes == journal_bytes
        assert client.metrics.trainer_spool_files == 0
        assert client.metrics.trainer_spool_bytes == 0
        result = client.control(Command.FINALIZE_OWNERS)
        with client.result_view(result):
            pass
        client.control(Command.COMMIT).close()
        result.close()
        for submission in submissions:
            submission.close()
        assert not list(tmp_path.iterdir())


def test_default_steady_state_writes_no_replay_or_trainer_spool(monkeypatch, tmp_path):
    monkeypatch.delenv("EMENDER_NDP_FALLBACK_SPOOL_DIR", raising=False)
    with open_client(191) as client:
        client.install_flat_layout(16, source_dtype=DType.F32, payload_max=64)
        client.install_generation(1, deadline_s=20).close()
        source = _sealed_source(client, np.arange(16, dtype=np.float32))
        submission = client.submit(
            source, trainer_key=key(192), trainer_incarnation=key(193),
            submission_seq=1, weight=1,
        )
        source.close()
        client.control(Command.FREEZE).close()
        result = client.control(Command.FINALIZE_OWNERS)
        with client.result_view(result):
            pass
        client.control(Command.COMMIT).close()
        result.close()
        submission.close()
        metrics = client.metrics
        assert metrics.disk_replay_bytes == metrics.disk_replay_files == 0
        assert metrics.trainer_spool_bytes == metrics.trainer_spool_files == 0
        assert metrics.python_dense_socket_bytes == metrics.handoff_full_copy_bytes == 0
        assert not list(tmp_path.iterdir())
