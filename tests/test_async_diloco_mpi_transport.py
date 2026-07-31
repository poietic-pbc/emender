import pytest

torch = pytest.importorskip("torch")

from ndm.async_diloco import AsyncDiLoCoUpdate
from ndm.async_diloco_mpi import (
    DenseTransportQuorumConfig,
    DenseUpdateEnvelope,
    collect_dense_quorum_from_envelopes,
    pack_dense_update,
    unpack_dense_update,
)


def _update(worker_id, value, *, generation=0, tokens=10):
    return AsyncDiLoCoUpdate(
        worker_id=worker_id,
        base_generation=generation,
        delta={
            "a": torch.tensor([float(value), float(value) + 1.0], dtype=torch.float32),
            "b": torch.tensor([[float(value) + 2.0]], dtype=torch.float32),
        },
        tokens=tokens,
        local_steps=1,
        loss_moving_average={"loss": 1.0 + float(value), "loss_100": 1.0 + float(value)},
    )


def test_dense_update_pack_unpack_preserves_wire_metadata_and_tensors():
    update = _update("rank-1", 3.0, tokens=17)
    envelope = pack_dense_update(
        update,
        run_id="wire",
        rank=1,
        generation=5,
        base_checkpoint="seed/latest.pt",
        bucket_bytes=8,
    )

    assert envelope.header["schema_version"] == 1
    assert envelope.header["transport"] == "cray-mpich-gpu-aware-p2p"
    assert envelope.header["generation"] == 5
    assert envelope.header["base_generation"] == 0
    assert envelope.header["base_checkpoint"] == "seed/latest.pt"
    assert envelope.header["global_generation"] == 5
    assert envelope.header["update_id"] == "rank-1:g000005:base000000"
    assert envelope.header["tokens"] == 17
    assert envelope.header["loss_window"]["loss"] == 4.0
    assert envelope.header["bucket_count"] > 1
    assert all(bucket.checksum_sha256 for bucket in envelope.buckets)

    restored = unpack_dense_update(envelope)
    assert restored.worker_id == update.worker_id
    assert restored.tokens == update.tokens
    assert restored.update_id == "rank-1:g000005:base000000"
    assert restored.global_generation == 5
    assert restored.loss_moving_average["loss_100"] == 4.0
    assert torch.equal(restored.delta["a"], update.delta["a"])
    assert torch.equal(restored.delta["b"], update.delta["b"])


def test_dense_update_unpack_rejects_corrupt_bucket_checksum():
    envelope = pack_dense_update(
        _update("rank-0", 1.0),
        run_id="checksum",
        rank=0,
        generation=0,
        bucket_bytes=64,
    )
    bucket = envelope.buckets[0]
    corrupt = DenseUpdateEnvelope(
        header=dict(envelope.header),
        buckets=(
            type(bucket)(
                index=bucket.index,
                offset=bucket.offset,
                payload=b"X" + bucket.payload[1:],
                checksum_sha256=bucket.checksum_sha256,
                tensor_entries=bucket.tensor_entries,
            ),
            *envelope.buckets[1:],
        ),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        unpack_dense_update(corrupt)


def test_dense_quorum_rejects_stale_generation_and_advances_with_fresh_quorum():
    base = {"a": torch.zeros(2), "b": torch.zeros(1, 1)}
    stale = pack_dense_update(
        _update("stale", 100.0, generation=-1),
        run_id="stale",
        rank=0,
        generation=0,
        bucket_bytes=64,
        staleness=1,
    )
    fresh0 = pack_dense_update(_update("fresh-0", 1.0, tokens=1), run_id="stale", rank=1, generation=0)
    fresh1 = pack_dense_update(_update("fresh-1", 3.0, tokens=3), run_id="stale", rank=2, generation=0)

    result = collect_dense_quorum_from_envelopes(
        base,
        (stale, fresh0, fresh1),
        config=DenseTransportQuorumConfig(
            run_id="stale",
            generation=0,
            base_generation=0,
            requested_ranks=3,
            quorum=2,
        ),
    )

    assert result.metrics.quorum_status == "advanced"
    assert result.metrics.accepted_updates == 2
    assert result.metrics.stale_updates == 1
    assert result.metrics.catchup_events[0]["worker_id"] == "stale"
    assert result.transport_metrics.stale_ranks == (0,)
    assert result.transport_metrics.accepted_ranks == (1, 2)
    assert result.transport_metrics.rejected_ranks == (0,)
    expected_a = (fresh0.header["tokens"] * torch.tensor([1.0, 2.0]) + fresh1.header["tokens"] * torch.tensor([3.0, 4.0])) / 4.0
    assert torch.allclose(result.state["a"], expected_a)


def test_dense_quorum_timeout_advances_without_unanimity():
    base = {"a": torch.zeros(2), "b": torch.zeros(1, 1)}
    envelopes = (
        pack_dense_update(_update("rank-0", 1.0), run_id="timeout", rank=0, generation=0),
        pack_dense_update(_update("rank-1", 2.0), run_id="timeout", rank=1, generation=0),
    )

    result = collect_dense_quorum_from_envelopes(
        base,
        envelopes,
        config=DenseTransportQuorumConfig(
            run_id="timeout",
            generation=0,
            base_generation=0,
            requested_ranks=4,
            quorum=2,
            timeout_s=0.01,
        ),
        timed_out_ranks=(2, 3),
    )

    assert result.metrics.quorum_status == "advanced"
    assert result.metrics.quorum_size == 2
    assert result.metrics.timed_out_updates == 2
    assert result.metrics.missing_updates == 2
    assert [event["worker_id"] for event in result.metrics.catchup_events] == [
        "rank-2",
        "rank-3",
    ]
    assert result.transport_metrics.timed_out_ranks == (2, 3)
    assert result.transport_metrics.missing_ranks == (2, 3)
    assert result.transport_metrics.bytes_received > 0


def test_dense_quorum_fraction_threshold_and_bucket_merge_math():
    base = {"a": torch.ones(2), "b": torch.ones(1, 1)}
    envelopes = tuple(
        pack_dense_update(_update(f"rank-{idx}", float(idx), tokens=idx + 1), run_id="fraction", rank=idx, generation=0, bucket_bytes=4)
        for idx in range(3)
    )

    result = collect_dense_quorum_from_envelopes(
        base,
        envelopes,
        config=DenseTransportQuorumConfig(
            run_id="fraction",
            generation=0,
            base_generation=0,
            requested_ranks=4,
            quorum_fraction=0.75,
            weight_by="tokens",
        ),
        timed_out_ranks=(3,),
    )

    assert result.metrics.quorum_status == "advanced"
    assert result.metrics.quorum_threshold == 3
    assert result.metrics.update_bytes["mpi_dense_payload_received"] > 0
    weighted_delta_a = (
        1 * torch.tensor([0.0, 1.0])
        + 2 * torch.tensor([1.0, 2.0])
        + 3 * torch.tensor([2.0, 3.0])
    ) / 6.0
    assert torch.allclose(result.state["a"], base["a"] + weighted_delta_a)
