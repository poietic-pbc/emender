import hashlib
import math

import pytest
import torch

from ndm.resilient_e97_reducer import ExactWeightedShardReducer, ShardChunk, TensorLayout


def _state(seed, *, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    # Representative full-state features: embedding, E97 split edit gates,
    # matrix state projections, normalization, and tied-output-shaped weights.
    shapes = {
        "embed.weight": (17, 7),
        "layers.0.e97.erase.weight": (7, 7),
        "layers.0.e97.write.weight": (7, 7),
        "layers.0.e97.key.weight": (7, 3),
        "layers.1.e97.state_projection": (3, 7, 7),
        "norm.weight": (7,),
        "output.weight": (17, 7),
    }
    return {name: torch.randn(shape, generator=generator, dtype=torch.float32) * scale
            for name, shape in shapes.items()}


def _reduce(layout, states, weights, order):
    packed = [layout.pack(state) for state in states]
    outputs = []
    for shard in range(layout.shard_count):
        reducer = ExactWeightedShardReducer(layout, shard, max_inflight_bytes=4096)
        for index in order:
            assert reducer.submit(f"worker-{index}", weight=weights[index], chunk=packed[index][shard])
        outputs.append(reducer.finalize([f"worker-{index}" for index in range(len(states))]))
        assert reducer.inflight_bytes == 0
    return layout.unpack(outputs)


def test_incremental_shards_match_float64_reference_for_unequal_token_weights():
    states = [_state(3, scale=1e-4), _state(5, scale=1e4), _state(7, scale=1.0)]
    weights = [3, 1_000_003, 29]
    layout = TensorLayout.from_state(states[0], max_chunk_bytes=104)
    actual = _reduce(layout, states, weights, order=[2, 0, 1])
    reverse = _reduce(layout, states, weights, order=[1, 0, 2])
    for name in states[0]:
        reference = sum(state[name].double() * weight for state, weight in zip(states, weights)) / sum(weights)
        assert torch.equal(actual[name], reverse[name])
        assert torch.allclose(actual[name].double(), reference, rtol=6e-8, atol=2e-7)


def test_full_fresh_equal_weight_cohort_matches_synchronous_diloco():
    base = _state(11)
    deltas = [_state(seed, scale=.01) for seed in range(20, 28)]
    layout = TensorLayout.from_state(deltas[0], max_chunk_bytes=80)
    mean = _reduce(layout, deltas, [4096] * 8, order=[7, 3, 0, 5, 1, 6, 2, 4])
    eta_outer = .7
    for name in base:
        synchronous = base[name] + torch.stack([delta[name] for delta in deltas]).mean(0) * eta_outer
        resilient = base[name] + mean[name] * eta_outer
        assert torch.allclose(resilient, synchronous, rtol=1e-6, atol=1e-7)


def test_layout_is_deterministic_bounded_and_owner_mapping_is_stable():
    state = _state(13)
    layout = TensorLayout.from_state(state, max_chunk_bytes=72)
    again = TensorLayout.from_state(dict(reversed(tuple(state.items()))), max_chunk_bytes=72)
    chunks = layout.pack(state)
    assert layout.digest == again.digest
    assert all(chunk.nbytes <= 72 for chunk in chunks)
    assert sum(chunk.elements for chunk in chunks) == layout.total_elements
    owners = [layout.owner(i, ("node-c", "node-a", "node-b"), run_id="e97",
                           generation=9, attempt=2) for i in range(layout.shard_count)]
    assert owners == [again.owner(i, ("node-b", "node-c", "node-a"), run_id="e97",
                                  generation=9, attempt=2) for i in range(layout.shard_count)]
    assert len(set(owners)) > 1  # no central full-model owner in this fixture


def test_checksum_backpressure_idempotence_conflicts_and_prompt_release():
    state = _state(17)
    layout = TensorLayout.from_state(state, max_chunk_bytes=64)
    chunk = layout.pack(state)[0]
    reducer = ExactWeightedShardReducer(layout, 0, max_inflight_bytes=chunk.nbytes)
    assert reducer.submit("worker", weight=7, chunk=chunk)
    assert not reducer.submit("worker", weight=7, chunk=chunk)
    with pytest.raises(ValueError, match="conflicting"):
        reducer.submit("worker", weight=8, chunk=chunk)
    with pytest.raises(BufferError, match="backpressure"):
        reducer.submit("other", weight=1, chunk=chunk)
    corrupt = ShardChunk(chunk.layout_digest, chunk.shard_id, chunk.element_offset,
                         chunk.elements, b"x" + chunk.payload[1:], chunk.checksum_sha256)
    with pytest.raises(ValueError, match="checksum"):
        ExactWeightedShardReducer(layout, 0, max_inflight_bytes=1024).submit(
            "bad", weight=1, chunk=corrupt)
    result = reducer.finalize(["worker"])
    assert reducer.inflight_bytes == 0
    assert reducer.high_water_bytes == chunk.nbytes
    assert result.checksum_sha256 == hashlib.sha256(result.payload).hexdigest()


def test_layout_rejects_shape_nonfinite_and_invalid_chunk_bounds():
    state = _state(19)
    layout = TensorLayout.from_state(state, max_chunk_bytes=64)
    wrong = dict(state); wrong["norm.weight"] = torch.ones(8)
    with pytest.raises(ValueError, match="layout"):
        layout.pack(wrong)
    nonfinite = dict(state); nonfinite["norm.weight"] = torch.full((7,), math.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        layout.pack(nonfinite)
    chunk = layout.pack(state)[0]
    invalid = ShardChunk(chunk.layout_digest, chunk.shard_id, 1, chunk.elements,
                         chunk.payload, chunk.checksum_sha256)
    with pytest.raises(ValueError, match="bounds"):
        ExactWeightedShardReducer(layout, 0, max_inflight_bytes=1024).submit(
            "worker", weight=1, chunk=invalid)
