import pytest

from ndm.data.tokenized_dataset import (
    COUNTER_SAMPLER_SCHEMA,
    LEGACY_SAMPLER_SCHEMA,
    CounterSamplerIdentity,
    sampler_checkpoint_metadata,
)
from ndm.e97_moe_checkpoint import (
    SCHEMA,
    _replicated_owner,
    validate_moe_sampler_manifest,
)


def _identity(**changes):
    values = dict(
        schema=COUNTER_SAMPLER_SCHEMA,
        corpus_sha256="1" * 64,
        tokenizer_sha256="2" * 64,
        sampler_key=42,
        data_world_size=2048,
        context_size=2048,
    )
    values.update(changes)
    return CounterSamplerIdentity(**values)


def _manifest(*, step=80, accepted_tokens=2048 * 2048 * 2,
              sampler="counter"):
    result = {
        "schema": SCHEMA,
        "complete": True,
        "step": step,
        "accepted_tokens": accepted_tokens,
    }
    if sampler == "counter":
        result["sampler"] = sampler_checkpoint_metadata(
            _identity(), total_accepted_tokens=accepted_tokens)
    elif sampler == "legacy":
        result["sampler"] = {
            "schema": LEGACY_SAMPLER_SCHEMA, "status": "legacy"}
    elif sampler == "missing":
        pass
    else:
        raise AssertionError(sampler)
    return result


def test_checkpoint_schema_and_replicated_ownership_are_stable_and_complete():
    assert SCHEMA == "emender-e97-moe-sharded-v1"
    names = [f"layers.{layer}.mixer.parameter.{index}"
             for layer in range(11) for index in range(17)]
    first = [_replicated_owner(name) for name in names]
    second = [_replicated_owner(name) for name in names]
    assert first == second
    assert all(0 <= owner < 8 for owner in first)
    assert set(first) == set(range(8))


def test_counter_sampler_manifest_restores_exact_authority():
    assert validate_moe_sampler_manifest(
        _manifest(), expected_identity=_identity()) == "counter"


@pytest.mark.parametrize("field,replacement", [
    ("schema", "future"),
    ("corpus_sha256", "3" * 64),
    ("tokenizer_sha256", "4" * 64),
    ("sampler_key", 43),
    ("data_world_size", 1024),
    ("context_size", 4096),
])
def test_counter_sampler_manifest_identity_drift_fails_closed(field, replacement):
    expected = _identity()
    manifest = _manifest()
    manifest["sampler"]["identity"][field] = replacement
    with pytest.raises(RuntimeError, match="sampler metadata mismatch"):
        validate_moe_sampler_manifest(manifest, expected_identity=expected)


def test_counter_sampler_manifest_cursor_drift_fails_closed():
    manifest = _manifest()
    manifest["sampler"]["absolute_rank_sample_index"] += 1
    with pytest.raises(RuntimeError, match="sampler metadata mismatch"):
        validate_moe_sampler_manifest(manifest, expected_identity=_identity())


@pytest.mark.parametrize("legacy_kind", ["legacy", "missing"])
def test_legacy_checkpoint_is_not_silently_relabelled(legacy_kind):
    with pytest.raises(RuntimeError, match="cannot be silently relabelled"):
        validate_moe_sampler_manifest(
            _manifest(sampler=legacy_kind), expected_identity=_identity())


@pytest.mark.parametrize("legacy_kind", ["legacy", "missing"])
def test_explicit_legacy_transition_requires_complete_k_boundary(legacy_kind):
    manifest = _manifest(step=81, sampler=legacy_kind)
    with pytest.raises(RuntimeError, match="K-aligned"):
        validate_moe_sampler_manifest(
            manifest, expected_identity=_identity(),
            allow_legacy_transition=True, diloco_k=40)

    manifest["step"] = 80
    assert validate_moe_sampler_manifest(
        manifest, expected_identity=_identity(),
        allow_legacy_transition=True, diloco_k=40) == "legacy-transition"


def test_legacy_launch_refuses_counter_checkpoint():
    with pytest.raises(RuntimeError, match="launch selects legacy"):
        validate_moe_sampler_manifest(
            _manifest(), expected_identity=None)


def test_historical_missing_metadata_remains_explicitly_legacy():
    assert validate_moe_sampler_manifest(
        _manifest(sampler="missing"), expected_identity=None) == "legacy"
