import json

import pytest

from ndm.data.masked_sft_dataset import (
    SFTSamplerIdentity,
    sft_checkpoint_metadata,
)
from ndm.data.tokenized_dataset import (
    BOUNDARY_COUNTER_SAMPLER_SCHEMA,
    COUNTER_SAMPLER_SCHEMA,
    LEGACY_SAMPLER_SCHEMA,
    CounterSamplerIdentity,
    sampler_checkpoint_metadata,
)
from ndm.e97_moe_checkpoint import (
    SCHEMA,
    _replicated_owner,
    _resolve_complete_generation,
    validate_moe_sampler_manifest,
    validate_sft_parent_optimizer_transition,
    validate_sft_sampler_manifest,
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


def test_complete_generation_resolves_direct_path_or_latest(tmp_path):
    generation = tmp_path / "step-00000080-tokens-0000000000002048"
    generation.mkdir()
    manifest = {"schema": SCHEMA, "complete": True, "step": 80}
    (generation / "manifest.json").write_text(json.dumps(manifest))
    resolved, observed = _resolve_complete_generation(generation)
    assert resolved == generation
    assert observed == manifest

    (tmp_path / "latest").symlink_to(generation.name)
    resolved, observed = _resolve_complete_generation(tmp_path)
    assert resolved == generation.resolve()
    assert observed == manifest


def test_incomplete_direct_generation_fails_closed(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "complete": False}))
    with pytest.raises(RuntimeError, match="not a complete"):
        _resolve_complete_generation(tmp_path)


def test_sft_parent_optimizer_transition_requires_exact_mature_authority(tmp_path):
    generation = tmp_path / "parent"
    generation.mkdir()
    parent = {
        "manifest_sha256": "a" * 64, "step": 2_338_536,
        "accepted_tokens": 282_070_089_728,
        "generation": str(generation.resolve()),
    }
    group = {
        "k": 16016, "weight_sum": 0.016241008784005282,
        "lr": 0.0001, "lr_max": 0.001007, "train_mode": False,
    }
    manifest = {
        "step": parent["step"], "accepted_tokens": parent["accepted_tokens"],
        "optimizer_groups": [group],
    }
    validate_sft_parent_optimizer_transition(generation, manifest, parent)

    broken = json.loads(json.dumps(manifest))
    broken["optimizer_groups"][0]["k"] = 0
    with pytest.raises(RuntimeError, match="not mature"):
        validate_sft_parent_optimizer_transition(generation, broken, parent)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        validate_sft_parent_optimizer_transition(tmp_path / "wrong", manifest, parent)


def test_sft_world_size_transition_is_explicit_and_k_aligned():
    parent = {
        "manifest_sha256": "a" * 64, "step": 2_338_536,
        "accepted_tokens": 282_070_089_728, "generation": "/authority",
    }
    previous = SFTSamplerIdentity(
        authority_manifest_sha256="b" * 64,
        pack_manifest_sha256="c" * 64, sampler_key=42,
        data_world_size=64, context_size=4096)
    expected = SFTSamplerIdentity(
        authority_manifest_sha256="b" * 64,
        pack_manifest_sha256="c" * 64, sampler_key=42,
        data_world_size=512, context_size=4096)
    sampler = sft_checkpoint_metadata(
        previous, parent=parent, total_tokens=1000,
        assistant_target_tokens=700, absolute_rank_sample_index=4096)
    manifest = {"accepted_tokens": parent["accepted_tokens"] + 1000,
                "sampler": sampler}
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        validate_sft_sampler_manifest(
            manifest, expected_identity=expected, expected_parent=parent,
            diloco_k=64)
    clocks, status, observed = validate_sft_sampler_manifest(
        manifest, expected_identity=expected, expected_parent=parent,
        allow_world_size_transition=True, diloco_k=64)
    assert clocks == (1000, 700, 4096)
    assert status == "sft-world-size-transition"
    assert observed == previous

    sampler["absolute_rank_sample_index"] = 4097
    with pytest.raises(RuntimeError, match="K-aligned cursor"):
        validate_sft_sampler_manifest(
            manifest, expected_identity=expected, expected_parent=parent,
            allow_world_size_transition=True, diloco_k=64)


def test_sft_world_size_transition_rejects_other_identity_changes():
    parent = {
        "manifest_sha256": "a" * 64, "step": 1,
        "accepted_tokens": 10, "generation": "/authority",
    }
    previous = SFTSamplerIdentity(
        authority_manifest_sha256="b" * 64,
        pack_manifest_sha256="c" * 64, sampler_key=42,
        data_world_size=64, context_size=4096)
    expected = SFTSamplerIdentity(
        authority_manifest_sha256="b" * 64,
        pack_manifest_sha256="d" * 64, sampler_key=42,
        data_world_size=512, context_size=4096)
    manifest = {"accepted_tokens": 11, "sampler": sft_checkpoint_metadata(
        previous, parent=parent, total_tokens=1,
        assistant_target_tokens=1, absolute_rank_sample_index=64)}
    with pytest.raises(RuntimeError, match="may change only data_world_size"):
        validate_sft_sampler_manifest(
            manifest, expected_identity=expected, expected_parent=parent,
            allow_world_size_transition=True, diloco_k=64)


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
    accepted = 150_134_063_104
    expected = _identity(
        schema=BOUNDARY_COUNTER_SAMPLER_SCHEMA,
        stream_origin_accepted_tokens=accepted)
    manifest = _manifest(step=81, accepted_tokens=accepted, sampler=legacy_kind)
    with pytest.raises(RuntimeError, match="K-aligned"):
        validate_moe_sampler_manifest(
            manifest, expected_identity=expected,
            allow_legacy_transition=True, diloco_k=40)

    manifest["step"] = 80
    assert validate_moe_sampler_manifest(
        manifest, expected_identity=expected,
        allow_legacy_transition=True, diloco_k=40) == "legacy-transition"


def test_legacy_transition_rejects_v1_and_wrong_v2_origin():
    accepted = 150_134_063_104
    manifest = _manifest(
        step=2_332_080, accepted_tokens=accepted, sampler="missing")
    with pytest.raises(RuntimeError, match="counter-v2"):
        validate_moe_sampler_manifest(
            manifest, expected_identity=_identity(),
            allow_legacy_transition=True, diloco_k=40)
    with pytest.raises(RuntimeError, match="must equal"):
        validate_moe_sampler_manifest(
            manifest, expected_identity=_identity(
                schema=BOUNDARY_COUNTER_SAMPLER_SCHEMA,
                stream_origin_accepted_tokens=accepted - 1),
            allow_legacy_transition=True, diloco_k=40)


def test_counter_context_and_world_transition_is_explicit_and_boundary_relative():
    accepted = 2048 * 2048 * 2
    expected = _identity(
        schema=BOUNDARY_COUNTER_SAMPLER_SCHEMA,
        context_size=131072,
        data_world_size=256,
        stream_origin_accepted_tokens=accepted)
    manifest = _manifest(step=80, accepted_tokens=accepted)
    with pytest.raises(RuntimeError, match="sampler metadata mismatch"):
        validate_moe_sampler_manifest(manifest, expected_identity=expected, diloco_k=1)
    assert validate_moe_sampler_manifest(
        manifest, expected_identity=expected,
        allow_counter_transition=True, diloco_k=1) == "counter-transition"


def test_counter_transition_rejects_wrong_origin_and_incomplete_k_boundary():
    accepted = 2048 * 2048 * 2
    manifest = _manifest(step=81, accepted_tokens=accepted)
    expected = _identity(
        schema=BOUNDARY_COUNTER_SAMPLER_SCHEMA,
        context_size=32768,
        data_world_size=256,
        stream_origin_accepted_tokens=accepted)
    with pytest.raises(RuntimeError, match="K-aligned"):
        validate_moe_sampler_manifest(
            manifest, expected_identity=expected,
            allow_counter_transition=True, diloco_k=2)
    with pytest.raises(RuntimeError, match="stream origin"):
        validate_moe_sampler_manifest(
            _manifest(step=80, accepted_tokens=accepted),
            expected_identity=_identity(
                schema=BOUNDARY_COUNTER_SAMPLER_SCHEMA,
                context_size=32768, data_world_size=256,
                stream_origin_accepted_tokens=accepted - 1),
            allow_counter_transition=True, diloco_k=2)


def test_legacy_launch_refuses_counter_checkpoint():
    with pytest.raises(RuntimeError, match="launch selects legacy"):
        validate_moe_sampler_manifest(
            _manifest(), expected_identity=None)


def test_historical_missing_metadata_remains_explicitly_legacy():
    assert validate_moe_sampler_manifest(
        _manifest(sampler="missing"), expected_identity=None) == "legacy"
