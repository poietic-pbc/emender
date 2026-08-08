import json

import pytest
import torch

from ndm.data import tokenized_dataset as td


CORPUS_DIGEST = "1" * 64
TOKENIZER_DIGEST = "2" * 64


class _FakeEncoding:
    n_vocab = 512

    def encode(self, text, disallowed_special=()):
        del disallowed_special
        return [ord(char) % self.n_vocab for char in text]


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch):
    monkeypatch.setattr(td.tiktoken, "get_encoding", lambda _name: _FakeEncoding())


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.bin"
    # ASCII keeps byte and character positions aligned while varying each window.
    path.write_bytes(bytes(32 + (index * 37) % 95 for index in range(8192)))
    return path


def identity(*, world=4, context=8, corpus_digest=CORPUS_DIGEST,
             tokenizer_digest=TOKENIZER_DIGEST, schema=td.COUNTER_SAMPLER_SCHEMA):
    return td.CounterSamplerIdentity(
        schema=schema,
        corpus_sha256=corpus_digest,
        tokenizer_sha256=tokenizer_digest,
        sampler_key=42,
        data_world_size=world,
        context_size=context,
    )


def dataset(corpus, *, rank=0, accepted=0, world=4, context=8, max_retries=64):
    return td.TokenizedStreamDataset(
        str(corpus), context, rank=rank, world_size=world,
        tokenizer_name="p50k_base", sampler_identity=identity(
            world=world, context=context), total_accepted_tokens=accepted,
        max_retries=max_retries,
    )


def tensors(batch):
    return batch[0].clone()


def test_uninterrupted_and_checkpoint_resumed_tensors_are_byte_identical(corpus):
    uninterrupted = dataset(corpus)
    uninterrupted.get_batch(3)
    expected = tensors(uninterrupted.get_batch(3))

    accepted = 3 * 4 * 8
    resumed = dataset(corpus, accepted=accepted)
    observed = tensors(resumed.get_batch(3))

    assert torch.equal(observed, expected)
    assert resumed.initial_absolute_rank_sample_index == 3


def test_retry_reproduces_unaccepted_provisional_work(corpus):
    first_attempt = dataset(corpus, accepted=2 * 4 * 8)
    retry_attempt = dataset(corpus, accepted=2 * 4 * 8)

    first = tensors(first_attempt.get_batch(2))
    retried = tensors(retry_attempt.get_batch(2))

    assert torch.equal(retried, first)
    assert retry_attempt.last_batch_sample_ids == first_attempt.last_batch_sample_ids


def test_successful_continuation_advances_to_exact_next_sample(corpus):
    initial = dataset(corpus)
    initial.get_batch(2)
    expected, expected_id = initial.sample_at(2)

    continued = dataset(corpus, accepted=2 * 4 * 8)
    observed = tensors(continued.get_batch(1))[0]

    assert torch.equal(observed, expected)
    assert continued.last_batch_sample_ids == (expected_id,)


def test_batch_grouping_does_not_change_sample_identity_or_tensors(corpus):
    grouped = dataset(corpus)
    grouped_tensors = tensors(grouped.get_batch(4))
    grouped_ids = grouped.last_batch_sample_ids

    split = dataset(corpus)
    split_tensors = torch.cat([
        tensors(split.get_batch(2)), tensors(split.get_batch(2))], dim=0)
    split_ids = tuple(
        split.sample_id(index) for index in range(4))

    assert torch.equal(split_tensors, grouped_tensors)
    assert grouped_ids == split_ids


def test_ranks_are_deterministic_and_distinct(corpus):
    rank0_a = dataset(corpus, rank=0)
    rank0_b = dataset(corpus, rank=0)
    rank1 = dataset(corpus, rank=1)

    sample0_a, id0_a = rank0_a.sample_at(0)
    sample0_b, id0_b = rank0_b.sample_at(0)
    sample1, id1 = rank1.sample_at(0)

    assert id0_a == id0_b
    assert torch.equal(sample0_a, sample0_b)
    assert id1 != id0_a
    assert rank1.candidate_byte_position(0, 0) != rank0_a.candidate_byte_position(0, 0)
    assert not torch.equal(sample1, sample0_a)


def test_bounded_retry_is_pure_and_does_not_perturb_later_samples(corpus, monkeypatch):
    retried = dataset(corpus, max_retries=3)
    ordinary = dataset(corpus, max_retries=3)
    original = retried._tokenize_position
    rejected_position = retried.candidate_byte_position(0, 0)

    monkeypatch.setattr(
        retried, "_tokenize_position",
        lambda pos: None if pos == rejected_position else original(pos))

    sample, sample_id = retried.sample_at(0)
    assert sample_id == retried.sample_id(0, 1)
    assert torch.equal(sample, ordinary._tokenize_position(
        ordinary.candidate_byte_position(0, 1)))
    later_retried, later_id = retried.sample_at(1)
    later_ordinary, ordinary_id = ordinary.sample_at(1)
    assert later_id == ordinary_id
    assert torch.equal(later_retried, later_ordinary)


def test_retry_exhaustion_fails_closed(corpus, monkeypatch):
    stream = dataset(corpus, max_retries=2)
    monkeypatch.setattr(stream, "_tokenize_position", lambda _pos: None)
    with pytest.raises(RuntimeError, match="exhausted 2 retries"):
        stream.sample_at(7)


@pytest.mark.parametrize("accepted", [-1, 1, 31, 33])
def test_accepted_token_cursor_requires_exact_division(accepted):
    with pytest.raises(ValueError):
        td.absolute_rank_sample_index(
            accepted, data_world_size=4, context_size=8)


def test_identity_and_cursor_metadata_round_trip_through_json():
    expected = identity()
    metadata = td.sampler_checkpoint_metadata(
        expected, total_accepted_tokens=6 * 4 * 8)
    restored = json.loads(json.dumps(metadata))

    observed_identity, accepted, cursor = td.restore_sampler_checkpoint_metadata(
        restored, expected_identity=expected)

    assert observed_identity == expected
    assert accepted == 192
    assert cursor == 6


@pytest.mark.parametrize("field,replacement", [
    ("schema", "future-schema"),
    ("corpus_sha256", "3" * 64),
    ("tokenizer_sha256", "4" * 64),
    ("sampler_key", 43),
    ("data_world_size", 8),
    ("context_size", 16),
])
def test_schema_cursor_world_context_corpus_tokenizer_mismatch_fails_closed(
        field, replacement):
    expected = identity()
    metadata = td.sampler_checkpoint_metadata(expected, total_accepted_tokens=0)
    metadata["identity"][field] = replacement
    with pytest.raises(ValueError):
        td.restore_sampler_checkpoint_metadata(
            metadata, expected_identity=expected)


def test_stored_cursor_mismatch_fails_closed():
    expected = identity()
    metadata = td.sampler_checkpoint_metadata(expected, total_accepted_tokens=0)
    metadata["absolute_rank_sample_index"] = 1
    with pytest.raises(ValueError, match="stored sampler cursor"):
        td.restore_sampler_checkpoint_metadata(
            metadata, expected_identity=expected)


def test_counter_dataset_requires_complete_matching_authority(corpus):
    with pytest.raises(ValueError, match="requires total_accepted_tokens"):
        td.TokenizedStreamDataset(
            str(corpus), 8, world_size=4, sampler_identity=identity())
    with pytest.raises(ValueError, match="world_size mismatches"):
        td.TokenizedStreamDataset(
            str(corpus), 8, world_size=2, sampler_identity=identity(),
            total_accepted_tokens=0)
    with pytest.raises(ValueError, match="chunk_size mismatches"):
        td.TokenizedStreamDataset(
            str(corpus), 16, world_size=4, sampler_identity=identity(),
            total_accepted_tokens=0)
    with pytest.raises(ValueError, match="requires a counter sampler identity"):
        td.TokenizedStreamDataset(
            str(corpus), 8, world_size=4, total_accepted_tokens=0)


def test_legacy_mode_is_explicitly_labelled(corpus):
    legacy = td.TokenizedStreamDataset(str(corpus), 8, seed=42)
    assert legacy.get_stats()["sampler_schema"] == td.LEGACY_SAMPLER_SCHEMA
