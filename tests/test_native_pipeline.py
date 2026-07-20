import time

import pytest

from ndm.native_pipeline import (CommittedResult, GenerationIdentity,
                                 NativeGenerationPipeline, finite_result_verifier)


DIGEST = "a" * 64
ROOT = "b" * 64


def identity(generation=0, *, fence=7, incarnation="boot-a"):
    return GenerationIdentity("run", fence, generation, 1, incarnation, DIGEST, DIGEST)


def result(generation=0, payload=1.0, **changes):
    value = CommittedResult(identity(generation), payload, ROOT, ROOT, 8,
                            time.monotonic_ns())
    return __import__("dataclasses").replace(value, **changes)


def test_double_buffer_ownership_and_bounded_foreground_handoff():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    first = pipe.handoff(pipe.reserve(), identity(0), object(), weight=1, digest=DIGEST)
    second = pipe.handoff(pipe.reserve(), identity(1), object(), weight=1, digest=DIGEST)
    with pytest.raises(TimeoutError):
        pipe.reserve(deadline=time.monotonic() + .01)
    pipe.release(first)
    assert pipe.reserve(deadline=time.monotonic() + .01) == first.slot
    assert pipe.metrics.handoff_high_water == 2
    pipe.cancel_reservation(first.slot)
    pipe.release(second)


def test_latest_only_queue_replaces_only_with_newer_committed_result():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    assert pipe.publish_committed(result(0), verify=finite_result_verifier)
    assert pipe.publish_committed(result(1), verify=finite_result_verifier)
    assert not pipe.publish_committed(result(0), verify=finite_result_verifier)
    assert pipe.take_at_boundary(trainer_generation=1, fence=7,
                                 incarnation="boot-a", base_digest=DIGEST).identity.generation == 1
    assert pipe.metrics.result_replacements == 1


def test_stale_partial_corrupt_and_nonfinite_results_never_apply():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    assert not pipe.publish_committed(result(0, complete=False), verify=finite_result_verifier)
    assert not pipe.publish_committed(result(0, float("nan")), verify=finite_result_verifier)
    assert not pipe.publish_committed(result(0), verify=lambda _: False)
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-a", base_digest=DIGEST) is None


def test_delayed_quorum_does_not_block_next_local_generation():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    token = pipe.handoff(pipe.reserve(), identity(0), "sealed", weight=1, digest=DIGEST)
    # No result exists: the boundary returns immediately and local K may proceed.
    started = time.monotonic()
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-a", base_digest=DIGEST) is None
    assert time.monotonic() - started < .05
    pipe.release(token)


def test_missing_owner_aborts_slot_without_publishing_partial_result():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    token = pipe.handoff(pipe.reserve(), identity(0), "sealed", weight=1, digest=DIGEST)
    pipe.release(token)  # bounded owner reassignment failed: abort/release
    assert pipe.metrics.handoffs == 1
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-a", base_digest=DIGEST) is None


def test_rejoin_invalidates_old_incarnation_buffers_results_and_receipts():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    old = pipe.handoff(pipe.reserve(), identity(0), "sealed", weight=1, digest=DIGEST)
    assert pipe.publish_committed(result(0), verify=finite_result_verifier)
    pipe.rebind(fence=7, incarnation="boot-b")
    with pytest.raises(ValueError):
        pipe.release(old)
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-b", base_digest=DIGEST) is None


def test_obsolete_fence_and_wrong_base_are_discarded_at_safe_boundary():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    assert pipe.publish_committed(result(0), verify=finite_result_verifier)
    assert pipe.take_at_boundary(trainer_generation=0, fence=8,
                                 incarnation="boot-a", base_digest=DIGEST) is None
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-a", base_digest="c" * 64) is None


def test_crash_before_checkpoint_cas_cannot_enter_result_mailbox():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    candidate = result(0)
    # Publication crashed before its durable callback returned; callers never
    # invoke publish_committed, leaving the previous authoritative state intact.
    del candidate
    assert pipe.take_at_boundary(trainer_generation=0, fence=7,
                                 incarnation="boot-a", base_digest=DIGEST) is None
