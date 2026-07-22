import threading
import time

import pytest

from ndm.native_pipeline import (BackgroundWork, CommittedResult,
                                 GenerationIdentity, LiveNativeGenerationScheduler,
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


def test_live_scheduler_direct_timestamps_prove_g0_background_overlaps_g1_k40():
    """Regression for job 5047497's foreground result_shards wait."""
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    events = []
    scheduler = LiveNativeGenerationScheduler(pipe, telemetry=events.append)
    background_entered = threading.Event()
    release_background = threading.Event()

    def background(payload, phase):
        phase("collection_start")
        background_entered.set()
        assert release_background.wait(.5)
        phase("checkpoint_publication")
        return result(0, payload)

    scheduler.enqueue(BackgroundWork(identity(0), 2.0, background))
    assert background_entered.wait(.2)
    scheduler.event(identity(1), "k40_start")
    k40_start = time.monotonic_ns()
    time.sleep(.01)
    k40_end = time.monotonic_ns()
    scheduler.event(identity(1), "k40_end")
    release_background.set()
    scheduler.close()
    collection_start = next(e.monotonic_ns for e in events
                            if e.phase == "collection_start")
    publication = next(e.monotonic_ns for e in events
                       if e.phase == "checkpoint_publication")
    assert collection_start < k40_end and publication > k40_start


def test_foreground_enqueue_never_waits_for_background_phases():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    scheduler = LiveNativeGenerationScheduler(pipe)
    release = threading.Event()

    def slow(payload, phase):
        for name in LiveNativeGenerationScheduler.BACKGROUND_PHASES:
            phase(name)
        release.wait(.5)
        return result(0, payload)

    started = time.monotonic()
    assert scheduler.enqueue(BackgroundWork(identity(0), 1.0, slow))
    assert time.monotonic() - started < .05
    release.set()
    scheduler.close()


def test_latest_only_slow_work_is_bounded_and_reports_replacement():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    entered, release = threading.Event(), threading.Event()
    scheduler = LiveNativeGenerationScheduler(pipe)

    def slow(payload, phase):
        entered.set(); release.wait(.5)
        return result(0, payload)

    assert scheduler.enqueue(BackgroundWork(identity(0), 0.0, slow))
    assert entered.wait(.2)
    assert scheduler.enqueue(BackgroundWork(identity(1), 1.0, slow))
    assert scheduler.enqueue(BackgroundWork(identity(2), 2.0, slow))
    assert pipe.metrics.handoff_replacements == 1
    assert pipe.metrics.dropped_handoffs == 1
    release.set(); scheduler.close()


def test_publication_failure_is_nonparticipation_and_restart_pending_result_applies_once():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    events = []
    scheduler = LiveNativeGenerationScheduler(pipe, telemetry=events.append)

    def failed(payload, phase):
        phase("integrity_scan")
        raise OSError("atomic checkpoint promotion failed")

    scheduler.enqueue(BackgroundWork(identity(0), 1.0, failed))
    scheduler.close()
    assert any(e.phase == "background_failed" for e in events)
    assert scheduler.apply_at_safe_boundary(identity(0), apply=lambda _: None) is False

    # A checkpoint-restored delayed result remains identity-bound and applies
    # exactly once at its next safe boundary.
    restarted = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    assert restarted.publish_committed(result(1), verify=finite_result_verifier)
    resumed = LiveNativeGenerationScheduler(restarted)
    applied = []
    assert resumed.apply_at_safe_boundary(identity(1), apply=applied.append)
    assert not resumed.apply_at_safe_boundary(identity(1), apply=applied.append)
    assert len(applied) == 1
    resumed.close()


def test_stale_duplicate_nonfinite_and_wrong_route_identity_are_rejected():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    assert pipe.publish_committed(result(1), verify=finite_result_verifier)
    assert not pipe.publish_committed(result(1), verify=finite_result_verifier)
    assert not pipe.publish_committed(result(2, float("inf")), verify=finite_result_verifier)
    bad = __import__("dataclasses").replace(identity(2), route_id="")
    with pytest.raises(ValueError):
        BackgroundWork(bad, 1.0, lambda *_: None).identity.validate()


def test_production_one_generation_delay_applies_g_at_g_plus_1_boundary():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    scheduler = LiveNativeGenerationScheduler(pipe, result_delay=1)
    assert pipe.publish_committed(result(0), verify=finite_result_verifier)
    applied = []
    assert scheduler.apply_at_safe_boundary(identity(1), apply=applied.append)
    assert [item.identity.generation for item in applied] == [0]
    assert not scheduler.apply_at_safe_boundary(identity(1), apply=applied.append)
    scheduler.close()


def test_delayed_boundary_never_admits_future_or_two_generation_stale_result():
    pipe = NativeGenerationPipeline(run_id="run", fence=7, incarnation="boot-a")
    scheduler = LiveNativeGenerationScheduler(pipe, result_delay=1)
    assert pipe.publish_committed(result(0), verify=finite_result_verifier)
    assert not scheduler.apply_at_safe_boundary(identity(2), apply=lambda _: None)
    assert pipe.metrics.stale_results == 1
    scheduler.close()
