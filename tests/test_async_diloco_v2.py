from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, getcontext
import json
import threading
import time

import numpy as np
import pytest
import torch

from ndm.async_diloco_v2 import (
    ASYNC_DECOUPLED_V2,
    AsyncV2CommitAuthority,
    AsyncV2Policy,
    AsyncV2WorkerLane,
    Backpressure,
    ContributionEnvelope,
    LatestResultMailbox,
    OuterState,
    ResultEnvelope,
    ScheduleFreeLocalState,
    StaleContribution,
    build_contribution,
    digest_array,
    reference_aggregate,
    rebase_schedulefree_torch_state,
)
from ndm.native_e97_runtime import GenerationMetadata, NativeTrainerDataPlane


ZERO = "0" * 64
ONE = "1" * 64
TWO = "2" * 64
THREE = "3" * 64


def contribution(
    *,
    worker: str,
    incarnation: str = "boot-a",
    sequence: int = 0,
    q0: int = 0,
    q1: int = 1,
    base_version: int = 6,
    current_version: int = 6,
    tokens: int = 10,
    start=(0.0, 0.0),
    end=(1.0, 1.0),
    fence: int = 7,
    policy: AsyncV2Policy = ASYNC_DECOUPLED_V2,
) -> ContributionEnvelope:
    return build_contribution(
        run_id="run",
        allocation_fence=fence,
        worker_id=worker,
        worker_incarnation=incarnation,
        contribution_sequence=sequence,
        local_window_start=q0,
        local_window_end=q1,
        base_global_version=base_version,
        base_global_digest=ZERO,
        current_global_version=current_version,
        policy=policy,
        layout_digest=ONE,
        code_digest=TWO,
        exact_tokens=tokens,
        interval_start=np.asarray(start, dtype=np.float64),
        interval_end=np.asarray(end, dtype=np.float64),
        window_monotonic_ns=tuple(
            (100 + 20 * offset, 110 + 20 * offset)
            for offset in range(q1 - q0)
        ),
        endpoint_digest=digest_array(np.asarray(end, dtype=np.float64)),
        local_trainer_set_digest=THREE,
        source_dtype="float32",
        shard_roots=(ONE,),
    )


def result(
    *,
    version: int,
    state=(1.0, 2.0),
    fence: int = 7,
    base_version: int | None = None,
    policy: AsyncV2Policy = ASYNC_DECOUPLED_V2,
) -> ResultEnvelope:
    state_array = np.asarray(state, dtype=np.float64)
    return ResultEnvelope.create(
        run_id="run",
        allocation_fence=fence,
        version=version,
        base_version=version - 1 if base_version is None else base_version,
        base_digest=ZERO,
        state=state_array,
        outer=OuterState(step=version, accepted_tokens=version * 10),
        policy_digest=policy.digest,
        layout_digest=ONE,
        code_digest=TWO,
        manifest_digest=THREE,
        selected_contribution_digests=(),
        reload_verified=True,
        latest_cas_verified=True,
    )


def test_reference_math_fresh_lagged_unequal_tokens_membership_and_tau_boundary():
    records = (
        contribution(
            worker="node-a",
            base_version=6,
            current_version=6,
            tokens=11,
            start=(1.0, -4.0),
            end=(3.0, 2.0),
        ),
        contribution(
            worker="node-c",
            incarnation="rejoin-c",
            sequence=4,
            q0=9,
            q1=12,
            base_version=0,
            current_version=6,
            tokens=5,
            start=(-1.0, 3.0),
            end=(6.0, 1.0),
        ),
    )
    mean, exact_tokens, admitted = reference_aggregate(6, records)

    getcontext().prec = 80
    expected = []
    for component in range(2):
        fresh_delta = Decimal(str(records[0].delta[component]))
        tau_delta = Decimal(str(records[1].delta[component]))
        numerator = Decimal(11 * 7) * fresh_delta + Decimal(5) * tau_delta
        expected.append(float(numerator / Decimal(11 * 7 + 5)))
    np.testing.assert_array_equal(mean, np.asarray(expected, dtype=np.float64))
    assert exact_tokens == 16
    assert [entry.commit_lag for entry in admitted] == [6, 0]
    assert {entry.worker_id for entry in admitted} == {"node-a", "node-c"}

    with pytest.raises(StaleContribution, match="lag 7"):
        reference_aggregate(
            7,
            (replace(records[1], identity=replace(
                records[1].identity, base_lag_at_seal=6)),),
        )


def test_half_step_outer_state_and_current_membership_are_authoritative():
    authority = AsyncV2CommitAuthority(
        run_id="run",
        fence=7,
        state=np.asarray([10.0, 20.0]),
        version=6,
        outer=OuterState(step=6, accepted_tokens=100),
        policy=ASYNC_DECOUPLED_V2,
        layout_digest=ONE,
        code_digest=TWO,
        version_digests={6: ZERO, 0: ZERO},
        minimum_tokens=1,
    )
    authority.install_membership({"node-a": "boot-a", "node-c": "rejoin-c"})
    records = (
        contribution(worker="node-a", tokens=11, start=(0, 0), end=(2, 6)),
        contribution(
            worker="node-c",
            incarnation="rejoin-c",
            base_version=0,
            current_version=6,
            tokens=5,
            start=(0, 0),
            end=(7, -2),
        ),
    )
    committed = authority.commit(records)
    mean, _, _ = reference_aggregate(6, records)
    np.testing.assert_array_equal(
        committed.state, np.asarray([10.0, 20.0]) + 0.5 * mean)
    assert committed.outer == OuterState(step=7, accepted_tokens=116)
    assert authority.version == 7

    late_old_incarnation = contribution(
        worker="node-c",
        incarnation="boot-c",
        sequence=9,
        base_version=6,
        current_version=7,
    )
    with pytest.raises(ValueError, match="incarnation"):
        authority.commit((late_old_incarnation,))


def test_continuous_lane_progresses_g_plus_one_while_g_service_is_delayed():
    lane = AsyncV2WorkerLane(
        run_id="run",
        fence=7,
        worker_id="node-a",
        incarnation="boot-a",
        local=ScheduleFreeLocalState(
            x=np.asarray([0.0]),
            parameter_points={"z": np.asarray([-1.0])},
            retained_buffers={"step": 40, "exp_avg_sq": np.asarray([3.0])},
        ),
        anchor_version=0,
        anchor_state=np.asarray([0.0]),
        anchor_digest=ZERO,
        layout_digest=ONE,
        code_digest=TWO,
        policy=ASYNC_DECOUPLED_V2,
    )
    lane.finish_window(np.asarray([1.0]), exact_tokens=10, begin_ns=1, end_ns=2)
    sealed = lane.seal()

    service_started = threading.Event()
    service_release = threading.Event()
    service_done = threading.Event()

    def delayed_native_service() -> None:
        service_started.set()
        assert service_release.wait(1.0)
        lane.release_sealed(sealed.digest, outcome="accepted")
        service_done.set()

    thread = threading.Thread(target=delayed_native_service)
    thread.start()
    assert service_started.wait(0.2)

    # No unavailable global base is invented: the next local K window begins
    # at the local endpoint while the interval retains anchor S_0.
    lane.finish_window(np.asarray([3.0]), exact_tokens=12, begin_ns=3, end_ns=4)
    assert lane.local_window == 2
    assert lane.applied_anchor_version == 0
    assert sealed.identity.base_global_version == 0
    assert sealed.identity.local_window_start == 0
    assert sealed.identity.local_window_end == 1
    assert lane.mutable_window_range == (1, 2)

    service_release.set()
    assert service_done.wait(0.2)
    thread.join()


def test_one_sealed_one_mutable_coalescing_and_lag_limit_pause_drop_catchup():
    lane = AsyncV2WorkerLane.for_test()
    lane.finish_window(np.asarray([1.0]), exact_tokens=1, begin_ns=1, end_ns=2)
    first = lane.seal()
    for index in range(2, 10):
        lane.finish_window(
            np.asarray([float(index)]),
            exact_tokens=index,
            begin_ns=index * 2 - 1,
            end_ns=index * 2,
        )
    assert lane.mutable_window_range == (1, 9)
    assert lane.mutable_window_count == ASYNC_DECOUPLED_V2.sigma_hard
    assert lane.paused_reason == "mutable_interval_limit"
    assert lane.high_water == {
        "sealed_descriptors": 1,
        "mutable_intervals": 1,
        "mutable_windows": 8,
    }
    with pytest.raises(Backpressure, match="paused"):
        lane.finish_window(np.asarray([10.0]), exact_tokens=1, begin_ns=19, end_ns=20)

    lane.release_sealed(first.digest, outcome="stale_drop")
    second = lane.seal()
    assert second.identity.local_window_start == 1
    assert second.identity.local_window_end == 9
    assert second.identity.window_count == 8
    assert second.exact_tokens == sum(range(2, 10))

    # A known global version at the hard anchor bound pauses if no verified
    # mailbox result exists; an explicit catch-up result resumes the lane.
    assert lane.apply_latest_at_boundary(known_global_version=6) is False
    assert lane.paused_reason == "catch_up_required"
    lane.mailbox.publish(result(version=6, state=(6.0,), base_version=0))
    assert lane.apply_latest_at_boundary(known_global_version=6)
    assert lane.paused_reason is None
    assert lane.applied_anchor_version == 6


def test_unsealed_interval_drops_beyond_tau_after_verified_catchup():
    lane = AsyncV2WorkerLane.for_test()
    lane.finish_window(
        np.asarray([1.0]), exact_tokens=5, begin_ns=1, end_ns=2)
    lane.mailbox.publish(result(version=6, state=(6.0,), base_version=0))
    assert lane.apply_latest_at_boundary(known_global_version=6)
    assert lane.mutable_window_range == (0, 1)

    lane.mailbox.publish(result(version=7, state=(8.0,), base_version=6))
    assert lane.apply_latest_at_boundary(known_global_version=7)
    assert lane.stale_drop_count == 1
    assert lane.mutable_window_range == (1, 1)
    # Dropping an inadmissible contribution interval does not erase the
    # trainer's disposable local displacement.
    np.testing.assert_array_equal(lane.local.x, np.asarray([9.0]))


def test_latest_only_mailbox_view_staging_idempotence_and_rejections():
    mailbox = LatestResultMailbox(
        run_id="run",
        fence=7,
        policy_digest=ASYNC_DECOUPLED_V2.digest,
        layout_digest=ONE,
        code_digest=TWO,
    )
    first = result(version=1)
    assert mailbox.publish(first) == "published"
    assert mailbox.publish(first) == "duplicate"
    lease = mailbox.take()
    assert lease is not None and lease.result.version == 1
    assert mailbox.publish(result(version=2, state=(2.0, 3.0))) == "staged"
    with pytest.raises(Backpressure, match="staging"):
        mailbox.publish(result(version=3, state=(3.0, 4.0)))
    lease.release()
    newest = mailbox.take()
    assert newest is not None and newest.result.version == 2
    newest.release()
    assert mailbox.high_water == {"visible": 1, "staging": 1, "held": 1}

    assert mailbox.publish(result(version=1)) == "duplicate"
    with pytest.raises(ValueError, match="equal-version conflict"):
        mailbox.publish(replace(first, result_digest=TWO))
    with pytest.raises(ValueError, match="fence"):
        mailbox.publish(result(version=4, fence=8))
    with pytest.raises(ValueError, match="nonfinite"):
        mailbox.publish(result(version=4, state=(float("nan"), 0.0)))


def test_safe_boundary_rebase_uses_accepted_ledger_and_translates_only_points():
    lane = AsyncV2WorkerLane.for_test(
        local=ScheduleFreeLocalState(
            x=np.asarray([18.0, 4.0]),
            parameter_points={"z": np.asarray([100.0, -3.0])},
            retained_buffers={
                "step": 81,
                "exp_avg_sq": np.asarray([7.0, 8.0]),
                "loss_scale": 1024.0,
            },
        ),
        anchor_state=np.asarray([10.0, 1.0]),
    )
    accepted = contribution(
        worker="node-0",
        base_version=0,
        current_version=0,
        start=(10.0, 1.0),
        end=(13.0, 0.0),
    )
    lane.record_accepted(commit_version=1, contribution=accepted)
    before_buffers = json.dumps(
        lane.local.retained_buffers,
        default=lambda value: np.asarray(value).tolist(),
        sort_keys=True,
    )
    lane.mailbox.publish(result(
        version=2, state=(20.0, 5.0), base_version=0))
    assert lane.apply_latest_at_boundary(known_global_version=2)
    # correction = ([20,5]-[10,1]) - ([3,-1]) = [7,5]
    np.testing.assert_array_equal(lane.local.x, np.asarray([25.0, 9.0]))
    np.testing.assert_array_equal(
        lane.local.parameter_points["z"], np.asarray([107.0, 2.0]))
    assert before_buffers == json.dumps(
        lane.local.retained_buffers,
        default=lambda value: np.asarray(value).tolist(),
        sort_keys=True,
    )

    nonaccepted = AsyncV2WorkerLane.for_test(
        local=ScheduleFreeLocalState(
            x=np.asarray([18.0]),
            parameter_points={"z": np.asarray([100.0])},
        ),
        anchor_state=np.asarray([10.0]),
    )
    nonaccepted.mailbox.publish(result(version=2, state=(20.0,), base_version=0))
    assert nonaccepted.apply_latest_at_boundary(known_global_version=2)
    np.testing.assert_array_equal(nonaccepted.local.x, np.asarray([28.0]))


def test_replay_wrong_identity_corrupt_nonfinite_owner_retry_and_failed_publication():
    authority = AsyncV2CommitAuthority.for_test()
    value = contribution(
        worker="node-a",
        base_version=0,
        current_version=0,
        start=(0.0,),
        end=(2.0,),
    )
    authority.install_membership({"node-a": "boot-a"})
    first = authority.commit((value,))
    assert authority.replay_receipt(value) == first.result_digest
    assert authority.replay_receipt(value) == first.result_digest
    replayed = authority.commit((value,))
    assert replayed is first
    assert authority.version == 1
    assert authority.outer.step == 1

    conflict = replace(
        value,
        identity=replace(value.identity, exact_tokens=value.exact_tokens + 1),
    )
    with pytest.raises(ValueError, match="conflicting replay"):
        authority.replay_receipt(conflict)
    for field, replacement in (
        ("allocation_fence", 8),
        ("base_global_digest", ONE),
        ("layout_digest", TWO),
        ("code_digest", THREE),
        ("policy_digest", ZERO),
    ):
        bad_identity = replace(value.identity, **{field: replacement})
        with pytest.raises(ValueError):
            AsyncV2CommitAuthority.for_test().commit(
                (replace(value, identity=bad_identity),))
    with pytest.raises(ValueError, match="payload digest"):
        AsyncV2CommitAuthority.for_test().commit(
            (replace(value, delta=np.asarray([99.0])),))
    with pytest.raises(ValueError, match="nonfinite"):
        AsyncV2CommitAuthority.for_test().commit(
            (replace(value, delta=np.asarray([float("inf")])),))

    before = AsyncV2CommitAuthority.for_test()
    before.install_membership({"node-a": "boot-a"})
    state = before.state.copy()
    outer = before.outer
    with pytest.raises(OSError, match="publication"):
        before.commit(
            (value,),
            publish=lambda _bundle: (_ for _ in ()).throw(
                OSError("publication failed")),
        )
    np.testing.assert_array_equal(before.state, state)
    assert before.version == 0 and before.outer == outer

    attempts = []
    committed = before.commit(
        (value,),
        owner_apply=lambda attempt, _records: (
            attempts.append(attempt),
            (_ for _ in ()).throw(OSError("lost owner"))
            if attempt == 0 else None,
        )[-1],
    )
    assert attempts == [0, 1]
    assert committed.version == 1


def test_fresh_allocation_checkpoint_restart_restores_only_global_outer_authority(tmp_path):
    authority = AsyncV2CommitAuthority.for_test()
    authority.install_membership({"node-a": "boot-a"})
    value = contribution(
        worker="node-a",
        base_version=0,
        current_version=0,
        start=(0.0,),
        end=(4.0,),
    )
    committed = authority.commit((value,))
    checkpoint = tmp_path / "global-v1.json"
    authority.checkpoint(checkpoint)

    restored = AsyncV2CommitAuthority.restore(
        checkpoint,
        new_fence=8,
        expected_run_id="run",
        expected_policy=ASYNC_DECOUPLED_V2,
        expected_layout_digest=ONE,
        expected_code_digest=TWO,
    )
    np.testing.assert_array_equal(restored.state, committed.state)
    assert restored.outer == committed.outer
    assert restored.version == committed.version
    assert restored.fence == 8
    assert restored.membership == {}
    with pytest.raises(ValueError, match="newer fence"):
        AsyncV2CommitAuthority.restore(
            checkpoint,
            new_fence=7,
            expected_run_id="run",
            expected_policy=ASYNC_DECOUPLED_V2,
            expected_layout_digest=ONE,
            expected_code_digest=TWO,
        )
    with pytest.raises(ValueError, match="current fence"):
        restored.commit((value,))


def test_native_metadata_abi_carries_exact_tokens_separate_from_lag_weight(tmp_path):
    policy = ASYNC_DECOUPLED_V2
    metadata = GenerationMetadata(
        run_id="run",
        fence_epoch=7,
        generation=6,
        attempt=1,
        owner_epoch=1,
        total_elements=4,
        layout_digest=ONE,
        base_digest=ZERO,
        plan_digest=THREE,
        deadline_unix_ns=time.time_ns() + 10_000_000_000,
        runtime_digests={"bundle": TWO},
        policy_id=policy.policy_id,
        policy_digest=policy.digest,
        code_digest=TWO,
        base_global_version=6,
        local_window_start=11,
        local_window_end=13,
    )
    assert metadata.as_json()["schema"] == "emender-native-e97-generation-v2"
    assert GenerationMetadata.from_json(metadata.as_json()) == metadata

    class Buffer:
        def sha256(self):
            return bytes.fromhex(THREE)

        def seal(self):
            return None

        def close(self):
            return None

    class Client:
        weight = None

        def submit(self, _buffer, **kwargs):
            self.weight = kwargs["weight"]
            return object()

    client = Client()
    plane = NativeTrainerDataPlane(
        client, metadata, rank=0, identity="trainer-0",
        incarnation="boot-a", control_root=tmp_path)
    plane.buffer = Buffer()
    identity = {
        "policy_id": policy.policy_id,
        "policy_digest": policy.digest,
        "code_digest": TWO,
        "base_global_version": 6,
        "base_global_digest": ZERO,
        "base_lag_at_seal": 2,
        "local_window_start": 11,
        "local_window_end": 13,
        "window_count": 2,
        "contribution_sequence": 9,
    }
    marker = plane._seal_submit(
        tokens=10,
        aggregation_weight=50,
        contribution_identity=identity,
        deadline_s=1.0,
    )
    assert client.weight == 50
    assert marker["tokens"] == 10
    assert marker["aggregation_weight"] == 50
    assert marker["base_lag_at_seal"] == 2
    assert marker["payload_digest"] == THREE
    assert marker["schema"] == "emender-native-e97-submission-v2"
    assert len(marker["interval_endpoint_digest"]) == 64
    assert len(marker["descriptor_digest"]) == 64

    plane.buffer = Buffer()
    with pytest.raises(ValueError, match="token/lag/window"):
        plane._seal_submit(
            tokens=10,
            aggregation_weight=49,
            contribution_identity=identity,
            deadline_s=1.0,
        )


def test_real_schedulefree_x_z_translation_retains_moments_scalars_and_rejects_unknown():
    local = {"a": torch.tensor([18.0, 4.0])}
    old = {"a": torch.tensor([10.0, 1.0])}
    new = {"a": torch.tensor([20.0, 5.0])}
    coalescing = {"a": torch.tensor([13.0, 0.0])}
    optimizer = {
        "state": {
            4: {
                "z": torch.tensor([100.0, -3.0]),
                "exp_avg_sq": torch.tensor([7.0, 8.0]),
                "step": 81,
                "loss_scale": 1024.0,
            },
        },
        "param_groups": [{"params": [4], "train_mode": True}],
    }
    moment = optimizer["state"][4]["exp_avg_sq"].clone()
    corrections = rebase_schedulefree_torch_state(
        local_state=local,
        old_anchor=old,
        new_anchor=new,
        accepted_local_deltas=({"a": torch.tensor([3.0, -1.0])},),
        optimizer_state_dict=optimizer,
        coalescing_start=coalescing,
    )
    torch.testing.assert_close(corrections["a"], torch.tensor([7.0, 5.0]))
    torch.testing.assert_close(local["a"], torch.tensor([25.0, 9.0]))
    torch.testing.assert_close(
        optimizer["state"][4]["z"], torch.tensor([107.0, 2.0]))
    torch.testing.assert_close(coalescing["a"], torch.tensor([20.0, 5.0]))
    torch.testing.assert_close(optimizer["state"][4]["exp_avg_sq"], moment)
    assert optimizer["state"][4]["step"] == 81
    assert optimizer["state"][4]["loss_scale"] == 1024.0

    optimizer["state"][4]["mystery_point"] = torch.zeros(2)
    with pytest.raises(ValueError, match="unknown parameter-valued"):
        rebase_schedulefree_torch_state(
            local_state=local,
            old_anchor=old,
            new_anchor=new,
            accepted_local_deltas=(),
            optimizer_state_dict=optimizer,
            coalescing_start=coalescing,
        )
