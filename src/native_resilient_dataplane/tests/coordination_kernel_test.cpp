#include "coordination_kernel.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

namespace coordination = emender_ndp::coordination;

namespace {

coordination::Key key(std::uint8_t value) {
    coordination::Key result{};
    result.fill(value);
    return result;
}

coordination::Digest digest(std::uint8_t value) {
    coordination::Digest result{};
    result.fill(value);
    return result;
}

coordination::Event event(coordination::EventKind kind,
                          const coordination::AuthorityState& state) {
    coordination::Event value;
    value.kind = kind;
    value.run = state.run;
    value.fence = state.fence;
    value.policy_digest = state.policy_digest;
    return value;
}

coordination::AuthorityState apply(
        const coordination::AuthorityState& state,
        const coordination::Event& input,
        coordination::Disposition expected) {
    const auto transition = coordination::step(state, input);
    if (transition.disposition != expected) {
        std::cerr << "unexpected " << coordination::disposition_name(
            transition.disposition) << " for "
                  << coordination::event_name(input.kind) << ", expected "
                  << coordination::disposition_name(expected) << "\n"
                  << transition.trace << "\n";
    }
    assert(transition.disposition == expected);
    assert(transition.trace.find(
        "\"schema\":\"emender-native-coordination-trace-v1\"")
        != std::string::npos);
    assert(transition.trace.find("\"post_state_digest\":\""
        + coordination::hex(transition.post_state_digest) + "\"")
        != std::string::npos);
    assert(transition.trace.size() < coordination::kMaximumTraceBytes);
    assert(!transition.effects.empty());
    assert(transition.effects.back().kind
        == coordination::EffectKind::EmitTrace);
    assert(coordination::invariant(transition.state));
    return transition.state;
}

coordination::AuthorityState configured(std::uint32_t minimum_nodes = 2) {
    coordination::AuthorityState empty;
    coordination::Event recover;
    recover.kind = coordination::EventKind::RecoverAuthority;
    recover.run = key(0x11);
    recover.fence = 5105811;
    recover.policy_digest = digest(0x21);
    recover.minimum_nodes = minimum_nodes;
    recover.minimum_tokens = minimum_nodes;
    return apply(empty, recover, coordination::Disposition::Accepted);
}

coordination::AuthorityState recover_ready(
        coordination::AuthorityState state, std::uint8_t node_value,
        std::uint8_t incarnation_value, std::uint64_t sequence,
        std::uint64_t generation = 0,
        coordination::Digest commit_receipt = {}) {
    auto recover = event(coordination::EventKind::RecoverPeer, state);
    recover.node = key(node_value);
    recover.incarnation = key(incarnation_value);
    recover.sequence = sequence;
    recover.generation = generation;
    recover.receipt_digest = commit_receipt;
    state = apply(state, recover, coordination::Disposition::Accepted);

    auto ready = event(coordination::EventKind::Ready, state);
    ready.node = recover.node;
    ready.incarnation = recover.incarnation;
    ready.sequence = sequence;
    ready.generation = generation;
    return apply(state, ready, coordination::Disposition::Accepted);
}

void test_job5105811_and_next_generation_rejoin() {
    auto state = configured();
    state = recover_ready(state, 1, 11, 1);  // node 0
    state = recover_ready(state, 2, 12, 1);  // node 1
    state = recover_ready(state, 3, 13, 1);  // an independently live peer

    auto open = event(coordination::EventKind::OpenGeneration, state);
    open.generation = 0;
    open.attempt = 1;
    state = apply(state, open, coordination::Disposition::Accepted);
    assert(state.active.cohort.size() == 3);

    auto contribution0 = event(coordination::EventKind::Contribution, state);
    contribution0.generation = 0;
    contribution0.attempt = 1;
    contribution0.node = key(1);
    contribution0.incarnation = key(11);
    contribution0.sequence = 1;
    contribution0.exact_tokens = 10;
    contribution0.payload_digest = digest(0x31);
    state = apply(state, contribution0, coordination::Disposition::Accepted);

    const auto duplicate_before = coordination::state_digest(state);
    const auto duplicate = coordination::step(state, contribution0);
    assert(duplicate.disposition
        == coordination::Disposition::IdenticalDuplicate);
    assert(duplicate.pre_state_digest == duplicate.post_state_digest);
    assert(duplicate.post_state_digest == duplicate_before);

    auto conflicting = contribution0;
    conflicting.payload_digest = digest(0x32);
    const auto conflict = coordination::step(state, conflicting);
    assert(conflict.disposition
        == coordination::Disposition::ConflictingDuplicate);
    assert(conflict.pre_state_digest == conflict.post_state_digest);

    auto contribution2 = contribution0;
    contribution2.node = key(3);
    contribution2.incarnation = key(13);
    contribution2.exact_tokens = 20;
    contribution2.payload_digest = digest(0x33);
    state = apply(state, contribution2, coordination::Disposition::Accepted);

    auto close = event(coordination::EventKind::CloseGeneration, state);
    close.generation = 0;
    close.attempt = 1;
    close.flags = coordination::EventFlagFiniteClose;
    state = apply(state, close, coordination::Disposition::Accepted);
    assert(state.active.phase == coordination::GenerationPhase::Closed);
    const auto frozen_digest = coordination::state_digest(state);
    auto late_different_sequence = contribution0;
    late_different_sequence.sequence = 99;
    const auto late_different =
        coordination::step(state, late_different_sequence);
    assert(late_different.disposition
        == coordination::Disposition::GenerationClosed);
    assert(late_different.pre_state_digest
        == late_different.post_state_digest);
    state = apply(state, close,
                  coordination::Disposition::IdenticalDuplicate);
    assert(coordination::state_digest(state) == frozen_digest);

    for (const auto& node : {key(1), key(3)}) {
        auto result = event(coordination::EventKind::ResultReceipt, state);
        result.generation = 0;
        result.attempt = 1;
        result.node = node;
        result.incarnation = state.active.contributions.at(node).incarnation;
        result.sequence = 1;
        result.exact_tokens = 30;
        result.result_digest = digest(0x41);
        state = apply(state, result, coordination::Disposition::Accepted);
    }

    auto commit = event(coordination::EventKind::Commit, state);
    commit.generation = 1;
    commit.attempt = 1;
    commit.exact_tokens = 30;
    commit.previous_receipt_digest = {};
    commit.receipt_digest = digest(0x51);
    commit.manifest_digest = digest(0x52);
    commit.result_digest = digest(0x41);
    auto corrupt_commit = commit;
    corrupt_commit.exact_tokens = 29;
    const auto corrupt_commit_result =
        coordination::step(state, corrupt_commit);
    assert(corrupt_commit_result.disposition
        == coordination::Disposition::Corrupt);
    assert(corrupt_commit_result.pre_state_digest
        == corrupt_commit_result.post_state_digest);
    state = apply(state, commit, coordination::Disposition::Accepted);
    assert(state.committed_generation == 1);
    assert(state.accepted_token_clock == 30);
    assert(state.active.phase == coordination::GenerationPhase::Committed);
    state = apply(state, commit,
                  coordination::Disposition::IdenticalDuplicate);
    auto conflicting_commit = commit;
    conflicting_commit.receipt_digest = digest(0x53);
    const auto conflicting_commit_result =
        coordination::step(state, conflicting_commit);
    assert(conflicting_commit_result.disposition
        == coordination::Disposition::ConflictingDuplicate);
    assert(conflicting_commit_result.pre_state_digest
        == conflicting_commit_result.post_state_digest);

    // Node 0's complete local cohort fails and recovers under a new,
    // monotonically sequenced incarnation.
    auto recover0 = event(coordination::EventKind::RecoverPeer, state);
    recover0.node = key(1);
    recover0.incarnation = key(21);
    recover0.sequence = 2;
    recover0.generation = 1;
    recover0.receipt_digest = digest(0x51);
    state = apply(state, recover0, coordination::Disposition::Accepted);

    // Permanent job5105811 ordering: node 1 is still alive while submitting
    // new work for the already closed source generation.  This is a typed,
    // nonfatal, non-mutating result whose only action is catch-up/retry.
    auto late1 = event(coordination::EventKind::Contribution, state);
    late1.generation = 0;
    late1.attempt = 1;
    late1.node = key(2);
    late1.incarnation = key(12);
    late1.sequence = 7;
    late1.exact_tokens = 10;
    late1.payload_digest = digest(0x61);
    const auto late_before = coordination::state_digest(state);
    const auto late = coordination::step(state, late1);
    assert(late.disposition == coordination::Disposition::GenerationClosed);
    assert(late.pre_state_digest == late.post_state_digest);
    assert(late.post_state_digest == late_before);
    assert(late.state.members.at(key(2)).live);
    assert(late.state.members.at(key(2)).incarnation == key(12));
    assert(std::none_of(
        late.effects.begin(), late.effects.end(),
        [](const coordination::Effect& item) {
            // Process kill and restart-budget effects do not exist in the
            // pure authority vocabulary.
            return item.kind == coordination::EffectKind::ExpirePeer;
        }));
    state = late.state;

    // A partial seven-trainer report never becomes node apply authority.
    auto partial_apply = event(coordination::EventKind::NodeApply, state);
    partial_apply.generation = 1;
    partial_apply.node = key(1);
    partial_apply.incarnation = key(21);
    partial_apply.sequence = 2;
    partial_apply.trainer_count = 7;
    partial_apply.receipt_digest = digest(0x71);
    partial_apply.previous_receipt_digest = digest(0x51);
    const auto partial = coordination::step(state, partial_apply);
    assert(partial.disposition == coordination::Disposition::Corrupt);
    assert(partial.pre_state_digest == partial.post_state_digest);

    // Node 0 and the uninjected node 1 both complete an atomic all-eight
    // apply/reload, advertise READY, and form the next generation cohort.
    partial_apply.trainer_count = 8;
    state = apply(state, partial_apply, coordination::Disposition::Accepted);
    assert(state.active.phase == coordination::GenerationPhase::Committed);
    auto ready0 = event(coordination::EventKind::Ready, state);
    ready0.generation = 1;
    ready0.node = key(1);
    ready0.incarnation = key(21);
    ready0.sequence = 2;
    ready0.receipt_digest = digest(0x71);
    state = apply(state, ready0, coordination::Disposition::Accepted);

    auto recover1 = event(coordination::EventKind::RecoverPeer, state);
    recover1.generation = 1;
    recover1.node = key(2);
    recover1.incarnation = key(12);
    recover1.sequence = 2;
    recover1.receipt_digest = digest(0x51);
    state = apply(state, recover1, coordination::Disposition::Accepted);
    auto apply1 = partial_apply;
    apply1.node = key(2);
    apply1.incarnation = key(12);
    apply1.receipt_digest = digest(0x72);
    state = apply(state, apply1, coordination::Disposition::Accepted);
    assert(state.active.phase == coordination::GenerationPhase::Committed);
    auto apply2 = partial_apply;
    apply2.node = key(3);
    apply2.incarnation = key(13);
    apply2.receipt_digest = digest(0x73);
    state = apply(state, apply2, coordination::Disposition::Accepted);
    assert(state.active.phase == coordination::GenerationPhase::Applied);
    auto ready1 = ready0;
    ready1.node = key(2);
    ready1.incarnation = key(12);
    ready1.receipt_digest = digest(0x72);
    state = apply(state, ready1, coordination::Disposition::Accepted);

    auto next_open = event(coordination::EventKind::OpenGeneration, state);
    next_open.generation = 1;
    next_open.attempt = 1;
    state = apply(state, next_open, coordination::Disposition::Accepted);
    assert(state.active.cohort.at(key(1)) == key(21));
    assert(state.active.cohort.at(key(2)) == key(12));

    // A later commit preserves the prior apply receipts as monotonic history
    // while requiring a fresh all-eight receipt before another READY.
    auto next_contribution = event(
        coordination::EventKind::Contribution, state);
    next_contribution.generation = 1;
    next_contribution.attempt = 1;
    next_contribution.node = key(1);
    next_contribution.incarnation = key(21);
    next_contribution.sequence = 3;
    next_contribution.exact_tokens = 5;
    next_contribution.payload_digest = digest(0xa1);
    state = apply(
        state, next_contribution, coordination::Disposition::Accepted);
    next_contribution.node = key(2);
    next_contribution.incarnation = key(12);
    next_contribution.exact_tokens = 7;
    next_contribution.payload_digest = digest(0xa2);
    state = apply(
        state, next_contribution, coordination::Disposition::Accepted);
    auto next_close = event(
        coordination::EventKind::CloseGeneration, state);
    next_close.generation = 1;
    next_close.attempt = 1;
    next_close.flags = coordination::EventFlagFiniteClose;
    state = apply(state, next_close, coordination::Disposition::Accepted);
    for (const auto& node : {key(1), key(2)}) {
        auto result = event(
            coordination::EventKind::ResultReceipt, state);
        result.generation = 1;
        result.attempt = 1;
        result.node = node;
        result.incarnation = state.active.contributions.at(node).incarnation;
        result.sequence = 3;
        result.exact_tokens = 12;
        result.result_digest = digest(0xb1);
        state = apply(state, result, coordination::Disposition::Accepted);
    }
    auto next_commit = event(coordination::EventKind::Commit, state);
    next_commit.generation = 2;
    next_commit.attempt = 1;
    next_commit.exact_tokens = 42;
    next_commit.previous_receipt_digest = digest(0x51);
    next_commit.receipt_digest = digest(0xb2);
    next_commit.manifest_digest = digest(0xb3);
    next_commit.result_digest = digest(0xb1);
    state = apply(
        state, next_commit, coordination::Disposition::Accepted);
    assert(state.committed_generation == 2);
    assert(state.members.at(key(1)).applied_generation == 1);
    assert(state.members.at(key(1)).apply_receipt == digest(0x71));
}

void test_stale_fence_expiry_owner_replay_and_fail_closed_state() {
    auto state = configured(1);
    state = recover_ready(state, 1, 11, 1);

    // Node-apply authority cannot exist at generation zero.  A malformed
    // external report is typed invalid input, not an invariant crash.
    auto zero_apply = event(coordination::EventKind::NodeApply, state);
    zero_apply.node = key(1);
    zero_apply.incarnation = key(11);
    zero_apply.sequence = 1;
    zero_apply.trainer_count = 8;
    zero_apply.receipt_digest = digest(0x19);
    const auto zero_apply_result = coordination::step(state, zero_apply);
    assert(zero_apply_result.disposition
        == coordination::Disposition::InvalidEvent);
    assert(zero_apply_result.pre_state_digest
        == zero_apply_result.post_state_digest);

    auto open = event(coordination::EventKind::OpenGeneration, state);
    open.attempt = 1;
    state = apply(state, open, coordination::Disposition::Accepted);

    auto stale = event(coordination::EventKind::Contribution, state);
    stale.fence -= 1;
    stale.node = key(1);
    stale.incarnation = key(11);
    stale.sequence = 1;
    stale.exact_tokens = 1;
    stale.payload_digest = digest(0x31);
    const auto stale_result = coordination::step(state, stale);
    assert(stale_result.disposition
        == coordination::Disposition::StaleFence);
    assert(stale_result.pre_state_digest == stale_result.post_state_digest);

    auto contribution = stale;
    contribution.fence = state.fence;
    contribution.attempt = 1;
    state = apply(state, contribution,
                  coordination::Disposition::Accepted);

    // Lease expiry is an external liveness fact.  It may remove READY but
    // cannot revoke an already acknowledged immutable contribution receipt.
    auto expire = event(coordination::EventKind::ExpirePeer, state);
    expire.node = key(1);
    expire.incarnation = key(11);
    expire.sequence = 1;
    state = apply(state, expire, coordination::Disposition::Accepted);
    assert(!state.members.at(key(1)).live);
    assert(state.active.contributions.count(key(1)) == 1);
    const auto receipt_replay = coordination::step(state, contribution);
    assert(receipt_replay.disposition
        == coordination::Disposition::IdenticalDuplicate);
    assert(receipt_replay.pre_state_digest
        == receipt_replay.post_state_digest);

    auto close = event(coordination::EventKind::CloseGeneration, state);
    close.attempt = 1;
    close.flags = coordination::EventFlagFiniteClose;
    state = apply(state, close, coordination::Disposition::Accepted);

    auto owner_lost = event(coordination::EventKind::OwnerLost, state);
    owner_lost.attempt = 1;
    owner_lost.node = key(1);
    owner_lost.incarnation = key(11);
    owner_lost.sequence = 1;
    state = apply(state, owner_lost, coordination::Disposition::Accepted);
    assert(state.active.owner_epoch == 2);
    state = apply(
        state, owner_lost,
        coordination::Disposition::IdenticalDuplicate);
    assert(state.active.owner_epoch == 2);
    owner_lost.sequence = 2;
    state = apply(state, owner_lost, coordination::Disposition::Accepted);
    assert(state.active.owner_epoch == 3);
    owner_lost.sequence = 3;
    state = apply(state, owner_lost,
                  coordination::Disposition::RetryNextGeneration);
    assert(state.active.phase == coordination::GenerationPhase::Aborted);

    // An invalid authoritative input state is the sole fatal class, and the
    // kernel cannot "repair" it by silently mutating authority.
    auto corrupt = configured(1);
    corrupt.commit_receipt = digest(0xee);
    auto query = event(coordination::EventKind::QueryCommit, corrupt);
    const auto fatal = coordination::step(corrupt, query);
    assert(fatal.disposition
        == coordination::Disposition::FatalInvariant);
    assert(fatal.pre_state_digest == fatal.post_state_digest);
}

void test_recovery_is_monotonic_and_ready_requires_all_eight_apply() {
    coordination::AuthorityState empty;
    coordination::Event authority;
    authority.kind = coordination::EventKind::RecoverAuthority;
    authority.run = key(0x81);
    authority.fence = 41;
    authority.generation = 5;
    authority.exact_tokens = 100;
    authority.minimum_nodes = 1;
    authority.minimum_tokens = 1;
    authority.policy_digest = digest(0x82);
    authority.receipt_digest = digest(0x83);
    authority.manifest_digest = digest(0x84);
    authority.result_digest = digest(0x85);
    auto state = apply(
        empty, authority, coordination::Disposition::Accepted);

    auto peer = event(coordination::EventKind::RecoverPeer, state);
    peer.generation = 5;
    peer.node = key(7);
    peer.incarnation = key(17);
    peer.sequence = 1;
    peer.receipt_digest = digest(0x83);
    state = apply(state, peer, coordination::Disposition::Accepted);

    auto ready = event(coordination::EventKind::Ready, state);
    ready.generation = 5;
    ready.node = peer.node;
    ready.incarnation = peer.incarnation;
    ready.sequence = peer.sequence;
    ready.receipt_digest = digest(0x91);
    const auto before_apply = coordination::step(state, ready);
    assert(before_apply.disposition == coordination::Disposition::Deferred);
    assert(before_apply.pre_state_digest == before_apply.post_state_digest);

    auto partial = event(coordination::EventKind::NodeApply, state);
    partial.generation = 5;
    partial.node = peer.node;
    partial.incarnation = peer.incarnation;
    partial.sequence = peer.sequence;
    partial.trainer_count = 7;
    partial.receipt_digest = digest(0x91);
    partial.previous_receipt_digest = digest(0x83);
    const auto seven = coordination::step(state, partial);
    assert(seven.disposition == coordination::Disposition::Corrupt);
    assert(seven.pre_state_digest == seven.post_state_digest);

    partial.trainer_count = 8;
    state = apply(state, partial, coordination::Disposition::Accepted);
    state = apply(state, ready, coordination::Disposition::Accepted);

    auto recovered_apply =
        event(coordination::EventKind::RecoverNodeApply, state);
    recovered_apply.generation = 5;
    recovered_apply.node = key(8);
    recovered_apply.incarnation = key(18);
    recovered_apply.trainer_count = 8;
    recovered_apply.receipt_digest = digest(0x92);
    state = apply(
        state, recovered_apply, coordination::Disposition::Accepted);
    state = apply(
        state, recovered_apply,
        coordination::Disposition::IdenticalDuplicate);
    auto recovered_peer = peer;
    recovered_peer.node = key(8);
    recovered_peer.incarnation = key(18);
    state = apply(
        state, recovered_peer, coordination::Disposition::Accepted);
    auto recovered_ready = ready;
    recovered_ready.node = recovered_peer.node;
    recovered_ready.incarnation = recovered_peer.incarnation;
    recovered_ready.receipt_digest = recovered_apply.receipt_digest;
    state = apply(
        state, recovered_ready, coordination::Disposition::Accepted);

    // A new fenced controller may only recover equal-or-newer durable
    // authority.  A rollback proposal is a typed non-mutating stale result.
    auto rollback = authority;
    rollback.fence = 42;
    rollback.generation = 4;
    rollback.exact_tokens = 90;
    rollback.previous_receipt_digest = digest(0x83);
    rollback.receipt_digest = digest(0xa1);
    rollback.manifest_digest = digest(0xa2);
    rollback.result_digest = digest(0xa3);
    const auto rejected = coordination::step(state, rollback);
    assert(rejected.disposition
        == coordination::Disposition::StaleGeneration);
    assert(rejected.pre_state_digest == rejected.post_state_digest);

    // Re-fencing may reload a newer durable generation, but it cannot rewrite
    // the already unique commit authority for the same generation.
    auto conflicting_same_generation = authority;
    conflicting_same_generation.fence = 43;
    conflicting_same_generation.exact_tokens = 101;
    conflicting_same_generation.previous_receipt_digest = digest(0x83);
    conflicting_same_generation.receipt_digest = digest(0xb1);
    conflicting_same_generation.manifest_digest = digest(0xb2);
    conflicting_same_generation.result_digest = digest(0xb3);
    const auto conflicting_recovery =
        coordination::step(state, conflicting_same_generation);
    assert(conflicting_recovery.disposition
        == coordination::Disposition::FatalInvariant);
    assert(conflicting_recovery.pre_state_digest
        == conflicting_recovery.post_state_digest);

    // An unknown event tag is total and nonfatal.
    auto unknown = event(static_cast<coordination::EventKind>(999), state);
    const auto invalid = coordination::step(state, unknown);
    assert(invalid.disposition == coordination::Disposition::InvalidEvent);
    assert(invalid.pre_state_digest == invalid.post_state_digest);
}

void test_bounded_membership() {
    auto state = configured(1);
    for (std::size_t index = 0;
         index != coordination::kMaximumNodes; ++index) {
        coordination::Key node{};
        node[0] = static_cast<std::uint8_t>(index);
        node[1] = static_cast<std::uint8_t>(index >> 8);
        node[15] = 1;
        coordination::Key incarnation = node;
        incarnation[14] = 7;
        auto recover = event(coordination::EventKind::RecoverPeer, state);
        recover.node = node;
        recover.incarnation = incarnation;
        recover.sequence = 1;
        state = apply(state, recover, coordination::Disposition::Accepted);
        auto ready = event(coordination::EventKind::Ready, state);
        ready.node = node;
        ready.incarnation = incarnation;
        ready.sequence = 1;
        state = apply(state, ready, coordination::Disposition::Accepted);
    }
    assert(state.members.size() == coordination::kMaximumNodes);
    auto extra = event(coordination::EventKind::Ready, state);
    extra.node = key(0xfe);
    extra.node[15] = 2;
    extra.incarnation = key(0xfd);
    extra.sequence = 1;
    const auto bounded = coordination::step(state, extra);
    assert(bounded.disposition == coordination::Disposition::Deferred);
    assert(bounded.state.members.size() == coordination::kMaximumNodes);
    assert(bounded.pre_state_digest == bounded.post_state_digest);
}

void test_open_snapshots_below_floor_and_deadline_aborts_without_commit() {
    auto state = configured(2);
    state = recover_ready(state, 1, 11, 1);

    auto open = event(coordination::EventKind::OpenGeneration, state);
    open.generation = 0;
    open.attempt = 1;
    state = apply(state, open, coordination::Disposition::Accepted);
    assert(state.active.cohort.size() == 1);

    auto contribution = event(coordination::EventKind::Contribution, state);
    contribution.generation = 0;
    contribution.attempt = 1;
    contribution.node = key(1);
    contribution.incarnation = key(11);
    contribution.sequence = 1;
    contribution.exact_tokens = 2;
    contribution.payload_digest = digest(0x44);
    state = apply(
        state, contribution, coordination::Disposition::Accepted);

    auto close = event(coordination::EventKind::CloseGeneration, state);
    close.generation = 0;
    close.attempt = 1;
    close.flags = coordination::EventFlagDeadlineExpired;
    state = apply(
        state, close, coordination::Disposition::InsufficientCohort);
    assert(state.active.phase == coordination::GenerationPhase::Aborted);
    assert(state.committed_generation == 0);
    assert(coordination::is_zero(state.commit_receipt));
}

}  // namespace

int main() {
    test_job5105811_and_next_generation_rejoin();
    test_stale_fence_expiry_owner_replay_and_fail_closed_state();
    test_recovery_is_monotonic_and_ready_requires_all_eight_apply();
    test_bounded_membership();
    test_open_snapshots_below_floor_and_deadline_aborts_without_commit();
    std::cout << "native coordination kernel tests passed\n";
    return 0;
}
