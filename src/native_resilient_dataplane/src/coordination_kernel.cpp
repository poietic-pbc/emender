#include "coordination_kernel.hpp"

#include "sha256.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>

namespace emender_ndp::coordination {
namespace {

template <typename Bytes>
void append_hex(std::ostringstream& output, const Bytes& value) {
    static constexpr char digits[] = "0123456789abcdef";
    for (const std::uint8_t byte : value) {
        output << digits[byte >> 4] << digits[byte & 15U];
    }
}

template <typename Integer>
void append_integer(std::string& output, Integer value) {
    static_assert(std::is_integral<Integer>::value, "integer required");
    for (std::size_t index = 0; index != sizeof(value); ++index) {
        output.push_back(static_cast<char>(
            static_cast<std::uint64_t>(value) >> (index * 8U)));
    }
}

template <typename Bytes>
void append_bytes(std::string& output, const Bytes& value) {
    output.append(reinterpret_cast<const char*>(value.data()), value.size());
}

Digest digest_string(const std::string& value) noexcept {
    try {
        return Sha256::digest(value.data(), value.size());
    } catch (...) {
        return {};
    }
}

bool same_configuration(const AuthorityState& state, const Event& event) {
    return state.run == event.run && state.fence == event.fence
        && state.policy_digest == event.policy_digest
        && state.minimum_nodes == event.minimum_nodes
        && state.minimum_tokens == event.minimum_tokens
        && state.committed_generation == event.generation
        && state.accepted_token_clock == event.exact_tokens
        && state.commit_receipt == event.receipt_digest
        && state.commit_manifest == event.manifest_digest
        && state.committed_result == event.result_digest;
}

std::uint64_t frozen_tokens(const Generation& generation) noexcept {
    std::uint64_t total = 0;
    for (const auto& item : generation.contributions) {
        if (item.second.exact_tokens > std::numeric_limits<std::uint64_t>::max() - total)
            return 0;
        total += item.second.exact_tokens;
    }
    return total;
}

bool phase_at_least(GenerationPhase left, GenerationPhase right) noexcept {
    return static_cast<std::uint32_t>(left) >= static_cast<std::uint32_t>(right);
}

Transition finish(const AuthorityState& before, AuthorityState after,
                  const Event& event, Disposition disposition,
                  std::vector<Effect> effects);

Transition unchanged(const AuthorityState& state, const Event& event,
                     Disposition disposition,
                     std::vector<Effect> effects = {}) {
    return finish(state, state, event, disposition, std::move(effects));
}

Effect effect(EffectKind kind, std::uint64_t generation,
              const Key& node = {}, const Digest& digest = {}) {
    Effect value;
    value.kind = kind;
    value.generation = generation;
    value.node = node;
    value.digest = digest;
    return value;
}

bool matching_event_identity(const AuthorityState& state,
                             const Event& event) noexcept {
    return state.run == event.run && state.fence == event.fence;
}

Transition recover_authority(const AuthorityState& before, const Event& event) {
    if (is_zero(event.run) || event.fence == 0 || is_zero(event.policy_digest)
        || event.minimum_nodes == 0 || event.minimum_nodes > kMaximumNodes
        || event.minimum_tokens == 0
        || (event.generation == 0
            && (!is_zero(event.receipt_digest) || !is_zero(event.manifest_digest)
                || !is_zero(event.result_digest) || event.exact_tokens != 0))
        || (event.generation != 0
            && (is_zero(event.receipt_digest) || is_zero(event.manifest_digest)
                || is_zero(event.result_digest)
                || event.exact_tokens == 0))) {
        return unchanged(before, event, Disposition::InvalidEvent);
    }
    if (before.configured) {
        if (same_configuration(before, event))
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        if (event.run != before.run || event.fence < before.fence)
            return unchanged(before, event, Disposition::StaleFence);
        if (event.fence == before.fence)
            return unchanged(before, event, Disposition::FatalInvariant);
        if (event.generation < before.committed_generation)
            return unchanged(before, event, Disposition::StaleGeneration);
        if (event.generation == before.committed_generation) {
            if (event.exact_tokens != before.accepted_token_clock
                || event.receipt_digest != before.commit_receipt
                || event.manifest_digest != before.commit_manifest
                || event.result_digest != before.committed_result
                || event.previous_receipt_digest != before.commit_receipt)
                return unchanged(
                    before, event, Disposition::FatalInvariant);
        } else if (event.exact_tokens <= before.accepted_token_clock
                   || event.previous_receipt_digest
                       != before.commit_receipt) {
            return unchanged(before, event, Disposition::StaleGeneration);
        }
    }
    AuthorityState after;
    after.configured = true;
    after.run = event.run;
    after.fence = event.fence;
    after.policy_digest = event.policy_digest;
    after.minimum_nodes = event.minimum_nodes;
    after.minimum_tokens = event.minimum_tokens;
    after.committed_generation = event.generation;
    after.accepted_token_clock = event.exact_tokens;
    after.commit_receipt = event.receipt_digest;
    after.commit_manifest = event.manifest_digest;
    after.committed_result = event.result_digest;
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::BindFence, event.generation),
        effect(EffectKind::SyncAuthority, event.generation),
    });
}

Transition recover_node_apply(const AuthorityState& before, const Event& event) {
    if (is_zero(event.node) || is_zero(event.incarnation)
        || is_zero(event.receipt_digest)
        || before.committed_generation == 0
        || event.generation != before.committed_generation
        || event.trainer_count != kRequiredTrainerReceipts) {
        return unchanged(before, event, Disposition::InvalidEvent);
    }
    const auto prior = before.recovered_node_applies.find(event.node);
    if (prior != before.recovered_node_applies.end()) {
        if (prior->second.incarnation == event.incarnation
            && prior->second.receipt_digest == event.receipt_digest)
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        return unchanged(before, event, Disposition::FatalInvariant);
    }
    if (before.recovered_node_applies.size() == kMaximumNodes)
        return unchanged(before, event, Disposition::Deferred);
    AuthorityState after = before;
    after.recovered_node_applies.emplace(
        event.node, RecoveredNodeApply{
            event.incarnation, event.receipt_digest});
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::RecordNodeApply, event.generation,
               event.node, event.receipt_digest),
    });
}

Transition recover_peer(const AuthorityState& before, const Event& event) {
    if (is_zero(event.node) || is_zero(event.incarnation) || event.sequence == 0)
        return unchanged(before, event, Disposition::InvalidEvent);
    if (event.generation != before.committed_generation
        || event.receipt_digest != before.commit_receipt)
        return unchanged(before, event, Disposition::StaleGeneration);
    AuthorityState after = before;
    auto found = after.members.find(event.node);
    if (found == after.members.end()) {
        if (after.members.size() == kMaximumNodes)
            return unchanged(before, event, Disposition::Deferred);
        found = after.members.emplace(event.node, Member{}).first;
    }
    Member& member = found->second;
    if (!is_zero(member.incarnation)) {
        if (event.sequence < member.control_sequence)
            return unchanged(before, event, Disposition::StaleIncarnation);
        if (event.sequence == member.control_sequence) {
            if (event.incarnation == member.incarnation && member.recovering)
                return unchanged(before, event, Disposition::IdenticalDuplicate);
            return unchanged(before, event, Disposition::StaleIncarnation);
        }
    }
    member.incarnation = event.incarnation;
    member.pending_incarnation = event.incarnation;
    member.control_sequence = event.sequence;
    member.live = true;
    member.ready = false;
    member.recovering = true;
    member.synchronized_generation =
        before.committed_generation == 0 ? 0 : member.applied_generation;
    const auto recovered_apply =
        after.recovered_node_applies.find(event.node);
    if (recovered_apply != after.recovered_node_applies.end()) {
        member.applied_generation = before.committed_generation;
        member.apply_receipt = recovered_apply->second.receipt_digest;
        member.synchronized_generation = before.committed_generation;
    }
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::SyncAuthority, before.committed_generation,
               event.node, before.commit_receipt),
    });
}

Transition ready(const AuthorityState& before, const Event& event) {
    if (is_zero(event.node) || is_zero(event.incarnation) || event.sequence == 0)
        return unchanged(before, event, Disposition::InvalidEvent);
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::RetryNextGeneration, {
            effect(EffectKind::RetryNextGeneration, before.committed_generation,
                   event.node, before.commit_receipt),
        });
    if (event.generation > before.committed_generation)
        return unchanged(before, event, Disposition::Deferred);
    AuthorityState after = before;
    auto found = after.members.find(event.node);
    if (found == after.members.end()) {
        if (after.members.size() == kMaximumNodes)
            return unchanged(before, event, Disposition::Deferred);
        Member member;
        member.incarnation = event.incarnation;
        member.pending_incarnation = event.incarnation;
        member.control_sequence = event.sequence;
        member.synchronized_generation = event.generation;
        found = after.members.emplace(event.node, member).first;
    }
    Member& member = found->second;
    if (member.incarnation != event.incarnation)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (event.sequence < member.control_sequence)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (member.ready && member.ready_generation == event.generation
        && event.sequence == member.control_sequence)
        return unchanged(before, event, Disposition::IdenticalDuplicate);
    if (event.generation != 0) {
        if (member.synchronized_generation != event.generation
            || member.applied_generation != event.generation
            || is_zero(member.apply_receipt))
            return unchanged(before, event, Disposition::Deferred);
        if (member.apply_receipt != event.receipt_digest)
            return unchanged(before, event, Disposition::Corrupt);
    }
    member.control_sequence = event.sequence;
    member.ready_generation = event.generation;
    member.live = true;
    member.ready = true;
    member.recovering = false;
    member.pending_incarnation = {};
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::AdvertiseReady, event.generation, event.node),
    });
}

Transition open_generation(const AuthorityState& before, const Event& event) {
    if (event.attempt == 0)
        return unchanged(before, event, Disposition::InvalidEvent);
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::GenerationClosed, {
            effect(EffectKind::RetryNextGeneration, before.committed_generation),
        });
    if (event.generation > before.committed_generation)
        return unchanged(before, event, Disposition::Deferred);
    if (before.active.present) {
        if (before.active.generation == event.generation
            && before.active.attempt == event.attempt)
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        if (before.active.generation == event.generation
            && before.active.phase == GenerationPhase::Aborted
            && event.attempt > before.active.attempt) {
            // A failed finite attempt advances its attempt identity without
            // inventing a commit or rolling back the global generation.
        } else if (before.active.generation >= event.generation)
            return unchanged(before, event, Disposition::StaleGeneration);
    }
    AuthorityState after = before;
    Generation generation;
    generation.present = true;
    generation.generation = event.generation;
    generation.attempt = event.attempt;
    generation.phase = GenerationPhase::Open;
    generation.owner_epoch = 1;
    for (const auto& item : after.members) {
        if (item.second.ready
            && item.second.ready_generation == event.generation)
            generation.cohort.emplace(item.first, item.second.incarnation);
    }
    if (generation.cohort.size() < after.minimum_nodes)
        return unchanged(before, event, Disposition::InsufficientCohort);
    after.active = std::move(generation);
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::StartGeneration, event.generation),
    });
}

Transition contribution(const AuthorityState& before, const Event& event) {
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::GenerationClosed, {
            effect(EffectKind::RetryNextGeneration, before.committed_generation,
                   event.node, before.commit_receipt),
        });
    if (!before.active.present || event.generation != before.active.generation
        || event.attempt != before.active.attempt)
        return unchanged(before, event, Disposition::StaleGeneration);
    const auto prior = before.active.contributions.find(event.node);
    if (prior != before.active.contributions.end()) {
        const Contribution& receipt = prior->second;
        if (receipt.incarnation == event.incarnation
            && receipt.sequence == event.sequence) {
            if (receipt.exact_tokens == event.exact_tokens
                && receipt.payload_digest == event.payload_digest)
                return unchanged(before, event,
                                 Disposition::IdenticalDuplicate, {
                    effect(EffectKind::AcknowledgeReceipt, event.generation,
                           event.node, receipt.receipt_digest),
                });
            return unchanged(before, event,
                             Disposition::ConflictingDuplicate);
        }
        return unchanged(before, event, Disposition::ConflictingDuplicate);
    }
    if (before.active.phase != GenerationPhase::Open)
        return unchanged(before, event, Disposition::GenerationClosed, {
            effect(EffectKind::RetryNextGeneration, before.committed_generation,
                   event.node, before.commit_receipt),
        });
    const auto cohort = before.active.cohort.find(event.node);
    if (cohort == before.active.cohort.end())
        return unchanged(before, event, Disposition::Deferred);
    if (cohort->second != event.incarnation)
        return unchanged(before, event, Disposition::StaleIncarnation);
    const auto member = before.members.find(event.node);
    if (member == before.members.end()
        || member->second.incarnation != event.incarnation
        || !member->second.ready)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (event.sequence == 0 || event.exact_tokens == 0
        || is_zero(event.payload_digest))
        return unchanged(before, event, Disposition::Corrupt);
    const std::uint64_t prior_tokens = frozen_tokens(before.active);
    if ((!before.active.contributions.empty() && prior_tokens == 0)
        || event.exact_tokens
            > std::numeric_limits<std::uint64_t>::max() - prior_tokens)
        return unchanged(before, event, Disposition::Corrupt);
    AuthorityState after = before;
    Contribution accepted;
    accepted.incarnation = event.incarnation;
    accepted.sequence = event.sequence;
    accepted.exact_tokens = event.exact_tokens;
    accepted.payload_digest = event.payload_digest;
    accepted.receipt_digest = contribution_receipt(event);
    after.active.contributions.emplace(event.node, accepted);
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::AcknowledgeReceipt, event.generation,
               event.node, accepted.receipt_digest),
    });
}

Transition close_generation(const AuthorityState& before, const Event& event) {
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::GenerationClosed, {
            effect(EffectKind::RetryNextGeneration, before.committed_generation),
        });
    if (!before.active.present || event.generation != before.active.generation
        || event.attempt != before.active.attempt)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (before.active.phase == GenerationPhase::Closed
        || phase_at_least(before.active.phase, GenerationPhase::Committed))
        return unchanged(before, event, Disposition::IdenticalDuplicate);
    if (before.active.phase == GenerationPhase::Aborted)
        return unchanged(before, event, Disposition::RetryNextGeneration);
    const bool finite_close = (event.flags & EventFlagFiniteClose) != 0;
    const bool deadline = (event.flags & EventFlagDeadlineExpired) != 0;
    if (!finite_close && !deadline)
        return unchanged(before, event, Disposition::Deferred);
    const std::uint64_t tokens = frozen_tokens(before.active);
    const bool enough = before.active.contributions.size() >= before.minimum_nodes
        && tokens >= before.minimum_tokens && tokens != 0;
    if (!enough) {
        if (!deadline)
            return unchanged(before, event, Disposition::InsufficientCohort);
        AuthorityState after = before;
        after.active.phase = GenerationPhase::Aborted;
        return finish(before, std::move(after), event,
                      Disposition::RetryNextGeneration, {
            effect(EffectKind::RetryNextGeneration, event.generation + 1),
        });
    }
    AuthorityState after = before;
    after.active.phase = GenerationPhase::Closed;
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::FreezeCohort, event.generation),
    });
}

Transition result_receipt(const AuthorityState& before, const Event& event) {
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::GenerationClosed);
    if (!before.active.present || event.generation != before.active.generation
        || event.attempt != before.active.attempt)
        return unchanged(before, event, Disposition::StaleGeneration);
    const auto prior = before.active.result_receipts.find(event.node);
    if (prior != before.active.result_receipts.end()) {
        if (prior->second.incarnation == event.incarnation
            && prior->second.sequence == event.sequence
            && prior->second.exact_tokens == event.exact_tokens
            && prior->second.result_digest == event.result_digest)
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        return unchanged(before, event, Disposition::ConflictingDuplicate);
    }
    if (before.active.phase != GenerationPhase::Closed)
        return unchanged(before, event,
                         phase_at_least(before.active.phase,
                                        GenerationPhase::Committed)
                             ? Disposition::GenerationClosed
                             : Disposition::Deferred);
    const auto accepted = before.active.contributions.find(event.node);
    const std::uint64_t exact_tokens = frozen_tokens(before.active);
    if (accepted == before.active.contributions.end())
        return unchanged(before, event, Disposition::Deferred);
    if (accepted->second.incarnation != event.incarnation)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (event.sequence == 0
        || event.sequence != accepted->second.sequence
        || event.exact_tokens != exact_tokens
        || exact_tokens == 0 || is_zero(event.result_digest))
        return unchanged(before, event, Disposition::Corrupt);
    for (const auto& item : before.active.result_receipts) {
        if (item.second.result_digest != event.result_digest)
            return unchanged(before, event,
                             Disposition::ConflictingDuplicate);
    }
    AuthorityState after = before;
    after.active.result_receipts.emplace(event.node, ResultReceipt{
        event.incarnation, event.sequence, event.exact_tokens,
        event.result_digest,
    });
    std::vector<Effect> effects;
    effects.push_back(effect(EffectKind::AcknowledgeReceipt,
                             event.generation, event.node,
                             event.result_digest));
    if (after.active.result_receipts.size()
        == after.active.contributions.size()) {
        effects.push_back(effect(EffectKind::CommitEligible,
                                 event.generation + 1, {}, event.result_digest));
    }
    return finish(before, std::move(after), event,
                  Disposition::Accepted, std::move(effects));
}

Transition commit(const AuthorityState& before, const Event& event) {
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (event.generation == before.committed_generation) {
        if (event.receipt_digest == before.commit_receipt
            && event.manifest_digest == before.commit_manifest
            && event.result_digest == before.committed_result
            && event.exact_tokens == before.accepted_token_clock)
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        return unchanged(before, event, Disposition::FatalInvariant);
    }
    if (event.generation != before.committed_generation + 1
        || !before.active.present
        || before.active.generation + 1 != event.generation
        || before.active.attempt != event.attempt)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (before.active.phase != GenerationPhase::Closed)
        return unchanged(before, event, Disposition::Deferred);
    if (before.active.result_receipts.size()
        != before.active.contributions.size()
        || before.active.result_receipts.empty())
        return unchanged(before, event, Disposition::Deferred);
    const std::uint64_t generation_tokens = frozen_tokens(before.active);
    if (generation_tokens == 0
        || generation_tokens
            > std::numeric_limits<std::uint64_t>::max()
                - before.accepted_token_clock
        || event.exact_tokens
            != before.accepted_token_clock + generation_tokens
        || event.previous_receipt_digest != before.commit_receipt
        || is_zero(event.receipt_digest) || is_zero(event.manifest_digest)
        || is_zero(event.result_digest))
        return unchanged(before, event, Disposition::Corrupt);
    for (const auto& item : before.active.result_receipts) {
        if (item.second.result_digest != event.result_digest
            || item.second.exact_tokens != generation_tokens)
            return unchanged(before, event, Disposition::Corrupt);
    }
    AuthorityState after = before;
    after.committed_generation = event.generation;
    after.accepted_token_clock = event.exact_tokens;
    after.commit_receipt = event.receipt_digest;
    after.commit_manifest = event.manifest_digest;
    after.committed_result = event.result_digest;
    after.active.phase = GenerationPhase::Committed;
    after.recovered_node_applies.clear();
    for (auto& item : after.members) {
        item.second.ready = false;
        // Retain the prior applied-generation receipt as monotonic recovery
        // evidence.  It no longer grants READY because Ready requires the
        // receipt's generation to equal the new commit.
    }
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::PublishCommit, event.generation,
               {}, event.receipt_digest),
    });
}

Transition node_apply(const AuthorityState& before, const Event& event) {
    if (event.generation == 0)
        return unchanged(before, event, Disposition::InvalidEvent);
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (event.generation > before.committed_generation)
        return unchanged(before, event, Disposition::Deferred);
    auto found = before.members.find(event.node);
    if (found == before.members.end()
        || found->second.incarnation != event.incarnation)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (event.sequence < found->second.control_sequence)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (found->second.applied_generation == event.generation
        && !is_zero(found->second.apply_receipt)) {
        if (found->second.apply_receipt == event.receipt_digest)
            return unchanged(before, event, Disposition::IdenticalDuplicate);
        return unchanged(before, event, Disposition::FatalInvariant);
    }
    if (event.trainer_count != kRequiredTrainerReceipts
        || is_zero(event.receipt_digest)
        || event.previous_receipt_digest != before.commit_receipt)
        return unchanged(before, event, Disposition::Corrupt);
    AuthorityState after = before;
    Member& member = after.members[event.node];
    member.control_sequence = event.sequence;
    member.applied_generation = event.generation;
    member.apply_receipt = event.receipt_digest;
    member.synchronized_generation = event.generation;
    after.recovered_node_applies[event.node] = RecoveredNodeApply{
        event.incarnation, event.receipt_digest};
    if (after.active.present
        && after.active.generation + 1 == event.generation
        && after.active.phase == GenerationPhase::Committed)
        after.active.phase = GenerationPhase::Applied;
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::RecordNodeApply, event.generation,
               event.node, event.receipt_digest),
    });
}

Transition expire_peer(const AuthorityState& before, const Event& event) {
    const auto found = before.members.find(event.node);
    if (found == before.members.end()
        || found->second.incarnation != event.incarnation
        || event.sequence < found->second.control_sequence)
        return unchanged(before, event, Disposition::StaleIncarnation);
    if (!found->second.live && !found->second.ready
        && !found->second.recovering)
        return unchanged(before, event, Disposition::IdenticalDuplicate);
    AuthorityState after = before;
    Member& member = after.members[event.node];
    member.live = false;
    member.ready = false;
    member.recovering = false;
    member.pending_incarnation = {};
    member.control_sequence = event.sequence;
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::ExpirePeer, event.generation, event.node),
    });
}

Transition owner_lost(const AuthorityState& before, const Event& event) {
    if (!before.active.present
        || event.generation != before.active.generation
        || event.attempt != before.active.attempt)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (is_zero(event.node) || is_zero(event.incarnation)
        || event.sequence == 0)
        return unchanged(before, event, Disposition::InvalidEvent);
    const auto cohort = before.active.cohort.find(event.node);
    if (cohort == before.active.cohort.end())
        return unchanged(before, event, Disposition::Deferred);
    if (cohort->second != event.incarnation)
        return unchanged(before, event, Disposition::StaleIncarnation);
    const auto duplicate = std::find_if(
        before.active.owner_loss_receipts.begin(),
        before.active.owner_loss_receipts.end(),
        [&](const OwnerLossReceipt& item) {
            return item.node == event.node
                && item.incarnation == event.incarnation
                && item.sequence == event.sequence;
        });
    if (duplicate != before.active.owner_loss_receipts.end())
        return unchanged(
            before, event, Disposition::IdenticalDuplicate);
    if (before.active.phase == GenerationPhase::Aborted)
        return unchanged(
            before, event, Disposition::RetryNextGeneration);
    if (before.active.phase != GenerationPhase::Closed)
        return unchanged(before, event,
                         phase_at_least(before.active.phase,
                                        GenerationPhase::Committed)
                             ? Disposition::GenerationClosed
                             : Disposition::Deferred);
    AuthorityState after = before;
    if (after.active.owner_reassignments
        == kMaximumOwnerReassignments) {
        after.active.phase = GenerationPhase::Aborted;
        return finish(before, std::move(after), event,
                      Disposition::RetryNextGeneration, {
            effect(EffectKind::RetryNextGeneration, event.generation + 1),
        });
    }
    ++after.active.owner_reassignments;
    ++after.active.owner_epoch;
    after.active.owner_loss_receipts.push_back(OwnerLossReceipt{
        event.node, event.incarnation, event.sequence});
    after.active.result_receipts.clear();
    return finish(before, std::move(after), event, Disposition::Accepted, {
        effect(EffectKind::ReassignOwner, event.generation, event.node),
    });
}

Transition query_commit(const AuthorityState& before, const Event& event) {
    if (event.generation < before.committed_generation)
        return unchanged(before, event, Disposition::StaleGeneration);
    if (event.generation == before.committed_generation)
        return unchanged(before, event, Disposition::Accepted);
    if (event.generation == before.committed_generation + 1)
        return unchanged(before, event, Disposition::Deferred);
    return unchanged(before, event, Disposition::StaleGeneration);
}

std::string canonical_trace(const AuthorityState& before,
                            const AuthorityState& after,
                            const Event& event,
                            Disposition disposition,
                            const std::vector<Effect>& effects,
                            const Digest& pre_digest,
                            const Digest& post_digest,
                            bool valid_after) {
    std::ostringstream output;
    output << "{\"schema\":\"emender-native-coordination-trace-v1\","
           << "\"kernel\":\"emender-native-coordination-kernel-v1\","
           << "\"pre_state_digest\":\"";
    append_hex(output, pre_digest);
    output << "\",\"event\":{\"kind\":\"" << event_name(event.kind)
           << "\",\"run\":\"";
    append_hex(output, event.run);
    output << "\",\"fence\":" << event.fence
           << ",\"generation\":" << event.generation
           << ",\"attempt\":" << event.attempt
           << ",\"node\":\"";
    append_hex(output, event.node);
    output << "\",\"incarnation\":\"";
    append_hex(output, event.incarnation);
    output << "\",\"sequence\":" << event.sequence
           << ",\"exact_tokens\":" << event.exact_tokens
           << ",\"trainer_count\":" << event.trainer_count
           << ",\"minimum_nodes\":" << event.minimum_nodes
           << ",\"minimum_tokens\":" << event.minimum_tokens
           << ",\"flags\":" << event.flags
           << ",\"policy_digest\":\"";
    append_hex(output, event.policy_digest);
    output << "\",\"payload_digest\":\"";
    append_hex(output, event.payload_digest);
    output << "\",\"result_digest\":\"";
    append_hex(output, event.result_digest);
    output << "\",\"receipt_digest\":\"";
    append_hex(output, event.receipt_digest);
    output << "\",\"previous_receipt_digest\":\"";
    append_hex(output, event.previous_receipt_digest);
    output << "\",\"manifest_digest\":\"";
    append_hex(output, event.manifest_digest);
    output << "\""
           << "},\"disposition\":\"" << disposition_name(disposition)
           << "\",\"effects\":[";
    for (std::size_t index = 0; index != effects.size(); ++index) {
        if (index != 0) output << ',';
        output << "{\"kind\":\"" << effect_name(effects[index].kind)
               << "\",\"generation\":" << effects[index].generation
               << ",\"node\":\"";
        append_hex(output, effects[index].node);
        output << "\",\"digest\":\"";
        append_hex(output, effects[index].digest);
        output << "\"}";
    }
    output << "],\"authority\":{\"fence\":" << after.fence
           << ",\"committed_generation\":" << after.committed_generation
           << ",\"accepted_token_clock\":" << after.accepted_token_clock
           << ",\"active_generation\":" << after.active.generation
           << ",\"active_attempt\":" << after.active.attempt
           << ",\"generation_phase\":\""
           << phase_name(after.active.phase)
           << "\",\"owner_epoch\":" << after.active.owner_epoch
           << ",\"owner_reassignments\":"
           << after.active.owner_reassignments
           << ",\"cohort_count\":" << after.active.cohort.size()
           << ",\"contribution_count\":"
           << after.active.contributions.size()
           << ",\"result_receipt_count\":"
           << after.active.result_receipts.size()
           << ",\"member_count\":" << after.members.size()
           << ",\"commit_receipt\":\"";
    append_hex(output, after.commit_receipt);
    output << "\",\"commit_manifest\":\"";
    append_hex(output, after.commit_manifest);
    output << "\",\"result_digest\":\"";
    append_hex(output, after.committed_result);
    output << "\"},\"post_state_digest\":\"";
    append_hex(output, post_digest);
    output << "\",\"invariant\":\""
           << (valid_after ? "ok" : "fatal") << "\"}";
    (void)before;
    return output.str();
}

Transition finish(const AuthorityState& before, AuthorityState after,
                  const Event& event, Disposition disposition,
                  std::vector<Effect> effects) {
    Transition result;
    result.pre_state_digest = state_digest(before);
    std::string reason;
    bool valid_after = invariant(after, &reason);
    if (!valid_after) {
        after = before;
        disposition = Disposition::FatalInvariant;
        effects.clear();
    }
    if (effects.size() >= kMaximumEffects) {
        after = before;
        disposition = Disposition::FatalInvariant;
        effects.clear();
        valid_after = false;
    }
    effects.push_back(effect(EffectKind::EmitTrace,
                             after.committed_generation));
    result.state = std::move(after);
    result.disposition = disposition;
    result.effects = std::move(effects);
    result.post_state_digest = state_digest(result.state);
    result.trace = canonical_trace(before, result.state, event,
                                   result.disposition, result.effects,
                                   result.pre_state_digest,
                                   result.post_state_digest, valid_after);
    return result;
}

}  // namespace

bool is_zero(const Key& value) noexcept {
    return std::all_of(value.begin(), value.end(),
                       [](std::uint8_t item) { return item == 0; });
}

bool is_zero(const Digest& value) noexcept {
    return std::all_of(value.begin(), value.end(),
                       [](std::uint8_t item) { return item == 0; });
}

std::string hex(const Key& value) {
    std::ostringstream output;
    append_hex(output, value);
    return output.str();
}

std::string hex(const Digest& value) {
    std::ostringstream output;
    append_hex(output, value);
    return output.str();
}

Digest contribution_receipt(const Event& event) noexcept {
    std::string encoded("emender-native-coordination-receipt-v1");
    encoded.push_back('\0');
    append_bytes(encoded, event.run);
    append_integer(encoded, event.fence);
    append_integer(encoded, event.generation);
    append_integer(encoded, event.attempt);
    append_bytes(encoded, event.node);
    append_bytes(encoded, event.incarnation);
    append_integer(encoded, event.sequence);
    append_integer(encoded, event.exact_tokens);
    append_bytes(encoded, event.payload_digest);
    return digest_string(encoded);
}

bool invariant(const AuthorityState& state, std::string* reason) noexcept {
    const auto fail = [&](const char* value) {
        if (reason) *reason = value;
        return false;
    };
    try {
        if (!state.configured) {
            if (state.fence != 0 || !state.members.empty()
                || state.active.present)
                return fail("unconfigured state carries authority");
            return true;
        }
        if (is_zero(state.run) || state.fence == 0
            || is_zero(state.policy_digest) || state.minimum_nodes == 0
            || state.minimum_nodes > kMaximumNodes
            || state.minimum_tokens == 0
            || state.members.size() > kMaximumNodes
            || state.recovered_node_applies.size() > kMaximumNodes)
            return fail("configured authority identity/bounds are invalid");
        if (state.committed_generation == 0) {
            if (!is_zero(state.commit_receipt)
                || !is_zero(state.commit_manifest)
                || !is_zero(state.committed_result)
                || state.accepted_token_clock != 0)
                return fail("generation-zero authority has a commit");
        } else if (is_zero(state.commit_receipt)
                   || is_zero(state.commit_manifest)
                   || is_zero(state.committed_result)
                   || state.accepted_token_clock == 0) {
            return fail("committed authority lacks immutable digests");
        }
        for (const auto& item : state.recovered_node_applies) {
            if (is_zero(item.first) || is_zero(item.second.incarnation)
                || is_zero(item.second.receipt_digest)
                || state.committed_generation == 0)
                return fail("recovered node apply is invalid");
        }
        for (const auto& item : state.members) {
            if (is_zero(item.first) || is_zero(item.second.incarnation))
                return fail("member identity is incomplete");
            if ((item.second.ready && !item.second.live)
                || (item.second.ready
                && item.second.ready_generation
                    != state.committed_generation))
                return fail("READY member is not synchronized");
            if (item.second.synchronized_generation
                    > state.committed_generation
                || item.second.applied_generation
                    > state.committed_generation
                || ((item.second.applied_generation == 0)
                    != is_zero(item.second.apply_receipt)))
                return fail("node apply authority is invalid");
            if (item.second.recovering
                && (!item.second.live || item.second.ready
                    || item.second.pending_incarnation
                        != item.second.incarnation))
                return fail("recovering member identity is invalid");
            if (!item.second.recovering
                && !is_zero(item.second.pending_incarnation))
                return fail("non-recovering member has a pending identity");
            if (item.second.ready && state.committed_generation != 0
                && (item.second.synchronized_generation
                        != state.committed_generation
                    || item.second.applied_generation
                        != state.committed_generation
                    || is_zero(item.second.apply_receipt)))
                return fail("READY member lacks all-eight apply authority");
        }
        if (!state.active.present) return true;
        const Generation& generation = state.active;
        if (generation.attempt == 0
            || generation.phase == GenerationPhase::None
            || generation.cohort.size() > kMaximumNodes
            || generation.contributions.size() > generation.cohort.size()
            || generation.result_receipts.size()
                > generation.contributions.size()
            || generation.owner_epoch == 0
            || generation.owner_reassignments
                > kMaximumOwnerReassignments
            || generation.owner_loss_receipts.size()
                != generation.owner_reassignments)
            return fail("generation identity/bounds are invalid");
        if (phase_at_least(generation.phase, GenerationPhase::Closed)
            && generation.phase != GenerationPhase::Aborted
            && (generation.contributions.size() < state.minimum_nodes
                || frozen_tokens(generation) < state.minimum_tokens
                || frozen_tokens(generation) == 0))
            return fail("closed generation is below immutable Q/T");
        if (phase_at_least(generation.phase, GenerationPhase::Committed)) {
            if (generation.generation + 1
                    != state.committed_generation
                || generation.result_receipts.size()
                    != generation.contributions.size()
                || generation.result_receipts.empty())
                return fail("committed generation lacks complete results");
        } else if (generation.generation
                   != state.committed_generation) {
            return fail("open/closed generation does not extend commit");
        }
        for (const auto& item : generation.cohort) {
            if (is_zero(item.first) || is_zero(item.second))
                return fail("cohort identity is incomplete");
        }
        std::set<std::tuple<Key, Key, std::uint64_t>> owner_losses;
        for (const auto& item : generation.owner_loss_receipts) {
            const auto cohort = generation.cohort.find(item.node);
            if (is_zero(item.node) || is_zero(item.incarnation)
                || item.sequence == 0
                || cohort == generation.cohort.end()
                || cohort->second != item.incarnation
                || !owner_losses.emplace(
                    item.node, item.incarnation, item.sequence).second)
                return fail("owner-loss receipt is invalid");
        }
        for (const auto& item : generation.contributions) {
            const auto cohort = generation.cohort.find(item.first);
            if (cohort == generation.cohort.end()
                || cohort->second != item.second.incarnation
                || item.second.sequence == 0
                || item.second.exact_tokens == 0
                || is_zero(item.second.payload_digest)
                || is_zero(item.second.receipt_digest))
                return fail("admitted contribution is invalid");
        }
        if (!generation.contributions.empty()
            && frozen_tokens(generation) == 0)
            return fail("contribution token total overflowed");
        Digest result{};
        for (const auto& item : generation.result_receipts) {
            const auto contribution =
                generation.contributions.find(item.first);
            if (contribution == generation.contributions.end()
                || contribution->second.incarnation
                    != item.second.incarnation
                || item.second.sequence == 0
                || item.second.exact_tokens != frozen_tokens(generation)
                || is_zero(item.second.result_digest))
                return fail("result receipt is invalid");
            if (is_zero(result)) result = item.second.result_digest;
            else if (result != item.second.result_digest)
                return fail("result receipts disagree");
        }
        return true;
    } catch (...) {
        return fail("invariant evaluation failed");
    }
}

Digest state_digest(const AuthorityState& state) noexcept {
    try {
        std::string encoded("emender-native-coordination-state-v1");
        encoded.push_back('\0');
        append_integer(encoded, static_cast<std::uint8_t>(state.configured));
        append_bytes(encoded, state.run);
        append_integer(encoded, state.fence);
        append_bytes(encoded, state.policy_digest);
        append_integer(encoded, state.minimum_nodes);
        append_integer(encoded, state.minimum_tokens);
        append_integer(encoded, state.committed_generation);
        append_integer(encoded, state.accepted_token_clock);
        append_bytes(encoded, state.commit_receipt);
        append_bytes(encoded, state.commit_manifest);
        append_bytes(encoded, state.committed_result);
        append_integer(encoded, static_cast<std::uint64_t>(
            state.recovered_node_applies.size()));
        for (const auto& item : state.recovered_node_applies) {
            append_bytes(encoded, item.first);
            append_bytes(encoded, item.second.incarnation);
            append_bytes(encoded, item.second.receipt_digest);
        }
        append_integer(encoded, static_cast<std::uint64_t>(state.members.size()));
        for (const auto& item : state.members) {
            append_bytes(encoded, item.first);
            append_bytes(encoded, item.second.incarnation);
            append_bytes(encoded, item.second.pending_incarnation);
            append_integer(encoded, item.second.control_sequence);
            append_integer(encoded, item.second.ready_generation);
            append_integer(encoded, item.second.synchronized_generation);
            append_integer(encoded, item.second.applied_generation);
            append_integer(encoded, static_cast<std::uint8_t>(
                item.second.live));
            append_integer(encoded, static_cast<std::uint8_t>(
                item.second.ready));
            append_integer(encoded, static_cast<std::uint8_t>(
                item.second.recovering));
            append_bytes(encoded, item.second.apply_receipt);
        }
        append_integer(encoded, static_cast<std::uint8_t>(
            state.active.present));
        append_integer(encoded, state.active.generation);
        append_integer(encoded, state.active.attempt);
        append_integer(encoded, static_cast<std::uint32_t>(
            state.active.phase));
        append_integer(encoded, state.active.owner_epoch);
        append_integer(encoded, state.active.owner_reassignments);
        append_integer(encoded, static_cast<std::uint64_t>(
            state.active.owner_loss_receipts.size()));
        for (const auto& item : state.active.owner_loss_receipts) {
            append_bytes(encoded, item.node);
            append_bytes(encoded, item.incarnation);
            append_integer(encoded, item.sequence);
        }
        append_integer(encoded, static_cast<std::uint64_t>(
            state.active.cohort.size()));
        for (const auto& item : state.active.cohort) {
            append_bytes(encoded, item.first);
            append_bytes(encoded, item.second);
        }
        append_integer(encoded, static_cast<std::uint64_t>(
            state.active.contributions.size()));
        for (const auto& item : state.active.contributions) {
            append_bytes(encoded, item.first);
            append_bytes(encoded, item.second.incarnation);
            append_integer(encoded, item.second.sequence);
            append_integer(encoded, item.second.exact_tokens);
            append_bytes(encoded, item.second.payload_digest);
            append_bytes(encoded, item.second.receipt_digest);
        }
        append_integer(encoded, static_cast<std::uint64_t>(
            state.active.result_receipts.size()));
        for (const auto& item : state.active.result_receipts) {
            append_bytes(encoded, item.first);
            append_bytes(encoded, item.second.incarnation);
            append_integer(encoded, item.second.sequence);
            append_integer(encoded, item.second.exact_tokens);
            append_bytes(encoded, item.second.result_digest);
        }
        return digest_string(encoded);
    } catch (...) {
        return {};
    }
}

Transition step(const AuthorityState& state, const Event& event) noexcept {
    try {
        std::string reason;
        if (!invariant(state, &reason))
            return unchanged(state, event, Disposition::FatalInvariant);
        if (event.kind == EventKind::RecoverAuthority)
            return recover_authority(state, event);
        if (!state.configured)
            return unchanged(state, event, Disposition::Deferred);
        if (!matching_event_identity(state, event))
            return unchanged(state, event, Disposition::StaleFence);
        if (event.policy_digest != state.policy_digest
            && !is_zero(event.policy_digest))
            return unchanged(state, event, Disposition::Corrupt);
        switch (event.kind) {
            case EventKind::RecoverAuthority:
                break;
            case EventKind::RecoverNodeApply:
                return recover_node_apply(state, event);
            case EventKind::RecoverPeer:
                return recover_peer(state, event);
            case EventKind::Ready:
                return ready(state, event);
            case EventKind::OpenGeneration:
                return open_generation(state, event);
            case EventKind::Contribution:
                return contribution(state, event);
            case EventKind::CloseGeneration:
                return close_generation(state, event);
            case EventKind::ResultReceipt:
                return result_receipt(state, event);
            case EventKind::Commit:
                return commit(state, event);
            case EventKind::NodeApply:
                return node_apply(state, event);
            case EventKind::ExpirePeer:
                return expire_peer(state, event);
            case EventKind::OwnerLost:
                return owner_lost(state, event);
            case EventKind::QueryCommit:
                return query_commit(state, event);
        }
        return unchanged(state, event, Disposition::InvalidEvent);
    } catch (...) {
        return unchanged(state, event, Disposition::FatalInvariant);
    }
}

const char* event_name(EventKind value) noexcept {
    switch (value) {
        case EventKind::RecoverAuthority: return "recover-authority";
        case EventKind::RecoverNodeApply: return "recover-node-apply";
        case EventKind::RecoverPeer: return "recover-peer";
        case EventKind::Ready: return "ready";
        case EventKind::OpenGeneration: return "open-generation";
        case EventKind::Contribution: return "contribution";
        case EventKind::CloseGeneration: return "close-generation";
        case EventKind::ResultReceipt: return "result-receipt";
        case EventKind::Commit: return "commit";
        case EventKind::NodeApply: return "node-apply";
        case EventKind::ExpirePeer: return "expire-peer";
        case EventKind::OwnerLost: return "owner-lost";
        case EventKind::QueryCommit: return "query-commit";
    }
    return "invalid-event";
}

const char* disposition_name(Disposition value) noexcept {
    switch (value) {
        case Disposition::Accepted: return "accepted";
        case Disposition::IdenticalDuplicate:
            return "identical-duplicate";
        case Disposition::ConflictingDuplicate:
            return "conflicting-duplicate";
        case Disposition::StaleFence: return "stale-fence";
        case Disposition::StaleIncarnation: return "stale-incarnation";
        case Disposition::StaleGeneration: return "stale-generation";
        case Disposition::GenerationClosed: return "generation-closed";
        case Disposition::Deferred: return "deferred";
        case Disposition::RetryNextGeneration:
            return "retry-next-generation";
        case Disposition::InsufficientCohort:
            return "insufficient-cohort";
        case Disposition::Corrupt: return "corrupt";
        case Disposition::InvalidEvent: return "invalid-event";
        case Disposition::FatalInvariant: return "fatal-invariant";
    }
    return "invalid-disposition";
}

const char* effect_name(EffectKind value) noexcept {
    switch (value) {
        case EffectKind::BindFence: return "bind-fence";
        case EffectKind::SyncAuthority: return "sync-authority";
        case EffectKind::AdvertiseReady: return "advertise-ready";
        case EffectKind::StartGeneration: return "start-generation";
        case EffectKind::AcknowledgeReceipt: return "acknowledge-receipt";
        case EffectKind::FreezeCohort: return "freeze-cohort";
        case EffectKind::ReassignOwner: return "reassign-owner";
        case EffectKind::CommitEligible: return "commit-eligible";
        case EffectKind::PublishCommit: return "publish-commit";
        case EffectKind::RecordNodeApply: return "record-node-apply";
        case EffectKind::ExpirePeer: return "expire-peer";
        case EffectKind::RetryNextGeneration:
            return "retry-next-generation";
        case EffectKind::EmitTrace: return "emit-trace";
    }
    return "invalid-effect";
}

const char* phase_name(GenerationPhase value) noexcept {
    switch (value) {
        case GenerationPhase::None: return "none";
        case GenerationPhase::Open: return "open";
        case GenerationPhase::Closed: return "closed";
        case GenerationPhase::Aborted: return "aborted";
        case GenerationPhase::Committed: return "committed";
        case GenerationPhase::Applied: return "applied";
    }
    return "invalid-phase";
}

}  // namespace emender_ndp::coordination
