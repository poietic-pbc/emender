#ifndef EMENDER_NDP_COORDINATION_KERNEL_HPP
#define EMENDER_NDP_COORDINATION_KERNEL_HPP

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace emender_ndp::coordination {

/*
 * Production coordination is deliberately model-free.  These identities are
 * opaque hashes at this boundary; networking, timers, storage, process
 * supervision, and dense-buffer execution remain outside this module.
 */
using Key = std::array<std::uint8_t, 16>;
using Digest = std::array<std::uint8_t, 32>;

constexpr std::size_t kMaximumNodes = 256;
constexpr std::size_t kMaximumEffects = 8;
constexpr std::size_t kMaximumTraceBytes = 4096;
constexpr std::uint32_t kMaximumOwnerReassignments = 2;
constexpr std::uint32_t kRequiredTrainerReceipts = 8;

enum class EventKind : std::uint32_t {
    RecoverAuthority = 1,
    RecoverNodeApply = 2,
    RecoverPeer = 3,
    Ready = 4,
    OpenGeneration = 5,
    Contribution = 6,
    CloseGeneration = 7,
    ResultReceipt = 8,
    Commit = 9,
    NodeApply = 10,
    ExpirePeer = 11,
    OwnerLost = 12,
    QueryCommit = 13,
};

enum class Disposition : std::uint32_t {
    Accepted = 1,
    IdenticalDuplicate = 2,
    ConflictingDuplicate = 3,
    StaleFence = 4,
    StaleIncarnation = 5,
    StaleGeneration = 6,
    GenerationClosed = 7,
    Deferred = 8,
    RetryNextGeneration = 9,
    InsufficientCohort = 10,
    Corrupt = 11,
    InvalidEvent = 12,
    FatalInvariant = 13,
};

enum class EffectKind : std::uint32_t {
    BindFence = 1,
    SyncAuthority = 2,
    AdvertiseReady = 3,
    StartGeneration = 4,
    AcknowledgeReceipt = 5,
    FreezeCohort = 6,
    ReassignOwner = 7,
    CommitEligible = 8,
    PublishCommit = 9,
    RecordNodeApply = 10,
    ExpirePeer = 11,
    RetryNextGeneration = 12,
    EmitTrace = 13,
};

enum class GenerationPhase : std::uint32_t {
    None = 0,
    Open = 1,
    Closed = 2,
    Aborted = 3,
    Committed = 4,
    Applied = 5,
};

enum EventFlags : std::uint32_t {
    EventFlagNone = 0,
    EventFlagFiniteClose = 1U << 0,
    EventFlagDeadlineExpired = 1U << 1,
};

struct Event {
    EventKind kind = EventKind::QueryCommit;
    std::uint32_t flags = EventFlagNone;
    Key run{};
    std::uint64_t fence = 0;
    std::uint64_t generation = 0;
    std::uint32_t attempt = 0;
    Key node{};
    Key incarnation{};
    std::uint64_t sequence = 0;
    std::uint64_t exact_tokens = 0;
    std::uint32_t trainer_count = 0;
    std::uint32_t minimum_nodes = 0;
    std::uint64_t minimum_tokens = 0;
    Digest policy_digest{};
    Digest payload_digest{};
    Digest result_digest{};
    Digest receipt_digest{};
    Digest previous_receipt_digest{};
    Digest manifest_digest{};
};

struct Effect {
    EffectKind kind = EffectKind::EmitTrace;
    std::uint64_t generation = 0;
    Key node{};
    Digest digest{};
};

struct Member {
    Key incarnation{};
    Key pending_incarnation{};
    std::uint64_t control_sequence = 0;
    std::uint64_t ready_generation = 0;
    std::uint64_t synchronized_generation = 0;
    std::uint64_t applied_generation = 0;
    bool live = false;
    bool ready = false;
    bool recovering = false;
    Digest apply_receipt{};
};

struct Contribution {
    Key incarnation{};
    std::uint64_t sequence = 0;
    std::uint64_t exact_tokens = 0;
    Digest payload_digest{};
    Digest receipt_digest{};
};

struct ResultReceipt {
    Key incarnation{};
    std::uint64_t sequence = 0;
    std::uint64_t exact_tokens = 0;
    Digest result_digest{};
};

struct RecoveredNodeApply {
    Key incarnation{};
    Digest receipt_digest{};
};

struct OwnerLossReceipt {
    Key node{};
    Key incarnation{};
    std::uint64_t sequence = 0;
};

struct Generation {
    bool present = false;
    std::uint64_t generation = 0;
    std::uint32_t attempt = 0;
    GenerationPhase phase = GenerationPhase::None;
    std::uint64_t owner_epoch = 0;
    std::uint32_t owner_reassignments = 0;
    std::vector<OwnerLossReceipt> owner_loss_receipts;
    std::map<Key, Key> cohort;
    std::map<Key, Contribution> contributions;
    std::map<Key, ResultReceipt> result_receipts;
};

struct AuthorityState {
    bool configured = false;
    Key run{};
    std::uint64_t fence = 0;
    Digest policy_digest{};
    std::uint32_t minimum_nodes = 0;
    std::uint64_t minimum_tokens = 0;
    std::uint64_t committed_generation = 0;
    std::uint64_t accepted_token_clock = 0;
    Digest commit_receipt{};
    Digest commit_manifest{};
    Digest committed_result{};
    std::map<Key, RecoveredNodeApply> recovered_node_applies;
    std::map<Key, Member> members;
    Generation active;
};

struct Transition {
    AuthorityState state;
    Disposition disposition = Disposition::InvalidEvent;
    std::vector<Effect> effects;
    std::string trace;
    Digest pre_state_digest{};
    Digest post_state_digest{};
};

/*
 * The sole production decision function.  It is total: malformed or expected
 * race inputs return typed dispositions, and a corrupt authoritative input
 * state returns FatalInvariant while preserving the supplied state.
 */
Transition step(const AuthorityState& state, const Event& event) noexcept;

bool invariant(const AuthorityState& state, std::string* reason = nullptr) noexcept;
Digest state_digest(const AuthorityState& state) noexcept;
Digest contribution_receipt(const Event& event) noexcept;

const char* event_name(EventKind value) noexcept;
const char* disposition_name(Disposition value) noexcept;
const char* effect_name(EffectKind value) noexcept;
const char* phase_name(GenerationPhase value) noexcept;

bool is_zero(const Key& value) noexcept;
bool is_zero(const Digest& value) noexcept;
std::string hex(const Key& value);
std::string hex(const Digest& value);

}  // namespace emender_ndp::coordination

#endif
