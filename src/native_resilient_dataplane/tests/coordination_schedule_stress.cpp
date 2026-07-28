#include "coordination_kernel.hpp"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "emender/ndp.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "sha256.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace coordination = emender_ndp::coordination;
namespace fs = std::filesystem;

namespace {

constexpr char kManifestSchema[] =
    "emender-native-coordination-stress-manifest-v1";
constexpr char kScheduleSchema[] =
    "emender-native-coordination-schedule-v1";
constexpr char kGeneratorSchema[] =
    "emender-native-coordination-generator-v1";
constexpr char kPrngSchema[] = "pcg-xsh-rr-64-32-v1";
constexpr char kShrinkSchema[] =
    "causal-ddmin-suffix-chunk-single-scalar-v1";
constexpr char kKernelSchema[] = "emender-native-coordination-kernel-v1";
constexpr char kTraceSchema[] = "emender-native-coordination-trace-v1";
constexpr char kHardenCommit[] =
    "e3243158b27ce6f54a4b6543199e40b87be691f8";
constexpr char kHardenManifestRelative[] =
    "docs/validation/harden-native-coordination-kernel-20260728.md";
constexpr char kHardenManifestSha256[] =
    "ba5fb5052b7aa10840613d69b91f280e64ffa2a1a2f6c33beb0f4aa4d53a1f57";
constexpr std::uint64_t kJob5105811 = UINT64_C(5105811);
constexpr std::size_t kDefaultMaximumEvents = 32;
constexpr std::size_t kDefaultRandomSchedules = 50000;

constexpr char kSchemaAuthority[] = R"SCHEMA(
manifest=emender-native-coordination-stress-manifest-v1
schedule=emender-native-coordination-schedule-v1
generator=emender-native-coordination-generator-v1
kernel=emender-native-coordination-kernel-v1
trace=emender-native-coordination-trace-v1
prng=pcg-xsh-rr-64-32-v1
prng-transition=state:=state*6364136223846793005+(stream|1) mod 2^64
prng-output=xorshifted:=uint32(((old>>18)^old)>>27);rot:=old>>59;rotr32(xorshifted,rot)
prng-seeding=state=0;step;state+=seed mod 2^64;step
grammar=authority (recover-peer ready|delay|expire){2,4} open generation-event{0,32}
generation-event=ready|expire|contribution|duplicate|conflict|drop|finite-close|deadline-close|result-receipt|owner-loss|commit|node-apply|query|restart
causal=authority precedes all; peer identity precedes ready/expire/apply; open snapshots prior ready identities; contribution references a prior open and previously recovered peer identity while the kernel decides immutable-cohort admissibility; close/owner-loss reference a prior open; result references a planned contribution; commit references a planned contribution/result identity; restart recovers only the prior durable commit
bounds=nodes[2,4];random-events<=32;one-active-generation;members<=256;effects<=8;trace-bytes<4096;owner-reassignments<=2;exact-token-integers-only
shrink=remove failing suffix;ddmin largest contiguous chunks left-to-right;remove singleton newest-to-oldest;simplify flags,sequence,tokens,node scalars
shrink-accept=causal-well-formed && same-original-predicate
)SCHEMA";

struct Options {
    fs::path source_root;
    fs::path corpus_dir;
    fs::path output;
    fs::path failure_dir;
    std::string source_commit = "working-tree";
    std::size_t random_schedules = kDefaultRandomSchedules;
    std::size_t maximum_events = kDefaultMaximumEvents;
    unsigned int determinism_repeats = 2;
    bool output_enabled = true;
    bool replay_only = false;
    std::uint64_t replay_seed = kJob5105811;
    std::size_t replay_index = 0;
    fs::path replay_file;
};

struct Pcg32 {
    std::uint64_t state = 0;
    std::uint64_t increment = 0;

    Pcg32(std::uint64_t seed, std::uint64_t stream) {
        increment = (stream << 1U) | UINT64_C(1);
        (void)next();
        state += seed;
        (void)next();
    }

    std::uint32_t next() noexcept {
        const std::uint64_t old = state;
        state = old * UINT64_C(6364136223846793005) + increment;
        const auto shifted =
            static_cast<std::uint32_t>(((old >> 18U) ^ old) >> 27U);
        const auto rotation = static_cast<std::uint32_t>(old >> 59U);
        return static_cast<std::uint32_t>(
            (shifted >> rotation)
            | (shifted << ((0U - rotation) & 31U)));
    }

    std::uint64_t next64() noexcept {
        return (static_cast<std::uint64_t>(next()) << 32U) | next();
    }

    std::size_t bounded(std::size_t bound) {
        if (bound == 0) throw std::invalid_argument("zero PRNG bound");
        const std::uint32_t value_bound =
            static_cast<std::uint32_t>(bound);
        const std::uint32_t threshold =
            static_cast<std::uint32_t>(0U - value_bound) % value_bound;
        for (;;) {
            const std::uint32_t value = next();
            if (value >= threshold)
                return static_cast<std::size_t>(value % value_bound);
        }
    }
};

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
    std::vector<std::string> result;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = value.find(delimiter, begin);
        result.push_back(value.substr(
            begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) return result;
        begin = end + 1;
    }
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (byte < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<unsigned int>(byte)
                           << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(byte);
                }
        }
    }
    return output.str();
}

template <typename Bytes>
Bytes parse_hex(const std::string& value) {
    Bytes result{};
    if (value.size() != result.size() * 2U)
        throw std::runtime_error("invalid fixed hexadecimal width");
    const auto nibble = [](char item) -> std::uint8_t {
        if (item >= '0' && item <= '9')
            return static_cast<std::uint8_t>(item - '0');
        if (item >= 'a' && item <= 'f')
            return static_cast<std::uint8_t>(item - 'a' + 10);
        if (item >= 'A' && item <= 'F')
            return static_cast<std::uint8_t>(item - 'A' + 10);
        throw std::runtime_error("invalid hexadecimal digit");
    };
    for (std::size_t index = 0; index != result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(
            (nibble(value[index * 2U]) << 4U)
            | nibble(value[index * 2U + 1U]));
    }
    return result;
}

coordination::Digest hash_bytes(const std::string& value) {
    return emender_ndp::Sha256::digest(value.data(), value.size());
}

coordination::Digest hash_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open digest input: " + path.string());
    emender_ndp::Sha256 hash;
    std::array<char, 64 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0)
            hash.update(buffer.data(), static_cast<std::size_t>(count));
    }
    if (!input.eof())
        throw std::runtime_error("cannot read digest input: " + path.string());
    return hash.finish();
}

coordination::Digest hash_bundle(
        const fs::path& root, const std::vector<fs::path>& relative_paths) {
    emender_ndp::Sha256 hash;
    const std::string domain =
        "emender-native-coordination-source-bundle-v1";
    hash.update(domain.data(), domain.size());
    std::vector<fs::path> ordered = relative_paths;
    std::sort(ordered.begin(), ordered.end());
    for (const fs::path& relative : ordered) {
        const std::string name = relative.generic_string();
        const auto file_digest = hash_file(root / relative);
        const char zero = '\0';
        hash.update(name.data(), name.size());
        hash.update(&zero, 1);
        hash.update(file_digest.data(), file_digest.size());
    }
    return hash.finish();
}

coordination::Digest hash_corpus(const fs::path& corpus_dir) {
    std::vector<fs::path> files;
    for (const fs::directory_entry& entry :
         fs::directory_iterator(corpus_dir)) {
        if (entry.is_regular_file()
            && entry.path().extension() == ".schedule")
            files.push_back(entry.path().filename());
    }
    if (files.empty())
        throw std::runtime_error("native coordination corpus is empty");
    return hash_bundle(corpus_dir, files);
}

coordination::Key opaque_key(const std::string& value) {
    const coordination::Digest full = hash_bytes("key:" + value);
    coordination::Key result{};
    std::copy_n(full.begin(), result.size(), result.begin());
    return result;
}

coordination::Digest opaque_digest(const std::string& value) {
    return hash_bytes("digest:" + value);
}

std::uint64_t checked_add(
        std::uint64_t left, std::uint64_t right, const char* field) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::runtime_error(std::string(field) + " overflow");
    return left + right;
}

std::uint64_t contribution_tokens(
        const coordination::AuthorityState& state) {
    std::uint64_t total = 0;
    for (const auto& item : state.active.contributions)
        total = checked_add(total, item.second.exact_tokens,
                            "contribution token sum");
    return total;
}

struct Step {
    enum class Kind {
        NativeEvent,
        Restart,
    };

    Kind kind = Kind::NativeEvent;
    std::string label;
    coordination::Event event;
    std::string restart_role;
    bool bump_fence = false;
};

struct Schedule {
    std::string name;
    std::string expected = "pass";
    std::vector<Step> steps;
};

Step native_step(std::string label, const coordination::Event& event) {
    Step result;
    result.kind = Step::Kind::NativeEvent;
    result.label = std::move(label);
    result.event = event;
    return result;
}

Step restart_step(std::string label, std::string role, bool bump_fence) {
    Step result;
    result.kind = Step::Kind::Restart;
    result.label = std::move(label);
    result.restart_role = std::move(role);
    result.bump_fence = bump_fence;
    return result;
}

coordination::Event recover_event_for_restart(
        const coordination::AuthorityState& prior, bool bump_fence) {
    coordination::Event event;
    event.kind = coordination::EventKind::RecoverAuthority;
    event.run = prior.run;
    event.fence = prior.fence + static_cast<std::uint64_t>(bump_fence);
    event.generation = prior.committed_generation;
    event.exact_tokens = prior.accepted_token_clock;
    event.minimum_nodes = prior.minimum_nodes;
    event.minimum_tokens = prior.minimum_tokens;
    event.policy_digest = prior.policy_digest;
    event.receipt_digest = prior.commit_receipt;
    event.previous_receipt_digest = prior.commit_receipt;
    event.manifest_digest = prior.commit_manifest;
    event.result_digest = prior.committed_result;
    return event;
}

struct AppliedStep {
    coordination::Transition transition;
    coordination::Event event;
    bool restart_boundary = false;
    std::string restart_role;
};

AppliedStep apply_step(
        const coordination::AuthorityState& prior, const Step& step) {
    if (step.kind == Step::Kind::NativeEvent) {
        return AppliedStep{
            coordination::step(prior, step.event), step.event, false, {}};
    }
    if (!prior.configured)
        throw std::runtime_error("restart before authority recovery");
    const coordination::Event recovery =
        recover_event_for_restart(prior, step.bump_fence);
    coordination::AuthorityState empty;
    return AppliedStep{
        coordination::step(empty, recovery), recovery, true,
        step.restart_role};
}

void write_schedule(const Schedule& schedule, const fs::path& path) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write schedule: " + path.string());
    output << "schema|" << kScheduleSchema << "\n"
           << "name|" << schedule.name << "\n"
           << "expected|" << schedule.expected << "\n";
    for (const Step& step : schedule.steps) {
        if (step.kind == Step::Kind::Restart) {
            output << "restart|" << step.label << '|'
                   << step.restart_role << '|'
                   << (step.bump_fence ? 1 : 0) << "\n";
            continue;
        }
        const coordination::Event& event = step.event;
        output << "event|" << step.label << '|'
               << static_cast<std::uint32_t>(event.kind) << '|'
               << event.flags << '|' << coordination::hex(event.run) << '|'
               << event.fence << '|' << event.generation << '|'
               << event.attempt << '|' << coordination::hex(event.node) << '|'
               << coordination::hex(event.incarnation) << '|'
               << event.sequence << '|' << event.exact_tokens << '|'
               << event.trainer_count << '|' << event.minimum_nodes << '|'
               << event.minimum_tokens << '|'
               << coordination::hex(event.policy_digest) << '|'
               << coordination::hex(event.payload_digest) << '|'
               << coordination::hex(event.result_digest) << '|'
               << coordination::hex(event.receipt_digest) << '|'
               << coordination::hex(event.previous_receipt_digest) << '|'
               << coordination::hex(event.manifest_digest) << "\n";
    }
    if (!output)
        throw std::runtime_error("cannot finish schedule: " + path.string());
}

Schedule built_in_fixture(const std::string& fixture);

Schedule read_schedule(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read schedule: " + path.string());
    Schedule result;
    std::string schema;
    std::string fixture;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::vector<std::string> fields = split(line, '|');
        if (fields.empty()) continue;
        if (fields[0] == "schema" && fields.size() == 2) {
            schema = fields[1];
        } else if (fields[0] == "name" && fields.size() == 2) {
            result.name = fields[1];
        } else if (fields[0] == "expected" && fields.size() == 2) {
            result.expected = fields[1];
        } else if (fields[0] == "fixture" && fields.size() == 2) {
            fixture = fields[1];
        } else if (fields[0] == "restart" && fields.size() == 4) {
            result.steps.push_back(restart_step(
                fields[1], fields[2], std::stoul(fields[3]) != 0));
        } else if (fields[0] == "event" && fields.size() == 21) {
            coordination::Event event;
            event.kind = static_cast<coordination::EventKind>(
                std::stoul(fields[2]));
            event.flags = static_cast<std::uint32_t>(
                std::stoul(fields[3]));
            event.run = parse_hex<coordination::Key>(fields[4]);
            event.fence = std::stoull(fields[5]);
            event.generation = std::stoull(fields[6]);
            event.attempt = static_cast<std::uint32_t>(
                std::stoul(fields[7]));
            event.node = parse_hex<coordination::Key>(fields[8]);
            event.incarnation =
                parse_hex<coordination::Key>(fields[9]);
            event.sequence = std::stoull(fields[10]);
            event.exact_tokens = std::stoull(fields[11]);
            event.trainer_count = static_cast<std::uint32_t>(
                std::stoul(fields[12]));
            event.minimum_nodes = static_cast<std::uint32_t>(
                std::stoul(fields[13]));
            event.minimum_tokens = std::stoull(fields[14]);
            event.policy_digest =
                parse_hex<coordination::Digest>(fields[15]);
            event.payload_digest =
                parse_hex<coordination::Digest>(fields[16]);
            event.result_digest =
                parse_hex<coordination::Digest>(fields[17]);
            event.receipt_digest =
                parse_hex<coordination::Digest>(fields[18]);
            event.previous_receipt_digest =
                parse_hex<coordination::Digest>(fields[19]);
            event.manifest_digest =
                parse_hex<coordination::Digest>(fields[20]);
            result.steps.push_back(native_step(fields[1], event));
        } else {
            throw std::runtime_error(
                "invalid schedule line in " + path.string() + ": " + line);
        }
    }
    if (schema != kScheduleSchema)
        throw std::runtime_error("unsupported schedule schema: " + schema);
    if (!fixture.empty()) {
        Schedule expanded = built_in_fixture(fixture);
        if (!result.name.empty()) expanded.name = result.name;
        if (!result.expected.empty()) expanded.expected = result.expected;
        result = std::move(expanded);
    }
    if (result.name.empty() || result.steps.empty())
        throw std::runtime_error("schedule lacks name/events: " + path.string());
    return result;
}

std::string canonical_schedule(const Schedule& schedule) {
    std::ostringstream output;
    output << "schema|" << kScheduleSchema << "\n"
           << "name|" << schedule.name << "\n"
           << "expected|" << schedule.expected << "\n";
    for (const Step& step : schedule.steps) {
        if (step.kind == Step::Kind::Restart) {
            output << "restart|" << step.label << '|'
                   << step.restart_role << '|'
                   << (step.bump_fence ? 1 : 0) << "\n";
            continue;
        }
        const coordination::Event& event = step.event;
        output << "event|" << step.label << '|'
               << static_cast<std::uint32_t>(event.kind) << '|'
               << event.flags << '|' << coordination::hex(event.run) << '|'
               << event.fence << '|' << event.generation << '|'
               << event.attempt << '|' << coordination::hex(event.node) << '|'
               << coordination::hex(event.incarnation) << '|'
               << event.sequence << '|' << event.exact_tokens << '|'
               << event.trainer_count << '|' << event.minimum_nodes << '|'
               << event.minimum_tokens << '|'
               << coordination::hex(event.policy_digest) << '|'
               << coordination::hex(event.payload_digest) << '|'
               << coordination::hex(event.result_digest) << '|'
               << coordination::hex(event.receipt_digest) << '|'
               << coordination::hex(event.previous_receipt_digest) << '|'
               << coordination::hex(event.manifest_digest) << "\n";
    }
    return output.str();
}

coordination::Key node_key(std::size_t index) {
    return opaque_key("logical-node-" + std::to_string(index));
}

coordination::Key incarnation_key(
        std::size_t index, std::uint64_t version) {
    return opaque_key(
        "logical-node-" + std::to_string(index)
        + "-incarnation-" + std::to_string(version));
}

coordination::Event configured_event(
        std::uint64_t generation = 0, std::uint64_t tokens = 0,
        std::uint64_t fence = kJob5105811,
        std::uint32_t minimum_nodes = 2,
        std::uint64_t minimum_tokens = 2) {
    coordination::Event event;
    event.kind = coordination::EventKind::RecoverAuthority;
    event.run = opaque_key("stress-native-coordinator-schedules");
    event.fence = fence;
    event.generation = generation;
    event.exact_tokens = tokens;
    event.minimum_nodes = minimum_nodes;
    event.minimum_tokens = minimum_tokens;
    event.policy_digest = opaque_digest(
        "async-decoupled-v2.1-simple-stress-policy-v1");
    if (generation != 0) {
        event.receipt_digest =
            opaque_digest("recovered-commit-" + std::to_string(generation));
        event.previous_receipt_digest = event.receipt_digest;
        event.manifest_digest =
            opaque_digest("recovered-manifest-" + std::to_string(generation));
        event.result_digest =
            opaque_digest("recovered-result-" + std::to_string(generation));
    }
    return event;
}

coordination::Event base_event(
        coordination::EventKind kind,
        const coordination::AuthorityState& state) {
    coordination::Event event;
    event.kind = kind;
    event.run = state.run;
    event.fence = state.fence;
    event.policy_digest = state.policy_digest;
    return event;
}

coordination::Event recover_peer_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t incarnation_version, std::uint64_t sequence) {
    coordination::Event event =
        base_event(coordination::EventKind::RecoverPeer, state);
    event.generation = state.committed_generation;
    event.node = node_key(node);
    event.incarnation = incarnation_key(node, incarnation_version);
    event.sequence = sequence;
    event.receipt_digest = state.commit_receipt;
    return event;
}

coordination::Event ready_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t incarnation_version, std::uint64_t sequence,
        coordination::Digest receipt = {}) {
    coordination::Event event =
        base_event(coordination::EventKind::Ready, state);
    event.generation = state.committed_generation;
    event.node = node_key(node);
    event.incarnation = incarnation_key(node, incarnation_version);
    event.sequence = sequence;
    if (coordination::is_zero(receipt)) {
        const auto found = state.members.find(event.node);
        if (found != state.members.end()) receipt = found->second.apply_receipt;
    }
    event.receipt_digest = receipt;
    return event;
}

coordination::Event open_event(
        const coordination::AuthorityState& state,
        std::uint32_t attempt = 1) {
    coordination::Event event =
        base_event(coordination::EventKind::OpenGeneration, state);
    event.generation = state.committed_generation;
    event.attempt = attempt;
    return event;
}

coordination::Event contribution_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t sequence, std::uint64_t tokens,
        const std::string& variant = "primary") {
    coordination::Event event =
        base_event(coordination::EventKind::Contribution, state);
    event.generation = state.active.present
        ? state.active.generation : state.committed_generation;
    event.attempt = state.active.present ? state.active.attempt : 1;
    event.node = node_key(node);
    const auto cohort = state.active.cohort.find(event.node);
    event.incarnation = cohort == state.active.cohort.end()
        ? incarnation_key(node, 1) : cohort->second;
    event.sequence = sequence;
    event.exact_tokens = tokens;
    event.payload_digest = opaque_digest(
        "payload-g" + std::to_string(event.generation)
        + "-n" + std::to_string(node) + "-" + variant);
    return event;
}

coordination::Event close_event(
        const coordination::AuthorityState& state, std::uint32_t flags) {
    coordination::Event event =
        base_event(coordination::EventKind::CloseGeneration, state);
    event.generation = state.active.present
        ? state.active.generation : state.committed_generation;
    event.attempt = state.active.present ? state.active.attempt : 1;
    event.flags = flags;
    return event;
}

coordination::Event result_event(
        const coordination::AuthorityState& state, std::size_t node,
        const std::string& variant = "primary") {
    coordination::Event event =
        base_event(coordination::EventKind::ResultReceipt, state);
    event.generation = state.active.present
        ? state.active.generation : state.committed_generation;
    event.attempt = state.active.present ? state.active.attempt : 1;
    event.node = node_key(node);
    const auto accepted = state.active.contributions.find(event.node);
    if (accepted != state.active.contributions.end()) {
        event.incarnation = accepted->second.incarnation;
        event.sequence = accepted->second.sequence;
    } else {
        event.incarnation = incarnation_key(node, 1);
        event.sequence = 100 + event.generation;
    }
    event.exact_tokens = contribution_tokens(state);
    event.result_digest = opaque_digest(
        "result-g" + std::to_string(event.generation)
        + "-" + variant);
    return event;
}

coordination::Event commit_event(
        const coordination::AuthorityState& state,
        const std::string& variant = "primary") {
    coordination::Event event =
        base_event(coordination::EventKind::Commit, state);
    event.generation = state.committed_generation + 1;
    event.attempt = state.active.present ? state.active.attempt : 1;
    event.exact_tokens = checked_add(
        state.accepted_token_clock, contribution_tokens(state),
        "commit token clock");
    event.previous_receipt_digest = state.commit_receipt;
    event.receipt_digest = opaque_digest(
        "commit-g" + std::to_string(event.generation) + "-" + variant);
    event.manifest_digest = opaque_digest(
        "manifest-g" + std::to_string(event.generation) + "-" + variant);
    if (!state.active.result_receipts.empty()) {
        event.result_digest =
            state.active.result_receipts.begin()->second.result_digest;
    } else {
        event.result_digest = opaque_digest(
            "result-g" + std::to_string(event.generation - 1) + "-primary");
    }
    return event;
}

coordination::Event apply_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t incarnation_version, std::uint64_t sequence,
        std::uint32_t trainer_count = coordination::kRequiredTrainerReceipts,
        const std::string& variant = "primary") {
    coordination::Event event =
        base_event(coordination::EventKind::NodeApply, state);
    event.generation = state.committed_generation;
    event.node = node_key(node);
    event.incarnation = incarnation_key(node, incarnation_version);
    event.sequence = sequence;
    event.trainer_count = trainer_count;
    event.receipt_digest = opaque_digest(
        "apply-g" + std::to_string(event.generation)
        + "-n" + std::to_string(node) + "-" + variant);
    event.previous_receipt_digest = state.commit_receipt;
    return event;
}

coordination::Event expire_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t incarnation_version, std::uint64_t sequence) {
    coordination::Event event =
        base_event(coordination::EventKind::ExpirePeer, state);
    event.generation = state.committed_generation;
    event.node = node_key(node);
    event.incarnation = incarnation_key(node, incarnation_version);
    event.sequence = sequence;
    return event;
}

coordination::Event owner_lost_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t sequence) {
    coordination::Event event =
        base_event(coordination::EventKind::OwnerLost, state);
    event.generation = state.active.present
        ? state.active.generation : state.committed_generation;
    event.attempt = state.active.present ? state.active.attempt : 1;
    event.node = node_key(node);
    const auto cohort = state.active.cohort.find(event.node);
    event.incarnation = cohort == state.active.cohort.end()
        ? incarnation_key(node, 1) : cohort->second;
    event.sequence = sequence;
    return event;
}

coordination::Event query_event(
        const coordination::AuthorityState& state,
        std::uint64_t generation) {
    coordination::Event event =
        base_event(coordination::EventKind::QueryCommit, state);
    event.generation = generation;
    return event;
}

coordination::Event recover_node_apply_event(
        const coordination::AuthorityState& state, std::size_t node,
        std::uint64_t incarnation_version,
        const coordination::Digest& receipt) {
    coordination::Event event =
        base_event(coordination::EventKind::RecoverNodeApply, state);
    event.generation = state.committed_generation;
    event.node = node_key(node);
    event.incarnation = incarnation_key(node, incarnation_version);
    event.trainer_count = coordination::kRequiredTrainerReceipts;
    event.receipt_digest = receipt;
    return event;
}

class Builder {
public:
    explicit Builder(std::string name) {
        schedule_.name = std::move(name);
    }

    Builder(const Schedule& schedule,
            const coordination::AuthorityState& state)
        : schedule_(schedule), state_(state) {}

    void append(const Step& step) {
        const AppliedStep applied = apply_step(state_, step);
        std::string reason;
        if (!coordination::invariant(applied.transition.state, &reason)) {
            throw std::runtime_error(
                "fixture created invalid state at " + step.label + ": "
                + reason);
        }
        schedule_.steps.push_back(step);
        state_ = applied.transition.state;
    }

    void append(const std::string& label,
                const coordination::Event& event) {
        append(native_step(label, event));
    }

    const coordination::AuthorityState& state() const noexcept {
        return state_;
    }

    const Schedule& schedule() const noexcept { return schedule_; }

    Schedule finish() const { return schedule_; }

private:
    Schedule schedule_;
    coordination::AuthorityState state_;
};

Builder ready_builder(
        const std::string& name, std::size_t nodes,
        std::uint32_t minimum_nodes = 2,
        std::uint64_t minimum_tokens = 2) {
    Builder builder(name);
    builder.append("recover-authority", configured_event(
        0, 0, kJob5105811, minimum_nodes, minimum_tokens));
    for (std::size_t node = 0; node != nodes; ++node) {
        builder.append(
            "recover-peer-n" + std::to_string(node),
            recover_peer_event(builder.state(), node, 1, 1));
        builder.append(
            "ready-n" + std::to_string(node),
            ready_event(builder.state(), node, 1, 1));
    }
    return builder;
}

Builder open_builder(const std::string& name, std::size_t nodes) {
    Builder builder = ready_builder(name, nodes);
    builder.append("open-generation", open_event(builder.state()));
    return builder;
}

void append_primary_contribution(Builder& builder, std::size_t node) {
    builder.append(
        "contribution-n" + std::to_string(node),
        contribution_event(
            builder.state(), node, 100 + builder.state().active.generation,
            static_cast<std::uint64_t>(node + 1)));
}

Builder contributed_builder(
        const std::string& name, std::size_t nodes,
        std::size_t contributions = 2) {
    Builder builder = open_builder(name, nodes);
    for (std::size_t node = 0;
         node != std::min(nodes, contributions); ++node)
        append_primary_contribution(builder, node);
    return builder;
}

Builder closed_builder(
        const std::string& name, std::size_t nodes,
        std::size_t contributions = 2) {
    Builder builder = contributed_builder(name, nodes, contributions);
    builder.append(
        "finite-close",
        close_event(builder.state(), coordination::EventFlagFiniteClose));
    return builder;
}

void append_primary_results(Builder& builder) {
    std::vector<std::size_t> nodes;
    for (std::size_t node = 0; node != 4; ++node) {
        if (builder.state().active.contributions.count(node_key(node)) != 0)
            nodes.push_back(node);
    }
    for (const std::size_t node : nodes) {
        builder.append(
            "result-receipt-n" + std::to_string(node),
            result_event(builder.state(), node));
    }
}

Builder results_builder(
        const std::string& name, std::size_t nodes,
        std::size_t contributions = 2) {
    Builder builder = closed_builder(name, nodes, contributions);
    append_primary_results(builder);
    return builder;
}

Builder committed_builder(
        const std::string& name, std::size_t nodes,
        std::size_t contributions = 2) {
    Builder builder = results_builder(name, nodes, contributions);
    builder.append("commit", commit_event(builder.state()));
    return builder;
}

Builder applied_builder(
        const std::string& name, std::size_t nodes) {
    Builder builder = committed_builder(name, nodes);
    builder.append(
        "node-apply-n0",
        apply_event(builder.state(), 0, 1, 2));
    return builder;
}

Schedule built_in_job5105811() {
    Builder builder = ready_builder("job5105811", 3);
    builder.append("open-generation", open_event(builder.state()));
    append_primary_contribution(builder, 0);
    append_primary_contribution(builder, 2);
    builder.append(
        "finite-close",
        close_event(builder.state(), coordination::EventFlagFiniteClose));
    append_primary_results(builder);
    builder.append("commit-generation-1", commit_event(builder.state()));

    builder.append(
        "recover-node0-new-incarnation",
        recover_peer_event(builder.state(), 0, 2, 2));

    coordination::Event closed =
        contribution_event(builder.state(), 1, 7, 10, "job5105811-late");
    closed.generation = 0;
    closed.attempt = 1;
    closed.incarnation = incarnation_key(1, 1);
    builder.append("closed-generation-contribution-node1", closed);

    builder.append(
        "partial-seven-trainer-apply",
        apply_event(builder.state(), 0, 2, 2, 7, "job5105811"));
    const coordination::Event node0_apply =
        apply_event(builder.state(), 0, 2, 2, 8, "job5105811");
    builder.append("all-eight-node0-apply", node0_apply);
    builder.append(
        "node0-ready-generation-1",
        ready_event(
            builder.state(), 0, 2, 2, node0_apply.receipt_digest));

    builder.append(
        "recover-live-node1",
        recover_peer_event(builder.state(), 1, 1, 2));
    const coordination::Event node1_apply =
        apply_event(builder.state(), 1, 1, 2, 8, "job5105811");
    builder.append("all-eight-node1-apply", node1_apply);
    builder.append(
        "node1-ready-generation-1",
        ready_event(
            builder.state(), 1, 1, 2, node1_apply.receipt_digest));
    builder.append("open-generation-1", open_event(builder.state()));
    return builder.finish();
}

Schedule built_in_mutant_stale_fence() {
    Builder builder("known-bad-stale-fence");
    builder.append("recover-authority", configured_event());
    coordination::Event stale = query_event(builder.state(), 0);
    --stale.fence;
    builder.append("stale-fence-query", stale);
    Schedule result = builder.finish();
    result.expected = "mutant:stale-noninterference";
    return result;
}

Schedule built_in_mutant_partial_apply() {
    Builder builder("known-bad-partial-apply");
    builder.append(
        "recover-committed-authority",
        configured_event(1, 8, kJob5105811, 2, 2));
    builder.append(
        "recover-peer",
        recover_peer_event(builder.state(), 0, 1, 1));
    builder.append(
        "partial-seven-trainer-apply",
        apply_event(builder.state(), 0, 1, 1, 7, "known-bad"));
    Schedule result = builder.finish();
    result.expected = "mutant:no-partial-authority";
    return result;
}

Schedule built_in_fixture(const std::string& fixture) {
    if (fixture == "job5105811-v1") return built_in_job5105811();
    if (fixture == "known-bad-stale-fence-v1")
        return built_in_mutant_stale_fence();
    if (fixture == "known-bad-partial-apply-v1")
        return built_in_mutant_partial_apply();
    throw std::runtime_error("unknown native coordination fixture: " + fixture);
}

bool causally_well_formed(
        const Schedule& schedule, std::string* reason = nullptr) {
    const auto fail = [&](const std::string& value) {
        if (reason != nullptr) *reason = value;
        return false;
    };
    bool configured = false;
    coordination::Key run{};
    coordination::Digest policy{};
    std::set<std::pair<coordination::Key, coordination::Key>> peers;
    std::set<std::pair<coordination::Key, coordination::Key>> ready;
    using GenerationIdentity = std::pair<std::uint64_t, std::uint32_t>;
    std::map<GenerationIdentity, std::set<
        std::pair<coordination::Key, coordination::Key>>> cohorts;
    std::set<std::tuple<
        std::uint64_t, std::uint32_t, coordination::Key>> contributions;

    for (std::size_t index = 0; index != schedule.steps.size(); ++index) {
        const Step& step = schedule.steps[index];
        if (step.kind == Step::Kind::Restart) {
            if (!configured)
                return fail("restart precedes authority at step "
                            + std::to_string(index));
            // Native service/peer-control recovery retains only durable
            // commit authority. READY membership and every unfinished
            // generation cause are volatile and must be established again.
            ready.clear();
            cohorts.clear();
            contributions.clear();
            continue;
        }
        const coordination::Event& event = step.event;
        if (event.kind == coordination::EventKind::RecoverAuthority) {
            if (coordination::is_zero(event.run)
                || coordination::is_zero(event.policy_digest)
                || event.fence == 0)
                return fail("invalid authority cause at step "
                            + std::to_string(index));
            if (!configured) {
                configured = true;
                run = event.run;
                policy = event.policy_digest;
            }
            continue;
        }
        if (!configured)
            return fail("native event precedes authority at step "
                        + std::to_string(index));
        if (event.run != run)
            return fail("event changes logical run at step "
                        + std::to_string(index));
        if (!coordination::is_zero(event.policy_digest)
            && event.policy_digest != policy)
            return fail("event changes policy at step "
                        + std::to_string(index));

        const auto peer =
            std::make_pair(event.node, event.incarnation);
        const GenerationIdentity generation =
            std::make_pair(event.generation, event.attempt);
        switch (event.kind) {
            case coordination::EventKind::RecoverAuthority:
                break;
            case coordination::EventKind::RecoverNodeApply:
                if (event.generation == 0)
                    return fail("recovered apply lacks committed cause");
                break;
            case coordination::EventKind::RecoverPeer:
                peers.insert(peer);
                break;
            case coordination::EventKind::Ready:
                if (peers.count(peer) == 0)
                    return fail("READY lacks recover-peer cause at step "
                                + std::to_string(index));
                ready.insert(peer);
                break;
            case coordination::EventKind::OpenGeneration:
                if (event.attempt == 0)
                    return fail("open lacks attempt identity");
                // A duplicate open is an observation race, not permission to
                // rewrite the first immutable READY snapshot.
                cohorts.emplace(generation, ready);
                break;
            case coordination::EventKind::Contribution:
                if (cohorts.count(generation) == 0)
                    return fail("contribution lacks open cause at step "
                                + std::to_string(index));
                if (peers.count(peer) == 0)
                    return fail("contribution lacks recovered peer cause at step "
                                + std::to_string(index));
                contributions.emplace(
                    event.generation, event.attempt, event.node);
                break;
            case coordination::EventKind::CloseGeneration:
                if (cohorts.count(generation) == 0)
                    return fail("close lacks open cause at step "
                                + std::to_string(index));
                break;
            case coordination::EventKind::ResultReceipt:
                if (contributions.count(std::make_tuple(
                        event.generation, event.attempt, event.node)) == 0)
                    return fail("result lacks contribution cause at step "
                                + std::to_string(index));
                break;
            case coordination::EventKind::Commit: {
                bool planned = false;
                for (const auto& item : contributions) {
                    if (std::get<0>(item) + 1 == event.generation) {
                        planned = true;
                        break;
                    }
                }
                if (!planned)
                    return fail("commit lacks contribution/result cause at step "
                                + std::to_string(index));
                break;
            }
            case coordination::EventKind::NodeApply:
                if (peers.count(peer) == 0)
                    return fail("node apply lacks peer/commit cause at step "
                                + std::to_string(index));
                break;
            case coordination::EventKind::ExpirePeer:
                if (peers.count(peer) == 0)
                    return fail("expiry lacks peer cause at step "
                                + std::to_string(index));
                // This layer records causal history, not the disposition of
                // an expiry race. A stale expiry does not revoke the prior
                // READY cause; production state decides whether it applied.
                break;
            case coordination::EventKind::OwnerLost:
                if (cohorts.count(generation) == 0
                    || peers.count(peer) == 0)
                    return fail("owner loss lacks open/peer cause at step "
                                + std::to_string(index));
                break;
            case coordination::EventKind::QueryCommit:
                break;
        }
    }
    return true;
}

enum class Mutant {
    None,
    IgnoreStaleFence,
    GrantPartialApply,
};

struct Failure {
    std::string predicate;
    std::string message;
    std::size_t step_index = 0;
};

struct CommitIdentity {
    std::uint64_t tokens = 0;
    coordination::Digest receipt{};
    coordination::Digest manifest{};
    coordination::Digest result{};
};

bool same_commit(
        const CommitIdentity& left, const CommitIdentity& right) {
    return left.tokens == right.tokens
        && left.receipt == right.receipt
        && left.manifest == right.manifest
        && left.result == right.result;
}

bool same_contribution(
        const coordination::Contribution& left,
        const coordination::Contribution& right) {
    return left.incarnation == right.incarnation
        && left.sequence == right.sequence
        && left.exact_tokens == right.exact_tokens
        && left.payload_digest == right.payload_digest
        && left.receipt_digest == right.receipt_digest;
}

bool same_contributions(
        const std::map<coordination::Key, coordination::Contribution>& left,
        const std::map<coordination::Key, coordination::Contribution>& right) {
    if (left.size() != right.size()) return false;
    auto left_it = left.begin();
    auto right_it = right.begin();
    while (left_it != left.end()) {
        if (left_it->first != right_it->first
            || !same_contribution(left_it->second, right_it->second))
            return false;
        ++left_it;
        ++right_it;
    }
    return true;
}

bool same_effects(
        const std::vector<coordination::Effect>& left,
        const std::vector<coordination::Effect>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index != left.size(); ++index) {
        if (left[index].kind != right[index].kind
            || left[index].generation != right[index].generation
            || left[index].node != right[index].node
            || left[index].digest != right[index].digest)
            return false;
    }
    return true;
}

bool disposition_must_not_mutate(coordination::Disposition value) {
    switch (value) {
        case coordination::Disposition::IdenticalDuplicate:
        case coordination::Disposition::ConflictingDuplicate:
        case coordination::Disposition::StaleFence:
        case coordination::Disposition::StaleIncarnation:
        case coordination::Disposition::StaleGeneration:
        case coordination::Disposition::GenerationClosed:
        case coordination::Disposition::Deferred:
        case coordination::Disposition::InsufficientCohort:
        case coordination::Disposition::Corrupt:
        case coordination::Disposition::InvalidEvent:
        case coordination::Disposition::FatalInvariant:
            return true;
        case coordination::Disposition::Accepted:
        case coordination::Disposition::RetryNextGeneration:
            return false;
    }
    return true;
}

struct SafetyOracle {
    bool initialized = false;
    std::uint64_t maximum_fence = 0;
    std::uint64_t maximum_generation = 0;
    std::uint64_t maximum_tokens = 0;
    std::map<std::uint64_t, CommitIdentity> commits;
    std::map<coordination::Key, std::uint64_t> applied_generation;
    std::set<std::pair<std::uint64_t, std::uint32_t>> named_closures;
    std::size_t progress_commits = 0;

    std::optional<Failure> check(
            const coordination::AuthorityState& logical_before,
            const AppliedStep& applied, std::size_t step_index,
            Mutant mutant) {
        const coordination::Transition& transition = applied.transition;
        const coordination::Event& event = applied.event;
        const coordination::AuthorityState& state = transition.state;

        if (mutant == Mutant::IgnoreStaleFence
            && transition.disposition
                == coordination::Disposition::StaleFence) {
            return Failure{
                "stale-noninterference",
                "known-bad mutation writes stale-fence authority",
                step_index};
        }
        if (mutant == Mutant::GrantPartialApply
            && event.kind == coordination::EventKind::NodeApply
            && event.trainer_count
                != coordination::kRequiredTrainerReceipts
            && transition.disposition == coordination::Disposition::Corrupt) {
            return Failure{
                "no-partial-authority",
                "known-bad mutation grants partial trainer apply",
                step_index};
        }

        std::string reason;
        if (!coordination::invariant(state, &reason)) {
            return Failure{
                "native-invariant", "native invariant failed: " + reason,
                step_index};
        }
        coordination::AuthorityState kernel_before =
            applied.restart_boundary
                ? coordination::AuthorityState{} : logical_before;
        if (transition.pre_state_digest
                != coordination::state_digest(kernel_before)
            || transition.post_state_digest
                != coordination::state_digest(state)) {
            return Failure{
                "deterministic-state-digest",
                "transition state digest does not bind exact state",
                step_index};
        }
        if (transition.trace.empty()
            || transition.trace.size() >= coordination::kMaximumTraceBytes
            || transition.trace.find(
                "\"schema\":\"emender-native-coordination-trace-v1\"")
                == std::string::npos
            || transition.trace.find(
                "\"kernel\":\"emender-native-coordination-kernel-v1\"")
                == std::string::npos
            || transition.trace.find(
                "\"post_state_digest\":\""
                + coordination::hex(transition.post_state_digest) + "\"")
                == std::string::npos) {
            return Failure{
                "canonical-trace",
                "transition did not emit one bounded canonical trace",
                step_index};
        }
        if (transition.effects.empty()
            || transition.effects.size() > coordination::kMaximumEffects
            || transition.effects.back().kind
                != coordination::EffectKind::EmitTrace) {
            return Failure{
                "bounded-protocol-state",
                "transition effects violate fixed production bounds",
                step_index};
        }
        if (state.members.size() > coordination::kMaximumNodes
            || state.recovered_node_applies.size()
                > coordination::kMaximumNodes
            || state.active.cohort.size() > coordination::kMaximumNodes
            || state.active.contributions.size()
                > coordination::kMaximumNodes
            || state.active.result_receipts.size()
                > coordination::kMaximumNodes
            || state.active.owner_reassignments
                > coordination::kMaximumOwnerReassignments) {
            return Failure{
                "bounded-protocol-state",
                "authoritative maps/reassignment counters exceeded bounds",
                step_index};
        }
        if (disposition_must_not_mutate(transition.disposition)
            && transition.pre_state_digest != transition.post_state_digest) {
            return Failure{
                transition.disposition
                        == coordination::Disposition::StaleFence
                    || transition.disposition
                        == coordination::Disposition::StaleIncarnation
                    || transition.disposition
                        == coordination::Disposition::StaleGeneration
                    ? "stale-noninterference" : "idempotence",
                "typed non-mutating disposition changed authority",
                step_index};
        }
        if (event.kind == coordination::EventKind::NodeApply
            && event.trainer_count
                != coordination::kRequiredTrainerReceipts
            && transition.pre_state_digest != transition.post_state_digest) {
            return Failure{
                "no-partial-authority",
                "partial trainer receipt changed node authority",
                step_index};
        }
        if (event.kind == coordination::EventKind::RecoverNodeApply
            && event.trainer_count
                != coordination::kRequiredTrainerReceipts
            && transition.pre_state_digest != transition.post_state_digest) {
            return Failure{
                "no-partial-authority",
                "partial recovered receipt changed node authority",
                step_index};
        }

        const coordination::Transition repeat =
            coordination::step(kernel_before, event);
        if (repeat.disposition != transition.disposition
            || repeat.pre_state_digest != transition.pre_state_digest
            || repeat.post_state_digest != transition.post_state_digest
            || repeat.trace != transition.trace
            || !same_effects(repeat.effects, transition.effects)) {
            return Failure{
                "deterministic-transition",
                "identical pre-state/event did not produce byte-identical output",
                step_index};
        }
        if (transition.disposition == coordination::Disposition::Accepted) {
            const coordination::Transition replay =
                coordination::step(state, event);
            if (replay.post_state_digest != transition.post_state_digest
                || (replay.disposition
                        != coordination::Disposition::IdenticalDuplicate
                    && !(event.kind
                            == coordination::EventKind::QueryCommit
                        && replay.disposition
                            == coordination::Disposition::Accepted))) {
                return Failure{
                    "idempotence",
                    "accepted event replay changed authority or outcome",
                    step_index};
            }
        }

        if (!applied.restart_boundary
            && logical_before.active.present && state.active.present
            && logical_before.active.generation == state.active.generation
            && logical_before.active.attempt == state.active.attempt
            && static_cast<std::uint32_t>(
                    logical_before.active.phase)
                >= static_cast<std::uint32_t>(
                    coordination::GenerationPhase::Closed)) {
            if (logical_before.active.cohort != state.active.cohort
                || !same_contributions(
                    logical_before.active.contributions,
                    state.active.contributions)) {
                return Failure{
                    "immutable-cohort-closure",
                    "closed cohort or contribution receipts changed",
                    step_index};
            }
        }

        for (const auto& item : state.members) {
            const coordination::Member& member = item.second;
            if (member.ready
                && (!member.live
                    || member.ready_generation
                        != state.committed_generation
                    || (state.committed_generation != 0
                        && (member.applied_generation
                                != state.committed_generation
                            || member.synchronized_generation
                                != state.committed_generation
                            || coordination::is_zero(
                                member.apply_receipt))))) {
                return Failure{
                    "no-partial-authority",
                    "READY member lacks complete current apply authority",
                    step_index};
            }
            if (member.applied_generation != 0) {
                const std::uint64_t prior = applied_generation[item.first];
                if (member.applied_generation < prior) {
                    return Failure{
                        "monotonic-recovery",
                        "node applied generation rolled backward",
                        step_index};
                }
                applied_generation[item.first] =
                    std::max(prior, member.applied_generation);
            }
        }

        if (state.configured) {
            if (initialized) {
                if (state.fence < maximum_fence
                    || state.committed_generation < maximum_generation
                    || state.accepted_token_clock < maximum_tokens) {
                    return Failure{
                        "monotonic-recovery",
                        "fence/generation/token authority rolled backward",
                        step_index};
                }
            }
            initialized = true;
            maximum_fence = std::max(maximum_fence, state.fence);
            maximum_generation =
                std::max(maximum_generation, state.committed_generation);
            maximum_tokens =
                std::max(maximum_tokens, state.accepted_token_clock);
            if (state.committed_generation != 0) {
                const CommitIdentity identity{
                    state.accepted_token_clock, state.commit_receipt,
                    state.commit_manifest, state.committed_result};
                const auto prior = commits.find(state.committed_generation);
                if (prior != commits.end()
                    && !same_commit(prior->second, identity)) {
                    return Failure{
                        "unique-commit",
                        "one committed generation acquired two identities",
                        step_index};
                }
                commits[state.committed_generation] = identity;
            }
        }

        if (event.kind == coordination::EventKind::CloseGeneration
            && transition.disposition
                == coordination::Disposition::Accepted) {
            const bool named =
                (event.flags & coordination::EventFlagFiniteClose) != 0
                || (event.flags
                    & coordination::EventFlagDeadlineExpired) != 0;
            if (!named) {
                return Failure{
                    "named-progress-predicates",
                    "generation closed without finite/deadline predicate",
                    step_index};
            }
            named_closures.emplace(event.generation, event.attempt);
        }
        if (event.kind == coordination::EventKind::Commit
            && transition.disposition
                == coordination::Disposition::Accepted
            && state.committed_generation
                > logical_before.committed_generation) {
            const bool quorum =
                logical_before.active.contributions.size()
                    >= logical_before.minimum_nodes
                && contribution_tokens(logical_before)
                    >= logical_before.minimum_tokens;
            const bool delivery =
                !logical_before.active.result_receipts.empty()
                && logical_before.active.result_receipts.size()
                    == logical_before.active.contributions.size();
            const bool deadline_or_finite =
                named_closures.count(std::make_pair(
                    logical_before.active.generation,
                    logical_before.active.attempt)) != 0;
            const bool fairness = event.kind
                == coordination::EventKind::Commit;
            if (!quorum || !delivery || !deadline_or_finite || !fairness) {
                return Failure{
                    "named-progress-predicates",
                    "commit advanced without quorum/deadline-or-finite/"
                    "delivery/fairness predicates",
                    step_index};
            }
            ++progress_commits;
        }
        return std::nullopt;
    }
};

struct Evaluation {
    std::optional<Failure> failure;
    std::size_t transitions = 0;
    std::array<std::uint64_t, 14> event_counts{};
    std::array<std::uint64_t, 14> disposition_counts{};
    std::array<std::uint64_t, 14> effect_counts{};
    std::size_t progress_commits = 0;
    coordination::Digest transcript_digest{};
    std::string transcript;
};

Evaluation evaluate_schedule(
        const Schedule& schedule, Mutant mutant = Mutant::None,
        bool capture_transcript = false) {
    Evaluation evaluation;
    std::string causal_reason;
    if (!causally_well_formed(schedule, &causal_reason)) {
        evaluation.failure = Failure{
            "causal-preconditions", causal_reason, 0};
        return evaluation;
    }
    coordination::AuthorityState state;
    SafetyOracle oracle;
    emender_ndp::Sha256 transcript_hash;
    const std::string schedule_header =
        "schedule:" + schedule.name + "\n";
    transcript_hash.update(schedule_header.data(), schedule_header.size());
    if (capture_transcript) evaluation.transcript += schedule_header;

    for (std::size_t index = 0; index != schedule.steps.size(); ++index) {
        AppliedStep applied;
        try {
            applied = apply_step(state, schedule.steps[index]);
        } catch (const std::exception& error) {
            evaluation.failure = Failure{
                "restart-recovery", error.what(), index};
            return evaluation;
        }
        if (applied.restart_boundary) {
            const std::string marker =
                "{\"control\":\"restart\",\"role\":\""
                + json_escape(applied.restart_role)
                + "\",\"prior_state_digest\":\""
                + coordination::hex(coordination::state_digest(state))
                + "\"}\n";
            transcript_hash.update(marker.data(), marker.size());
            if (capture_transcript) evaluation.transcript += marker;
        }
        const std::string trace = applied.transition.trace + "\n";
        transcript_hash.update(trace.data(), trace.size());
        if (capture_transcript) evaluation.transcript += trace;

        ++evaluation.transitions;
        const std::size_t event_index =
            static_cast<std::size_t>(applied.event.kind);
        const std::size_t disposition_index =
            static_cast<std::size_t>(applied.transition.disposition);
        if (event_index < evaluation.event_counts.size())
            ++evaluation.event_counts[event_index];
        if (disposition_index < evaluation.disposition_counts.size())
            ++evaluation.disposition_counts[disposition_index];
        for (const coordination::Effect& effect :
             applied.transition.effects) {
            const std::size_t effect_index =
                static_cast<std::size_t>(effect.kind);
            if (effect_index < evaluation.effect_counts.size())
                ++evaluation.effect_counts[effect_index];
        }
        const std::optional<Failure> failure =
            oracle.check(state, applied, index, mutant);
        if (failure.has_value()) {
            evaluation.failure = failure;
            evaluation.transcript_digest = transcript_hash.finish();
            return evaluation;
        }
        state = applied.transition.state;
    }
    evaluation.progress_commits = oracle.progress_commits;
    evaluation.transcript_digest = transcript_hash.finish();
    return evaluation;
}

bool retains_predicate(
        const Schedule& candidate, Mutant mutant,
        const std::string& predicate) {
    std::string reason;
    if (!causally_well_formed(candidate, &reason)) return false;
    const Evaluation evaluation = evaluate_schedule(candidate, mutant, false);
    return evaluation.failure.has_value()
        && evaluation.failure->predicate == predicate;
}

Schedule shrink_failure(
        const Schedule& input, Mutant mutant,
        const std::string& predicate) {
    Schedule current = input;

    // 1. Remove the largest failing suffix first.
    for (std::size_t keep = 1; keep < current.steps.size(); ++keep) {
        Schedule candidate = current;
        candidate.steps.resize(keep);
        if (retains_predicate(candidate, mutant, predicate)) {
            current = std::move(candidate);
            break;
        }
    }

    // 2. Causal delta debugging: largest contiguous chunks, left-to-right.
    std::size_t chunk = current.steps.size() / 2U;
    while (chunk != 0) {
        bool changed = false;
        for (std::size_t begin = 0;
             begin + chunk <= current.steps.size(); ++begin) {
            Schedule candidate = current;
            candidate.steps.erase(
                candidate.steps.begin()
                    + static_cast<std::ptrdiff_t>(begin),
                candidate.steps.begin()
                    + static_cast<std::ptrdiff_t>(begin + chunk));
            if (!candidate.steps.empty()
                && retains_predicate(candidate, mutant, predicate)) {
                current = std::move(candidate);
                changed = true;
                break;
            }
        }
        if (!changed) chunk /= 2U;
    }

    // 3. Remove single events in newest-to-oldest order to a fixed point.
    bool removed = true;
    while (removed) {
        removed = false;
        for (std::size_t offset = 0;
             offset != current.steps.size(); ++offset) {
            const std::size_t index = current.steps.size() - 1U - offset;
            Schedule candidate = current;
            candidate.steps.erase(
                candidate.steps.begin()
                + static_cast<std::ptrdiff_t>(index));
            if (!candidate.steps.empty()
                && retains_predicate(candidate, mutant, predicate)) {
                current = std::move(candidate);
                removed = true;
                break;
            }
        }
    }

    // 4. Simplify event scalars in stable field order.
    for (std::size_t index = 0; index != current.steps.size(); ++index) {
        if (current.steps[index].kind != Step::Kind::NativeEvent) continue;
        const auto try_event = [&](const coordination::Event& event) {
            Schedule candidate = current;
            candidate.steps[index].event = event;
            if (retains_predicate(candidate, mutant, predicate)) {
                current = std::move(candidate);
                return true;
            }
            return false;
        };
        coordination::Event event = current.steps[index].event;
        if (event.flags != 0) {
            auto simplified = event;
            simplified.flags =
                (event.flags & coordination::EventFlagFiniteClose) != 0
                ? coordination::EventFlagFiniteClose
                : coordination::EventFlagDeadlineExpired;
            (void)try_event(simplified);
            event = current.steps[index].event;
        }
        if (event.sequence > 1) {
            auto simplified = event;
            simplified.sequence = 1;
            (void)try_event(simplified);
            event = current.steps[index].event;
        }
        if (event.exact_tokens > 1) {
            auto simplified = event;
            simplified.exact_tokens = 1;
            (void)try_event(simplified);
        }
    }
    return current;
}

struct SystematicSuite {
    std::vector<Schedule> schedules;
    std::set<std::string> pair_ids;
    std::set<std::string> ordered_pair_cases;
    std::set<std::string> three_race_ids;
    std::set<std::string> three_race_permutations;
    std::set<std::string> restart_phase_cases;
};

void require_causal_schedule(const Schedule& schedule) {
    std::string reason;
    if (!causally_well_formed(schedule, &reason)) {
        throw std::runtime_error(
            "generated schedule is not causal: " + schedule.name
            + ": " + reason);
    }
}

void add_pair(
        SystematicSuite& suite, const std::string& pair_id,
        const Builder& base, const Step& first, const Step& second,
        std::size_t nodes) {
    suite.pair_ids.insert(pair_id);
    for (const bool reverse : {false, true}) {
        Schedule schedule = base.finish();
        const std::string order = reverse ? "ba" : "ab";
        schedule.name =
            "pair-" + pair_id + "-" + order
            + "-" + std::to_string(nodes) + "n";
        schedule.steps.push_back(reverse ? second : first);
        schedule.steps.push_back(reverse ? first : second);
        require_causal_schedule(schedule);
        suite.ordered_pair_cases.insert(pair_id + ":" + order);
        suite.schedules.push_back(std::move(schedule));
    }
}

coordination::Event duplicate_contribution_event(
        const coordination::AuthorityState& state, std::size_t node) {
    const auto found = state.active.contributions.find(node_key(node));
    if (found == state.active.contributions.end())
        throw std::runtime_error("fixture lacks duplicate contribution");
    coordination::Event event =
        base_event(coordination::EventKind::Contribution, state);
    event.generation = state.active.generation;
    event.attempt = state.active.attempt;
    event.node = found->first;
    event.incarnation = found->second.incarnation;
    event.sequence = found->second.sequence;
    event.exact_tokens = found->second.exact_tokens;
    event.payload_digest = found->second.payload_digest;
    return event;
}

coordination::Event duplicate_result_event(
        const coordination::AuthorityState& state, std::size_t node) {
    const auto found = state.active.result_receipts.find(node_key(node));
    if (found == state.active.result_receipts.end())
        throw std::runtime_error("fixture lacks duplicate result");
    coordination::Event event =
        base_event(coordination::EventKind::ResultReceipt, state);
    event.generation = state.active.generation;
    event.attempt = state.active.attempt;
    event.node = found->first;
    event.incarnation = found->second.incarnation;
    event.sequence = found->second.sequence;
    event.exact_tokens = found->second.exact_tokens;
    event.result_digest = found->second.result_digest;
    return event;
}

void add_pair_schedules(SystematicSuite& suite, std::size_t nodes) {
    {
        Builder base = ready_builder("pair-base", nodes);
        const Step ready = native_step(
            "ready-replay", ready_event(base.state(), 0, 1, 1));
        const Step expire = native_step(
            "ready-expiry", expire_event(base.state(), 0, 1, 1));
        add_pair(suite, "ready-expire", base, ready, expire, nodes);
    }
    {
        Builder base = ready_builder("pair-base", nodes);
        add_pair(
            suite, "ready-open", base,
            native_step(
                "ready-replay", ready_event(base.state(), 0, 1, 1)),
            native_step("open", open_event(base.state())), nodes);
        add_pair(
            suite, "open-expire", base,
            native_step("open", open_event(base.state())),
            native_step(
                "expire", expire_event(base.state(), nodes - 1, 1, 1)),
            nodes);
    }
    {
        Builder base = open_builder("pair-base", nodes);
        add_pair(
            suite, "contribution-contribution", base,
            native_step(
                "contribution-n0",
                contribution_event(base.state(), 0, 100, 1)),
            native_step(
                "contribution-n1",
                contribution_event(base.state(), 1, 100, 2)),
            nodes);
        add_pair(
            suite, "contribution-expire", base,
            native_step(
                "contribution-n0",
                contribution_event(base.state(), 0, 100, 1)),
            native_step(
                "expire-n0", expire_event(base.state(), 0, 1, 1)),
            nodes);
        coordination::Event valid =
            contribution_event(base.state(), 0, 100, 1);
        coordination::Event stale_fence = valid;
        --stale_fence.fence;
        add_pair(
            suite, "contribution-stale-fence", base,
            native_step("valid-contribution", valid),
            native_step("stale-fence-contribution", stale_fence), nodes);
    }
    {
        Builder base = ready_builder("pair-base", nodes);
        base.append(
            "recover-new-incarnation",
            recover_peer_event(base.state(), 0, 2, 2));
        base.append(
            "ready-new-incarnation",
            ready_event(base.state(), 0, 2, 2));
        base.append("open", open_event(base.state()));
        coordination::Event valid =
            contribution_event(base.state(), 0, 100, 1);
        coordination::Event stale_incarnation = valid;
        stale_incarnation.incarnation = incarnation_key(0, 1);
        add_pair(
            suite, "contribution-stale-incarnation", base,
            native_step("valid-contribution", valid),
            native_step(
                "stale-incarnation-contribution", stale_incarnation),
            nodes);
    }
    if (nodes >= 3) {
        Builder base = contributed_builder("pair-base", nodes, 2);
        const Step late = native_step(
            "late-contribution",
            contribution_event(base.state(), 2, 100, 3));
        add_pair(
            suite, "contribution-finite-close", base, late,
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            nodes);
        add_pair(
            suite, "contribution-deadline-close", base, late,
            native_step(
                "deadline-close",
                close_event(
                    base.state(),
                    coordination::EventFlagDeadlineExpired)),
            nodes);
    }
    {
        Builder base = contributed_builder("pair-base", nodes, 2);
        coordination::Event duplicate =
            duplicate_contribution_event(base.state(), 0);
        coordination::Event conflict = duplicate;
        conflict.payload_digest =
            opaque_digest("conflicting-contribution-payload");
        add_pair(
            suite, "contribution-duplicate-conflict", base,
            native_step("identical-contribution", duplicate),
            native_step("conflicting-contribution", conflict), nodes);
        add_pair(
            suite, "duplicate-contribution-close", base,
            native_step("identical-contribution", duplicate),
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            nodes);
        add_pair(
            suite, "conflicting-contribution-close", base,
            native_step("conflicting-contribution", conflict),
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            nodes);
        add_pair(
            suite, "close-expire", base,
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            native_step(
                "expire-n0", expire_event(base.state(), 0, 1, 1)),
            nodes);
        add_pair(
            suite, "close-owner-loss", base,
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            native_step(
                "owner-loss-n0",
                owner_lost_event(base.state(), 0, 1)),
            nodes);
        add_pair(
            suite, "finite-close-deadline-close", base,
            native_step(
                "finite-close",
                close_event(
                    base.state(), coordination::EventFlagFiniteClose)),
            native_step(
                "deadline-close",
                close_event(
                    base.state(),
                    coordination::EventFlagDeadlineExpired)),
            nodes);
    }
    {
        Builder base = closed_builder("pair-base", nodes);
        add_pair(
            suite, "result-result", base,
            native_step("result-n0", result_event(base.state(), 0)),
            native_step("result-n1", result_event(base.state(), 1)),
            nodes);
        add_pair(
            suite, "result-owner-loss", base,
            native_step("result-n0", result_event(base.state(), 0)),
            native_step(
                "owner-loss", owner_lost_event(base.state(), 0, 1)),
            nodes);
    }
    {
        Builder base = closed_builder("pair-base", nodes);
        base.append("result-n0", result_event(base.state(), 0));
        const coordination::Event duplicate =
            duplicate_result_event(base.state(), 0);
        coordination::Event conflict = duplicate;
        conflict.result_digest = opaque_digest("conflicting-result");
        add_pair(
            suite, "result-duplicate-conflict", base,
            native_step("identical-result", duplicate),
            native_step("conflicting-result", conflict), nodes);
        coordination::Event final_result = result_event(base.state(), 1);
        const coordination::Event commit = commit_event(base.state());
        add_pair(
            suite, "result-commit", base,
            native_step("final-result", final_result),
            native_step("commit", commit), nodes);
    }
    {
        Builder base = results_builder("pair-base", nodes);
        const coordination::Event commit = commit_event(base.state());
        add_pair(
            suite, "commit-expire", base,
            native_step("commit", commit),
            native_step(
                "expire-n0", expire_event(base.state(), 0, 1, 1)),
            nodes);
        add_pair(
            suite, "commit-owner-loss", base,
            native_step("commit", commit),
            native_step(
                "owner-loss", owner_lost_event(base.state(), 0, 1)),
            nodes);
        add_pair(
            suite, "commit-recover", base,
            native_step("commit", commit),
            native_step(
                "recover-new-incarnation",
                recover_peer_event(base.state(), 0, 2, 2)),
            nodes);
        add_pair(
            suite, "commit-query-checkpoint", base,
            native_step("commit", commit),
            native_step(
                "query-checkpoint",
                query_event(base.state(), base.state().committed_generation + 1)),
            nodes);
    }
    {
        Builder base = committed_builder("pair-base", nodes);
        const coordination::Event apply =
            apply_event(base.state(), 0, 1, 2);
        add_pair(
            suite, "apply-ready", base,
            native_step("all-eight-apply", apply),
            native_step(
                "ready-after-apply",
                ready_event(
                    base.state(), 0, 1, 2, apply.receipt_digest)),
            nodes);
        add_pair(
            suite, "apply-expire", base,
            native_step("all-eight-apply", apply),
            native_step(
                "expire", expire_event(base.state(), 0, 1, 1)),
            nodes);
        add_pair(
            suite, "apply-duplicate-receipt", base,
            native_step("all-eight-apply", apply),
            native_step("duplicate-apply", apply), nodes);
    }
    {
        Builder base = ready_builder("pair-base", nodes);
        const coordination::Event recover =
            recover_peer_event(base.state(), 0, 2, 2);
        add_pair(
            suite, "recover-stale-incarnation", base,
            native_step("recover-new-incarnation", recover),
            native_step(
                "old-incarnation-ready",
                ready_event(base.state(), 0, 1, 1)),
            nodes);
    }
    {
        Builder base = closed_builder("pair-base", nodes);
        const coordination::Event owner =
            owner_lost_event(base.state(), 0, 1);
        add_pair(
            suite, "owner-loss-replay", base,
            native_step("owner-loss", owner),
            native_step("owner-loss-replay", owner), nodes);
    }
    {
        Builder base("pair-base");
        base.append(
            "recover-committed-authority",
            configured_event(1, 8, kJob5105811, 2, 2));
        const coordination::Digest apply_receipt =
            opaque_digest("recovered-apply-n0");
        const coordination::Event recovered_apply =
            recover_node_apply_event(
                base.state(), 0, 1, apply_receipt);
        const coordination::Event recover =
            recover_peer_event(base.state(), 0, 1, 1);
        add_pair(
            suite, "recover-apply-recover-peer", base,
            native_step("recover-node-apply", recovered_apply),
            native_step("recover-peer", recover), nodes);
    }
}

void add_permuted_three_race(
        SystematicSuite& suite, const std::string& race_id,
        const Builder& base, const std::array<Step, 3>& events,
        std::size_t nodes) {
    suite.three_race_ids.insert(race_id);
    std::array<int, 3> order{{0, 1, 2}};
    do {
        Schedule schedule = base.finish();
        std::string permutation;
        for (const int index : order) {
            schedule.steps.push_back(
                events[static_cast<std::size_t>(index)]);
            permutation += static_cast<char>('a' + index);
        }
        schedule.name =
            "three-" + race_id + "-" + permutation
            + "-" + std::to_string(nodes) + "n";
        require_causal_schedule(schedule);
        suite.three_race_permutations.insert(
            race_id + ":" + permutation);
        suite.schedules.push_back(std::move(schedule));
    } while (std::next_permutation(order.begin(), order.end()));
}

void add_three_race_schedules(
        SystematicSuite& suite, std::size_t nodes) {
    if (nodes >= 3) {
        const std::size_t cohort_nodes = nodes;
        Builder base =
            contributed_builder("three-base", cohort_nodes, 2);
        add_permuted_three_race(
            suite, "close-failure-contribution", base,
            std::array<Step, 3>{{
                native_step(
                    "finite-close",
                    close_event(
                        base.state(),
                        coordination::EventFlagFiniteClose)),
                native_step(
                    "participant-failure",
                    expire_event(base.state(), 2, 1, 1)),
                native_step(
                    "late-contribution",
                    contribution_event(base.state(), 2, 100, 3)),
            }}, nodes);
    }
    {
        Builder base = results_builder("three-base", nodes);
        add_permuted_three_race(
            suite, "commit-failure-rejoin", base,
            std::array<Step, 3>{{
                native_step("commit", commit_event(base.state())),
                native_step(
                    "participant-failure",
                    expire_event(base.state(), 0, 1, 1)),
                native_step(
                    "rejoin-new-incarnation",
                    recover_peer_event(base.state(), 0, 2, 2)),
            }}, nodes);
    }
    {
        Builder base = closed_builder("three-base", nodes);
        add_permuted_three_race(
            suite, "owner-loss-replay-receipt", base,
            std::array<Step, 3>{{
                native_step(
                    "owner-loss",
                    owner_lost_event(base.state(), 0, 1)),
                native_step(
                    "contribution-replay",
                    duplicate_contribution_event(base.state(), 0)),
                native_step(
                    "result-receipt", result_event(base.state(), 0)),
            }}, nodes);
    }
    {
        Builder base = committed_builder("three-base", nodes);
        const coordination::Event apply =
            apply_event(base.state(), 0, 1, 2);
        add_permuted_three_race(
            suite, "apply-restart-duplicate-receipt", base,
            std::array<Step, 3>{{
                native_step("all-eight-apply", apply),
                restart_step("service-restart", "service", false),
                native_step("duplicate-apply-receipt", apply),
            }}, nodes);
    }
}

Builder phase_builder(
        const std::string& phase, std::size_t nodes) {
    if (phase == "authority") {
        Builder builder("phase-base");
        builder.append("recover-authority", configured_event());
        return builder;
    }
    if (phase == "ready") return ready_builder("phase-base", nodes);
    if (phase == "open") return open_builder("phase-base", nodes);
    if (phase == "contribution")
        return contributed_builder("phase-base", nodes, 2);
    if (phase == "closed") return closed_builder("phase-base", nodes);
    if (phase == "result-publication") {
        Builder builder = closed_builder("phase-base", nodes);
        builder.append("result-n0", result_event(builder.state(), 0));
        return builder;
    }
    if (phase == "commit") return committed_builder("phase-base", nodes);
    if (phase == "apply") return applied_builder("phase-base", nodes);
    throw std::runtime_error("unknown restart phase: " + phase);
}

void add_restart_matrix(
        SystematicSuite& suite, std::size_t nodes) {
    static const std::array<const char*, 8> phases{{
        "authority", "ready", "open", "contribution", "closed",
        "result-publication", "commit", "apply",
    }};
    static const std::array<const char*, 4> roles{{
        "trainer", "manager", "service", "peer-control",
    }};
    for (const char* phase : phases) {
        for (const char* role : roles) {
            Builder base = phase_builder(phase, nodes);
            Schedule schedule = base.finish();
            schedule.name =
                "restart-" + std::string(role) + "-"
                + phase + "-" + std::to_string(nodes) + "n";
            schedule.steps.push_back(restart_step(
                "restart-" + std::string(role), role,
                std::string(role) == "peer-control"));
            require_causal_schedule(schedule);
            suite.restart_phase_cases.insert(
                std::string(role) + ":" + phase);
            suite.schedules.push_back(std::move(schedule));
        }
    }
}

void add_targeted_schedules(
        SystematicSuite& suite, std::size_t nodes) {
    {
        Builder builder = open_builder("drop-deadline", nodes);
        append_primary_contribution(builder, 0);
        builder.append(
            "dropped-contribution-deadline",
            close_event(
                builder.state(),
                coordination::EventFlagDeadlineExpired));
        Schedule schedule = builder.finish();
        schedule.name =
            "contribution-drop-deadline-" + std::to_string(nodes) + "n";
        suite.schedules.push_back(std::move(schedule));
    }
    {
        Builder builder = open_builder("insufficient-finite-close", nodes);
        append_primary_contribution(builder, 0);
        builder.append(
            "insufficient-finite-close",
            close_event(
                builder.state(),
                coordination::EventFlagFiniteClose));
        Schedule schedule = builder.finish();
        schedule.name =
            "insufficient-finite-close-" + std::to_string(nodes) + "n";
        suite.schedules.push_back(std::move(schedule));
    }
    {
        Builder builder("ready-delay-expiry");
        builder.append("recover-authority", configured_event());
        for (std::size_t node = 0; node != nodes; ++node) {
            builder.append(
                "recover-peer-n" + std::to_string(node),
                recover_peer_event(builder.state(), node, 1, 1));
            if (node < 2) {
                builder.append(
                    "ready-n" + std::to_string(node),
                    ready_event(builder.state(), node, 1, 1));
            }
        }
        builder.append("open-with-delayed-ready", open_event(builder.state()));
        if (nodes > 2) {
            builder.append(
                "delayed-ready-n2",
                ready_event(builder.state(), 2, 1, 1));
            builder.append(
                "expire-delayed-peer",
                expire_event(builder.state(), 2, 1, 1));
        }
        Schedule schedule = builder.finish();
        schedule.name =
            "ready-delay-expiry-" + std::to_string(nodes) + "n";
        suite.schedules.push_back(std::move(schedule));
    }
    if (nodes == 3) {
        Schedule schedule = built_in_job5105811();
        schedule.name = "systematic-job5105811-3n";
        suite.schedules.push_back(std::move(schedule));
    }
    {
        Builder builder("invalid-event");
        builder.append("recover-authority", configured_event());
        builder.append(
            "recover-peer",
            recover_peer_event(builder.state(), 0, 1, 1));
        coordination::Event invalid =
            base_event(coordination::EventKind::NodeApply, builder.state());
        invalid.node = node_key(0);
        invalid.incarnation = incarnation_key(0, 1);
        invalid.sequence = 1;
        invalid.trainer_count = 7;
        builder.append("generation-zero-partial-apply", invalid);
        Schedule schedule = builder.finish();
        schedule.name =
            "invalid-generation-zero-apply-" + std::to_string(nodes) + "n";
        suite.schedules.push_back(std::move(schedule));
    }
    {
        Builder builder = committed_builder("conflicting-commit", nodes);
        coordination::Event identical =
            commit_event(builder.state(), "conflict");
        identical.generation = builder.state().committed_generation;
        identical.exact_tokens = builder.state().accepted_token_clock;
        identical.receipt_digest = builder.state().commit_receipt;
        identical.manifest_digest = builder.state().commit_manifest;
        identical.result_digest = builder.state().committed_result;
        coordination::Event conflict = identical;
        conflict.receipt_digest =
            opaque_digest("conflicting-current-commit-authority");
        builder.append("conflicting-current-commit", conflict);
        Schedule schedule = builder.finish();
        schedule.name =
            "fatal-conflicting-commit-" + std::to_string(nodes) + "n";
        suite.schedules.push_back(std::move(schedule));
    }
}

SystematicSuite build_systematic_suite() {
    SystematicSuite suite;
    for (std::size_t nodes = 2; nodes <= 4; ++nodes) {
        add_pair_schedules(suite, nodes);
        add_three_race_schedules(suite, nodes);
        add_restart_matrix(suite, nodes);
        add_targeted_schedules(suite, nodes);
    }
    return suite;
}

std::optional<std::size_t> node_index_for_key(
        const coordination::Key& key, std::size_t nodes) {
    for (std::size_t node = 0; node != nodes; ++node) {
        if (node_key(node) == key) return node;
    }
    return std::nullopt;
}

Schedule random_schedule(
        std::uint64_t base_seed, std::size_t schedule_index,
        std::size_t maximum_events) {
    const std::uint64_t derived_seed = splitmix64(
        base_seed
        + static_cast<std::uint64_t>(schedule_index)
            * UINT64_C(0x9e3779b97f4a7c15));
    Pcg32 random(
        derived_seed,
        splitmix64(base_seed ^ static_cast<std::uint64_t>(schedule_index)));
    const std::size_t nodes = 2 + random.bounded(3);
    Builder builder(
        "random-seed-" + std::to_string(base_seed)
        + "-index-" + std::to_string(schedule_index));
    builder.append("recover-authority", configured_event());
    std::array<std::uint64_t, 4> incarnation_versions{{1, 1, 1, 1}};
    std::array<std::uint64_t, 4> control_sequences{{1, 1, 1, 1}};
    for (std::size_t node = 0; node != nodes; ++node) {
        builder.append(
            "recover-peer-n" + std::to_string(node),
            recover_peer_event(builder.state(), node, 1, 1));
        if (node < 2 || random.bounded(2) == 0) {
            builder.append(
                "ready-n" + std::to_string(node),
                ready_event(builder.state(), node, 1, 1));
        }
    }
    builder.append("open-generation", open_event(builder.state()));

    const auto append_query = [&]() {
        const std::uint64_t generation = builder.state().committed_generation
            + static_cast<std::uint64_t>(random.bounded(3));
        builder.append(
            "query-commit", query_event(builder.state(), generation));
    };

    for (std::size_t offset = 0; offset != maximum_events; ++offset) {
        const std::size_t operation = random.bounded(18);
        const std::size_t node = random.bounded(nodes);
        const coordination::AuthorityState& state = builder.state();
        switch (operation) {
            case 0:
                append_query();
                break;
            case 1: {
                coordination::Event stale = query_event(
                    state, state.committed_generation);
                if (stale.fence > 1) --stale.fence;
                builder.append("stale-fence-query", stale);
                break;
            }
            case 2:
                builder.append(
                    "ready-race",
                    ready_event(
                        state, node, incarnation_versions[node],
                        control_sequences[node]));
                break;
            case 3:
                builder.append(
                    "expire-race",
                    expire_event(
                        state, node, incarnation_versions[node],
                        control_sequences[node]));
                break;
            case 4:
                ++incarnation_versions[node];
                ++control_sequences[node];
                builder.append(
                    "recover-new-incarnation",
                    recover_peer_event(
                        state, node, incarnation_versions[node],
                        control_sequences[node]));
                break;
            case 5:
                builder.append(
                    "open-race",
                    open_event(
                        state, state.active.present
                            ? state.active.attempt
                                + static_cast<std::uint32_t>(
                                    state.active.phase
                                        == coordination::GenerationPhase::Aborted)
                            : 1));
                break;
            case 6: {
                if (!state.active.present || state.active.cohort.empty()) {
                    append_query();
                    break;
                }
                auto item = state.active.cohort.begin();
                std::advance(
                    item,
                    static_cast<std::ptrdiff_t>(
                        random.bounded(state.active.cohort.size())));
                const auto selected =
                    node_index_for_key(item->first, nodes);
                if (!selected.has_value()) {
                    append_query();
                    break;
                }
                builder.append(
                    "contribution",
                    contribution_event(
                        state, *selected,
                        100 + state.active.generation,
                        1 + random.bounded(17)));
                break;
            }
            case 7: {
                if (state.active.contributions.empty()) {
                    append_query();
                    break;
                }
                auto item = state.active.contributions.begin();
                std::advance(
                    item,
                    static_cast<std::ptrdiff_t>(
                        random.bounded(
                            state.active.contributions.size())));
                const auto selected =
                    node_index_for_key(item->first, nodes);
                if (!selected.has_value()) {
                    append_query();
                    break;
                }
                coordination::Event replay =
                    duplicate_contribution_event(state, *selected);
                if (random.bounded(3) == 0)
                    replay.payload_digest =
                        opaque_digest("random-conflicting-payload-"
                                      + std::to_string(offset));
                builder.append("contribution-replay", replay);
                break;
            }
            case 8: {
                if (!state.active.present) {
                    append_query();
                    break;
                }
                static constexpr std::array<std::uint32_t, 3> flags{{
                    coordination::EventFlagFiniteClose,
                    coordination::EventFlagDeadlineExpired,
                    coordination::EventFlagNone,
                }};
                builder.append(
                    "close-race",
                    close_event(state, flags[random.bounded(flags.size())]));
                break;
            }
            case 9: {
                if (state.active.contributions.empty()) {
                    append_query();
                    break;
                }
                auto item = state.active.contributions.begin();
                std::advance(
                    item,
                    static_cast<std::ptrdiff_t>(
                        random.bounded(
                            state.active.contributions.size())));
                const auto selected =
                    node_index_for_key(item->first, nodes);
                if (!selected.has_value()) {
                    append_query();
                    break;
                }
                builder.append(
                    "result-receipt",
                    result_event(state, *selected));
                break;
            }
            case 10: {
                if (state.active.result_receipts.empty()) {
                    append_query();
                    break;
                }
                auto item = state.active.result_receipts.begin();
                std::advance(
                    item,
                    static_cast<std::ptrdiff_t>(
                        random.bounded(
                            state.active.result_receipts.size())));
                const auto selected =
                    node_index_for_key(item->first, nodes);
                if (!selected.has_value()) {
                    append_query();
                    break;
                }
                coordination::Event replay =
                    duplicate_result_event(state, *selected);
                if (random.bounded(3) == 0)
                    replay.result_digest =
                        opaque_digest("random-conflicting-result-"
                                      + std::to_string(offset));
                builder.append("result-replay", replay);
                break;
            }
            case 11: {
                if (!state.active.present || state.active.cohort.empty()) {
                    append_query();
                    break;
                }
                auto item = state.active.cohort.begin();
                std::advance(
                    item,
                    static_cast<std::ptrdiff_t>(
                        random.bounded(state.active.cohort.size())));
                const auto selected =
                    node_index_for_key(item->first, nodes);
                if (!selected.has_value()) {
                    append_query();
                    break;
                }
                builder.append(
                    "owner-loss",
                    owner_lost_event(
                        state, *selected,
                        1 + static_cast<std::uint64_t>(random.bounded(4))));
                break;
            }
            case 12:
                if (!state.active.present
                    || state.active.contributions.empty()) {
                    append_query();
                } else {
                    coordination::Event commit = commit_event(state);
                    if (random.bounded(5) == 0
                        && commit.exact_tokens > 0)
                        --commit.exact_tokens;
                    builder.append("commit-race", commit);
                }
                break;
            case 13:
                if (state.committed_generation == 0
                    || state.members.empty()) {
                    append_query();
                } else {
                    const std::uint32_t trainers =
                        random.bounded(4) == 0 ? 7U : 8U;
                    builder.append(
                        "node-apply",
                        apply_event(
                            state, node, incarnation_versions[node],
                            control_sequences[node] + 1, trainers,
                            "random-" + std::to_string(offset)));
                    if (trainers == 8)
                        ++control_sequences[node];
                }
                break;
            case 14:
                if (state.committed_generation == 0) {
                    append_query();
                } else {
                    builder.append(
                        "recover-node-apply",
                        recover_node_apply_event(
                            state, node, incarnation_versions[node],
                            opaque_digest(
                                "random-recovered-apply-"
                                + std::to_string(offset))));
                }
                break;
            case 15:
                builder.append(restart_step(
                    "service-restart", "service", false));
                break;
            case 16:
                builder.append(restart_step(
                    "peer-control-restart", "peer-control", true));
                break;
            case 17:
                if (state.active.present
                    && !state.active.contributions.empty()) {
                    auto item = state.active.contributions.begin();
                    const auto selected =
                        node_index_for_key(item->first, nodes);
                    if (selected.has_value()) {
                        coordination::Event stale =
                            duplicate_contribution_event(state, *selected);
                        if (state.committed_generation != 0)
                            stale.generation =
                                state.committed_generation - 1;
                        builder.append("closed-or-stale-contribution", stale);
                        break;
                    }
                }
                append_query();
                break;
        }
    }
    Schedule schedule = builder.finish();
    require_causal_schedule(schedule);
    return schedule;
}

constexpr std::array<std::uint64_t, 4> kRandomSeeds{{
    kJob5105811,
    UINT64_C(0x243f6a8885a308d3),
    UINT64_C(0x9e3779b97f4a7c15),
    UINT64_C(0xd1b54a32d192ed03),
}};

struct FailureCase {
    Schedule schedule;
    Failure failure;
    Mutant mutant = Mutant::None;
};

struct CampaignSummary {
    std::size_t systematic_schedules = 0;
    std::size_t random_schedules = 0;
    std::size_t corpus_schedules = 0;
    std::size_t total_schedules = 0;
    std::size_t transitions = 0;
    std::size_t progress_commits = 0;
    std::array<std::uint64_t, 14> event_counts{};
    std::array<std::uint64_t, 14> disposition_counts{};
    std::array<std::uint64_t, 14> effect_counts{};
    std::set<std::string> pair_ids;
    std::set<std::string> ordered_pair_cases;
    std::set<std::string> three_race_ids;
    std::set<std::string> three_race_permutations;
    std::set<std::string> restart_phase_cases;
    std::size_t known_bad_detected = 0;
    std::size_t known_bad_minimized = 0;
    std::size_t known_bad_original_steps = 0;
    std::size_t known_bad_minimized_steps = 0;
    coordination::Digest campaign_digest{};
};

bool same_summary(
        const CampaignSummary& left, const CampaignSummary& right) {
    return left.systematic_schedules == right.systematic_schedules
        && left.random_schedules == right.random_schedules
        && left.corpus_schedules == right.corpus_schedules
        && left.total_schedules == right.total_schedules
        && left.transitions == right.transitions
        && left.progress_commits == right.progress_commits
        && left.event_counts == right.event_counts
        && left.disposition_counts == right.disposition_counts
        && left.effect_counts == right.effect_counts
        && left.pair_ids == right.pair_ids
        && left.ordered_pair_cases == right.ordered_pair_cases
        && left.three_race_ids == right.three_race_ids
        && left.three_race_permutations
            == right.three_race_permutations
        && left.restart_phase_cases == right.restart_phase_cases
        && left.known_bad_detected == right.known_bad_detected
        && left.known_bad_minimized == right.known_bad_minimized
        && left.known_bad_original_steps
            == right.known_bad_original_steps
        && left.known_bad_minimized_steps
            == right.known_bad_minimized_steps
        && left.campaign_digest == right.campaign_digest;
}

std::vector<std::pair<fs::path, Schedule>> load_corpus(
        const fs::path& corpus_dir) {
    std::vector<fs::path> paths;
    for (const fs::directory_entry& entry :
         fs::directory_iterator(corpus_dir)) {
        if (entry.is_regular_file()
            && entry.path().extension() == ".schedule")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    std::vector<std::pair<fs::path, Schedule>> result;
    for (const fs::path& path : paths)
        result.emplace_back(path, read_schedule(path));
    return result;
}

struct CampaignRun {
    CampaignSummary summary;
    std::optional<FailureCase> failure;
};

void add_evaluation(
        CampaignSummary& summary, emender_ndp::Sha256& campaign_hash,
        const Schedule& schedule, const Evaluation& evaluation) {
    ++summary.total_schedules;
    summary.transitions += evaluation.transitions;
    summary.progress_commits += evaluation.progress_commits;
    for (std::size_t index = 0; index != summary.event_counts.size(); ++index) {
        summary.event_counts[index] += evaluation.event_counts[index];
        summary.disposition_counts[index] +=
            evaluation.disposition_counts[index];
        summary.effect_counts[index] += evaluation.effect_counts[index];
    }
    const std::string canonical = canonical_schedule(schedule);
    const coordination::Digest schedule_digest = hash_bytes(canonical);
    campaign_hash.update(
        schedule_digest.data(), schedule_digest.size());
    campaign_hash.update(
        evaluation.transcript_digest.data(),
        evaluation.transcript_digest.size());
}

std::optional<FailureCase> run_production_schedule(
        CampaignSummary& summary, emender_ndp::Sha256& campaign_hash,
        const Schedule& schedule) {
    const Evaluation evaluation =
        evaluate_schedule(schedule, Mutant::None, false);
    if (evaluation.failure.has_value()) {
        return FailureCase{
            schedule, *evaluation.failure, Mutant::None};
    }
    add_evaluation(summary, campaign_hash, schedule, evaluation);
    return std::nullopt;
}

Mutant mutant_for_expected(const std::string& expected) {
    if (expected == "mutant:stale-noninterference")
        return Mutant::IgnoreStaleFence;
    if (expected == "mutant:no-partial-authority")
        return Mutant::GrantPartialApply;
    return Mutant::None;
}

CampaignRun run_campaign(const Options& options) {
    CampaignRun run;
    emender_ndp::Sha256 campaign_hash;
    const std::string domain =
        "emender-native-coordination-campaign-transcript-v1";
    campaign_hash.update(domain.data(), domain.size());

    const SystematicSuite systematic = build_systematic_suite();
    run.summary.systematic_schedules = systematic.schedules.size();
    run.summary.pair_ids = systematic.pair_ids;
    run.summary.ordered_pair_cases = systematic.ordered_pair_cases;
    run.summary.three_race_ids = systematic.three_race_ids;
    run.summary.three_race_permutations =
        systematic.three_race_permutations;
    run.summary.restart_phase_cases = systematic.restart_phase_cases;
    for (const Schedule& schedule : systematic.schedules) {
        const auto failure = run_production_schedule(
            run.summary, campaign_hash, schedule);
        if (failure.has_value()) {
            run.failure = failure;
            return run;
        }
    }

    const auto corpus = load_corpus(options.corpus_dir);
    run.summary.corpus_schedules = corpus.size();
    bool saw_job5105811 = false;
    for (const auto& item : corpus) {
        const Schedule& schedule = item.second;
        if (schedule.name == "job5105811") saw_job5105811 = true;
        const auto failure = run_production_schedule(
            run.summary, campaign_hash, schedule);
        if (failure.has_value()) {
            run.failure = failure;
            return run;
        }
        const Mutant mutant = mutant_for_expected(schedule.expected);
        if (mutant == Mutant::None) continue;
        const Evaluation mutation =
            evaluate_schedule(schedule, mutant, false);
        const std::string predicate =
            schedule.expected.substr(std::string("mutant:").size());
        if (!mutation.failure.has_value()
            || mutation.failure->predicate != predicate) {
            run.failure = FailureCase{
                schedule,
                Failure{
                    "known-bad-detection",
                    "permanent mutation corpus did not reproduce "
                    + predicate,
                    0},
                mutant};
            return run;
        }
        ++run.summary.known_bad_detected;
        run.summary.known_bad_original_steps += schedule.steps.size();
        const Schedule minimized =
            shrink_failure(schedule, mutant, predicate);
        std::string causal_reason;
        if (!causally_well_formed(minimized, &causal_reason)
            || !retains_predicate(minimized, mutant, predicate)
            || minimized.steps.size() > schedule.steps.size()) {
            run.failure = FailureCase{
                schedule,
                Failure{
                    "causal-shrinking",
                    "known-bad shrink failed causal/predicate preservation: "
                    + causal_reason,
                    0},
                mutant};
            return run;
        }
        ++run.summary.known_bad_minimized;
        run.summary.known_bad_minimized_steps += minimized.steps.size();
        const std::string minimized_bytes = canonical_schedule(minimized);
        const coordination::Digest minimized_digest =
            hash_bytes(minimized_bytes);
        campaign_hash.update(
            minimized_digest.data(), minimized_digest.size());
    }
    if (!saw_job5105811) {
        run.failure = FailureCase{
            built_in_job5105811(),
            Failure{
                "permanent-corpus",
                "job 5105811 is absent from the permanent corpus",
                0},
            Mutant::None};
        return run;
    }

    run.summary.random_schedules = options.random_schedules;
    std::size_t remaining = options.random_schedules;
    for (std::size_t seed_index = 0;
         seed_index != kRandomSeeds.size(); ++seed_index) {
        const std::size_t seeds_left = kRandomSeeds.size() - seed_index;
        const std::size_t count =
            remaining / seeds_left
            + static_cast<std::size_t>(remaining % seeds_left != 0);
        remaining -= count;
        for (std::size_t index = 0; index != count; ++index) {
            Schedule schedule = random_schedule(
                kRandomSeeds[seed_index], index, options.maximum_events);
            const auto failure = run_production_schedule(
                run.summary, campaign_hash, schedule);
            if (failure.has_value()) {
                run.failure = failure;
                return run;
            }
        }
    }
    run.summary.campaign_digest = campaign_hash.finish();
    return run;
}

std::string event_coverage_json(
        const std::array<std::uint64_t, 14>& counts) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (std::uint32_t value = 1; value <= 13; ++value) {
        if (!first) output << ',';
        first = false;
        output << '"' << coordination::event_name(
            static_cast<coordination::EventKind>(value))
               << "\":" << counts[value];
    }
    output << '}';
    return output.str();
}

std::string disposition_coverage_json(
        const std::array<std::uint64_t, 14>& counts) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (std::uint32_t value = 1; value <= 13; ++value) {
        if (!first) output << ',';
        first = false;
        output << '"' << coordination::disposition_name(
            static_cast<coordination::Disposition>(value))
               << "\":" << counts[value];
    }
    output << '}';
    return output.str();
}

std::string effect_coverage_json(
        const std::array<std::uint64_t, 14>& counts) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (std::uint32_t value = 1; value <= 13; ++value) {
        if (!first) output << ',';
        first = false;
        output << '"' << coordination::effect_name(
            static_cast<coordination::EffectKind>(value))
               << "\":" << counts[value];
    }
    output << '}';
    return output.str();
}

template <typename Values>
std::string string_array_json(const Values& values) {
    std::ostringstream output;
    output << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) output << ',';
        first = false;
        output << '"' << json_escape(value) << '"';
    }
    output << ']';
    return output.str();
}

void enforce_coverage(const CampaignSummary& summary) {
    if (summary.total_schedules < 10000)
        throw std::runtime_error(
            "stress campaign executed fewer than tens of thousands schedules");
    for (std::uint32_t value = 1; value <= 13; ++value) {
        if (summary.event_counts[value] == 0) {
            throw std::runtime_error(
                "missing event coverage: "
                + std::string(coordination::event_name(
                    static_cast<coordination::EventKind>(value))));
        }
        if (summary.disposition_counts[value] == 0) {
            throw std::runtime_error(
                "missing disposition coverage: "
                + std::string(coordination::disposition_name(
                    static_cast<coordination::Disposition>(value))));
        }
    }
    if (summary.pair_ids.empty()
        || summary.ordered_pair_cases.size()
            != summary.pair_ids.size() * 2U)
        throw std::runtime_error(
            "pair-order coverage is not complete in both orders");
    if (summary.three_race_ids.size() != 4
        || summary.three_race_permutations.size()
            != summary.three_race_ids.size() * 6U)
        throw std::runtime_error(
            "targeted three-race permutation coverage is incomplete");
    if (summary.restart_phase_cases.size() != 4U * 8U)
        throw std::runtime_error(
            "role/phase restart matrix coverage is incomplete");
    if (summary.known_bad_detected != 2
        || summary.known_bad_minimized != 2)
        throw std::runtime_error(
            "known-bad variants were not detected and minimized");
}

std::vector<std::string> requirement_ids(
        const std::string& prefix, unsigned int first,
        unsigned int last) {
    std::vector<std::string> result;
    for (unsigned int value = first; value <= last; ++value) {
        std::ostringstream item;
        item << prefix << std::setw(2) << std::setfill('0') << value;
        result.push_back(item.str());
    }
    return result;
}

std::string abi_descriptor() {
    std::ostringstream output;
    output << "schema=emender-native-coordination-abi-fingerprint-v1\n"
           << "NDP_COORD_ABI_V1=" << NDP_COORD_ABI_V1 << "\n"
           << "NDP_COORD_MAX_NODES=" << NDP_COORD_MAX_NODES << "\n"
           << "NDP_COORD_MAX_EFFECTS=" << NDP_COORD_MAX_EFFECTS << "\n"
           << "NDP_COORD_TRACE_CAPACITY=" << NDP_COORD_TRACE_CAPACITY << "\n"
           << "sizeof(ndp_coord_event_v1)="
           << sizeof(ndp_coord_event_v1) << "\n"
           << "sizeof(ndp_coord_effect_v1)="
           << sizeof(ndp_coord_effect_v1) << "\n"
           << "sizeof(ndp_coord_member_v1)="
           << sizeof(ndp_coord_member_v1) << "\n"
           << "sizeof(ndp_coord_result_v1)="
           << sizeof(ndp_coord_result_v1) << "\n"
           << "kernel.maximum_nodes=" << coordination::kMaximumNodes << "\n"
           << "kernel.maximum_effects=" << coordination::kMaximumEffects
           << "\n"
           << "kernel.maximum_trace_bytes="
           << coordination::kMaximumTraceBytes << "\n"
           << "kernel.maximum_owner_reassignments="
           << coordination::kMaximumOwnerReassignments << "\n"
           << "kernel.required_trainer_receipts="
           << coordination::kRequiredTrainerReceipts << "\n";
    return output.str();
}

std::vector<fs::path> source_bundle_paths() {
    return {
        "src/native_resilient_dataplane/CMakeLists.txt",
        "src/native_resilient_dataplane/include/emender/ndp.h",
        "src/native_resilient_dataplane/src/coordination_kernel.cpp",
        "src/native_resilient_dataplane/src/coordination_kernel.hpp",
        "src/native_resilient_dataplane/src/ndp.cpp",
        "src/native_resilient_dataplane/src/rpc_protocol.hpp",
        "src/native_resilient_dataplane/src/rpc_server.cpp",
        "src/native_resilient_dataplane/src/service_core.hpp",
        "src/native_resilient_dataplane/tests/"
            "coordination_schedule_stress.cpp",
        "scripts/frontier/run_native_coordination_stress.sh",
    };
}

std::string canonical_command(int argc, char** argv) {
    const auto quote = [](const std::string& value) {
        if (value.find_first_of(" \t\n'\"\\$`") == std::string::npos)
            return value;
        std::string result = "'";
        for (const char item : value) {
            if (item == '\'') result += "'\\''";
            else result.push_back(item);
        }
        result.push_back('\'');
        return result;
    };
    std::ostringstream output;
    for (int index = 0; index != argc; ++index) {
        if (index != 0) output << ' ';
        output << quote(argv[index]);
    }
    return output.str();
}

std::string random_seed_counts_json(std::size_t total) {
    std::ostringstream output;
    output << '[';
    std::size_t remaining = total;
    for (std::size_t seed_index = 0;
         seed_index != kRandomSeeds.size(); ++seed_index) {
        const std::size_t seeds_left = kRandomSeeds.size() - seed_index;
        const std::size_t count =
            remaining / seeds_left
            + static_cast<std::size_t>(remaining % seeds_left != 0);
        remaining -= count;
        if (seed_index != 0) output << ',';
        output << "{\"seed\":" << kRandomSeeds[seed_index]
               << ",\"schedules\":" << count << '}';
    }
    output << ']';
    return output.str();
}

std::string manifest_json(
        const Options& options, const CampaignSummary& summary,
        const coordination::Digest& source_digest,
        const coordination::Digest& binary_digest,
        const coordination::Digest& abi_header_digest,
        const coordination::Digest& abi_descriptor_digest,
        const coordination::Digest& schema_digest,
        const coordination::Digest& corpus_digest,
        const std::string& exact_command) {
    const auto r_ids = requirement_ids("R", 1, 16);
    const auto ndp_ids = requirement_ids("NDP", 1, 17);
    const auto v21s_ids = requirement_ids("V21S", 1, 17);
    const auto isp_ids = requirement_ids("ISP", 1, 7);
    const std::size_t logical_evaluations =
        summary.transitions
        * static_cast<std::size_t>(options.determinism_repeats);

    std::ostringstream output;
    output << "{\n"
           << "  \"schema\":\"" << kManifestSchema << "\",\n"
           << "  \"status\":\"passed\",\n"
           << "  \"task\":\"stress-native-coordinator-schedules\",\n"
           << "  \"authoritative_scope\":\"native-transition-safety-only\",\n"
           << "  \"kernel\":\"" << kKernelSchema << "\",\n"
           << "  \"trace_schema\":\"" << kTraceSchema << "\",\n"
           << "  \"harden_binding\":{\"task\":\"harden-native-coordination-kernel\","
           << "\"commit\":\"" << kHardenCommit << "\","
           << "\"manifest_path\":\"" << kHardenManifestRelative << "\","
           << "\"manifest_sha256\":\"" << kHardenManifestSha256
           << "\"},\n"
           << "  \"identities\":{"
           << "\"source_commit\":\""
           << json_escape(options.source_commit) << "\","
           << "\"source_bundle_sha256\":\""
           << coordination::hex(source_digest) << "\","
           << "\"native_binary_sha256\":\""
           << coordination::hex(binary_digest) << "\","
           << "\"coordination_abi_header_sha256\":\""
           << coordination::hex(abi_header_digest) << "\","
           << "\"coordination_abi_descriptor_sha256\":\""
           << coordination::hex(abi_descriptor_digest) << "\","
           << "\"generator_schema_sha256\":\""
           << coordination::hex(schema_digest) << "\","
           << "\"permanent_corpus_sha256\":\""
           << coordination::hex(corpus_digest) << "\"},\n"
           << "  \"generator\":{\"schema\":\"" << kGeneratorSchema
           << "\",\"prng\":\"" << kPrngSchema
           << "\",\"prng_multiplier\":6364136223846793005,"
           << "\"prng_output\":\"xsh-rr-64-32\","
           << "\"independent_schedule_seed\":\"splitmix64-v1\","
           << "\"grammar\":\"authority (recover-peer ready|delay|expire){2,4} "
              "open generation-event{0,32}\","
           << "\"causal_preconditions\":\"versioned in generator schema; "
              "shrinks must remain causal and reproduce the original predicate\","
           << "\"shrink_order\":\"" << kShrinkSchema << "\"},\n"
           << "  \"bounds\":{\"logical_nodes_min\":2,"
           << "\"logical_nodes_max\":4,"
           << "\"random_generation_events_max\":"
           << options.maximum_events << ','
           << "\"exact_token_type\":\"uint64\","
           << "\"maximum_members\":" << coordination::kMaximumNodes << ','
           << "\"maximum_effects\":" << coordination::kMaximumEffects << ','
           << "\"maximum_trace_bytes\":"
           << coordination::kMaximumTraceBytes << ','
           << "\"maximum_owner_reassignments\":"
           << coordination::kMaximumOwnerReassignments << "},\n"
           << "  \"counts\":{\"systematic_schedules\":"
           << summary.systematic_schedules
           << ",\"random_schedules\":" << summary.random_schedules
           << ",\"permanent_corpus_schedules\":"
           << summary.corpus_schedules
           << ",\"total_schedules\":" << summary.total_schedules
           << ",\"native_transitions\":" << summary.transitions
           << ",\"progress_commits_under_named_predicates\":"
           << summary.progress_commits
           << ",\"pair_classes\":" << summary.pair_ids.size()
           << ",\"ordered_pair_cases\":"
           << summary.ordered_pair_cases.size()
           << ",\"three_race_classes\":"
           << summary.three_race_ids.size()
           << ",\"three_race_permutations\":"
           << summary.three_race_permutations.size()
           << ",\"restart_role_phase_cases\":"
           << summary.restart_phase_cases.size()
           << ",\"known_bad_detected\":"
           << summary.known_bad_detected
           << ",\"known_bad_minimized\":"
           << summary.known_bad_minimized
           << ",\"known_bad_original_steps\":"
           << summary.known_bad_original_steps
           << ",\"known_bad_minimized_steps\":"
           << summary.known_bad_minimized_steps
           << ",\"native_safety_failures\":0},\n"
           << "  \"random_seed_partitions\":"
           << random_seed_counts_json(summary.random_schedules) << ",\n"
           << "  \"coverage\":{\"event_kinds\":"
           << event_coverage_json(summary.event_counts)
           << ",\"dispositions\":"
           << disposition_coverage_json(summary.disposition_counts)
           << ",\"effects\":" << effect_coverage_json(summary.effect_counts)
           << ",\"pair_ids\":" << string_array_json(summary.pair_ids)
           << ",\"ordered_pair_cases\":"
           << string_array_json(summary.ordered_pair_cases)
           << ",\"three_race_ids\":"
           << string_array_json(summary.three_race_ids)
           << ",\"three_race_permutations\":"
           << string_array_json(summary.three_race_permutations)
           << ",\"restart_role_phase_cases\":"
           << string_array_json(summary.restart_phase_cases) << "},\n"
           << "  \"safety_invariants\":["
           << "\"unique-commit\",\"stale-noninterference\","
           << "\"idempotence\",\"immutable-cohort-closure\","
           << "\"no-partial-authority\",\"monotonic-recovery\","
           << "\"bounded-protocol-state\","
           << "\"deterministic-state-digest\"],\n"
           << "  \"progress_predicates\":["
           << "\"q-min-and-t-min\",\"finite-close-or-deadline\","
           << "\"complete-result-delivery\",\"scheduled-commit-fairness\"],\n"
           << "  \"requirements\":{\"R01_R16\":"
           << string_array_json(r_ids)
           << ",\"NDP01_NDP17\":" << string_array_json(ndp_ids)
           << ",\"V21S01_V21S17\":" << string_array_json(v21s_ids)
           << ",\"ISP01_ISP07\":" << string_array_json(isp_ids) << "},\n"
           << "  \"determinism\":{\"campaign_transcript_sha256\":\""
           << coordination::hex(summary.campaign_digest)
           << "\",\"full_campaign_repeats\":"
           << options.determinism_repeats
           << ",\"byte_identical\":true,"
           << "\"non_authoritative_timestamp_fields\":[]},\n"
           << "  \"runtime\":{\"unit\":\"deterministic-native-transition-"
              "evaluations\",\"value\":"
           << logical_evaluations
           << ",\"wall_clock\":\"reported outside this byte-identical "
              "manifest\"},\n"
           << "  \"execution\":{\"exact_command\":\""
           << json_escape(exact_command)
           << "\",\"replay_random_template\":\""
           << json_escape(
                "ndp_coordination_schedule_stress --source-root . "
                "--corpus-dir tests/corpus/native_coordination "
                "--replay-seed <seed> --replay-index <index>")
           << "\",\"replay_corpus_template\":\""
           << json_escape(
                "ndp_coordination_schedule_stress --source-root . "
                "--corpus-dir tests/corpus/native_coordination "
                "--replay-file <minimal.schedule>")
           << "\"},\n"
           << "  \"forbidden_facilities\":{\"slurm_calls\":0,"
           << "\"external_peers\":0,\"model_loads\":0,\"gpu_calls\":0,"
           << "\"libfabric_calls\":0,\"frontier_allocations\":0},\n"
           << "  \"boundary_nonclaims\":["
           << "\"numerical-parity\",\"dense-byte-path\","
           << "\"snapshot-ownership\",\"foreground-timing\","
           << "\"two-node-cxi\",\"frontier-scheduler-evidence\","
           << "\"scale-rung-authorization\"],\n"
           << "  \"scale_gate_use\":\"immutable native-first stress input "
              "for scale-v21-direct-8n; not a substitute for later gates\"\n"
           << "}\n";
    return output.str();
}

void write_text(const fs::path& path, const std::string& value) {
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot write output: " + path.string());
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    if (!output)
        throw std::runtime_error("cannot finish output: " + path.string());
}

std::string sanitized_predicate(const std::string& value) {
    std::string result;
    for (const char item : value) {
        if ((item >= 'a' && item <= 'z')
            || (item >= '0' && item <= '9') || item == '-')
            result.push_back(item);
        else
            result.push_back('-');
    }
    return result;
}

int emit_failure(
        const Options& options, const FailureCase& failure_case,
        const std::string& binary_name) {
    const Schedule minimized = shrink_failure(
        failure_case.schedule, failure_case.mutant,
        failure_case.failure.predicate);
    const coordination::Digest digest =
        hash_bytes(canonical_schedule(minimized));
    const std::string stem =
        "regression-" + sanitized_predicate(failure_case.failure.predicate)
        + "-" + coordination::hex(digest).substr(0, 16);
    const fs::path schedule_path =
        options.failure_dir / (stem + ".schedule");
    const fs::path trace_path =
        options.failure_dir / (stem + ".native-trace.jsonl");
    write_schedule(minimized, schedule_path);
    const Evaluation replay = evaluate_schedule(
        minimized, failure_case.mutant, true);
    write_text(trace_path, replay.transcript);
    std::cerr
        << "native coordination stress failure\n"
        << "predicate: " << failure_case.failure.predicate << "\n"
        << "message: " << failure_case.failure.message << "\n"
        << "original_steps: " << failure_case.schedule.steps.size() << "\n"
        << "minimal_steps: " << minimized.steps.size() << "\n"
        << "promoted_corpus: " << schedule_path.string() << "\n"
        << "native_trace: " << trace_path.string() << "\n"
        << "replay: " << binary_name
        << " --source-root " << options.source_root.string()
        << " --corpus-dir " << options.corpus_dir.string()
        << " --replay-file " << schedule_path.string() << "\n";
    return 2;
}

Options parse_options(int argc, char** argv) {
    Options options;
#ifdef NDP_STRESS_DEFAULT_SOURCE_ROOT
    options.source_root = NDP_STRESS_DEFAULT_SOURCE_ROOT;
#else
    options.source_root = fs::current_path();
#endif
    for (int index = 1; index != argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&](const char* name) -> std::string {
            if (index + 1 >= argc)
                throw std::runtime_error(
                    std::string("missing value for ") + name);
            return argv[++index];
        };
        if (argument == "--source-root") {
            options.source_root = value("--source-root");
        } else if (argument == "--corpus-dir") {
            options.corpus_dir = value("--corpus-dir");
        } else if (argument == "--output") {
            options.output = value("--output");
        } else if (argument == "--failure-dir") {
            options.failure_dir = value("--failure-dir");
        } else if (argument == "--source-commit") {
            options.source_commit = value("--source-commit");
        } else if (argument == "--random-schedules") {
            options.random_schedules =
                static_cast<std::size_t>(
                    std::stoull(value("--random-schedules")));
        } else if (argument == "--maximum-events") {
            options.maximum_events =
                static_cast<std::size_t>(
                    std::stoull(value("--maximum-events")));
        } else if (argument == "--determinism-repeats") {
            options.determinism_repeats =
                static_cast<unsigned int>(
                    std::stoul(value("--determinism-repeats")));
        } else if (argument == "--replay-seed") {
            options.replay_seed =
                std::stoull(value("--replay-seed"));
            options.replay_only = true;
        } else if (argument == "--replay-index") {
            options.replay_index =
                static_cast<std::size_t>(
                    std::stoull(value("--replay-index")));
        } else if (argument == "--replay-file") {
            options.replay_file = value("--replay-file");
            options.replay_only = true;
        } else if (argument == "--no-output") {
            options.output_enabled = false;
        } else if (argument == "--help") {
            std::cout
                << "usage: ndp_coordination_schedule_stress [options]\n"
                << "  --source-root PATH --corpus-dir PATH\n"
                << "  --random-schedules N --maximum-events N\n"
                << "  --determinism-repeats N --source-commit SHA\n"
                << "  --output PATH --failure-dir PATH\n"
                << "  --replay-seed N --replay-index N\n"
                << "  --replay-file PATH\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    options.source_root = fs::absolute(options.source_root).lexically_normal();
    if (options.corpus_dir.empty())
        options.corpus_dir =
            options.source_root / "tests/corpus/native_coordination";
    else if (options.corpus_dir.is_relative())
        options.corpus_dir =
            fs::absolute(options.corpus_dir).lexically_normal();
    if (options.failure_dir.empty())
        options.failure_dir = options.corpus_dir;
    else if (options.failure_dir.is_relative())
        options.failure_dir =
            fs::absolute(options.failure_dir).lexically_normal();
    if (options.output.is_relative() && !options.output.empty())
        options.output = fs::absolute(options.output).lexically_normal();
    if (options.determinism_repeats == 0)
        throw std::runtime_error("determinism repeats must be positive");
    if (options.maximum_events == 0 || options.maximum_events > 128)
        throw std::runtime_error("maximum events must be in [1,128]");
    return options;
}

int replay(const Options& options) {
    Schedule schedule = options.replay_file.empty()
        ? random_schedule(
            options.replay_seed, options.replay_index,
            options.maximum_events)
        : read_schedule(options.replay_file);
    const Mutant mutant = mutant_for_expected(schedule.expected);
    const Evaluation evaluation =
        evaluate_schedule(schedule, mutant, true);
    std::cout << evaluation.transcript;
    if (evaluation.failure.has_value()) {
        std::cerr << "reproduced predicate: "
                  << evaluation.failure->predicate << "\n";
        return mutant == Mutant::None ? 2 : 0;
    }
    std::cerr << "schedule passed: " << schedule.name
              << " transitions=" << evaluation.transitions
              << " final_transcript_sha256="
              << coordination::hex(evaluation.transcript_digest) << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.replay_only) return replay(options);
        if (!fs::is_directory(options.source_root)
            || !fs::is_directory(options.corpus_dir))
            throw std::runtime_error(
                "source root or permanent corpus directory is absent");

        const coordination::Digest harden_manifest_digest =
            hash_file(options.source_root / kHardenManifestRelative);
        if (coordination::hex(harden_manifest_digest)
            != kHardenManifestSha256) {
            throw std::runtime_error(
                "harden-native-coordination-kernel manifest digest drifted");
        }

        CampaignRun first = run_campaign(options);
        if (first.failure.has_value())
            return emit_failure(options, *first.failure, argv[0]);
        enforce_coverage(first.summary);
        for (unsigned int repeat = 1;
             repeat < options.determinism_repeats; ++repeat) {
            CampaignRun repeated = run_campaign(options);
            if (repeated.failure.has_value())
                return emit_failure(options, *repeated.failure, argv[0]);
            if (!same_summary(first.summary, repeated.summary))
                throw std::runtime_error(
                    "identical campaign inputs were not byte-deterministic");
        }

        const coordination::Digest source_digest =
            hash_bundle(options.source_root, source_bundle_paths());
        fs::path executable = fs::canonical("/proc/self/exe");
        const coordination::Digest binary_digest = hash_file(executable);
        const coordination::Digest abi_header_digest = hash_file(
            options.source_root
            / "src/native_resilient_dataplane/include/emender/ndp.h");
        const coordination::Digest abi_descriptor_digest =
            hash_bytes(abi_descriptor());
        const coordination::Digest schema_digest =
            hash_bytes(kSchemaAuthority);
        const coordination::Digest corpus_digest =
            hash_corpus(options.corpus_dir);
        const std::string exact_command =
            canonical_command(argc, argv);
        const std::string manifest = manifest_json(
            options, first.summary, source_digest, binary_digest,
            abi_header_digest, abi_descriptor_digest, schema_digest,
            corpus_digest, exact_command);
        if (options.output_enabled) {
            if (options.output.empty())
                std::cout << manifest;
            else
                write_text(options.output, manifest);
        }
        std::cout
            << "native coordination stress passed"
            << " systematic=" << first.summary.systematic_schedules
            << " random=" << first.summary.random_schedules
            << " corpus=" << first.summary.corpus_schedules
            << " total=" << first.summary.total_schedules
            << " transitions=" << first.summary.transitions
            << " pairs=" << first.summary.pair_ids.size()
            << " ordered_pairs="
            << first.summary.ordered_pair_cases.size()
            << " three_races="
            << first.summary.three_race_permutations.size()
            << " digest="
            << coordination::hex(first.summary.campaign_digest)
            << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native coordination stress error: "
                  << error.what() << "\n";
        return 1;
    }
}
