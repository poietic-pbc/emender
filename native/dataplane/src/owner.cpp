#include "owner.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace emender::ndp {
namespace {

constexpr std::uint64_t kMiB = UINT64_C(1024) * UINT64_C(1024);
constexpr std::string_view kOrderDomain{"emender-ndp-order-v1\0", 21};
constexpr std::string_view kReceiptDomain{"emender-ndp-receipt-v1\0", 23};
constexpr std::string_view kOwnerResultDomain{"emender-ndp-owner-result-v1\0", 28};
constexpr std::string_view kResultDomain{"emender-ndp-result-v1\0", 22};

bool add_checked(std::uint64_t a, std::uint64_t b, std::uint64_t *out) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
  *out = a + b;
  return true;
}

bool multiply_checked(std::uint64_t a, std::uint64_t b, std::uint64_t *out) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) return false;
  *out = a * b;
  return true;
}

void append_u32(std::vector<std::uint8_t> *out, std::uint32_t value) {
  for (unsigned i = 0; i != 4; ++i) out->push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}

void append_u64(std::vector<std::uint8_t> *out, std::uint64_t value) {
  for (unsigned i = 0; i != 8; ++i) out->push_back(static_cast<std::uint8_t>(value >> (8U * i)));
}

template <std::size_t N>
void append(std::vector<std::uint8_t> *out, const std::array<std::uint8_t, N> &value) {
  out->insert(out->end(), value.begin(), value.end());
}

bool same_key(const Key128 &a, const Key128 &b) {
  return std::equal(a.begin(), a.end(), b.begin());
}

bool is_zero_digest(const Digest &digest) {
  return std::all_of(digest.begin(), digest.end(), [](std::uint8_t value) { return value == 0; });
}

}  // namespace

int validate_plan_bounds(const GenerationPlan &plan, std::uint64_t *required_bytes) {
  if (plan.layout_bytes == 0 || plan.layout_bytes > UINT64_C(16) * 1024 * 1024 * 1024 ||
      plan.layout_bytes % 8 != 0 || plan.payload_max == 0 ||
      plan.payload_max > NDP_TRANSPORT_MAX_PAYLOAD || plan.payload_max % 8 != 0 ||
      plan.shard_count == 0 || plan.shard_count > NDP_TRANSPORT_MAX_SHARDS ||
      plan.accepted.empty() || plan.accepted.size() > NDP_TRANSPORT_MAX_CONTRIBUTIONS ||
      plan.tx_slots == 0 || plan.tx_slots > 16 || plan.rx_slots == 0 ||
      plan.rx_slots > 16 || plan.owner_count == 0 ||
      plan.owner_count > NDP_TRANSPORT_MAX_CONTRIBUTIONS ||
      plan.assigned_bytes > plan.layout_bytes || plan.deadline_unix_ns == 0) {
    return NDP_T_EBOUNDS;
  }
  const std::uint64_t expected_shards =
      (plan.layout_bytes + plan.payload_max - 1) / plan.payload_max;
  if (expected_shards != plan.shard_count) return NDP_T_EBOUNDS;
  std::set<std::uint32_t> assigned;
  std::uint64_t observed_assigned_bytes = 0;
  for (const auto shard : plan.assigned_shards) {
    if (shard >= plan.shard_count || !assigned.insert(shard).second) return NDP_T_EBOUNDS;
    const std::uint64_t offset = static_cast<std::uint64_t>(shard) * plan.payload_max;
    const std::uint64_t bytes = std::min(plan.payload_max, plan.layout_bytes - offset);
    if (!add_checked(observed_assigned_bytes, bytes, &observed_assigned_bytes)) return NDP_T_EBOUNDS;
  }
  const std::uint64_t max_owner_bytes =
      (plan.layout_bytes + plan.owner_count - 1) / plan.owner_count + plan.payload_max;
  if (observed_assigned_bytes != plan.assigned_bytes || plan.assigned_bytes > max_owner_bytes ||
      (plan.layout_bytes == UINT64_C(5506770496) && plan.owner_count < 2)) return NDP_T_EBOUNDS;
  std::uint64_t total_weight = 0;
  std::set<Digest> contribution_ids;
  for (const auto &contribution : plan.accepted) {
    if (contribution.weight == 0 || contribution.weight > ((UINT64_C(1) << 53U) - 1) ||
        is_zero_digest(contribution.contribution_digest) ||
        !contribution_ids.insert(contribution.contribution_digest).second ||
        !add_checked(total_weight, contribution.weight, &total_weight) ||
        total_weight >= (UINT64_C(1) << 63U)) return NDP_T_EBOUNDS;
  }

  std::uint64_t two_layouts = 0, slots = 0, slots_bytes = 0, ledger = 0;
  if (!multiply_checked(plan.layout_bytes, 2, &two_layouts) ||
      !add_checked(plan.payload_max, kHeaderBytes, &slots_bytes) ||
      !multiply_checked(static_cast<std::uint64_t>(plan.tx_slots + plan.rx_slots),
                        slots_bytes, &slots) ||
      !multiply_checked(static_cast<std::uint64_t>(plan.accepted.size()),
                        plan.shard_count, &ledger) ||
      !multiply_checked(ledger, 128, &ledger)) return NDP_T_EBOUNDS;
  std::uint64_t required = two_layouts;
  if (!add_checked(required, plan.assigned_bytes, &required) ||
      !add_checked(required, slots, &required) ||
      !add_checked(required, ledger, &required) ||
      !add_checked(required, 64 * kMiB, &required)) return NDP_T_EBOUNDS;
  if (required > plan.resident_limit_bytes) return NDP_T_EBOUNDS;
  if (required_bytes != nullptr) *required_bytes = required;
  return NDP_T_OK;
}

std::vector<Contribution> contribution_order(const GenerationPlan &plan,
                                             std::uint32_t shard_id) {
  std::vector<Contribution> ordered = plan.accepted;
  std::sort(ordered.begin(), ordered.end(), [&](const Contribution &a, const Contribution &b) {
    std::vector<std::uint8_t> a_input, b_input;
    a_input.reserve(68); b_input.reserve(68);
    append(&a_input, plan.layout_digest); append_u32(&a_input, shard_id);
    append(&a_input, a.contribution_digest);
    append(&b_input, plan.layout_digest); append_u32(&b_input, shard_id);
    append(&b_input, b.contribution_digest);
    const auto a_order = sha256_domain(kOrderDomain, a_input.data(), a_input.size());
    const auto b_order = sha256_domain(kOrderDomain, b_input.data(), b_input.size());
    return std::tie(a_order, a.contribution_digest) <
           std::tie(b_order, b.contribution_digest);
  });
  return ordered;
}

std::vector<std::size_t> deterministic_owner_map(
    const Key128 &run_key, std::uint64_t fence_epoch,
    std::uint64_t generation, std::uint32_t attempt,
    std::uint64_t owner_epoch, std::size_t owner_count,
    std::size_t shard_count) {
  if (owner_count == 0) return {};
  std::vector<std::uint8_t> input;
  append(&input, run_key); append_u64(&input, fence_epoch);
  append_u64(&input, generation); append_u32(&input, attempt);
  append_u64(&input, owner_epoch);
  const Digest digest = sha256(input.data(), input.size());
  std::uint64_t h = 0;
  for (unsigned i = 0; i != 8; ++i) h |= static_cast<std::uint64_t>(digest[i]) << (8U * i);
  std::vector<std::size_t> result(shard_count);
  const auto start = static_cast<std::size_t>(h % owner_count);
  for (std::size_t shard = 0; shard != shard_count; ++shard) {
    result[shard] = (start + shard) % owner_count;
  }
  return result;
}

int compute_owner_result_root(
    const GenerationPlan &plan,
    const std::map<std::uint32_t, std::vector<std::uint8_t>> &shards,
    Digest *out) {
  if (out == nullptr || shards.size() != plan.assigned_shards.size()) return NDP_T_EINVAL;
  std::vector<std::uint8_t> encoded;
  append(&encoded, plan.run_key); append_u64(&encoded, plan.fence_epoch);
  append_u64(&encoded, plan.generation); append_u32(&encoded, plan.attempt);
  append_u64(&encoded, plan.owner_epoch); append(&encoded, plan.owner_worker_key);
  append(&encoded, plan.owner_incarnation);
  std::set<std::uint32_t> expected(plan.assigned_shards.begin(), plan.assigned_shards.end());
  for (const auto &[shard_id, payload] : shards) {
    if (expected.erase(shard_id) != 1 || shard_id >= plan.shard_count) return NDP_T_EBOUNDS;
    const std::uint64_t offset = static_cast<std::uint64_t>(shard_id) * plan.payload_max;
    const std::uint64_t bytes = std::min(plan.payload_max, plan.layout_bytes - offset);
    if (payload.size() != bytes) return NDP_T_EBOUNDS;
    append_u32(&encoded, shard_id); append_u64(&encoded, bytes);
    append(&encoded, sha256(payload.data(), payload.size()));
  }
  if (!expected.empty()) return NDP_T_EBOUNDS;
  *out = sha256_domain(kOwnerResultDomain, encoded.data(), encoded.size());
  return NDP_T_OK;
}

int compute_result_root(
    const GenerationPlan &plan, std::uint64_t global_weight,
    const std::map<std::uint32_t, std::vector<std::uint8_t>> &shards,
    Digest *out) {
  if (out == nullptr || global_weight == 0 || global_weight >= (UINT64_C(1) << 63U) ||
      shards.size() != plan.shard_count) return NDP_T_EINVAL;
  std::vector<std::uint8_t> encoded;
  append(&encoded, plan.run_key); append_u64(&encoded, plan.fence_epoch);
  append_u64(&encoded, plan.generation); append_u32(&encoded, plan.attempt);
  append_u64(&encoded, plan.owner_epoch); append(&encoded, plan.layout_digest);
  append(&encoded, plan.base_digest); append_u64(&encoded, global_weight);
  for (std::uint32_t shard_id = 0; shard_id != plan.shard_count; ++shard_id) {
    const auto found = shards.find(shard_id);
    if (found == shards.end()) return NDP_T_EBOUNDS;
    const std::uint64_t offset = static_cast<std::uint64_t>(shard_id) * plan.payload_max;
    const std::uint64_t bytes = std::min(plan.payload_max, plan.layout_bytes - offset);
    if (found->second.size() != bytes) return NDP_T_EBOUNDS;
    append_u32(&encoded, shard_id); append_u64(&encoded, bytes);
    append(&encoded, sha256(found->second.data(), found->second.size()));
  }
  *out = sha256_domain(kResultDomain, encoded.data(), encoded.size());
  return NDP_T_OK;
}

OwnerEngine::OwnerEngine(GenerationPlan plan) : plan_(std::move(plan)) {
  static_assert(std::numeric_limits<double>::is_iec559,
                "v1 requires IEC 60559 binary64");
  static_assert(sizeof(LedgerEntry) <= 128,
                "receipt ledger entry exceeds the admitted v1 bound");
  const std::uint16_t endian_probe = 1;
  if (*reinterpret_cast<const std::uint8_t *>(&endian_probe) != 1 ||
      std::fesetround(FE_TONEAREST) != 0) {
    throw std::runtime_error("v1 requires little-endian FE_TONEAREST binary64");
  }
  std::uint64_t required = 0;
  if (validate_plan_bounds(plan_, &required) != NDP_T_OK) {
    throw std::invalid_argument("generation plan exceeds v1 bounds");
  }
  for (const auto shard_id : plan_.assigned_shards) {
    ShardState shard{};
    shard.order = contribution_order(plan_, shard_id);
    const std::uint64_t offset = static_cast<std::uint64_t>(shard_id) * plan_.payload_max;
    const std::uint64_t bytes = std::min(plan_.payload_max, plan_.layout_bytes - offset);
    shard.accumulator.assign(static_cast<std::size_t>(bytes), 0);
    shard.ledger.resize(shard.order.size());
    shards_.emplace(shard_id, std::move(shard));
  }
  (void)required;
}

int OwnerEngine::freeze() {
  if (state_ != 0) return NDP_T_ESTATE;
  if (unix_time_ns() >= plan_.deadline_unix_ns) return NDP_T_EDEADLINE;
  state_ = 1;
  return NDP_T_OK;
}

int OwnerEngine::grant_next(std::uint32_t shard_id, std::uint64_t credit_epoch,
                            FrameHeader *credit) {
  if (state_ != 1 || credit == nullptr) return NDP_T_ESTATE;
  if (unix_time_ns() >= plan_.deadline_unix_ns) return NDP_T_EDEADLINE;
  auto found = shards_.find(shard_id);
  if (found == shards_.end()) return NDP_T_EINVAL;
  auto &shard = found->second;
  if (shard.next >= shard.order.size()) return NDP_T_ESTATE;
  if (credit_epoch <= shard.credit_epoch) return NDP_T_ESTALE;
  shard.credit_epoch = credit_epoch;
  shard.credit_bytes = std::min(plan_.payload_max,
      plan_.layout_bytes - static_cast<std::uint64_t>(shard_id) * plan_.payload_max);
  shard.credited_contribution = shard.order[shard.next].contribution_digest;
  metrics_.credit_high_water = std::max(metrics_.credit_high_water, shard.credit_bytes);
  FrameHeader out{};
  out.type = MessageType::credit;
  out.run_key = plan_.run_key; out.fence_epoch = plan_.fence_epoch;
  out.generation = plan_.generation; out.attempt = plan_.attempt;
  out.shard_id = shard_id; out.owner_epoch = plan_.owner_epoch;
  out.layout_digest = plan_.layout_digest; out.base_digest = plan_.base_digest;
  out.contribution_digest = shard.credited_contribution;
  out.payload_offset = static_cast<std::uint64_t>(shard_id) * plan_.payload_max;
  out.payload_bytes = shard.credit_bytes; out.shard_bytes = shard.credit_bytes;
  out.message_seq = credit_epoch; out.deadline_unix_ns = plan_.deadline_unix_ns;
  out.credit_bytes = shard.credit_bytes; out.chunk_index = shard_id;
  out.chunk_count = plan_.shard_count;
  *credit = out;
  return NDP_T_OK;
}

bool OwnerEngine::identity_matches(const FrameHeader &h, WireReason *reason) const {
  if (!same_key(h.run_key, plan_.run_key) || h.fence_epoch != plan_.fence_epoch) {
    *reason = WireReason::stale_fence; return false;
  }
  if (h.generation != plan_.generation || h.attempt != plan_.attempt) {
    *reason = WireReason::stale_generation_or_attempt; return false;
  }
  if (h.owner_epoch != plan_.owner_epoch) {
    *reason = WireReason::stale_owner_epoch; return false;
  }
  if (!constant_time_equal(h.layout_digest, plan_.layout_digest) ||
      !constant_time_equal(h.base_digest, plan_.base_digest)) {
    *reason = WireReason::layout_or_base; return false;
  }
  return true;
}

Receipt OwnerEngine::make_receipt(const FrameHeader &h, WireStatus status,
                                  WireReason reason, std::uint32_t ordinal) const {
  std::vector<std::uint8_t> encoded;
  append(&encoded, h.run_key); append_u64(&encoded, h.fence_epoch);
  append_u64(&encoded, h.generation); append_u32(&encoded, h.attempt);
  append_u64(&encoded, h.owner_epoch); append(&encoded, h.contribution_digest);
  append_u64(&encoded, h.contribution_seq); append_u32(&encoded, h.shard_id);
  append_u64(&encoded, h.payload_offset); append_u64(&encoded, h.payload_bytes);
  append(&encoded, h.payload_digest); append_u64(&encoded, h.weight);
  // The transport owner identity is bound by the installed endpoint record in
  // the service. The pure engine uses the plan identity as the receiver key so
  // its golden receipts remain deterministic.
  append(&encoded, plan_.owner_worker_key); append(&encoded, plan_.owner_incarnation);
  append_u64(&encoded, plan_.owner_endpoint_epoch); append_u32(&encoded, ordinal);
  append_u32(&encoded, static_cast<std::uint32_t>(status));
  append_u32(&encoded, static_cast<std::uint32_t>(reason));
  Receipt receipt{};
  receipt.status = status; receipt.reason = reason; receipt.application_ordinal = ordinal;
  receipt.receipt_digest = sha256_domain(kReceiptDomain, encoded.data(), encoded.size());
  receipt.header = h;
  receipt.header.type = MessageType::receipt;
  receipt.header.status = status; receipt.header.reason = reason;
  receipt.header.payload_digest = h.payload_digest;
  return receipt;
}

Receipt OwnerEngine::reject(const FrameHeader &header, WireReason reason) {
  if (reason == WireReason::stale_fence ||
      reason == WireReason::stale_generation_or_attempt ||
      reason == WireReason::stale_owner_epoch) ++metrics_.stale_rejects;
  else if (reason == WireReason::checksum) ++metrics_.checksum_rejects;
  else if (reason == WireReason::nonfinite) ++metrics_.nonfinite_rejects;
  else if (reason == WireReason::no_credit) ++metrics_.no_credit_rejects;
  else if (reason == WireReason::conflict) ++metrics_.conflicts;
  return make_receipt(header,
      reason == WireReason::no_credit ? WireStatus::retryable : WireStatus::rejected,
      reason, 0);
}

Receipt OwnerEngine::apply(const DecodedFrame &frame, std::uint64_t now_unix_ns) {
  const auto &h = frame.header;
  if (state_ != 1) return reject(h, state_ == 3 ? WireReason::shutdown : WireReason::not_accepted);
  if (h.type != MessageType::contribution_data) return reject(h, WireReason::not_accepted);
  WireReason identity_reason = WireReason::none;
  if (!identity_matches(h, &identity_reason)) return reject(h, identity_reason);
  if (now_unix_ns >= plan_.deadline_unix_ns || now_unix_ns >= h.deadline_unix_ns) {
    return reject(h, WireReason::deadline);
  }
  auto found = shards_.find(h.shard_id);
  if (found == shards_.end()) return reject(h, WireReason::not_accepted);
  auto &shard = found->second;
  const std::uint64_t expected_offset = static_cast<std::uint64_t>(h.shard_id) * plan_.payload_max;
  const std::uint64_t expected_bytes = std::min(plan_.payload_max, plan_.layout_bytes - expected_offset);
  if (h.chunk_index != h.shard_id || h.chunk_count != plan_.shard_count ||
      h.payload_offset != expected_offset || h.payload_bytes != expected_bytes ||
      h.shard_bytes != expected_bytes || frame.payload.size() != expected_bytes ||
      expected_bytes % 8 != 0) return reject(h, WireReason::byte_bounds);

  const auto accepted = std::find_if(shard.order.begin(), shard.order.end(), [&](const Contribution &c) {
    return constant_time_equal(c.contribution_digest, h.contribution_digest);
  });
  if (accepted == shard.order.end()) return reject(h, WireReason::not_accepted);
  const std::size_t accepted_index = static_cast<std::size_t>(
      std::distance(shard.order.begin(), accepted));
  if (shard.ledger[accepted_index].has_value()) {
    const auto &entry = *shard.ledger[accepted_index];
    if (entry.payload_offset == h.payload_offset && entry.payload_bytes == h.payload_bytes &&
        entry.weight == h.weight && entry.owner_epoch == h.owner_epoch &&
        constant_time_equal(entry.payload_digest, h.payload_digest)) {
      ++metrics_.duplicate_frames;
      Receipt duplicate{};
      duplicate.status = entry.status; duplicate.reason = entry.reason;
      duplicate.application_ordinal = entry.application_ordinal;
      duplicate.receipt_digest = entry.receipt_digest;
      duplicate.header = h; duplicate.header.type = MessageType::receipt;
      duplicate.header.status = entry.status; duplicate.header.reason = entry.reason;
      return duplicate;
    }
    return reject(h, WireReason::conflict);
  }
  if (shard.next >= shard.order.size()) return reject(h, WireReason::not_accepted);
  const Contribution &expected = shard.order[shard.next];
  if (!constant_time_equal(expected.contribution_digest, h.contribution_digest) ||
      !same_key(expected.worker_key, h.worker_key) ||
      !same_key(expected.incarnation, h.incarnation) ||
      expected.contribution_seq != h.contribution_seq || expected.weight != h.weight) {
    return reject(h, WireReason::no_credit);
  }
  if (shard.credit_bytes < h.payload_bytes ||
      !constant_time_equal(shard.credited_contribution, h.contribution_digest)) {
    return reject(h, WireReason::no_credit);
  }
  if (!frame.payload_validated) {
    const Digest observed_digest = sha256(frame.payload.data(), frame.payload.size());
    if (!constant_time_equal(observed_digest, h.payload_digest)) {
      return reject(h, WireReason::checksum);
    }
  }

  const auto *raw = frame.payload.data();
  const std::size_t elements = shard.accumulator.size() / sizeof(double);
  for (std::size_t i = 0; i != elements; ++i) {
    double value = 0.0;
    std::memcpy(&value, raw + i * sizeof(double), sizeof(double));
    if (!std::isfinite(value)) return reject(h, WireReason::nonfinite);
    double current = 0.0;
    std::memcpy(&current, shard.accumulator.data() + i * sizeof(double), sizeof(double));
    const double sum = current + value;
    if (!std::isfinite(sum)) return reject(h, WireReason::nonfinite);
  }
  for (std::size_t i = 0; i != elements; ++i) {
    double value = 0.0;
    std::memcpy(&value, raw + i * sizeof(double), sizeof(double));
    double current = 0.0;
    std::memcpy(&current, shard.accumulator.data() + i * sizeof(double), sizeof(double));
    current += value;
    std::memcpy(shard.accumulator.data() + i * sizeof(double), &current, sizeof(double));
  }
  const std::uint32_t ordinal = static_cast<std::uint32_t>(shard.next);
  Receipt receipt = make_receipt(h, WireStatus::applied, WireReason::none, ordinal);
  LedgerEntry entry{};
  entry.payload_offset = h.payload_offset; entry.payload_bytes = h.payload_bytes;
  entry.payload_digest = h.payload_digest; entry.weight = h.weight;
  entry.owner_epoch = h.owner_epoch; entry.status = receipt.status;
  entry.reason = receipt.reason; entry.application_ordinal = receipt.application_ordinal;
  entry.receipt_digest = receipt.receipt_digest;
  shard.ledger[accepted_index] = entry;
  ++shard.next;
  shard.credit_bytes = 0;
  shard.credited_contribution.fill(0);
  metrics_.useful_bytes += h.payload_bytes;
  metrics_.wire_bytes += h.payload_bytes + kHeaderBytes;
  ++metrics_.receipt_count;
  return receipt;
}

int OwnerEngine::finalize(std::uint64_t now_unix_ns) {
  if (state_ != 1) return NDP_T_ESTATE;
  if (now_unix_ns >= plan_.deadline_unix_ns) return NDP_T_EDEADLINE;
  std::uint64_t total_weight = 0;
  for (const auto &contribution : plan_.accepted) {
    if (!add_checked(total_weight, contribution.weight, &total_weight)) return NDP_T_EBOUNDS;
  }
  for (auto &[shard_id, shard] : shards_) {
    (void)shard_id;
    if (shard.next != shard.order.size()) return NDP_T_ESTATE;
    const double divisor = static_cast<double>(total_weight);
    const std::size_t elements = shard.accumulator.size() / sizeof(double);
    for (std::size_t i = 0; i != elements; ++i) {
      double numerator = 0.0;
      std::memcpy(&numerator, shard.accumulator.data() + i * sizeof(double), sizeof(double));
      const double value = numerator / divisor;
      if (!std::isfinite(value)) return NDP_T_ENONFINITE;
      std::memcpy(shard.accumulator.data() + i * sizeof(double), &value, sizeof(double));
    }
    metrics_.retained_bytes += shard.accumulator.size();
  }
  metrics_.retained_high_water = std::max(metrics_.retained_high_water, metrics_.retained_bytes);
  state_ = 2;
  return NDP_T_OK;
}

int OwnerEngine::reassign(std::uint64_t new_owner_epoch) {
  if (unix_time_ns() >= plan_.deadline_unix_ns) return NDP_T_EDEADLINE;
  if (state_ == 3 || new_owner_epoch <= plan_.owner_epoch || metrics_.reassignment_count >= 2) {
    return metrics_.reassignment_count >= 2 ? NDP_T_EBOUNDS : NDP_T_ESTALE;
  }
  plan_.owner_epoch = new_owner_epoch;
  ++metrics_.reassignment_count;
  metrics_.released_bytes += metrics_.retained_bytes;
  metrics_.retained_bytes = 0;
  for (auto &[id, shard] : shards_) {
    (void)id;
    shard.next = 0; shard.credit_epoch = 0; shard.credit_bytes = 0;
    shard.credited_contribution.fill(0);
    std::fill(shard.accumulator.begin(), shard.accumulator.end(), 0);
    std::fill(shard.ledger.begin(), shard.ledger.end(), std::nullopt);
  }
  state_ = 1;
  return NDP_T_OK;
}

void OwnerEngine::cancel() {
  if (state_ == 3) return;
  metrics_.released_bytes += metrics_.retained_bytes;
  metrics_.retained_bytes = 0;
  for (auto &[id, shard] : shards_) {
    (void)id;
    shard.accumulator.clear(); shard.ledger.clear();
  }
  state_ = 3;
}

const std::vector<std::uint8_t> *OwnerEngine::result(std::uint32_t shard_id) const {
  const auto found = shards_.find(shard_id);
  if (state_ != 2 || found == shards_.end()) return nullptr;
  return &found->second.accumulator;
}

int OwnerEngine::owner_result_root(Digest *out) const {
  if (state_ != 2) return NDP_T_ESTATE;
  if (out == nullptr) return NDP_T_EINVAL;
  std::vector<std::uint8_t> encoded;
  encoded.reserve(76 + shards_.size() * 44);
  append(&encoded, plan_.run_key); append_u64(&encoded, plan_.fence_epoch);
  append_u64(&encoded, plan_.generation); append_u32(&encoded, plan_.attempt);
  append_u64(&encoded, plan_.owner_epoch); append(&encoded, plan_.owner_worker_key);
  append(&encoded, plan_.owner_incarnation);
  for (const auto &[shard_id, shard] : shards_) {
    append_u32(&encoded, shard_id);
    append_u64(&encoded, shard.accumulator.size());
    append(&encoded, sha256(shard.accumulator.data(), shard.accumulator.size()));
  }
  *out = sha256_domain(kOwnerResultDomain, encoded.data(), encoded.size());
  return NDP_T_OK;
}

ReplayBuffer::ReplayBuffer(std::uint64_t layout_bytes, std::uint64_t deadline_unix_ns)
    : layout_bytes_(layout_bytes), deadline_unix_ns_(deadline_unix_ns) {
  if (layout_bytes == 0 || layout_bytes > UINT64_C(16) * 1024 * 1024 * 1024 ||
      deadline_unix_ns == 0) throw std::invalid_argument("invalid replay bound");
}

int ReplayBuffer::retain(std::uint32_t shard_id, std::vector<std::uint8_t> frame) {
  if (unix_time_ns() >= deadline_unix_ns_) return NDP_T_EDEADLINE;
  if (frames_.count(shard_id) != 0) return NDP_T_ECONFLICT;
  if (frame.size() > layout_bytes_ - std::min(layout_bytes_, metrics_.retained_bytes)) return NDP_T_EBOUNDS;
  metrics_.retained_bytes += frame.size();
  metrics_.retained_high_water = std::max(metrics_.retained_high_water, metrics_.retained_bytes);
  frames_.emplace(shard_id, std::move(frame));
  return NDP_T_OK;
}

int ReplayBuffer::acknowledge(std::uint32_t shard_id) {
  const auto found = frames_.find(shard_id);
  if (found == frames_.end()) return NDP_T_ESTALE;
  metrics_.retained_bytes -= found->second.size();
  metrics_.released_bytes += found->second.size();
  frames_.erase(found);
  return NDP_T_OK;
}

int ReplayBuffer::begin_reassignment(std::uint64_t now_unix_ns) {
  if (now_unix_ns >= deadline_unix_ns_) return NDP_T_EDEADLINE;
  if (metrics_.reassignment_count >= 2) return NDP_T_EBOUNDS;
  if (metrics_.retained_bytes > UINT64_C(2) * layout_bytes_ - metrics_.replay_bytes) return NDP_T_EBOUNDS;
  metrics_.replay_bytes += metrics_.retained_bytes;
  ++metrics_.reassignment_count;
  return NDP_T_OK;
}

const std::vector<std::uint8_t> *ReplayBuffer::frame(std::uint32_t shard_id) const {
  const auto found = frames_.find(shard_id);
  return found == frames_.end() ? nullptr : &found->second;
}

void ReplayBuffer::cancel() {
  metrics_.released_bytes += metrics_.retained_bytes;
  metrics_.retained_bytes = 0;
  frames_.clear();
}

ResultAssembler::ResultAssembler(GenerationPlan plan)
    : plan_(std::move(plan)), payload_max_(plan_.payload_max),
      deadline_unix_ns_(plan_.deadline_unix_ns) {
  std::uint64_t required_bytes = 0;
  if (validate_plan_bounds(plan_, &required_bytes) != NDP_T_OK ||
      plan_.layout_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("invalid result assembler bounds");
  }
  for (const auto &contribution : plan_.accepted) {
    if (contribution.weight == 0 ||
        !add_checked(global_weight_, contribution.weight, &global_weight_) ||
        global_weight_ >= (UINT64_C(1) << 63U)) {
      throw std::invalid_argument("invalid result assembler global weight");
    }
  }
  aggregate_.resize(static_cast<std::size_t>(plan_.layout_bytes));
  received_.resize(plan_.shard_count, false);
  digests_.resize(plan_.shard_count);
}

int ResultAssembler::accept(const DecodedFrame &frame, std::uint64_t now_unix_ns) {
  const auto &h = frame.header;
  if (h.type != MessageType::result_data) return NDP_T_EINVAL;
  if (now_unix_ns >= deadline_unix_ns_ || now_unix_ns >= h.deadline_unix_ns) {
    return NDP_T_EDEADLINE;
  }
  if (!same_key(h.run_key, plan_.run_key) || h.fence_epoch != plan_.fence_epoch) {
    return NDP_T_EFENCE;
  }
  if (h.generation != plan_.generation || h.attempt != plan_.attempt ||
      h.owner_epoch != plan_.owner_epoch ||
      !constant_time_equal(h.layout_digest, plan_.layout_digest) ||
      !constant_time_equal(h.base_digest, plan_.base_digest)) {
    return NDP_T_ESTALE;
  }
  if (h.weight != global_weight_) return NDP_T_ECONFLICT;
  if (h.shard_id >= received_.size()) return NDP_T_EBOUNDS;
  const std::uint64_t offset = static_cast<std::uint64_t>(h.shard_id) * payload_max_;
  const std::uint64_t bytes = std::min(payload_max_, aggregate_.size() - offset);
  if (h.chunk_index != h.shard_id || h.chunk_count != plan_.shard_count ||
      h.payload_offset != offset || h.payload_bytes != bytes ||
      h.shard_bytes != bytes || frame.payload.size() != bytes) return NDP_T_EBOUNDS;
  if (!frame.payload_validated &&
      !constant_time_equal(sha256(frame.payload.data(), frame.payload.size()),
                           h.payload_digest)) return NDP_T_ECHECKSUM;
  if (received_[h.shard_id]) {
    return constant_time_equal(digests_[h.shard_id], h.payload_digest)
        ? NDP_T_OK : NDP_T_ECONFLICT;
  }
  std::copy(frame.payload.begin(), frame.payload.end(), aggregate_.begin() + static_cast<std::ptrdiff_t>(offset));
  received_[h.shard_id] = true;
  digests_[h.shard_id] = h.payload_digest;
  return NDP_T_OK;
}

bool ResultAssembler::complete() const noexcept {
  return std::all_of(received_.begin(), received_.end(), [](bool value) { return value; });
}

void ResultAssembler::release() {
  released_bytes_ += aggregate_.size();
  aggregate_.clear(); aggregate_.shrink_to_fit(); received_.clear(); digests_.clear();
}

}  // namespace emender::ndp
