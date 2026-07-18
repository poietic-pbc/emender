#ifndef EMENDER_NDP_OWNER_HPP
#define EMENDER_NDP_OWNER_HPP

#include "protocol.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace emender::ndp {

struct Contribution {
  Key128 worker_key{};
  Key128 incarnation{};
  std::uint64_t contribution_seq{0};
  Digest contribution_digest{};
  std::uint64_t weight{0};
};

struct GenerationPlan {
  Key128 run_key{};
  std::uint64_t fence_epoch{0};
  std::uint64_t generation{0};
  std::uint32_t attempt{0};
  std::uint64_t owner_epoch{0};
  Digest layout_digest{};
  Digest base_digest{};
  Key128 owner_worker_key{};
  Key128 owner_incarnation{};
  std::uint64_t owner_endpoint_epoch{0};
  std::uint64_t layout_bytes{0};
  std::uint64_t payload_max{0};
  std::uint32_t shard_count{0};
  std::uint32_t tx_slots{0};
  std::uint32_t rx_slots{0};
  std::uint32_t owner_count{0};
  std::uint64_t assigned_bytes{0};
  std::uint64_t resident_limit_bytes{0};
  std::uint64_t deadline_unix_ns{0};
  std::vector<std::uint32_t> assigned_shards;
  std::vector<Contribution> accepted;
};

struct OwnerMetrics {
  std::uint64_t useful_bytes{0};
  std::uint64_t wire_bytes{0};
  std::uint64_t duplicate_frames{0};
  std::uint64_t conflicts{0};
  std::uint64_t checksum_rejects{0};
  std::uint64_t nonfinite_rejects{0};
  std::uint64_t stale_rejects{0};
  std::uint64_t no_credit_rejects{0};
  std::uint64_t retained_bytes{0};
  std::uint64_t retained_high_water{0};
  std::uint64_t released_bytes{0};
  std::uint64_t replay_bytes{0};
  std::uint64_t receipt_count{0};
  std::uint64_t credit_high_water{0};
  std::uint32_t reassignment_count{0};
};

struct Receipt {
  WireStatus status{WireStatus::rejected};
  WireReason reason{WireReason::none};
  std::uint32_t application_ordinal{0};
  Digest receipt_digest{};
  FrameHeader header{};
};

int validate_plan_bounds(const GenerationPlan &plan, std::uint64_t *required_bytes);
std::vector<Contribution> contribution_order(const GenerationPlan &plan,
                                             std::uint32_t shard_id);
std::vector<std::size_t> deterministic_owner_map(
    const Key128 &run_key, std::uint64_t fence_epoch,
    std::uint64_t generation, std::uint32_t attempt,
    std::uint64_t owner_epoch, std::size_t owner_count,
    std::size_t shard_count);
int compute_owner_result_root(
    const GenerationPlan &plan,
    const std::map<std::uint32_t, std::vector<std::uint8_t>> &shards,
    Digest *out);
int compute_result_root(
    const GenerationPlan &plan, std::uint64_t global_weight,
    const std::map<std::uint32_t, std::vector<std::uint8_t>> &shards,
    Digest *out);

class OwnerEngine {
 public:
  explicit OwnerEngine(GenerationPlan plan);

  int freeze();
  int grant_next(std::uint32_t shard_id, std::uint64_t credit_epoch,
                 FrameHeader *credit);
  Receipt apply(const DecodedFrame &frame, std::uint64_t now_unix_ns);
  int finalize(std::uint64_t now_unix_ns);
  int reassign(std::uint64_t new_owner_epoch);
  void cancel();

  const std::vector<std::uint8_t> *result(std::uint32_t shard_id) const;
  int owner_result_root(Digest *out) const;
  const OwnerMetrics &metrics() const noexcept { return metrics_; }
  std::uint32_t state() const noexcept { return state_; }
  std::uint64_t retained_bytes() const noexcept { return metrics_.retained_bytes; }

 private:
  struct LedgerEntry {
    std::uint64_t payload_offset{0};
    std::uint64_t payload_bytes{0};
    Digest payload_digest{};
    std::uint64_t weight{0};
    std::uint64_t owner_epoch{0};
    WireStatus status{WireStatus::applied};
    WireReason reason{WireReason::none};
    std::uint32_t application_ordinal{0};
    Digest receipt_digest{};
  };
  struct ShardState {
    std::vector<Contribution> order;
    std::size_t next{0};
    std::uint64_t credit_epoch{0};
    std::uint64_t credit_bytes{0};
    Digest credited_contribution{};
    std::vector<std::uint8_t> accumulator;
    std::vector<std::optional<LedgerEntry>> ledger;
  };

  Receipt reject(const FrameHeader &header, WireReason reason);
  Receipt make_receipt(const FrameHeader &header, WireStatus status,
                       WireReason reason, std::uint32_t ordinal) const;
  bool identity_matches(const FrameHeader &header, WireReason *reason) const;
  GenerationPlan plan_;
  std::map<std::uint32_t, ShardState> shards_;
  OwnerMetrics metrics_{};
  std::uint32_t state_{0};  // 0 installed, 1 frozen, 2 ready, 3 cancelled
};

class ReplayBuffer {
 public:
  ReplayBuffer(std::uint64_t layout_bytes, std::uint64_t deadline_unix_ns);
  int retain(std::uint32_t shard_id, std::vector<std::uint8_t> frame);
  int acknowledge(std::uint32_t shard_id);
  int begin_reassignment(std::uint64_t now_unix_ns);
  const std::vector<std::uint8_t> *frame(std::uint32_t shard_id) const;
  void cancel();
  const OwnerMetrics &metrics() const noexcept { return metrics_; }

 private:
  std::uint64_t layout_bytes_;
  std::uint64_t deadline_unix_ns_;
  std::map<std::uint32_t, std::vector<std::uint8_t>> frames_;
  OwnerMetrics metrics_{};
};

class ResultAssembler {
 public:
  explicit ResultAssembler(GenerationPlan plan);
  int accept(const DecodedFrame &frame, std::uint64_t now_unix_ns);
  bool complete() const noexcept;
  const std::vector<std::uint8_t> &aggregate() const noexcept { return aggregate_; }
  std::uint64_t released_bytes() const noexcept { return released_bytes_; }
  void release();

 private:
  GenerationPlan plan_;
  std::uint64_t global_weight_{0};
  std::uint64_t payload_max_;
  std::uint64_t deadline_unix_ns_;
  std::vector<std::uint8_t> aggregate_;
  std::vector<bool> received_;
  std::vector<Digest> digests_;
  std::uint64_t released_bytes_{0};
};

}  // namespace emender::ndp

#endif
