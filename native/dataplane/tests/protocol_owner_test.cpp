#include "owner.hpp"
#include "protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(expression) do {                                                   \
  if (!(expression)) {                                                           \
    std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: "             \
              << #expression << '\n';                                            \
    ++failures;                                                                  \
  }                                                                              \
} while (false)

using namespace emender::ndp;

Digest digest_of(std::string value) {
  return sha256(reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
}

Key128 key(std::uint8_t seed) {
  Key128 result{};
  for (std::size_t i = 0; i != result.size(); ++i) result[i] = static_cast<std::uint8_t>(seed + i);
  return result;
}

std::vector<std::uint8_t> doubles(std::initializer_list<double> values) {
  std::vector<std::uint8_t> result(values.size() * sizeof(double));
  std::size_t offset = 0;
  for (double value : values) {
    std::memcpy(result.data() + offset, &value, sizeof(value));
    offset += sizeof(value);
  }
  return result;
}

GenerationPlan plan() {
  GenerationPlan result{};
  result.run_key = key(1); result.fence_epoch = 7; result.generation = 9;
  result.attempt = 2; result.owner_epoch = 4;
  result.layout_digest = digest_of("layout"); result.base_digest = digest_of("base");
  result.owner_worker_key = key(141); result.owner_incarnation = key(161);
  result.owner_endpoint_epoch = 8;
  result.layout_bytes = 16; result.payload_max = 16; result.shard_count = 1;
  result.tx_slots = 1; result.rx_slots = 1; result.owner_count = 1;
  result.assigned_bytes = 16; result.resident_limit_bytes = 128 * 1024 * 1024;
  result.deadline_unix_ns = unix_time_ns() + UINT64_C(60) * 1000 * 1000 * 1000;
  result.assigned_shards = {0};
  Contribution a{}; a.worker_key = key(31); a.incarnation = key(51);
  a.contribution_seq = 3; a.contribution_digest = digest_of("contribution-a"); a.weight = 3;
  Contribution b{}; b.worker_key = key(71); b.incarnation = key(91);
  b.contribution_seq = 5; b.contribution_digest = digest_of("contribution-b"); b.weight = 5;
  result.accepted = {a, b};
  return result;
}

DecodedFrame contribution_frame(const GenerationPlan &p, const Contribution &c,
                                const std::vector<std::uint8_t> &payload) {
  FrameHeader header{};
  header.type = MessageType::contribution_data; header.run_key = p.run_key;
  header.fence_epoch = p.fence_epoch; header.generation = p.generation;
  header.attempt = p.attempt; header.shard_id = 0; header.owner_epoch = p.owner_epoch;
  header.contribution_seq = c.contribution_seq; header.worker_key = c.worker_key;
  header.incarnation = c.incarnation; header.layout_digest = p.layout_digest;
  header.base_digest = p.base_digest; header.payload_digest = sha256(payload.data(), payload.size());
  header.contribution_digest = c.contribution_digest; header.payload_offset = 0;
  header.payload_bytes = payload.size(); header.shard_bytes = payload.size();
  header.weight = c.weight; header.message_seq = 10;
  header.deadline_unix_ns = p.deadline_unix_ns;
  header.chunk_index = 0; header.chunk_count = 1;
  return DecodedFrame{header, payload};
}

void test_wire_codec() {
  const std::string vector = "123456789";
  CHECK(crc32c(reinterpret_cast<const std::uint8_t *>(vector.data()), vector.size()) ==
        UINT32_C(0xe3069283));
  const auto p = plan();
  const auto payload = doubles({6.0, 12.0});
  auto decoded = contribution_frame(p, p.accepted[0], payload);
  std::vector<std::uint8_t> encoded;
  CHECK(encode_frame(decoded.header, payload.data(), payload.size(), &encoded) == NDP_T_OK);
  CHECK(encoded.size() == kHeaderBytes + payload.size());
  DecodedFrame roundtrip{};
  CHECK(decode_frame(encoded.data(), encoded.size(), 16, &roundtrip) == NDP_T_OK);
  CHECK(roundtrip.header.fence_epoch == p.fence_epoch);
  CHECK(roundtrip.payload == payload);

  // Partial/truncated local writes cannot become valid frames.
  for (const std::size_t length : {std::size_t{0}, std::size_t{20}, kHeaderBytes - 1,
                                   encoded.size() - 1}) {
    CHECK(decode_frame(encoded.data(), length, 16, &roundtrip) == NDP_T_EBOUNDS);
  }
  auto corrupt_header = encoded;
  corrupt_header[48] ^= 1;
  CHECK(decode_frame(corrupt_header.data(), corrupt_header.size(), 16, &roundtrip) ==
        NDP_T_ECHECKSUM);
  auto corrupt_payload = encoded;
  corrupt_payload.back() ^= 1;
  CHECK(decode_frame(corrupt_payload.data(), corrupt_payload.size(), 16, &roundtrip) ==
        NDP_T_ECHECKSUM);

  EndpointRecord endpoint{};
  endpoint.run_key = p.run_key; endpoint.fence_epoch = p.fence_epoch;
  endpoint.worker_key = key(111); endpoint.incarnation = key(131);
  endpoint.endpoint_epoch = 3; endpoint.expires_unix_ns = p.deadline_unix_ns;
  endpoint.provider_name = "tcp;ofi_rxm"; endpoint.fabric_name = "127.0.0.1/32";
  endpoint.domain_name = "lo"; endpoint.addr_format = 2;
  endpoint.endpoint_name = {1, 2, 3, 4, 0, 255};
  std::vector<std::uint8_t> endpoint_bytes;
  CHECK(encode_endpoint_record(endpoint, &endpoint_bytes) == NDP_T_OK);
  EndpointRecord endpoint_roundtrip{};
  CHECK(decode_endpoint_record(endpoint_bytes.data(), endpoint_bytes.size(),
                               &endpoint_roundtrip) == NDP_T_OK);
  CHECK(endpoint_roundtrip.endpoint_name == endpoint.endpoint_name);
  endpoint_bytes.back() ^= 1;
  CHECK(decode_endpoint_record(endpoint_bytes.data(), endpoint_bytes.size(),
                               &endpoint_roundtrip) == NDP_T_ECHECKSUM);
}

void test_owner_order_duplicate_corruption_timeout_replay() {
  const auto p = plan();
  std::uint64_t required = 0;
  CHECK(validate_plan_bounds(p, &required) == NDP_T_OK);
  CHECK(required <= p.resident_limit_bytes);
  auto invalid = p; invalid.resident_limit_bytes = required - 1;
  CHECK(validate_plan_bounds(invalid, nullptr) == NDP_T_EBOUNDS);

  OwnerEngine owner(p);
  CHECK(owner.freeze() == NDP_T_OK);
  const auto order = contribution_order(p, 0);
  CHECK(order.size() == 2);
  const auto payload_for = [&](const Contribution &c) {
    return constant_time_equal(c.contribution_digest, p.accepted[0].contribution_digest)
        ? doubles({6.0, 12.0}) : doubles({20.0, 28.0});
  };

  // A valid but out-of-order frame is retryable and cannot mutate the sum.
  const Contribution &second = order[1];
  auto second_frame = contribution_frame(p, second, payload_for(second));
  auto receipt = owner.apply(second_frame, unix_time_ns());
  CHECK(receipt.status == WireStatus::retryable);
  CHECK(receipt.reason == WireReason::no_credit);

  FrameHeader credit{};
  CHECK(owner.grant_next(0, 1, &credit) == NDP_T_OK);
  CHECK(constant_time_equal(credit.contribution_digest, order[0].contribution_digest));
  const Contribution &first = order[0];
  auto first_frame = contribution_frame(p, first, payload_for(first));
  receipt = owner.apply(first_frame, unix_time_ns());
  CHECK(receipt.status == WireStatus::applied);
  const Digest first_receipt = receipt.receipt_digest;

  // Lost application receipt causes an identical replay: one add, stable digest.
  receipt = owner.apply(first_frame, unix_time_ns());
  CHECK(receipt.status == WireStatus::applied);
  CHECK(constant_time_equal(receipt.receipt_digest, first_receipt));
  CHECK(owner.metrics().duplicate_frames == 1);

  auto conflict = first_frame;
  ++conflict.header.weight;
  receipt = owner.apply(conflict, unix_time_ns());
  CHECK(receipt.reason == WireReason::conflict);

  CHECK(owner.grant_next(0, 2, &credit) == NDP_T_OK);
  receipt = owner.apply(second_frame, unix_time_ns());
  CHECK(receipt.status == WireStatus::applied);
  CHECK(owner.finalize(unix_time_ns()) == NDP_T_OK);
  const auto *result = owner.result(0);
  CHECK(result != nullptr && result->size() == 16);
  if (result != nullptr) {
    double first_value = 0, second_value = 0;
    std::memcpy(&first_value, result->data(), 8);
    std::memcpy(&second_value, result->data() + 8, 8);
    CHECK(first_value == 26.0 / 8.0);
    CHECK(second_value == 40.0 / 8.0);
    Digest owner_root{}, full_root{};
    std::map<std::uint32_t, std::vector<std::uint8_t>> result_shards{{0, *result}};
    CHECK(owner.owner_result_root(&owner_root) == NDP_T_OK);
    CHECK(compute_result_root(p, 8, result_shards, &full_root) == NDP_T_OK);
    CHECK(!constant_time_equal(owner_root, full_root));
    result_shards[0][0] ^= 1;
    Digest changed_root{};
    CHECK(compute_result_root(p, 8, result_shards, &changed_root) == NDP_T_OK);
    CHECK(!constant_time_equal(full_root, changed_root));
  }
  CHECK(owner.metrics().retained_bytes == 16);

  // Owner loss/reassignment clears the epoch-local ledger and accepts bounded replay.
  CHECK(owner.reassign(p.owner_epoch + 1) == NDP_T_OK);
  auto replay_plan = p; replay_plan.owner_epoch = p.owner_epoch + 1;
  const auto replay_order = contribution_order(replay_plan, 0);
  for (std::size_t i = 0; i != replay_order.size(); ++i) {
    CHECK(owner.grant_next(0, i + 1, &credit) == NDP_T_OK);
    auto replay = contribution_frame(replay_plan, replay_order[i], payload_for(replay_order[i]));
    CHECK(owner.apply(replay, unix_time_ns()).status == WireStatus::applied);
  }
  CHECK(owner.finalize(unix_time_ns()) == NDP_T_OK);
  CHECK(owner.metrics().reassignment_count == 1);

  // Stale fences, expired deadlines, and nonfinite frames never enter accumulators.
  OwnerEngine stale_owner(p); CHECK(stale_owner.freeze() == NDP_T_OK);
  CHECK(stale_owner.grant_next(0, 1, &credit) == NDP_T_OK);
  auto stale = contribution_frame(p, order[0], payload_for(order[0]));
  --stale.header.fence_epoch;
  CHECK(stale_owner.apply(stale, unix_time_ns()).reason == WireReason::stale_fence);
  auto expired = contribution_frame(p, order[0], payload_for(order[0]));
  expired.header.deadline_unix_ns = unix_time_ns() - 1;
  CHECK(stale_owner.apply(expired, unix_time_ns()).reason == WireReason::deadline);
  auto bad = doubles({std::numeric_limits<double>::infinity(), 1.0});
  auto nonfinite = contribution_frame(p, order[0], bad);
  CHECK(stale_owner.apply(nonfinite, unix_time_ns()).reason == WireReason::nonfinite);
  stale_owner.cancel();
  CHECK(stale_owner.state() == 3);
}

void test_replay_and_redistribution_bounds() {
  const std::uint64_t deadline = unix_time_ns() + UINT64_C(10) * 1000 * 1000 * 1000;
  ReplayBuffer replay(1024, deadline);
  CHECK(replay.retain(0, std::vector<std::uint8_t>(400, 1)) == NDP_T_OK);
  CHECK(replay.retain(1, std::vector<std::uint8_t>(400, 2)) == NDP_T_OK);
  CHECK(replay.retain(2, std::vector<std::uint8_t>(300, 3)) == NDP_T_EBOUNDS);
  CHECK(replay.begin_reassignment(unix_time_ns()) == NDP_T_OK);
  CHECK(replay.begin_reassignment(unix_time_ns()) == NDP_T_OK);
  CHECK(replay.begin_reassignment(unix_time_ns()) == NDP_T_EBOUNDS);
  CHECK(replay.metrics().replay_bytes <= 2 * 1024);
  CHECK(replay.acknowledge(0) == NDP_T_OK);
  replay.cancel();
  CHECK(replay.metrics().retained_bytes == 0);
  CHECK(replay.metrics().released_bytes == 800);

  ResultAssembler assembler(32, 16, 2, deadline);
  FrameHeader h{}; h.type = MessageType::result_data; h.deadline_unix_ns = deadline;
  h.chunk_count = 2; h.shard_bytes = 16; h.payload_bytes = 16;
  auto second_payload = doubles({3.0, 4.0});
  h.shard_id = 1; h.chunk_index = 1; h.payload_offset = 16;
  h.payload_digest = sha256(second_payload.data(), second_payload.size());
  CHECK(assembler.accept(DecodedFrame{h, second_payload}, unix_time_ns()) == NDP_T_OK);
  auto first_payload = doubles({1.0, 2.0});
  h.shard_id = 0; h.chunk_index = 0; h.payload_offset = 0;
  h.payload_digest = sha256(first_payload.data(), first_payload.size());
  CHECK(assembler.accept(DecodedFrame{h, first_payload}, unix_time_ns()) == NDP_T_OK);
  CHECK(assembler.complete());
  CHECK(std::equal(first_payload.begin(), first_payload.end(), assembler.aggregate().begin()));
  CHECK(assembler.accept(DecodedFrame{h, first_payload}, unix_time_ns()) == NDP_T_OK);
  auto conflicting_payload = doubles({9.0, 10.0});
  h.payload_digest = sha256(conflicting_payload.data(), conflicting_payload.size());
  CHECK(assembler.accept(DecodedFrame{h, conflicting_payload}, unix_time_ns()) ==
        NDP_T_ECONFLICT);
  assembler.release();
  CHECK(assembler.released_bytes() == 32);

  const auto map1 = deterministic_owner_map(key(1), 7, 9, 2, 1, 3, 17);
  const auto map2 = deterministic_owner_map(key(1), 7, 9, 2, 3, 3, 17);
  const auto map1_repeat = deterministic_owner_map(key(1), 7, 9, 2, 1, 3, 17);
  CHECK(map1.size() == 17 && map2.size() == 17);
  CHECK(map1 == map1_repeat);
  CHECK(map1[0] == 0 && map2[0] == 1);
  for (std::size_t i = 0; i != map1.size(); ++i) {
    CHECK(map1[i] == (map1[0] + i) % 3);
  }
  CHECK(*std::max_element(map1.begin(), map1.end()) < 3);
}

}  // namespace

int main() {
  test_wire_codec();
  test_owner_order_duplicate_corruption_timeout_replay();
  test_replay_and_redistribution_bounds();
  if (failures != 0) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "protocol/owner tests passed\n";
  return 0;
}
