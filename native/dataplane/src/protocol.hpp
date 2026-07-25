#ifndef EMENDER_NDP_PROTOCOL_HPP
#define EMENDER_NDP_PROTOCOL_HPP

#include "emender/ndp_transport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace emender::ndp {

using Key128 = std::array<std::uint8_t, 16>;
using Digest = std::array<std::uint8_t, 32>;

constexpr std::size_t kHeaderBytes = 320;
constexpr std::size_t kHeaderCrcOffset = 312;
constexpr std::uint32_t kNoShard = UINT32_C(0xffffffff);

enum class MessageType : std::uint16_t {
  route_probe = 1,
  route_probe_ack = 2,
  credit = 3,
  contribution_data = 4,
  receipt = 5,
  result_announce = 6,
  fetch = 7,
  result_data = 8,
  cancel = 9,
  goodbye = 10,
};

enum class WireStatus : std::uint32_t {
  none = 0,
  applied = 1,
  duplicate = 2,
  finalized = 3,
  rejected = 4,
  retryable = 5,
};

enum class WireReason : std::uint32_t {
  none = 0,
  stale_fence = 1,
  stale_generation_or_attempt = 2,
  stale_owner_epoch = 3,
  not_accepted = 4,
  layout_or_base = 5,
  byte_bounds = 6,
  checksum = 7,
  nonfinite = 8,
  conflict = 9,
  no_credit = 10,
  deadline = 11,
  route = 12,
  provider = 13,
  shutdown = 14,
};

struct FrameHeader {
  std::uint16_t protocol_major{1};
  std::uint16_t protocol_minor{0};
  MessageType type{MessageType::route_probe};
  std::uint16_t flags{0};
  Key128 run_key{};
  std::uint64_t fence_epoch{0};
  std::uint64_t generation{0};
  std::uint32_t attempt{0};
  std::uint32_t shard_id{kNoShard};
  std::uint64_t owner_epoch{0};
  std::uint64_t contribution_seq{0};
  Key128 worker_key{};
  Key128 incarnation{};
  Digest layout_digest{};
  Digest base_digest{};
  Digest payload_digest{};
  Digest contribution_digest{};
  std::uint64_t payload_offset{0};
  std::uint64_t payload_bytes{0};
  std::uint64_t shard_bytes{0};
  std::uint64_t weight{0};
  std::uint64_t message_seq{0};
  std::uint64_t deadline_unix_ns{0};
  std::uint64_t credit_bytes{0};
  std::uint32_t chunk_index{0};
  std::uint32_t chunk_count{0};
  WireStatus status{WireStatus::none};
  WireReason reason{WireReason::none};
};

struct DecodedFrame {
  FrameHeader header{};
  std::vector<std::uint8_t> payload;
  // True only when decode_frame authenticated payload_digest. Pure-engine
  // callers leave this false so owner/assembler entry points still verify
  // direct or deliberately corrupted frames independently.
  bool payload_validated{false};
};

struct EndpointRecord {
  Key128 run_key{};
  std::uint64_t fence_epoch{0};
  Key128 worker_key{};
  Key128 incarnation{};
  std::uint64_t endpoint_epoch{0};
  std::uint64_t expires_unix_ns{0};
  std::string provider_name;
  std::string fabric_name;
  std::string domain_name;
  std::uint32_t addr_format{0};
  std::vector<std::uint8_t> endpoint_name;
};

Digest sha256(const std::uint8_t *data, std::size_t bytes);
Digest sha256_domain(std::string_view domain,
                     const std::uint8_t *data, std::size_t bytes);
std::uint32_t crc32c(const std::uint8_t *data, std::size_t bytes);
bool constant_time_equal(const Digest &a, const Digest &b) noexcept;

bool message_has_body(MessageType type) noexcept;
int encode_frame(const FrameHeader &header,
                 const std::uint8_t *payload, std::size_t payload_bytes,
                 std::vector<std::uint8_t> *out);
int encode_frame_prehashed(const FrameHeader &header,
                           const std::uint8_t *payload,
                           std::size_t payload_bytes,
                           std::vector<std::uint8_t> *out);
int decode_frame_view(const std::uint8_t *frame, std::size_t frame_bytes,
                      std::uint64_t payload_max, FrameHeader *header,
                      const std::uint8_t **payload, std::size_t *payload_bytes);
int decode_frame_view_header_only(const std::uint8_t *frame,
                                  std::size_t frame_bytes,
                                  std::uint64_t payload_max,
                                  FrameHeader *header,
                                  const std::uint8_t **payload,
                                  std::size_t *payload_bytes);
int decode_frame(const std::uint8_t *frame, std::size_t frame_bytes,
                 std::uint64_t payload_max, DecodedFrame *out);

int encode_endpoint_record(const EndpointRecord &record,
                           std::vector<std::uint8_t> *out);
int decode_endpoint_record(const std::uint8_t *record, std::size_t record_bytes,
                           EndpointRecord *out);

std::uint64_t unix_time_ns();
std::uint64_t monotonic_time_ns();
std::string hex(const std::uint8_t *data, std::size_t bytes);

}  // namespace emender::ndp

#endif
