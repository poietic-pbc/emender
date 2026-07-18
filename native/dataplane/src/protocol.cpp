#include "protocol.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace emender::ndp {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{
    0x45, 0x4d, 0x4e, 0x44, 0x50, 0x31, 0x00, 0x00};
constexpr std::string_view kEndpointDomain{"emender-ndp-endpoint-v1\0", 24};

class Writer {
 public:
  explicit Writer(std::vector<std::uint8_t> *out) : out_(out) {}

  void bytes(const std::uint8_t *data, std::size_t size) {
    out_->insert(out_->end(), data, data + size);
  }
  void u16(std::uint16_t value) {
    for (unsigned i = 0; i != 2; ++i) out_->push_back(static_cast<std::uint8_t>(value >> (8U * i)));
  }
  void u32(std::uint32_t value) {
    for (unsigned i = 0; i != 4; ++i) out_->push_back(static_cast<std::uint8_t>(value >> (8U * i)));
  }
  void u64(std::uint64_t value) {
    for (unsigned i = 0; i != 8; ++i) out_->push_back(static_cast<std::uint8_t>(value >> (8U * i)));
  }

 private:
  std::vector<std::uint8_t> *out_;
};

class Reader {
 public:
  Reader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

  bool bytes(std::uint8_t *out, std::size_t size) {
    if (size > size_ - offset_) return false;
    std::memcpy(out, data_ + offset_, size);
    offset_ += size;
    return true;
  }
  bool u16(std::uint16_t *out) {
    std::uint64_t value = 0;
    if (!integer(2, &value)) return false;
    *out = static_cast<std::uint16_t>(value);
    return true;
  }
  bool u32(std::uint32_t *out) {
    std::uint64_t value = 0;
    if (!integer(4, &value)) return false;
    *out = static_cast<std::uint32_t>(value);
    return true;
  }
  bool u64(std::uint64_t *out) { return integer(8, out); }
  std::size_t remaining() const { return size_ - offset_; }

 private:
  bool integer(std::size_t bytes, std::uint64_t *out) {
    if (bytes > size_ - offset_) return false;
    std::uint64_t value = 0;
    for (std::size_t i = 0; i != bytes; ++i) {
      value |= static_cast<std::uint64_t>(data_[offset_ + i]) << (8U * i);
    }
    offset_ += bytes;
    *out = value;
    return true;
  }

  const std::uint8_t *data_;
  std::size_t size_;
  std::size_t offset_{0};
};

bool valid_type(std::uint16_t value) {
  return value >= static_cast<std::uint16_t>(MessageType::route_probe) &&
         value <= static_cast<std::uint16_t>(MessageType::goodbye);
}

bool valid_status(std::uint32_t value) {
  return value <= static_cast<std::uint32_t>(WireStatus::retryable);
}

bool valid_reason(std::uint32_t value) {
  return value <= static_cast<std::uint32_t>(WireReason::shutdown);
}

bool known_utf8(std::string_view text) {
  // Provider/fabric/domain names are normally ASCII. Reject embedded NUL and
  // malformed continuation/overlong forms without pulling a locale library
  // into the wire parser.
  std::size_t i = 0;
  while (i < text.size()) {
    const auto c = static_cast<unsigned char>(text[i]);
    if (c == 0) return false;
    if (c < 0x80) {
      ++i;
      continue;
    }
    std::size_t need = 0;
    std::uint32_t cp = 0;
    if ((c & 0xe0U) == 0xc0U) { need = 1; cp = c & 0x1fU; if (cp < 2) return false; }
    else if ((c & 0xf0U) == 0xe0U) { need = 2; cp = c & 0x0fU; }
    else if ((c & 0xf8U) == 0xf0U) { need = 3; cp = c & 0x07U; }
    else return false;
    if (need > text.size() - i - 1) return false;
    for (std::size_t j = 1; j <= need; ++j) {
      const auto cc = static_cast<unsigned char>(text[i + j]);
      if ((cc & 0xc0U) != 0x80U) return false;
      cp = (cp << 6U) | (cc & 0x3fU);
    }
    if ((need == 2 && cp < 0x800U) || (need == 3 && cp < 0x10000U) ||
        cp > 0x10ffffU || (cp >= 0xd800U && cp <= 0xdfffU)) return false;
    i += need + 1;
  }
  return true;
}

}  // namespace

Digest sha256(const std::uint8_t *data, std::size_t bytes) {
  Digest out{};
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  if (raw == nullptr) throw std::bad_alloc();
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(raw, EVP_MD_CTX_free);
  unsigned int result_bytes = 0;
  if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1 ||
      (bytes != 0 && EVP_DigestUpdate(ctx.get(), data, bytes) != 1) ||
      EVP_DigestFinal_ex(ctx.get(), out.data(), &result_bytes) != 1 ||
      result_bytes != out.size()) {
    throw std::runtime_error("OpenSSL SHA-256 failure");
  }
  return out;
}

Digest sha256_domain(std::string_view domain,
                     const std::uint8_t *data, std::size_t bytes) {
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  if (raw == nullptr) throw std::bad_alloc();
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(raw, EVP_MD_CTX_free);
  Digest out{};
  unsigned int result_bytes = 0;
  if (EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(ctx.get(), domain.data(), domain.size()) != 1 ||
      (bytes != 0 && EVP_DigestUpdate(ctx.get(), data, bytes) != 1) ||
      EVP_DigestFinal_ex(ctx.get(), out.data(), &result_bytes) != 1 ||
      result_bytes != out.size()) {
    throw std::runtime_error("OpenSSL domain SHA-256 failure");
  }
  return out;
}

std::uint32_t crc32c(const std::uint8_t *data, std::size_t bytes) {
  std::uint32_t crc = UINT32_C(0xffffffff);
  for (std::size_t i = 0; i != bytes; ++i) {
    crc ^= data[i];
    for (unsigned bit = 0; bit != 8; ++bit) {
      const std::uint32_t mask = static_cast<std::uint32_t>(-
          static_cast<std::int32_t>(crc & 1U));
      crc = (crc >> 1U) ^ (UINT32_C(0x82f63b78) & mask);
    }
  }
  return crc ^ UINT32_C(0xffffffff);
}

bool constant_time_equal(const Digest &a, const Digest &b) noexcept {
  std::uint8_t different = 0;
  for (std::size_t i = 0; i != a.size(); ++i) different |= a[i] ^ b[i];
  return different == 0;
}

bool message_has_body(MessageType type) noexcept {
  return type == MessageType::contribution_data || type == MessageType::result_data;
}

int encode_frame(const FrameHeader &header,
                 const std::uint8_t *payload, std::size_t payload_bytes,
                 std::vector<std::uint8_t> *out) {
  if (out == nullptr || (payload_bytes != 0 && payload == nullptr)) return NDP_T_EINVAL;
  const bool has_body = message_has_body(header.type);
  if ((has_body && payload_bytes != header.payload_bytes) ||
      (!has_body && payload_bytes != 0) || header.flags != 0 ||
      header.payload_bytes > NDP_TRANSPORT_MAX_PAYLOAD) return NDP_T_EBOUNDS;
  if (has_body && !constant_time_equal(sha256(payload, payload_bytes), header.payload_digest)) {
    return NDP_T_ECHECKSUM;
  }

  out->clear();
  out->reserve(kHeaderBytes + payload_bytes);
  Writer w(out);
  w.bytes(kMagic.data(), kMagic.size());
  w.u16(1); w.u16(0);
  w.u16(static_cast<std::uint16_t>(header.type));
  w.u16(header.flags);
  w.u32(static_cast<std::uint32_t>(kHeaderBytes));
  w.u32(0);
  w.bytes(header.run_key.data(), header.run_key.size());
  w.u64(header.fence_epoch); w.u64(header.generation);
  w.u32(header.attempt); w.u32(header.shard_id);
  w.u64(header.owner_epoch); w.u64(header.contribution_seq);
  w.bytes(header.worker_key.data(), header.worker_key.size());
  w.bytes(header.incarnation.data(), header.incarnation.size());
  w.bytes(header.layout_digest.data(), header.layout_digest.size());
  w.bytes(header.base_digest.data(), header.base_digest.size());
  w.bytes(header.payload_digest.data(), header.payload_digest.size());
  w.bytes(header.contribution_digest.data(), header.contribution_digest.size());
  w.u64(header.payload_offset); w.u64(header.payload_bytes);
  w.u64(header.shard_bytes); w.u64(header.weight);
  w.u64(header.message_seq); w.u64(header.deadline_unix_ns);
  w.u64(header.credit_bytes);
  w.u32(header.chunk_index); w.u32(header.chunk_count);
  w.u32(static_cast<std::uint32_t>(header.status));
  w.u32(static_cast<std::uint32_t>(header.reason));
  if (out->size() != kHeaderCrcOffset) return NDP_T_EIO;
  w.u32(crc32c(out->data(), out->size()));
  w.u32(0);
  if (out->size() != kHeaderBytes) return NDP_T_EIO;
  if (payload_bytes != 0) w.bytes(payload, payload_bytes);
  return NDP_T_OK;
}

int decode_frame_view(const std::uint8_t *frame, std::size_t frame_bytes,
                      std::uint64_t payload_max, FrameHeader *header,
                      const std::uint8_t **payload, std::size_t *payload_bytes) {
  if (frame == nullptr || header == nullptr || payload == nullptr ||
      payload_bytes == nullptr) return NDP_T_EINVAL;
  if (frame_bytes < kHeaderBytes || payload_max == 0 ||
      payload_max > NDP_TRANSPORT_MAX_PAYLOAD) return NDP_T_EBOUNDS;
  if (!std::equal(kMagic.begin(), kMagic.end(), frame)) return NDP_T_EVERSION;
  Reader r(frame, kHeaderBytes);
  std::array<std::uint8_t, 8> magic{};
  std::uint16_t major = 0, minor = 0, type = 0, flags = 0;
  std::uint32_t header_bytes = 0, reserved = 0, status = 0, reason = 0;
  std::uint32_t encoded_crc = 0, reserved_tail = 0;
  FrameHeader h{};
  if (!r.bytes(magic.data(), magic.size()) || !r.u16(&major) || !r.u16(&minor) ||
      !r.u16(&type) || !r.u16(&flags) || !r.u32(&header_bytes) || !r.u32(&reserved) ||
      !r.bytes(h.run_key.data(), h.run_key.size()) || !r.u64(&h.fence_epoch) ||
      !r.u64(&h.generation) || !r.u32(&h.attempt) || !r.u32(&h.shard_id) ||
      !r.u64(&h.owner_epoch) || !r.u64(&h.contribution_seq) ||
      !r.bytes(h.worker_key.data(), h.worker_key.size()) ||
      !r.bytes(h.incarnation.data(), h.incarnation.size()) ||
      !r.bytes(h.layout_digest.data(), h.layout_digest.size()) ||
      !r.bytes(h.base_digest.data(), h.base_digest.size()) ||
      !r.bytes(h.payload_digest.data(), h.payload_digest.size()) ||
      !r.bytes(h.contribution_digest.data(), h.contribution_digest.size()) ||
      !r.u64(&h.payload_offset) || !r.u64(&h.payload_bytes) ||
      !r.u64(&h.shard_bytes) || !r.u64(&h.weight) || !r.u64(&h.message_seq) ||
      !r.u64(&h.deadline_unix_ns) || !r.u64(&h.credit_bytes) ||
      !r.u32(&h.chunk_index) || !r.u32(&h.chunk_count) || !r.u32(&status) ||
      !r.u32(&reason) || !r.u32(&encoded_crc) || !r.u32(&reserved_tail) ||
      r.remaining() != 0) return NDP_T_EBOUNDS;
  if (major != 1 || minor != 0 || header_bytes != kHeaderBytes || flags != 0 ||
      reserved != 0 || reserved_tail != 0 || !valid_type(type) ||
      !valid_status(status) || !valid_reason(reason)) return NDP_T_EVERSION;
  if (encoded_crc != crc32c(frame, kHeaderCrcOffset)) return NDP_T_ECHECKSUM;
  h.type = static_cast<MessageType>(type);
  h.flags = flags;
  h.status = static_cast<WireStatus>(status);
  h.reason = static_cast<WireReason>(reason);
  if (h.payload_bytes > payload_max || h.shard_bytes > payload_max) return NDP_T_EBOUNDS;
  const std::size_t body_bytes = message_has_body(h.type) ?
      static_cast<std::size_t>(h.payload_bytes) : 0;
  if (body_bytes > std::numeric_limits<std::size_t>::max() - kHeaderBytes ||
      frame_bytes != kHeaderBytes + body_bytes) return NDP_T_EBOUNDS;
  if (body_bytes != 0 &&
      !constant_time_equal(sha256(frame + kHeaderBytes, body_bytes), h.payload_digest)) {
    return NDP_T_ECHECKSUM;
  }
  *header = h;
  *payload = frame + kHeaderBytes;
  *payload_bytes = body_bytes;
  return NDP_T_OK;
}

int decode_frame(const std::uint8_t *frame, std::size_t frame_bytes,
                 std::uint64_t payload_max, DecodedFrame *out) {
  if (out == nullptr) return NDP_T_EINVAL;
  const std::uint8_t *payload = nullptr;
  std::size_t payload_bytes = 0;
  FrameHeader header{};
  const int rc = decode_frame_view(frame, frame_bytes, payload_max, &header,
                                   &payload, &payload_bytes);
  if (rc != NDP_T_OK) return rc;
  out->header = header;
  out->payload.assign(payload, payload + payload_bytes);
  return NDP_T_OK;
}

int encode_endpoint_record(const EndpointRecord &record,
                           std::vector<std::uint8_t> *out) {
  if (out == nullptr || record.provider_name.empty() || record.fabric_name.empty() ||
      record.domain_name.empty() || record.endpoint_name.empty() ||
      record.provider_name.size() > UINT16_MAX || record.fabric_name.size() > UINT16_MAX ||
      record.domain_name.size() > UINT16_MAX || record.endpoint_name.size() > UINT16_MAX ||
      !known_utf8(record.provider_name) || !known_utf8(record.fabric_name) ||
      !known_utf8(record.domain_name)) return NDP_T_EINVAL;
  out->clear();
  Writer w(out);
  w.bytes(record.run_key.data(), record.run_key.size());
  w.u64(record.fence_epoch);
  w.bytes(record.worker_key.data(), record.worker_key.size());
  w.bytes(record.incarnation.data(), record.incarnation.size());
  w.u64(record.endpoint_epoch); w.u64(record.expires_unix_ns);
  const auto write_string = [&w](std::string_view value) {
    w.u16(static_cast<std::uint16_t>(value.size()));
    w.bytes(reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
  };
  write_string(record.provider_name); write_string(record.fabric_name);
  write_string(record.domain_name); w.u32(record.addr_format);
  w.u16(static_cast<std::uint16_t>(record.endpoint_name.size()));
  w.bytes(record.endpoint_name.data(), record.endpoint_name.size());
  const Digest digest = sha256_domain(kEndpointDomain, out->data(), out->size());
  w.bytes(digest.data(), digest.size());
  if (out->size() > NDP_TRANSPORT_ENDPOINT_MAX) {
    out->clear();
    return NDP_T_EBOUNDS;
  }
  return NDP_T_OK;
}

int decode_endpoint_record(const std::uint8_t *record, std::size_t record_bytes,
                           EndpointRecord *out) {
  if (record == nullptr || out == nullptr || record_bytes > NDP_TRANSPORT_ENDPOINT_MAX ||
      record_bytes < 16 + 8 + 16 + 16 + 8 + 8 + 2 + 2 + 2 + 4 + 2 + 32) {
    return NDP_T_EBOUNDS;
  }
  Reader r(record, record_bytes - 32);
  EndpointRecord parsed{};
  if (!r.bytes(parsed.run_key.data(), parsed.run_key.size()) ||
      !r.u64(&parsed.fence_epoch) ||
      !r.bytes(parsed.worker_key.data(), parsed.worker_key.size()) ||
      !r.bytes(parsed.incarnation.data(), parsed.incarnation.size()) ||
      !r.u64(&parsed.endpoint_epoch) || !r.u64(&parsed.expires_unix_ns)) return NDP_T_EBOUNDS;
  const auto read_string = [&r](std::string *value) {
    std::uint16_t size = 0;
    if (!r.u16(&size) || size > r.remaining()) return false;
    std::vector<std::uint8_t> bytes(size);
    if (!r.bytes(bytes.data(), bytes.size())) return false;
    value->assign(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return known_utf8(*value);
  };
  std::uint16_t endpoint_bytes = 0;
  if (!read_string(&parsed.provider_name) || !read_string(&parsed.fabric_name) ||
      !read_string(&parsed.domain_name) || !r.u32(&parsed.addr_format) ||
      !r.u16(&endpoint_bytes) || endpoint_bytes == 0 || endpoint_bytes > r.remaining()) {
    return NDP_T_EBOUNDS;
  }
  parsed.endpoint_name.resize(endpoint_bytes);
  if (!r.bytes(parsed.endpoint_name.data(), parsed.endpoint_name.size()) || r.remaining() != 0) {
    return NDP_T_EBOUNDS;
  }
  Digest encoded{};
  std::memcpy(encoded.data(), record + record_bytes - encoded.size(), encoded.size());
  const Digest expected = sha256_domain(kEndpointDomain, record, record_bytes - encoded.size());
  if (!constant_time_equal(encoded, expected)) return NDP_T_ECHECKSUM;
  *out = std::move(parsed);
  return NDP_T_OK;
}

std::uint64_t unix_time_ns() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t monotonic_time_ns() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::string hex(const std::uint8_t *data, std::size_t bytes) {
  static constexpr char chars[] = "0123456789abcdef";
  std::string out(bytes * 2, '0');
  for (std::size_t i = 0; i != bytes; ++i) {
    out[2 * i] = chars[data[i] >> 4U];
    out[2 * i + 1] = chars[data[i] & 0x0fU];
  }
  return out;
}

}  // namespace emender::ndp
