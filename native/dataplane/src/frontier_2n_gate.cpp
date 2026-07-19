#include "emender/ndp_transport.h"

#include "owner.hpp"
#include "protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <malloc.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <unistd.h>

namespace {

using emender::ndp::Contribution;
using emender::ndp::DecodedFrame;
using emender::ndp::Digest;
using emender::ndp::FrameHeader;
using emender::ndp::GenerationPlan;
using emender::ndp::Key128;
using emender::ndp::MessageType;
using emender::ndp::OwnerEngine;
using emender::ndp::ResultAssembler;
using emender::ndp::WireReason;
using emender::ndp::WireStatus;

constexpr std::uint64_t kE97LayoutBytes = UINT64_C(5506770496);
constexpr std::uint64_t kE97Elements = UINT64_C(688346312);
constexpr std::uint64_t kPayloadMax = UINT64_C(67108864);
constexpr std::uint64_t kNode0Weight = UINT64_C(1966080);
constexpr std::uint64_t kNode1Weight = UINT64_C(1968000);
constexpr std::uint64_t kGlobalWeight = UINT64_C(3934080);
constexpr std::uint32_t kTrainers = 8;
constexpr std::uint32_t kSlots = 4;
constexpr std::uint64_t kGiB = UINT64_C(1024) * 1024 * 1024;
constexpr std::array<std::uint8_t, 8> kControlMagic{{'N', 'D', 'P', 'C', 'T', 'L', '1', 0}};

struct Options {
  std::string controller_host;
  std::string provider;
  std::string mode{"clean"};
  std::string output;
  std::string run_id;
  std::string payload_id;
  std::uint16_t controller_port{0};
  std::uint32_t rank{0};
  std::uint32_t generations{3};
  std::uint32_t warmup{1};
  std::uint32_t reduction_threads{16};
  std::uint64_t layout_bytes{kE97LayoutBytes};
  std::uint64_t payload_max{kPayloadMax};
  std::uint64_t deadline_seconds{900};
  bool production{false};
};

struct PeerRecord {
  std::uint32_t rank{0};
  Key128 worker_key{};
  Key128 incarnation{};
  std::uint64_t endpoint_epoch{0};
  std::uint64_t expires_unix_ns{0};
  std::vector<std::uint8_t> encoded;
};

struct MetricSnapshot {
  struct ndp_transport_metrics_v1 value{};
};

struct Sample {
  double transfer_redistribution_seconds{0.0};
  std::uint64_t useful_tx_bytes{0};
  std::uint64_t useful_rx_bytes{0};
  std::uint64_t wire_tx_bytes{0};
  std::uint64_t wire_rx_bytes{0};
  std::uint64_t retries{0};
  std::uint64_t released_bytes{0};
  std::uint64_t contribution_tx_bytes{0};
  std::uint64_t redistribution_tx_bytes{0};
  std::uint64_t stale_rejects{0};
  std::uint64_t checksum_rejects{0};
  std::string result_root;
  std::string result_payload_sha256;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

void require_code(int code, const std::string &operation, bool accepted = false) {
  if (code == NDP_T_OK || (accepted && code == NDP_T_ACCEPTED)) return;
  fail(operation + ": " + ndp_transport_error_string(code) +
       " (" + std::to_string(code) + ")");
}

std::uint64_t parse_u64(const std::string &text, const char *name) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != 0) {
    fail(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<std::uint64_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto next = [&]() -> std::string {
      if (++index >= argc) fail("missing value after " + argument);
      return argv[index];
    };
    if (argument == "--controller-host") options.controller_host = next();
    else if (argument == "--controller-port") options.controller_port =
        static_cast<std::uint16_t>(parse_u64(next(), "controller port"));
    else if (argument == "--rank") options.rank =
        static_cast<std::uint32_t>(parse_u64(next(), "rank"));
    else if (argument == "--provider") options.provider = next();
    else if (argument == "--mode") options.mode = next();
    else if (argument == "--output") options.output = next();
    else if (argument == "--run-id") options.run_id = next();
    else if (argument == "--payload-id") options.payload_id = next();
    else if (argument == "--layout-bytes") options.layout_bytes =
        parse_u64(next(), "layout bytes");
    else if (argument == "--payload-max") options.payload_max =
        parse_u64(next(), "payload max");
    else if (argument == "--generations") options.generations =
        static_cast<std::uint32_t>(parse_u64(next(), "generations"));
    else if (argument == "--warmup") options.warmup =
        static_cast<std::uint32_t>(parse_u64(next(), "warmup"));
    else if (argument == "--reduction-threads") options.reduction_threads =
        static_cast<std::uint32_t>(parse_u64(next(), "reduction threads"));
    else if (argument == "--deadline-seconds") options.deadline_seconds =
        parse_u64(next(), "deadline seconds");
    else if (argument == "--production") options.production = true;
    else if (argument == "--test-only") options.production = false;
    else fail("unknown option: " + argument);
  }
  if (options.controller_host.empty() || options.controller_port == 0 ||
      options.rank > 1 || options.provider.empty() || options.output.empty() ||
      options.run_id.empty() || options.payload_id.empty() ||
      (options.mode != "clean" && options.mode != "fault") ||
      options.layout_bytes == 0 || options.layout_bytes % 8 != 0 ||
      options.payload_max == 0 || options.payload_max > kPayloadMax ||
      options.payload_max % 8 != 0 || options.generations == 0 ||
      options.reduction_threads == 0 || options.reduction_threads > 128) {
    fail("invalid or incomplete gate configuration");
  }
  if (options.production &&
      (options.provider != "cxi" || options.layout_bytes != kE97LayoutBytes ||
       options.layout_bytes / 8 != kE97Elements ||
       options.payload_max != kPayloadMax || options.warmup != 1 ||
       (options.mode == "clean" && options.generations != 3) ||
       (options.mode == "fault" && options.generations != 1))) {
    fail("production G2 configuration is not the exact E97 contract");
  }
  return options;
}

void append_u32(std::vector<std::uint8_t> *output, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8) {
    output->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_u64(std::vector<std::uint8_t> *output, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8) {
    output->push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

std::uint32_t read_u32(const std::uint8_t *input) {
  std::uint32_t value = 0;
  for (unsigned index = 0; index != 4; ++index) {
    value |= static_cast<std::uint32_t>(input[index]) << (index * 8U);
  }
  return value;
}

std::uint64_t read_u64(const std::uint8_t *input) {
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index) {
    value |= static_cast<std::uint64_t>(input[index]) << (index * 8U);
  }
  return value;
}

Key128 key_from(const std::string &value) {
  const Digest digest = emender::ndp::sha256(
      reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
  Key128 key{};
  std::copy_n(digest.begin(), key.size(), key.begin());
  return key;
}

Digest digest_from(const std::string &value) {
  return emender::ndp::sha256(
      reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
}

bool same_key(const Key128 &left, const Key128 &right) {
  return std::equal(left.begin(), left.end(), right.begin());
}

std::string json_escape(const std::string &value) {
  std::ostringstream output;
  for (const char raw_character : value) {
    const auto character = static_cast<unsigned char>(raw_character);
    if (character == '"' || character == '\\') output << '\\' << character;
    else if (character >= 0x20) output << character;
    else output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned>(character) << std::dec;
  }
  return output.str();
}

bool write_all(int descriptor, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const std::uint8_t *>(data);
  while (bytes != 0) {
    const ssize_t written = ::send(descriptor, cursor, bytes, MSG_NOSIGNAL);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    cursor += written;
    bytes -= static_cast<std::size_t>(written);
  }
  return true;
}

bool read_all(int descriptor, void *data, std::size_t bytes) {
  auto *cursor = static_cast<std::uint8_t *>(data);
  while (bytes != 0) {
    const ssize_t received = ::recv(descriptor, cursor, bytes, 0);
    if (received < 0 && errno == EINTR) continue;
    if (received <= 0) return false;
    cursor += received;
    bytes -= static_cast<std::size_t>(received);
  }
  return true;
}

int connect_controller(const std::string &host, std::uint16_t port,
                       std::uint64_t deadline_unix_ns) {
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(port);
  if (::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses) != 0) {
    fail("controller address resolution failed");
  }
  int connected = -1;
  while (emender::ndp::unix_time_ns() < deadline_unix_ns && connected < 0) {
    for (addrinfo *address = addresses; address != nullptr; address = address->ai_next) {
      const int descriptor = ::socket(address->ai_family,
                                      address->ai_socktype | SOCK_CLOEXEC,
                                      address->ai_protocol);
      if (descriptor < 0) continue;
      if (::connect(descriptor, address->ai_addr, address->ai_addrlen) == 0) {
        connected = descriptor;
        break;
      }
      ::close(descriptor);
    }
    if (connected < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  ::freeaddrinfo(addresses);
  if (connected < 0) fail("bounded controller connection failed");
  timeval timeout{};
  timeout.tv_sec = 60;
  (void)::setsockopt(connected, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)::setsockopt(connected, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  return connected;
}

class NativeTransport {
 public:
  NativeTransport(const Options &options, const Key128 &run_key,
                  const Key128 &worker_key, const Key128 &incarnation,
                  std::uint64_t endpoint_epoch, std::uint64_t expires_unix_ns,
                  std::uint64_t operation_deadline_unix_ns)
      : payload_max_(options.payload_max) {
    struct ndp_transport_open_v1 config{};
    config.struct_size = sizeof(config);
    config.abi_version = NDP_TRANSPORT_ABI_V1;
    config.mode = options.production ? NDP_T_MODE_PRODUCTION : NDP_T_MODE_TEST_ONLY;
    config.tx_slots = kSlots;
    config.rx_slots = kSlots;
    config.payload_max = options.payload_max;
    config.resident_limit_bytes = 64 * kGiB;
    config.operation_deadline_unix_ns = operation_deadline_unix_ns;
    config.telemetry_fd = -1;
    config.provider_len = static_cast<std::uint32_t>(options.provider.size());
    if (config.provider_len > sizeof(config.provider)) fail("provider name too long");
    std::copy(options.provider.begin(), options.provider.end(), config.provider);
    if (options.production) {
      const std::string required{"cxi"};
      config.require_provider_len = static_cast<std::uint32_t>(required.size());
      std::copy(required.begin(), required.end(), config.require_provider);
      // Frontier exposes cxi0 and cxi1 on each compute node.  Pin one native
      // domain so provider resolution cannot silently pick a rail or reject
      // an otherwise valid but ambiguous set of CXI matches.
      const std::string domain{"cxi0"};
      config.domain_len = static_cast<std::uint32_t>(domain.size());
      std::copy(domain.begin(), domain.end(), config.domain);
    } else {
      const std::string bind{"127.0.0.1"};
      config.bind_node_len = static_cast<std::uint32_t>(bind.size());
      std::copy(bind.begin(), bind.end(), config.bind_node);
    }
    require_code(ndp_transport_open_v1(&config, &handle_), "transport_open");

    ndp_transport_identity_v1 identity{};
    identity.struct_size = sizeof(identity);
    identity.abi_version = NDP_TRANSPORT_ABI_V1;
    std::copy(run_key.begin(), run_key.end(), identity.run_key);
    identity.fence_epoch = 1;
    std::copy(worker_key.begin(), worker_key.end(), identity.worker_key);
    std::copy(incarnation.begin(), incarnation.end(), identity.incarnation);
    identity.endpoint_epoch = endpoint_epoch;
    identity.expires_unix_ns = expires_unix_ns;
    require_code(ndp_transport_bind_identity_v1(handle_, &identity),
                 "transport_bind_identity");

    struct ndp_transport_endpoint_v1 endpoint{};
    endpoint.struct_size = sizeof(endpoint);
    endpoint.abi_version = NDP_TRANSPORT_ABI_V1;
    require_code(ndp_transport_endpoint_v1(handle_, &endpoint), "transport_endpoint");
    endpoint_record_.assign(endpoint.record, endpoint.record + endpoint.record_bytes);
  }

  NativeTransport(const NativeTransport &) = delete;
  NativeTransport &operator=(const NativeTransport &) = delete;

  ~NativeTransport() {
    if (handle_ != 0) (void)ndp_transport_close_v1(handle_);
  }

  const std::vector<std::uint8_t> &endpoint_record() const noexcept {
    return endpoint_record_;
  }

  void install_peer(const PeerRecord &peer) {
    ndp_transport_peer_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = NDP_TRANSPORT_ABI_V1;
    std::copy(peer.worker_key.begin(), peer.worker_key.end(), value.worker_key);
    std::copy(peer.incarnation.begin(), peer.incarnation.end(), value.incarnation);
    value.endpoint_epoch = peer.endpoint_epoch;
    value.expires_unix_ns = peer.expires_unix_ns;
    value.endpoint_name_bytes = static_cast<std::uint32_t>(peer.encoded.size());
    if (peer.encoded.size() > sizeof(value.endpoint_name)) fail("peer record too large");
    std::copy(peer.encoded.begin(), peer.encoded.end(), value.endpoint_name);
    require_code(ndp_transport_peer_upsert_v1(handle_, &value, &peer_id_),
                 "transport_peer_upsert");
  }

  void remove_peer() {
    if (peer_id_ != 0) {
      require_code(ndp_transport_peer_remove_v1(handle_, peer_id_),
                   "transport_peer_remove");
      peer_id_ = 0;
    }
  }

  void send(const std::vector<std::uint8_t> &frame, std::uint64_t deadline_unix_ns) {
    if (peer_id_ == 0) fail("send attempted without current peer route");
    require_code(ndp_transport_send_v1(handle_, peer_id_, frame.data(), frame.size(),
                                       deadline_unix_ns),
                 "transport_send", true);
  }

  std::vector<DecodedFrame> receive(int timeout_ms) {
    ndp_transport_event_v1 events[32]{};
    for (auto &event : events) {
      event.struct_size = sizeof(event);
      event.abi_version = NDP_TRANSPORT_ABI_V1;
    }
    std::uint32_t count = 0;
    require_code(ndp_transport_poll_v1(handle_, events, 32, &count, timeout_ms),
                 "transport_poll");
    for (std::uint32_t index = 0; index != count; ++index) {
      if (events[index].event == NDP_T_EVENT_CQ_ERROR ||
          events[index].event == NDP_T_EVENT_ROUTE_DOWN ||
          events[index].status < 0) {
        fail("native CQ/route event failed: event=" +
             std::to_string(events[index].event) + " status=" +
             std::to_string(events[index].status) + " provider_errno=" +
             std::to_string(events[index].provider_errno) + " reason=" +
             std::to_string(events[index].reason));
      }
    }
    std::vector<DecodedFrame> frames;
    std::vector<std::uint8_t> storage(static_cast<std::size_t>(payload_max_ + 320));
    while (true) {
      std::uint64_t bytes = 0;
      std::uint64_t peer = 0;
      const int received = ndp_transport_receive_v1(
          handle_, storage.data(), storage.size(), &bytes, &peer);
      if (received == NDP_T_ECREDIT) break;
      require_code(received, "transport_receive");
      if (peer != peer_id_) fail("received frame from non-current peer route");
      DecodedFrame decoded{};
      require_code(emender::ndp::decode_frame(storage.data(),
                                              static_cast<std::size_t>(bytes),
                                              payload_max_, &decoded),
                   "decode_frame");
      frames.push_back(std::move(decoded));
    }
    return frames;
  }

  MetricSnapshot metrics() const {
    MetricSnapshot snapshot{};
    snapshot.value.struct_size = sizeof(snapshot.value);
    snapshot.value.abi_version = NDP_TRANSPORT_ABI_V1;
    require_code(ndp_transport_metrics_v1(handle_, &snapshot.value),
                 "transport_metrics");
    return snapshot;
  }

  void wait_idle(std::uint64_t deadline_unix_ns) {
    while (emender::ndp::unix_time_ns() < deadline_unix_ns) {
      (void)receive(10);
      const auto current = metrics().value;
      if (current.in_flight_bytes == 0 && current.retained_bytes == 0) return;
    }
    fail("transport did not return to its post-release floor");
  }

 private:
  ndp_transport_t handle_{0};
  std::uint64_t peer_id_{0};
  std::uint64_t payload_max_{0};
  std::vector<std::uint8_t> endpoint_record_;
};

PeerRecord exchange_endpoint(int control_fd, std::uint32_t rank,
                             std::uint32_t phase,
                             const std::vector<std::uint8_t> &record) {
  std::vector<std::uint8_t> request;
  request.insert(request.end(), kControlMagic.begin(), kControlMagic.end());
  append_u32(&request, 1);
  append_u32(&request, phase);
  append_u32(&request, rank);
  append_u32(&request, static_cast<std::uint32_t>(record.size()));
  append_u64(&request, emender::ndp::unix_time_ns());
  request.insert(request.end(), record.begin(), record.end());
  if (!write_all(control_fd, request.data(), request.size())) {
    fail("controller endpoint request failed");
  }

  std::array<std::uint8_t, 68> header{};
  if (!read_all(control_fd, header.data(), header.size()) ||
      !std::equal(kControlMagic.begin(), kControlMagic.end(), header.begin())) {
    fail("controller endpoint reply was truncated or invalid");
  }
  const std::uint32_t status = read_u32(header.data() + 8);
  if (status != 0) fail("controller rejected endpoint exchange: " + std::to_string(status));
  PeerRecord peer{};
  peer.rank = read_u32(header.data() + 12);
  std::copy_n(header.data() + 16, 16, peer.worker_key.begin());
  std::copy_n(header.data() + 32, 16, peer.incarnation.begin());
  peer.endpoint_epoch = read_u64(header.data() + 48);
  peer.expires_unix_ns = read_u64(header.data() + 56);
  const std::uint32_t record_bytes = read_u32(header.data() + 64);
  if (peer.rank == rank || record_bytes == 0 ||
      record_bytes > NDP_TRANSPORT_ENDPOINT_MAX) {
    fail("controller returned invalid peer identity");
  }
  peer.encoded.resize(record_bytes);
  if (!read_all(control_fd, peer.encoded.data(), peer.encoded.size())) {
    fail("controller returned a truncated endpoint record");
  }
  return peer;
}

std::uint64_t rss_bytes() {
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:") {
      std::uint64_t kib = 0;
      status >> kib;
      return kib * 1024;
    }
    std::string rest;
    std::getline(status, rest);
  }
  return 0;
}

std::uint64_t rss_high_water_bytes() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) {
    fail("cannot read process resident high-water");
  }
  // Linux reports ru_maxrss in KiB.
  return static_cast<std::uint64_t>(usage.ru_maxrss) * UINT64_C(1024);
}

std::vector<double> local_reduce(const Options &options) {
  const std::uint64_t elements = options.layout_bytes / sizeof(double);
  std::vector<double> numerator(static_cast<std::size_t>(elements));
  const std::uint64_t lane_weight = options.rank == 0 ? UINT64_C(245760) : UINT64_C(246000);
  const std::uint32_t threads = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(options.reduction_threads, elements));
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (std::uint32_t thread = 0; thread != threads; ++thread) {
    const std::uint64_t begin = elements * thread / threads;
    const std::uint64_t end = elements * (thread + 1) / threads;
    workers.emplace_back([&, begin, end] {
      for (std::uint64_t index = begin; index != end; ++index) {
        double value = 0.0;
        for (std::uint32_t lane = 0; lane != kTrainers; ++lane) {
          const float source = static_cast<float>(options.rank * 16 + lane) +
              0.25F + ((index & 1U) == 0 ? 0.0F : 0.5F);
          const double term = static_cast<double>(source) *
                              static_cast<double>(lane_weight);
          value = value + term;
        }
        numerator[static_cast<std::size_t>(index)] = value;
      }
    });
  }
  for (auto &worker : workers) worker.join();
  return numerator;
}

std::uint64_t shard_bytes(const Options &options, std::uint32_t shard) {
  const std::uint64_t offset = static_cast<std::uint64_t>(shard) * options.payload_max;
  return std::min(options.payload_max, options.layout_bytes - offset);
}

Contribution make_contribution(const Options &options, std::uint32_t rank,
                               const Key128 &incarnation,
                               std::uint64_t generation) {
  Contribution contribution{};
  contribution.worker_key = key_from("worker:" + options.run_id + ":" + std::to_string(rank));
  contribution.incarnation = incarnation;
  contribution.contribution_seq = generation + 1;
  contribution.contribution_digest = digest_from(
      "contribution:" + options.payload_id + ":" + std::to_string(generation) +
      ":" + std::to_string(rank));
  contribution.weight = rank == 0 ? kNode0Weight : kNode1Weight;
  return contribution;
}

GenerationPlan make_plan(const Options &options, const Key128 &run_key,
                         const std::array<Key128, 2> &contribution_incarnations,
                         const std::array<Key128, 2> &endpoint_incarnations,
                         std::uint64_t generation, std::uint64_t owner_epoch,
                         std::uint64_t deadline_unix_ns,
                         const std::vector<std::size_t> &owners) {
  GenerationPlan plan{};
  plan.run_key = run_key;
  plan.fence_epoch = 1;
  plan.generation = generation;
  plan.attempt = 1;
  plan.owner_epoch = owner_epoch;
  plan.layout_digest = digest_from("e97-layout-v1:" +
      std::to_string(options.layout_bytes) + ":" + std::to_string(options.payload_max));
  plan.base_digest = digest_from("synthetic-base:" + options.payload_id + ":" +
                                 std::to_string(generation));
  plan.owner_worker_key = key_from("worker:" + options.run_id + ":" +
                                   std::to_string(options.rank));
  plan.owner_incarnation = endpoint_incarnations[options.rank];
  plan.owner_endpoint_epoch = owner_epoch;
  plan.layout_bytes = options.layout_bytes;
  plan.payload_max = options.payload_max;
  plan.shard_count = static_cast<std::uint32_t>(
      (options.layout_bytes + options.payload_max - 1) / options.payload_max);
  plan.tx_slots = kSlots;
  plan.rx_slots = kSlots;
  plan.owner_count = 2;
  plan.resident_limit_bytes = 64 * kGiB;
  plan.deadline_unix_ns = deadline_unix_ns;
  for (std::uint32_t shard = 0; shard != plan.shard_count; ++shard) {
    if (owners[shard] == options.rank) {
      plan.assigned_shards.push_back(shard);
      plan.assigned_bytes += shard_bytes(options, shard);
    }
  }
  plan.accepted.push_back(make_contribution(options, 0,
                                             contribution_incarnations[0], generation));
  plan.accepted.push_back(make_contribution(options, 1,
                                             contribution_incarnations[1], generation));
  return plan;
}

std::vector<std::uint8_t> encode_header(const FrameHeader &header) {
  std::vector<std::uint8_t> encoded;
  require_code(emender::ndp::encode_frame(header, nullptr, 0, &encoded),
               "encode_header");
  return encoded;
}

std::vector<std::uint8_t> contribution_frame(
    const GenerationPlan &plan, const Contribution &contribution,
    const double *numerator, std::uint32_t shard, std::uint64_t message_seq,
    bool stale_owner_epoch = false, bool bad_checksum = false) {
  const std::uint64_t offset = static_cast<std::uint64_t>(shard) * plan.payload_max;
  const std::uint64_t bytes = std::min(plan.payload_max, plan.layout_bytes - offset);
  const auto *payload = reinterpret_cast<const std::uint8_t *>(numerator) + offset;
  FrameHeader header{};
  header.type = MessageType::contribution_data;
  header.run_key = plan.run_key;
  header.fence_epoch = plan.fence_epoch;
  header.generation = plan.generation;
  header.attempt = plan.attempt;
  header.shard_id = shard;
  header.owner_epoch = stale_owner_epoch ? plan.owner_epoch - 1 : plan.owner_epoch;
  header.contribution_seq = contribution.contribution_seq;
  header.worker_key = contribution.worker_key;
  header.incarnation = contribution.incarnation;
  header.layout_digest = plan.layout_digest;
  header.base_digest = plan.base_digest;
  header.payload_digest = emender::ndp::sha256(payload, static_cast<std::size_t>(bytes));
  header.contribution_digest = contribution.contribution_digest;
  header.payload_offset = offset;
  header.payload_bytes = bytes;
  header.shard_bytes = bytes;
  header.weight = contribution.weight;
  header.message_seq = bad_checksum
      ? (message_seq | (UINT64_C(1) << 63U)) : message_seq;
  header.deadline_unix_ns = plan.deadline_unix_ns;
  header.chunk_index = shard;
  header.chunk_count = plan.shard_count;
  std::vector<std::uint8_t> encoded;
  require_code(emender::ndp::encode_frame(header, payload,
                                          static_cast<std::size_t>(bytes), &encoded),
               "encode_contribution");
  return encoded;
}

DecodedFrame local_contribution_frame(const GenerationPlan &plan,
                                      const Contribution &contribution,
                                      const double *numerator,
                                      std::uint32_t shard,
                                      std::uint64_t message_seq) {
  const auto encoded = contribution_frame(plan, contribution, numerator,
                                          shard, message_seq, false, false);
  DecodedFrame decoded{};
  require_code(emender::ndp::decode_frame(encoded.data(), encoded.size(),
                                          plan.payload_max, &decoded),
               "decode_local_contribution");
  return decoded;
}

std::vector<std::uint8_t> result_frame(const GenerationPlan &plan,
                                       const std::vector<std::uint8_t> &payload,
                                       std::uint32_t shard,
                                       std::uint64_t message_seq) {
  FrameHeader header{};
  header.type = MessageType::result_data;
  header.run_key = plan.run_key;
  header.fence_epoch = plan.fence_epoch;
  header.generation = plan.generation;
  header.attempt = plan.attempt;
  header.shard_id = shard;
  header.owner_epoch = plan.owner_epoch;
  header.worker_key = plan.owner_worker_key;
  header.incarnation = plan.owner_incarnation;
  header.layout_digest = plan.layout_digest;
  header.base_digest = plan.base_digest;
  header.payload_digest = emender::ndp::sha256(payload.data(), payload.size());
  header.payload_offset = static_cast<std::uint64_t>(shard) * plan.payload_max;
  header.payload_bytes = payload.size();
  header.shard_bytes = payload.size();
  header.weight = kGlobalWeight;
  header.message_seq = message_seq;
  header.deadline_unix_ns = plan.deadline_unix_ns;
  header.chunk_index = shard;
  header.chunk_count = plan.shard_count;
  std::vector<std::uint8_t> encoded;
  require_code(emender::ndp::encode_frame(header, payload.data(), payload.size(),
                                          &encoded), "encode_result");
  return encoded;
}

FrameHeader generation_signal(const GenerationPlan &plan, MessageType type,
                              std::uint64_t message_seq) {
  FrameHeader header{};
  header.type = type;
  header.run_key = plan.run_key;
  header.fence_epoch = plan.fence_epoch;
  header.generation = plan.generation;
  header.attempt = plan.attempt;
  header.owner_epoch = plan.owner_epoch;
  header.worker_key = plan.owner_worker_key;
  header.incarnation = plan.owner_incarnation;
  header.layout_digest = plan.layout_digest;
  header.base_digest = plan.base_digest;
  header.message_seq = message_seq;
  header.deadline_unix_ns = plan.deadline_unix_ns;
  return header;
}

bool generation_signal_matches(const FrameHeader &header,
                               const GenerationPlan &plan) {
  return same_key(header.run_key, plan.run_key) &&
         header.fence_epoch == plan.fence_epoch &&
         header.generation == plan.generation &&
         header.attempt == plan.attempt &&
         header.owner_epoch == plan.owner_epoch &&
         emender::ndp::constant_time_equal(header.layout_digest,
                                            plan.layout_digest) &&
         emender::ndp::constant_time_equal(header.base_digest,
                                            plan.base_digest) &&
         header.shard_id == emender::ndp::kNoShard &&
         header.payload_bytes == 0 &&
         header.deadline_unix_ns > emender::ndp::unix_time_ns();
}

Digest result_root(const GenerationPlan &plan,
                   const std::vector<std::uint8_t> &aggregate) {
  std::vector<std::uint8_t> encoded;
  encoded.insert(encoded.end(), plan.run_key.begin(), plan.run_key.end());
  append_u64(&encoded, plan.fence_epoch);
  append_u64(&encoded, plan.generation);
  append_u32(&encoded, plan.attempt);
  append_u64(&encoded, plan.owner_epoch);
  encoded.insert(encoded.end(), plan.layout_digest.begin(), plan.layout_digest.end());
  encoded.insert(encoded.end(), plan.base_digest.begin(), plan.base_digest.end());
  append_u64(&encoded, kGlobalWeight);
  for (std::uint32_t shard = 0; shard != plan.shard_count; ++shard) {
    const std::uint64_t offset = static_cast<std::uint64_t>(shard) * plan.payload_max;
    const std::uint64_t bytes = std::min(plan.payload_max, plan.layout_bytes - offset);
    append_u32(&encoded, shard);
    append_u64(&encoded, bytes);
    const Digest shard_digest = emender::ndp::sha256(
        aggregate.data() + static_cast<std::size_t>(offset),
        static_cast<std::size_t>(bytes));
    encoded.insert(encoded.end(), shard_digest.begin(), shard_digest.end());
  }
  const std::string_view domain{"emender-ndp-result-v1\0", 22};
  return emender::ndp::sha256_domain(domain, encoded.data(), encoded.size());
}

class ContributionStage {
 public:
  ContributionStage(const Options &options, NativeTransport *transport,
                    GenerationPlan plan, std::vector<std::size_t> owners,
                    const std::vector<double> *local_numerator,
                    bool inject_clean_rejects, bool inject_old_epoch,
                    bool prime_only, std::uint32_t prime_shard)
      : options_(options), transport_(transport), plan_(std::move(plan)),
        owners_(std::move(owners)), local_numerator_(local_numerator),
        owner_(std::make_unique<OwnerEngine>(plan_)),
        owner_next_(plan_.shard_count, 0), remote_acked_(plan_.shard_count, false),
        injection_state_(plan_.shard_count, 0), active_(plan_.shard_count, false),
        inject_clean_rejects_(inject_clean_rejects),
        inject_old_epoch_(inject_old_epoch), prime_only_(prime_only),
        prime_shard_(prime_shard) {
    require_code(owner_->freeze(), "owner_freeze");
    for (std::uint32_t shard = 0; shard != plan_.shard_count; ++shard) {
      if (owners_[shard] == options_.rank && (!prime_only_ || shard == prime_shard_)) {
        owned_.push_back(shard);
      }
    }
  }

  void run() {
    activate_more();
    for (;;) {
      if (complete()) {
        if (prime_only_) break;
        if (!result_announced_) send_result_announce();
        if (peer_result_announced_) break;
      }
      if (emender::ndp::unix_time_ns() >= plan_.deadline_unix_ns) {
        fail("contribution stage exceeded its absolute deadline");
      }
      const auto frames = transport_->receive(10);
      for (const auto &frame : frames) dispatch(frame);
      activate_more();
    }
    transport_->wait_idle(plan_.deadline_unix_ns);
  }

  std::unique_ptr<OwnerEngine> take_owner() { return std::move(owner_); }
  std::uint64_t contribution_tx_bytes() const noexcept { return contribution_tx_bytes_; }
  std::uint64_t stale_rejects() const noexcept { return stale_rejects_; }
  std::uint64_t checksum_rejects() const noexcept { return checksum_rejects_; }

 private:
  const Contribution &local_contribution() const {
    return plan_.accepted[options_.rank];
  }

  bool contribution_is_local(const Contribution &contribution) const {
    return same_key(contribution.worker_key, local_contribution().worker_key);
  }

  bool complete() const {
    if (prime_only_) {
      if (options_.rank == 1) return prime_route_ack_;
      return remote_acked_[prime_shard_] && prime_route_ack_sent_;
    }
    for (const std::uint32_t shard : owned_) {
      if (owner_next_[shard] != plan_.accepted.size()) return false;
    }
    for (std::uint32_t shard = 0; shard != plan_.shard_count; ++shard) {
      if (owners_[shard] != options_.rank && !remote_acked_[shard]) return false;
    }
    return true;
  }

  void activate_more() {
    while (active_count_ < 2 && next_owned_ < owned_.size()) {
      const std::uint32_t shard = owned_[next_owned_++];
      active_[shard] = true;
      ++active_count_;
      advance_owner(shard);
    }
  }

  void owner_completed(std::uint32_t shard) {
    if (active_[shard]) {
      active_[shard] = false;
      --active_count_;
    }
  }

  void advance_owner(std::uint32_t shard) {
    const auto order = emender::ndp::contribution_order(plan_, shard);
    while (owner_next_[shard] < order.size()) {
      FrameHeader credit{};
      require_code(owner_->grant_next(shard, ++credit_epoch_, &credit),
                   "owner_grant_next");
      const Contribution &expected = order[owner_next_[shard]];
      if (!contribution_is_local(expected)) {
        send_control_header(credit);
        return;
      }
      DecodedFrame local = local_contribution_frame(
          plan_, local_contribution(), local_numerator_->data(), shard,
          ++message_sequence_);
      const auto receipt = owner_->apply(local, emender::ndp::unix_time_ns());
      if (receipt.status != WireStatus::applied) fail("local owner application rejected");
      ++owner_next_[shard];
    }
    owner_completed(shard);
  }

  void send_contribution(std::uint32_t shard, int variant) {
    bool stale_epoch = false;
    bool bad_checksum = false;
    if (variant == 1) stale_epoch = true;
    if (variant == 2) bad_checksum = true;
    auto frame = contribution_frame(plan_, local_contribution(),
                                    local_numerator_->data(), shard,
                                    ++message_sequence_, stale_epoch,
                                    bad_checksum);
    contribution_tx_bytes_ += shard_bytes(options_, shard);
    transport_->send(frame, plan_.deadline_unix_ns);
  }

  void send_control_header(FrameHeader header) {
    header.worker_key = plan_.owner_worker_key;
    header.incarnation = plan_.owner_incarnation;
    transport_->send(encode_header(header), plan_.deadline_unix_ns);
  }

  void send_result_announce() {
    FrameHeader announcement = generation_signal(
        plan_, MessageType::result_announce, ++message_sequence_);
    send_control_header(announcement);
    result_announced_ = true;
  }

  void handle_result_announce(const FrameHeader &header) {
    if (prime_only_ || !generation_signal_matches(header, plan_)) {
      fail("result announce did not match the frozen generation");
    }
    peer_result_announced_ = true;
  }

  void handle_credit(const FrameHeader &header) {
    const std::uint32_t shard = header.shard_id;
    if (shard >= plan_.shard_count || owners_[shard] == options_.rank ||
        !emender::ndp::constant_time_equal(header.contribution_digest,
                                           local_contribution().contribution_digest)) {
      fail("received credit outside the frozen local contribution plan");
    }
    if (inject_clean_rejects_ && injection_state_[shard] == 0) {
      injection_state_[shard] = 1;
      inject_clean_rejects_ = false;
      send_contribution(shard, 1);
    } else if (inject_old_epoch_ && injection_state_[shard] == 0) {
      injection_state_[shard] = 3;
      inject_old_epoch_ = false;
      send_contribution(shard, 1);
    } else {
      injection_state_[shard] = 5;
      send_contribution(shard, 0);
    }
  }

  void handle_contribution(const DecodedFrame &frame) {
    const std::uint32_t shard = frame.header.shard_id;
    if (shard >= plan_.shard_count || owners_[shard] != options_.rank) {
      fail("received contribution for a non-owned shard");
    }
    DecodedFrame candidate = frame;
    if ((candidate.header.message_seq & (UINT64_C(1) << 63U)) != 0 &&
        !candidate.payload.empty()) {
      // The gate corrupts the already-received native RX payload. This is the
      // same receiver-side fault boundary as a provider/data-integrity error;
      // sender-side frame validation remains fail-closed.
      candidate.payload[0] ^= UINT8_C(0xff);
    }
    const auto receipt = owner_->apply(candidate, emender::ndp::unix_time_ns());
    send_control_header(receipt.header);
    if (receipt.status == WireStatus::applied || receipt.status == WireStatus::duplicate) {
      ++owner_next_[shard];
      advance_owner(shard);
    } else if (receipt.reason == WireReason::stale_owner_epoch ||
               receipt.reason == WireReason::stale_fence ||
               receipt.reason == WireReason::stale_generation_or_attempt) {
      ++stale_rejects_;
    } else if (receipt.reason == WireReason::checksum) {
      ++checksum_rejects_;
    } else {
      fail("unexpected rejecting contribution receipt");
    }
  }

  void handle_receipt(const FrameHeader &header) {
    const std::uint32_t shard = header.shard_id;
    if (shard >= plan_.shard_count || owners_[shard] == options_.rank) {
      fail("received receipt outside a remote-owned shard");
    }
    if (header.status == WireStatus::rejected) {
      if (injection_state_[shard] == 1 &&
          (header.reason == WireReason::stale_owner_epoch ||
           header.reason == WireReason::stale_fence)) {
        injection_state_[shard] = 2;
        send_contribution(shard, 2);
        return;
      }
      if (injection_state_[shard] == 2 && header.reason == WireReason::checksum) {
        injection_state_[shard] = 5;
        send_contribution(shard, 0);
        return;
      }
      if (injection_state_[shard] == 3 &&
          header.reason == WireReason::stale_owner_epoch) {
        injection_state_[shard] = 5;
        send_contribution(shard, 0);
        return;
      }
      fail("unexpected injected rejection sequence");
    }
    if (header.status != WireStatus::applied && header.status != WireStatus::duplicate) {
      fail("remote application did not return an applied receipt");
    }
    remote_acked_[shard] = true;
    if (prime_only_ && shard == prime_shard_) {
      FrameHeader acknowledgement{};
      acknowledgement.type = MessageType::route_probe_ack;
      acknowledgement.run_key = plan_.run_key;
      acknowledgement.fence_epoch = plan_.fence_epoch;
      acknowledgement.generation = plan_.generation;
      acknowledgement.attempt = plan_.attempt;
      acknowledgement.owner_epoch = plan_.owner_epoch;
      acknowledgement.deadline_unix_ns = plan_.deadline_unix_ns;
      acknowledgement.message_seq = ++message_sequence_;
      send_control_header(acknowledgement);
      prime_route_ack_sent_ = true;
    }
  }

  void dispatch(const DecodedFrame &frame) {
    switch (frame.header.type) {
      case MessageType::credit: handle_credit(frame.header); break;
      case MessageType::contribution_data: handle_contribution(frame); break;
      case MessageType::receipt: handle_receipt(frame.header); break;
      case MessageType::route_probe_ack: prime_route_ack_ = true; break;
      case MessageType::result_announce: handle_result_announce(frame.header); break;
      default: fail("unexpected frame in contribution stage");
    }
  }

  const Options &options_;
  NativeTransport *transport_;
  GenerationPlan plan_;
  std::vector<std::size_t> owners_;
  const std::vector<double> *local_numerator_;
  std::unique_ptr<OwnerEngine> owner_;
  std::vector<std::size_t> owner_next_;
  std::vector<bool> remote_acked_;
  std::vector<int> injection_state_;
  std::vector<bool> active_;
  std::vector<std::uint32_t> owned_;
  std::size_t next_owned_{0};
  std::size_t active_count_{0};
  std::uint64_t credit_epoch_{0};
  std::uint64_t message_sequence_{0};
  std::uint64_t contribution_tx_bytes_{0};
  std::uint64_t stale_rejects_{0};
  std::uint64_t checksum_rejects_{0};
  bool inject_clean_rejects_{false};
  bool inject_old_epoch_{false};
  bool prime_only_{false};
  std::uint32_t prime_shard_{0};
  bool prime_route_ack_{false};
  bool prime_route_ack_sent_{false};
  bool result_announced_{false};
  bool peer_result_announced_{false};
};

Sample run_generation(const Options &options, NativeTransport *transport,
                      const Key128 &run_key,
                      const std::array<Key128, 2> &contribution_incarnations,
                      const std::array<Key128, 2> &endpoint_incarnations,
                      const std::vector<double> &local_numerator,
                      std::uint64_t generation, std::uint64_t owner_epoch,
                      bool inject_clean_rejects, bool inject_old_epoch,
                      std::uint64_t absolute_deadline) {
  GenerationPlan mapping_plan{};
  mapping_plan.run_key = run_key;
  const std::uint32_t shard_count = static_cast<std::uint32_t>(
      (options.layout_bytes + options.payload_max - 1) / options.payload_max);
  const auto owners = emender::ndp::deterministic_owner_map(
      run_key, 1, generation, 1, owner_epoch, 2, shard_count);
  GenerationPlan plan = make_plan(options, run_key, contribution_incarnations,
                                  endpoint_incarnations, generation, owner_epoch,
                                  absolute_deadline, owners);
  std::uint64_t admitted_resident = 0;
  require_code(emender::ndp::validate_plan_bounds(plan, &admitted_resident),
               "validate_generation_plan");
  (void)admitted_resident;

  const MetricSnapshot before = transport->metrics();
  const auto started = std::chrono::steady_clock::now();
  ContributionStage contribution(options, transport, plan, owners,
                                 &local_numerator, inject_clean_rejects,
                                 inject_old_epoch, false, 0);
  contribution.run();
  std::unique_ptr<OwnerEngine> owner = contribution.take_owner();
  require_code(owner->finalize(emender::ndp::unix_time_ns()), "owner_finalize");

  ResultAssembler assembler(plan);
  std::uint64_t redistribution_tx = 0;
  std::uint64_t sequence = 1000000;
  bool peer_redistribution_complete = false;
  const auto handle_generation_goodbye = [&](const FrameHeader &header) {
    if (!generation_signal_matches(header, plan)) {
      fail("generation goodbye did not match the frozen generation");
    }
    peer_redistribution_complete = true;
  };
  std::vector<std::uint32_t> remote_shards;
  for (std::uint32_t shard = 0; shard != plan.shard_count; ++shard) {
    if (owners[shard] == options.rank) {
      const auto *payload = owner->result(shard);
      if (payload == nullptr) fail("owned result shard is missing");
      const auto encoded = result_frame(plan, *payload, shard, ++sequence);
      DecodedFrame decoded{};
      require_code(emender::ndp::decode_frame(encoded.data(), encoded.size(),
                                              plan.payload_max, &decoded),
                   "decode_local_result");
      require_code(assembler.accept(decoded, emender::ndp::unix_time_ns()),
                   "assemble_local_result");
    } else {
      remote_shards.push_back(shard);
    }
  }

  for (const std::uint32_t requested : remote_shards) {
    FrameHeader fetch{};
    fetch.type = MessageType::fetch;
    fetch.run_key = plan.run_key;
    fetch.fence_epoch = plan.fence_epoch;
    fetch.generation = plan.generation;
    fetch.attempt = plan.attempt;
    fetch.shard_id = requested;
    fetch.owner_epoch = plan.owner_epoch;
    fetch.layout_digest = plan.layout_digest;
    fetch.base_digest = plan.base_digest;
    fetch.payload_offset = static_cast<std::uint64_t>(requested) * plan.payload_max;
    fetch.payload_bytes = shard_bytes(options, requested);
    fetch.shard_bytes = fetch.payload_bytes;
    fetch.weight = kGlobalWeight;
    fetch.message_seq = ++sequence;
    fetch.deadline_unix_ns = plan.deadline_unix_ns;
    fetch.chunk_index = requested;
    fetch.chunk_count = plan.shard_count;
    fetch.worker_key = plan.owner_worker_key;
    fetch.incarnation = plan.owner_incarnation;
    transport->send(encode_header(fetch), plan.deadline_unix_ns);
    bool received_requested = false;
    while (!received_requested) {
      for (const auto &frame : transport->receive(10)) {
        if (frame.header.type == MessageType::fetch) {
          const auto *payload = owner->result(frame.header.shard_id);
          if (payload == nullptr) fail("peer fetched a non-owned result shard");
          auto encoded = result_frame(plan, *payload, frame.header.shard_id,
                                      ++sequence);
          redistribution_tx += payload->size();
          transport->send(encoded, plan.deadline_unix_ns);
        } else if (frame.header.type == MessageType::result_data) {
          require_code(assembler.accept(frame, emender::ndp::unix_time_ns()),
                       "assemble_remote_result");
          if (frame.header.shard_id == requested) received_requested = true;
        } else if (frame.header.type == MessageType::goodbye) {
          handle_generation_goodbye(frame.header);
        } else {
          fail("unexpected frame during redistribution: type=" +
               std::to_string(static_cast<std::uint16_t>(frame.header.type)) +
               " generation=" + std::to_string(frame.header.generation));
        }
      }
    }
  }
  while (!assembler.complete()) {
    for (const auto &frame : transport->receive(10)) {
      if (frame.header.type == MessageType::fetch) {
        const auto *payload = owner->result(frame.header.shard_id);
        if (payload == nullptr) fail("late fetch named a non-owned shard");
        auto encoded = result_frame(plan, *payload, frame.header.shard_id, ++sequence);
        redistribution_tx += payload->size();
        transport->send(encoded, plan.deadline_unix_ns);
      } else if (frame.header.type == MessageType::result_data) {
        require_code(assembler.accept(frame, emender::ndp::unix_time_ns()),
                     "assemble_late_result");
      } else if (frame.header.type == MessageType::goodbye) {
        handle_generation_goodbye(frame.header);
      } else {
        fail("unexpected late redistribution frame: type=" +
             std::to_string(static_cast<std::uint16_t>(frame.header.type)) +
             " generation=" + std::to_string(frame.header.generation));
      }
    }
  }
  const auto send_generation_goodbye = [&]() {
    FrameHeader goodbye = generation_signal(plan, MessageType::goodbye, ++sequence);
    transport->send(encode_header(goodbye), plan.deadline_unix_ns);
  };
  send_generation_goodbye();
  while (!peer_redistribution_complete) {
    if (emender::ndp::unix_time_ns() >= plan.deadline_unix_ns) {
      fail("generation completion exchange exceeded its absolute deadline");
    }
    for (const auto &frame : transport->receive(10)) {
      if (frame.header.type == MessageType::fetch) {
        const auto *payload = owner->result(frame.header.shard_id);
        if (payload == nullptr) fail("completion fetch named a non-owned shard");
        auto encoded = result_frame(plan, *payload, frame.header.shard_id, ++sequence);
        redistribution_tx += payload->size();
        transport->send(encoded, plan.deadline_unix_ns);
      } else if (frame.header.type == MessageType::goodbye) {
        handle_generation_goodbye(frame.header);
      } else {
        fail("unexpected frame during generation completion: type=" +
             std::to_string(static_cast<std::uint16_t>(frame.header.type)) +
             " generation=" + std::to_string(frame.header.generation));
      }
    }
  }
  transport->wait_idle(plan.deadline_unix_ns);
  const auto finished = std::chrono::steady_clock::now();
  const MetricSnapshot after = transport->metrics();
  if (after.value.cq_errors != before.value.cq_errors ||
      after.value.route_errors != before.value.route_errors ||
      after.value.in_flight_bytes != 0 || after.value.retained_bytes != 0) {
    fail("transport metrics did not return cleanly to the release floor");
  }

  Sample sample{};
  sample.transfer_redistribution_seconds =
      std::chrono::duration<double>(finished - started).count();
  sample.useful_tx_bytes = after.value.useful_tx_bytes - before.value.useful_tx_bytes;
  sample.useful_rx_bytes = after.value.useful_rx_bytes - before.value.useful_rx_bytes;
  sample.wire_tx_bytes = after.value.wire_tx_bytes - before.value.wire_tx_bytes;
  sample.wire_rx_bytes = after.value.wire_rx_bytes - before.value.wire_rx_bytes;
  sample.retries = after.value.retries - before.value.retries;
  sample.released_bytes = after.value.released_bytes - before.value.released_bytes;
  sample.contribution_tx_bytes = contribution.contribution_tx_bytes();
  sample.redistribution_tx_bytes = redistribution_tx;
  sample.stale_rejects = contribution.stale_rejects();
  sample.checksum_rejects = contribution.checksum_rejects();
  const Digest root = result_root(plan, assembler.aggregate());
  sample.result_root = emender::ndp::hex(root.data(), root.size());
  const Digest payload = emender::ndp::sha256(assembler.aggregate().data(),
                                              assembler.aggregate().size());
  sample.result_payload_sha256 = emender::ndp::hex(payload.data(), payload.size());
  assembler.release();
  owner->cancel();
  return sample;
}

std::uint32_t first_owner_one_shard(const Options &options, const Key128 &run_key,
                                    std::uint64_t generation) {
  const std::uint32_t shards = static_cast<std::uint32_t>(
      (options.layout_bytes + options.payload_max - 1) / options.payload_max);
  const auto owners = emender::ndp::deterministic_owner_map(
      run_key, 1, generation, 1, 1, 2, shards);
  for (std::uint32_t shard = 0; shard != shards; ++shard) {
    if (owners[shard] == 1) return shard;
  }
  fail("two-owner plan did not assign a shard to owner one");
}

void run_fault_prime(const Options &options, NativeTransport *transport,
                     const Key128 &run_key,
                     const std::array<Key128, 2> &contribution_incarnations,
                     const std::array<Key128, 2> &endpoint_incarnations,
                     const std::vector<double> &local_numerator,
                     std::uint64_t generation, std::uint32_t prime_shard,
                     std::uint64_t deadline_unix_ns) {
  const std::uint32_t shards = static_cast<std::uint32_t>(
      (options.layout_bytes + options.payload_max - 1) / options.payload_max);
  const auto owners = emender::ndp::deterministic_owner_map(
      run_key, 1, generation, 1, 1, 2, shards);
  GenerationPlan plan = make_plan(options, run_key, contribution_incarnations,
                                  endpoint_incarnations, generation, 1,
                                  deadline_unix_ns, owners);
  ContributionStage prime(options, transport, plan, owners, &local_numerator,
                          false, false, true, prime_shard);
  prime.run();
}

void write_summary(const Options &options, const std::vector<Sample> &warmups,
                   const std::vector<Sample> &samples,
                   const MetricSnapshot &metrics,
                   std::uint64_t baseline_rss, std::uint64_t high_water_rss,
                   std::uint64_t post_release_rss,
                   std::uint64_t local_reduction_seconds_ns,
                   std::uint64_t admitted_resident_bytes,
                   const std::array<Key128, 2> &initial_endpoint_incarnations,
                   const std::array<Key128, 2> &final_endpoint_incarnations,
                   std::uint64_t fault_replay_bytes,
                   std::uint32_t fault_reassignment_count,
                   std::uint32_t old_epoch_rejects) {
  std::ofstream output(options.output, std::ios::trunc);
  if (!output) fail("cannot create node summary: " + options.output);
  output << std::setprecision(17);
  output << "{\"schema\":\"emender-native-dataplane-node-gate-v1\","
         << "\"status\":\"passed\",\"rank\":" << options.rank
         << ",\"mode\":\"" << options.mode << "\","
         << "\"provider\":\"" << json_escape(options.provider) << "\","
         << "\"endpoint_type\":\"FI_EP_RDM\","
         << "\"production_provider\":" << (options.production ? "true" : "false")
         << ",\"run_id\":\"" << json_escape(options.run_id) << "\","
         << "\"payload_id\":\"" << json_escape(options.payload_id) << "\","
         << "\"layout_bytes\":" << options.layout_bytes
         << ",\"total_elements\":" << options.layout_bytes / 8
         << ",\"payload_max\":" << options.payload_max
         << ",\"shard_count\":"
         << (options.layout_bytes + options.payload_max - 1) / options.payload_max
         << ",\"trainers_per_node\":8,\"node_weight\":"
         << (options.rank == 0 ? kNode0Weight : kNode1Weight)
         << ",\"global_weight\":" << kGlobalWeight
         << ",\"local_reduction_input_bytes\":"
         << options.layout_bytes / 8 * sizeof(float) * kTrainers
         << ",\"local_reduction_seconds\":"
         << static_cast<double>(local_reduction_seconds_ns) / 1.0e9
         << ",\"admitted_resident_bytes\":" << admitted_resident_bytes
         << ",\"baseline_rss_bytes\":" << baseline_rss
         << ",\"rss_high_water_bytes\":" << high_water_rss
         << ",\"post_release_rss_bytes\":" << post_release_rss
         << ",\"python_dense_socket_bytes\":0,\"trainer_spool_bytes\":0,"
         << "\"trainer_spool_files\":0,\"disk_replay_bytes\":0,"
         << "\"handoff_full_copy_bytes\":0,\"central_full_model_broker\":false,"
         << "\"mpi_collectives\":0,\"all_rank_barriers\":0,"
         << "\"fault_replay_bytes\":" << fault_replay_bytes
         << ",\"fault_reassignment_count\":" << fault_reassignment_count
         << ",\"old_epoch_rejects\":" << old_epoch_rejects
         << ",\"partial_commit\":false,"
         << "\"initial_endpoint_incarnation\":\""
         << emender::ndp::hex(initial_endpoint_incarnations[options.rank].data(), 16)
         << "\",\"final_endpoint_incarnation\":\""
         << emender::ndp::hex(final_endpoint_incarnations[options.rank].data(), 16)
         << "\",\"transport\":{"
         << "\"useful_tx_bytes\":" << metrics.value.useful_tx_bytes
         << ",\"useful_rx_bytes\":" << metrics.value.useful_rx_bytes
         << ",\"wire_tx_bytes\":" << metrics.value.wire_tx_bytes
         << ",\"wire_rx_bytes\":" << metrics.value.wire_rx_bytes
         << ",\"retries\":" << metrics.value.retries
         << ",\"cq_errors\":" << metrics.value.cq_errors
         << ",\"route_errors\":" << metrics.value.route_errors
         << ",\"in_flight_bytes\":" << metrics.value.in_flight_bytes
         << ",\"in_flight_high_water\":" << metrics.value.in_flight_high_water
         << ",\"retained_bytes\":" << metrics.value.retained_bytes
         << ",\"retained_high_water\":" << metrics.value.retained_high_water
         << ",\"released_bytes\":" << metrics.value.released_bytes
         << ",\"tx_slot_high_water\":" << metrics.value.tx_slot_high_water
         << ",\"rx_slot_high_water\":" << metrics.value.rx_slot_high_water
         << "},\"warmups\":[";
  const auto write_samples = [&](const std::vector<Sample> &values) {
    for (std::size_t index = 0; index != values.size(); ++index) {
      if (index != 0) output << ',';
      const Sample &sample = values[index];
      output << "{\"transfer_redistribution_seconds\":"
             << sample.transfer_redistribution_seconds
             << ",\"useful_tx_bytes\":" << sample.useful_tx_bytes
             << ",\"useful_rx_bytes\":" << sample.useful_rx_bytes
             << ",\"wire_tx_bytes\":" << sample.wire_tx_bytes
             << ",\"wire_rx_bytes\":" << sample.wire_rx_bytes
             << ",\"retries\":" << sample.retries
             << ",\"released_bytes\":" << sample.released_bytes
             << ",\"contribution_tx_bytes\":" << sample.contribution_tx_bytes
             << ",\"redistribution_tx_bytes\":" << sample.redistribution_tx_bytes
             << ",\"stale_rejects\":" << sample.stale_rejects
             << ",\"checksum_rejects\":" << sample.checksum_rejects
             << ",\"result_root\":\"" << sample.result_root
             << "\",\"result_payload_sha256\":\""
             << sample.result_payload_sha256 << "\"}";
    }
  };
  write_samples(warmups);
  output << "],\"samples\":[";
  write_samples(samples);
  output << "]}\n";
  if (!output) fail("node summary write failed");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::uint64_t started_unix_ns = emender::ndp::unix_time_ns();
    const std::uint64_t absolute_deadline = started_unix_ns +
        options.deadline_seconds * UINT64_C(1000000000);
    const std::uint64_t expires_unix_ns = absolute_deadline + UINT64_C(60000000000);
    const Key128 run_key = key_from("run:" + options.run_id);
    std::array<Key128, 2> endpoint_incarnations{{
        key_from("endpoint-incarnation:" + options.run_id + ":0:1"),
        key_from("endpoint-incarnation:" + options.run_id + ":1:1")}};
    std::array<Key128, 2> contribution_incarnations = endpoint_incarnations;
    const auto initial_endpoint_incarnations = endpoint_incarnations;
    const Key128 worker_key = key_from("worker:" + options.run_id + ":" +
                                       std::to_string(options.rank));
    auto transport = std::make_unique<NativeTransport>(
        options, run_key, worker_key, endpoint_incarnations[options.rank], 1,
        expires_unix_ns, absolute_deadline);
    const int control_fd = connect_controller(options.controller_host,
                                              options.controller_port,
                                              absolute_deadline);
    PeerRecord peer = exchange_endpoint(control_fd, options.rank, 0,
                                        transport->endpoint_record());
    endpoint_incarnations[peer.rank] = peer.incarnation;
    transport->install_peer(peer);

    const std::uint64_t baseline_rss = rss_bytes();
    const auto reduction_started = std::chrono::steady_clock::now();
    std::vector<double> local_numerator = local_reduce(options);
    const auto reduction_finished = std::chrono::steady_clock::now();
    const std::uint64_t reduction_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            reduction_finished - reduction_started).count());

    std::vector<Sample> warmups;
    std::vector<Sample> samples;
    std::uint64_t fault_replay_bytes = 0;
    std::uint32_t fault_reassignment_count = 0;
    std::uint32_t old_epoch_rejects = 0;
    std::uint64_t generation = 0;
    for (std::uint32_t index = 0; index != options.warmup; ++index, ++generation) {
      warmups.push_back(run_generation(
          options, transport.get(), run_key, contribution_incarnations,
          endpoint_incarnations, local_numerator, generation, 1,
          index == 0, false, absolute_deadline));
    }

    if (options.mode == "fault") {
      const std::uint32_t prime_shard = first_owner_one_shard(options, run_key,
                                                              generation);
      run_fault_prime(options, transport.get(), run_key,
                      contribution_incarnations, endpoint_incarnations,
                      local_numerator, generation, prime_shard,
                      absolute_deadline);
      fault_replay_bytes = UINT64_C(2) * shard_bytes(options, prime_shard);
      fault_reassignment_count = 1;

      if (options.rank == 1) {
        transport.reset();
        endpoint_incarnations[1] = key_from(
            "endpoint-incarnation:" + options.run_id + ":1:2");
        contribution_incarnations[1] = endpoint_incarnations[1];
        transport = std::make_unique<NativeTransport>(
            options, run_key, worker_key, endpoint_incarnations[1], 2,
            expires_unix_ns, absolute_deadline);
      }
      PeerRecord replacement = exchange_endpoint(
          control_fd, options.rank, 1, transport->endpoint_record());
      endpoint_incarnations[replacement.rank] = replacement.incarnation;
      contribution_incarnations[1] = endpoint_incarnations[1];
      if (options.rank == 0) transport->remove_peer();
      transport->install_peer(replacement);
      samples.push_back(run_generation(
          options, transport.get(), run_key, contribution_incarnations,
          endpoint_incarnations, local_numerator, generation, 2,
          false, options.rank == 0, absolute_deadline));
      old_epoch_rejects = static_cast<std::uint32_t>(samples.back().stale_rejects);
      ++generation;
    } else {
      for (std::uint32_t index = 0; index != options.generations;
           ++index, ++generation) {
        samples.push_back(run_generation(
            options, transport.get(), run_key, contribution_incarnations,
            endpoint_incarnations, local_numerator, generation, 1,
            false, false, absolute_deadline));
      }
    }

    transport->wait_idle(absolute_deadline);
    const MetricSnapshot final_metrics = transport->metrics();
    if (final_metrics.value.cq_errors != 0 || final_metrics.value.route_errors != 0 ||
        final_metrics.value.in_flight_bytes != 0 ||
        final_metrics.value.retained_bytes != 0 ||
        final_metrics.value.tx_slot_high_water > kSlots ||
        final_metrics.value.rx_slot_high_water > kSlots) {
      fail("terminal transport bounds/release invariant failed");
    }

    const std::uint32_t shard_count = static_cast<std::uint32_t>(
        (options.layout_bytes + options.payload_max - 1) / options.payload_max);
    const auto owners = emender::ndp::deterministic_owner_map(
        run_key, 1, 0, 1, 1, 2, shard_count);
    const GenerationPlan admission_plan = make_plan(
        options, run_key, contribution_incarnations, endpoint_incarnations,
        0, 1, absolute_deadline, owners);
    std::uint64_t admitted_resident = 0;
    require_code(emender::ndp::validate_plan_bounds(admission_plan,
                                                    &admitted_resident),
                 "terminal_admission_check");

    const std::uint64_t high_water_rss = rss_high_water_bytes();
    local_numerator.clear();
    local_numerator.shrink_to_fit();
    (void)::malloc_trim(0);
    const std::uint64_t post_release_rss = rss_bytes();
    write_summary(options, warmups, samples, final_metrics, baseline_rss,
                  high_water_rss, post_release_rss, reduction_ns, admitted_resident,
                  initial_endpoint_incarnations, endpoint_incarnations,
                  fault_replay_bytes, fault_reassignment_count,
                  old_epoch_rejects);
    ::close(control_fd);
    std::cout << "native_frontier_2n_gate=passed rank=" << options.rank
              << " mode=" << options.mode << " output=" << options.output << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "native_frontier_2n_gate=failed error=" << error.what() << '\n';
    return 1;
  }
}
