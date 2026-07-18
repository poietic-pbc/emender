#include "fabric.hpp"
#include "owner.hpp"
#include "protocol.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using namespace emender::ndp;

bool write_fragmented(int fd, const void *data, std::size_t bytes) {
  const auto *cursor = static_cast<const std::uint8_t *>(data);
  while (bytes != 0) {
    const std::size_t fragment = std::min<std::size_t>(bytes, 3);
    const ssize_t written = ::write(fd, cursor, fragment);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    cursor += written; bytes -= static_cast<std::size_t>(written);
  }
  return true;
}

bool read_exact(int fd, void *data, std::size_t bytes) {
  auto *cursor = static_cast<std::uint8_t *>(data);
  while (bytes != 0) {
    const ssize_t got = ::read(fd, cursor, bytes);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    cursor += got; bytes -= static_cast<std::size_t>(got);
  }
  return true;
}

bool exchange_name(int fd, const std::vector<std::uint8_t> &local,
                   std::vector<std::uint8_t> *remote) {
  const std::uint32_t size = static_cast<std::uint32_t>(local.size());
  if (!write_fragmented(fd, &size, sizeof(size)) ||
      !write_fragmented(fd, local.data(), local.size())) return false;
  std::uint32_t remote_size = 0;
  if (!read_exact(fd, &remote_size, sizeof(remote_size)) || remote_size == 0 ||
      remote_size > NDP_TRANSPORT_ENDPOINT_MAX) return false;
  remote->resize(remote_size);
  return read_exact(fd, remote->data(), remote->size());
}

Key128 key(std::uint8_t seed) {
  Key128 result{};
  for (std::size_t i = 0; i != result.size(); ++i) result[i] = static_cast<std::uint8_t>(seed + i);
  return result;
}

Digest digest(std::string value) {
  return sha256(reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
}

std::vector<std::uint8_t> doubles(double a, double b) {
  std::vector<std::uint8_t> result(2 * sizeof(double));
  std::memcpy(result.data(), &a, sizeof(double));
  std::memcpy(result.data() + sizeof(double), &b, sizeof(double));
  return result;
}

GenerationPlan owner_plan(std::uint64_t deadline) {
  GenerationPlan plan{};
  plan.run_key = key(1); plan.fence_epoch = 11; plan.generation = 5;
  plan.attempt = 1; plan.owner_epoch = 2;
  plan.layout_digest = digest("fabric-layout"); plan.base_digest = digest("fabric-base");
  plan.owner_worker_key = key(81); plan.owner_incarnation = key(101);
  plan.owner_endpoint_epoch = 1;
  plan.layout_bytes = 16; plan.payload_max = 16; plan.shard_count = 1;
  plan.tx_slots = 2; plan.rx_slots = 2; plan.owner_count = 1;
  plan.assigned_bytes = 16; plan.resident_limit_bytes = 128 * 1024 * 1024;
  plan.deadline_unix_ns = deadline; plan.assigned_shards = {0};
  Contribution a{}; a.worker_key = key(21); a.incarnation = key(41);
  a.contribution_seq = 1; a.contribution_digest = digest("fabric-contribution-a"); a.weight = 3;
  Contribution b{}; b.worker_key = key(61); b.incarnation = key(71);
  b.contribution_seq = 2; b.contribution_digest = digest("fabric-contribution-b"); b.weight = 5;
  plan.accepted = {a, b};
  return plan;
}

std::vector<std::uint8_t> contribution_frame(
    const GenerationPlan &plan, const Contribution &contribution,
    const std::vector<std::uint8_t> &payload, std::uint64_t message_seq,
    bool stale_fence = false) {
  FrameHeader h{}; h.type = MessageType::contribution_data; h.run_key = plan.run_key;
  h.fence_epoch = stale_fence ? plan.fence_epoch - 1 : plan.fence_epoch;
  h.generation = plan.generation; h.attempt = plan.attempt; h.shard_id = 0;
  h.owner_epoch = plan.owner_epoch; h.contribution_seq = contribution.contribution_seq;
  h.worker_key = contribution.worker_key; h.incarnation = contribution.incarnation;
  h.layout_digest = plan.layout_digest; h.base_digest = plan.base_digest;
  h.payload_digest = sha256(payload.data(), payload.size());
  h.contribution_digest = contribution.contribution_digest;
  h.payload_bytes = payload.size(); h.shard_bytes = payload.size();
  h.weight = contribution.weight; h.message_seq = message_seq;
  h.deadline_unix_ns = plan.deadline_unix_ns; h.chunk_count = 1;
  std::vector<std::uint8_t> result;
  if (encode_frame(h, payload.data(), payload.size(), &result) != NDP_T_OK) return {};
  return result;
}

int endpoint_process(int fd, bool child) {
  const std::uint64_t deadline = unix_time_ns() + UINT64_C(20) * 1000 * 1000 * 1000;
  FabricConfig config{};
  config.production = false; config.provider = "tcp;ofi_rxm";
  config.bind_node = "127.0.0.1"; config.tx_slots = 2; config.rx_slots = 2;
  config.payload_max = 4096; config.resident_limit_bytes = 1024 * 1024;
  config.deadline_unix_ns = deadline;
  FabricEndpoint endpoint(config);
  if (endpoint.start() != NDP_T_OK) return 10;
  const auto facts = endpoint.facts();
  if (facts.provider != "tcp;ofi_rxm" || facts.production_provider ||
      facts.endpoint_name.empty()) return 11;
  std::vector<std::uint8_t> remote;
  if (!exchange_name(fd, facts.endpoint_name, &remote)) return 12;
  std::uint64_t peer = 0;
  if (endpoint.add_peer(remote, deadline, &peer) != NDP_T_OK) return 13;

  // Cross sends exercise one persistent RDM endpoint, one AV route, fixed
  // pools, distinct completion queues, and the native progress thread in two
  // OS processes. The delivered sequence injects reordering, duplicate replay,
  // stale fencing, and receiver-side corruption into the real provider path.
  const GenerationPlan plan = owner_plan(deadline);
  OwnerEngine owner(plan);
  if (owner.freeze() != NDP_T_OK) return 14;
  FrameHeader credit{};
  if (owner.grant_next(0, 1, &credit) != NDP_T_OK) return 15;
  const auto order = contribution_order(plan, 0);
  const auto payload_for = [&](const Contribution &c) {
    return constant_time_equal(c.contribution_digest, plan.accepted[0].contribution_digest)
        ? doubles(6.0, 12.0) : doubles(20.0, 28.0);
  };
  std::vector<std::vector<std::uint8_t>> frames;
  frames.push_back(contribution_frame(plan, order[1], payload_for(order[1]), 20));
  frames.push_back(contribution_frame(plan, order[0], payload_for(order[0]), 10));
  frames.push_back(frames.back());
  frames.push_back(contribution_frame(plan, order[1], payload_for(order[1]), 21, true));
  frames.push_back(contribution_frame(plan, order[1], payload_for(order[1]), 21));
  frames.push_back(contribution_frame(plan, order[1], payload_for(order[1]), 99));
  for (const auto &frame : frames) {
    if (frame.empty() || endpoint.send(peer, frame.data(), frame.size(), deadline) != NDP_T_ACCEPTED) {
      return 29;
    }
  }

  std::size_t sent = 0;
  std::size_t received = 0;
  std::size_t sequence10 = 0;
  bool saw_reordering = false, saw_stale = false, saw_corruption = false;
  bool saw_preserved_short_buffer = false;
  while (unix_time_ns() < deadline && (sent < frames.size() || received < frames.size())) {
    std::vector<FabricEvent> events;
    if (endpoint.poll(&events, 8, 100) != NDP_T_OK) return 16;
    for (const auto &event : events) {
      if (event.kind == NDP_T_EVENT_CQ_ERROR) return 17;
      if (event.kind == NDP_T_EVENT_SENT) ++sent;
      if (event.kind != NDP_T_EVENT_RECEIVED) continue;
      ++received;
      if (!saw_preserved_short_buffer) {
        std::array<std::uint8_t, 8> too_small{};
        std::size_t required = 0;
        std::uint64_t short_source = 0;
        if (endpoint.receive(too_small.data(), too_small.size(), &required,
                             &short_source) != NDP_T_EBOUNDS ||
            required <= too_small.size()) return 36;
        saw_preserved_short_buffer = true;
      }
      std::vector<std::uint8_t> received_frame;
      std::uint64_t source = 0;
      if (endpoint.receive(&received_frame, &source) != NDP_T_OK || source != peer) return 19;
      DecodedFrame decoded{};
      if (decode_frame(received_frame.data(), received_frame.size(), 4096, &decoded) != NDP_T_OK) {
        return 20;
      }
      if (decoded.header.message_seq == 99) {
        received_frame.back() ^= 1;
        DecodedFrame corrupt{};
        if (decode_frame(received_frame.data(), received_frame.size(), 4096, &corrupt) !=
            NDP_T_ECHECKSUM) return 30;
        saw_corruption = true;
        continue;
      }
      const Receipt receipt = owner.apply(decoded, unix_time_ns());
      if (decoded.header.message_seq == 20) {
        if (receipt.reason != WireReason::no_credit) return 31;
        saw_reordering = true;
      } else if (decoded.header.message_seq == 10) {
        if (receipt.status != WireStatus::applied) return 32;
        if (++sequence10 == 1 && owner.grant_next(0, 2, &credit) != NDP_T_OK) return 33;
      } else if (decoded.header.fence_epoch != plan.fence_epoch) {
        if (receipt.reason != WireReason::stale_fence) return 34;
        saw_stale = true;
      } else if (decoded.header.message_seq == 21 && receipt.status != WireStatus::applied) {
        return 35;
      }
    }
  }
  if (sent < frames.size() || received < frames.size() || sequence10 != 2 ||
      !saw_reordering || !saw_stale || !saw_corruption ||
      !saw_preserved_short_buffer ||
      owner.metrics().duplicate_frames != 1 || owner.metrics().stale_rejects != 1 ||
      owner.metrics().no_credit_rejects != 1 || owner.finalize(unix_time_ns()) != NDP_T_OK) {
    return 18;
  }
  const auto counters = endpoint.counters();
  const std::uint64_t expected_wire = frames.size() * (kHeaderBytes + 16);
  if (counters.wire_tx_bytes < expected_wire || counters.wire_rx_bytes < expected_wire ||
      counters.useful_tx_bytes != frames.size() * 16 ||
      counters.useful_rx_bytes != frames.size() * 16 || counters.cq_errors != 0 ||
      counters.in_flight_bytes != 0 || counters.retained_bytes != 0 ||
      counters.released_bytes < 2 * expected_wire ||
      counters.tx_slot_high_water > 2 || counters.rx_slot_high_water > 2 ||
      counters.latency_count < frames.size()) {
    std::cerr << (child ? "child" : "parent")
              << " counters tx=" << counters.wire_tx_bytes
              << " rx=" << counters.wire_rx_bytes
              << " cq=" << counters.cq_errors
              << " inflight=" << counters.in_flight_bytes
              << " released=" << counters.released_bytes
              << " tx_hwm=" << counters.tx_slot_high_water
              << " rx_hwm=" << counters.rx_slot_high_water
              << " latency=" << counters.latency_count << '\n';
    return 21;
  }

  if (child) {
    const std::uint8_t ready = 1;
    if (!write_fragmented(fd, &ready, 1)) return 22;
    // Simulate abrupt peer loss: skip C++ destructors and let the kernel close
    // the endpoint. The surviving process must stay alive and bound its route.
    ::close(fd);
    _exit(0);
  }
  std::uint8_t ready = 0;
  if (!read_exact(fd, &ready, 1) || ready != 1) return 23;
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto &frame = frames.front();
  if (endpoint.send(peer, frame.data(), frame.size(), unix_time_ns() - 1) != NDP_T_EDEADLINE) {
    return 24;
  }
  if (endpoint.cancel(peer) != NDP_T_OK) return 25;
  if (endpoint.remove_peer(peer) != NDP_T_OK) return 26;
  if (endpoint.send(peer, frame.data(), frame.size(), deadline) != NDP_T_EROUTE) return 27;
  endpoint.shutdown();
  return 0;
}

}  // namespace

int main() {
  int sockets[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets) != 0) {
    std::perror("socketpair"); return 1;
  }
  const pid_t child = ::fork();
  if (child < 0) { std::perror("fork"); return 1; }
  if (child == 0) {
    ::close(sockets[0]);
    const int rc = endpoint_process(sockets[1], true);
    ::close(sockets[1]);
    _exit(rc);
  }
  ::close(sockets[1]);
  const int parent_rc = endpoint_process(sockets[0], false);
  ::close(sockets[0]);
  int status = 0;
  if (::waitpid(child, &status, 0) != child) { std::perror("waitpid"); return 2; }
  if (parent_rc != 0) {
    std::cerr << "parent endpoint process failed: " << parent_rc << '\n';
    return parent_rc;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "child endpoint process failed: status=" << status << '\n';
    return 3;
  }
  std::cout << "multi-process FI_EP_RDM provider test passed\n";
  return 0;
}
