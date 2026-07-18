#include "emender/ndp_transport.h"

#include "fabric.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <vector>

namespace emender::ndp {
namespace {

std::mutex registry_mutex;
struct PeerBinding {
  std::uint64_t peer_id{0};
  std::uint64_t endpoint_epoch{0};
  Key128 worker_key{};
  Key128 incarnation{};
  Digest endpoint_digest{};
};
struct TransportInstance {
  std::shared_ptr<FabricEndpoint> endpoint;
  std::uint64_t payload_max{0};
  std::mutex binding_mutex;
  std::optional<EndpointRecord> identity;
  std::map<std::string, PeerBinding> bindings;
  std::map<std::uint64_t, std::string> ids;
};
std::map<ndp_transport_t, std::shared_ptr<TransportInstance>> registry;
std::atomic<std::uint64_t> next_handle{1};

template <typename T>
bool valid_prefix(const T *value) {
  return value != nullptr && value->struct_size >= sizeof(T) &&
         value->abi_version == NDP_TRANSPORT_ABI_V1;
}

std::string span(const std::uint8_t *data, std::uint32_t bytes,
                 std::uint32_t capacity) {
  if (bytes > capacity || (bytes != 0 && std::memchr(data, 0, bytes) != nullptr)) return {};
  return std::string(reinterpret_cast<const char *>(data), bytes);
}

std::shared_ptr<TransportInstance> lookup(ndp_transport_t handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto found = registry.find(handle);
  return found == registry.end() ? nullptr : found->second;
}

std::string key_string(const std::uint8_t *key) {
  return std::string(reinterpret_cast<const char *>(key), 16);
}

bool nonzero_key(const std::uint8_t *key) {
  for (std::size_t i = 0; i != 16; ++i) if (key[i] != 0) return true;
  return false;
}

template <std::size_t N>
bool same_bytes(const std::array<std::uint8_t, N> &left,
                const std::uint8_t *right) {
  return std::equal(left.begin(), left.end(), right);
}

}  // namespace
}  // namespace emender::ndp

extern "C" {

uint32_t ndp_transport_abi_version(void) { return NDP_TRANSPORT_ABI_V1; }

const char *ndp_transport_error_string(int code) {
  switch (code) {
    case NDP_T_OK: return "success";
    case NDP_T_ACCEPTED: return "accepted/in progress";
    case NDP_T_EINVAL: return "invalid argument";
    case NDP_T_EVERSION: return "ABI/protocol version mismatch";
    case NDP_T_ESTATE: return "invalid state";
    case NDP_T_EFENCE: return "stale fence";
    case NDP_T_ESTALE: return "stale identity or handle";
    case NDP_T_ECONFLICT: return "conflicting duplicate";
    case NDP_T_ECHECKSUM: return "checksum failure";
    case NDP_T_ENONFINITE: return "nonfinite value";
    case NDP_T_EBOUNDS: return "configured bound exceeded";
    case NDP_T_ECREDIT: return "no credit/resource slot";
    case NDP_T_EDEADLINE: return "deadline expired";
    case NDP_T_EROUTE: return "route failure";
    case NDP_T_EPROVIDER: return "provider policy/setup failure";
    case NDP_T_ENOMEM: return "out of bounded memory";
    case NDP_T_EIO: return "I/O failure";
    case NDP_T_ESHUTDOWN: return "service shutting down";
    default: return "unknown native transport error";
  }
}

int ndp_transport_open_v1(const struct ndp_transport_open_v1 *config,
                          ndp_transport_t *out) {
  using namespace emender::ndp;
  if (!valid_prefix(config) || out == nullptr) return NDP_T_EVERSION;
  if (config->reserved0 != 0 || config->reserved1 != 0) return NDP_T_EINVAL;
  if ((config->provider_len != 0 && config->provider_len > NDP_TRANSPORT_PROVIDER_MAX) ||
      config->require_provider_len > NDP_TRANSPORT_PROVIDER_MAX ||
      config->fabric_len > NDP_TRANSPORT_FABRIC_MAX ||
      config->domain_len > NDP_TRANSPORT_DOMAIN_MAX ||
      config->bind_node_len > NDP_TRANSPORT_DOMAIN_MAX) return NDP_T_EINVAL;
  FabricConfig native{};
  native.production = config->mode == NDP_T_MODE_PRODUCTION;
  if (!native.production && config->mode != NDP_T_MODE_TEST_ONLY) return NDP_T_EINVAL;
  native.provider = span(config->provider, config->provider_len, NDP_TRANSPORT_PROVIDER_MAX);
  native.require_provider = span(config->require_provider, config->require_provider_len,
                                 NDP_TRANSPORT_PROVIDER_MAX);
  native.fabric = span(config->fabric, config->fabric_len, NDP_TRANSPORT_FABRIC_MAX);
  native.domain = span(config->domain, config->domain_len, NDP_TRANSPORT_DOMAIN_MAX);
  native.bind_node = span(config->bind_node, config->bind_node_len, NDP_TRANSPORT_DOMAIN_MAX);
  native.tx_slots = config->tx_slots; native.rx_slots = config->rx_slots;
  native.payload_max = config->payload_max;
  native.resident_limit_bytes = config->resident_limit_bytes;
  native.deadline_unix_ns = config->operation_deadline_unix_ns;
  native.telemetry_fd = config->telemetry_fd;
  try {
    auto endpoint = std::make_shared<FabricEndpoint>(std::move(native));
    const int rc = endpoint->start();
    if (rc != NDP_T_OK) return rc;
    auto instance = std::make_shared<TransportInstance>();
    instance->endpoint = std::move(endpoint);
    instance->payload_max = config->payload_max;
    const ndp_transport_t handle = next_handle.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      registry.emplace(handle, std::move(instance));
    }
    *out = handle;
    return NDP_T_OK;
  } catch (const std::bad_alloc &) {
    return NDP_T_ENOMEM;
  } catch (...) {
    return NDP_T_EIO;
  }
}

int ndp_transport_bind_identity_v1(
    ndp_transport_t transport,
    const struct ndp_transport_identity_v1 *identity) {
  using namespace emender::ndp;
  if (!valid_prefix(identity)) return NDP_T_EVERSION;
  if (!nonzero_key(identity->run_key) || !nonzero_key(identity->worker_key) ||
      !nonzero_key(identity->incarnation) || identity->fence_epoch == 0 ||
      identity->endpoint_epoch == 0 ||
      identity->expires_unix_ns <= unix_time_ns()) {
    return NDP_T_EINVAL;
  }
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  const FabricFacts facts = instance->endpoint->facts();
  EndpointRecord record{};
  std::copy_n(identity->run_key, 16, record.run_key.begin());
  record.fence_epoch = identity->fence_epoch;
  std::copy_n(identity->worker_key, 16, record.worker_key.begin());
  std::copy_n(identity->incarnation, 16, record.incarnation.begin());
  record.endpoint_epoch = identity->endpoint_epoch;
  record.expires_unix_ns = identity->expires_unix_ns;
  record.provider_name = facts.provider;
  record.fabric_name = facts.fabric;
  record.domain_name = facts.domain;
  record.addr_format = facts.addr_format;
  record.endpoint_name = facts.endpoint_name;
  std::lock_guard<std::mutex> lock(instance->binding_mutex);
  if (instance->identity.has_value()) {
    const auto &current = *instance->identity;
    if (!same_bytes(current.run_key, identity->run_key) ||
        current.fence_epoch != identity->fence_epoch ||
        !same_bytes(current.worker_key, identity->worker_key) ||
        !same_bytes(current.incarnation, identity->incarnation) ||
        current.endpoint_epoch != identity->endpoint_epoch ||
        current.expires_unix_ns != identity->expires_unix_ns) {
      return NDP_T_ECONFLICT;
    }
    return NDP_T_OK;
  }
  instance->identity = std::move(record);
  return NDP_T_OK;
}

int ndp_transport_endpoint_v1(ndp_transport_t transport,
                              struct ndp_transport_endpoint_v1 *out) {
  using namespace emender::ndp;
  if (!valid_prefix(out)) return NDP_T_EVERSION;
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  std::vector<std::uint8_t> encoded;
  {
    std::lock_guard<std::mutex> lock(instance->binding_mutex);
    if (!instance->identity.has_value()) return NDP_T_ESTATE;
    const int rc = encode_endpoint_record(*instance->identity, &encoded);
    if (rc != NDP_T_OK) return rc;
  }
  if (encoded.size() > sizeof(out->record)) return NDP_T_EBOUNDS;
  std::memset(out->record, 0, sizeof(out->record));
  std::copy(encoded.begin(), encoded.end(), out->record);
  out->record_bytes = static_cast<std::uint32_t>(encoded.size());
  out->reserved0 = 0;
  return NDP_T_OK;
}

int ndp_transport_peer_upsert_v1(ndp_transport_t transport,
                                 const struct ndp_transport_peer_v1 *peer,
                                 uint64_t *peer_id) {
  using namespace emender::ndp;
  if (!valid_prefix(peer) || peer_id == nullptr) return NDP_T_EVERSION;
  if (peer->endpoint_name_bytes == 0 ||
      peer->endpoint_name_bytes > NDP_TRANSPORT_ENDPOINT_MAX || peer->reserved0 != 0 ||
      peer->endpoint_epoch == 0 || !nonzero_key(peer->worker_key) ||
      !nonzero_key(peer->incarnation)) {
    return NDP_T_EINVAL;
  }
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  EndpointRecord record{};
  const int decoded = decode_endpoint_record(
      peer->endpoint_name, peer->endpoint_name_bytes, &record);
  if (decoded != NDP_T_OK) return decoded;
  {
    std::lock_guard<std::mutex> lock(instance->binding_mutex);
    if (!instance->identity.has_value()) return NDP_T_ESTATE;
    const auto &local = *instance->identity;
    if (!same_bytes(record.run_key, local.run_key.data()) ||
        record.fence_epoch != local.fence_epoch ||
        !same_bytes(record.worker_key, peer->worker_key) ||
        !same_bytes(record.incarnation, peer->incarnation) ||
        record.endpoint_epoch != peer->endpoint_epoch ||
        record.expires_unix_ns != peer->expires_unix_ns ||
        record.expires_unix_ns <= unix_time_ns() ||
        record.provider_name != local.provider_name) {
      return NDP_T_ECONFLICT;
    }
  }
  std::vector<std::uint8_t> name = record.endpoint_name;
  const std::string worker = key_string(peer->worker_key);
  const Digest endpoint_digest = sha256(name.data(), name.size());
  std::lock_guard<std::mutex> lock(instance->binding_mutex);
  const auto current = instance->bindings.find(worker);
  if (current != instance->bindings.end()) {
    const bool same_incarnation = std::equal(
        current->second.incarnation.begin(), current->second.incarnation.end(),
        peer->incarnation);
    if (peer->endpoint_epoch < current->second.endpoint_epoch) return NDP_T_ESTALE;
    if (peer->endpoint_epoch == current->second.endpoint_epoch) {
      if (!same_incarnation ||
          !constant_time_equal(current->second.endpoint_digest, endpoint_digest)) {
        return NDP_T_ECONFLICT;
      }
      *peer_id = current->second.peer_id;
      return NDP_T_OK;
    }
    const std::uint64_t old_id = current->second.peer_id;
    const int remove_rc = instance->endpoint->remove_peer(old_id);
    if (remove_rc != NDP_T_OK && remove_rc != NDP_T_EROUTE) return remove_rc;
    instance->ids.erase(old_id);
    instance->bindings.erase(current);
  }
  std::uint64_t installed_id = 0;
  const int rc = instance->endpoint->add_peer(name, peer->expires_unix_ns, &installed_id);
  if (rc != NDP_T_OK) return rc;
  PeerBinding binding{};
  binding.peer_id = installed_id; binding.endpoint_epoch = peer->endpoint_epoch;
  std::copy_n(peer->worker_key, 16, binding.worker_key.begin());
  std::copy_n(peer->incarnation, 16, binding.incarnation.begin());
  binding.endpoint_digest = endpoint_digest;
  instance->bindings.emplace(worker, binding);
  instance->ids.emplace(installed_id, worker);
  *peer_id = installed_id;
  return NDP_T_OK;
}

int ndp_transport_peer_remove_v1(ndp_transport_t transport, uint64_t peer_id) {
  using namespace emender::ndp;
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  {
    std::lock_guard<std::mutex> lock(instance->binding_mutex);
    const auto id = instance->ids.find(peer_id);
    if (id == instance->ids.end()) return NDP_T_ESTALE;
    instance->bindings.erase(id->second);
    instance->ids.erase(id);
  }
  return instance->endpoint->remove_peer(peer_id);
}

int ndp_transport_send_v1(ndp_transport_t transport, uint64_t peer_id,
                          const uint8_t *frame, uint64_t frame_bytes,
                          uint64_t deadline_unix_ns) {
  using namespace emender::ndp;
  if (frame_bytes > std::numeric_limits<std::size_t>::max()) return NDP_T_EBOUNDS;
  const auto instance = lookup(transport);
  return instance == nullptr ? NDP_T_ESTALE : instance->endpoint->send(
      peer_id, frame, static_cast<std::size_t>(frame_bytes), deadline_unix_ns);
}

int ndp_transport_poll_v1(ndp_transport_t transport,
                          struct ndp_transport_event_v1 *events,
                          uint32_t capacity, uint32_t *count, int timeout_ms) {
  using namespace emender::ndp;
  if (events == nullptr || count == nullptr || capacity == 0 || timeout_ms < 0) {
    return NDP_T_EINVAL;
  }
  for (std::uint32_t i = 0; i != capacity; ++i) {
    if (!valid_prefix(&events[i])) return NDP_T_EVERSION;
  }
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  std::vector<FabricEvent> native;
  const int rc = instance->endpoint->poll(&native, capacity, timeout_ms);
  if (rc != NDP_T_OK) return rc;
  for (std::size_t i = 0; i != native.size(); ++i) {
    auto &output = events[i];
    output.event = native[i].kind; output.status = native[i].status;
    output.peer_id = native[i].peer_id; output.message_seq = native[i].message_seq;
    output.useful_bytes = native[i].useful_bytes; output.wire_bytes = native[i].wire_bytes;
    output.provider_errno = native[i].provider_errno; output.reason = native[i].reason;
    std::copy(native[i].detail.begin(), native[i].detail.end(), output.detail);
    output.reserved0 = 0;
  }
  *count = static_cast<std::uint32_t>(native.size());
  return NDP_T_OK;
}

int ndp_transport_receive_v1(ndp_transport_t transport, uint8_t *frame,
                             uint64_t capacity, uint64_t *frame_bytes,
                             uint64_t *peer_id) {
  using namespace emender::ndp;
  if (frame == nullptr || frame_bytes == nullptr || peer_id == nullptr ||
      capacity > std::numeric_limits<std::size_t>::max()) return NDP_T_EBOUNDS;
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  std::size_t native_bytes = 0;
  std::uint64_t source_peer = 0;
  const int rc = instance->endpoint->receive(
      frame, static_cast<std::size_t>(capacity), &native_bytes, &source_peer);
  *frame_bytes = native_bytes;
  if (rc != NDP_T_OK) return rc;
  FrameHeader decoded{};
  const std::uint8_t *payload = nullptr;
  std::size_t payload_bytes = 0;
  const int decode_rc = decode_frame_view(frame, native_bytes, instance->payload_max,
                                          &decoded, &payload, &payload_bytes);
  if (decode_rc != NDP_T_OK) return decode_rc;
  (void)payload;
  (void)payload_bytes;
  {
    std::lock_guard<std::mutex> lock(instance->binding_mutex);
    const std::string worker = key_string(decoded.worker_key.data());
    const auto binding = instance->bindings.find(worker);
    if (binding == instance->bindings.end() ||
        !std::equal(binding->second.incarnation.begin(), binding->second.incarnation.end(),
                    decoded.incarnation.begin()) ||
        (source_peer != 0 && source_peer != binding->second.peer_id)) return NDP_T_EROUTE;
    *peer_id = binding->second.peer_id;
  }
  return NDP_T_OK;
}

int ndp_transport_metrics_v1(ndp_transport_t transport,
                             struct ndp_transport_metrics_v1 *out) {
  using namespace emender::ndp;
  if (!valid_prefix(out)) return NDP_T_EVERSION;
  const auto instance = lookup(transport);
  if (instance == nullptr) return NDP_T_ESTALE;
  const FabricCounters metrics = instance->endpoint->counters();
  const FabricFacts facts = instance->endpoint->facts();
  out->useful_tx_bytes = metrics.useful_tx_bytes;
  out->useful_rx_bytes = metrics.useful_rx_bytes;
  out->wire_tx_bytes = metrics.wire_tx_bytes; out->wire_rx_bytes = metrics.wire_rx_bytes;
  out->retries = metrics.retries; out->replay_bytes = metrics.replay_bytes;
  out->duplicate_frames = 0; out->checksum_rejects = 0; out->stale_rejects = 0;
  out->cq_errors = metrics.cq_errors; out->route_errors = metrics.route_errors;
  out->in_flight_bytes = metrics.in_flight_bytes;
  out->in_flight_high_water = metrics.in_flight_high_water;
  out->retained_bytes = metrics.retained_bytes;
  out->retained_high_water = metrics.retained_high_water;
  out->released_bytes = metrics.released_bytes;
  out->tx_slot_high_water = metrics.tx_slot_high_water;
  out->rx_slot_high_water = metrics.rx_slot_high_water;
  out->latency_count = metrics.latency_count; out->latency_total_ns = metrics.latency_total_ns;
  out->service_started_unix_ns = metrics.service_started_unix_ns;
  out->last_progress_unix_ns = metrics.last_progress_unix_ns;
  out->live_peers = static_cast<std::uint32_t>(instance->endpoint->peer_count());
  out->owner_state = out->live_peers == 0 ? NDP_T_OWNER_ROUTE_IDLE
                                          : NDP_T_OWNER_ROUTE_READY;
  out->provider_name_len = static_cast<std::uint32_t>(
      std::min<std::size_t>(facts.provider.size(), NDP_TRANSPORT_PROVIDER_MAX));
  std::memset(out->provider_name, 0, sizeof(out->provider_name));
  std::copy_n(facts.provider.begin(), out->provider_name_len, out->provider_name);
  return NDP_T_OK;
}

int ndp_transport_cancel_v1(ndp_transport_t transport, uint64_t peer_id) {
  using namespace emender::ndp;
  const auto instance = lookup(transport);
  return instance == nullptr ? NDP_T_ESTALE : instance->endpoint->cancel(peer_id);
}

int ndp_transport_close_v1(ndp_transport_t transport) {
  using namespace emender::ndp;
  std::shared_ptr<TransportInstance> instance;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = registry.find(transport);
    if (found == registry.end()) return NDP_T_ESTALE;
    instance = std::move(found->second); registry.erase(found);
  }
  instance->endpoint->shutdown();
  return NDP_T_OK;
}

}  // extern "C"
