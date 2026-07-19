#include "fabric.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string_view>

#include <unistd.h>

namespace emender::ndp {
namespace {

constexpr std::size_t kSlotAlignment = 2 * 1024 * 1024;

std::string safe(const char *value) { return value == nullptr ? std::string{} : value; }

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  for (const char raw : value) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

bool is_test_provider(const std::string &provider) {
  return provider == "tcp;ofi_rxm" || provider == "shm";
}

bool equivalent(const fi_info *a, const fi_info *b) {
  return safe(a->fabric_attr->prov_name) == safe(b->fabric_attr->prov_name) &&
         safe(a->fabric_attr->name) == safe(b->fabric_attr->name) &&
         safe(a->domain_attr->name) == safe(b->domain_attr->name) &&
         a->addr_format == b->addr_format && a->ep_attr->type == b->ep_attr->type &&
         a->ep_attr->max_msg_size == b->ep_attr->max_msg_size;
}

void set_name(char **destination, const std::string &name) {
  if (name.empty()) return;
  *destination = ::strdup(name.c_str());
  if (*destination == nullptr) throw std::bad_alloc();
}

std::uint64_t body_bytes(const std::uint8_t *frame, std::size_t frame_bytes) {
  if (frame_bytes < kHeaderBytes) return 0;
  // The decoder owns validation. For telemetry only, contribution/result are
  // the two message types that carry their bytes after the fixed header.
  const std::uint16_t type = static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(frame[12]) |
      static_cast<std::uint16_t>(static_cast<std::uint16_t>(frame[13]) << 8U));
  return (type == static_cast<std::uint16_t>(MessageType::contribution_data) ||
          type == static_cast<std::uint16_t>(MessageType::result_data))
             ? frame_bytes - kHeaderBytes : 0;
}

}  // namespace

FabricEndpoint::FabricEndpoint(FabricConfig config) : config_(std::move(config)) {
  counters_.service_started_unix_ns = unix_time_ns();
  counters_.last_progress_unix_ns = counters_.service_started_unix_ns;
}

FabricEndpoint::~FabricEndpoint() { shutdown(); }

int FabricEndpoint::validate_config(const FabricConfig &config, std::string *why) {
  const auto fail = [&](std::string message) {
    if (why != nullptr) *why = std::move(message);
    return NDP_T_EPROVIDER;
  };
  if (config.provider.empty()) return fail("provider must be explicit");
  if (config.production) {
    if (config.provider != "cxi" || config.require_provider != "cxi") {
      return fail("production requires --provider=cxi and --require-provider=cxi");
    }
    const char *effective = std::getenv("FI_PROVIDER");
    if (effective != nullptr && *effective != '\0' && std::string_view(effective) != "cxi") {
      return fail("FI_PROVIDER cannot weaken the required production provider");
    }
  } else {
    if (!is_test_provider(config.provider)) {
      return fail("test-only mode permits only tcp;ofi_rxm or shm");
    }
    if (!config.require_provider.empty() && config.require_provider != config.provider) {
      return fail("test provider requirement must match the explicit provider");
    }
  }
  if (config.tx_slots == 0 || config.tx_slots > 16 || config.rx_slots == 0 ||
      config.rx_slots > 16 || config.payload_max == 0 ||
      config.payload_max > NDP_TRANSPORT_MAX_PAYLOAD || config.payload_max % 8 != 0 ||
      config.deadline_unix_ns == 0) return fail("slot, payload, or deadline bound invalid");
  std::uint64_t pool_bytes = 0;
  const std::uint64_t slot_bytes = config.payload_max + kHeaderBytes;
  if (slot_bytes < config.payload_max ||
      static_cast<std::uint64_t>(config.tx_slots + config.rx_slots) >
          std::numeric_limits<std::uint64_t>::max() / slot_bytes) {
    return fail("registered pool size overflow");
  }
  pool_bytes = static_cast<std::uint64_t>(config.tx_slots + config.rx_slots) * slot_bytes;
  if (config.resident_limit_bytes == 0 || pool_bytes > config.resident_limit_bytes) {
    return fail("registered pools exceed resident limit");
  }
  return NDP_T_OK;
}

int FabricEndpoint::resolve_provider() {
  fi_info *hints = fi_allocinfo();
  if (hints == nullptr) return NDP_T_ENOMEM;
  // Every accepted RDM frame must be attributable to its installed AV route.
  // FI_SOURCE makes fi_cq_readfrom return that address on both the layered
  // local provider and native CXI.  Production separately pins cxi0, so this
  // capability no longer leaves provider/domain resolution ambiguous.
  hints->caps = FI_MSG | FI_SOURCE;
  hints->mode = FI_CONTEXT;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->threading = FI_THREAD_SAFE;
  try {
    set_name(&hints->fabric_attr->prov_name, config_.provider);
    set_name(&hints->fabric_attr->name, config_.fabric);
    set_name(&hints->domain_attr->name, config_.domain);
  } catch (...) {
    fi_freeinfo(hints);
    return NDP_T_ENOMEM;
  }
  const char *node = config_.bind_node.empty() ? nullptr : config_.bind_node.c_str();
  const std::uint64_t flags = node == nullptr ? 0 : FI_SOURCE;
  fi_info *matches = nullptr;
  const int rc = fi_getinfo(FI_VERSION(1, 18), node, nullptr, flags, hints, &matches);
  fi_freeinfo(hints);
  if (rc != 0 || matches == nullptr) return NDP_T_EPROVIDER;

  fi_info *selected = nullptr;
  for (fi_info *candidate = matches; candidate != nullptr; candidate = candidate->next) {
    if (safe(candidate->fabric_attr->prov_name) != config_.provider ||
        candidate->ep_attr->type != FI_EP_RDM || (candidate->caps & FI_MSG) == 0) continue;
    if (selected == nullptr) selected = candidate;
    else if (!equivalent(selected, candidate)) {
      fi_freeinfo(matches);
      return NDP_T_EPROVIDER;
    }
  }
  if (selected == nullptr) {
    fi_freeinfo(matches);
    return NDP_T_EPROVIDER;
  }
  info_ = fi_dupinfo(selected);
  fi_freeinfo(matches);
  if (info_ == nullptr) return NDP_T_ENOMEM;
  if (info_->ep_attr->max_msg_size <= kHeaderBytes ||
      config_.payload_max > info_->ep_attr->max_msg_size - kHeaderBytes) {
    return NDP_T_EPROVIDER;
  }
  facts_.provider = safe(info_->fabric_attr->prov_name);
  facts_.fabric = safe(info_->fabric_attr->name);
  facts_.domain = safe(info_->domain_attr->name);
  facts_.addr_format = info_->addr_format;
  facts_.api_version = FI_VERSION(1, 18);
  facts_.provider_version = info_->fabric_attr->prov_version;
  facts_.max_msg_size = info_->ep_attr->max_msg_size;
  facts_.mr_mode = static_cast<std::uint64_t>(info_->domain_attr->mr_mode);
  facts_.production_provider = config_.production && facts_.provider == "cxi";
  if (config_.production && !facts_.production_provider) return NDP_T_EPROVIDER;
  return NDP_T_OK;
}

int FabricEndpoint::open_fabric() {
  int rc = fi_fabric(info_->fabric_attr, &fabric_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;
  rc = fi_domain(fabric_, info_, &domain_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;

  fi_cq_attr cq_attr{};
  cq_attr.format = FI_CQ_FORMAT_MSG;
  cq_attr.wait_obj = FI_WAIT_NONE;
  cq_attr.size = std::max<std::uint32_t>(config_.tx_slots * 4, 64);
  rc = fi_cq_open(domain_, &cq_attr, &tx_cq_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;
  cq_attr.size = std::max<std::uint32_t>(config_.rx_slots * 4, 64);
  rc = fi_cq_open(domain_, &cq_attr, &rx_cq_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;

  fi_av_attr av_attr{};
  // Table addresses make CQ source identity stable and comparable across
  // providers; the opaque endpoint name itself remains control-plane data.
  av_attr.type = FI_AV_TABLE;
  av_attr.count = NDP_TRANSPORT_MAX_CONTRIBUTIONS;
  rc = fi_av_open(domain_, &av_attr, &av_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;
  rc = fi_endpoint(domain_, info_, &endpoint_, nullptr);
  if (rc != 0) return NDP_T_EPROVIDER;
  if (fi_ep_bind(endpoint_, &tx_cq_->fid, FI_SEND) != 0 ||
      fi_ep_bind(endpoint_, &rx_cq_->fid, FI_RECV) != 0 ||
      fi_ep_bind(endpoint_, &av_->fid, 0) != 0 || fi_enable(endpoint_) != 0) {
    return NDP_T_EPROVIDER;
  }

  std::size_t name_bytes = 0;
  rc = fi_getname(&endpoint_->fid, nullptr, &name_bytes);
  if (rc != -FI_ETOOSMALL || name_bytes == 0 || name_bytes > NDP_TRANSPORT_ENDPOINT_MAX) {
    return NDP_T_EPROVIDER;
  }
  facts_.endpoint_name.resize(name_bytes);
  rc = fi_getname(&endpoint_->fid, facts_.endpoint_name.data(), &name_bytes);
  if (rc != 0) return NDP_T_EPROVIDER;
  facts_.endpoint_name.resize(name_bytes);
  return NDP_T_OK;
}

int FabricEndpoint::allocate_slots(std::vector<std::unique_ptr<Slot>> *slots,
                                   std::uint32_t count, bool receive) {
  const std::uint64_t capacity64 = config_.payload_max + kHeaderBytes;
  if (capacity64 > std::numeric_limits<std::size_t>::max()) return NDP_T_EBOUNDS;
  const std::size_t capacity = static_cast<std::size_t>(capacity64);
  slots->reserve(count);
  for (std::uint32_t i = 0; i != count; ++i) {
    auto slot = std::make_unique<Slot>();
    if (posix_memalign(&slot->buffer, kSlotAlignment, capacity) != 0) return NDP_T_ENOMEM;
    std::memset(slot->buffer, 0, capacity);
    slot->capacity = capacity; slot->index = i; slot->receive = receive;
    // CXI production must prove explicit registration. Some bounded local
    // RDM providers advertise mr_mode=0 and reject fi_mr_reg because their
    // host buffers require no registration; those test slots remain fixed and
    // reusable, but are never reported as production evidence.
    if (config_.production || info_->domain_attr->mr_mode != 0) {
      const std::uint64_t access = FI_SEND | FI_RECV;
      const int rc = fi_mr_reg(domain_, slot->buffer, capacity, access,
                               0, 0, 0, &slot->mr, nullptr);
      if (rc != 0) {
        if (config_.production) return NDP_T_EPROVIDER;
        slot->mr = nullptr;
      }
      if (slot->mr != nullptr &&
          (info_->domain_attr->mr_mode & FI_MR_ENDPOINT) != 0) {
        // CXI advertises FI_MR_ENDPOINT: the region is disabled after
        // fi_mr_reg and is unusable until it is bound to this endpoint and
        // explicitly enabled.
        if (fi_mr_bind(slot->mr, &endpoint_->fid, 0) != 0 ||
            fi_mr_enable(slot->mr) != 0) {
          (void)fi_close(&slot->mr->fid);
          slot->mr = nullptr;
          std::free(slot->buffer);
          slot->buffer = nullptr;
          return NDP_T_EPROVIDER;
        }
      }
    }
    slots->push_back(std::move(slot));
  }
  return NDP_T_OK;
}

int FabricEndpoint::post_receive(Slot *slot) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot->active) return NDP_T_OK;
    slot->used = 0; slot->peer_id = 0; slot->message_seq = 0;
    slot->active = true;
  }
  void *descriptor = slot->mr == nullptr ? nullptr : fi_mr_desc(slot->mr);
  const ssize_t rc = fi_recv(endpoint_, slot->buffer, slot->capacity,
                             descriptor, FI_ADDR_UNSPEC, &slot->context);
  if (rc == 0) return NDP_T_OK;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    slot->active = false;
  }
  return rc == -FI_EAGAIN ? NDP_T_ECREDIT : NDP_T_EROUTE;
}

int FabricEndpoint::start() {
  std::string why;
  int rc = validate_config(config_, &why);
  if (rc != NDP_T_OK) return rc;
  rc = resolve_provider();
  if (rc != NDP_T_OK) return rc;
  rc = open_fabric();
  if (rc != NDP_T_OK) return rc;
  rc = allocate_slots(&tx_slots_, config_.tx_slots, false);
  if (rc != NDP_T_OK) return rc;
  rc = allocate_slots(&rx_slots_, config_.rx_slots, true);
  if (rc != NDP_T_OK) return rc;
  for (auto &slot : rx_slots_) {
    rc = post_receive(slot.get());
    if (rc != NDP_T_OK) return rc;
  }
  counters_.rx_slot_high_water = rx_slots_.size();
  event_capacity_ = static_cast<std::size_t>(config_.tx_slots + config_.rx_slots) * 4 + 16;
  running_.store(true);
  progress_thread_ = std::thread(&FabricEndpoint::progress_loop, this);
  telemetry("fabric_ready");
  return NDP_T_OK;
}

int FabricEndpoint::add_peer(const std::vector<std::uint8_t> &endpoint_name,
                             std::uint64_t expires_unix_ns, std::uint64_t *peer_id) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  if (peer_id == nullptr || endpoint_name.empty() ||
      endpoint_name.size() > NDP_TRANSPORT_ENDPOINT_MAX ||
      expires_unix_ns <= unix_time_ns() || shutting_down_.load()) return NDP_T_EINVAL;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (peers_.size() >= NDP_TRANSPORT_MAX_CONTRIBUTIONS) return NDP_T_EBOUNDS;
  }
  fi_addr_t address = FI_ADDR_UNSPEC;
  const ssize_t inserted = fi_av_insert(av_, endpoint_name.data(), 1, &address, 0, nullptr);
  if (inserted != 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.route_errors;
    return NDP_T_EROUTE;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const std::uint64_t id = next_peer_id_++;
  peers_.emplace(id, Peer{address, expires_unix_ns, false});
  *peer_id = id;
  return NDP_T_OK;
}

int FabricEndpoint::remove_peer(std::uint64_t peer_id) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  fi_addr_t address = FI_ADDR_UNSPEC;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = peers_.find(peer_id);
    if (found == peers_.end()) return NDP_T_ESTALE;
    address = found->second.address;
    peers_.erase(found);
  }
  return fi_av_remove(av_, &address, 1, 0) == 0 ? NDP_T_OK : NDP_T_EROUTE;
}

int FabricEndpoint::send(std::uint64_t peer_id, const std::uint8_t *frame,
                         std::size_t frame_bytes, std::uint64_t deadline_unix_ns,
                         bool replay) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  if (frame == nullptr || frame_bytes < kHeaderBytes ||
      frame_bytes > config_.payload_max + kHeaderBytes || deadline_unix_ns == 0) {
    return NDP_T_EBOUNDS;
  }
  FrameHeader decoded{};
  const std::uint8_t *payload = nullptr;
  std::size_t payload_bytes = 0;
  const int decoded_rc = decode_frame_view(frame, frame_bytes, config_.payload_max,
                                           &decoded, &payload, &payload_bytes);
  if (decoded_rc != NDP_T_OK) return decoded_rc;
  (void)payload;
  (void)payload_bytes;
  while (!shutting_down_.load()) {
    if (unix_time_ns() >= deadline_unix_ns || unix_time_ns() >= config_.deadline_unix_ns) {
      return NDP_T_EDEADLINE;
    }
    Slot *slot = nullptr;
    fi_addr_t address = FI_ADDR_UNSPEC;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto peer = peers_.find(peer_id);
      if (peer == peers_.end() || peer->second.cancelled ||
          unix_time_ns() >= peer->second.expires_unix_ns) return NDP_T_EROUTE;
      address = peer->second.address;
      const auto available = std::find_if(tx_slots_.begin(), tx_slots_.end(),
          [](const std::unique_ptr<Slot> &candidate) { return !candidate->active; });
      if (available != tx_slots_.end()) {
        slot = available->get();
        std::memcpy(slot->buffer, frame, frame_bytes);
        slot->used = frame_bytes; slot->peer_id = peer_id;
        slot->message_seq = decoded.message_seq;
        slot->started_ns = monotonic_time_ns(); slot->active = true;
      }
    }
    if (slot == nullptr) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.retries;
      }
      std::this_thread::yield();
      continue;
    }
    ssize_t rc = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      void *descriptor = slot->mr == nullptr ? nullptr : fi_mr_desc(slot->mr);
      rc = fi_send(endpoint_, slot->buffer, slot->used, descriptor,
                   address, &slot->context);
      if (rc == 0) {
      counters_.in_flight_bytes += slot->used;
      counters_.in_flight_high_water = std::max(counters_.in_flight_high_water,
                                                 counters_.in_flight_bytes);
      const auto active = static_cast<std::uint64_t>(std::count_if(
          tx_slots_.begin(), tx_slots_.end(),
          [](const std::unique_ptr<Slot> &candidate) { return candidate->active; }));
      counters_.tx_slot_high_water = std::max(counters_.tx_slot_high_water, active);
      if (replay) counters_.replay_bytes += body_bytes(frame, frame_bytes);
      }
      if (rc != 0) {
        slot->active = false;
        if (rc == -FI_EAGAIN) ++counters_.retries;
        else ++counters_.route_errors;
      }
    }
    if (rc == 0) return NDP_T_ACCEPTED;
    if (rc != -FI_EAGAIN) return NDP_T_EROUTE;
    std::this_thread::yield();
  }
  return NDP_T_ESHUTDOWN;
}

int FabricEndpoint::drain_cq_error(fid_cq *cq, bool receive) {
  fi_cq_err_entry error{};
  const ssize_t rc = fi_cq_readerr(cq, &error, 0);
  if (rc <= 0) return rc == -FI_EAGAIN ? NDP_T_OK : NDP_T_EROUTE;
  FabricEvent event{};
  event.kind = NDP_T_EVENT_CQ_ERROR; event.status = NDP_T_EROUTE;
  event.provider_errno = error.prov_errno == 0 ? error.err : error.prov_errno;
  event.reason = static_cast<std::uint32_t>(WireReason::provider);
  Slot *repost = nullptr;
  if (error.op_context != nullptr) {
    auto *slot = reinterpret_cast<Slot *>(error.op_context);
    event.peer_id = slot->peer_id; event.message_seq = slot->message_seq;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!receive && slot->active) {
      counters_.in_flight_bytes -= std::min(counters_.in_flight_bytes,
          static_cast<std::uint64_t>(slot->used));
      counters_.released_bytes += slot->used;
    }
    slot->active = false;
    if (receive) repost = slot;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.cq_errors;
  }
  emit_event(std::move(event));
  if (repost != nullptr && !shutting_down_.load()) {
    const int posted = post_receive(repost);
    if (posted != NDP_T_OK && posted != NDP_T_ECREDIT) return posted;
  }
  return NDP_T_OK;
}

int FabricEndpoint::drain_cq(fid_cq *cq, bool receive) {
  std::array<fi_cq_msg_entry, 16> entries{};
  std::array<fi_addr_t, 16> sources{};
  sources.fill(FI_ADDR_NOTAVAIL);
  const ssize_t count = receive
      ? fi_cq_readfrom(cq, entries.data(), entries.size(), sources.data())
      : fi_cq_read(cq, entries.data(), entries.size());
  if (count == -FI_EAGAIN) return NDP_T_OK;
  if (count == -FI_EAVAIL) return drain_cq_error(cq, receive);
  if (count < 0) return NDP_T_EROUTE;
  for (ssize_t i = 0; i != count; ++i) {
    if (entries[static_cast<std::size_t>(i)].op_context == nullptr) continue;
    auto *slot = reinterpret_cast<Slot *>(entries[static_cast<std::size_t>(i)].op_context);
    FabricEvent event{};
    event.kind = receive ? NDP_T_EVENT_RECEIVED : NDP_T_EVENT_SENT;
    event.peer_id = slot->peer_id; event.message_seq = slot->message_seq;
    bool accept_receive = !receive;
    if (receive) {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto peer = std::find_if(peers_.begin(), peers_.end(), [&](const auto &candidate) {
        return candidate.second.address == sources[static_cast<std::size_t>(i)];
      });
      if (peer == peers_.end() || peer->second.cancelled ||
          unix_time_ns() >= peer->second.expires_unix_ns) {
        event.status = NDP_T_EROUTE;
        event.reason = static_cast<std::uint32_t>(WireReason::route);
        ++counters_.route_errors;
      } else {
        event.peer_id = peer->first;
        accept_receive = true;
      }
    }
    const std::size_t completed_bytes = receive
        ? entries[static_cast<std::size_t>(i)].len : slot->used;
    if (completed_bytes > slot->capacity) {
      accept_receive = false;
      event.status = NDP_T_EBOUNDS;
      event.reason = static_cast<std::uint32_t>(WireReason::byte_bounds);
      std::lock_guard<std::mutex> lock(mutex_);
      ++counters_.route_errors;
    }
    event.wire_bytes = completed_bytes;
    event.useful_bytes = accept_receive && completed_bytes >= kHeaderBytes
        ? body_bytes(static_cast<const std::uint8_t *>(slot->buffer),
                     completed_bytes) : 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (receive) {
        counters_.wire_rx_bytes += event.wire_bytes;
        counters_.useful_rx_bytes += event.useful_bytes;
      } else {
        counters_.wire_tx_bytes += event.wire_bytes;
        counters_.useful_tx_bytes += event.useful_bytes;
        counters_.in_flight_bytes -= std::min(counters_.in_flight_bytes,
            static_cast<std::uint64_t>(slot->used));
        counters_.released_bytes += slot->used;
        const std::uint64_t latency = monotonic_time_ns() - slot->started_ns;
        ++counters_.latency_count; counters_.latency_total_ns += latency;
      }
      counters_.last_progress_unix_ns = unix_time_ns();
      slot->active = false;
    }
    if (receive && accept_receive) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        counters_.retained_bytes += completed_bytes;
        counters_.retained_high_water = std::max(counters_.retained_high_water,
                                                  counters_.retained_bytes);
        received_.push_back(ReceivedFrame{slot, completed_bytes, event.peer_id,
                                          event.detail});
      }
    } else if (receive && !shutting_down_.load()) {
      // Unknown, expired, truncated, or otherwise unauthenticated sources are
      // observable metadata events only. They never enter the receive queue.
      const int posted = post_receive(slot);
      if (posted != NDP_T_OK && posted != NDP_T_ECREDIT) return posted;
    }
    emit_event(std::move(event));
  }
  return NDP_T_OK;
}

int FabricEndpoint::progress_once() {
  int rc = drain_cq(tx_cq_, false);
  if (rc != NDP_T_OK) return rc;
  return drain_cq(rx_cq_, true);
}

void FabricEndpoint::progress_loop() {
  while (running_.load()) {
    int progress = NDP_T_EIO;
    try {
      progress = progress_once();
    } catch (...) {
      progress = NDP_T_EIO;
    }
    if (progress != NDP_T_OK) {
      FabricEvent event{}; event.kind = NDP_T_EVENT_CQ_ERROR;
      event.status = NDP_T_EROUTE; event.reason = static_cast<std::uint32_t>(WireReason::provider);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.cq_errors;
      }
      emit_event(std::move(event));
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  for (unsigned i = 0; i != 100; ++i) {
    try {
      if (progress_once() != NDP_T_OK) break;
    } catch (...) {
      break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (counters_.in_flight_bytes == 0) break;
  }
}

void FabricEndpoint::emit_event(FabricEvent event) {
  telemetry("completion", &event);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (events_.size() >= event_capacity_) events_.pop_front();
    events_.push_back(std::move(event));
  }
  event_cv_.notify_all();
}

int FabricEndpoint::poll(std::vector<FabricEvent> *events, std::size_t capacity,
                         int timeout_ms) {
  if (events == nullptr || capacity == 0 || timeout_ms < 0) return NDP_T_EINVAL;
  std::unique_lock<std::mutex> lock(mutex_);
  if (events_.empty() && !shutting_down_.load()) {
    event_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                       [&] { return !events_.empty() || shutting_down_.load(); });
  }
  events->clear();
  while (!events_.empty() && events->size() < capacity) {
    events->push_back(std::move(events_.front())); events_.pop_front();
  }
  return NDP_T_OK;
}

int FabricEndpoint::receive(std::vector<std::uint8_t> *frame, std::uint64_t *peer_id) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  if (frame == nullptr || peer_id == nullptr) return NDP_T_EINVAL;
  Slot *repost = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (received_.empty()) return NDP_T_ECREDIT;
    const ReceivedFrame &received = received_.front();
    try {
      const auto *begin = static_cast<const std::uint8_t *>(received.slot->buffer);
      frame->assign(begin, begin + received.bytes);
    } catch (const std::bad_alloc &) {
      return NDP_T_ENOMEM;
    }
    *peer_id = received_.front().peer_id;
    repost = received_.front().slot;
    received_.pop_front();
    counters_.retained_bytes -= std::min(counters_.retained_bytes,
                                          static_cast<std::uint64_t>(frame->size()));
    counters_.released_bytes += frame->size();
  }
  if (repost != nullptr && !shutting_down_.load()) {
    const int rc = post_receive(repost);
    if (rc != NDP_T_OK) return rc;
  }
  return NDP_T_OK;
}

int FabricEndpoint::receive(std::uint8_t *frame, std::size_t capacity,
                            std::size_t *frame_bytes, std::uint64_t *peer_id) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  if (frame == nullptr || frame_bytes == nullptr || peer_id == nullptr) {
    return NDP_T_EINVAL;
  }
  Slot *repost = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (received_.empty()) return NDP_T_ECREDIT;
    const ReceivedFrame &received = received_.front();
    *frame_bytes = received.bytes;
    if (received.bytes > capacity) return NDP_T_EBOUNDS;
    std::memcpy(frame, received.slot->buffer, received.bytes);
    *peer_id = received.peer_id;
    repost = received.slot;
    received_.pop_front();
    counters_.retained_bytes -= std::min(counters_.retained_bytes,
                                          static_cast<std::uint64_t>(*frame_bytes));
    counters_.released_bytes += *frame_bytes;
  }
  if (repost != nullptr && !shutting_down_.load()) {
    const int rc = post_receive(repost);
    if (rc != NDP_T_OK) return rc;
  }
  return NDP_T_OK;
}

int FabricEndpoint::cancel(std::uint64_t peer_id) {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto peer = peers_.find(peer_id);
    if (peer == peers_.end()) return NDP_T_ESTALE;
    peer->second.cancelled = true;
    for (auto &slot : tx_slots_) {
      if (slot->active && slot->peer_id == peer_id) {
        (void)fi_cancel(&endpoint_->fid, &slot->context);
      }
    }
  }
  FabricEvent event{}; event.kind = NDP_T_EVENT_CANCELLED;
  event.status = NDP_T_ESHUTDOWN; event.peer_id = peer_id;
  emit_event(std::move(event));
  return NDP_T_OK;
}

FabricFacts FabricEndpoint::facts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return facts_;
}

FabricCounters FabricEndpoint::counters() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return counters_;
}

std::size_t FabricEndpoint::peer_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return peers_.size();
}

void FabricEndpoint::telemetry(std::string_view event, const FabricEvent *detail) {
  if (config_.telemetry_fd < 0) return;
  std::ostringstream out;
  const FabricCounters snapshot = counters();
  const std::uint64_t elapsed_ns = unix_time_ns() - snapshot.service_started_unix_ns;
  const double throughput = elapsed_ns == 0 ? 0.0 :
      static_cast<double>(snapshot.useful_tx_bytes + snapshot.useful_rx_bytes) * 1.0e9 /
      static_cast<double>(elapsed_ns);
  const double mean_latency = snapshot.latency_count == 0 ? 0.0 :
      static_cast<double>(snapshot.latency_total_ns) /
      static_cast<double>(snapshot.latency_count);
  const char *owner_state = shutting_down_.load() ? "DRAINING" :
      (peer_count() == 0 ? "ROUTE_IDLE" : "ROUTE_READY");
  const char *fi_provider = std::getenv("FI_PROVIDER");
  const char *mr_monitor = std::getenv("FI_MR_CACHE_MONITOR");
  const char *cxi_ats = std::getenv("FI_CXI_ATS");
  out << "{\"schema\":\"emender-native-dataplane-telemetry-v1\","
      << "\"event\":\"" << event << "\",\"unix_ns\":" << unix_time_ns()
      << ",\"provider\":\"" << json_escape(facts_.provider)
      << "\",\"fabric\":\"" << json_escape(facts_.fabric)
      << "\",\"domain\":\"" << json_escape(facts_.domain)
      << "\",\"endpoint_type\":\"FI_EP_RDM\""
      << ",\"libfabric_api_version\":" << facts_.api_version
      << ",\"provider_version\":" << facts_.provider_version
      << ",\"max_msg_size\":" << facts_.max_msg_size
      << ",\"mr_mode\":" << facts_.mr_mode
      << ",\"effective_fi_provider\":\""
      << json_escape(fi_provider == nullptr ? "" : fi_provider)
      << "\",\"fi_mr_cache_monitor\":\""
      << json_escape(mr_monitor == nullptr ? "" : mr_monitor)
      << "\",\"fi_cxi_ats\":\""
      << json_escape(cxi_ats == nullptr ? "" : cxi_ats)
      << "\",\"production_provider\":"
      << (facts_.production_provider ? "true" : "false")
      << ",\"useful_tx_bytes\":" << snapshot.useful_tx_bytes
      << ",\"useful_rx_bytes\":" << snapshot.useful_rx_bytes
      << ",\"wire_tx_bytes\":" << snapshot.wire_tx_bytes
      << ",\"wire_rx_bytes\":" << snapshot.wire_rx_bytes
      << ",\"retries\":" << snapshot.retries
      << ",\"replay_bytes\":" << snapshot.replay_bytes
      << ",\"cq_errors\":" << snapshot.cq_errors
      << ",\"route_errors\":" << snapshot.route_errors
      << ",\"throughput_bytes_per_second\":" << throughput
      << ",\"mean_send_latency_ns\":" << mean_latency
      << ",\"in_flight_bytes\":" << snapshot.in_flight_bytes
      << ",\"in_flight_high_water\":" << snapshot.in_flight_high_water
      << ",\"retained_bytes\":" << snapshot.retained_bytes
      << ",\"retained_high_water\":" << snapshot.retained_high_water
      << ",\"released_bytes\":" << snapshot.released_bytes
      << ",\"owner_state\":\"" << owner_state << "\"";
  if (detail != nullptr) {
    out << ",\"status\":" << detail->status << ",\"peer_id\":" << detail->peer_id
        << ",\"useful_bytes\":" << detail->useful_bytes
        << ",\"wire_bytes\":" << detail->wire_bytes
        << ",\"provider_errno\":" << detail->provider_errno;
  }
  out << "}\n";
  const std::string line = out.str();
  std::lock_guard<std::mutex> lock(telemetry_mutex_);
  std::size_t written = 0;
  while (written != line.size()) {
    const ssize_t result = ::write(config_.telemetry_fd, line.data() + written,
                                   line.size() - written);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) break;
    written += static_cast<std::size_t>(result);
  }
}

void FabricEndpoint::close_fid(fid *object) {
  if (object != nullptr) (void)fi_close(object);
}

void FabricEndpoint::shutdown() {
  std::lock_guard<std::mutex> api_lock(api_mutex_);
  if (shutting_down_.exchange(true)) return;
  running_.store(false);
  event_cv_.notify_all();
  // Cancel while the endpoint and registered storage are still live. Closing
  // or freeing an MR after an ignored FI_EBUSY would permit provider access to
  // released memory, so endpoint close is the hard ownership handoff below.
  if (endpoint_ != nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &slot : tx_slots_) if (slot->active) (void)fi_cancel(&endpoint_->fid, &slot->context);
    for (auto &slot : rx_slots_) if (slot->active) (void)fi_cancel(&endpoint_->fid, &slot->context);
  }
  if (progress_thread_.joinable()) progress_thread_.join();
  for (unsigned i = 0; i != 100; ++i) {
    try {
      if (tx_cq_ != nullptr) (void)drain_cq(tx_cq_, false);
      if (rx_cq_ != nullptr) (void)drain_cq(rx_cq_, true);
    } catch (...) {
      break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (counters_.in_flight_bytes == 0) break;
  }
  if (endpoint_ != nullptr && fi_close(&endpoint_->fid) != 0) {
    // Leak provider-owned objects and their backing buffers rather than
    // freeing memory that may still be DMA-visible. Process supervision owns
    // the final bounded kill if a provider cannot close locally.
    telemetry("shutdown_endpoint_busy");
    return;
  }
  endpoint_ = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &event : received_) {
      counters_.released_bytes += event.bytes;
    }
    counters_.retained_bytes = 0;
    received_.clear();
  }
  for (auto &slot : tx_slots_) {
    close_fid(slot->mr == nullptr ? nullptr : &slot->mr->fid); slot->mr = nullptr;
    std::free(slot->buffer); slot->buffer = nullptr;
  }
  for (auto &slot : rx_slots_) {
    close_fid(slot->mr == nullptr ? nullptr : &slot->mr->fid); slot->mr = nullptr;
    std::free(slot->buffer); slot->buffer = nullptr;
  }
  tx_slots_.clear(); rx_slots_.clear();
  close_fid(tx_cq_ == nullptr ? nullptr : &tx_cq_->fid); tx_cq_ = nullptr;
  close_fid(rx_cq_ == nullptr ? nullptr : &rx_cq_->fid); rx_cq_ = nullptr;
  close_fid(av_ == nullptr ? nullptr : &av_->fid); av_ = nullptr;
  close_fid(domain_ == nullptr ? nullptr : &domain_->fid); domain_ = nullptr;
  close_fid(fabric_ == nullptr ? nullptr : &fabric_->fid); fabric_ = nullptr;
  if (info_ != nullptr) { fi_freeinfo(info_); info_ = nullptr; }
  telemetry("shutdown");
}

}  // namespace emender::ndp
