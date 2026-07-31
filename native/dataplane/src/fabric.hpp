#ifndef EMENDER_NDP_FABRIC_HPP
#define EMENDER_NDP_FABRIC_HPP

#include "protocol.hpp"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace emender::ndp {

struct FabricConfig {
  bool production{false};
  std::string provider;
  std::string require_provider;
  std::string fabric;
  std::string domain;
  std::string bind_node;
  std::uint32_t tx_slots{4};
  std::uint32_t rx_slots{4};
  std::uint64_t payload_max{NDP_TRANSPORT_MAX_PAYLOAD};
  std::uint64_t resident_limit_bytes{0};
  std::uint64_t deadline_unix_ns{0};
  int telemetry_fd{-1};
};

struct FabricFacts {
  std::string provider;
  std::string fabric;
  std::string domain;
  std::uint32_t addr_format{0};
  std::uint32_t api_version{0};
  std::uint32_t provider_version{0};
  std::uint64_t max_msg_size{0};
  std::uint64_t mr_mode{0};
  bool production_provider{false};
  std::vector<std::uint8_t> endpoint_name;
};

struct FabricCounters {
  std::uint64_t useful_tx_bytes{0};
  std::uint64_t useful_rx_bytes{0};
  std::uint64_t wire_tx_bytes{0};
  std::uint64_t wire_rx_bytes{0};
  std::uint64_t retries{0};
  std::uint64_t replay_bytes{0};
  std::uint64_t cq_errors{0};
  std::uint64_t route_errors{0};
  std::uint64_t in_flight_bytes{0};
  std::uint64_t in_flight_high_water{0};
  std::uint64_t retained_bytes{0};
  std::uint64_t retained_high_water{0};
  std::uint64_t released_bytes{0};
  std::uint64_t tx_slot_high_water{0};
  std::uint64_t rx_slot_high_water{0};
  std::uint64_t latency_count{0};
  std::uint64_t latency_total_ns{0};
  std::uint64_t service_started_unix_ns{0};
  std::uint64_t last_progress_unix_ns{0};
};

struct FabricEvent {
  std::uint32_t kind{NDP_T_EVENT_NONE};
  int status{NDP_T_OK};
  std::uint64_t peer_id{0};
  std::uint64_t message_seq{0};
  std::uint64_t useful_bytes{0};
  std::uint64_t wire_bytes{0};
  int provider_errno{0};
  std::uint32_t reason{0};
  Digest detail{};
};

class FabricEndpoint {
 public:
  explicit FabricEndpoint(FabricConfig config);
  ~FabricEndpoint();
  FabricEndpoint(const FabricEndpoint &) = delete;
  FabricEndpoint &operator=(const FabricEndpoint &) = delete;

  static int validate_config(const FabricConfig &config, std::string *why);
  int start();
  int add_peer(const std::vector<std::uint8_t> &endpoint_name,
               std::uint64_t expires_unix_ns, std::uint64_t *peer_id);
  int remove_peer(std::uint64_t peer_id);
  int send(std::uint64_t peer_id, const std::uint8_t *frame,
           std::size_t frame_bytes, std::uint64_t deadline_unix_ns,
           bool replay = false);
  int poll(std::vector<FabricEvent> *events, std::size_t capacity,
           int timeout_ms);
  int receive(std::vector<std::uint8_t> *frame, std::uint64_t *peer_id);
  int receive(std::uint8_t *frame, std::size_t capacity,
              std::size_t *frame_bytes, std::uint64_t *peer_id);
  int cancel(std::uint64_t peer_id);
  void shutdown();

  FabricFacts facts() const;
  FabricCounters counters() const;
  std::size_t peer_count() const;

 private:
  struct Slot {
    fi_context context{};
    void *buffer{nullptr};
    fid_mr *mr{nullptr};
    std::size_t capacity{0};
    std::size_t used{0};
    std::size_t index{0};
    std::uint64_t peer_id{0};
    std::uint64_t message_seq{0};
    std::uint64_t started_ns{0};
    bool active{false};
    bool receive{false};
  };
  struct Peer {
    fi_addr_t address{FI_ADDR_UNSPEC};
    std::uint64_t expires_unix_ns{0};
    bool cancelled{false};
  };
  struct ReceivedFrame {
    Slot *slot{nullptr};
    std::size_t bytes{0};
    std::uint64_t peer_id{0};
    Digest detail{};
  };

  int resolve_provider();
  int open_fabric();
  int allocate_slots(std::vector<std::unique_ptr<Slot>> *slots,
                     std::uint32_t count, bool receive);
  int post_receive(Slot *slot);
  int progress_once();
  int drain_cq(fid_cq *cq, bool receive);
  int drain_cq_error(fid_cq *cq, bool receive);
  void progress_loop();
  void emit_event(FabricEvent event);
  void telemetry(std::string_view event, const FabricEvent *detail = nullptr);
  void close_fid(fid *object);

  FabricConfig config_;
  fi_info *info_{nullptr};
  fid_fabric *fabric_{nullptr};
  fid_domain *domain_{nullptr};
  fid_ep *endpoint_{nullptr};
  fid_av *av_{nullptr};
  fid_cq *tx_cq_{nullptr};
  fid_cq *rx_cq_{nullptr};
  FabricFacts facts_{};
  std::vector<std::unique_ptr<Slot>> tx_slots_;
  std::vector<std::unique_ptr<Slot>> rx_slots_;
  std::map<std::uint64_t, Peer> peers_;
  std::uint64_t next_peer_id_{1};

  mutable std::mutex mutex_;
  mutable std::mutex api_mutex_;
  std::condition_variable event_cv_;
  std::deque<FabricEvent> events_;
  std::deque<ReceivedFrame> received_;
  std::size_t event_capacity_{0};
  FabricCounters counters_{};
  std::thread progress_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> shutting_down_{false};
  mutable std::mutex telemetry_mutex_;
};

}  // namespace emender::ndp

#endif
