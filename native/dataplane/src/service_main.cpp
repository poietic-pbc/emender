#include "emender/ndp_transport.h"

#include "fabric.hpp"
#include "protocol.hpp"
#include "service_core.hpp"
#include "service_rpc_server.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace {

std::atomic<bool> stop_requested{false};

void stop_handler(int) { stop_requested.store(true); }

bool parse_u64(const std::string &text, std::uint64_t *out) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || *end != '\0') return false;
  *out = static_cast<std::uint64_t>(value);
  return true;
}

void usage() {
  std::cerr << "usage: ndp_cxi_service --provider NAME [--require-provider NAME] "
               "(--production|--test-only) [--probe|--serve] "
               "[--bind-node NODE] [--payload-max BYTES] [--tx-slots N] "
               "[--rx-slots N] [--telemetry PATH] [--deadline-seconds N] "
               "[--socket PATH --run-key HEX32 --admission-token HEX64 "
               "--initial-fence N]\n";
}

int hex_value(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

template <std::size_t N>
bool parse_hex(const std::string &text, std::array<std::uint8_t, N> *output) {
  if (output == nullptr || text.size() != N * 2) return false;
  for (std::size_t index = 0; index != N; ++index) {
    const int high = hex_value(text[index * 2]);
    const int low = hex_value(text[index * 2 + 1]);
    if (high < 0 || low < 0) return false;
    (*output)[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  using namespace emender::ndp;
  FabricConfig config{};
  config.payload_max = 64 * 1024;
  config.resident_limit_bytes = 64 * 1024 * 1024;
  std::uint64_t deadline_seconds = 24 * 60 * 60;
  bool mode_set = false;
  bool serve = false;
  std::string telemetry_path;
  emender_ndp::ServiceRpcConfig rpc_config{};
  std::string run_key_hex;
  std::string admission_token_hex;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&](std::string *out) {
      if (++i >= argc) return false;
      *out = argv[i]; return true;
    };
    if (arg == "--provider") { if (!value(&config.provider)) { usage(); return 2; } }
    else if (arg == "--require-provider") { if (!value(&config.require_provider)) { usage(); return 2; } }
    else if (arg == "--fabric") { if (!value(&config.fabric)) { usage(); return 2; } }
    else if (arg == "--domain") { if (!value(&config.domain)) { usage(); return 2; } }
    else if (arg == "--bind-node") { if (!value(&config.bind_node)) { usage(); return 2; } }
    else if (arg == "--telemetry") { if (!value(&telemetry_path)) { usage(); return 2; } }
    else if (arg == "--socket") { if (!value(&rpc_config.socket_path)) { usage(); return 2; } }
    else if (arg == "--run-key") { if (!value(&run_key_hex)) { usage(); return 2; } }
    else if (arg == "--admission-token") { if (!value(&admission_token_hex)) { usage(); return 2; } }
    else if (arg == "--production") { config.production = true; mode_set = true; }
    else if (arg == "--test-only") { config.production = false; mode_set = true; }
    else if (arg == "--probe") { serve = false; }
    else if (arg == "--serve") { serve = true; }
    else if (arg == "--payload-max" || arg == "--tx-slots" || arg == "--rx-slots"
             || arg == "--deadline-seconds" || arg == "--initial-fence") {
      std::string text;
      std::uint64_t number = 0;
      if (!value(&text) || !parse_u64(text, &number)) { usage(); return 2; }
      if (arg == "--payload-max") config.payload_max = number;
      else if (arg == "--tx-slots" && number <= UINT32_MAX) config.tx_slots = static_cast<std::uint32_t>(number);
      else if (arg == "--rx-slots" && number <= UINT32_MAX) config.rx_slots = static_cast<std::uint32_t>(number);
      else if (arg == "--deadline-seconds" && number > 0) deadline_seconds = number;
      else if (arg == "--initial-fence" && number > 0) rpc_config.minimum_fence = number;
      else { usage(); return 2; }
    } else { usage(); return 2; }
  }
  if (deadline_seconds > std::numeric_limits<std::uint64_t>::max()
          / (UINT64_C(1000) * 1000 * 1000)
      || unix_time_ns() > std::numeric_limits<std::uint64_t>::max()
          - deadline_seconds * UINT64_C(1000) * 1000 * 1000) {
    usage(); return 2;
  }
  config.deadline_unix_ns = unix_time_ns()
      + deadline_seconds * UINT64_C(1000) * 1000 * 1000;
  if (!mode_set || config.provider.empty()) { usage(); return 2; }
  if (serve && (rpc_config.socket_path.empty()
      || !parse_hex(run_key_hex, &rpc_config.run_key)
      || !parse_hex(admission_token_hex, &rpc_config.admission_token)
      || rpc_config.minimum_fence == 0)) {
    std::cerr << "serve mode requires a bounded socket and exact run/token/fence identity\n";
    return 2;
  }
  rpc_config.allowed_uid = ::getuid();
  if (!telemetry_path.empty()) {
    config.telemetry_fd = ::open(telemetry_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (config.telemetry_fd < 0) { std::perror("open telemetry"); return 1; }
  }
  std::string why;
  const int policy = FabricEndpoint::validate_config(config, &why);
  if (policy != NDP_T_OK) {
    std::cerr << "provider policy rejected: " << why << '\n';
    if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
    return 1;
  }
  emender_ndp::ServiceCore local_service;
  emender_ndp::ServiceRpcServer rpc_server(local_service, rpc_config);
  if (serve) {
    const int local_result = rpc_server.start(&why);
    if (local_result != NDP_OK) {
      std::cerr << "local RPC setup failed: " << why << '\n';
      if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
      return 1;
    }
  }
  FabricEndpoint endpoint(config);
  const int rc = endpoint.start();
  if (rc != NDP_T_OK) {
    std::cerr << "fabric setup failed: " << ndp_transport_error_string(rc) << '\n';
    rpc_server.shutdown();
    if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
    return 1;
  }
  const FabricFacts facts = endpoint.facts();
  std::cout << "{\"schema\":\"emender-native-dataplane-endpoint-v1\","
            << "\"provider\":\"" << facts.provider << "\","
            << "\"fabric\":\"" << facts.fabric << "\","
            << "\"domain\":\"" << facts.domain << "\","
            << "\"endpoint_type\":\"FI_EP_RDM\","
            << "\"production_provider\":" << (facts.production_provider ? "true" : "false") << ','
            << "\"local_service_core\":true,"
            << "\"metadata_rpc\":\"AF_UNIX/SOCK_SEQPACKET/v1\","
            << "\"max_msg_size\":" << facts.max_msg_size << ','
            << "\"mr_mode\":" << facts.mr_mode << ','
            << "\"endpoint_name_hex\":\""
            << hex(facts.endpoint_name.data(), facts.endpoint_name.size()) << "\"}\n";
  std::cout.flush();
  if (serve) {
    std::signal(SIGINT, stop_handler); std::signal(SIGTERM, stop_handler);
    while (!stop_requested.load() && unix_time_ns() < config.deadline_unix_ns) {
      const int local_result = rpc_server.poll_once(0);
      if (local_result != NDP_OK) {
        std::cerr << "local RPC accept failed with code " << local_result << '\n';
        break;
      }
      std::vector<FabricEvent> events;
      (void)endpoint.poll(&events, 16, 100);
    }
  }
  rpc_server.shutdown();
  endpoint.shutdown();
  if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
  return 0;
}
