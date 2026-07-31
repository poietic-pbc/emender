#include "emender/ndp_transport.h"

#include "fabric.hpp"
#include "protocol.hpp"
#include "rpc_server.hpp"
#include "service_core.hpp"

#include <atomic>
#include <array>
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
               "[--bind-node NODE] [--payload-max BYTES] "
               "[--resident-limit BYTES] [--tx-slots N] [--rx-slots N] "
               "[--telemetry PATH] "
               "[--socket PATH (--admission-token-fd FD|"
               "--admission-token-hex HEX64)]\n";
}

bool parse_token(const std::string &text, std::array<std::uint8_t, 32> *out) {
  if (text.size() != out->size() * 2) return false;
  const auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (std::size_t index = 0; index != out->size(); ++index) {
    const int high = nibble(text[index * 2]);
    const int low = nibble(text[index * 2 + 1]);
    if (high < 0 || low < 0) return false;
    (*out)[index] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  using namespace emender::ndp;
  FabricConfig config{};
  config.payload_max = 64 * 1024;
  config.resident_limit_bytes = 64 * 1024 * 1024;
  config.deadline_unix_ns = unix_time_ns() + UINT64_C(30) * 1000 * 1000 * 1000;
  bool mode_set = false;
  bool serve = false;
  std::string telemetry_path;
  emender_ndp::LocalRpcServerConfig rpc_config;
  bool token_set = false;
  bool token_from_hex = false;
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
    else if (arg == "--admission-token-hex") {
      std::string token;
      if (!value(&token) || !parse_token(token, &rpc_config.admission_token)) {
        usage(); return 2;
      }
      token_set = true;
      token_from_hex = true;
    }
    else if (arg == "--admission-token-fd") {
      std::string descriptor;
      std::uint64_t number = 0;
      if (!value(&descriptor) || !parse_u64(descriptor, &number)
          || number > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        usage(); return 2;
      }
      const int fd = static_cast<int>(number);
      std::size_t offset = 0;
      while (offset != rpc_config.admission_token.size()) {
        const ssize_t got = ::read(fd, rpc_config.admission_token.data() + offset,
                                   rpc_config.admission_token.size() - offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { usage(); return 2; }
        offset += static_cast<std::size_t>(got);
      }
      ::close(fd);
      token_set = true;
    }
    else if (arg == "--production") { config.production = true; mode_set = true; }
    else if (arg == "--test-only") { config.production = false; mode_set = true; }
    else if (arg == "--probe") { serve = false; }
    else if (arg == "--serve") { serve = true; }
    else if (arg == "--payload-max" || arg == "--resident-limit"
             || arg == "--tx-slots" || arg == "--rx-slots") {
      std::string text;
      std::uint64_t number = 0;
      if (!value(&text) || !parse_u64(text, &number)) { usage(); return 2; }
      if (arg == "--payload-max") config.payload_max = number;
      else if (arg == "--resident-limit") config.resident_limit_bytes = number;
      else if (arg == "--tx-slots" && number <= UINT32_MAX) config.tx_slots = static_cast<std::uint32_t>(number);
      else if (arg == "--rx-slots" && number <= UINT32_MAX) config.rx_slots = static_cast<std::uint32_t>(number);
      else { usage(); return 2; }
    } else { usage(); return 2; }
  }
  if (!mode_set || config.provider.empty()) { usage(); return 2; }
  if (serve && (rpc_config.socket_path.empty() || !token_set)) { usage(); return 2; }
  if (serve && config.production && token_from_hex) {
    std::cerr << "production admission token must arrive through a protected fd\n";
    return 2;
  }
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
  FabricEndpoint endpoint(config);
  const int rc = endpoint.start();
  if (rc != NDP_T_OK) {
    std::cerr << "fabric setup failed: " << ndp_transport_error_string(rc) << '\n';
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
            << "\"max_msg_size\":" << facts.max_msg_size << ','
            << "\"mr_mode\":" << facts.mr_mode << ','
            << "\"endpoint_name_hex\":\""
            << hex(facts.endpoint_name.data(), facts.endpoint_name.size()) << "\""
            << (serve ? ",\"control_socket\":\"" + rpc_config.socket_path + "\"" : "")
            << "}\n";
  std::cout.flush();
  if (serve) {
    emender_ndp::LocalServiceCore local_core;
    emender_ndp::LocalRpcServer rpc_server(local_core, rpc_config);
    const int rpc_result = rpc_server.start();
    if (rpc_result != NDP_OK) {
      std::cerr << "local RPC setup failed with NDP status " << rpc_result << '\n';
      endpoint.shutdown();
      if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
      return 1;
    }
    std::signal(SIGINT, stop_handler); std::signal(SIGTERM, stop_handler);
    while (!stop_requested.load()) {
      std::vector<FabricEvent> events;
      (void)endpoint.poll(&events, 16, 100);
    }
    rpc_server.shutdown();
  }
  endpoint.shutdown();
  if (config.telemetry_fd >= 0) ::close(config.telemetry_fd);
  return 0;
}
