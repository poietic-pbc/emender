#ifndef EMENDER_NDP_SERVICE_RPC_SERVER_HPP
#define EMENDER_NDP_SERVICE_RPC_SERVER_HPP

#include "service_core.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>

namespace emender_ndp {

struct ServiceRpcConfig {
    std::string socket_path;
    std::array<std::uint8_t, 16> run_key{};
    std::array<std::uint8_t, 32> admission_token{};
    std::uint64_t minimum_fence = 0;
    uid_t allowed_uid = 0;
};

class ServiceRpcServer final {
 public:
    ServiceRpcServer(ServiceCore& core, ServiceRpcConfig config);
    ~ServiceRpcServer();
    ServiceRpcServer(const ServiceRpcServer&) = delete;
    ServiceRpcServer& operator=(const ServiceRpcServer&) = delete;

    int start(std::string* why);
    int poll_once(int timeout_ms);
    void shutdown();
    std::size_t connection_count() const;

 private:
    void serve_connection(int socket_fd, uid_t peer_uid);
    void remove_connection(int socket_fd);
    void reap_threads();

    ServiceCore& core_;
    ServiceRpcConfig config_;
    int listener_ = -1;
    std::atomic<bool> running_{false};
    mutable std::mutex mutex_;
    std::set<int> connections_;
    std::vector<std::thread> threads_;
    std::set<std::thread::id> finished_threads_;
};

}  // namespace emender_ndp

#endif
