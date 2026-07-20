#ifndef EMENDER_NDP_RPC_SERVER_HPP
#define EMENDER_NDP_RPC_SERVER_HPP

#include "service_core.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace emender_ndp {

struct LocalRpcServerConfig {
    std::string socket_path;
    std::array<std::uint8_t, 32> admission_token{};
    // One controller plus the 64 bounded compact reduction sources.
    std::uint32_t max_clients = 65;
};

class LocalRpcServer {
public:
    LocalRpcServer(LocalServiceCore& core, LocalRpcServerConfig config);
    ~LocalRpcServer();
    LocalRpcServer(const LocalRpcServer&) = delete;
    LocalRpcServer& operator=(const LocalRpcServer&) = delete;

    int start();
    void shutdown() noexcept;
    bool running() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace emender_ndp

#endif
