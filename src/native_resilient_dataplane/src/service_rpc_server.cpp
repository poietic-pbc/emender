#include "service_rpc_server.hpp"

#include "service_rpc_protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace emender_ndp {
namespace {

template <typename T>
bool body_as(const rpc::Packet& packet, T* output) noexcept {
    if (output == nullptr || packet.header.body_bytes != sizeof(T)) return false;
    std::memcpy(output, packet.body.data(), sizeof(T));
    return true;
}

bool constant_time_equal(const std::uint8_t* left, const std::uint8_t* right,
                         std::size_t bytes) noexcept {
    std::uint8_t difference = 0;
    for (std::size_t index = 0; index != bytes; ++index)
        difference |= static_cast<std::uint8_t>(left[index] ^ right[index]);
    return difference == 0;
}

bool valid_request_header(const rpc::Packet& packet, std::uint64_t expected) noexcept {
    return packet.header.flags == 0 && packet.header.status == 0
        && packet.header.sequence == expected && packet.header.opcode != 0;
}

int send_status(int fd, const rpc::Packet& request, int status,
                const void* body = nullptr, std::size_t body_bytes = 0,
                int send_fd = -1) noexcept {
    return rpc::send_packet(fd, static_cast<rpc::Opcode>(request.header.opcode),
                            request.header.sequence, status, rpc::kResponse,
                            body, body_bytes, send_fd);
}

}  // namespace

ServiceRpcServer::ServiceRpcServer(ServiceCore& core, ServiceRpcConfig config)
    : core_(core), config_(std::move(config)) {}

ServiceRpcServer::~ServiceRpcServer() { shutdown(); }

int ServiceRpcServer::start(std::string* why) {
    const auto fail = [&](int result, const char* message) {
        if (why != nullptr) *why = message;
        return result;
    };
    if (running_.load()) return fail(NDP_ESTATE, "RPC service is already running");
    sockaddr_un address{};
    if (config_.socket_path.empty()
        || config_.socket_path.size() >= sizeof(address.sun_path)
        || config_.minimum_fence == 0
        || std::all_of(config_.run_key.begin(), config_.run_key.end(),
                       [](std::uint8_t value) { return value == 0; })
        || std::all_of(config_.admission_token.begin(), config_.admission_token.end(),
                       [](std::uint8_t value) { return value == 0; })) {
        return fail(NDP_EINVAL, "RPC socket/run/fence/token configuration is invalid");
    }
    listener_ = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (listener_ < 0) return fail(NDP_EIO, "cannot create SOCK_SEQPACKET listener");
    const int enabled = 1;
    if (::setsockopt(listener_, SOL_SOCKET, SO_PASSCRED, &enabled, sizeof(enabled)) != 0) {
        ::close(listener_); listener_ = -1;
        return fail(NDP_EIO, "cannot enable peer credentials");
    }
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, config_.socket_path.data(), config_.socket_path.size());
    address.sun_path[config_.socket_path.size()] = '\0';
    if (::bind(listener_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(listener_); listener_ = -1;
        return fail(NDP_EIO, "cannot bind RPC socket (refusing to unlink an unknown owner)");
    }
    if (::chmod(config_.socket_path.c_str(), 0600) != 0 || ::listen(listener_, 64) != 0) {
        ::close(listener_); listener_ = -1;
        (void)::unlink(config_.socket_path.c_str());
        return fail(NDP_EIO, "cannot protect/listen on RPC socket");
    }
    running_.store(true);
    return NDP_OK;
}

int ServiceRpcServer::poll_once(int timeout_ms) {
    if (!running_.load() || listener_ < 0 || timeout_ms < 0 || timeout_ms > 1000)
        return NDP_EINVAL;
    reap_threads();
    pollfd ready{listener_, POLLIN, 0};
    int polled;
    do { polled = ::poll(&ready, 1, timeout_ms); } while (polled < 0 && errno == EINTR);
    if (polled < 0) return NDP_EIO;
    if (polled == 0 || (ready.revents & POLLIN) == 0) return NDP_OK;
    for (;;) {
        const int connection = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
        if (connection < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            return running_.load() ? NDP_EIO : NDP_ESHUTDOWN;
        }
        ucred credentials{};
        socklen_t credential_bytes = sizeof(credentials);
        if (::getsockopt(connection, SOL_SOCKET, SO_PEERCRED,
                         &credentials, &credential_bytes) != 0
            || credential_bytes != sizeof(credentials)
            || credentials.uid != config_.allowed_uid) {
            ::close(connection);
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (connections_.size() >= 64) {
                ::close(connection);
                continue;
            }
            connections_.insert(connection);
            threads_.emplace_back(&ServiceRpcServer::serve_connection, this,
                                  connection, credentials.uid);
        }
    }
    return NDP_OK;
}

void ServiceRpcServer::remove_connection(int socket_fd) {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(socket_fd);
}

void ServiceRpcServer::reap_threads() {
    std::vector<std::thread> completed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto thread = threads_.begin();
        while (thread != threads_.end()) {
            if (finished_threads_.erase(thread->get_id()) != 0) {
                completed.push_back(std::move(*thread));
                thread = threads_.erase(thread);
            } else {
                ++thread;
            }
        }
    }
    for (auto& thread : completed) {
        if (thread.joinable()) thread.join();
    }
}

void ServiceRpcServer::serve_connection(int socket_fd, uid_t peer_uid) {
    ndp_client_t client = 0;
    std::uint64_t expected_sequence = 1;
    bool close_requested = false;
    while (running_.load() && !close_requested) {
        rpc::Packet request;
        const int received = rpc::receive_packet(socket_fd, &request, false);
        if (received != NDP_OK) {
            if (received != NDP_ESHUTDOWN)
                std::fprintf(stderr, "native RPC packet rejected before dispatch: %d\n", received);
            break;
        }
        if (!valid_request_header(request, expected_sequence)) {
            (void)send_status(socket_fd, request, NDP_ECONFLICT);
            break;
        }
        ++expected_sequence;
        const auto opcode = static_cast<rpc::Opcode>(request.header.opcode);
        if (client == 0 && opcode != rpc::Opcode::open) {
            (void)send_status(socket_fd, request, NDP_ESTATE);
            break;
        }
        int result = NDP_EINVAL;
        std::uint64_t handle = 0;
        int response_fd = -1;
        switch (opcode) {
            case rpc::Opcode::open: {
                ndp_open_v1 value{};
                if (client != 0 || !request.fds.empty() || !body_as(request, &value)) break;
                const std::string supplied_path(
                    reinterpret_cast<const char*>(value.socket_path),
                    std::min<std::size_t>(value.socket_path_len, sizeof(value.socket_path)));
                if (value.socket_path_len > sizeof(value.socket_path)
                    || supplied_path != config_.socket_path
                    || peer_uid != config_.allowed_uid
                    || !constant_time_equal(value.run_key, config_.run_key.data(),
                                            config_.run_key.size())
                    || value.fence_epoch < config_.minimum_fence) {
                    result = NDP_EFENCE;
                    break;
                }
                if (!constant_time_equal(value.admission_token,
                                         config_.admission_token.data(),
                                         config_.admission_token.size())) {
                    result = NDP_EINVAL;
                    break;
                }
                result = core_.client_open(&value, &client);
                handle = client;
                break;
            }
            case rpc::Opcode::poll_fd:
                if (request.header.body_bytes == 0 && request.fds.empty())
                    result = core_.client_poll_fd(client, &response_fd);
                break;
            case rpc::Opcode::close:
                if (request.header.body_bytes == 0 && request.fds.empty()) {
                    result = core_.client_close(client);
                    if (result == NDP_OK) client = 0;
                    close_requested = true;
                }
                break;
            case rpc::Opcode::layout_install: {
                ndp_layout_v1 value{};
                if (request.fds.size() != 1 || !body_as(request, &value)) break;
                value.descriptor_fd = request.fds.front();
                result = core_.layout_install(client, &value);
                break;
            }
            case rpc::Opcode::buffer_register: {
                ndp_buffer_v1 value{};
                if (request.fds.size() != 1 || !body_as(request, &value)) break;
                value.fd = request.fds.front();
                result = core_.buffer_register(client, &value, &handle);
                break;
            }
            case rpc::Opcode::buffer_allocate: {
                ndp_alloc_v1 value{};
                if (!request.fds.empty() || !body_as(request, &value)) break;
                result = core_.buffer_allocate(client, &value, &handle, &response_fd);
                break;
            }
            case rpc::Opcode::buffer_seal:
            case rpc::Opcode::buffer_release:
            case rpc::Opcode::op_release: {
                std::uint64_t remote = 0;
                if (!request.fds.empty() || !body_as(request, &remote)) break;
                if (opcode == rpc::Opcode::buffer_seal)
                    result = core_.buffer_seal(client, remote);
                else if (opcode == rpc::Opcode::buffer_release)
                    result = core_.buffer_release(client, remote);
                else
                    result = core_.op_release(client, remote);
                break;
            }
            case rpc::Opcode::submit: {
                ndp_submit_v1 value{};
                if (!request.fds.empty() || !body_as(request, &value)) break;
                result = core_.submit(client, &value, &handle);
                break;
            }
            case rpc::Opcode::control: {
                struct ndp_control_v1 value{};
                if (!body_as(request, &value)) break;
                const std::size_t expected_fds = value.metadata_kind == 0 ? 0 : 1;
                if (request.fds.size() != expected_fds) break;
                value.metadata_fd = request.fds.empty() ? -1 : request.fds.front();
                result = core_.control(client, &value, &handle);
                break;
            }
            case rpc::Opcode::poll: {
                rpc::PollRequest value{};
                if (!request.fds.empty() || !body_as(request, &value)
                    || value.capacity == 0 || value.capacity > 16
                    || value.timeout_ms < 0 || value.timeout_ms > 30000) break;
                std::array<ndp_event_v1, 16> events{};
                std::uint32_t count = 0;
                result = core_.poll(client, events.data(), value.capacity,
                                    &count, value.timeout_ms);
                std::array<std::uint8_t, sizeof(rpc::PollResponsePrefix)
                    + 16 * sizeof(ndp_event_v1)> response{};
                rpc::PollResponsePrefix prefix{count, 0};
                std::memcpy(response.data(), &prefix, sizeof(prefix));
                std::memcpy(response.data() + sizeof(prefix), events.data(),
                            count * sizeof(ndp_event_v1));
                if (send_status(socket_fd, request, result, response.data(),
                                sizeof(prefix) + count * sizeof(ndp_event_v1)) != NDP_OK)
                    close_requested = true;
                continue;
            }
            case rpc::Opcode::result_view: {
                rpc::ResultViewRequest value{};
                if (!request.fds.empty() || !body_as(request, &value)) break;
                rpc::ResultViewResponse response{};
                response.result = value.result_prefix;
                result = core_.result_view(client, value.operation, &response.result,
                                           &response.buffer, &response_fd);
                if (send_status(socket_fd, request, result,
                                result == NDP_OK ? &response : nullptr,
                                result == NDP_OK ? sizeof(response) : 0,
                                result == NDP_OK ? response_fd : -1) != NDP_OK)
                    close_requested = true;
                if (response_fd >= 0) ::close(response_fd);
                continue;
            }
            case rpc::Opcode::metrics: {
                ndp_metrics_v1 value{};
                if (!request.fds.empty() || request.header.body_bytes != sizeof(value)) break;
                std::memcpy(&value, request.body.data(), sizeof(value));
                result = core_.metrics(client, &value);
                if (send_status(socket_fd, request, result,
                                result == NDP_OK ? &value : nullptr,
                                result == NDP_OK ? sizeof(value) : 0) != NDP_OK)
                    close_requested = true;
                continue;
            }
            default:
                result = NDP_EVERSION;
                break;
        }
        const void* response_body = nullptr;
        std::size_t response_bytes = 0;
        if (result == NDP_OK && (opcode == rpc::Opcode::open
            || opcode == rpc::Opcode::buffer_register
            || opcode == rpc::Opcode::buffer_allocate
            || opcode == rpc::Opcode::submit || opcode == rpc::Opcode::control)) {
            response_body = &handle;
            response_bytes = sizeof(handle);
        }
        if (send_status(socket_fd, request, result, response_body, response_bytes,
                        result == NDP_OK ? response_fd : -1) != NDP_OK)
            close_requested = true;
        if (response_fd >= 0) ::close(response_fd);
    }
    if (client != 0) (void)core_.client_close(client);
    (void)::shutdown(socket_fd, SHUT_RDWR);
    remove_connection(socket_fd);
    ::close(socket_fd);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_threads_.insert(std::this_thread::get_id());
    }
}

void ServiceRpcServer::shutdown() {
    if (!running_.exchange(false)) return;
    if (listener_ >= 0) {
        (void)::shutdown(listener_, SHUT_RDWR);
        ::close(listener_);
        listener_ = -1;
    }
    std::vector<int> connections;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        connections.assign(connections_.begin(), connections_.end());
    }
    for (const int fd : connections) (void)::shutdown(fd, SHUT_RDWR);
    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        threads.swap(threads_);
        finished_threads_.clear();
    }
    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        connections_.clear();
    }
    if (!config_.socket_path.empty()) (void)::unlink(config_.socket_path.c_str());
}

std::size_t ServiceRpcServer::connection_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return connections_.size();
}

}  // namespace emender_ndp
