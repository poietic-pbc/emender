#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "emender/ndp.h"
#include "service_rpc_protocol.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace emender_ndp {
namespace {

std::uint64_t unix_ns() noexcept {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

template <typename T>
bool valid_input(const T* value) noexcept {
    return value != nullptr && value->struct_size >= sizeof(T)
        && (value->abi_version >> 16) == (NDP_ABI_V1 >> 16);
}

bool is_zero(const std::uint8_t* value, std::size_t bytes) noexcept {
    std::uint8_t combined = 0;
    for (std::size_t index = 0; index != bytes; ++index) combined |= value[index];
    return combined == 0;
}

class HandleSource {
 public:
    HandleSource() {
        std::random_device random;
        cookie_ = random() ^ static_cast<std::uint32_t>(::getpid())
            ^ static_cast<std::uint32_t>(unix_ns());
        if (cookie_ == 0) cookie_ = 1;
    }
    std::uint64_t next() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return (static_cast<std::uint64_t>(cookie_) << 32) | counter_++;
    }
 private:
    std::mutex mutex_;
    std::uint32_t cookie_ = 1;
    std::uint32_t counter_ = 1;
};

HandleSource& handles() {
    static HandleSource source;
    return source;
}

struct Client {
    int socket_fd = -1;
    ndp_client_t public_handle = 0;
    std::uint64_t remote_handle = 0;
    std::uint64_t next_sequence = 2;
    bool closed = false;
    std::mutex mutex;
    std::unordered_map<ndp_buffer_t, std::uint64_t> buffers;
    std::unordered_map<ndp_op_t, std::uint64_t> operations;
    std::unordered_map<std::uint64_t, ndp_op_t> remote_operations;
    ~Client() { if (socket_fd >= 0) ::close(socket_fd); }
};

class Registry {
 public:
    std::shared_ptr<Client> find(ndp_client_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = clients_.find(handle);
        return found == clients_.end() ? nullptr : found->second;
    }
    void add(const std::shared_ptr<Client>& client) {
        std::lock_guard<std::mutex> lock(mutex_);
        clients_[client->public_handle] = client;
    }
    std::shared_ptr<Client> remove(ndp_client_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = clients_.find(handle);
        if (found == clients_.end()) return nullptr;
        auto result = found->second;
        clients_.erase(found);
        return result;
    }
 private:
    std::mutex mutex_;
    std::unordered_map<ndp_client_t, std::shared_ptr<Client>> clients_;
};

Registry& registry() {
    static Registry value;
    return value;
}

int connect_socket(const ndp_open_v1& input) noexcept {
    sockaddr_un address{};
    if (input.socket_path_len == 0 || input.socket_path_len >= sizeof(address.sun_path)
        || std::memchr(input.socket_path, 0, input.socket_path_len) != nullptr)
        return -1;
    const int descriptor = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (descriptor < 0) return -1;
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, input.socket_path, input.socket_path_len);
    address.sun_path[input.socket_path_len] = '\0';
    int result;
    do {
        result = ::connect(descriptor, reinterpret_cast<const sockaddr*>(&address),
                           sizeof(address));
    } while (result < 0 && errno == EINTR);
    if (result != 0) { ::close(descriptor); return -1; }
    return descriptor;
}

int validate_response(const rpc::Packet& response, rpc::Opcode opcode,
                      std::uint64_t sequence, std::size_t expected_fds) noexcept {
    if (response.header.flags != rpc::kResponse
        || response.header.opcode != static_cast<std::uint16_t>(opcode)
        || response.header.sequence != sequence
        || response.fds.size() != expected_fds) return NDP_EIO;
    return response.header.status;
}

int transact_locked(Client& client, rpc::Opcode opcode,
                    const void* request, std::size_t request_bytes,
                    int request_fd, rpc::Packet* response,
                    std::size_t expected_success_fds = 0) noexcept {
    if (client.closed || client.socket_fd < 0 || response == nullptr)
        return NDP_ESHUTDOWN;
    const std::uint64_t sequence = client.next_sequence++;
    int result = rpc::send_packet(client.socket_fd, opcode, sequence, NDP_OK, 0,
                                  request, request_bytes, request_fd);
    if (result != NDP_OK) return result;
    result = rpc::receive_packet(client.socket_fd, response, false);
    if (result != NDP_OK) return result;
    const std::size_t response_fds = response->header.status == NDP_OK
        ? expected_success_fds : 0;
    return validate_response(*response, opcode, sequence, response_fds);
}

template <typename T>
bool response_as(const rpc::Packet& response, T* output) noexcept {
    if (output == nullptr || response.header.body_bytes != sizeof(T)) return false;
    std::memcpy(output, response.body.data(), sizeof(T));
    return true;
}

int remote_buffer(Client& client, ndp_buffer_t local, std::uint64_t* remote) noexcept {
    const auto found = client.buffers.find(local);
    if (found == client.buffers.end()) return NDP_EINVAL;
    *remote = found->second;
    return NDP_OK;
}

int remote_operation(Client& client, ndp_op_t local, std::uint64_t* remote) noexcept {
    const auto found = client.operations.find(local);
    if (found == client.operations.end()) return NDP_EINVAL;
    *remote = found->second;
    return NDP_OK;
}

ndp_buffer_t add_buffer(Client& client, std::uint64_t remote) {
    const ndp_buffer_t local = handles().next();
    client.buffers[local] = remote;
    return local;
}

ndp_op_t add_operation(Client& client, std::uint64_t remote) {
    const auto prior = client.remote_operations.find(remote);
    if (prior != client.remote_operations.end()) return prior->second;
    const ndp_op_t local = handles().next();
    client.operations[local] = remote;
    client.remote_operations[remote] = local;
    return local;
}

template <typename Callable>
int guarded(Callable&& callable) noexcept {
    try { return callable(); }
    catch (const std::bad_alloc&) { return NDP_ENOMEM; }
    catch (...) { return NDP_EIO; }
}

int open_client(const ndp_open_v1* input, ndp_client_t* output) {
    if (!valid_input(input) || output == nullptr) return input
        && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    if ((input->role != NDP_ROLE_TRAINER && input->role != NDP_ROLE_CONTROLLER)
        || input->flags != 0 || input->socket_path_len == 0
        || input->socket_path_len > sizeof(input->socket_path)
        || input->fence_epoch == 0 || input->deadline_unix_ns <= unix_ns()
        || is_zero(input->run_key, sizeof(input->run_key))
        || is_zero(input->admission_token, sizeof(input->admission_token))) return NDP_EINVAL;
    const int socket_fd = connect_socket(*input);
    if (socket_fd < 0) return NDP_EROUTE;
    rpc::Packet response;
    int result = rpc::send_packet(socket_fd, rpc::Opcode::open, 1, NDP_OK, 0,
                                  input, sizeof(*input));
    if (result == NDP_OK) result = rpc::receive_packet(socket_fd, &response, false);
    std::uint64_t remote = 0;
    if (result == NDP_OK) result = validate_response(response, rpc::Opcode::open, 1, 0);
    if (result == NDP_OK && !response_as(response, &remote)) result = NDP_EIO;
    if (result != NDP_OK || remote == 0) { ::close(socket_fd); return result; }
    auto client = std::make_shared<Client>();
    client->socket_fd = socket_fd;
    client->remote_handle = remote;
    client->public_handle = handles().next();
    registry().add(client);
    *output = client->public_handle;
    return NDP_OK;
}

}  // namespace
}  // namespace emender_ndp

extern "C" {

uint32_t ndp_abi_version(void) { return NDP_ABI_V1; }

const char* ndp_error_string(int code) {
    switch (code) {
        case NDP_OK: return "success";
        case NDP_IN_PROGRESS: return "accepted/in progress";
        case NDP_EINVAL: return "invalid argument";
        case NDP_EVERSION: return "ABI version mismatch";
        case NDP_ESTATE: return "invalid lifecycle state";
        case NDP_EFENCE: return "stale allocation fence";
        case NDP_ESTALE: return "stale generation, attempt, owner, or incarnation";
        case NDP_ECONFLICT: return "conflicting identity replay";
        case NDP_ECHECKSUM: return "checksum mismatch";
        case NDP_ENONFINITE: return "nonfinite dense input";
        case NDP_EBOUNDS: return "configured byte or slot bound exceeded";
        case NDP_ECREDIT: return "credit exhausted";
        case NDP_EDEADLINE: return "absolute deadline expired";
        case NDP_EROUTE: return "route failure";
        case NDP_EPROVIDER: return "provider or numerical platform unsupported";
        case NDP_ENOMEM: return "bounded allocation unavailable";
        case NDP_EIO: return "local I/O failure";
        case NDP_ESHUTDOWN: return "native service is draining or stopped";
        default: return "unknown native data-plane result";
    }
}

int ndp_client_open_v1(const ndp_open_v1* input, ndp_client_t* output) {
    return emender_ndp::guarded([&] { return emender_ndp::open_client(input, output); });
}

int ndp_client_poll_fd_v1(ndp_client_t handle, int* output) {
    if (output == nullptr) return NDP_EINVAL;
    *output = -1;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    const int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::poll_fd, nullptr, 0, -1, &response, 1);
    if (result != NDP_OK) return result;
    if (response.header.body_bytes != 0) return NDP_EIO;
    *output = response.take_fd();
    return *output >= 0 ? NDP_OK : NDP_EIO;
}

int ndp_client_close_v1(ndp_client_t handle) {
    auto client = emender_ndp::registry().remove(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    const int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::close, nullptr, 0, -1, &response);
    client->closed = true;
    client->buffers.clear();
    client->operations.clear();
    client->remote_operations.clear();
    if (client->socket_fd >= 0) {
        ::close(client->socket_fd);
        client->socket_fd = -1;
    }
    return result;
}

int ndp_layout_install_v1(ndp_client_t handle, const ndp_layout_v1* input) {
    if (!emender_ndp::valid_input(input)) return input
        && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    auto client = emender_ndp::registry().find(handle);
    if (!client || input->descriptor_fd < 0) return NDP_EINVAL;
    ndp_layout_v1 wire = *input;
    wire.descriptor_fd = -1;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    const int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::layout_install, &wire, sizeof(wire),
        input->descriptor_fd, &response);
    return result == NDP_OK && response.header.body_bytes != 0 ? NDP_EIO : result;
}

int ndp_buffer_register_v1(ndp_client_t handle, const ndp_buffer_v1* input,
                           ndp_buffer_t* output) {
    if (!emender_ndp::valid_input(input) || output == nullptr) return input
        && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    auto client = emender_ndp::registry().find(handle);
    if (!client || input->kind != NDP_BUFFER_MEMFD || input->fd < 0) return NDP_EINVAL;
    ndp_buffer_v1 wire = *input;
    wire.fd = -1;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    std::uint64_t remote = 0;
    int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::buffer_register, &wire, sizeof(wire),
        input->fd, &response);
    if (result == NDP_OK && !emender_ndp::response_as(response, &remote)) result = NDP_EIO;
    if (result == NDP_OK) *output = emender_ndp::add_buffer(*client, remote);
    return result;
}

int ndp_buffer_allocate_v1(ndp_client_t handle, const ndp_alloc_v1* input,
                           ndp_buffer_t* output, int* output_fd) {
    if (!emender_ndp::valid_input(input) || output == nullptr || output_fd == nullptr)
        return input && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16)
            ? NDP_EVERSION : NDP_EINVAL;
    *output = 0; *output_fd = -1;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    std::uint64_t remote = 0;
    int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::buffer_allocate, input, sizeof(*input),
        -1, &response, 1);
    if (result == NDP_OK && !emender_ndp::response_as(response, &remote)) result = NDP_EIO;
    if (result == NDP_OK) {
        const int descriptor = response.take_fd();
        if (descriptor < 0) return NDP_EIO;
        *output = emender_ndp::add_buffer(*client, remote);
        *output_fd = descriptor;
    }
    return result;
}

int ndp_buffer_seal_v1(ndp_client_t handle, ndp_buffer_t buffer) {
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    std::uint64_t remote = 0;
    int result = emender_ndp::remote_buffer(*client, buffer, &remote);
    if (result != NDP_OK) return result;
    emender_ndp::rpc::Packet response;
    result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::buffer_seal, &remote, sizeof(remote),
        -1, &response);
    return result == NDP_OK && response.header.body_bytes != 0 ? NDP_EIO : result;
}

int ndp_buffer_release_v1(ndp_client_t handle, ndp_buffer_t buffer) {
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    std::uint64_t remote = 0;
    int result = emender_ndp::remote_buffer(*client, buffer, &remote);
    if (result != NDP_OK) return result;
    emender_ndp::rpc::Packet response;
    result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::buffer_release, &remote, sizeof(remote),
        -1, &response);
    if (result == NDP_OK || result == NDP_EFENCE || result == NDP_ESHUTDOWN) {
        client->buffers.erase(buffer);
    }
    return result == NDP_OK && response.header.body_bytes != 0 ? NDP_EIO : result;
}

int ndp_submit_local_v1(ndp_client_t handle, const ndp_submit_v1* input,
                        ndp_op_t* output) {
    if (!emender_ndp::valid_input(input) || output == nullptr) return input
        && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    ndp_submit_v1 wire = *input;
    int result = emender_ndp::remote_buffer(*client, input->buffer, &wire.buffer);
    if (result != NDP_OK) return result;
    emender_ndp::rpc::Packet response;
    std::uint64_t remote = 0;
    result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::submit, &wire, sizeof(wire), -1, &response);
    if (result == NDP_OK && !emender_ndp::response_as(response, &remote)) result = NDP_EIO;
    if (result == NDP_OK) *output = emender_ndp::add_operation(*client, remote);
    return result;
}

int ndp_control_v1(ndp_client_t handle, const struct ndp_control_v1* input,
                   ndp_op_t* output) {
    if (!emender_ndp::valid_input(input) || output == nullptr) return input
        && (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    struct ndp_control_v1 wire = *input;
    const int descriptor = input->metadata_kind == 0 ? -1 : input->metadata_fd;
    wire.metadata_fd = -1;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    std::uint64_t remote = 0;
    int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::control, &wire, sizeof(wire),
        descriptor, &response);
    if (result == NDP_OK && !emender_ndp::response_as(response, &remote)) result = NDP_EIO;
    if (result == NDP_OK) *output = emender_ndp::add_operation(*client, remote);
    return result;
}

int ndp_poll_v1(ndp_client_t handle, ndp_event_v1* events,
                uint32_t capacity, uint32_t* count, int timeout_ms) {
    if (events == nullptr || count == nullptr || capacity == 0 || timeout_ms < 0
        || timeout_ms > 30000) return NDP_EINVAL;
    *count = 0;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::PollRequest request{std::min<std::uint32_t>(capacity, 16), timeout_ms};
    emender_ndp::rpc::Packet response;
    int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::poll, &request, sizeof(request), -1, &response);
    if (result != NDP_OK) return result;
    emender_ndp::rpc::PollResponsePrefix prefix{};
    if (response.header.body_bytes < sizeof(prefix)) return NDP_EIO;
    std::memcpy(&prefix, response.body.data(), sizeof(prefix));
    if (prefix.reserved != 0 || prefix.count > request.capacity
        || response.header.body_bytes != sizeof(prefix) + prefix.count * sizeof(ndp_event_v1))
        return NDP_EIO;
    std::memcpy(events, response.body.data() + sizeof(prefix),
                prefix.count * sizeof(ndp_event_v1));
    for (std::uint32_t index = 0; index != prefix.count; ++index) {
        if (events[index].op != 0) {
            const auto found = client->remote_operations.find(events[index].op);
            events[index].op = found == client->remote_operations.end() ? 0 : found->second;
        }
    }
    *count = prefix.count;
    return NDP_OK;
}

int ndp_result_view_v1(ndp_client_t handle, ndp_op_t op,
                       ndp_result_v1* result, ndp_buffer_t* buffer, int* output_fd) {
    if (!emender_ndp::valid_input(result) || buffer == nullptr || output_fd == nullptr)
        return result && (result->abi_version >> 16) != (NDP_ABI_V1 >> 16)
            ? NDP_EVERSION : NDP_EINVAL;
    *buffer = 0; *output_fd = -1;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::ResultViewRequest request{};
    int status = emender_ndp::remote_operation(*client, op, &request.operation);
    if (status != NDP_OK) return status;
    request.result_prefix = *result;
    emender_ndp::rpc::Packet response;
    status = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::result_view, &request, sizeof(request),
        -1, &response, 1);
    emender_ndp::rpc::ResultViewResponse value{};
    if (status == NDP_OK && !emender_ndp::response_as(response, &value)) status = NDP_EIO;
    if (status == NDP_OK) {
        const int descriptor = response.take_fd();
        if (descriptor < 0) return NDP_EIO;
        *result = value.result;
        *buffer = emender_ndp::add_buffer(*client, value.buffer);
        *output_fd = descriptor;
    }
    return status;
}

int ndp_op_release_v1(ndp_client_t handle, ndp_op_t op) {
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    std::uint64_t remote = 0;
    int result = emender_ndp::remote_operation(*client, op, &remote);
    if (result != NDP_OK) return result;
    emender_ndp::rpc::Packet response;
    result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::op_release, &remote, sizeof(remote),
        -1, &response);
    if (result == NDP_OK || result == NDP_EFENCE || result == NDP_ESHUTDOWN) {
        client->operations.erase(op);
        client->remote_operations.erase(remote);
    }
    return result == NDP_OK && response.header.body_bytes != 0 ? NDP_EIO : result;
}

int ndp_client_metrics_v1(ndp_client_t handle, ndp_metrics_v1* output) {
    if (!emender_ndp::valid_input(output)) return output
        && (output->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    auto client = emender_ndp::registry().find(handle);
    if (!client) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(client->mutex);
    emender_ndp::rpc::Packet response;
    int result = emender_ndp::transact_locked(
        *client, emender_ndp::rpc::Opcode::metrics, output, sizeof(*output),
        -1, &response);
    if (result == NDP_OK && !emender_ndp::response_as(response, output)) result = NDP_EIO;
    return result;
}

}  // extern "C"
