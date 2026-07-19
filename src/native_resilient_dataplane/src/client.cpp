#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "emender/ndp.h"
#include "rpc_protocol.hpp"

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
#include <sys/socket.h>
#include <sys/un.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace emender_ndp::client {
namespace {

using rpc::Header;
using rpc::Opcode;
using rpc::Packet;

template <typename T>
bool valid_input(const T* value) noexcept {
    return value != nullptr && value->struct_size >= sizeof(T)
        && (value->abi_version >> 16) == (NDP_ABI_V1 >> 16);
}

template <typename T>
int input_error(const T* value) noexcept {
    return value && (value->abi_version >> 16) != (NDP_ABI_V1 >> 16)
        ? NDP_EVERSION : NDP_EINVAL;
}

template <typename T>
T prefix_copy(const T* input) noexcept {
    T output{};
    if (input) std::memcpy(&output, input, sizeof(T));
    output.struct_size = sizeof(T);
    output.abi_version = NDP_ABI_V1;
    return output;
}

template <std::size_t N>
std::array<std::uint8_t, N> fixed_bytes(const std::uint8_t (&input)[N]) noexcept {
    std::array<std::uint8_t, N> output{};
    std::copy(input, input + N, output.begin());
    return output;
}

struct Connection {
    int socket_fd = -1;
    ndp_client_t handle = 0;
    pid_t process = 0;
    std::uint64_t request_id = 1;
    std::array<std::uint8_t, 16> run{};
    std::array<std::uint8_t, 16> incarnation{};
    std::array<std::uint8_t, 32> layout_digest{};
    std::uint64_t fence = 0;
    std::uint64_t generation = 0;
    std::uint32_t attempt = 0;
    std::mutex mutex;

    ~Connection() { if (socket_fd >= 0) ::close(socket_fd); }

    Header base(Opcode opcode) const noexcept {
        Header header{};
        header.opcode = opcode;
        header.client = handle;
        header.run = run;
        header.fence = fence;
        header.generation = generation;
        header.attempt = attempt;
        header.incarnation = incarnation;
        header.layout_digest = layout_digest;
        return header;
    }

    int call(Header header, const std::vector<std::uint8_t>& payload,
             const std::vector<int>& fds, Packet& response) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (socket_fd < 0 || process != ::getpid()) return NDP_EINVAL;
        header.request_id = request_id++;
        if (rpc::send_packet(socket_fd, header, payload, fds) != 0) return NDP_EROUTE;
        const int received = rpc::recv_packet(socket_fd, response);
        if (received != 0) return received == 1 ? NDP_ESHUTDOWN : NDP_EROUTE;
        if (response.header.flags != rpc::kResponse
            || response.header.opcode != header.opcode
            || response.header.request_id != header.request_id
            || (handle != 0 && response.header.client != handle)) return NDP_EROUTE;
        if (response.header.status >= 0) {
            generation = response.header.generation;
            attempt = response.header.attempt;
            layout_digest = response.header.layout_digest;
        }
        return response.header.status;
    }
};

std::mutex registry_mutex;
std::unordered_map<ndp_client_t, std::shared_ptr<Connection>> registry;

std::shared_ptr<Connection> connection(ndp_client_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = registry.find(handle);
    return found == registry.end() ? nullptr : found->second;
}

std::uint64_t unix_ns() noexcept {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool zero_token(const std::uint8_t* token, std::size_t size) noexcept {
    std::uint8_t combined = 0;
    for (std::size_t index = 0; index != size; ++index) combined |= token[index];
    return combined == 0;
}

int connect_socket(const ndp_open_v1& input) noexcept {
    sockaddr_un address{};
    if (input.socket_path_len == 0 || input.socket_path_len > sizeof(address.sun_path)
        || std::find(input.socket_path, input.socket_path + input.socket_path_len,
                     static_cast<std::uint8_t>(0)) != input.socket_path + input.socket_path_len)
        return -1;
    const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, input.socket_path, input.socket_path_len);
    const socklen_t length = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + input.socket_path_len);
    int result;
    do { result = ::connect(fd, reinterpret_cast<const sockaddr*>(&address), length); }
    while (result < 0 && errno == EINTR);
    if (result != 0) { ::close(fd); return -1; }
    return fd;
}

int expect_fds(Packet& packet, std::size_t expected) noexcept {
    if (packet.fds.size() != expected) return NDP_EROUTE;
    return NDP_OK;
}

template <typename Callable>
int guarded(Callable&& callable) noexcept {
    try { return callable(); }
    catch (const std::bad_alloc&) { return NDP_ENOMEM; }
    catch (...) { return NDP_EIO; }
}

struct PollRequest {
    std::uint32_t capacity;
    std::int32_t timeout_ms;
};

struct ResultResponse {
    ndp_result_v1 result;
    ndp_buffer_t buffer;
};

}  // namespace
}  // namespace emender_ndp::client

namespace rpc = emender_ndp::rpc;

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
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input) || output == nullptr) return input_error(input);
        *output = 0;
        ndp_open_v1 request = prefix_copy(input);
        if ((request.role != NDP_ROLE_TRAINER && request.role != NDP_ROLE_CONTROLLER)
            || request.flags != 0 || request.socket_path_len == 0
            || request.socket_path_len > sizeof(request.socket_path)
            || request.fence_epoch == 0 || request.deadline_unix_ns <= unix_ns()
            || zero_token(request.admission_token, sizeof(request.admission_token)))
            return NDP_EINVAL;
        const int socket_fd = connect_socket(request);
        if (socket_fd < 0) return NDP_EROUTE;
        auto opened = std::make_shared<Connection>();
        opened->socket_fd = socket_fd;
        opened->process = ::getpid();
        opened->run = fixed_bytes(request.run_key);
        opened->fence = request.fence_epoch;
        opened->incarnation = fixed_bytes(request.incarnation);
        Header header = opened->base(Opcode::Open);
        Packet response;
        const int result = opened->call(header, rpc::object_payload(request), {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK || !response.payload.empty()
            || response.header.client == 0) return NDP_EROUTE;
        opened->handle = response.header.client;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            if (!registry.emplace(opened->handle, opened).second) return NDP_ECONFLICT;
        }
        *output = opened->handle;
        return NDP_OK;
    });
}

int ndp_client_poll_fd_v1(ndp_client_t client, int* output) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (output == nullptr) return NDP_EINVAL;
        *output = -1;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Packet response;
        const int result = current->call(current->base(Opcode::PollFd), {}, {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 1) != NDP_OK || !response.payload.empty()) return NDP_EROUTE;
        *output = response.release_fd();
        return *output >= 0 ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_client_close_v1(ndp_client_t client) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        std::shared_ptr<Connection> current;
        {
            std::lock_guard<std::mutex> lock(registry_mutex);
            const auto found = registry.find(client);
            if (found == registry.end()) return NDP_EINVAL;
            current = found->second;
            registry.erase(found);
        }
        Packet response;
        const int result = current->call(current->base(Opcode::Close), {}, {}, response);
        if (result >= 0 && (expect_fds(response, 0) != NDP_OK || !response.payload.empty()))
            return NDP_EROUTE;
        return result;
    });
}

int ndp_layout_install_v1(ndp_client_t client, const ndp_layout_v1* input) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input)) return input_error(input);
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        ndp_layout_v1 request = prefix_copy(input);
        if (request.descriptor_fd < 0) return NDP_EINVAL;
        const int fd = request.descriptor_fd;
        request.descriptor_fd = -1;
        Header header = current->base(Opcode::LayoutInstall);
        header.extent = request.descriptor_bytes;
        header.layout_digest = fixed_bytes(request.layout_digest);
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), {fd}, response);
        if (result != NDP_OK) return result;
        return expect_fds(response, 0) == NDP_OK && response.payload.empty()
            ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_buffer_register_v1(ndp_client_t client, const ndp_buffer_v1* input,
                           ndp_buffer_t* output) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input) || output == nullptr) return input_error(input);
        *output = 0;
        if (input->kind != NDP_BUFFER_MEMFD || input->fd < 0) return NDP_EPROVIDER;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        ndp_buffer_v1 request = prefix_copy(input);
        const int fd = request.fd;
        request.fd = -1;
        Header header = current->base(Opcode::BufferRegister);
        header.sequence = request.handle_generation;
        header.extent = request.length;
        header.layout_digest = fixed_bytes(request.layout_digest);
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), {fd}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK
            || !rpc::payload_object(response, *output)) return NDP_EROUTE;
        return NDP_OK;
    });
}

int ndp_buffer_allocate_v1(ndp_client_t client, const ndp_alloc_v1* input,
                           ndp_buffer_t* output, int* output_fd) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input) || output == nullptr || output_fd == nullptr)
            return input_error(input);
        *output = 0; *output_fd = -1;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        const ndp_alloc_v1 request = prefix_copy(input);
        Header header = current->base(Opcode::BufferAllocate);
        header.extent = request.bytes;
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 1) != NDP_OK
            || !rpc::payload_object(response, *output)) return NDP_EROUTE;
        *output_fd = response.release_fd();
        return *output_fd >= 0 ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_buffer_seal_v1(ndp_client_t client, ndp_buffer_t buffer) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Header header = current->base(Opcode::BufferSeal); header.sequence = buffer;
        Packet response;
        const int result = current->call(header, rpc::object_payload(buffer), {}, response);
        if (result != NDP_OK) return result;
        return expect_fds(response, 0) == NDP_OK && response.payload.empty()
            ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_buffer_release_v1(ndp_client_t client, ndp_buffer_t buffer) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Header header = current->base(Opcode::BufferRelease); header.sequence = buffer;
        Packet response;
        const int result = current->call(header, rpc::object_payload(buffer), {}, response);
        if (result != NDP_OK) return result;
        return expect_fds(response, 0) == NDP_OK && response.payload.empty()
            ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_submit_local_v1(ndp_client_t client, const ndp_submit_v1* input,
                        ndp_op_t* output) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input) || output == nullptr) return input_error(input);
        *output = 0;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        const ndp_submit_v1 request = prefix_copy(input);
        std::uint64_t width = request.source_dtype == NDP_DTYPE_BF16 ? 2
            : request.source_dtype == NDP_DTYPE_F32 ? 4
            : request.source_dtype == NDP_DTYPE_F64 ? 8 : 0;
        if (width == 0 || request.element_count > UINT64_MAX / width) return NDP_EBOUNDS;
        Header header = current->base(Opcode::Submit);
        header.sequence = request.submission_seq;
        header.extent = request.element_count * width;
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK
            || !rpc::payload_object(response, *output)) return NDP_EROUTE;
        return NDP_OK;
    });
}

int ndp_control_v1(ndp_client_t client, const struct ndp_control_v1* input,
                   ndp_op_t* output) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(input) || output == nullptr) return input_error(input);
        *output = 0;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        struct ndp_control_v1 request = prefix_copy(input);
        std::vector<int> fds;
        if (request.metadata_kind == 0) {
            if (request.metadata_fd != -1) return NDP_EINVAL;
        } else {
            if (request.metadata_fd < 0) return NDP_EINVAL;
            fds.push_back(request.metadata_fd);
            request.metadata_fd = -1;
        }
        Header header = current->base(Opcode::Control);
        header.generation = request.generation;
        header.attempt = request.attempt;
        header.sequence = request.owner_epoch;
        header.extent = request.metadata_bytes;
        header.layout_digest = fixed_bytes(request.layout_digest);
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), fds, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK
            || !rpc::payload_object(response, *output)) return NDP_EROUTE;
        return NDP_OK;
    });
}

int ndp_poll_v1(ndp_client_t client, ndp_event_v1* events,
                uint32_t capacity, uint32_t* count, int timeout_ms) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (events == nullptr || count == nullptr || capacity == 0 || timeout_ms < 0)
            return NDP_EINVAL;
        *count = 0;
        if (capacity > rpc::kMaxPayloadBytes / sizeof(ndp_event_v1)) return NDP_EBOUNDS;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        const PollRequest request{capacity, timeout_ms};
        Header header = current->base(Opcode::Poll);
        header.sequence = static_cast<std::uint64_t>(timeout_ms);
        header.extent = static_cast<std::uint64_t>(capacity) * sizeof(ndp_event_v1);
        Packet response;
        const int result = current->call(header, rpc::object_payload(request), {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK
            || response.payload.size() % sizeof(ndp_event_v1) != 0
            || response.payload.size() / sizeof(ndp_event_v1) > capacity) return NDP_EROUTE;
        *count = static_cast<std::uint32_t>(response.payload.size() / sizeof(ndp_event_v1));
        std::memcpy(events, response.payload.data(), response.payload.size());
        return NDP_OK;
    });
}

int ndp_result_view_v1(ndp_client_t client, ndp_op_t op,
                       ndp_result_v1* result, ndp_buffer_t* buffer, int* output_fd) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(result) || buffer == nullptr || output_fd == nullptr)
            return input_error(result);
        *buffer = 0; *output_fd = -1;
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Header header = current->base(Opcode::ResultView); header.sequence = op;
        Packet response;
        const int code = current->call(header, rpc::object_payload(op), {}, response);
        if (code != NDP_OK) return code;
        ResultResponse wire{};
        if (expect_fds(response, 1) != NDP_OK
            || !rpc::payload_object(response, wire)) return NDP_EROUTE;
        *result = wire.result;
        *buffer = wire.buffer;
        *output_fd = response.release_fd();
        return *output_fd >= 0 ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_op_release_v1(ndp_client_t client, ndp_op_t op) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Header header = current->base(Opcode::OpRelease); header.sequence = op;
        Packet response;
        const int result = current->call(header, rpc::object_payload(op), {}, response);
        if (result != NDP_OK) return result;
        return expect_fds(response, 0) == NDP_OK && response.payload.empty()
            ? NDP_OK : NDP_EROUTE;
    });
}

int ndp_client_metrics_v1(ndp_client_t client, ndp_metrics_v1* output) {
    using namespace emender_ndp::client;
    return guarded([&]() -> int {
        if (!valid_input(output)) return input_error(output);
        auto current = connection(client);
        if (!current) return NDP_EINVAL;
        Packet response;
        const int result = current->call(current->base(Opcode::Metrics), {}, {}, response);
        if (result != NDP_OK) return result;
        if (expect_fds(response, 0) != NDP_OK
            || !rpc::payload_object(response, *output)) return NDP_EROUTE;
        return NDP_OK;
    });
}

}  // extern "C"
