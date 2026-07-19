#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "rpc_server.hpp"
#include "rpc_protocol.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace emender_ndp {
namespace {

using rpc::Header;
using rpc::Opcode;
using rpc::Packet;

struct PollRequest {
    std::uint32_t capacity;
    std::int32_t timeout_ms;
};

struct ResultResponse {
    ndp_result_v1 result;
    ndp_buffer_t buffer;
};

struct Session {
    ndp_client_t handle = 0;
    std::array<std::uint8_t, 16> run{};
    std::array<std::uint8_t, 16> incarnation{};
    std::uint64_t fence = 0;
    std::uint32_t role = 0;
    bool opened = false;
};

bool constant_equal(const std::uint8_t* left, const std::uint8_t* right,
                    std::size_t size) noexcept {
    std::uint8_t difference = 0;
    for (std::size_t index = 0; index != size; ++index)
        difference |= static_cast<std::uint8_t>(left[index] ^ right[index]);
    return difference == 0;
}

bool all_zero(const std::array<std::uint8_t, 32>& value) noexcept {
    std::uint8_t combined = 0;
    for (const auto byte : value) combined |= byte;
    return combined == 0;
}

bool request_identity(const Header& header, const Session& session) noexcept {
    return header.flags == 0 && header.client == session.handle
        && header.run == session.run && header.fence == session.fence
        && header.incarnation == session.incarnation && header.status == 0;
}

bool metadata_fd_valid(int fd, std::uint64_t expected_bytes,
                       const std::uint8_t expected_digest[32]) noexcept {
    if (fd < 0 || expected_bytes == 0 || expected_bytes > (UINT64_C(16) << 20))
        return false;
    struct stat status{};
    if (::fstat(fd, &status) != 0 || status.st_size < 0
        || static_cast<std::uint64_t>(status.st_size) != expected_bytes) return false;
    const int seals = ::fcntl(fd, F_GET_SEALS);
    if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE))
            != (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE)) return false;
    Sha256 hash;
    std::array<std::uint8_t, 4096> bytes{};
    std::uint64_t offset = 0;
    while (offset != expected_bytes) {
        const std::size_t wanted = static_cast<std::size_t>(
            std::min<std::uint64_t>(bytes.size(), expected_bytes - offset));
        ssize_t got;
        do { got = ::pread(fd, bytes.data(), wanted, static_cast<off_t>(offset)); }
        while (got < 0 && errno == EINTR);
        if (got <= 0) return false;
        hash.update(bytes.data(), static_cast<std::size_t>(got));
        offset += static_cast<std::uint64_t>(got);
    }
    const auto digest = hash.finish();
    return constant_equal(digest.data(), expected_digest, digest.size());
}

Header response_header(const Header& request, int status,
                       const ServiceSnapshot& snapshot, const Session& session) {
    Header response{};
    response.opcode = request.opcode;
    response.flags = rpc::kResponse;
    response.request_id = request.request_id;
    response.client = session.handle;
    response.run = session.run;
    response.fence = session.fence;
    response.generation = snapshot.generation;
    response.attempt = snapshot.attempt;
    response.incarnation = session.incarnation;
    response.sequence = request.sequence;
    response.extent = request.extent;
    response.layout_digest = snapshot.layout_digest;
    response.status = status;
    return response;
}

}  // namespace

struct LocalRpcServer::Impl {
    LocalServiceCore& core;
    LocalRpcServerConfig config;
    int listener = -1;
    std::atomic<bool> stopping{false};
    std::thread accept_thread;
    std::mutex clients_mutex;
    std::unordered_set<int> client_fds;
    std::vector<std::thread> client_threads;

    Impl(LocalServiceCore& service, LocalRpcServerConfig value)
        : core(service), config(std::move(value)) {}

    ~Impl() { shutdown(); }

    int start();
    void shutdown() noexcept;
    void accept_loop();
    void connection_loop(int fd, uid_t peer_uid);
    bool open_session(const Packet& request, uid_t peer_uid, Session& session,
                      Header& response, int& status);
    bool dispatch(Packet& request, Session& session, std::vector<std::uint8_t>& payload,
                  std::vector<int>& fds, int& status, bool& terminate);
    int send_response(int fd, const Header& request, const Session& session, int status,
                      const std::vector<std::uint8_t>& payload,
                      const std::vector<int>& fds);
};

int LocalRpcServer::Impl::start() {
    sockaddr_un address{};
    if (listener >= 0 || config.socket_path.empty()
        || config.socket_path.size() >= sizeof(address.sun_path)
        || config.max_clients == 0 || all_zero(config.admission_token)) return NDP_EINVAL;
    const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return NDP_EIO;
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, config.socket_path.data(), config.socket_path.size());
    const socklen_t length = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + config.socket_path.size());
    if (::bind(fd, reinterpret_cast<const sockaddr*>(&address), length) != 0) {
        const int saved = errno;
        ::close(fd);
        errno = saved;
        return NDP_EIO;
    }
    if (::chmod(config.socket_path.c_str(), 0600) != 0
        || ::listen(fd, static_cast<int>(config.max_clients)) != 0) {
        const int saved = errno;
        ::close(fd);
        ::unlink(config.socket_path.c_str());
        errno = saved;
        return NDP_EIO;
    }
    listener = fd;
    stopping.store(false);
    accept_thread = std::thread(&Impl::accept_loop, this);
    return NDP_OK;
}

void LocalRpcServer::Impl::accept_loop() {
    while (!stopping.load()) {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        if (fd < 0) {
            if (errno == EINTR) continue;
            if (stopping.load() || errno == EBADF || errno == EINVAL) break;
            continue;
        }
        ucred credentials{};
        socklen_t length = sizeof(credentials);
        if (::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0
            || length != sizeof(credentials)) {
            ::close(fd);
            continue;
        }
        std::lock_guard<std::mutex> lock(clients_mutex);
        if (client_fds.size() >= config.max_clients) {
            ::close(fd);
            continue;
        }
        client_fds.insert(fd);
        client_threads.emplace_back(&Impl::connection_loop, this, fd, credentials.uid);
    }
}

bool LocalRpcServer::Impl::open_session(const Packet& request, uid_t peer_uid,
                                        Session& session, Header& response,
                                        int& status) {
    ndp_open_v1 input{};
    if (request.header.opcode != Opcode::Open || request.header.flags != 0
        || request.header.client != 0 || request.header.status != 0
        || !request.fds.empty() || !rpc::payload_object(request, input)
        || peer_uid != ::geteuid()) {
        status = NDP_EINVAL;
        return false;
    }
    if (input.socket_path_len == 0 || input.socket_path_len > sizeof(input.socket_path)) {
        status = NDP_EINVAL;
        return false;
    }
    const std::string requested_path(
        reinterpret_cast<const char*>(input.socket_path), input.socket_path_len);
    const std::array<std::uint8_t, 16> run = [&] {
        std::array<std::uint8_t, 16> value{};
        std::copy(input.run_key, input.run_key + 16, value.begin()); return value;
    }();
    const std::array<std::uint8_t, 16> incarnation = [&] {
        std::array<std::uint8_t, 16> value{};
        std::copy(input.incarnation, input.incarnation + 16, value.begin()); return value;
    }();
    if (input.struct_size < sizeof(input)
        || (input.abi_version >> 16) != (NDP_ABI_V1 >> 16)
        || requested_path != config.socket_path || request.header.run != run
        || request.header.fence != input.fence_epoch
        || request.header.incarnation != incarnation
        || !constant_equal(input.admission_token, config.admission_token.data(),
                           config.admission_token.size())) {
        status = NDP_EFENCE;
        return false;
    }
    ndp_client_t handle = 0;
    status = core.client_open(&input, &handle);
    if (status != NDP_OK) return false;
    session.handle = handle;
    session.run = run;
    session.incarnation = incarnation;
    session.fence = input.fence_epoch;
    session.role = input.role;
    session.opened = true;
    response = response_header(request.header, status, core.snapshot(handle), session);
    return true;
}

int LocalRpcServer::Impl::send_response(
        int fd, const Header& request, const Session& session, int status,
        const std::vector<std::uint8_t>& payload, const std::vector<int>& fds) {
    const ServiceSnapshot snapshot = session.opened
        ? core.snapshot(session.handle) : ServiceSnapshot{};
    Header response = response_header(request, status, snapshot, session);
    return rpc::send_packet(fd, response, payload, fds);
}

bool LocalRpcServer::Impl::dispatch(
        Packet& request, Session& session, std::vector<std::uint8_t>& output,
        std::vector<int>& output_fds, int& status, bool& terminate) {
    output.clear(); output_fds.clear(); status = NDP_EINVAL; terminate = false;
    const Header& header = request.header;
    if (!request_identity(header, session) || header.opcode == Opcode::Open)
        return false;
    const ServiceSnapshot before = core.snapshot(session.handle);
    auto require_generation_header = [&] {
        return header.generation == before.generation && header.attempt == before.attempt
            && header.layout_digest == before.layout_digest;
    };
    switch (header.opcode) {
        case Opcode::Close:
            if (!request.payload.empty() || !request.fds.empty()) return false;
            status = core.client_close(session.handle);
            session.opened = false;
            terminate = true;
            return true;
        case Opcode::PollFd: {
            if (!request.payload.empty() || !request.fds.empty()) return false;
            int fd = -1;
            status = core.client_poll_fd(session.handle, &fd);
            if (status == NDP_OK) output_fds.push_back(fd);
            return true;
        }
        case Opcode::LayoutInstall: {
            ndp_layout_v1 input{};
            if (request.fds.size() != 1 || !rpc::payload_object(request, input)
                || header.extent != input.descriptor_bytes
                || header.layout_digest != [&] { std::array<std::uint8_t, 32> value{};
                    std::copy(input.layout_digest, input.layout_digest + 32, value.begin());
                    return value; }()) return false;
            input.descriptor_fd = request.fds[0];
            status = core.layout_install(session.handle, &input);
            return true;
        }
        case Opcode::BufferRegister: {
            ndp_buffer_v1 input{};
            if (request.fds.size() != 1 || !rpc::payload_object(request, input)
                || input.kind != NDP_BUFFER_MEMFD || header.sequence != input.handle_generation
                || header.extent != input.length || !require_generation_header()) return false;
            input.fd = request.fds[0];
            ndp_buffer_t handle = 0;
            status = core.buffer_register(session.handle, &input, &handle);
            if (status == NDP_OK) output = rpc::object_payload(handle);
            return true;
        }
        case Opcode::BufferAllocate: {
            ndp_alloc_v1 input{};
            if (!request.fds.empty() || !rpc::payload_object(request, input)
                || header.extent != input.bytes) return false;
            ndp_buffer_t handle = 0;
            int fd = -1;
            status = core.buffer_allocate(session.handle, &input, &handle, &fd);
            if (status == NDP_OK) {
                output = rpc::object_payload(handle);
                output_fds.push_back(fd);
            }
            return true;
        }
        case Opcode::BufferSeal:
        case Opcode::BufferRelease: {
            ndp_buffer_t handle = 0;
            if (!request.fds.empty() || !rpc::payload_object(request, handle)
                || header.sequence != handle) return false;
            status = header.opcode == Opcode::BufferSeal
                ? core.buffer_seal(session.handle, handle)
                : core.buffer_release(session.handle, handle);
            return true;
        }
        case Opcode::Submit: {
            ndp_submit_v1 input{};
            if (!request.fds.empty() || !rpc::payload_object(request, input)
                || header.sequence != input.submission_seq || !require_generation_header())
                return false;
            const std::uint64_t width = input.source_dtype == NDP_DTYPE_BF16 ? 2
                : input.source_dtype == NDP_DTYPE_F32 ? 4
                : input.source_dtype == NDP_DTYPE_F64 ? 8 : 0;
            if (width == 0 || input.element_count > UINT64_MAX / width
                || header.extent != input.element_count * width) return false;
            ndp_op_t operation = 0;
            status = core.submit(session.handle, &input, &operation);
            if (status == NDP_OK) output = rpc::object_payload(operation);
            return true;
        }
        case Opcode::Control: {
            struct ndp_control_v1 input{};
            if (!rpc::payload_object(request, input)
                || header.generation != input.generation || header.attempt != input.attempt
                || header.sequence != input.owner_epoch || header.extent != input.metadata_bytes
                || header.layout_digest != [&] { std::array<std::uint8_t, 32> value{};
                    std::copy(input.layout_digest, input.layout_digest + 32, value.begin());
                    return value; }())
                return false;
            if (input.metadata_kind == 0) {
                if (!request.fds.empty() || input.metadata_fd != -1) return false;
            } else {
                if (request.fds.size() != 1
                    || !metadata_fd_valid(request.fds[0], input.metadata_bytes,
                                          input.metadata_sha256)) {
                    status = NDP_ECHECKSUM;
                    return true;
                }
                input.metadata_fd = request.fds[0];
            }
            ndp_op_t operation = 0;
            status = core.control(session.handle, &input, &operation);
            if (status == NDP_OK) output = rpc::object_payload(operation);
            return true;
        }
        case Opcode::Poll: {
            PollRequest input{};
            if (!request.fds.empty() || !rpc::payload_object(request, input)
                || input.capacity == 0 || input.timeout_ms < 0 || input.timeout_ms > 30000
                || input.capacity > rpc::kMaxPayloadBytes / sizeof(ndp_event_v1)
                || header.sequence != static_cast<std::uint64_t>(input.timeout_ms)
                || header.extent != static_cast<std::uint64_t>(input.capacity)
                    * sizeof(ndp_event_v1)) return false;
            std::vector<ndp_event_v1> events(input.capacity);
            std::uint32_t count = 0;
            status = core.poll(session.handle, events.data(), input.capacity, &count,
                               input.timeout_ms);
            if (status == NDP_OK) {
                const auto* begin = reinterpret_cast<const std::uint8_t*>(events.data());
                output.assign(begin, begin + static_cast<std::size_t>(count)
                    * sizeof(ndp_event_v1));
            }
            return true;
        }
        case Opcode::ResultView: {
            ndp_op_t operation = 0;
            if (!request.fds.empty() || !rpc::payload_object(request, operation)
                || header.sequence != operation || !require_generation_header()) return false;
            ResultResponse response{};
            response.result.struct_size = sizeof(response.result);
            response.result.abi_version = NDP_ABI_V1;
            int fd = -1;
            status = core.result_view(session.handle, operation, &response.result,
                                      &response.buffer, &fd);
            if (status == NDP_OK) {
                output = rpc::object_payload(response);
                output_fds.push_back(fd);
            }
            return true;
        }
        case Opcode::OpRelease: {
            ndp_op_t operation = 0;
            if (!request.fds.empty() || !rpc::payload_object(request, operation)
                || header.sequence != operation) return false;
            status = core.op_release(session.handle, operation);
            return true;
        }
        case Opcode::Metrics: {
            if (!request.payload.empty() || !request.fds.empty()) return false;
            ndp_metrics_v1 metrics{};
            metrics.struct_size = sizeof(metrics);
            metrics.abi_version = NDP_ABI_V1;
            status = core.metrics(session.handle, &metrics);
            if (status == NDP_OK) output = rpc::object_payload(metrics);
            return true;
        }
        case Opcode::Open:
            return false;
    }
    return false;
}

void LocalRpcServer::Impl::connection_loop(int fd, uid_t peer_uid) {
    Session session{};
    Packet request;
    int received = rpc::recv_packet(fd, request);
    if (received == 0) {
        Header open_response{};
        int status = NDP_EINVAL;
        const bool opened = open_session(request, peer_uid, session, open_response, status);
        Header response = opened ? open_response
            : response_header(request.header, status, ServiceSnapshot{}, session);
        if (rpc::send_packet(fd, response, {}) == 0 && opened) {
            while (!stopping.load()) {
                Packet call;
                received = rpc::recv_packet(fd, call);
                if (received != 0) break;
                std::vector<std::uint8_t> payload;
                std::vector<int> fds;
                bool terminate = false;
                int call_status = NDP_EINVAL;
                const bool valid = dispatch(call, session, payload, fds,
                                            call_status, terminate);
                if (!valid) break;
                const int sent = send_response(fd, call.header, session, call_status,
                                               payload, fds);
                for (const int output_fd : fds) if (output_fd >= 0) ::close(output_fd);
                if (sent != 0 || terminate) break;
            }
        }
    }
    if (session.opened) (void)core.client_close(session.handle);
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    std::lock_guard<std::mutex> lock(clients_mutex);
    client_fds.erase(fd);
}

void LocalRpcServer::Impl::shutdown() noexcept {
    if (stopping.exchange(true)) return;
    const int listening = listener;
    listener = -1;
    if (listening >= 0) {
        ::shutdown(listening, SHUT_RDWR);
        ::close(listening);
    }
    if (accept_thread.joinable()) accept_thread.join();
    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        for (const int fd : client_fds) ::shutdown(fd, SHUT_RDWR);
    }
    for (auto& thread : client_threads) if (thread.joinable()) thread.join();
    client_threads.clear();
    {
        std::lock_guard<std::mutex> lock(clients_mutex);
        client_fds.clear();
    }
    if (!config.socket_path.empty()) ::unlink(config.socket_path.c_str());
}

LocalRpcServer::LocalRpcServer(LocalServiceCore& core, LocalRpcServerConfig config)
    : impl_(std::make_unique<Impl>(core, std::move(config))) {}
LocalRpcServer::~LocalRpcServer() = default;
int LocalRpcServer::start() { return impl_->start(); }
void LocalRpcServer::shutdown() noexcept { impl_->shutdown(); }
bool LocalRpcServer::running() const noexcept {
    return impl_->listener >= 0 && !impl_->stopping.load();
}

}  // namespace emender_ndp
