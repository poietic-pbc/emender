#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "emender/ndp.h"
#include "service_rpc_protocol.hpp"
#include "sha256.hpp"

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/memfd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

using Digest = std::array<std::uint8_t, 32>;
constexpr std::array<std::uint8_t, 16> kRun{{
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
constexpr std::array<std::uint8_t, 32> kToken{{
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}};

std::uint64_t deadline(unsigned seconds = 30) {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count())
        + static_cast<std::uint64_t>(seconds) * UINT64_C(1000000000);
}

std::string hex(const std::uint8_t* data, std::size_t bytes) {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::string result(bytes * 2, '0');
    for (std::size_t index = 0; index != bytes; ++index) {
        result[index * 2] = alphabet[data[index] >> 4];
        result[index * 2 + 1] = alphabet[data[index] & 15U];
    }
    return result;
}

int memfd(const char* name) {
    return static_cast<int>(::syscall(SYS_memfd_create, name,
        static_cast<unsigned>(MFD_CLOEXEC | MFD_ALLOW_SEALING)));
}

bool write_all(int fd, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    while (bytes != 0) {
        const ssize_t wrote = ::write(fd, cursor, bytes);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return false;
        cursor += static_cast<std::size_t>(wrote);
        bytes -= static_cast<std::size_t>(wrote);
    }
    return true;
}

int sealed_memfd(const char* name, const void* data, std::size_t bytes) {
    const int fd = memfd(name);
    if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0
        || !write_all(fd, data, bytes)
        || ::fcntl(fd, F_ADD_SEALS,
                   F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
        if (fd >= 0) ::close(fd);
        return -1;
    }
    return fd;
}

void append_u16(std::vector<std::uint8_t>* output, std::uint16_t value) {
    output->push_back(static_cast<std::uint8_t>(value));
    output->push_back(static_cast<std::uint8_t>(value >> 8));
}
void append_u32(std::vector<std::uint8_t>* output, std::uint32_t value) {
    for (unsigned index = 0; index != 4; ++index)
        output->push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}
void append_u64(std::vector<std::uint8_t>* output, std::uint64_t value) {
    for (unsigned index = 0; index != 8; ++index)
        output->push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

std::vector<std::uint8_t> layout_descriptor(Digest* digest) {
    std::vector<std::uint8_t> value{'N','D','P','L','A','Y','1',0};
    append_u32(&value, 1); append_u32(&value, 1);
    append_u64(&value, 4); append_u64(&value, 64);
    append_u32(&value, 1); append_u32(&value, 0);
    append_u16(&value, 4);
    value.insert(value.end(), {'f','l','a','t'});
    append_u16(&value, NDP_DTYPE_F32); append_u16(&value, 1);
    append_u64(&value, 4); append_u64(&value, 0); append_u64(&value, 4);
    emender_ndp::Sha256 hash;
    static constexpr char domain[] = "emender-ndp-layout-v1\0";
    hash.update(domain, sizeof(domain) - 1);
    hash.update(value.data(), value.size());
    *digest = hash.finish();
    return value;
}

ndp_open_v1 open_request(const std::string& socket_path, std::uint32_t role,
                         std::uint64_t fence, std::uint8_t identity,
                         bool correct_token = true) {
    ndp_open_v1 value{};
    value.struct_size = sizeof(value); value.abi_version = NDP_ABI_V1;
    value.role = role;
    value.socket_path_len = static_cast<std::uint32_t>(socket_path.size());
    std::memcpy(value.socket_path, socket_path.data(), socket_path.size());
    std::copy(kRun.begin(), kRun.end(), value.run_key);
    value.fence_epoch = fence;
    std::fill(std::begin(value.worker_key), std::end(value.worker_key), identity);
    std::fill(std::begin(value.incarnation), std::end(value.incarnation), identity + 1);
    if (correct_token) std::copy(kToken.begin(), kToken.end(), value.admission_token);
    else std::fill(std::begin(value.admission_token), std::end(value.admission_token), 9);
    value.deadline_unix_ns = deadline();
    return value;
}

int open_client(const std::string& path, std::uint32_t role, std::uint64_t fence,
                std::uint8_t identity, ndp_client_t* client) {
    const auto value = open_request(path, role, fence, identity);
    return ndp_client_open_v1(&value, client);
}

struct ndp_control_v1 control_request(std::uint32_t command, std::uint64_t fence,
                                      const Digest& layout,
                                      std::uint64_t generation = 7,
                                      std::uint32_t attempt = 2,
                                      std::uint64_t owner_epoch = 3) {
    struct ndp_control_v1 value{};
    value.struct_size = sizeof(value); value.abi_version = NDP_ABI_V1;
    value.command = command;
    std::copy(kRun.begin(), kRun.end(), value.run_key);
    value.fence_epoch = fence; value.generation = generation;
    value.attempt = attempt; value.owner_epoch = owner_epoch;
    value.deadline_unix_ns = deadline(); value.metadata_fd = -1;
    std::copy(layout.begin(), layout.end(), value.layout_digest);
    std::fill(std::begin(value.base_digest), std::end(value.base_digest), 5);
    std::fill(std::begin(value.plan_digest), std::end(value.plan_digest), 6);
    return value;
}

pid_t start_service(const std::string& executable, const std::string& socket_path,
                    std::uint64_t initial_fence) {
    const pid_t child = ::fork();
    if (child != 0) return child;
    const std::string run = hex(kRun.data(), kRun.size());
    const std::string token = hex(kToken.data(), kToken.size());
    const std::string fence = std::to_string(initial_fence);
    ::execl(executable.c_str(), executable.c_str(),
            "--provider", "tcp;ofi_rxm", "--test-only", "--serve",
            "--bind-node", "127.0.0.1", "--payload-max", "4096",
            "--tx-slots", "1", "--rx-slots", "1",
            "--socket", socket_path.c_str(), "--run-key", run.c_str(),
            "--admission-token", token.c_str(), "--initial-fence", fence.c_str(),
            "--deadline-seconds", "120", static_cast<char*>(nullptr));
    _exit(127);
}

bool wait_socket(const std::string& path, pid_t service) {
    for (unsigned attempt = 0; attempt != 500; ++attempt) {
        struct stat status{};
        if (::lstat(path.c_str(), &status) == 0 && S_ISSOCK(status.st_mode)
            && (status.st_mode & 0777) == 0600) return true;
        int child_status = 0;
        if (::waitpid(service, &child_status, WNOHANG) == service) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool stop_service(pid_t service, const std::string& socket_path) {
    if (::kill(service, SIGTERM) != 0) return false;
    for (unsigned attempt = 0; attempt != 500; ++attempt) {
        int status = 0;
        const pid_t result = ::waitpid(service, &status, WNOHANG);
        if (result == service) {
            struct stat ignored{};
            return WIFEXITED(status) && WEXITSTATUS(status) == 0
                && ::lstat(socket_path.c_str(), &ignored) != 0 && errno == ENOENT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    (void)::kill(service, SIGKILL);
    (void)::waitpid(service, nullptr, 0);
    return false;
}

struct ServiceGuard {
    pid_t process = -1;
    std::string socket_path;
    ~ServiceGuard() {
        if (process <= 0) return;
        (void)::kill(process, SIGTERM);
        for (unsigned attempt = 0; attempt != 200; ++attempt) {
            if (::waitpid(process, nullptr, WNOHANG) == process) {
                (void)::unlink(socket_path.c_str());
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        (void)::kill(process, SIGKILL);
        (void)::waitpid(process, nullptr, 0);
        (void)::unlink(socket_path.c_str());
    }
};

int trainer_submission(const std::string& path, const std::array<float, 4>& values,
                       std::uint64_t weight, int expected_result) {
    ndp_client_t trainer = 0;
    if (open_client(path, NDP_ROLE_TRAINER, 1, 40, &trainer) != NDP_OK) return 10;
    const int source_fd = sealed_memfd("ndp-trainer-input", values.data(), sizeof(values));
    if (source_fd < 0) return 11;
    Digest layout{};
    (void)layout_descriptor(&layout);
    ndp_buffer_v1 buffer{};
    buffer.struct_size = sizeof(buffer); buffer.abi_version = NDP_ABI_V1;
    buffer.kind = NDP_BUFFER_MEMFD; buffer.flags = NDP_BUFFER_READ;
    buffer.length = sizeof(values); buffer.handle_generation = 1;
    buffer.fd = source_fd;
    std::copy(layout.begin(), layout.end(), buffer.layout_digest);
    ndp_buffer_t buffer_handle = 0;
    if (ndp_buffer_register_v1(trainer, &buffer, &buffer_handle) != NDP_OK) return 12;
    ndp_submit_v1 submit{};
    submit.struct_size = sizeof(submit); submit.abi_version = NDP_ABI_V1;
    submit.buffer = buffer_handle;
    std::fill(std::begin(submit.trainer_key), std::end(submit.trainer_key), 41);
    std::fill(std::begin(submit.trainer_incarnation),
              std::end(submit.trainer_incarnation), 42);
    submit.submission_seq = 9; submit.weight = weight;
    submit.element_count = values.size(); submit.source_dtype = NDP_DTYPE_F32;
    submit.deadline_unix_ns = deadline(20);
    const Digest source_digest = emender_ndp::Sha256::digest(values.data(), sizeof(values));
    std::copy(source_digest.begin(), source_digest.end(), submit.source_buffer_sha256);
    ndp_op_t operation = 0;
    const int result = ndp_submit_local_v1(trainer, &submit, &operation);
    ::close(source_fd);
    // Intentionally do not release the admitted buffer/operation: process exit
    // must leave the service-owned contribution replayable.
    const int closed = ndp_client_close_v1(trainer);
    if (result != expected_result || closed != NDP_OK) {
        std::fprintf(stderr,
            "trainer submission mismatch result=%d expected=%d close=%d weight=%llu\n",
            result, expected_result, closed,
            static_cast<unsigned long long>(weight));
        return 13;
    }
    return 0;
}

bool run_child_submission(const std::string& path,
                          const std::array<float, 4>& values,
                          std::uint64_t weight, int expected_result) {
    const pid_t child = ::fork();
    if (child == 0) _exit(trainer_submission(path, values, weight, expected_result));
    int status = 0;
    return ::waitpid(child, &status, 0) == child
        && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

int raw_connect(const std::string& path) {
    const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    sockaddr_un address{}; address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.data(), path.size());
    address.sun_path[path.size()] = 0;
    if (fd < 0 || ::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        if (fd >= 0) ::close(fd);
        return -1;
    }
    return fd;
}

bool reject_oversized_frame(const std::string& path) {
    const int socket_fd = raw_connect(path);
    if (socket_fd < 0) return false;
    std::array<std::uint8_t, emender_ndp::rpc::kMaxFrameBytes + 1> oversized{};
    const bool sent = ::send(socket_fd, oversized.data(), oversized.size(), MSG_NOSIGNAL)
        == static_cast<ssize_t>(oversized.size());
    char byte = 0;
    const ssize_t received = ::recv(socket_fd, &byte, 1, 0);
    ::close(socket_fd);
    return sent && received == 0;
}

bool reject_fd_smuggling(const std::string& path, int descriptor) {
    const int socket_fd = raw_connect(path);
    if (socket_fd < 0) return false;
    const ndp_open_v1 open = open_request(path, NDP_ROLE_TRAINER, 1, 60);
    if (emender_ndp::rpc::send_packet(socket_fd, emender_ndp::rpc::Opcode::open,
            1, NDP_OK, 0, &open, sizeof(open)) != NDP_OK) return false;
    emender_ndp::rpc::Packet response;
    const int open_receive = emender_ndp::rpc::receive_packet(socket_fd, &response, false);
    if (open_receive != NDP_OK || response.header.status != NDP_OK) {
        std::fprintf(stderr, "raw smuggling open failed receive=%d status=%d\n",
                     open_receive, response.header.status);
        return false;
    }

    ndp_layout_v1 layout{};
    layout.struct_size = sizeof(layout); layout.abi_version = NDP_ABI_V1;
    emender_ndp::rpc::Header header{};
    header.opcode = static_cast<std::uint16_t>(emender_ndp::rpc::Opcode::layout_install);
    header.body_bytes = sizeof(layout); header.fd_count = 2; header.sequence = 2;
    iovec vectors[2]{{&header, sizeof(header)}, {&layout, sizeof(layout)}};
    std::array<std::uint8_t, CMSG_SPACE(2 * sizeof(int))> ancillary{};
    msghdr message{}; message.msg_iov = vectors; message.msg_iovlen = 2;
    message.msg_control = ancillary.data(); message.msg_controllen = ancillary.size();
    cmsghdr* rights = CMSG_FIRSTHDR(&message);
    rights->cmsg_level = SOL_SOCKET; rights->cmsg_type = SCM_RIGHTS;
    rights->cmsg_len = CMSG_LEN(2 * sizeof(int));
    const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 3);
    if (duplicate < 0) { ::close(socket_fd); return false; }
    const int descriptors[2] = {descriptor, duplicate};
    std::memcpy(CMSG_DATA(rights), descriptors, sizeof(descriptors));
    const bool sent = ::sendmsg(socket_fd, &message, MSG_NOSIGNAL)
        == static_cast<ssize_t>(sizeof(header) + sizeof(layout));
    ::close(duplicate);
    emender_ndp::rpc::Packet rejected;
    const int reject_receive = sent
        ? emender_ndp::rpc::receive_packet(socket_fd, &rejected, false) : NDP_EIO;
    const bool correct = sent && reject_receive == NDP_OK
        && rejected.header.status == NDP_EINVAL && rejected.fds.empty();
    if (!correct) {
        std::fprintf(stderr,
            "raw smuggling reject failed sent=%d receive=%d status=%d fds=%zu flags=%u\n",
            sent ? 1 : 0, reject_receive, rejected.header.status,
            rejected.fds.size(), rejected.header.flags);
    }
    ::close(socket_fd);
    return correct;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    char directory_template[] = "/tmp/emender-ndp-rpc-XXXXXX";
    char* directory = ::mkdtemp(directory_template);
    if (directory == nullptr) return 2;
    const std::string socket_path = std::string(directory) + "/service.sock";
    pid_t service = start_service(argv[1], socket_path, 1);
    ServiceGuard service_guard{service, socket_path};
    if (service <= 0 || !wait_socket(socket_path, service)) return 3;

    // Admission token and frame/descriptor bounds fail before any core state.
    ndp_open_v1 bad = open_request(socket_path, NDP_ROLE_TRAINER, 1, 9, false);
    ndp_client_t ignored_client = 0;
    if (ndp_client_open_v1(&bad, &ignored_client) != NDP_EINVAL
        || ignored_client != 0 || !reject_oversized_frame(socket_path)) return 4;
    const std::array<float, 4> values{{1.0F, 3.0F, 5.0F, 7.0F}};
    const int scratch_fd = sealed_memfd("smuggled", values.data(), sizeof(values));
    if (scratch_fd < 0 || !reject_fd_smuggling(socket_path, scratch_fd)) return 5;
    ::close(scratch_fd);

    Digest layout_digest{};
    const auto descriptor = layout_descriptor(&layout_digest);
    ndp_client_t controller = 0;
    if (open_client(socket_path, NDP_ROLE_CONTROLLER, 1, 20, &controller) != NDP_OK)
        return 6;
    const int layout_fd = sealed_memfd("ndp-layout", descriptor.data(), descriptor.size());
    ndp_layout_v1 layout{};
    layout.struct_size = sizeof(layout); layout.abi_version = NDP_ABI_V1;
    layout.descriptor_fd = layout_fd; layout.descriptor_bytes = descriptor.size();
    std::copy(layout_digest.begin(), layout_digest.end(), layout.layout_digest);
    if (layout_fd < 0 || ndp_layout_install_v1(controller, &layout) != NDP_OK) return 7;
    ::close(layout_fd);
    auto install = control_request(NDP_CONTROL_INSTALL_GENERATION, 1, layout_digest);
    ndp_op_t install_op = 0;
    if (ndp_control_v1(controller, &install, &install_op) != NDP_OK
        || ndp_op_release_v1(controller, install_op) != NDP_OK
        || ndp_client_close_v1(controller) != NDP_OK) return 8;

    // A rejected nonfinite producer cannot mutate the service accumulator.
    const std::array<float, 4> nonfinite{{1.0F, 3.0F, NAN, 7.0F}};
    if (!run_child_submission(socket_path, nonfinite, 3, NDP_ENONFINITE)) return 9;
    // The admitted producer exits; identical reconnect is idempotent and a
    // conflicting replay is rejected while the first sealed memfd stays live.
    if (!run_child_submission(socket_path, values, 3, NDP_OK)
        || !run_child_submission(socket_path, values, 3, NDP_OK)
        || !run_child_submission(socket_path, values, 4, NDP_ECONFLICT)) return 10;

    if (open_client(socket_path, NDP_ROLE_CONTROLLER, 1, 21, &controller) != NDP_OK)
        return 11;
    auto freeze = control_request(NDP_CONTROL_FREEZE, 1, layout_digest);
    ndp_op_t freeze_op = 0;
    if (ndp_control_v1(controller, &freeze, &freeze_op) != NDP_OK) return 12;
    auto finalize = control_request(NDP_CONTROL_FINALIZE_OWNERS, 1, layout_digest);
    ndp_op_t result_op = 0;
    if (ndp_control_v1(controller, &finalize, &result_op) != NDP_OK) return 13;
    ndp_result_v1 result{};
    result.struct_size = sizeof(result); result.abi_version = NDP_ABI_V1;
    ndp_buffer_t result_buffer = 0; int result_fd = -1;
    if (ndp_result_view_v1(controller, result_op, &result, &result_buffer,
                           &result_fd) != NDP_OK
        || (fcntl(result_fd, F_GETFL) & O_ACCMODE) != O_RDONLY
        || result.global_weight != 3 || result.result_bytes != sizeof(values)
        || result.fence_epoch != 1 || result.generation != 7 || result.attempt != 2
        || std::memcmp(result.layout_digest, layout_digest.data(), 32) != 0) return 14;
    std::array<float, 4> observed{};
    if (::pread(result_fd, observed.data(), sizeof(observed), 0) != sizeof(observed)
        || observed != values || ::pwrite(result_fd, observed.data(), sizeof(observed), 0)
            != -1 || errno != EBADF) return 15;
    ::close(result_fd);
    if (ndp_buffer_release_v1(controller, result_buffer) != NDP_OK) return 16;
    auto commit = control_request(NDP_CONTROL_COMMIT, 1, layout_digest);
    ndp_op_t commit_op = 0;
    if (ndp_control_v1(controller, &commit, &commit_op) != NDP_OK
        || ndp_op_release_v1(controller, commit_op) != NDP_OK
        || ndp_op_release_v1(controller, result_op) != NDP_OK
        || ndp_op_release_v1(controller, freeze_op) != NDP_OK) return 17;

    // A newer authenticated fence invalidates the still-connected controller.
    ndp_client_t newer = 0;
    if (open_client(socket_path, NDP_ROLE_CONTROLLER, 2, 22, &newer) != NDP_OK)
        return 18;
    ndp_metrics_v1 metrics{};
    metrics.struct_size = sizeof(metrics); metrics.abi_version = NDP_ABI_V1;
    const int stale_metrics = ndp_client_metrics_v1(controller, &metrics);
    const int old_close = ndp_client_close_v1(controller);
    const int new_close = ndp_client_close_v1(newer);
    if (stale_metrics != NDP_EFENCE || old_close != NDP_OK || new_close != NDP_OK) {
        std::fprintf(stderr, "new fence mismatch metrics=%d old-close=%d new-close=%d\n",
                     stale_metrics, old_close, new_close);
        return 19;
    }

    // A process restart shuts down old sockets and changes the handle domain.
    ndp_client_t before_restart = 0;
    if (open_client(socket_path, NDP_ROLE_TRAINER, 2, 23, &before_restart) != NDP_OK)
        return 20;
    ndp_alloc_v1 allocation{};
    allocation.struct_size = sizeof(allocation); allocation.abi_version = NDP_ABI_V1;
    allocation.flags = NDP_BUFFER_READ | NDP_BUFFER_WRITE;
    allocation.bytes = 16; allocation.deadline_unix_ns = deadline();
    ndp_buffer_t stale_buffer = 0; int stale_fd = -1;
    if (ndp_buffer_allocate_v1(before_restart, &allocation, &stale_buffer,
                               &stale_fd) != NDP_OK) return 21;
    ::close(stale_fd);
    if (!stop_service(service, socket_path)) return 22;
    service_guard.process = -1;
    if (ndp_client_metrics_v1(before_restart, &metrics) != NDP_ESHUTDOWN) return 23;
    (void)ndp_client_close_v1(before_restart);

    service = start_service(argv[1], socket_path, 3);
    service_guard.process = service;
    if (service <= 0 || !wait_socket(socket_path, service)) return 24;
    ndp_client_t restarted = 0;
    if (open_client(socket_path, NDP_ROLE_TRAINER, 3, 24, &restarted) != NDP_OK
        || ndp_buffer_release_v1(restarted, stale_buffer) != NDP_EINVAL
        || ndp_client_close_v1(restarted) != NDP_OK
        || !stop_service(service, socket_path)) return 25;
    service_guard.process = -1;
    if (::rmdir(directory) != 0) return 26;
    std::puts("persistent compiled native service RPC v1 multiprocess integration passed");
    return 0;
}
