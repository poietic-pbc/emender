#include "emender/ndp.h"

#include "rpc_protocol.hpp"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <linux/memfd.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Digest = std::array<std::uint8_t, 32>;
using Key = std::array<std::uint8_t, 16>;

constexpr std::uint64_t kGeneration = 7;
constexpr std::uint32_t kAttempt = 3;
constexpr std::uint64_t kOwnerEpoch = 11;
constexpr std::uint64_t kWeight = 3;
constexpr std::uint64_t kSubmissionSequence = 9;
constexpr std::size_t kElements = 4;
constexpr std::array<float, kElements> kValues{{1.0F, 2.0F, 3.0F, 4.0F}};
constexpr std::array<std::uint8_t, 32> kToken{{
    0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,
    0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,
    0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,
    0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,0x5a,
}};

std::string socket_path;
std::vector<std::uint8_t> layout_descriptor;
Digest layout_digest{};
Digest base_digest{};
Digest plan_digest{};
Key run_key{};
std::uint64_t generation_deadline_ns = 0;

std::uint64_t unix_ns() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::uint64_t deadline() { return unix_ns() + UINT64_C(30) * 1000 * 1000 * 1000; }

Key key(std::uint8_t value) {
  Key output{};
  output.fill(value);
  return output;
}

void append_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>(value));
  out.push_back(static_cast<std::uint8_t>(value >> 8));
}
void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
  for (unsigned shift = 0; shift != 32; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}
void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
  for (unsigned shift = 0; shift != 64; shift += 8)
    out.push_back(static_cast<std::uint8_t>(value >> shift));
}

void initialize_metadata() {
  run_key = key(0x11);
  base_digest = emender_ndp::Sha256::digest("base", 4);
  plan_digest = emender_ndp::Sha256::digest("plan", 4);
  const std::array<std::uint8_t, 8> magic{{'N','D','P','L','A','Y','1',0}};
  layout_descriptor.insert(layout_descriptor.end(), magic.begin(), magic.end());
  append_u32(layout_descriptor, 1);
  append_u32(layout_descriptor, 1);
  append_u64(layout_descriptor, kElements);
  append_u64(layout_descriptor, 64);
  append_u32(layout_descriptor, 1);
  append_u32(layout_descriptor, 0);
  static constexpr char name[] = "flat";
  append_u16(layout_descriptor, 4);
  layout_descriptor.insert(layout_descriptor.end(), name, name + 4);
  append_u16(layout_descriptor, NDP_DTYPE_F32);
  append_u16(layout_descriptor, 1);
  append_u64(layout_descriptor, kElements);
  append_u64(layout_descriptor, 0);
  append_u64(layout_descriptor, kElements);
  emender_ndp::Sha256 hash;
  static constexpr char domain[] = "emender-ndp-layout-v1\0";
  hash.update(domain, sizeof(domain) - 1);
  hash.update(layout_descriptor.data(), layout_descriptor.size());
  layout_digest = hash.finish();
}

int memfd(const char* name) {
#if defined(SYS_memfd_create)
  return static_cast<int>(::syscall(SYS_memfd_create, name,
      static_cast<unsigned>(MFD_CLOEXEC | MFD_ALLOW_SEALING)));
#else
  (void)name;
  return -1;
#endif
}

bool pwrite_all(int fd, const void* input, std::size_t size) {
  const auto* bytes = static_cast<const std::uint8_t*>(input);
  std::size_t offset = 0;
  while (offset != size) {
    const ssize_t wrote = ::pwrite(fd, bytes + offset, size - offset,
                                   static_cast<off_t>(offset));
    if (wrote < 0 && errno == EINTR) continue;
    if (wrote <= 0) return false;
    offset += static_cast<std::size_t>(wrote);
  }
  return true;
}

int sealed_memfd(const char* name, const void* input, std::size_t size,
                 bool seal_write = true) {
  const int fd = memfd(name);
  if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(size)) != 0
      || !pwrite_all(fd, input, size)) {
    if (fd >= 0) ::close(fd);
    return -1;
  }
  int seals = F_SEAL_GROW | F_SEAL_SHRINK;
  if (seal_write) seals |= F_SEAL_WRITE | F_SEAL_SEAL;
  if (::fcntl(fd, F_ADD_SEALS, seals) != 0) { ::close(fd); return -1; }
  return fd;
}

ndp_client_t open_client(std::uint32_t role, std::uint64_t fence,
                         std::uint8_t worker_byte, std::uint8_t incarnation_byte,
                         const std::array<std::uint8_t, 32>& token = kToken,
                         int* status_out = nullptr) {
  ndp_open_v1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = NDP_ABI_V1;
  request.role = role;
  request.socket_path_len = static_cast<std::uint32_t>(socket_path.size());
  std::memcpy(request.socket_path, socket_path.data(), socket_path.size());
  std::copy(run_key.begin(), run_key.end(), request.run_key);
  request.fence_epoch = fence;
  const Key worker = key(worker_byte);
  const Key incarnation = key(incarnation_byte);
  std::copy(worker.begin(), worker.end(), request.worker_key);
  std::copy(incarnation.begin(), incarnation.end(), request.incarnation);
  std::copy(token.begin(), token.end(), request.admission_token);
  request.deadline_unix_ns = generation_deadline_ns;
  ndp_client_t client = 0;
  const int status = ndp_client_open_v1(&request, &client);
  if (status_out) *status_out = status;
  return status == NDP_OK ? client : 0;
}

struct ndp_control_v1 control_request(std::uint32_t command, std::uint64_t fence) {
  struct ndp_control_v1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = NDP_ABI_V1;
  request.command = command;
  std::copy(run_key.begin(), run_key.end(), request.run_key);
  request.fence_epoch = fence;
  request.generation = kGeneration;
  request.attempt = kAttempt;
  request.owner_epoch = kOwnerEpoch;
  request.deadline_unix_ns = generation_deadline_ns;
  request.metadata_fd = -1;
  std::copy(layout_digest.begin(), layout_digest.end(), request.layout_digest);
  std::copy(base_digest.begin(), base_digest.end(), request.base_digest);
  std::copy(plan_digest.begin(), plan_digest.end(), request.plan_digest);
  return request;
}

int control(ndp_client_t client, std::uint32_t command, ndp_op_t* operation,
            std::uint64_t fence = 1) {
  struct ndp_control_v1 request = control_request(command, fence);
  return ndp_control_v1(client, &request, operation);
}

int run_child(const char* name, const std::function<int()>& function) {
  const pid_t child = ::fork();
  if (child < 0) return 1;
  if (child == 0) {
    const int result = function();
    if (result != 0) std::fprintf(stderr, "%s child failed at check %d\n", name, result);
    _exit(result == 0 ? 0 : 1);
  }
  int status = 0;
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)
      || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "%s child process failed\n", name);
    return 1;
  }
  return 0;
}

pid_t start_service(const char* executable) {
  const pid_t child = ::fork();
  if (child != 0) return child;
  (void)::prctl(PR_SET_PDEATHSIG, SIGTERM);
  if (::getppid() == 1) _exit(126);
  static constexpr char token_hex[] =
      "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";
  ::execl(executable, executable, "--provider", "tcp;ofi_rxm", "--test-only",
          "--serve", "--bind-node", "127.0.0.1", "--socket", socket_path.c_str(),
          "--admission-token-hex", token_hex, static_cast<char*>(nullptr));
  _exit(127);
}

bool wait_for_socket(pid_t service) {
  for (int retry = 0; retry != 300; ++retry) {
    struct stat status{};
    if (::lstat(socket_path.c_str(), &status) == 0) {
      if (!S_ISSOCK(status.st_mode) || (status.st_mode & 0777) != 0600
          || status.st_uid != ::geteuid()) return false;
      return true;
    }
    int process_status = 0;
    if (::waitpid(service, &process_status, WNOHANG) == service) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

bool stop_service(pid_t service) {
  if (::kill(service, SIGTERM) != 0) return false;
  int status = 0;
  for (int retry = 0; retry != 200; ++retry) {
    const pid_t result = ::waitpid(service, &status, WNOHANG);
    if (result == service) {
      return WIFEXITED(status) && WEXITSTATUS(status) == 0
          && ::access(socket_path.c_str(), F_OK) != 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  ::kill(service, SIGKILL);
  (void)::waitpid(service, &status, 0);
  return false;
}

int raw_connect() {
  const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (fd < 0) return -1;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socket_path.data(), socket_path.size());
  const socklen_t length = static_cast<socklen_t>(
      offsetof(sockaddr_un, sun_path) + socket_path.size());
  if (::connect(fd, reinterpret_cast<const sockaddr*>(&address), length) != 0) {
    ::close(fd); return -1;
  }
  return fd;
}

ndp_open_v1 raw_open_request() {
  ndp_open_v1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = NDP_ABI_V1;
  request.role = NDP_ROLE_TRAINER;
  request.socket_path_len = static_cast<std::uint32_t>(socket_path.size());
  std::memcpy(request.socket_path, socket_path.data(), socket_path.size());
  std::copy(run_key.begin(), run_key.end(), request.run_key);
  request.fence_epoch = 1;
  const Key worker = key(0x21), incarnation = key(0x22);
  std::copy(worker.begin(), worker.end(), request.worker_key);
  std::copy(incarnation.begin(), incarnation.end(), request.incarnation);
  std::copy(kToken.begin(), kToken.end(), request.admission_token);
  request.deadline_unix_ns = deadline();
  return request;
}

emender_ndp::rpc::Header raw_header(const ndp_open_v1& request) {
  emender_ndp::rpc::Header header{};
  header.opcode = emender_ndp::rpc::Opcode::Open;
  std::copy(request.run_key, request.run_key + 16, header.run.begin());
  header.fence = request.fence_epoch;
  std::copy(request.incarnation, request.incarnation + 16, header.incarnation.begin());
  return header;
}

int send_truncated_rights(int socket, const ndp_open_v1& open) {
  auto header = raw_header(open);
  const auto payload = emender_ndp::rpc::object_payload(open);
  header.payload_bytes = static_cast<std::uint32_t>(payload.size());
  header.fd_count = emender_ndp::rpc::kMaxFds + 1;
  header.payload_digest = emender_ndp::rpc::payload_digest(payload);
  auto frame = emender_ndp::rpc::encode_header(header);
  frame.insert(frame.end(), payload.begin(), payload.end());

  std::array<int, emender_ndp::rpc::kMaxFds + 1> descriptors{};
  for (std::size_t index = 0; index != descriptors.size(); ++index) {
    descriptors[index] = sealed_memfd("rpc-control-truncation", "z", 1);
    if (descriptors[index] < 0) {
      for (std::size_t close_index = 0; close_index != index; ++close_index)
        ::close(descriptors[close_index]);
      return -1;
    }
  }
  std::array<std::uint8_t, CMSG_SPACE(sizeof(int) * descriptors.size())> control{};
  iovec vector{frame.data(), frame.size()};
  msghdr message{};
  message.msg_iov = &vector;
  message.msg_iovlen = 1;
  message.msg_control = control.data();
  message.msg_controllen = control.size();
  cmsghdr* rights = CMSG_FIRSTHDR(&message);
  rights->cmsg_level = SOL_SOCKET;
  rights->cmsg_type = SCM_RIGHTS;
  rights->cmsg_len = CMSG_LEN(sizeof(int) * descriptors.size());
  std::memcpy(CMSG_DATA(rights), descriptors.data(), sizeof(descriptors));
  ssize_t sent;
  do { sent = ::sendmsg(socket, &message, MSG_NOSIGNAL); }
  while (sent < 0 && errno == EINTR);
  for (const int descriptor : descriptors) ::close(descriptor);
  return sent == static_cast<ssize_t>(frame.size()) ? 0 : -1;
}

int test_rpc_envelope_and_authentication() {
  std::array<std::uint8_t, 32> wrong_token{};
  wrong_token.fill(0x6b);
  int status = 0;
  if (open_client(NDP_ROLE_TRAINER, 1, 0x31, 0x32, wrong_token, &status) != 0
      || status != NDP_EFENCE) return 1;

  int socket = raw_connect();
  if (socket < 0) return 2;
  const ndp_open_v1 open = raw_open_request();
  const int first = sealed_memfd("rpc-extra-one", "x", 1);
  const int second = sealed_memfd("rpc-extra-two", "y", 1);
  if (first < 0 || second < 0
      || emender_ndp::rpc::send_packet(socket, raw_header(open),
          emender_ndp::rpc::object_payload(open), {first, second}) != 0) return 3;
  ::close(first); ::close(second);
  emender_ndp::rpc::Packet response;
  if (emender_ndp::rpc::recv_packet(socket, response) != 0
      || response.header.status != NDP_EINVAL) return 4;
  ::close(socket);

  socket = raw_connect();
  if (socket < 0) return 5;
  auto encoded = emender_ndp::rpc::encode_header(raw_header(open));
  encoded[20] = encoded[21] = encoded[22] = encoded[23] = 0xff;
  if (::send(socket, encoded.data(), encoded.size(), MSG_NOSIGNAL)
      != static_cast<ssize_t>(encoded.size())) return 6;
  std::array<std::uint8_t, 16> ignored{};
  const ssize_t got = ::recv(socket, ignored.data(), ignored.size(), 0);
  ::close(socket);
  if (got != 0) return 7;

  socket = raw_connect();
  if (socket < 0 || send_truncated_rights(socket, open) != 0) return 8;
  const ssize_t truncated = ::recv(socket, ignored.data(), ignored.size(), 0);
  ::close(socket);
  return truncated == 0 ? 0 : 9;
}

int controller_install() {
  const ndp_client_t client = open_client(NDP_ROLE_CONTROLLER, 1, 0x41, 0x42);
  if (client == 0) return 1;
  const int descriptor = sealed_memfd("ndp-layout", layout_descriptor.data(),
                                      layout_descriptor.size());
  if (descriptor < 0) return 2;
  ndp_layout_v1 layout{};
  layout.struct_size = sizeof(layout);
  layout.abi_version = NDP_ABI_V1;
  layout.descriptor_fd = descriptor;
  layout.descriptor_bytes = layout_descriptor.size();
  std::copy(layout_digest.begin(), layout_digest.end(), layout.layout_digest);
  const int installed = ndp_layout_install_v1(client, &layout);
  ::close(descriptor);
  if (installed != NDP_OK) return 3;
  ndp_op_t operation = 0;
  if (control(client, NDP_CONTROL_INSTALL_GENERATION, &operation) != NDP_OK) return 4;
  if (ndp_op_release_v1(client, operation) != NDP_OK) return 5;
  return ndp_client_close_v1(client) == NDP_OK ? 0 : 6;
}

ndp_buffer_t register_dense(ndp_client_t client, const float* values,
                            bool seal_write, int* status_out) {
  const int fd = sealed_memfd("trainer-dense", values, sizeof(float) * kElements,
                              seal_write);
  if (fd < 0) { *status_out = NDP_EIO; return 0; }
  ndp_buffer_v1 request{};
  request.struct_size = sizeof(request);
  request.abi_version = NDP_ABI_V1;
  request.kind = NDP_BUFFER_MEMFD;
  request.flags = NDP_BUFFER_READ;
  request.length = sizeof(float) * kElements;
  request.handle_generation = 7;
  request.fd = fd;
  std::copy(layout_digest.begin(), layout_digest.end(), request.layout_digest);
  ndp_buffer_t buffer = 0;
  *status_out = ndp_buffer_register_v1(client, &request, &buffer);
  ::close(fd);
  return buffer;
}

int submit_buffer(ndp_client_t client, ndp_buffer_t buffer, std::uint64_t weight,
                  std::uint8_t trainer, const float* values, ndp_op_t* operation) {
  ndp_submit_v1 submit{};
  submit.struct_size = sizeof(submit);
  submit.abi_version = NDP_ABI_V1;
  submit.buffer = buffer;
  const Key trainer_key = key(trainer), incarnation = key(trainer + 1);
  std::copy(trainer_key.begin(), trainer_key.end(), submit.trainer_key);
  std::copy(incarnation.begin(), incarnation.end(), submit.trainer_incarnation);
  submit.submission_seq = kSubmissionSequence;
  submit.weight = weight;
  submit.element_count = kElements;
  submit.source_dtype = NDP_DTYPE_F32;
  submit.deadline_unix_ns = generation_deadline_ns;
  const Digest digest = emender_ndp::Sha256::digest(values, sizeof(float) * kElements);
  std::copy(digest.begin(), digest.end(), submit.source_buffer_sha256);
  return ndp_submit_local_v1(client, &submit, operation);
}

int trainer_submit_and_reconnect() {
  ndp_client_t client = open_client(NDP_ROLE_TRAINER, 1, 0x51, 0x52);
  if (client == 0) return 1;
  int status = 0;
  ndp_buffer_t buffer = register_dense(client, kValues.data(), false, &status);
  if (buffer != 0 || status != NDP_EINVAL) return 2;
  buffer = register_dense(client, kValues.data(), true, &status);
  if (buffer == 0 || status != NDP_OK) return 3;
  ndp_op_t operation = 0;
  const int first_submit = submit_buffer(
      client, buffer, kWeight, 0x61, kValues.data(), &operation);
  if (first_submit != NDP_OK) {
    std::fprintf(stderr, "first trainer submit status=%d\n", first_submit);
    return 4;
  }
  if (ndp_buffer_release_v1(client, buffer) != NDP_OK
      || ndp_op_release_v1(client, operation) != NDP_OK
      || ndp_client_close_v1(client) != NDP_OK) return 5;

  client = open_client(NDP_ROLE_TRAINER, 1, 0x51, 0x52);
  if (client == 0) return 6;
  buffer = register_dense(client, kValues.data(), true, &status);
  if (buffer == 0 || status != NDP_OK) return 7;
  if (submit_buffer(client, buffer, kWeight, 0x61, kValues.data(), &operation)
      != NDP_OK) return 8;
  if (ndp_op_release_v1(client, operation) != NDP_OK) return 9;
  if (submit_buffer(client, buffer, kWeight + 1, 0x61, kValues.data(), &operation)
      != NDP_ECONFLICT) return 10;
  if (ndp_buffer_release_v1(client, buffer) != NDP_OK) return 11;

  const std::array<float, kElements> nonfinite{{1.0F, 2.0F,
      std::numeric_limits<float>::quiet_NaN(), 4.0F}};
  buffer = register_dense(client, nonfinite.data(), true, &status);
  if (buffer == 0 || status != NDP_OK) return 12;
  if (submit_buffer(client, buffer, 1, 0x71, nonfinite.data(), &operation)
      != NDP_ENONFINITE) return 13;
  if (ndp_buffer_release_v1(client, buffer) != NDP_OK) return 14;
  return ndp_client_close_v1(client) == NDP_OK ? 0 : 15;
}

int controller_finalize(int result_pipe) {
  const ndp_client_t client = open_client(NDP_ROLE_CONTROLLER, 1, 0x41, 0x43);
  if (client == 0) return 1;
  ndp_op_t frozen = 0, result = 0;
  if (control(client, NDP_CONTROL_FREEZE, &frozen) != NDP_OK) return 2;
  if (control(client, NDP_CONTROL_FINALIZE_OWNERS, &result) != NDP_OK) return 3;
  if (::write(result_pipe, &result, sizeof(result)) != sizeof(result)) return 4;
  // Deliberately do not release either operation.  Their authoritative state
  // belongs to the service and must survive this controller process exiting.
  return ndp_client_close_v1(client) == NDP_OK ? 0 : 5;
}

int controller_read_commit_and_fence(ndp_op_t result_operation, int stale_pipe) {
  const ndp_client_t client = open_client(NDP_ROLE_CONTROLLER, 1, 0x41, 0x44);
  if (client == 0) return 1;
  ndp_result_v1 result{};
  result.struct_size = sizeof(result);
  result.abi_version = NDP_ABI_V1;
  ndp_buffer_t result_buffer = 0;
  int result_fd = -1;
  if (ndp_result_view_v1(client, result_operation, &result, &result_buffer,
                         &result_fd) != NDP_OK) return 2;
  if ((::fcntl(result_fd, F_GETFL) & O_ACCMODE) != O_RDONLY
      || (::fcntl(result_fd, F_GETFD) & FD_CLOEXEC) == 0) return 3;
  const int seals = ::fcntl(result_fd, F_GET_SEALS);
  if ((seals & (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL))
      != (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE | F_SEAL_SEAL)) return 4;
  struct stat result_stat{};
  if (::fstat(result_fd, &result_stat) != 0
      || result_stat.st_size != static_cast<off_t>(sizeof(float) * kElements)) return 5;
  const void* mapped = ::mmap(nullptr, sizeof(float) * kElements, PROT_READ,
                              MAP_SHARED, result_fd, 0);
  if (mapped == MAP_FAILED
      || std::memcmp(mapped, kValues.data(), sizeof(float) * kElements) != 0) return 6;
  ::munmap(const_cast<void*>(mapped), sizeof(float) * kElements);
  if (result.fence_epoch != 1 || result.generation != kGeneration
      || result.attempt != kAttempt || result.global_weight != kWeight
      || result.result_bytes != sizeof(float) * kElements
      || std::memcmp(result.run_key, run_key.data(), run_key.size()) != 0
      || std::memcmp(result.layout_digest, layout_digest.data(), layout_digest.size()) != 0
      || std::memcmp(result.base_digest, base_digest.data(), base_digest.size()) != 0)
    return 7;
  const void* writable = ::mmap(nullptr, sizeof(float) * kElements,
                                PROT_READ | PROT_WRITE, MAP_SHARED, result_fd, 0);
  if (writable != MAP_FAILED) { ::munmap(const_cast<void*>(writable),
                                         sizeof(float) * kElements); return 8; }
  ::close(result_fd);

  ndp_op_t commit = 0;
  if (control(client, NDP_CONTROL_COMMIT, &commit) != NDP_OK
      || ndp_op_release_v1(client, commit) != NDP_OK
      || ndp_buffer_release_v1(client, result_buffer) != NDP_OK
      || ndp_op_release_v1(client, result_operation) != NDP_OK) return 9;
  ndp_alloc_v1 allocation{};
  allocation.struct_size = sizeof(allocation);
  allocation.abi_version = NDP_ABI_V1;
  allocation.flags = NDP_BUFFER_READ | NDP_BUFFER_WRITE;
  allocation.bytes = 8;
  allocation.deadline_unix_ns = deadline();
  ndp_buffer_t stale_buffer = 0;
  int stale_fd = -1;
  if (ndp_buffer_allocate_v1(client, &allocation, &stale_buffer, &stale_fd) != NDP_OK)
    return 10;
  ::close(stale_fd);
  if (::write(stale_pipe, &stale_buffer, sizeof(stale_buffer)) != sizeof(stale_buffer))
    return 11;
  const ndp_client_t newer = open_client(NDP_ROLE_CONTROLLER, 2, 0x41, 0x45);
  if (newer == 0 || ndp_buffer_release_v1(client, stale_buffer) != NDP_EFENCE)
    return 12;
  if (ndp_client_close_v1(newer) != NDP_OK || ndp_client_close_v1(client) != NDP_OK)
    return 13;
  return 0;
}

int restart_rejects_stale_handle(ndp_buffer_t stale_buffer) {
  const ndp_client_t client = open_client(NDP_ROLE_CONTROLLER, 3, 0x41, 0x46);
  if (client == 0) return 1;
  if (ndp_buffer_release_v1(client, stale_buffer) != NDP_EINVAL) return 2;
  return ndp_client_close_v1(client) == NDP_OK ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  initialize_metadata();
  generation_deadline_ns = unix_ns() + UINT64_C(120) * 1000 * 1000 * 1000;
  char directory[] = "/tmp/emender-ndp-rpc-v1.XXXXXX";
  if (::mkdtemp(directory) == nullptr) return 3;
  socket_path = std::string(directory) + "/service.sock";

  pid_t service = start_service(argv[1]);
  if (service <= 0 || !wait_for_socket(service)) return 4;
  if (test_rpc_envelope_and_authentication() != 0) return 5;
  if (run_child("controller-install", controller_install) != 0) return 6;
  if (run_child("trainer", trainer_submit_and_reconnect) != 0) return 7;

  int result_pipe[2];
  if (::pipe2(result_pipe, O_CLOEXEC) != 0) return 8;
  if (run_child("controller-finalize", [&] {
        ::close(result_pipe[0]);
        return controller_finalize(result_pipe[1]);
      }) != 0) return 9;
  ::close(result_pipe[1]);
  ndp_op_t result_operation = 0;
  if (::read(result_pipe[0], &result_operation, sizeof(result_operation))
      != sizeof(result_operation)) return 10;
  ::close(result_pipe[0]);

  int stale_pipe[2];
  if (::pipe2(stale_pipe, O_CLOEXEC) != 0) return 11;
  if (run_child("controller-result", [&] {
        ::close(stale_pipe[0]);
        return controller_read_commit_and_fence(result_operation, stale_pipe[1]);
      }) != 0) return 12;
  ::close(stale_pipe[1]);
  ndp_buffer_t stale_buffer = 0;
  if (::read(stale_pipe[0], &stale_buffer, sizeof(stale_buffer)) != sizeof(stale_buffer))
    return 13;
  ::close(stale_pipe[0]);

  if (!stop_service(service)) return 14;
  service = start_service(argv[1]);
  if (service <= 0 || !wait_for_socket(service)) return 15;
  if (run_child("restart-stale", [&] { return restart_rejects_stale_handle(stale_buffer); })
      != 0) return 16;
  if (!stop_service(service)) return 17;
  if (::rmdir(directory) != 0) return 18;
  std::puts("persistent native service RPC v1 cross-process integration passed");
  return 0;
}
