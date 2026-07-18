#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "emender/ndp.h"
#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <limits.h>
#include <linux/memfd.h>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <string>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Native resilient data plane v1 requires a little-endian target"
#endif

#if defined(__clang__)
#pragma STDC FENV_ACCESS ON
#endif

namespace emender_ndp {
namespace {

constexpr std::uint64_t kMaxLayoutBytes = UINT64_C(16) << 30;
constexpr std::uint64_t kMaxDescriptorBytes = UINT64_C(16) << 20;
constexpr std::uint64_t kMaxPayloadBytes = UINT64_C(64) << 20;
constexpr std::size_t kMaxLocalBuffers = 64;
constexpr std::uint64_t kMaxWeight = (UINT64_C(1) << 53) - 1;
constexpr std::uint64_t kMaxTotalWeight = (UINT64_C(1) << 63) - 1;
constexpr long kLustreMagic = 0x0BD00BD0L;
constexpr long kNfsMagic = 0x6969L;

using Digest = std::array<std::uint8_t, 32>;
using Key16 = std::array<std::uint8_t, 16>;

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

bool is_zero(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint8_t value = 0;
    for (std::size_t index = 0; index != size; ++index) value |= data[index];
    return value == 0;
}

template <std::size_t N>
std::array<std::uint8_t, N> bytes(const std::uint8_t (&input)[N]) noexcept {
    std::array<std::uint8_t, N> result{};
    std::copy(input, input + N, result.begin());
    return result;
}

bool digest_equal(const Digest& left, const std::uint8_t* right) noexcept {
    return std::memcmp(left.data(), right, left.size()) == 0;
}

std::uint16_t read_u16(const std::uint8_t*& cursor,
                       const std::uint8_t* end, bool& ok) noexcept {
    if (static_cast<std::size_t>(end - cursor) < 2) { ok = false; return 0; }
    const std::uint16_t result = static_cast<std::uint16_t>(cursor[0])
        | (static_cast<std::uint16_t>(cursor[1]) << 8);
    cursor += 2;
    return result;
}

std::uint32_t read_u32(const std::uint8_t*& cursor,
                       const std::uint8_t* end, bool& ok) noexcept {
    if (static_cast<std::size_t>(end - cursor) < 4) { ok = false; return 0; }
    std::uint32_t result = 0;
    for (unsigned index = 0; index != 4; ++index)
        result |= static_cast<std::uint32_t>(cursor[index]) << (index * 8);
    cursor += 4;
    return result;
}

std::uint64_t read_u64(const std::uint8_t*& cursor,
                       const std::uint8_t* end, bool& ok) noexcept {
    if (static_cast<std::size_t>(end - cursor) < 8) { ok = false; return 0; }
    std::uint64_t result = 0;
    for (unsigned index = 0; index != 8; ++index)
        result |= static_cast<std::uint64_t>(cursor[index]) << (index * 8);
    cursor += 8;
    return result;
}

void append_u32(Sha256& hash, std::uint32_t value) noexcept {
    std::uint8_t encoded[4];
    for (unsigned index = 0; index != 4; ++index)
        encoded[index] = static_cast<std::uint8_t>(value >> (index * 8));
    hash.update(encoded, sizeof(encoded));
}

void append_u64(Sha256& hash, std::uint64_t value) noexcept {
    std::uint8_t encoded[8];
    for (unsigned index = 0; index != 8; ++index)
        encoded[index] = static_cast<std::uint8_t>(value >> (index * 8));
    hash.update(encoded, sizeof(encoded));
}

bool valid_utf8(const std::uint8_t* input, std::size_t length) noexcept {
    std::size_t index = 0;
    while (index < length) {
        const std::uint8_t first = input[index++];
        if (first == 0) return false;
        if (first < 0x80) continue;
        unsigned continuation = 0;
        std::uint32_t point = 0;
        std::uint32_t minimum = 0;
        if ((first & 0xe0U) == 0xc0U) { continuation = 1; point = first & 0x1fU; minimum = 0x80; }
        else if ((first & 0xf0U) == 0xe0U) { continuation = 2; point = first & 0x0fU; minimum = 0x800; }
        else if ((first & 0xf8U) == 0xf0U) { continuation = 3; point = first & 0x07U; minimum = 0x10000; }
        else return false;
        if (index + continuation > length) return false;
        while (continuation-- != 0) {
            const std::uint8_t next = input[index++];
            if ((next & 0xc0U) != 0x80U) return false;
            point = (point << 6) | (next & 0x3fU);
        }
        if (point < minimum || point > 0x10ffffU
            || (point >= 0xd800U && point <= 0xdfffU)) return false;
    }
    return true;
}

int duplicate_fd(int fd) noexcept {
    int result;
    do { result = ::fcntl(fd, F_DUPFD_CLOEXEC, 3); } while (result < 0 && errno == EINTR);
    return result;
}

int create_memfd(const char* name) noexcept {
#if defined(SYS_memfd_create)
    return static_cast<int>(::syscall(SYS_memfd_create, name,
        static_cast<unsigned>(MFD_CLOEXEC | MFD_ALLOW_SEALING)));
#else
    (void)name;
    errno = ENOSYS;
    return -1;
#endif
}

bool write_all(int fd, const void* source, std::size_t bytes_to_write) noexcept {
    const auto* data = static_cast<const std::uint8_t*>(source);
    while (bytes_to_write != 0) {
        const ssize_t wrote = ::write(fd, data, bytes_to_write);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return false;
        data += static_cast<std::size_t>(wrote);
        bytes_to_write -= static_cast<std::size_t>(wrote);
    }
    return true;
}

struct Layout {
    std::vector<std::uint8_t> descriptor;
    Digest digest{};
    std::uint64_t total_elements = 0;
    std::uint64_t layout_bytes = 0;
    std::uint64_t payload_max = 0;
    std::uint32_t shard_count = 0;
    std::uint32_t source_dtype_mask = 0;
};

int parse_layout(int fd, std::uint64_t descriptor_bytes,
                 const std::uint8_t expected_digest[32], Layout& output) {
    if (fd < 0 || descriptor_bytes < 40 || descriptor_bytes > kMaxDescriptorBytes)
        return NDP_EBOUNDS;
    struct stat status{};
    if (::fstat(fd, &status) != 0 || status.st_size < 0
        || static_cast<std::uint64_t>(status.st_size) != descriptor_bytes)
        return NDP_EBOUNDS;
    const int seals = ::fcntl(fd, F_GET_SEALS);
    if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE))
            != (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE))
        return NDP_EINVAL;
    output.descriptor.resize(static_cast<std::size_t>(descriptor_bytes));
    std::size_t consumed = 0;
    while (consumed != output.descriptor.size()) {
        const ssize_t got = ::pread(fd, output.descriptor.data() + consumed,
                                    output.descriptor.size() - consumed,
                                    static_cast<off_t>(consumed));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return NDP_EIO;
        consumed += static_cast<std::size_t>(got);
    }
    Sha256 hash;
    static constexpr char domain[] = "emender-ndp-layout-v1\0";
    hash.update(domain, sizeof(domain) - 1);
    hash.update(output.descriptor.data(), output.descriptor.size());
    output.digest = hash.finish();
    if (!digest_equal(output.digest, expected_digest)) return NDP_ECHECKSUM;

    const auto* cursor = output.descriptor.data();
    const auto* end = cursor + output.descriptor.size();
    static constexpr std::uint8_t magic[8] = {'N','D','P','L','A','Y','1',0};
    if (std::memcmp(cursor, magic, sizeof(magic)) != 0) return NDP_EINVAL;
    cursor += sizeof(magic);
    bool ok = true;
    const std::uint32_t record_count = read_u32(cursor, end, ok);
    output.source_dtype_mask = read_u32(cursor, end, ok);
    output.total_elements = read_u64(cursor, end, ok);
    output.payload_max = read_u64(cursor, end, ok);
    output.shard_count = read_u32(cursor, end, ok);
    const std::uint32_t reserved = read_u32(cursor, end, ok);
    if (!ok || record_count == 0 || reserved != 0 || output.total_elements == 0
        || output.total_elements > kMaxLayoutBytes / 8
        || output.payload_max == 0 || output.payload_max > kMaxPayloadBytes
        || output.payload_max % 8 != 0) return NDP_EBOUNDS;
    output.layout_bytes = output.total_elements * 8;
    const std::uint64_t expected_shards =
        (output.layout_bytes + output.payload_max - 1) / output.payload_max;
    if (expected_shards == 0 || expected_shards > 256
        || output.shard_count != expected_shards) return NDP_EBOUNDS;

    std::string prior_name;
    std::uint64_t expected_offset = 0;
    std::uint32_t observed_mask = 0;
    for (std::uint32_t record = 0; record != record_count; ++record) {
        const std::uint16_t name_length = read_u16(cursor, end, ok);
        if (!ok || name_length == 0 || static_cast<std::size_t>(end - cursor) < name_length
            || !valid_utf8(cursor, name_length)) return NDP_EINVAL;
        std::string name(reinterpret_cast<const char*>(cursor), name_length);
        cursor += name_length;
        if (!prior_name.empty() && !(prior_name < name)) return NDP_EINVAL;
        prior_name = std::move(name);
        const std::uint16_t dtype = read_u16(cursor, end, ok);
        const std::uint16_t ndim = read_u16(cursor, end, ok);
        if (!ok || dtype < NDP_DTYPE_F32 || dtype > NDP_DTYPE_F64 || ndim > 16)
            return NDP_EINVAL;
        observed_mask |= UINT32_C(1) << (dtype - 1);
        std::uint64_t product = 1;
        for (std::uint16_t dim = 0; dim != ndim; ++dim) {
            const std::uint64_t size = read_u64(cursor, end, ok);
            if (!ok || size == 0 || product > std::numeric_limits<std::uint64_t>::max() / size)
                return NDP_EBOUNDS;
            product *= size;
        }
        const std::uint64_t offset = read_u64(cursor, end, ok);
        const std::uint64_t count = read_u64(cursor, end, ok);
        if (!ok || count != product || offset != expected_offset
            || count > output.total_elements - expected_offset) return NDP_EBOUNDS;
        expected_offset += count;
    }
    if (cursor != end || expected_offset != output.total_elements
        || observed_mask != output.source_dtype_mask) return NDP_EINVAL;
    return NDP_OK;
}

struct MetricState {
    ndp_metrics_v1 value{};
    MetricState() {
        value.struct_size = sizeof(value);
        value.abi_version = NDP_ABI_V1;
    }
};

struct Buffer {
    ndp_client_t owner_client = 0;
    std::uint64_t handle_generation = 0;
    std::uint32_t kind = NDP_BUFFER_MEMFD;
    int fd = -1;
    const void* address = nullptr;
    std::uint64_t length = 0;
    std::uint64_t offset = 0;
    bool sealed = false;
    bool accounted = true;
    std::uint32_t public_refs = 0;
    std::uint32_t retained_refs = 0;
    std::shared_ptr<MetricState> metrics;
    ~Buffer() { if (fd >= 0) ::close(fd); }
};

struct Mapping {
    const void* data = nullptr;
    std::size_t bytes = 0;
    bool mapped = false;
    ~Mapping() { if (mapped && data != MAP_FAILED) ::munmap(const_cast<void*>(data), bytes); }
};

int map_readonly(const Buffer& buffer, Mapping& output) noexcept {
    if (buffer.length > std::numeric_limits<std::size_t>::max()) return NDP_EBOUNDS;
    output.bytes = static_cast<std::size_t>(buffer.length);
    if (buffer.kind == NDP_BUFFER_MEMFD) {
        void* mapped = ::mmap(nullptr, output.bytes, PROT_READ, MAP_PRIVATE,
                              buffer.fd, static_cast<off_t>(buffer.offset));
        if (mapped == MAP_FAILED) return NDP_EIO;
        output.data = mapped;
        output.mapped = true;
        return NDP_OK;
    }
    if (buffer.kind == NDP_BUFFER_XPMEM_ADDRESS && buffer.address != nullptr) {
        output.data = buffer.address;
        return NDP_OK;
    }
    return NDP_EPROVIDER;
}

struct SubmissionKey {
    Key16 trainer{};
    Key16 incarnation{};
    std::uint64_t sequence = 0;
    bool operator<(const SubmissionKey& other) const noexcept {
        if (trainer != other.trainer) return trainer < other.trainer;
        if (incarnation != other.incarnation) return incarnation < other.incarnation;
        return sequence < other.sequence;
    }
};

struct Receipt {
    std::uint64_t weight = 0, offset = 0, count = 0;
    std::uint32_t dtype = 0;
    Digest digest{};
};

struct Submission {
    SubmissionKey key;
    Receipt receipt;
    std::shared_ptr<Buffer> buffer;
    ndp_op_t op = 0;
};

enum class OperationKind { Submit, Control, Result };

struct Operation {
    ndp_op_t handle = 0;
    ndp_client_t owner = 0;
    OperationKind kind = OperationKind::Control;
    std::uint64_t generation = 0;
    std::uint32_t attempt = 0;
    std::uint64_t owner_epoch = 0;
    bool valid = true;
    std::shared_ptr<Buffer> result_buffer;
    ndp_result_v1 result{};
};

struct Client {
    ndp_client_t handle = 0;
    std::uint32_t role = 0;
    Key16 run{};
    Key16 worker{};
    Key16 incarnation{};
    std::uint64_t fence = 0;
    int event_fd = -1;
    bool closed = false;
    std::deque<ndp_event_v1> events;
    std::shared_ptr<MetricState> metrics = std::make_shared<MetricState>();
    ~Client() { if (event_fd >= 0) ::close(event_fd); }
};

class Service {
public:
    Service() : cookie_(boot_cookie()) {
        static_assert(std::numeric_limits<double>::is_iec559,
                      "v1 requires IEC 60559 binary64");
        std::fesetround(FE_TONEAREST);
    }

    int client_open(const ndp_open_v1* input, ndp_client_t* output);
    int client_poll_fd(ndp_client_t handle, int* output);
    int client_close(ndp_client_t handle);
    int layout_install(ndp_client_t handle, const ndp_layout_v1* input);
    int buffer_register(ndp_client_t handle, const ndp_buffer_v1* input,
                        ndp_buffer_t* output);
    int buffer_allocate(ndp_client_t handle, const ndp_alloc_v1* input,
                        ndp_buffer_t* output, int* output_fd);
    int buffer_seal(ndp_client_t handle, ndp_buffer_t buffer);
    int buffer_release(ndp_client_t handle, ndp_buffer_t buffer);
    int submit(ndp_client_t handle, const ndp_submit_v1* input, ndp_op_t* output);
    int control(ndp_client_t handle, const struct ndp_control_v1* input, ndp_op_t* output);
    int poll(ndp_client_t handle, ndp_event_v1* events, std::uint32_t capacity,
             std::uint32_t* count, int timeout_ms);
    int result_view(ndp_client_t handle, ndp_op_t op, ndp_result_v1* result,
                    ndp_buffer_t* buffer, int* output_fd);
    int op_release(ndp_client_t handle, ndp_op_t op);
    int metrics(ndp_client_t handle, ndp_metrics_v1* output);

private:
    static std::uint32_t boot_cookie();
    std::uint64_t next_handle() noexcept {
        return (static_cast<std::uint64_t>(cookie_) << 32) | counter_++;
    }
    std::shared_ptr<Client> current_client(ndp_client_t handle) const;
    int require_current(const std::shared_ptr<Client>& client) const noexcept;
    void enqueue(const std::shared_ptr<Client>& client, std::uint32_t event,
                 std::uint32_t status, std::uint32_t reason, ndp_op_t op,
                 std::uint64_t logical_bytes, const Digest* detail = nullptr);
    std::shared_ptr<Operation> make_operation(const std::shared_ptr<Client>& client,
                                              OperationKind kind);
    int create_buffer(const std::shared_ptr<Client>& client, std::uint64_t length,
                      std::shared_ptr<Buffer>& output);
    void account_maybe_release(const std::shared_ptr<Buffer>& buffer);
    void release_submissions();
    void remove_spool() noexcept;
    void abort_generation(bool fence_change);
    void maybe_finish_commit();
    int validate_control(const std::shared_ptr<Client>& client,
                         const struct ndp_control_v1* input) const;
    int reduce_local(const std::shared_ptr<Client>& client);
    int project_result(const std::shared_ptr<Client>& client,
                       const struct ndp_control_v1* input, std::shared_ptr<Operation>& output);
    int materialize_spool(const std::shared_ptr<Client>& client);

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::uint32_t cookie_;
    std::uint32_t counter_ = 1;
    std::unordered_map<ndp_client_t, std::shared_ptr<Client>> clients_;
    std::unordered_map<ndp_buffer_t, std::shared_ptr<Buffer>> buffers_;
    std::unordered_map<ndp_op_t, std::shared_ptr<Operation>> operations_;
    std::size_t accounted_buffers_ = 0;

    bool run_bound_ = false;
    Key16 run_{};
    std::uint64_t fence_ = 0;
    ndp_client_t controller_ = 0;
    std::uint32_t state_ = NDP_STATE_IDLE;
    bool stopped_ = false;
    std::shared_ptr<Layout> layout_;

    std::uint64_t generation_ = 0;
    std::uint32_t attempt_ = 0;
    std::uint64_t owner_epoch_ = 0;
    std::uint64_t generation_deadline_ = 0;
    Digest base_digest_{};
    Digest plan_digest_{};
    std::map<SubmissionKey, Submission> submissions_;
    std::map<SubmissionKey, Receipt> receipts_;
    std::vector<double> numerator_;
    std::uint64_t total_weight_ = 0;
    std::string spool_path_;
    std::shared_ptr<Operation> freeze_operation_;
    std::shared_ptr<Operation> result_operation_;
};

std::uint32_t Service::boot_cookie() {
    std::random_device source;
    std::uint32_t value = source();
    value ^= static_cast<std::uint32_t>(::getpid()) * 0x9e3779b9U;
    value ^= static_cast<std::uint32_t>(unix_ns());
    return value == 0 ? 1U : value;
}

std::shared_ptr<Client> Service::current_client(ndp_client_t handle) const {
    const auto found = clients_.find(handle);
    return found == clients_.end() ? nullptr : found->second;
}

int Service::require_current(const std::shared_ptr<Client>& client) const noexcept {
    if (!client || client->closed) return NDP_EINVAL;
    if (stopped_) return NDP_ESHUTDOWN;
    if (client->run != run_ || client->fence != fence_) return NDP_EFENCE;
    return NDP_OK;
}

void Service::enqueue(const std::shared_ptr<Client>& client, std::uint32_t event,
                      std::uint32_t status, std::uint32_t reason, ndp_op_t op,
                      std::uint64_t logical_bytes, const Digest* detail) {
    if (!client || client->closed) return;
    ndp_event_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = NDP_ABI_V1;
    value.event = event;
    value.status = status;
    value.reason = reason;
    value.state = state_;
    value.op = op;
    value.generation = generation_;
    value.attempt = attempt_;
    value.shard_id = UINT32_MAX;
    value.owner_epoch = owner_epoch_;
    value.logical_bytes = logical_bytes;
    if (detail) std::copy(detail->begin(), detail->end(), value.detail_digest);
    client->events.push_back(value);
    const std::uint64_t one = 1;
    const ssize_t ignored = ::write(client->event_fd, &one, sizeof(one));
    (void)ignored;
    condition_.notify_all();
}

std::shared_ptr<Operation> Service::make_operation(
        const std::shared_ptr<Client>& client, OperationKind kind) {
    auto operation = std::make_shared<Operation>();
    operation->handle = next_handle();
    operation->owner = client->handle;
    operation->kind = kind;
    operation->generation = generation_;
    operation->attempt = attempt_;
    operation->owner_epoch = owner_epoch_;
    operations_[operation->handle] = operation;
    return operation;
}

int Service::client_open(const ndp_open_v1* input, ndp_client_t* output) {
    if (!valid_input(input) || output == nullptr) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    if ((input->role != NDP_ROLE_TRAINER && input->role != NDP_ROLE_CONTROLLER)
        || input->flags != 0 || input->socket_path_len == 0
        || input->socket_path_len > sizeof(input->socket_path)
        || input->fence_epoch == 0 || input->deadline_unix_ns <= unix_ns()
        || is_zero(input->admission_token, sizeof(input->admission_token))) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(mutex_);
    const Key16 requested_run = bytes(input->run_key);
    if (!run_bound_) {
        run_ = requested_run;
        fence_ = input->fence_epoch;
        run_bound_ = true;
        stopped_ = false;
        state_ = NDP_STATE_IDLE;
    } else if (requested_run != run_) {
        if (!clients_.empty()) return NDP_ECONFLICT;
        run_ = requested_run;
        fence_ = input->fence_epoch;
        stopped_ = false;
        state_ = NDP_STATE_IDLE;
        layout_.reset();
    } else if (input->fence_epoch < fence_) {
        return NDP_EFENCE;
    } else if (input->fence_epoch > fence_) {
        abort_generation(true);
        fence_ = input->fence_epoch;
        controller_ = 0;
        stopped_ = false;
        state_ = NDP_STATE_IDLE;
    }
    if (input->role == NDP_ROLE_CONTROLLER && controller_ != 0) {
        const auto existing = current_client(controller_);
        if (existing && !existing->closed && existing->fence == fence_)
            return NDP_ECONFLICT;
    }
    auto client = std::make_shared<Client>();
    client->handle = next_handle();
    client->role = input->role;
    client->run = requested_run;
    client->worker = bytes(input->worker_key);
    client->incarnation = bytes(input->incarnation);
    client->fence = input->fence_epoch;
    client->event_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (client->event_fd < 0) return NDP_EIO;
    clients_[client->handle] = client;
    if (client->role == NDP_ROLE_CONTROLLER) controller_ = client->handle;
    *output = client->handle;
    enqueue(client, NDP_EVENT_STATE, NDP_STATUS_APPLIED, NDP_REASON_NONE, 0, 0);
    return NDP_OK;
}

int Service::client_poll_fd(ndp_client_t handle, int* output) {
    if (output == nullptr) return NDP_EINVAL;
    *output = -1;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    const int duplicate = duplicate_fd(client->event_fd);
    if (duplicate < 0) return NDP_EIO;
    *output = duplicate;
    return NDP_OK;
}

void Service::account_maybe_release(const std::shared_ptr<Buffer>& buffer) {
    if (!buffer || !buffer->accounted || buffer->public_refs != 0
        || buffer->retained_refs != 0) return;
    auto& metrics = buffer->metrics->value;
    metrics.shared_bytes_current -= buffer->length;
    metrics.released_shared_bytes += buffer->length;
    buffer->accounted = false;
    if (accounted_buffers_ != 0) --accounted_buffers_;
}

int Service::client_close(ndp_client_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    if (controller_ == handle && client->fence == fence_) {
        abort_generation(false);
        controller_ = 0;
    }
    std::vector<ndp_buffer_t> owned_buffers;
    for (const auto& item : buffers_)
        if (item.second->metrics == client->metrics) owned_buffers.push_back(item.first);
    for (const auto buffer_handle : owned_buffers) {
        auto buffer = buffers_[buffer_handle];
        buffers_.erase(buffer_handle);
        if (buffer->public_refs != 0) --buffer->public_refs;
        account_maybe_release(buffer);
    }
    std::vector<ndp_op_t> owned_ops;
    for (const auto& item : operations_)
        if (item.second->owner == handle) owned_ops.push_back(item.first);
    for (const auto op_handle : owned_ops) {
        auto op = operations_[op_handle];
        operations_.erase(op_handle);
        op->valid = false;
        if (op->result_buffer && op->result_buffer->retained_refs != 0) {
            --op->result_buffer->retained_refs;
            account_maybe_release(op->result_buffer);
        }
    }
    client->closed = true;
    clients_.erase(handle);
    maybe_finish_commit();
    return NDP_OK;
}

int Service::layout_install(ndp_client_t handle, const ndp_layout_v1* input) {
    if (!valid_input(input)) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    if (input->reserved0 != 0) return NDP_EINVAL;
    Layout parsed;
    const int parsed_result = parse_layout(input->descriptor_fd, input->descriptor_bytes,
                                           input->layout_digest, parsed);
    if (parsed_result != NDP_OK) return parsed_result;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    if (state_ != NDP_STATE_IDLE || client->role != NDP_ROLE_CONTROLLER)
        return NDP_ESTATE;
    layout_ = std::make_shared<Layout>(std::move(parsed));
    return NDP_OK;
}

int Service::create_buffer(const std::shared_ptr<Client>& client, std::uint64_t length,
                           std::shared_ptr<Buffer>& output) {
    if (length == 0 || length > kMaxLayoutBytes || accounted_buffers_ >= kMaxLocalBuffers) {
        ++client->metrics->value.buffer_exhaustions;
        return NDP_EBOUNDS;
    }
    const char* configured = std::getenv("EMENDER_NDP_MAX_SHARED_BYTES");
    std::uint64_t maximum = kMaxLayoutBytes;
    if (configured && *configured) {
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(configured, &end, 10);
        if (errno != 0 || end == configured || *end != '\0' || parsed == 0)
            return NDP_EINVAL;
        maximum = static_cast<std::uint64_t>(parsed);
    }
    auto& metrics = client->metrics->value;
    if (metrics.shared_bytes_current > maximum
        || length > maximum - metrics.shared_bytes_current) {
        ++metrics.buffer_exhaustions;
        return NDP_ENOMEM;
    }
    auto buffer = std::make_shared<Buffer>();
    buffer->owner_client = client->handle;
    buffer->handle_generation = next_handle();
    buffer->length = length;
    buffer->metrics = client->metrics;
    buffer->fd = create_memfd("emender-ndp-v1");
    if (buffer->fd < 0) return NDP_EIO;
    if (::ftruncate(buffer->fd, static_cast<off_t>(length)) != 0) return NDP_EIO;
    if (::fcntl(buffer->fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK) != 0)
        return NDP_EIO;
    ++accounted_buffers_;
    metrics.shared_bytes_current += length;
    metrics.admitted_shared_bytes += length;
    metrics.shared_bytes_high_water = std::max(metrics.shared_bytes_high_water,
                                                metrics.shared_bytes_current);
    output = std::move(buffer);
    return NDP_OK;
}

int Service::buffer_allocate(ndp_client_t handle, const ndp_alloc_v1* input,
                             ndp_buffer_t* output, int* output_fd) {
    if (!valid_input(input) || output == nullptr || output_fd == nullptr) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    *output_fd = -1;
    if (input->flags != (NDP_BUFFER_READ | NDP_BUFFER_WRITE) || input->reserved0 != 0
        || input->deadline_unix_ns <= unix_ns()) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    std::shared_ptr<Buffer> buffer;
    const int created = create_buffer(client, input->bytes, buffer);
    if (created != NDP_OK) return created;
    const int duplicate = duplicate_fd(buffer->fd);
    if (duplicate < 0) {
        buffer->public_refs = 0;
        account_maybe_release(buffer);
        return NDP_EIO;
    }
    const ndp_buffer_t buffer_handle = next_handle();
    buffer->public_refs = 1;
    buffers_[buffer_handle] = buffer;
    *output = buffer_handle;
    *output_fd = duplicate;
    return NDP_OK;
}

int Service::buffer_register(ndp_client_t handle, const ndp_buffer_v1* input,
                             ndp_buffer_t* output) {
    if (!valid_input(input) || output == nullptr) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    if (input->reserved0 != 0 || input->length == 0 || input->length > kMaxLayoutBytes
        || input->flags != NDP_BUFFER_READ) return NDP_EINVAL;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    if (!layout_ || !digest_equal(layout_->digest, input->layout_digest)) return NDP_EINVAL;
    if (accounted_buffers_ >= kMaxLocalBuffers) {
        ++client->metrics->value.buffer_exhaustions;
        return NDP_EBOUNDS;
    }
    auto buffer = std::make_shared<Buffer>();
    buffer->owner_client = client->handle;
    buffer->handle_generation = input->handle_generation;
    buffer->kind = input->kind;
    buffer->length = input->length;
    buffer->offset = input->offset;
    buffer->metrics = client->metrics;
    buffer->sealed = true;
    if (input->kind == NDP_BUFFER_MEMFD) {
        if (input->fd < 0 || input->address_or_segid != 0 || input->offset != 0)
            return NDP_EINVAL;
        const int seals = ::fcntl(input->fd, F_GET_SEALS);
        if (seals < 0 || (seals & (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE))
                != (F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_WRITE)) return NDP_EINVAL;
        buffer->fd = duplicate_fd(input->fd);
        if (buffer->fd < 0) return NDP_EIO;
    } else if (input->kind == NDP_BUFFER_XPMEM_ADDRESS) {
#if defined(NDP_ENABLE_XPMEM) && NDP_ENABLE_XPMEM
        if (input->address_or_segid == 0 || input->fd != -1) return NDP_EINVAL;
        buffer->address = reinterpret_cast<const void*>(input->address_or_segid + input->offset);
#else
        return NDP_EPROVIDER;
#endif
    } else return NDP_EINVAL;
    ++accounted_buffers_;
    auto& metrics = client->metrics->value;
    metrics.shared_bytes_current += buffer->length;
    metrics.admitted_shared_bytes += buffer->length;
    metrics.shared_bytes_high_water = std::max(metrics.shared_bytes_high_water,
                                                metrics.shared_bytes_current);
    const ndp_buffer_t buffer_handle = next_handle();
    buffer->public_refs = 1;
    buffers_[buffer_handle] = buffer;
    *output = buffer_handle;
    return NDP_OK;
}

int Service::buffer_seal(ndp_client_t handle, ndp_buffer_t buffer_handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    const auto found = buffers_.find(buffer_handle);
    if (found == buffers_.end() || found->second->metrics != client->metrics)
        return NDP_EINVAL;
    auto& buffer = *found->second;
    if (buffer.sealed) return NDP_OK;
    if (buffer.kind == NDP_BUFFER_MEMFD) {
        if (::fcntl(buffer.fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_SEAL) != 0)
            return errno == EBUSY ? NDP_ESTATE : NDP_EIO;
    }
    buffer.sealed = true;
    return NDP_OK;
}

int Service::buffer_release(ndp_client_t handle, ndp_buffer_t buffer_handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    const auto found = buffers_.find(buffer_handle);
    if (found == buffers_.end() || found->second->metrics != client->metrics)
        return NDP_EINVAL;
    auto buffer = found->second;
    buffers_.erase(found);
    if (buffer->public_refs == 0) return NDP_EINVAL;
    --buffer->public_refs;
    account_maybe_release(buffer);
    maybe_finish_commit();
    return NDP_OK;
}

int Service::validate_control(const std::shared_ptr<Client>& client,
                              const struct ndp_control_v1* input) const {
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    if (client->role != NDP_ROLE_CONTROLLER || controller_ != client->handle)
        return NDP_ESTATE;
    if (bytes(input->run_key) != run_ || input->fence_epoch != fence_)
        return NDP_EFENCE;
    if (input->deadline_unix_ns <= unix_ns()) return NDP_EDEADLINE;
    if (input->flags != 0 || input->reserved0 != 0) return NDP_EINVAL;
    if (input->metadata_kind == 0) {
        if (input->metadata_fd != -1 || input->metadata_bytes != 0
            || !is_zero(input->metadata_sha256, 32)) return NDP_EINVAL;
    } else if (input->metadata_fd < 0 || input->metadata_bytes == 0) {
        return NDP_EINVAL;
    }
    return NDP_OK;
}

int dtype_size(std::uint32_t dtype, std::uint64_t& output) noexcept {
    switch (dtype) {
        case NDP_DTYPE_F32: output = 4; return NDP_OK;
        case NDP_DTYPE_BF16: output = 2; return NDP_OK;
        case NDP_DTYPE_F64: output = 8; return NDP_OK;
        default: return NDP_EINVAL;
    }
}

double read_source(const std::uint8_t* data, std::uint32_t dtype,
                   std::uint64_t index) noexcept {
    if (dtype == NDP_DTYPE_F32) {
        float value;
        std::memcpy(&value, data + index * 4, sizeof(value));
        return static_cast<double>(value);
    }
    if (dtype == NDP_DTYPE_F64) {
        double value;
        std::memcpy(&value, data + index * 8, sizeof(value));
        return value;
    }
    std::uint16_t raw;
    std::memcpy(&raw, data + index * 2, sizeof(raw));
    const std::uint32_t expanded = static_cast<std::uint32_t>(raw) << 16;
    float value;
    std::memcpy(&value, &expanded, sizeof(value));
    return static_cast<double>(value);
}

int Service::submit(ndp_client_t handle, const ndp_submit_v1* input,
                    ndp_op_t* output) {
    if (!valid_input(input) || output == nullptr) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    if (state_ != NDP_STATE_LOCAL_COLLECT || !layout_) return NDP_ESTATE;
    if (input->flags != 0 || input->deadline_unix_ns <= unix_ns()
        || input->deadline_unix_ns > generation_deadline_ || input->weight == 0
        || input->weight > kMaxWeight || input->element_offset != 0
        || input->element_count != layout_->total_elements) return NDP_EINVAL;
    const auto buffer_found = buffers_.find(input->buffer);
    if (buffer_found == buffers_.end() || buffer_found->second->metrics != client->metrics
        || !buffer_found->second->sealed) return NDP_EINVAL;
    std::uint64_t width = 0;
    if (dtype_size(input->source_dtype, width) != NDP_OK
        || input->element_count > std::numeric_limits<std::uint64_t>::max() / width
        || buffer_found->second->length != input->element_count * width) return NDP_EBOUNDS;

    SubmissionKey key{bytes(input->trainer_key), bytes(input->trainer_incarnation),
                      input->submission_seq};
    Receipt receipt{input->weight, input->element_offset, input->element_count,
                    input->source_dtype, bytes(input->source_buffer_sha256)};
    for (const auto& prior : receipts_) {
        if (prior.first.trainer == key.trainer && prior.first.incarnation != key.incarnation) {
            ++client->metrics->value.stale_rejects;
            return NDP_ESTALE;
        }
        if (prior.first.trainer == key.trainer && prior.first.sequence != key.sequence) {
            ++client->metrics->value.conflict_count;
            return NDP_ECONFLICT;
        }
    }
    const auto seen = receipts_.find(key);
    if (seen != receipts_.end()) {
        const Receipt& old = seen->second;
        if (old.weight != receipt.weight || old.offset != receipt.offset
            || old.count != receipt.count || old.dtype != receipt.dtype
            || old.digest != receipt.digest) {
            ++client->metrics->value.conflict_count;
            return NDP_ECONFLICT;
        }
        auto duplicate = make_operation(client, OperationKind::Submit);
        *output = duplicate->handle;
        ++client->metrics->value.duplicate_count;
        enqueue(client, NDP_EVENT_LOCAL_DUPLICATE, NDP_STATUS_DUPLICATE,
                NDP_REASON_NONE, duplicate->handle, 0, &receipt.digest);
        return NDP_OK;
    }

    Mapping mapping;
    const int mapped = map_readonly(*buffer_found->second, mapping);
    if (mapped != NDP_OK) return mapped;
    auto& metrics = client->metrics->value;
    metrics.mapped_bytes_current += mapping.bytes;
    metrics.mapped_bytes_high_water = std::max(metrics.mapped_bytes_high_water,
                                                metrics.mapped_bytes_current);
    const Digest actual = Sha256::digest(mapping.data, mapping.bytes);
    if (actual != receipt.digest) {
        metrics.mapped_bytes_current -= mapping.bytes;
        ++metrics.checksum_rejects;
        return NDP_ECHECKSUM;
    }
    const auto* raw = static_cast<const std::uint8_t*>(mapping.data);
    for (std::uint64_t index = 0; index != input->element_count; ++index) {
        if (!std::isfinite(read_source(raw, input->source_dtype, index))) {
            metrics.mapped_bytes_current -= mapping.bytes;
            ++metrics.nonfinite_rejects;
            return NDP_ENONFINITE;
        }
    }
    metrics.mapped_bytes_current -= mapping.bytes;

    auto operation = make_operation(client, OperationKind::Submit);
    ++buffer_found->second->retained_refs;
    Submission submission{key, receipt, buffer_found->second, operation->handle};
    receipts_[key] = receipt;
    submissions_[key] = std::move(submission);
    *output = operation->handle;
    enqueue(client, NDP_EVENT_LOCAL_ACCEPTED, NDP_STATUS_APPLIED, NDP_REASON_NONE,
            operation->handle, buffer_found->second->length, &receipt.digest);
    return NDP_OK;
}

void Service::release_submissions() {
    for (auto& item : submissions_) {
        auto buffer = item.second.buffer;
        if (buffer && buffer->retained_refs != 0) {
            --buffer->retained_refs;
            buffer->metrics->value.prompt_source_released_bytes += buffer->length;
            const auto owner = current_client(buffer->owner_client);
            enqueue(owner, NDP_EVENT_BUFFER_RELEASED, NDP_STATUS_APPLIED,
                    NDP_REASON_NONE, item.second.op, buffer->length,
                    &item.second.receipt.digest);
            account_maybe_release(buffer);
        }
        item.second.buffer.reset();
    }
    submissions_.clear();
}

int Service::reduce_local(const std::shared_ptr<Client>& client) {
    if (submissions_.empty() || !layout_) return NDP_ESTATE;
    if (std::fesetround(FE_TONEAREST) != 0) return NDP_EPROVIDER;
    try { numerator_.assign(static_cast<std::size_t>(layout_->total_elements), 0.0); }
    catch (const std::bad_alloc&) { return NDP_ENOMEM; }
    total_weight_ = 0;
    for (const auto& item : submissions_) {
        const Submission& submission = item.second;
        if (submission.receipt.weight > kMaxTotalWeight - total_weight_) {
            release_submissions();
            numerator_.clear();
            return NDP_EBOUNDS;
        }
        Mapping mapping;
        const int mapped = map_readonly(*submission.buffer, mapping);
        if (mapped != NDP_OK) { release_submissions(); numerator_.clear(); return mapped; }
        auto& metrics = submission.buffer->metrics->value;
        metrics.mapped_bytes_current += mapping.bytes;
        metrics.mapped_bytes_high_water = std::max(metrics.mapped_bytes_high_water,
                                                    metrics.mapped_bytes_current);
        const Digest actual = Sha256::digest(mapping.data, mapping.bytes);
        if (actual != submission.receipt.digest) {
            metrics.mapped_bytes_current -= mapping.bytes;
            ++metrics.checksum_rejects;
            release_submissions();
            numerator_.clear();
            return NDP_ECHECKSUM;
        }
        const auto* raw = static_cast<const std::uint8_t*>(mapping.data);
        const double weight = static_cast<double>(submission.receipt.weight);
        for (std::uint64_t index = 0; index != layout_->total_elements; ++index) {
            const double source = read_source(raw, submission.receipt.dtype, index);
            const double term = source * weight;
            const double sum = numerator_[static_cast<std::size_t>(index)] + term;
            if (!std::isfinite(source) || !std::isfinite(term) || !std::isfinite(sum)) {
                metrics.mapped_bytes_current -= mapping.bytes;
                ++metrics.nonfinite_rejects;
                release_submissions();
                numerator_.clear();
                return NDP_ENONFINITE;
            }
            numerator_[static_cast<std::size_t>(index)] = sum;
        }
        metrics.mapped_bytes_current -= mapping.bytes;
        total_weight_ += submission.receipt.weight;
    }
    release_submissions();
    const int spool = materialize_spool(client);
    if (spool != NDP_OK) { numerator_.clear(); return spool; }
    return NDP_OK;
}

bool local_spool_path(const char* input, std::string& output) {
    if (!input || input[0] != '/') return false;
    char resolved[PATH_MAX];
    if (::realpath(input, resolved) == nullptr) return false;
    output = resolved;
    if (output.find("/lustre/") == 0 || output.find("/home/") == 0
        || output.find("/nfs/") == 0 || output.find("/autofs/") == 0) return false;
    struct statfs filesystem{};
    if (::statfs(output.c_str(), &filesystem) != 0
        || filesystem.f_type == kLustreMagic || filesystem.f_type == kNfsMagic) return false;
    return true;
}

int Service::materialize_spool(const std::shared_ptr<Client>& client) {
    const char* configured = std::getenv("EMENDER_NDP_FALLBACK_SPOOL_DIR");
    if (!configured || *configured == '\0') return NDP_OK;
    std::string directory;
    if (!local_spool_path(configured, directory)) return NDP_EBOUNDS;
    char name[256];
    std::snprintf(name, sizeof(name), "%s/ndp-g%llu-a%u-%llu.replay",
                  directory.c_str(), static_cast<unsigned long long>(generation_),
                  attempt_, static_cast<unsigned long long>(next_handle()));
    const int fd = ::open(name, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0600);
    if (fd < 0) return NDP_EIO;
    std::array<std::uint8_t, 64> header{};
    const char magic[] = "NDPJ1";
    std::memcpy(header.data(), magic, sizeof(magic) - 1);
    std::memcpy(header.data() + 8, run_.data(), run_.size());
    std::memcpy(header.data() + 24, &fence_, sizeof(fence_));
    std::memcpy(header.data() + 32, &generation_, sizeof(generation_));
    std::memcpy(header.data() + 40, &attempt_, sizeof(attempt_));
    const std::uint64_t payload_bytes = numerator_.size() * sizeof(double);
    std::memcpy(header.data() + 48, &payload_bytes, sizeof(payload_bytes));
    Sha256 digest;
    digest.update(numerator_.data(), static_cast<std::size_t>(payload_bytes));
    const Digest marker = digest.finish();
    bool success = write_all(fd, header.data(), header.size())
        && write_all(fd, numerator_.data(), static_cast<std::size_t>(payload_bytes))
        && write_all(fd, marker.data(), marker.size()) && ::fsync(fd) == 0;
    const int saved = errno;
    ::close(fd);
    if (!success) { ::unlink(name); errno = saved; return NDP_EIO; }
    const std::uint64_t total = header.size() + payload_bytes + marker.size();
    if (total > layout_->layout_bytes + (UINT64_C(1) << 20)) {
        ::unlink(name);
        return NDP_EBOUNDS;
    }
    spool_path_ = name;
    client->metrics->value.disk_replay_bytes += total;
    ++client->metrics->value.disk_replay_files;
    return NDP_OK;
}

void Service::remove_spool() noexcept {
    if (!spool_path_.empty()) {
        ::unlink(spool_path_.c_str());
        spool_path_.clear();
    }
}

void Service::abort_generation(bool fence_change) {
    if (state_ == NDP_STATE_IDLE || state_ == NDP_STATE_STOPPED) {
        remove_spool();
        return;
    }
    state_ = NDP_STATE_ABORTING;
    release_submissions();
    numerator_.clear();
    numerator_.shrink_to_fit();
    total_weight_ = 0;
    remove_spool();
    if (freeze_operation_) freeze_operation_->valid = false;
    if (result_operation_) result_operation_->valid = false;
    for (auto& item : clients_) {
        if (item.second->fence == fence_) {
            ++item.second->metrics->value.cancelled_ops;
            if (fence_change) ++item.second->metrics->value.stale_rejects;
            enqueue(item.second, NDP_EVENT_ABORTED, NDP_STATUS_REJECTED,
                    fence_change ? NDP_REASON_STALE_FENCE : NDP_REASON_NONE,
                    0, 0);
        }
    }
    freeze_operation_.reset();
    result_operation_.reset();
    receipts_.clear();
    state_ = NDP_STATE_IDLE;
}

int Service::project_result(const std::shared_ptr<Client>& client,
                            const struct ndp_control_v1* input,
                            std::shared_ptr<Operation>& output) {
    if (total_weight_ == 0 || numerator_.size() != layout_->total_elements)
        return NDP_ESTATE;
    const double total = static_cast<double>(total_weight_);
    for (double& value : numerator_) {
        value = value / total;
        if (!std::isfinite(value)) return NDP_ENONFINITE;
    }
    std::shared_ptr<Buffer> result_buffer;
    const std::uint64_t result_bytes = layout_->total_elements * sizeof(float);
    int created = create_buffer(client, result_bytes, result_buffer);
    if (created != NDP_OK) return created;
    void* mapped = ::mmap(nullptr, static_cast<std::size_t>(result_bytes),
                          PROT_READ | PROT_WRITE, MAP_SHARED, result_buffer->fd, 0);
    if (mapped == MAP_FAILED) { account_maybe_release(result_buffer); return NDP_EIO; }
    auto* projected = static_cast<float*>(mapped);
    for (std::size_t index = 0; index != numerator_.size(); ++index)
        projected[index] = static_cast<float>(numerator_[index]);
    ::munmap(mapped, static_cast<std::size_t>(result_bytes));
    if (::fcntl(result_buffer->fd, F_ADD_SEALS,
                F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
        account_maybe_release(result_buffer);
        return NDP_EIO;
    }
    result_buffer->sealed = true;
    result_buffer->retained_refs = 1;

    Sha256 root;
    static constexpr char domain[] = "emender-ndp-result-v1\0";
    root.update(domain, sizeof(domain) - 1);
    root.update(run_.data(), run_.size());
    append_u64(root, fence_);
    append_u64(root, generation_);
    append_u32(root, attempt_);
    append_u64(root, owner_epoch_);
    root.update(layout_->digest.data(), layout_->digest.size());
    root.update(base_digest_.data(), base_digest_.size());
    append_u64(root, total_weight_);
    root.update(numerator_.data(), numerator_.size() * sizeof(double));
    const Digest result_root = root.finish();

    auto operation = make_operation(client, OperationKind::Result);
    operation->result_buffer = result_buffer;
    operation->result.struct_size = sizeof(ndp_result_v1);
    operation->result.abi_version = NDP_ABI_V1;
    operation->result.flags = NDP_BUFFER_READ;
    operation->result.dtype = NDP_DTYPE_F32;
    std::copy(run_.begin(), run_.end(), operation->result.run_key);
    operation->result.fence_epoch = fence_;
    operation->result.generation = generation_;
    operation->result.attempt = attempt_;
    std::copy(layout_->digest.begin(), layout_->digest.end(), operation->result.layout_digest);
    std::copy(base_digest_.begin(), base_digest_.end(), operation->result.base_digest);
    std::copy(result_root.begin(), result_root.end(), operation->result.result_root);
    operation->result.global_weight = total_weight_;
    operation->result.result_bytes = result_bytes;
    client->metrics->value.result_bytes += result_bytes;
    ++client->metrics->value.projection_count;
    output = operation;
    (void)input;
    return NDP_OK;
}

int Service::control(ndp_client_t handle, const struct ndp_control_v1* input,
                     ndp_op_t* output) {
    if (!valid_input(input) || output == nullptr) return input &&
        (input->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    *output = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int validated = validate_control(client, input);
    if (validated != NDP_OK) return validated;
    if (input->command == NDP_CONTROL_BIND_FENCE) {
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_INSTALL_GENERATION) {
        if (state_ != NDP_STATE_IDLE || !layout_ || input->attempt == 0
            || !digest_equal(layout_->digest, input->layout_digest)) return NDP_ESTATE;
        generation_ = input->generation;
        attempt_ = input->attempt;
        owner_epoch_ = input->owner_epoch;
        generation_deadline_ = input->deadline_unix_ns;
        base_digest_ = bytes(input->base_digest);
        plan_digest_ = bytes(input->plan_digest);
        receipts_.clear();
        submissions_.clear();
        numerator_.clear();
        total_weight_ = 0;
        state_ = NDP_STATE_LOCAL_COLLECT;
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        enqueue(client, NDP_EVENT_STATE, NDP_STATUS_APPLIED, NDP_REASON_NONE,
                op->handle, 0);
        return NDP_OK;
    }
    if (input->generation != generation_ || input->attempt != attempt_)
        return NDP_ESTALE;
    if (input->command == NDP_CONTROL_FREEZE) {
        if (state_ == NDP_STATE_FROZEN && freeze_operation_) {
            *output = freeze_operation_->handle;
            return NDP_OK;
        }
        if (state_ != NDP_STATE_LOCAL_COLLECT) return NDP_ESTATE;
        state_ = NDP_STATE_PREPARED;
        const int reduced = reduce_local(client);
        if (reduced != NDP_OK) { abort_generation(false); return reduced; }
        state_ = NDP_STATE_FROZEN;
        freeze_operation_ = make_operation(client, OperationKind::Control);
        *output = freeze_operation_->handle;
        enqueue(client, NDP_EVENT_PREPARED, NDP_STATUS_APPLIED, NDP_REASON_NONE,
                freeze_operation_->handle, layout_->layout_bytes);
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_REASSIGN) {
        if (state_ != NDP_STATE_FROZEN && state_ != NDP_STATE_TRANSFERRING)
            return NDP_ESTATE;
        if (input->owner_epoch <= owner_epoch_ || input->owner_epoch > owner_epoch_ + 2)
            return NDP_ESTALE;
        owner_epoch_ = input->owner_epoch;
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_FINALIZE_OWNERS) {
        if (state_ == NDP_STATE_RESULT_READY && result_operation_) {
            *output = result_operation_->handle;
            return NDP_OK;
        }
        if (state_ != NDP_STATE_FROZEN || input->owner_epoch != owner_epoch_)
            return NDP_ESTATE;
        std::shared_ptr<Operation> result;
        const int projected = project_result(client, input, result);
        if (projected != NDP_OK) { abort_generation(false); return projected; }
        result_operation_ = result;
        state_ = NDP_STATE_RESULT_READY;
        *output = result->handle;
        const Digest result_detail = bytes(result->result.result_root);
        enqueue(client, NDP_EVENT_RESULT_READY, NDP_STATUS_FINALIZED,
                NDP_REASON_NONE, result->handle, result->result.result_bytes,
                &result_detail);
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_COMMIT) {
        if (state_ != NDP_STATE_RESULT_READY) return NDP_ESTATE;
        state_ = NDP_STATE_COMMITTED;
        remove_spool();
        numerator_.clear();
        numerator_.shrink_to_fit();
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        enqueue(client, NDP_EVENT_COMMITTED, NDP_STATUS_FINALIZED, NDP_REASON_NONE,
                op->handle, 0);
        maybe_finish_commit();
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_ABORT) {
        abort_generation(false);
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        return NDP_OK;
    }
    if (input->command == NDP_CONTROL_DRAIN) {
        if (state_ != NDP_STATE_IDLE && state_ != NDP_STATE_COMMITTED)
            abort_generation(false);
        state_ = NDP_STATE_DRAINING;
        auto op = make_operation(client, OperationKind::Control);
        *output = op->handle;
        state_ = NDP_STATE_STOPPED;
        stopped_ = true;
        enqueue(client, NDP_EVENT_DRAINED, NDP_STATUS_FINALIZED,
                NDP_REASON_SHUTDOWN, op->handle, 0);
        return NDP_OK;
    }
    return NDP_EINVAL;
}

void Service::maybe_finish_commit() {
    if (state_ != NDP_STATE_COMMITTED || !result_operation_
        || !result_operation_->result_buffer) return;
    const auto& buffer = result_operation_->result_buffer;
    if (buffer->public_refs == 0 && buffer->retained_refs == 0) {
        result_operation_.reset();
        freeze_operation_.reset();
        receipts_.clear();
        state_ = NDP_STATE_IDLE;
    }
}

int Service::poll(ndp_client_t handle, ndp_event_v1* events,
                  std::uint32_t capacity, std::uint32_t* count, int timeout_ms) {
    if (events == nullptr || count == nullptr || capacity == 0 || timeout_ms < 0)
        return NDP_EINVAL;
    *count = 0;
    std::unique_lock<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    if (client->events.empty() && timeout_ms != 0) {
        condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return client->closed || !client->events.empty();
        });
    }
    if (client->closed) return NDP_ESHUTDOWN;
    while (*count != capacity && !client->events.empty()) {
        events[*count] = client->events.front();
        client->events.pop_front();
        ++*count;
    }
    std::uint64_t ignored;
    while (::read(client->event_fd, &ignored, sizeof(ignored)) > 0) {}
    return NDP_OK;
}

int Service::result_view(ndp_client_t handle, ndp_op_t op_handle,
                         ndp_result_v1* result, ndp_buffer_t* buffer_handle,
                         int* output_fd) {
    if (!valid_input(result) || buffer_handle == nullptr || output_fd == nullptr)
        return result && (result->abi_version >> 16) != (NDP_ABI_V1 >> 16)
            ? NDP_EVERSION : NDP_EINVAL;
    *buffer_handle = 0;
    *output_fd = -1;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    const int current = require_current(client);
    if (current != NDP_OK) return current;
    const auto found = operations_.find(op_handle);
    if (found == operations_.end() || found->second->owner != handle
        || !found->second->valid || found->second->kind != OperationKind::Result
        || !found->second->result_buffer || state_ != NDP_STATE_RESULT_READY)
        return NDP_ESTATE;
    const int duplicate = duplicate_fd(found->second->result_buffer->fd);
    if (duplicate < 0) return NDP_EIO;
    const ndp_buffer_t view_handle = next_handle();
    ++found->second->result_buffer->public_refs;
    buffers_[view_handle] = found->second->result_buffer;
    *result = found->second->result;
    *buffer_handle = view_handle;
    *output_fd = duplicate;
    return NDP_OK;
}

int Service::op_release(ndp_client_t handle, ndp_op_t op_handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    const auto found = operations_.find(op_handle);
    if (found == operations_.end() || found->second->owner != handle)
        return NDP_EINVAL;
    auto op = found->second;
    operations_.erase(found);
    op->valid = false;
    if (op->result_buffer && op->result_buffer->retained_refs != 0) {
        --op->result_buffer->retained_refs;
        account_maybe_release(op->result_buffer);
    }
    maybe_finish_commit();
    return NDP_OK;
}

int Service::metrics(ndp_client_t handle, ndp_metrics_v1* output) {
    if (!valid_input(output)) return output &&
        (output->abi_version >> 16) != (NDP_ABI_V1 >> 16) ? NDP_EVERSION : NDP_EINVAL;
    std::lock_guard<std::mutex> lock(mutex_);
    auto client = current_client(handle);
    if (!client || client->closed) return NDP_EINVAL;
    *output = client->metrics->value;
    return NDP_OK;
}

Service& service() {
    static Service instance;
    return instance;
}

template <typename Callable>
int guarded(Callable&& callable) noexcept {
    try { return callable(); }
    catch (const std::bad_alloc&) { return NDP_ENOMEM; }
    catch (...) { return NDP_EIO; }
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
    return emender_ndp::guarded([&] { return emender_ndp::service().client_open(input, output); });
}
int ndp_client_poll_fd_v1(ndp_client_t client, int* output) {
    return emender_ndp::guarded([&] { return emender_ndp::service().client_poll_fd(client, output); });
}
int ndp_client_close_v1(ndp_client_t client) {
    return emender_ndp::guarded([&] { return emender_ndp::service().client_close(client); });
}
int ndp_layout_install_v1(ndp_client_t client, const ndp_layout_v1* input) {
    return emender_ndp::guarded([&] { return emender_ndp::service().layout_install(client, input); });
}
int ndp_buffer_register_v1(ndp_client_t client, const ndp_buffer_v1* input,
                           ndp_buffer_t* output) {
    return emender_ndp::guarded([&] { return emender_ndp::service().buffer_register(client, input, output); });
}
int ndp_buffer_allocate_v1(ndp_client_t client, const ndp_alloc_v1* input,
                           ndp_buffer_t* output, int* output_fd) {
    return emender_ndp::guarded([&] { return emender_ndp::service().buffer_allocate(client, input, output, output_fd); });
}
int ndp_buffer_seal_v1(ndp_client_t client, ndp_buffer_t buffer) {
    return emender_ndp::guarded([&] { return emender_ndp::service().buffer_seal(client, buffer); });
}
int ndp_buffer_release_v1(ndp_client_t client, ndp_buffer_t buffer) {
    return emender_ndp::guarded([&] { return emender_ndp::service().buffer_release(client, buffer); });
}
int ndp_submit_local_v1(ndp_client_t client, const ndp_submit_v1* input,
                        ndp_op_t* output) {
    return emender_ndp::guarded([&] { return emender_ndp::service().submit(client, input, output); });
}
int ndp_control_v1(ndp_client_t client, const struct ndp_control_v1* input,
                   ndp_op_t* output) {
    return emender_ndp::guarded([&] { return emender_ndp::service().control(client, input, output); });
}
int ndp_poll_v1(ndp_client_t client, ndp_event_v1* events,
                uint32_t capacity, uint32_t* count, int timeout_ms) {
    return emender_ndp::guarded([&] { return emender_ndp::service().poll(client, events, capacity, count, timeout_ms); });
}
int ndp_result_view_v1(ndp_client_t client, ndp_op_t op,
                       ndp_result_v1* result, ndp_buffer_t* buffer, int* output_fd) {
    return emender_ndp::guarded([&] { return emender_ndp::service().result_view(client, op, result, buffer, output_fd); });
}
int ndp_op_release_v1(ndp_client_t client, ndp_op_t op) {
    return emender_ndp::guarded([&] { return emender_ndp::service().op_release(client, op); });
}
int ndp_client_metrics_v1(ndp_client_t client, ndp_metrics_v1* output) {
    return emender_ndp::guarded([&] { return emender_ndp::service().metrics(client, output); });
}

}  // extern "C"
