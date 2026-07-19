#ifndef EMENDER_NDP_SERVICE_RPC_PROTOCOL_HPP
#define EMENDER_NDP_SERVICE_RPC_PROTOCOL_HPP

#include "emender/ndp.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace emender_ndp::rpc {

constexpr std::uint32_t kMagic = UINT32_C(0x3150444e);  // "NDP1" little endian.
constexpr std::uint16_t kVersion = 1;
constexpr std::size_t kMaxFrameBytes = 4096;
constexpr std::size_t kMaxAncillaryFds = 4;
constexpr std::uint16_t kResponse = 1;

enum class Opcode : std::uint16_t {
    open = 1,
    poll_fd = 2,
    close = 3,
    layout_install = 4,
    buffer_register = 5,
    buffer_allocate = 6,
    buffer_seal = 7,
    buffer_release = 8,
    submit = 9,
    control = 10,
    poll = 11,
    result_view = 12,
    op_release = 13,
    metrics = 14,
};

struct Header {
    std::uint32_t magic = kMagic;
    std::uint16_t version = kVersion;
    std::uint16_t opcode = 0;
    std::uint32_t body_bytes = 0;
    std::uint16_t fd_count = 0;
    std::uint16_t flags = 0;
    std::uint64_t sequence = 0;
    std::int32_t status = 0;
    std::uint32_t reserved = 0;
};
static_assert(sizeof(Header) == 32, "native service RPC header must be stable");
constexpr std::size_t kMaxBodyBytes = kMaxFrameBytes - sizeof(Header);

struct Packet {
    Header header{};
    std::array<std::uint8_t, kMaxBodyBytes> body{};
    std::vector<int> fds;

    Packet() = default;
    Packet(const Packet&) = delete;
    Packet& operator=(const Packet&) = delete;
    Packet(Packet&& other) noexcept;
    Packet& operator=(Packet&& other) noexcept;
    ~Packet();

    void close_fds() noexcept;
    int take_fd() noexcept;
};

struct PollRequest {
    std::uint32_t capacity;
    std::int32_t timeout_ms;
};
static_assert(sizeof(PollRequest) == 8, "RPC poll request size");

struct PollResponsePrefix {
    std::uint32_t count;
    std::uint32_t reserved;
};
static_assert(sizeof(PollResponsePrefix) == 8, "RPC poll response prefix size");

struct ResultViewRequest {
    std::uint64_t operation;
    ndp_result_v1 result_prefix;
};

struct ResultViewResponse {
    ndp_result_v1 result;
    std::uint64_t buffer;
};

int send_packet(int socket_fd, Opcode opcode, std::uint64_t sequence,
                std::int32_t status, std::uint16_t flags,
                const void* body, std::size_t body_bytes, int send_fd = -1) noexcept;

// NDP_IN_PROGRESS means the nonblocking socket has no packet ready. NDP_ESHUTDOWN
// means an orderly peer close. Every other nonnegative result produced here is
// NDP_OK; malformed/truncated packets fail closed with a stable negative code.
int receive_packet(int socket_fd, Packet* output, bool nonblocking) noexcept;

}  // namespace emender_ndp::rpc

#endif
