#ifndef EMENDER_NDP_RPC_PROTOCOL_HPP
#define EMENDER_NDP_RPC_PROTOCOL_HPP

#include "sha256.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace emender_ndp::rpc {

constexpr std::uint16_t kMajor = 1;
constexpr std::uint16_t kMinor = 0;
constexpr std::uint16_t kResponse = 1;
constexpr std::size_t kHeaderBytes = 188;
constexpr std::size_t kMaxPacketBytes = 64 * 1024;
constexpr std::size_t kMaxPayloadBytes = kMaxPacketBytes - kHeaderBytes;
constexpr std::size_t kMaxFds = 4;

enum class Opcode : std::uint16_t {
    Open = 1,
    Close = 2,
    PollFd = 3,
    LayoutInstall = 4,
    BufferRegister = 5,
    BufferAllocate = 6,
    BufferSeal = 7,
    BufferRelease = 8,
    Submit = 9,
    Control = 10,
    Poll = 11,
    ResultView = 12,
    OpRelease = 13,
    Metrics = 14,
};

struct Header {
    Opcode opcode = Opcode::Open;
    std::uint16_t flags = 0;
    std::uint32_t payload_bytes = 0;
    std::uint16_t fd_count = 0;
    std::uint64_t request_id = 0;
    std::uint64_t client = 0;
    std::array<std::uint8_t, 16> run{};
    std::uint64_t fence = 0;
    std::uint64_t generation = 0;
    std::uint32_t attempt = 0;
    std::array<std::uint8_t, 16> incarnation{};
    std::uint64_t sequence = 0;
    std::uint64_t extent = 0;
    std::array<std::uint8_t, 32> layout_digest{};
    std::array<std::uint8_t, 32> payload_digest{};
    std::int32_t status = 0;
};

struct Packet {
    Header header{};
    std::vector<std::uint8_t> payload;
    std::vector<int> fds;

    Packet() = default;
    Packet(const Packet&) = delete;
    Packet& operator=(const Packet&) = delete;
    Packet(Packet&& other) noexcept
        : header(other.header), payload(std::move(other.payload)),
          fds(std::move(other.fds)) {
        other.fds.clear();
    }
    Packet& operator=(Packet&& other) noexcept {
        if (this != &other) {
            close_fds();
            header = other.header;
            payload = std::move(other.payload);
            fds = std::move(other.fds);
            other.fds.clear();
        }
        return *this;
    }
    ~Packet() { close_fds(); }
    void close_fds() noexcept {
        for (const int fd : fds) if (fd >= 0) ::close(fd);
        fds.clear();
    }
    int release_fd(std::size_t index = 0) noexcept {
        if (index >= fds.size()) return -1;
        const int result = fds[index];
        fds[index] = -1;
        return result;
    }
};

inline void append_u16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value));
    out.push_back(static_cast<std::uint8_t>(value >> 8));
}
inline void append_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    for (unsigned shift = 0; shift != 32; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}
inline void append_u64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (unsigned shift = 0; shift != 64; shift += 8)
        out.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline std::uint16_t take_u16(const std::uint8_t*& cursor) {
    const std::uint16_t value = static_cast<std::uint16_t>(cursor[0])
        | (static_cast<std::uint16_t>(cursor[1]) << 8);
    cursor += 2;
    return value;
}
inline std::uint32_t take_u32(const std::uint8_t*& cursor) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift != 32; shift += 8)
        value |= static_cast<std::uint32_t>(*cursor++) << shift;
    return value;
}
inline std::uint64_t take_u64(const std::uint8_t*& cursor) {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift != 64; shift += 8)
        value |= static_cast<std::uint64_t>(*cursor++) << shift;
    return value;
}

inline std::array<std::uint8_t, 32> payload_digest(
        const std::vector<std::uint8_t>& payload) {
    return Sha256::digest(payload.data(), payload.size());
}

inline std::vector<std::uint8_t> encode_header(const Header& header) {
    std::vector<std::uint8_t> out;
    out.reserve(kHeaderBytes);
    static constexpr std::uint8_t magic[8] = {'N','D','P','R','P','C','1',0};
    out.insert(out.end(), magic, magic + sizeof(magic));
    append_u16(out, kMajor);
    append_u16(out, kMinor);
    append_u16(out, static_cast<std::uint16_t>(header.opcode));
    append_u16(out, header.flags);
    append_u32(out, static_cast<std::uint32_t>(kHeaderBytes));
    append_u32(out, header.payload_bytes);
    append_u16(out, header.fd_count);
    append_u16(out, 0);
    append_u64(out, header.request_id);
    append_u64(out, header.client);
    out.insert(out.end(), header.run.begin(), header.run.end());
    append_u64(out, header.fence);
    append_u64(out, header.generation);
    append_u32(out, header.attempt);
    append_u32(out, 0);
    out.insert(out.end(), header.incarnation.begin(), header.incarnation.end());
    append_u64(out, header.sequence);
    append_u64(out, header.extent);
    out.insert(out.end(), header.layout_digest.begin(), header.layout_digest.end());
    out.insert(out.end(), header.payload_digest.begin(), header.payload_digest.end());
    append_u32(out, static_cast<std::uint32_t>(header.status));
    append_u32(out, 0);
    return out;
}

inline bool decode_header(const std::uint8_t* bytes, std::size_t size,
                          Header& output) noexcept {
    if (size < kHeaderBytes) return false;
    static constexpr std::uint8_t magic[8] = {'N','D','P','R','P','C','1',0};
    if (std::memcmp(bytes, magic, sizeof(magic)) != 0) return false;
    const std::uint8_t* cursor = bytes + sizeof(magic);
    if (take_u16(cursor) != kMajor || take_u16(cursor) > kMinor) return false;
    const std::uint16_t opcode = take_u16(cursor);
    if (opcode < static_cast<std::uint16_t>(Opcode::Open)
        || opcode > static_cast<std::uint16_t>(Opcode::Metrics)) return false;
    output.opcode = static_cast<Opcode>(opcode);
    output.flags = take_u16(cursor);
    if ((output.flags & ~kResponse) != 0) return false;
    if (take_u32(cursor) != kHeaderBytes) return false;
    output.payload_bytes = take_u32(cursor);
    output.fd_count = take_u16(cursor);
    if (take_u16(cursor) != 0 || output.payload_bytes > kMaxPayloadBytes
        || output.fd_count > kMaxFds) return false;
    output.request_id = take_u64(cursor);
    output.client = take_u64(cursor);
    std::memcpy(output.run.data(), cursor, output.run.size()); cursor += output.run.size();
    output.fence = take_u64(cursor);
    output.generation = take_u64(cursor);
    output.attempt = take_u32(cursor);
    if (take_u32(cursor) != 0) return false;
    std::memcpy(output.incarnation.data(), cursor, output.incarnation.size());
    cursor += output.incarnation.size();
    output.sequence = take_u64(cursor);
    output.extent = take_u64(cursor);
    std::memcpy(output.layout_digest.data(), cursor, output.layout_digest.size());
    cursor += output.layout_digest.size();
    std::memcpy(output.payload_digest.data(), cursor, output.payload_digest.size());
    cursor += output.payload_digest.size();
    output.status = static_cast<std::int32_t>(take_u32(cursor));
    if (take_u32(cursor) != 0) return false;
    return static_cast<std::size_t>(cursor - bytes) == kHeaderBytes;
}

inline int send_packet(int socket_fd, Header header,
                       const std::vector<std::uint8_t>& payload,
                       const std::vector<int>& fds = {}) noexcept {
    if (payload.size() > kMaxPayloadBytes || fds.size() > kMaxFds) return -1;
    header.payload_bytes = static_cast<std::uint32_t>(payload.size());
    header.fd_count = static_cast<std::uint16_t>(fds.size());
    header.payload_digest = payload_digest(payload);
    std::vector<std::uint8_t> encoded = encode_header(header);
    iovec vectors[2] = {
        {encoded.data(), encoded.size()},
        {const_cast<std::uint8_t*>(payload.data()), payload.size()},
    };
    std::array<std::uint8_t, CMSG_SPACE(sizeof(int) * kMaxFds)> control{};
    msghdr message{};
    message.msg_iov = vectors;
    message.msg_iovlen = payload.empty() ? 1 : 2;
    if (!fds.empty()) {
        message.msg_control = control.data();
        message.msg_controllen = CMSG_SPACE(sizeof(int) * fds.size());
        cmsghdr* rights = CMSG_FIRSTHDR(&message);
        rights->cmsg_level = SOL_SOCKET;
        rights->cmsg_type = SCM_RIGHTS;
        rights->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
        std::memcpy(CMSG_DATA(rights), fds.data(), sizeof(int) * fds.size());
    }
    ssize_t sent;
    do { sent = ::sendmsg(socket_fd, &message, MSG_NOSIGNAL); }
    while (sent < 0 && errno == EINTR);
    const std::size_t expected = encoded.size() + payload.size();
    return sent >= 0 && static_cast<std::size_t>(sent) == expected ? 0 : -1;
}

inline void close_control_rights(msghdr& message) noexcept {
    for (cmsghdr* item = CMSG_FIRSTHDR(&message); item != nullptr;
         item = CMSG_NXTHDR(&message, item)) {
        if (item->cmsg_level != SOL_SOCKET || item->cmsg_type != SCM_RIGHTS
            || item->cmsg_len < CMSG_LEN(0)) continue;
        const std::size_t descriptor_bytes = item->cmsg_len - CMSG_LEN(0);
        if (descriptor_bytes % sizeof(int) != 0) continue;
        const std::size_t count = descriptor_bytes / sizeof(int);
        const auto* descriptors = reinterpret_cast<const int*>(CMSG_DATA(item));
        for (std::size_t index = 0; index != count; ++index)
            if (descriptors[index] >= 0) ::close(descriptors[index]);
    }
}

inline int recv_packet(int socket_fd, Packet& output) noexcept {
    std::vector<std::uint8_t> bytes(kMaxPacketBytes);
    std::array<std::uint8_t, CMSG_SPACE(sizeof(int) * kMaxFds)> control{};
    iovec vector{bytes.data(), bytes.size()};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    ssize_t received;
    do { received = ::recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC); }
    while (received < 0 && errno == EINTR);
    if (received < 0) return -1;
    if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
        close_control_rights(message);
        return -1;
    }
    std::vector<int> received_fds;
    for (cmsghdr* item = CMSG_FIRSTHDR(&message); item != nullptr;
         item = CMSG_NXTHDR(&message, item)) {
        if (item->cmsg_level != SOL_SOCKET || item->cmsg_type != SCM_RIGHTS
            || item->cmsg_len < CMSG_LEN(0)) {
            for (const int fd : received_fds) ::close(fd);
            return -1;
        }
        const std::size_t descriptor_bytes = item->cmsg_len - CMSG_LEN(0);
        if (descriptor_bytes % sizeof(int) != 0) {
            for (const int fd : received_fds) ::close(fd);
            return -1;
        }
        const std::size_t count = descriptor_bytes / sizeof(int);
        const auto* descriptors = reinterpret_cast<const int*>(CMSG_DATA(item));
        for (std::size_t index = 0; index != count; ++index) {
            if (received_fds.size() == kMaxFds) {
                for (const int fd : received_fds) ::close(fd);
                for (; index != count; ++index) ::close(descriptors[index]);
                return -1;
            }
            received_fds.push_back(descriptors[index]);
        }
    }
    if (received == 0) {
        for (const int fd : received_fds) ::close(fd);
        return 1;
    }
    Header header{};
    if (!decode_header(bytes.data(), static_cast<std::size_t>(received), header)
        || static_cast<std::size_t>(received) != kHeaderBytes + header.payload_bytes) {
        for (const int fd : received_fds) ::close(fd);
        return -1;
    }
    if (received_fds.size() != header.fd_count) {
        for (const int fd : received_fds) ::close(fd);
        return -1;
    }
    output.close_fds();
    output.header = header;
    output.payload.assign(bytes.begin() + static_cast<std::ptrdiff_t>(kHeaderBytes),
                          bytes.begin() + received);
    if (payload_digest(output.payload) != output.header.payload_digest) {
        for (const int fd : received_fds) ::close(fd);
        output.payload.clear();
        return -1;
    }
    output.fds = std::move(received_fds);
    return 0;
}

template <typename T>
inline std::vector<std::uint8_t> object_payload(const T& value) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    return std::vector<std::uint8_t>(begin, begin + sizeof(T));
}

template <typename T>
inline bool payload_object(const Packet& packet, T& output) noexcept {
    if (packet.payload.size() != sizeof(T)) return false;
    std::memcpy(&output, packet.payload.data(), sizeof(T));
    return true;
}

}  // namespace emender_ndp::rpc

#endif
