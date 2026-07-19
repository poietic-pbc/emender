#include "service_rpc_protocol.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace emender_ndp::rpc {

Packet::Packet(Packet&& other) noexcept
    : header(other.header), body(other.body), fds(std::move(other.fds)) {
    other.fds.clear();
}

Packet& Packet::operator=(Packet&& other) noexcept {
    if (this != &other) {
        close_fds();
        header = other.header;
        body = other.body;
        fds = std::move(other.fds);
        other.fds.clear();
    }
    return *this;
}

Packet::~Packet() { close_fds(); }

void Packet::close_fds() noexcept {
    for (const int fd : fds) {
        if (fd >= 0) ::close(fd);
    }
    fds.clear();
}

int Packet::take_fd() noexcept {
    if (fds.size() != 1) return -1;
    const int result = fds.front();
    fds.clear();
    return result;
}

int send_packet(int socket_fd, Opcode opcode, std::uint64_t sequence,
                std::int32_t status, std::uint16_t flags,
                const void* body, std::size_t body_bytes, int send_fd) noexcept {
    if (socket_fd < 0 || sequence == 0 || body_bytes > kMaxBodyBytes
        || (body_bytes != 0 && body == nullptr)) return NDP_EINVAL;
    Header header{};
    header.opcode = static_cast<std::uint16_t>(opcode);
    header.body_bytes = static_cast<std::uint32_t>(body_bytes);
    header.fd_count = send_fd >= 0 ? 1 : 0;
    header.flags = flags;
    header.sequence = sequence;
    header.status = status;
    iovec vectors[2]{};
    vectors[0].iov_base = &header;
    vectors[0].iov_len = sizeof(header);
    vectors[1].iov_base = const_cast<void*>(body);
    vectors[1].iov_len = body_bytes;
    std::array<std::uint8_t, CMSG_SPACE(sizeof(int))> control{};
    msghdr message{};
    message.msg_iov = vectors;
    message.msg_iovlen = body_bytes == 0 ? 1 : 2;
    if (send_fd >= 0) {
        message.msg_control = control.data();
        message.msg_controllen = control.size();
        cmsghdr* ancillary = CMSG_FIRSTHDR(&message);
        ancillary->cmsg_level = SOL_SOCKET;
        ancillary->cmsg_type = SCM_RIGHTS;
        ancillary->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(ancillary), &send_fd, sizeof(send_fd));
    }
    const std::size_t total = sizeof(header) + body_bytes;
    ssize_t sent;
    do { sent = ::sendmsg(socket_fd, &message, MSG_NOSIGNAL); }
    while (sent < 0 && errno == EINTR);
    if (sent < 0) return errno == EPIPE || errno == ECONNRESET ? NDP_ESHUTDOWN : NDP_EIO;
    return static_cast<std::size_t>(sent) == total ? NDP_OK : NDP_EIO;
}

int receive_packet(int socket_fd, Packet* output, bool nonblocking) noexcept {
    if (socket_fd < 0 || output == nullptr) return NDP_EINVAL;
    output->close_fds();
    std::array<std::uint8_t, kMaxFrameBytes> frame{};
    // SO_PASSCRED is enabled on the listener. Leave independent space for the
    // kernel's SCM_CREDENTIALS record and the maximum descriptor record so a
    // valid one-fd request is never mistaken for ancillary truncation.
    std::array<std::uint8_t,
        CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(kMaxAncillaryFds * sizeof(int))>
        control{};
    iovec vector{frame.data(), frame.size()};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    int flags = MSG_CMSG_CLOEXEC;
    if (nonblocking) flags |= MSG_DONTWAIT;
    ssize_t received;
    do { received = ::recvmsg(socket_fd, &message, flags); }
    while (received < 0 && errno == EINTR);
    if (received == 0) return NDP_ESHUTDOWN;
    if (received < 0) {
        if (nonblocking && (errno == EAGAIN || errno == EWOULDBLOCK))
            return NDP_IN_PROGRESS;
        return errno == ECONNRESET ? NDP_ESHUTDOWN : NDP_EIO;
    }
    for (cmsghdr* ancillary = CMSG_FIRSTHDR(&message); ancillary != nullptr;
         ancillary = CMSG_NXTHDR(&message, ancillary)) {
        if (ancillary->cmsg_level != SOL_SOCKET || ancillary->cmsg_type != SCM_RIGHTS
            || ancillary->cmsg_len < CMSG_LEN(0)) continue;
        const std::size_t data_bytes = ancillary->cmsg_len - CMSG_LEN(0);
        if (data_bytes % sizeof(int) != 0) {
            output->close_fds();
            return NDP_EINVAL;
        }
        const auto* values = reinterpret_cast<const int*>(CMSG_DATA(ancillary));
        for (std::size_t index = 0; index != data_bytes / sizeof(int); ++index) {
            if (output->fds.size() == kMaxAncillaryFds) {
                ::close(values[index]);
            } else {
                const int descriptor = values[index];
                const int current = ::fcntl(descriptor, F_GETFD);
                if (current >= 0) (void)::fcntl(descriptor, F_SETFD, current | FD_CLOEXEC);
                output->fds.push_back(descriptor);
            }
        }
    }
    if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0
        || static_cast<std::size_t>(received) < sizeof(Header)) {
        output->close_fds();
        return NDP_EBOUNDS;
    }
    std::memcpy(&output->header, frame.data(), sizeof(Header));
    const Header& header = output->header;
    const std::size_t expected = sizeof(Header) + header.body_bytes;
    if (header.magic != kMagic || header.version != kVersion || header.reserved != 0
        || header.body_bytes > kMaxBodyBytes
        || expected != static_cast<std::size_t>(received)
        || header.fd_count != output->fds.size()) {
        output->close_fds();
        return NDP_EINVAL;
    }
    if (header.body_bytes != 0) {
        std::memcpy(output->body.data(), frame.data() + sizeof(Header),
                    header.body_bytes);
    }
    return NDP_OK;
}

}  // namespace emender_ndp::rpc
