#ifndef EMENDER_NDP_SERVICE_CORE_HPP
#define EMENDER_NDP_SERVICE_CORE_HPP

#include "emender/ndp.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace emender_ndp {

struct ServiceSnapshot {
    std::array<std::uint8_t, 16> run{};
    std::array<std::uint8_t, 16> incarnation{};
    std::array<std::uint8_t, 32> layout_digest{};
    std::uint64_t fence = 0;
    std::uint64_t generation = 0;
    std::uint64_t owner_epoch = 0;
    std::uint32_t attempt = 0;
    std::uint32_t state = NDP_STATE_STARTING;
};

/*
 * Authoritative node-local state.  This object is deliberately linkable only
 * into the persistent service; libemender_ndp is an AF_UNIX RPC client and
 * never constructs one.
 */
class LocalServiceCore {
public:
    LocalServiceCore();
    ~LocalServiceCore();
    LocalServiceCore(const LocalServiceCore&) = delete;
    LocalServiceCore& operator=(const LocalServiceCore&) = delete;

    int client_open(const ndp_open_v1*, ndp_client_t*);
    int client_poll_fd(ndp_client_t, int*);
    int client_close(ndp_client_t);
    int layout_install(ndp_client_t, const ndp_layout_v1*);
    int buffer_register(ndp_client_t, const ndp_buffer_v1*, ndp_buffer_t*);
    int buffer_allocate(ndp_client_t, const ndp_alloc_v1*, ndp_buffer_t*, int*);
    int buffer_seal(ndp_client_t, ndp_buffer_t);
    int buffer_release(ndp_client_t, ndp_buffer_t);
    int submit(ndp_client_t, const ndp_submit_v1*, ndp_op_t*);
    int control(ndp_client_t, const struct ndp_control_v1*, ndp_op_t*);
    int poll(ndp_client_t, ndp_event_v1*, std::uint32_t, std::uint32_t*, int);
    int result_view(ndp_client_t, ndp_op_t, ndp_result_v1*, ndp_buffer_t*, int*);
    int op_release(ndp_client_t, ndp_op_t);
    int metrics(ndp_client_t, ndp_metrics_v1*);
    int coordination_step(ndp_client_t, const ndp_coord_event_v1*,
                          ndp_coord_result_v1*, std::string*);
    ServiceSnapshot snapshot(ndp_client_t) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace emender_ndp

#endif
