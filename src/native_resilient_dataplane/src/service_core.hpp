#ifndef EMENDER_NDP_SERVICE_CORE_HPP
#define EMENDER_NDP_SERVICE_CORE_HPP

#include "emender/ndp.h"

#include <cstdint>
#include <memory>

namespace emender_ndp {

// Authoritative node-local state machine.  This class is linked only into the
// persistent service executable; libemender_ndp is an AF_UNIX RPC client and
// therefore cannot instantiate a second process-local authority.
class ServiceCore final {
 public:
    ServiceCore();
    ~ServiceCore();
    ServiceCore(const ServiceCore&) = delete;
    ServiceCore& operator=(const ServiceCore&) = delete;

    int client_open(const ndp_open_v1* input, ndp_client_t* output);
    int client_poll_fd(ndp_client_t client, int* output);
    int client_close(ndp_client_t client);
    int layout_install(ndp_client_t client, const ndp_layout_v1* input);
    int buffer_register(ndp_client_t client, const ndp_buffer_v1* input,
                        ndp_buffer_t* output);
    int buffer_allocate(ndp_client_t client, const ndp_alloc_v1* input,
                        ndp_buffer_t* output, int* output_fd);
    int buffer_seal(ndp_client_t client, ndp_buffer_t buffer);
    int buffer_release(ndp_client_t client, ndp_buffer_t buffer);
    int submit(ndp_client_t client, const ndp_submit_v1* input, ndp_op_t* output);
    int control(ndp_client_t client, const struct ndp_control_v1* input,
                ndp_op_t* output);
    int poll(ndp_client_t client, ndp_event_v1* events, std::uint32_t capacity,
             std::uint32_t* count, int timeout_ms);
    int result_view(ndp_client_t client, ndp_op_t op, ndp_result_v1* result,
                    ndp_buffer_t* buffer, int* output_fd);
    int op_release(ndp_client_t client, ndp_op_t op);
    int metrics(ndp_client_t client, ndp_metrics_v1* output);

 private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace emender_ndp

#endif
