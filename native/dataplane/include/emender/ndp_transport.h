#ifndef EMENDER_NDP_TRANSPORT_H
#define EMENDER_NDP_TRANSPORT_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN32)
#define NDP_TRANSPORT_API __declspec(dllexport)
#else
#define NDP_TRANSPORT_API __attribute__((visibility("default")))
#endif

#define NDP_TRANSPORT_ABI_V1 UINT32_C(0x00010000)
#define NDP_TRANSPORT_ENDPOINT_MAX UINT32_C(4096)
#define NDP_TRANSPORT_PROVIDER_MAX UINT32_C(64)
#define NDP_TRANSPORT_FABRIC_MAX UINT32_C(128)
#define NDP_TRANSPORT_DOMAIN_MAX UINT32_C(128)
#define NDP_TRANSPORT_HEADER_BYTES UINT32_C(320)
#define NDP_TRANSPORT_MAX_PAYLOAD UINT64_C(67108864)
#define NDP_TRANSPORT_MAX_SHARDS UINT32_C(256)
#define NDP_TRANSPORT_MAX_CONTRIBUTIONS UINT32_C(4096)

typedef uint64_t ndp_transport_t;

enum ndp_transport_result_v1 {
  NDP_T_OK = 0,
  NDP_T_ACCEPTED = 1,
  NDP_T_EINVAL = -1,
  NDP_T_EVERSION = -2,
  NDP_T_ESTATE = -3,
  NDP_T_EFENCE = -4,
  NDP_T_ESTALE = -5,
  NDP_T_ECONFLICT = -6,
  NDP_T_ECHECKSUM = -7,
  NDP_T_ENONFINITE = -8,
  NDP_T_EBOUNDS = -9,
  NDP_T_ECREDIT = -10,
  NDP_T_EDEADLINE = -11,
  NDP_T_EROUTE = -12,
  NDP_T_EPROVIDER = -13,
  NDP_T_ENOMEM = -14,
  NDP_T_EIO = -15,
  NDP_T_ESHUTDOWN = -16
};

enum ndp_transport_mode_v1 {
  NDP_T_MODE_PRODUCTION = 1,
  NDP_T_MODE_TEST_ONLY = 2
};

enum ndp_transport_event_kind_v1 {
  NDP_T_EVENT_NONE = 0,
  NDP_T_EVENT_RECEIVED = 1,
  NDP_T_EVENT_SENT = 2,
  NDP_T_EVENT_CQ_ERROR = 3,
  NDP_T_EVENT_ROUTE_DOWN = 4,
  NDP_T_EVENT_CANCELLED = 5,
  NDP_T_EVENT_SHUTDOWN = 6
};

enum ndp_transport_owner_state_v1 {
  NDP_T_OWNER_ROUTE_IDLE = 0,
  NDP_T_OWNER_ROUTE_READY = 1,
  NDP_T_OWNER_DRAINING = 2
};

struct ndp_transport_open_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t mode;
  uint32_t tx_slots;
  uint32_t rx_slots;
  uint32_t reserved0;
  uint64_t payload_max;
  uint64_t resident_limit_bytes;
  uint64_t operation_deadline_unix_ns;
  int32_t telemetry_fd;
  uint32_t reserved1;
  uint32_t provider_len;
  uint8_t provider[NDP_TRANSPORT_PROVIDER_MAX];
  uint32_t require_provider_len;
  uint8_t require_provider[NDP_TRANSPORT_PROVIDER_MAX];
  uint32_t fabric_len;
  uint8_t fabric[NDP_TRANSPORT_FABRIC_MAX];
  uint32_t domain_len;
  uint8_t domain[NDP_TRANSPORT_DOMAIN_MAX];
  uint32_t bind_node_len;
  uint8_t bind_node[NDP_TRANSPORT_DOMAIN_MAX];
};

/*
 * The provider endpoint exists before Python admits this service to leased
 * membership.  Binding is therefore a distinct, additive ABI operation: the
 * allocation holder supplies the fenced identity only after acquiring the
 * run lease, and only then may the service publish a routable endpoint
 * record.  This keeps provider setup out of the Python dense path while
 * preventing an unfenced raw libfabric address from becoming READY.
 */
struct ndp_transport_identity_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint8_t run_key[16];
  uint64_t fence_epoch;
  uint8_t worker_key[16];
  uint8_t incarnation[16];
  uint64_t endpoint_epoch;
  uint64_t expires_unix_ns;
};

struct ndp_transport_endpoint_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t record_bytes;
  uint32_t reserved0;
  uint8_t record[NDP_TRANSPORT_ENDPOINT_MAX];
};

struct ndp_transport_peer_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint8_t worker_key[16];
  uint8_t incarnation[16];
  uint64_t endpoint_epoch;
  uint64_t expires_unix_ns;
  uint32_t endpoint_name_bytes;
  uint32_t reserved0;
  uint8_t endpoint_name[NDP_TRANSPORT_ENDPOINT_MAX];
};

struct ndp_transport_event_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint32_t event;
  int32_t status;
  uint64_t peer_id;
  uint64_t message_seq;
  uint64_t useful_bytes;
  uint64_t wire_bytes;
  int32_t provider_errno;
  uint32_t reason;
  uint8_t detail[32];
  uint64_t reserved0;
};

struct ndp_transport_metrics_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t useful_tx_bytes;
  uint64_t useful_rx_bytes;
  uint64_t wire_tx_bytes;
  uint64_t wire_rx_bytes;
  uint64_t retries;
  uint64_t replay_bytes;
  uint64_t duplicate_frames;
  uint64_t checksum_rejects;
  uint64_t stale_rejects;
  uint64_t cq_errors;
  uint64_t route_errors;
  uint64_t in_flight_bytes;
  uint64_t in_flight_high_water;
  uint64_t retained_bytes;
  uint64_t retained_high_water;
  uint64_t released_bytes;
  uint64_t tx_slot_high_water;
  uint64_t rx_slot_high_water;
  uint64_t latency_count;
  uint64_t latency_total_ns;
  uint64_t service_started_unix_ns;
  uint64_t last_progress_unix_ns;
  uint32_t owner_state;
  uint32_t live_peers;
  uint32_t provider_name_len;
  uint8_t provider_name[NDP_TRANSPORT_PROVIDER_MAX];
};

NDP_TRANSPORT_API uint32_t ndp_transport_abi_version(void);
NDP_TRANSPORT_API const char *ndp_transport_error_string(int code);
NDP_TRANSPORT_API int ndp_transport_open_v1(
    const struct ndp_transport_open_v1 *config, ndp_transport_t *out);
NDP_TRANSPORT_API int ndp_transport_bind_identity_v1(
    ndp_transport_t transport,
    const struct ndp_transport_identity_v1 *identity);
NDP_TRANSPORT_API int ndp_transport_endpoint_v1(
    ndp_transport_t transport, struct ndp_transport_endpoint_v1 *out);
NDP_TRANSPORT_API int ndp_transport_peer_upsert_v1(
    ndp_transport_t transport, const struct ndp_transport_peer_v1 *peer,
    uint64_t *peer_id);
NDP_TRANSPORT_API int ndp_transport_peer_remove_v1(
    ndp_transport_t transport, uint64_t peer_id);
NDP_TRANSPORT_API int ndp_transport_send_v1(
    ndp_transport_t transport, uint64_t peer_id, const uint8_t *frame,
    uint64_t frame_bytes, uint64_t deadline_unix_ns);
NDP_TRANSPORT_API int ndp_transport_poll_v1(
    ndp_transport_t transport, struct ndp_transport_event_v1 *events,
    uint32_t capacity, uint32_t *count, int timeout_ms);
NDP_TRANSPORT_API int ndp_transport_receive_v1(
    ndp_transport_t transport, uint8_t *frame, uint64_t capacity,
    uint64_t *frame_bytes, uint64_t *peer_id);
NDP_TRANSPORT_API int ndp_transport_metrics_v1(
    ndp_transport_t transport, struct ndp_transport_metrics_v1 *out);
NDP_TRANSPORT_API int ndp_transport_cancel_v1(
    ndp_transport_t transport, uint64_t peer_id);
NDP_TRANSPORT_API int ndp_transport_close_v1(ndp_transport_t transport);

#if defined(__cplusplus)
}
#define NDP_TRANSPORT_STATIC_ASSERT(type, bytes) \
  static_assert(sizeof(type) == (bytes), #type " ABI size changed")
#else
#define NDP_TRANSPORT_STATIC_ASSERT(type, bytes) \
  _Static_assert(sizeof(type) == (bytes), #type " ABI size changed")
#endif

NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_open_v1, 592);
NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_identity_v1, 80);
NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_endpoint_v1, 4112);
NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_peer_v1, 4160);
NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_event_v1, 96);
NDP_TRANSPORT_STATIC_ASSERT(struct ndp_transport_metrics_v1, 264);
#undef NDP_TRANSPORT_STATIC_ASSERT

#endif
