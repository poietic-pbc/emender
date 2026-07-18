#ifndef EMENDER_NDP_H
#define EMENDER_NDP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define NDP_API __declspec(dllexport)
#else
#define NDP_API __attribute__((visibility("default")))
#endif

#define NDP_ABI_V1 UINT32_C(0x00010000)

typedef uint64_t ndp_client_t;
typedef uint64_t ndp_buffer_t;
typedef uint64_t ndp_op_t;

enum ndp_result_code {
    NDP_OK = 0,
    NDP_IN_PROGRESS = 1,
    NDP_EINVAL = -1,
    NDP_EVERSION = -2,
    NDP_ESTATE = -3,
    NDP_EFENCE = -4,
    NDP_ESTALE = -5,
    NDP_ECONFLICT = -6,
    NDP_ECHECKSUM = -7,
    NDP_ENONFINITE = -8,
    NDP_EBOUNDS = -9,
    NDP_ECREDIT = -10,
    NDP_EDEADLINE = -11,
    NDP_EROUTE = -12,
    NDP_EPROVIDER = -13,
    NDP_ENOMEM = -14,
    NDP_EIO = -15,
    NDP_ESHUTDOWN = -16,
};

enum ndp_role {
    NDP_ROLE_TRAINER = 1,
    NDP_ROLE_CONTROLLER = 2,
};

enum ndp_buffer_kind {
    NDP_BUFFER_XPMEM_ADDRESS = 1,
    NDP_BUFFER_MEMFD = 2,
};

enum ndp_buffer_flags {
    NDP_BUFFER_READ = 1,
    NDP_BUFFER_WRITE = 2,
};

enum ndp_dtype {
    NDP_DTYPE_F32 = 1,
    NDP_DTYPE_BF16 = 2,
    NDP_DTYPE_F64 = 3,
};

enum ndp_control_command {
    NDP_CONTROL_BIND_FENCE = 1,
    NDP_CONTROL_INSTALL_GENERATION = 2,
    NDP_CONTROL_FREEZE = 3,
    NDP_CONTROL_REASSIGN = 4,
    NDP_CONTROL_FINALIZE_OWNERS = 5,
    NDP_CONTROL_COMMIT = 6,
    NDP_CONTROL_ABORT = 7,
    NDP_CONTROL_DRAIN = 8,
};

enum ndp_state {
    NDP_STATE_STARTING = 1,
    NDP_STATE_CONTROL_BOUND = 2,
    NDP_STATE_FABRIC_READY = 3,
    NDP_STATE_IDLE = 4,
    NDP_STATE_LOCAL_COLLECT = 5,
    NDP_STATE_PREPARED = 6,
    NDP_STATE_FROZEN = 7,
    NDP_STATE_TRANSFERRING = 8,
    NDP_STATE_OWNED_READY = 9,
    NDP_STATE_REDISTRIBUTING = 10,
    NDP_STATE_RESULT_READY = 11,
    NDP_STATE_COMMITTED = 12,
    NDP_STATE_ABORTING = 13,
    NDP_STATE_ABORTED = 14,
    NDP_STATE_DRAINING = 15,
    NDP_STATE_STOPPED = 16,
    NDP_STATE_FAULT = 17,
};

enum ndp_event_kind {
    NDP_EVENT_STATE = 1,
    NDP_EVENT_BUFFER_RELEASED = 2,
    NDP_EVENT_LOCAL_ACCEPTED = 3,
    NDP_EVENT_LOCAL_DUPLICATE = 4,
    NDP_EVENT_PREPARED = 5,
    NDP_EVENT_RESULT_READY = 6,
    NDP_EVENT_COMMITTED = 7,
    NDP_EVENT_ABORTED = 8,
    NDP_EVENT_DRAINED = 9,
};

enum ndp_status {
    NDP_STATUS_NONE = 0,
    NDP_STATUS_APPLIED = 1,
    NDP_STATUS_DUPLICATE = 2,
    NDP_STATUS_FINALIZED = 3,
    NDP_STATUS_REJECTED = 4,
    NDP_STATUS_RETRYABLE = 5,
};

enum ndp_reason {
    NDP_REASON_NONE = 0,
    NDP_REASON_STALE_FENCE = 1,
    NDP_REASON_STALE_GENERATION_OR_ATTEMPT = 2,
    NDP_REASON_STALE_OWNER_EPOCH = 3,
    NDP_REASON_NOT_ACCEPTED = 4,
    NDP_REASON_LAYOUT_OR_BASE = 5,
    NDP_REASON_BYTE_BOUNDS = 6,
    NDP_REASON_CHECKSUM = 7,
    NDP_REASON_NONFINITE = 8,
    NDP_REASON_CONFLICT = 9,
    NDP_REASON_NO_CREDIT = 10,
    NDP_REASON_DEADLINE = 11,
    NDP_REASON_ROUTE = 12,
    NDP_REASON_PROVIDER = 13,
    NDP_REASON_SHUTDOWN = 14,
};

struct ndp_open_v1 {
    uint32_t struct_size, abi_version, role, flags;
    uint32_t socket_path_len;
    uint8_t socket_path[108];
    uint8_t run_key[16];
    uint64_t fence_epoch;
    uint8_t worker_key[16], incarnation[16], admission_token[32];
    uint64_t deadline_unix_ns;
};

struct ndp_layout_v1 {
    uint32_t struct_size, abi_version;
    int32_t descriptor_fd;
    uint32_t reserved0;
    uint64_t descriptor_bytes;
    uint8_t layout_digest[32];
};

struct ndp_buffer_v1 {
    uint32_t struct_size, abi_version, kind, flags;
    uint64_t address_or_segid, offset, length, handle_generation;
    int32_t fd;
    uint32_t reserved0;
    uint8_t layout_digest[32];
};

struct ndp_alloc_v1 {
    uint32_t struct_size, abi_version, flags, reserved0;
    uint64_t bytes, deadline_unix_ns;
};

struct ndp_submit_v1 {
    uint32_t struct_size, abi_version;
    uint64_t buffer;
    uint8_t trainer_key[16], trainer_incarnation[16];
    uint64_t submission_seq, weight, element_offset, element_count;
    uint32_t source_dtype, flags;
    uint64_t deadline_unix_ns;
    uint8_t source_buffer_sha256[32];
};

struct ndp_control_v1 {
    uint32_t struct_size, abi_version, command, flags;
    uint8_t run_key[16];
    uint64_t fence_epoch, generation;
    uint32_t attempt, metadata_kind;
    uint64_t owner_epoch, deadline_unix_ns;
    int32_t metadata_fd;
    uint32_t reserved0;
    uint64_t metadata_bytes;
    uint8_t layout_digest[32], base_digest[32], plan_digest[32], metadata_sha256[32];
};

struct ndp_event_v1 {
    uint32_t struct_size, abi_version, event, status, reason, state;
    uint64_t op, generation;
    uint32_t attempt, shard_id;
    uint64_t owner_epoch, logical_bytes;
    uint8_t detail_digest[32];
};

struct ndp_result_v1 {
    uint32_t struct_size, abi_version, flags, dtype;
    uint8_t run_key[16];
    uint64_t fence_epoch, generation;
    uint32_t attempt, reserved0;
    uint8_t layout_digest[32], base_digest[32], result_root[32];
    uint64_t global_weight, result_bytes;
};

/* Optional v1 telemetry extension. It is prefix/versioned like the core ABI. */
struct ndp_metrics_v1 {
    uint32_t struct_size, abi_version;
    uint64_t shared_bytes_current, shared_bytes_high_water;
    uint64_t admitted_shared_bytes, released_shared_bytes;
    uint64_t mapped_bytes_current, mapped_bytes_high_water;
    uint64_t prompt_source_released_bytes, result_bytes;
    uint64_t disk_replay_bytes, disk_replay_files;
    uint64_t trainer_spool_bytes, trainer_spool_files;
    uint64_t python_dense_socket_bytes, handoff_full_copy_bytes;
    uint64_t projection_count, duplicate_count, conflict_count;
    uint64_t checksum_rejects, nonfinite_rejects, stale_rejects;
    uint64_t cancelled_ops, buffer_exhaustions;
};

NDP_API uint32_t ndp_abi_version(void);
NDP_API const char *ndp_error_string(int code);

NDP_API int ndp_client_open_v1(const struct ndp_open_v1 *, ndp_client_t *);
NDP_API int ndp_client_poll_fd_v1(ndp_client_t, int *dup_fd);
NDP_API int ndp_client_close_v1(ndp_client_t);

NDP_API int ndp_layout_install_v1(ndp_client_t, const struct ndp_layout_v1 *);
NDP_API int ndp_buffer_register_v1(ndp_client_t, const struct ndp_buffer_v1 *,
                                   ndp_buffer_t *);
NDP_API int ndp_buffer_allocate_v1(ndp_client_t, const struct ndp_alloc_v1 *,
                                   ndp_buffer_t *, int *dup_fd);
NDP_API int ndp_buffer_seal_v1(ndp_client_t, ndp_buffer_t);
NDP_API int ndp_buffer_release_v1(ndp_client_t, ndp_buffer_t);

NDP_API int ndp_submit_local_v1(ndp_client_t, const struct ndp_submit_v1 *,
                                ndp_op_t *);
NDP_API int ndp_control_v1(ndp_client_t, const struct ndp_control_v1 *, ndp_op_t *);
NDP_API int ndp_poll_v1(ndp_client_t, struct ndp_event_v1 *events,
                        uint32_t capacity, uint32_t *count, int timeout_ms);
NDP_API int ndp_result_view_v1(ndp_client_t, ndp_op_t,
                               struct ndp_result_v1 *, ndp_buffer_t *, int *dup_fd);
NDP_API int ndp_op_release_v1(ndp_client_t, ndp_op_t);
NDP_API int ndp_client_metrics_v1(ndp_client_t, struct ndp_metrics_v1 *);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus)
static_assert(sizeof(struct ndp_open_v1) == 224, "ndp_open_v1 ABI size");
static_assert(sizeof(struct ndp_layout_v1) == 56, "ndp_layout_v1 ABI size");
static_assert(sizeof(struct ndp_buffer_v1) == 88, "ndp_buffer_v1 ABI size");
static_assert(sizeof(struct ndp_alloc_v1) == 32, "ndp_alloc_v1 ABI size");
static_assert(sizeof(struct ndp_submit_v1) == 128, "ndp_submit_v1 ABI size");
static_assert(sizeof(struct ndp_control_v1) == 216, "ndp_control_v1 ABI size");
static_assert(sizeof(struct ndp_event_v1) == 96, "ndp_event_v1 ABI size");
static_assert(sizeof(struct ndp_result_v1) == 168, "ndp_result_v1 ABI size");
static_assert(sizeof(struct ndp_metrics_v1) == 184, "ndp_metrics_v1 ABI size");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(struct ndp_open_v1) == 224, "ndp_open_v1 ABI size");
_Static_assert(sizeof(struct ndp_layout_v1) == 56, "ndp_layout_v1 ABI size");
_Static_assert(sizeof(struct ndp_buffer_v1) == 88, "ndp_buffer_v1 ABI size");
_Static_assert(sizeof(struct ndp_alloc_v1) == 32, "ndp_alloc_v1 ABI size");
_Static_assert(sizeof(struct ndp_submit_v1) == 128, "ndp_submit_v1 ABI size");
_Static_assert(sizeof(struct ndp_control_v1) == 216, "ndp_control_v1 ABI size");
_Static_assert(sizeof(struct ndp_event_v1) == 96, "ndp_event_v1 ABI size");
_Static_assert(sizeof(struct ndp_result_v1) == 168, "ndp_result_v1 ABI size");
_Static_assert(sizeof(struct ndp_metrics_v1) == 184, "ndp_metrics_v1 ABI size");
#endif

#endif
