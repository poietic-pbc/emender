#include "emender/ndp.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    if (ndp_abi_version() != NDP_ABI_V1) return 1;
    if (ndp_abi_version_v21() != NDP_ABI_V21) return 6;
    if (strcmp(ndp_error_string(NDP_EFENCE), "stale allocation fence") != 0) return 2;
    if (sizeof(struct ndp_open_v1) != 224 || sizeof(struct ndp_result_v1) != 168) return 3;

    struct ndp_open_v1 open_request;
    memset(&open_request, 0, sizeof(open_request));
    open_request.struct_size = sizeof(open_request);
    open_request.abi_version = NDP_ABI_V1;
    ndp_client_t client = UINT64_C(123);
    if (ndp_client_open_v1(&open_request, &client) != NDP_EINVAL || client != 0) return 4;

    struct ndp_submit_v21 v21;
    memset(&v21, 0, sizeof(v21));
    v21.struct_size = sizeof(v21);
    v21.abi_version = NDP_ABI_V21;
    memcpy(v21.policy_id, "async-decoupled-v2.0-exp",
           sizeof("async-decoupled-v2.0-exp") - 1);
    v21.policy_id_len = 26;
    ndp_op_t operation = 0;
    if (ndp_submit_local_v21(0, &v21, &operation) != NDP_EINVAL) return 7;

    puts("native resilient data-plane ABI v1/v2.1 are compiler-checked");
    return 0;
}
