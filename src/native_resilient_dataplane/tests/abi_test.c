#include "emender/ndp.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    if (ndp_abi_version() != NDP_ABI_V1) return 1;
    if (strcmp(ndp_error_string(NDP_EFENCE), "stale allocation fence") != 0) return 2;
    if (sizeof(struct ndp_open_v1) != 224 || sizeof(struct ndp_result_v1) != 168) return 3;

    struct ndp_open_v1 open_request;
    memset(&open_request, 0, sizeof(open_request));
    open_request.struct_size = sizeof(open_request);
    open_request.abi_version = NDP_ABI_V1;
    ndp_client_t client = UINT64_C(123);
    if (ndp_client_open_v1(&open_request, &client) != NDP_EINVAL || client != 0) return 4;

    puts("native resilient data-plane ABI v1 is compiler-checked");
    return 0;
}
