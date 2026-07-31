#define _POSIX_C_SOURCE 200809L
#include "emender/ndp_transport.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t unix_ns(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static void copy_span(uint8_t *out, uint32_t *length, const char *value) {
  *length = (uint32_t)strlen(value);
  memcpy(out, value, *length);
}

int main(void) {
  if (ndp_transport_abi_version() != NDP_TRANSPORT_ABI_V1) return 1;
  if (strcmp(ndp_transport_error_string(NDP_T_EFENCE), "stale fence") != 0) return 2;

  struct ndp_transport_open_v1 open;
  FILE *telemetry = tmpfile();
  if (telemetry == NULL) return 16;
  memset(&open, 0, sizeof(open));
  open.struct_size = (uint32_t)sizeof(open);
  open.abi_version = NDP_TRANSPORT_ABI_V1;
  open.mode = NDP_T_MODE_TEST_ONLY;
  open.tx_slots = 1; open.rx_slots = 1;
  open.payload_max = 4096;
  open.resident_limit_bytes = UINT64_C(1048576);
  open.operation_deadline_unix_ns = unix_ns() + UINT64_C(10000000000);
  open.telemetry_fd = fileno(telemetry);
  copy_span(open.provider, &open.provider_len, "tcp;ofi_rxm");
  copy_span(open.bind_node, &open.bind_node_len, "127.0.0.1");

  ndp_transport_t transport = 0;
  open.struct_size = 8;
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_EVERSION) return 19;
  open.struct_size = (uint32_t)sizeof(open);
  open.abi_version = UINT32_C(0x00020000);
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_EVERSION) return 3;
  open.abi_version = NDP_TRANSPORT_ABI_V1;
  open.resident_limit_bytes = 0;
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_EPROVIDER) return 23;
  open.resident_limit_bytes = UINT64_C(1048576);
  open.reserved0 = 1;
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_EINVAL) return 24;
  open.reserved0 = 0;
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_OK || transport == 0) return 4;

  struct ndp_transport_endpoint_v1 endpoint;
  memset(&endpoint, 0, sizeof(endpoint));
  endpoint.struct_size = (uint32_t)sizeof(endpoint);
  endpoint.abi_version = NDP_TRANSPORT_ABI_V1;
  if (ndp_transport_endpoint_v1(transport, &endpoint) != NDP_T_ESTATE) return 25;
  struct ndp_transport_identity_v1 identity;
  memset(&identity, 0, sizeof(identity));
  identity.struct_size = (uint32_t)sizeof(identity);
  identity.abi_version = NDP_TRANSPORT_ABI_V1;
  memset(identity.run_key, 9, sizeof(identity.run_key));
  identity.fence_epoch = 7;
  memset(identity.worker_key, 1, sizeof(identity.worker_key));
  memset(identity.incarnation, 2, sizeof(identity.incarnation));
  identity.endpoint_epoch = 1;
  identity.expires_unix_ns = open.operation_deadline_unix_ns;
  if (ndp_transport_bind_identity_v1(transport, &identity) != NDP_T_OK) return 26;
  if (ndp_transport_endpoint_v1(transport, &endpoint) != NDP_T_OK ||
      endpoint.record_bytes == 0 || endpoint.record_bytes > NDP_TRANSPORT_ENDPOINT_MAX) return 5;
  struct ndp_transport_endpoint_v1 first_endpoint;
  memcpy(&first_endpoint, &endpoint, sizeof(endpoint));

  struct ndp_transport_peer_v1 peer;
  uint64_t peer_id = 0, same_id = 0, rejoined_id = 0;
  memset(&peer, 0, sizeof(peer));
  peer.struct_size = (uint32_t)sizeof(peer);
  peer.abi_version = NDP_TRANSPORT_ABI_V1;
  memset(peer.worker_key, 1, sizeof(peer.worker_key));
  memset(peer.incarnation, 2, sizeof(peer.incarnation));
  peer.endpoint_epoch = 1;
  peer.expires_unix_ns = open.operation_deadline_unix_ns;
  peer.endpoint_name_bytes = endpoint.record_bytes;
  memcpy(peer.endpoint_name, endpoint.record, endpoint.record_bytes);
  if (ndp_transport_peer_upsert_v1(transport, &peer, &peer_id) != NDP_T_OK ||
      peer_id == 0) return 11;
  if (ndp_transport_peer_upsert_v1(transport, &peer, &same_id) != NDP_T_OK ||
      same_id != peer_id) return 12;
  ndp_transport_t peer_transport = 0;
  if (ndp_transport_open_v1(&open, &peer_transport) != NDP_T_OK ||
      peer_transport == 0) return 27;
  identity.endpoint_epoch = 2;
  memset(identity.incarnation, 3, sizeof(identity.incarnation));
  if (ndp_transport_bind_identity_v1(peer_transport, &identity) != NDP_T_OK) return 28;
  if (ndp_transport_endpoint_v1(peer_transport, &endpoint) != NDP_T_OK) return 29;
  peer.endpoint_epoch = identity.endpoint_epoch;
  peer.expires_unix_ns = identity.expires_unix_ns;
  memcpy(peer.incarnation, identity.incarnation, sizeof(peer.incarnation));
  peer.endpoint_name_bytes = endpoint.record_bytes;
  memcpy(peer.endpoint_name, endpoint.record, endpoint.record_bytes);
  if (ndp_transport_peer_upsert_v1(transport, &peer, &rejoined_id) != NDP_T_OK ||
      rejoined_id == 0 || rejoined_id == peer_id) return 13;
  peer.endpoint_epoch = 1;
  memset(peer.incarnation, 2, sizeof(peer.incarnation));
  peer.endpoint_name_bytes = first_endpoint.record_bytes;
  memcpy(peer.endpoint_name, first_endpoint.record, first_endpoint.record_bytes);
  if (ndp_transport_peer_upsert_v1(transport, &peer, &same_id) != NDP_T_ESTALE) return 14;

  struct ndp_transport_event_v1 event;
  uint32_t count = 99;
  memset(&event, 0, sizeof(event));
  event.struct_size = (uint32_t)sizeof(event);
  event.abi_version = NDP_TRANSPORT_ABI_V1;
  if (ndp_transport_poll_v1(transport, &event, 1, &count, 1) != NDP_T_OK || count != 0) return 6;

  struct ndp_transport_metrics_v1 metrics;
  memset(&metrics, 0, sizeof(metrics));
  metrics.struct_size = (uint32_t)sizeof(metrics);
  metrics.abi_version = NDP_TRANSPORT_ABI_V1;
  if (ndp_transport_metrics_v1(transport, &metrics) != NDP_T_OK ||
      metrics.provider_name_len != strlen("tcp;ofi_rxm") || metrics.live_peers != 1 ||
      metrics.owner_state != NDP_T_OWNER_ROUTE_READY ||
      metrics.in_flight_bytes != 0 || metrics.retained_bytes != 0) return 7;

  // A future caller may append zeroed fields; v1 reads and writes only its
  // declared prefix.
  {
    struct extended_metrics {
      struct ndp_transport_metrics_v1 v1;
      uint64_t future_tail;
    } extended;
    memset(&extended, 0, sizeof(extended));
    extended.v1.struct_size = (uint32_t)sizeof(extended);
    extended.v1.abi_version = NDP_TRANSPORT_ABI_V1;
    extended.future_tail = UINT64_C(0x1122334455667788);
    if (ndp_transport_metrics_v1(transport, &extended.v1) != NDP_T_OK ||
        extended.future_tail != UINT64_C(0x1122334455667788)) return 20;
  }

  // Cancellation notifications use the same capped metadata event queue.
  for (unsigned i = 0; i != 64; ++i) {
    if (ndp_transport_cancel_v1(transport, rejoined_id) != NDP_T_OK) return 21;
  }
  {
    struct ndp_transport_event_v1 events[64];
    uint32_t bounded_count = 0;
    memset(events, 0, sizeof(events));
    for (unsigned i = 0; i != 64; ++i) {
      events[i].struct_size = (uint32_t)sizeof(events[i]);
      events[i].abi_version = NDP_TRANSPORT_ABI_V1;
    }
    if (ndp_transport_poll_v1(transport, events, 64, &bounded_count, 0) != NDP_T_OK ||
        bounded_count > 24) return 22;
  }

  if (ndp_transport_peer_remove_v1(transport, rejoined_id) != NDP_T_OK) return 15;
  if (ndp_transport_close_v1(peer_transport) != NDP_T_OK) return 30;

  if (ndp_transport_close_v1(transport) != NDP_T_OK) return 8;
  if (ndp_transport_close_v1(transport) != NDP_T_ESTALE) return 9;
  if (fflush(telemetry) != 0 || fseek(telemetry, 0, SEEK_SET) != 0) return 17;
  {
    char telemetry_text[16384];
    const size_t bytes = fread(telemetry_text, 1, sizeof(telemetry_text) - 1, telemetry);
    telemetry_text[bytes] = '\0';
    if (strstr(telemetry_text, "emender-native-dataplane-telemetry-v1") == NULL ||
        strstr(telemetry_text, "\"useful_tx_bytes\"") == NULL ||
        strstr(telemetry_text, "\"wire_rx_bytes\"") == NULL ||
        strstr(telemetry_text, "\"retries\"") == NULL ||
        strstr(telemetry_text, "\"cq_errors\"") == NULL ||
        strstr(telemetry_text, "\"endpoint_type\":\"FI_EP_RDM\"") == NULL ||
        strstr(telemetry_text, "\"max_msg_size\"") == NULL ||
        strstr(telemetry_text, "\"mr_mode\"") == NULL ||
        strstr(telemetry_text, "\"throughput_bytes_per_second\"") == NULL ||
        strstr(telemetry_text, "\"mean_send_latency_ns\"") == NULL ||
        strstr(telemetry_text, "\"owner_state\"") == NULL ||
        strstr(telemetry_text, "\"retained_bytes\":0") == NULL) return 18;
  }
  fclose(telemetry);

  // A test provider can never be promoted by changing only require_provider.
  open.mode = NDP_T_MODE_PRODUCTION;
  open.telemetry_fd = -1;
  copy_span(open.require_provider, &open.require_provider_len, "cxi");
  if (ndp_transport_open_v1(&open, &transport) != NDP_T_EPROVIDER) return 10;
  puts("C ABI provider/handle lifecycle test passed");
  return 0;
}
