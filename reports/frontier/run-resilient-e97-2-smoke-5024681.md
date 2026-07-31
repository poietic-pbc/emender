# Resilient E97 kernel-config startup smoke — job 5024681

Real submission at `2026-07-17T19:59` EDT from fetched authoritative commit
`64fa9a16210ef367eab37c2dbf73c2a02003b1a9`; not `--test-only`.

- Run: `run-resilient-e97-2-smoke-20260717T235921Z-64fa9a1`.
- Payload: `64fa9a1-20260717T235921Z-startup-smoke-e97-kernel-flags`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:20:00`, no injection.
- One finalized generation is required before any full gate submission.
- Pinned step-1525000 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Offline p50k cache SHA256:
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`.

The immutable run directory retains the exact command. This changed payload
follows job 5024460, whose node-local tokenizer staging passed before model
construction rejected an incomplete chunked-E97 kernel configuration. The
approved flat configuration now explicitly binds `use_triton=1`,
`use_split_edit=1`, `linear_state=1`, and `e88_raw_write=0`; its focused
launcher regression suite passed 22 tests before submission.

Conformance checked against *Resilient DiLoCo Compute Pool*, version 1;
applicable R02, R03, R04, R06, R08, R09, R10, R14, and R16. No R16 pass is
claimed until an immutable generation is finalized.
