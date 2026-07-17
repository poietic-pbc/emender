# Resilient E97 offline-tokenizer startup smoke — job 5024460

Real (not `--test-only`) submission at `2026-07-17T19:54:46-04:00` from
fetched authoritative commit `e7fb705f435a8809ef67a7456e2af6b10b75ca18`.
The compute-visible tracked checkout had `HEAD == origin/main`.

- Run: `run-resilient-e97-2-smoke-20260717T235305Z-e7fb705`.
- Payload: `e7fb705-20260717T235305Z-startup-smoke-offline-p50k`.
- Immediate state: `PENDING (Priority)`; runtime remains `00:00:00` while queued.
- Resources: exactly 2 nodes, 16 GPUs, debug QoS, `00:20:00`.
- Injection: none; requested finalized generations: one.
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Tokenizer input SHA256:
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`.

The immutable run directory retains `exact-command.sh`. The launcher verifies
the p50k input before staging it to the payload-specific node-local cache on
both nodes, and exports `TIKTOKEN_CACHE_DIR` before starting either manager or
trainer roles. Launcher regression validation passed 21 tests before this
submission. Rendered production parity reports `ok=true`, no forbidden or
missing fields, and identical model/data/optimizer/launcher/checkpoint/network
fields; this startup smoke additionally disables the allowlisted injection.
No full `02:00:00` gate has been submitted.

Conformance checked against *Resilient DiLoCo Compute Pool*, version 1;
applicable R02, R03, R04, R06, R08, R09, R10, R14, and R16. Acceptance still
requires both model-free managers, all sixteen real trainers, bounded liveness
and progress, network/node-local hot-path transport, and one immutable
finalized generation.
