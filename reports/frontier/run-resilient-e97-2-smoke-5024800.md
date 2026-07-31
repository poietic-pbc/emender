# Resilient E97 exact-model startup smoke — job 5024800

Real submission at `2026-07-17T20:05:59-04:00` from fetched authoritative
commit `2ee76fee8dfc4ea0d6ce1f5691cebf058778d144`; not `--test-only`.

- Run: `run-resilient-e97-2-smoke-20260718T000536Z-2ee76fe`.
- Payload: `2ee76fe-20260718T000536Z-startup-smoke-exact-e97-model`.
- Immediate state: `PENDING (Priority)`, runtime `00:00:00`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:20:00`, no injection.
- Pinned seed and offline p50k cache retain their previously verified SHA256s.

This changed payload follows 5024681's pre-generation invalid E97 tile error.
The flat config is now checked against the retained pinned-model command:
depth 11, 32 groups, 64 slots, MLP ratio 2.2623/multiple 64, batch 1,
sequence length 2048, E97 tile 32, and the production non-chunked Triton path.
The immutable run directory retains `exact-command.sh`; 22 launcher tests
passed pre-submit. No full gate was submitted.

Conformance checked against architecture version 1 and R02, R03, R04, R06,
R08, R09, R10, R14, R16. A pass still requires one immutable generation.
