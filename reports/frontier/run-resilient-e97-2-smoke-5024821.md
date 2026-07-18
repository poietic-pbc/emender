# Resilient E97 generation-duration startup smoke — job 5024821

Real submission from fetched authoritative commit
`52da7b19790c3d89e1ee0aacdf6da1740e39e683`; not `--test-only`.

- Run: `run-resilient-e97-2-smoke-20260718T002156Z-52da7b1`.
- Payload: `52da7b1-20260718T002156Z-startup-smoke-one-generation-runtime`.
- Exactly 2 nodes, 16 GPUs, debug QoS, `00:40:00`, no injection.
- Progress/generation bounds: 2100 seconds; Slurm still sends TERM@300.
- One finalized generation is mandatory before a full gate.

The immutable run directory retains the exact command. This changed payload
follows 5024800, which ran the real 1.3B model with all original roles for
14m32s without a runtime error but received TERM@300 before completing its
first 40-step generation. Forty minutes provides 35 minutes before TERM while
remaining a short debug-only startup gate. The full resilience gate remains
exactly `02:00:00`. Focused launcher validation passed 22 tests pre-submit.

Conformance checked against architecture version 1 and R02, R03, R04, R06,
R08, R09, R10, R14, R16. No generation pass is claimed yet.
