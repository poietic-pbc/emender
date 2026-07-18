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

## Terminal result

Job 5024821 started at `2026-07-17T20:31:03-04:00` after 8m43s queued and
ended at `2026-07-17T21:05:56-04:00` after 34m53s runtime. Slurm reported
`FAILED`, `RaisedSignal:15(Terminated)`, exit `0:15`; TERM@300 therefore
arrived at the expected 35-minute boundary. All two managers and sixteen real
trainers started. Their logs show the real HIP E97 split-edit Triton path and
real `train.py` optimizer work, but no trainer completed all 40 local steps and
no generation/checkpoint was finalized. This is a failed startup smoke, not
acceptance evidence.

The preserved supervision stream records all 18 role identities and the
allocation termination evictions. The immutable command and role logs remain
under `/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260718T002156Z-52da7b1`.
The next payload adds per-optimizer-step committed progress heartbeats and uses
a 50-minute short smoke (45 minutes before TERM) so the observed generation
duration is measurable and one finalized generation remains mandatory before
any full gate submission.
