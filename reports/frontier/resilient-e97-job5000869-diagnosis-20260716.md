# Terminal diagnosis: resilient E97 job 5000869

Task: `complete-resilient-e97`

Evidence was re-read on 2026-07-16 before changing or resubmitting the
payload. No retry was submitted during this diagnosis.

## Terminal scheduler state

`sacct -j 5000869` records the two-node job as `FAILED`, exit `90:0`, after
00:02:56 on debug QoS (2026-07-15 04:37:33--04:40:29 local scheduler time).
The only application step, `5000869.0`, is `OUT_OF_MEMORY`, exit `0:125`.
It ran on `frontier06637` and `frontier09088`; Slurm requested 1000 GB per
node and reports step `MaxRSS=64857540K` (average `62980045.50K` per task).
The retained stdout identifies `frontier06637: task 4: Out Of Memory` at
04:40:26, followed by cancellation of the step and termination of the other
tasks. The batch wrapper consequently exited 90.

## Exact last progress boundary

All expected 16 ranks started: ranks 0--7 on `frontier06637` and 8--15 on
`frontier09088`. All 16 checkpoint-load/trainer paths reached generation 0
and wrote a local-training heartbeat with a finite loss (2.3493--2.5444).
The last retained heartbeat is rank 11 or 15 at 04:40:03 scheduler time.
Thus the exact boundary was **generation-0 local training complete / local
update construction beginning**, before resilient manager exchange,
aggregation, apply, or global finalization. The manifest records zero accepted
updates, zero training tokens at the global boundary, no metrics JSON, no
checkpoint path, and `latest_path=.`. No generation finalized.

There is no evidence of an independently launched manager process in the job:
the command launched 16 identical trainer/rank processes and assigned each
`--node-rank=$SLURM_PROCID`, with `--node-count 16`. Consequently there are no
separate manager exit codes to report. Trainer/rank 4 was kernel-OOM-killed;
ranks 0--3 and 5--15 were terminated by Slurm when the single `srun` step
failed. This launch topology also does not yet meet the required one-manager-
per-physical-node supervision boundary.

## Strongest supported root cause

The strongest supported cause is host-memory multiplication in the 16-rank
payload, not a network or quorum deadline. Every GPU rank independently loaded
the complete model and optimizer checkpoint into anonymous CPU memory, then
retained a full CPU base state while training. Eight copies per node exhausted
the 1 TB node allocation. The observed roughly 63--65 GB RSS per task times
eight tasks is consistent with this mechanism, and Slurm's explicit OOM event
occurred before any manager exchange. The prior payload therefore violated the
bounded per-node-memory requirement even though GPU-local training advanced.

## Preserved immutable handoff

Job 5000869 produced no checkpoint. Its last valid input remains the immutable
job-5000436 handoff:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z/async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`

No partial job-5000869 artifact is eligible for restart. A changed retry is
blocked until the host-memory behavior is regression-tested and committed.

## Retained evidence

- `logs/frontier/trainpy_async_quorum/resilient-e97-2n-20260715T083455Z-5000869.out`
- `logs/frontier/trainpy_async_quorum/resilient-e97-2n-20260715T083455Z-5000869.err`
- shared run root recorded in the stdout, including `artifacts/manifest.json`,
  `artifacts/rank-start.tsv`, 16 progress heartbeats, and the train log

