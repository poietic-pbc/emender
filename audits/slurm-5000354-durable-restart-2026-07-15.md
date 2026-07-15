# Slurm job 5000354 durable-restart audit

Audit time: 2026-07-15 02:21–02:22 EDT (2026-07-15 06:21–06:22 UTC)

Verdict: **CANCEL**. Only debug job `5000354` was cancelled. Production job
`4980157` was not modified.

## Live scheduler evidence

- At 02:21:01 EDT, `squeue` reported `5000354|PENDING|Priority`, with submit
  time `2026-07-15T02:06:33`, no allocated nodes, and no estimated start.
- `scontrol show job -dd 5000354` reported `JobState=PENDING`,
  `Reason=Priority`, `StartTime=Unknown`, and `EndTime=Unknown`. Consequently,
  Slurm supplied no projected StartTime and no start delta could be calculated.
- At 02:22:11 EDT, after the deficient audit verdict, `scancel 5000354` was
  issued. `sacct -X` then reported `CANCELLED by 19032`, start `None`, end
  `2026-07-15T02:22:11`, and elapsed `00:00:00`.
- A read-only check after cancellation reported production `4980157` still
  `PENDING (Priority)` with projected start `2026-07-16T09:10:00`.

## Persistent roots and cadence

The exact submitted persistent output root was:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709`

The runner would derive the exact per-job run root at job start as:

`$OUTPUT_ROOT/$(date -u +%Y%m%d)/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000354-$(date -u +%Y%m%dT%H%M%SZ)`

and place restart state under `$RUN_ROOT/async_run`. This is on persistent
project-shared Lustre storage. The `step1065000` component is a stale cosmetic
label; it does not select the submitted checkpoint.

Submitted checkpoint cadence was:

- recovery checkpoint every `1` generation (`RECOVERY_EVERY_GENERATIONS=1`);
- export checkpoint every `1` generation (`EXPORT_EVERY_GENERATIONS=1`);
- wall-clock recovery/export cadence disabled (`-1` seconds);
- `40` local steps per generation (`DILOCO_K=40`, `ASYNC_LOCAL_STEPS=40`);
- model activation checkpoint interval `16` (a memory setting, not durable-save
  cadence);
- finalization reserve `1200` seconds.

The checkpoint implementation records model and optimizer state and atomically
advances run-local pointers (`latest.pt` and checkpoint-manager `latest.json`).
Those are useful state-completeness mechanisms during a clean generation
publication, but they do not cure the scheduler-exit and restart-selection
deficiencies below.

## Submitted seed identity

The exact submitted seed path was:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt`

The runner checks that `E97_CHECKPOINT` is readable and appends it to the real
trainer command as `--checkpoint "$E97_CHECKPOINT"`; therefore the stale
`SCALEOUT_VARIANT` label does not cause step 1065000 to load.

Both `manifest.json` and `latest_emender_E97_1.3B.json` identify:

- loaded/seed step: `1525000`;
- size: `7719679924` bytes;
- SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`;
- source URI:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1525000/checkpoint_step_1525000_loss_2.4378.pt`.

The adjacent checksum file contains the same SHA256, and
`latest_symlink_target.txt` points to
`checkpoint_step_1525000_loss_2.4378.pt`.
An independent `sha256sum` of the 7,719,679,924-byte persistent checkpoint
completed during this audit and reproduced the exact approved digest.

## Missing durability requirements

The job was cancelled because all required restart behavior was not present:

1. The submitted sbatch wrapper has no `#SBATCH --signal=...` directive.
2. Neither the wrapper nor the common runner installs a `TERM`, `USR1`, or
   `EXIT` trap that requests and waits for a final durable checkpoint.
3. The summary/restart manifest is generated only after the `srun` pipeline
   returns. A scheduler kill can prevent this post-processing entirely.
4. The submitted command supplies only the immutable seed checkpoint. It does
   not supply a restart manifest/pointer from a prior run, and every start
   creates a new timestamped `RUN_ROOT`. Thus a rerun has no deterministic
   submitted mechanism to discover and resume job 5000354's latest durable
   generation.

## Notification

A `CANCEL` verdict containing the persistent root, approved seed identity,
deficiencies, verified cancellation, scheduler status, and production
non-interference was sent with `wg msg send run-exact-256n-2h-debug`.
