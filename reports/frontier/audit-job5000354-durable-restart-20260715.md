# Durable restart audit: Slurm job 5000354

Audit time: 2026-07-15 (America/New_York)

Verdict: **PASS — keep debug job 5000354 queued.** No job was submitted or
cancelled during this audit. Production job 4980157 was inspected read-only and
left untouched.

## Scheduler and persistent roots

`scontrol show job -dd 5000354` reported the job `PENDING (Priority)`, QoS
`debug`, walltime `02:00:00`, 256 nodes and 2,048 tasks. Its immutable command
is `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch` in the retained
attested tree. The submitted environment fixes:

- `OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709`
- `SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n`
- `SMOKE_NAME=256n`
- `SLURM_JOB_ID=5000354` when allocated

The shared launcher constructs the run root as:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000354-<UTC-launch-stamp>
```

The payload root is `<RUN_ROOT>/async_run`. Because the job was pending during
the audit, the UTC allocation-time stamp had not yet been instantiated. The
entire root is on persistent project-shared Lustre; only compiled-helper IPC is
under node-local `${TMPDIR:-/tmp}`.

Evidence: the shared launcher defines `OUTPUT_ROOT`, `RUN_ROOT`, and `RUN_DIR`
at `scripts/frontier/trainpy_async_quorum_smoke_common.sh:19-24` and passes
`--run-dir "$RUN_DIR"` at lines 228-233.

## Checkpoint cadence and completeness

The submitted defaults set both `RECOVERY_EVERY_GENERATIONS=1` and
`EXPORT_EVERY_GENERATIONS=1`, with their wall-clock alternatives disabled.
They are passed to the async trainer at
`scripts/frontier/trainpy_async_quorum_smoke_common.sh:152-155,271-274`.
Consequently every successfully merged generation produces:

- a finalized generation manifest at
  `async_run/generations/gen_<generation>/manifest.json`;
- recovery and export records at
  `async_run/recovery_checkpoints/gen_<generation>/...json` and
  `async_run/export_checkpoints/gen_<generation>/...json`;
- a full chain checkpoint at
  `async_run/checkpoints/<model-label>/checkpoint_step_<step>_loss_<loss>.pt`.

The chain checkpoint writer is
`ndm/async_diloco_real.py:865-951`. It writes to a temporary file, reloads it
with mmap, requires `model_state_dict`, `optimizer_state_dict`, and the expected
step, and only then atomically renames it into place. Its payload contains the
model state, optimizer state, step, loss, and checkpoint metadata including
`kind=async_diloco_chain`, run ID, generation, source checkpoint and step,
model/tokenizer/optimizer identity, local-step count, token count, and update
accounting. This is sufficient state for the established trainer continuation
path; it is not merely a metrics manifest.

## Restart pointers and scheduler-exit persistence

After a chain checkpoint verifies, the writer atomically replaces
`async_run/latest.pt` with a relative symlink to that checkpoint
(`ndm/async_diloco_real.py:941-950`). The generation publisher then atomically
replaces `async_run/latest.json` with the run ID, generation, finalized manifest
path, all checkpoint paths, and `model_checkpoint_path`
(`ndm/async_diloco.py:876-909`). Resume selection ignores partial data and
selects only finalized authoritative global-generation manifests
(`ndm/async_diloco.py:927-955,1105-1121`).

Thus a scheduler exit can interrupt only the current temporary/in-flight
generation. All earlier completed `.pt` files and manifests remain on Lustre,
and the atomic pointers continue to name the previous verified generation.
The launcher also requests finalization inside its reserve window, but restart
safety does not depend on the final generation completing.

## Isolation and disposition

The PASS verdict was sent to WG task `run-exact-256n-2h-debug` as message 5.
Job 5000354 was not cancelled. A read-only `scontrol`/`sacct` snapshot showed
production job 4980157 still `PENDING (Priority)`, QoS `normal`, walltime
`12:00:00`; this audit performed no scheduler mutation against it.
