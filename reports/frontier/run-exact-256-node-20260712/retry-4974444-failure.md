# Exact E97 256-node smoke 4974444: rejected

Job 4974444 was submitted from the freshly rendered immutable smoke bundle
after the strengthened allocation-free prologue gate passed.  The allocation
used the required 256 nodes and 2048 tasks, but Slurm terminated the batch shell
with `SIGUSR1` nine seconds after start.  This attempt is rejected: no promotion
record was written and no production job was submitted.

## Immutable submission

- Command: `sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch`
- Job ID: `4974444`
- Bundle fingerprint: `338b05d24fba86b1a12d40413622ba258f389a43ff470e8e9ea5fed6ea509be3`
- Queue: partition `batch`, QoS `debug`, account `bif148`
- Reservation: absent
- Walltime: `00:20:00`
- Topology: 256 nodes, 2048 tasks, 8 tasks/node, 1 GPU/task, 7 CPUs/task
- Submit/start/end: `2026-07-12T04:22:45-04:00` /
  `2026-07-12T04:23:21-04:00` / `2026-07-12T04:23:30-04:00`

The exact bundle was retained in `retry-4974444-bundle/`; mandatory render and
parity evidence was retained with the submission checkpoint.

## Terminal evidence and root cause

`sacct` and `scontrol` report:

```text
4974444|e97-async-256-smoke|FAILED|0:10|00:00:09|256
4974444.batch|batch|CANCELLED|0:10|00:00:09|1|1
4974444.extern|extern|COMPLETED|0:0|00:00:09|256|256
JobState=FAILED Reason=RaisedSignal:10(User_defined_signal_1)
```

The immutable script requested `#SBATCH --signal=B:USR1@1200`, while the smoke
walltime is exactly 1200 seconds.  Slurm therefore delivered the advance signal
at the beginning of the allocation.  The batch script has no `USR1` trap, so
the signal terminated it during `preflight-before-helper`:

```text
e97-presrun phase=preflight-before-helper status=begin fingerprint=338b05d24fba86b1a12d40413622ba258f389a43ff470e8e9ea5fed6ea509be3
```

The stderr SHA256 is
`b95ebf99ca5150257c6d90bb91582e1856de8434753e6d65be3bf9441e79bb51`;
stdout is empty with SHA256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Acceptance disposition

- On-node preflight did not complete and the helper was not built.
- No `srun` step exists; zero of 2048 ranks started or contributed.
- No accepted updates, finite loss, DiLoCo merge, metrics, rank manifest,
  finalized checkpoint, or reload evidence exists.
- `build/e97-256/smoke/promotion.json` is absent.
- The configured external seed remains the pre-run object
  `checkpoint_step_1065000_loss_2.5386.pt`; the launcher never reached `srun`.
- Production submission is not authorized.

A prerequisite repair must make the advance-signal policy valid for the smoke
duration (and add deterministic validation that rejects an advance interval
greater than or equal to walltime) before another exact allocation is submitted.
