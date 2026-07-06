# Prepare real async E97 256n production decision

Task: `prepare-real-async`  
Date: 2026-07-06  
Decision scope: synthesize the real async implementation and validation ladder
evidence, then decide whether to unpause or replace the old
`retry-refreshed-e97` 256-node retry.  
Decision constraint: do not submit a 256-node job without fresh user approval
after this report.

## Recommendation

**No-go. Do not submit, unpause, or replace the 256-node retry yet.**

The real token training path is **not validated** in the current checkout. The
required real trainer entrypoint,
`scripts/frontier/e97_async_diloco_train.py`, is absent. The 1-node and 2-node
ladder jobs reached the production wrapper but failed before training with
wrapper exit code `65`; the attempted 8-node, 20-minute validation was accepted
by Slurm as job `4948004` but never received an allocation and was canceled at
`00:00:00` elapsed. Therefore there is no finite real loss, real tokens/sec,
quorum distribution, checkpoint recovery, export, or finalization evidence for
the real path.

Keep the old `retry-refreshed-e97` task paused. It is a synthetic/debug-era
256-node retry and should not be resumed from the current evidence. No
ready-but-paused submit task is created by this report because submission is not
recommended.

## Evidence reviewed

- `docs/QUALITY_PASS_REAL_ASYNC_E97_20260706.md`: current checkout was scored
  no-go for real async validation because the wrapper still defaults to the
  prototype path and the advertised real trainer files are not present in
  `HEAD`. It also flags that the dependency-branch helper, while closer to real
  token math, had not proven distributed production transport.
- `docs/validation/validate-real-async/VALIDATE_REAL_ASYNC_E97_1N2N_20260706.md`:
  the 1-node job `4947951` and 2-node job `4947962` were submitted through the
  production wrapper with `ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py`.
  Both failed before training because that file was missing.
- `docs/validation/validate-real-async-2/VALIDATE_REAL_ASYNC_E97_8N20M_20260706.md`:
  the 8-node job `4948004` used the production launcher and forced the same
  real entrypoint, but stayed pending on `batch/debug` with reason `Priority`
  and was canceled before start. No 8-node logs, metrics, or run artifacts were
  produced.
- `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`: source of the
  proposed 256-node production configuration below.
- `wg show retry-refreshed-e97`: the old 256-node retry is currently
  `open (PAUSED)`.

## Real token training path status

**Not validated.**

What is proven:

- The production wrapper can create pre-launch environment and command
  artifacts on 1-node and 2-node debug allocations.
- The wrapper correctly refuses to launch when the required real entrypoint is
  forced but missing.
- The refreshed seed path and intended token geometry were recorded by the
  failed 1-node and 2-node attempts.
- The external production `latest.pt` guard remained unchanged through the
  ladder.

What is not proven:

- The current checkout can import or execute a real async E97 trainer.
- Any real training step runs from real dataset batches.
- Any finite real loss or real tokens/sec is produced.
- Local and global quorum behavior is exercised by real worker updates.
- Deferred/stale/timed-out/failed update accounting is produced by real
  training.
- Recovery checkpoints, export checkpoints, finalization, or generation
  manifests are produced by the real path.
- 8-node, 20-minute production-wrapper behavior, because job `4948004` never
  started.
- 256-node distributed async transport, because neither the 1n/2n nor 8n job
  produced trainer evidence.

## Proposed 256n config under discussion

This is the exact production candidate configuration encoded in
`scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`, not a submit
authorization.

| Field | Value |
| --- | --- |
| Launcher | `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` |
| Slurm account | `bif148` |
| Partition / QoS | `batch` / normal production QoS |
| Node count | `256` |
| Tasks per node | `1` |
| CPUs per task | `56` |
| Walltime | `12:00:00` requested; wrapper `TRAIN_MINUTES=700` |
| Node-hours | `256 * 12 = 3072` requested node-hours |
| Training target | `E97_1.3B_step1065000_async_diloco_256n12h_20260706` |
| Scaleout variant | `E97_1.3B_step1065000_async_quorum_b4_k40_256n12h` |
| Seed/latest path | `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt` |
| Default wrapper entrypoint | `scripts/frontier/async_diloco_e97_multinode.py` |
| Required real entrypoint for future approval | `scripts/frontier/e97_async_diloco_train.py` or an equivalent real trainer wired into the launcher |
| Output root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco` |
| Run root pattern | `$OUTPUT_ROOT/$SCALEOUT_VARIANT/$RUN_DATE/$SLURM_JOB_ID-$RUN_STAMP` |
| Metrics JSON | `$RUN_ROOT/artifacts/async_diloco_e97_256n_metrics.json` |
| Run directory | `$RUN_ROOT/async_run` |
| Batch size | `4` |
| Chunk size | `2048` |
| DiLoCo K / local steps | `40` |
| Workers per node | `8` |
| Worker count | `2048` implied by `256 * 8` |
| Local quorum | `8` |
| Global quorum | `240` |
| Local timeout | `120` seconds |
| Global timeout | `240` seconds |
| Staleness policy | `reject_stale` |
| Update weighting | `tokens` |
| Transport mode | `cray-mpich-gpu-aware-p2p` |
| Group aggregator size | `16` |
| Dense update storage | `0` |
| Tokens per local step | `16,777,216` |
| Tokens per DiLoCo generation | `671,088,640` |
| Generation manifests | every generation |
| Recovery cadence | every `5` generations or `600` seconds, whichever comes first |
| Export cadence | every `45` generations or `3600` seconds, whichever comes first |
| Finalization buffer | `1200` seconds |
| Chain update on failure | `0` |
| Production latest guard | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt` |
| Latest policy | run-local `async_run/latest.json` with external chain `latest.pt` as guard-only |
| Approval gate | wrapper refuses to train unless `ASYNC_DILOCO_HUMAN_APPROVED=1` |

## Expected tokens and node-hours

Requested node-hours are exact from the wrapper: **3072.0 node-hours**.

Token accounting from the proposed geometry:

- Per local step: `16,777,216` tokens.
- Per DiLoCo generation: `40 * 16,777,216 = 671,088,640` tokens.
- Per recovery interval of 5 generations: `3,355,443,200` tokens.
- Per export interval of 45 generations: `30,198,988,800` tokens.

Because no real async trainer metrics exist, expected total tokens over the
700-minute training budget remain an **estimate only**. A linearized estimate
from the prior 8-node K40 no-DDP run's median global throughput of about
`164,697` tokens/sec would scale to roughly `5.27M` tokens/sec at 256 nodes and
about **221B tokens** over `700` minutes:

```text
164,697 tokens/s * (256 / 8) * 700 min * 60 s/min
= 221,356,608,000 tokens
```

That estimate is not production evidence. It assumes near-linear scaling from a
different non-async run family and should be treated only as a budgeting bound.
Under the proposed async geometry it corresponds to about `330` DiLoCo
generations (`221.36B / 671.09M`).

## Remaining risks

- **Hard blocker:** the required real trainer entrypoint is missing in the
  current checkout.
- **No real metrics:** there is no finite loss, real tokens/sec, quorum,
  deferral, recovery, export, or finalization evidence.
- **8-node smoke not run:** the 8-node production-wrapper attempt did not start,
  so it does not reduce runtime or transport risk.
- **Synthetic fallback risk:** the wrapper default still points to
  `scripts/frontier/async_diloco_e97_multinode.py`, which delegates to the
  debug harness. Future validation must force or default to a real trainer and
  reject synthetic metrics as production evidence.
- **Distributed semantics risk:** prior quality-pass review found the intended
  dependency-branch helper closer to real token work, but not yet proven as
  real inter-node async transport.
- **Long-job checkpoint risk:** quality-pass review flagged that long-job
  cadence/finalization knobs had not been proven wired into the real trainer
  command path.
- **Scheduler risk:** even the 8-node debug-QOS attempt could remain pending
  long enough to require explicit cancellation; a 256-node production request
  needs active queue monitoring.
- **Latest-chain safety:** the external production `latest.pt` must remain
  guard-only until a validated checkpoint is produced and separately approved.

## Monitor and cancel plan for any future approved submit

Do not use this plan until the real trainer exists, the ladder passes, and the
user gives fresh approval after reviewing the updated decision.

Pre-submit checks:

- Confirm `scripts/frontier/e97_async_diloco_train.py` or the approved real
  entrypoint exists in the submit checkout.
- Confirm the launcher command records the refreshed seed path exactly:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`.
- Record the production latest guard symlink target, metadata, and checksum
  before submit.
- Submit only through a WG-tracked task with the exact command and
  `ASYNC_DILOCO_HUMAN_APPROVED=1`.

Early monitor cadence:

- Immediately after submit: record job id, `squeue`, partition/QOS, reason or
  node list, and estimated start if pending.
- While pending: check at least every 15-30 minutes; cancel if the job would
  begin outside the intended supervised window.
- First 10 minutes after start: tail stdout/stderr and run artifacts; verify
  seed path, real entrypoint path, node count, local/global quorum, K, batch
  size, chunk size, output root, and production latest guard.
- First generation: require finite real loss, positive real tokens/sec,
  accepted/deferred/stale/timed_out/failed update counts, local/global quorum
  distributions, and generation manifest output.
- First recovery/export windows: require recovery checkpoint records by
  5 generations or 600 seconds, and export checkpoint records by 45 generations
  or 3600 seconds if the run lives that long.

Cancel criteria:

- Wrong or missing real trainer entrypoint.
- Seed path differs from the refreshed `step_1065000/latest.pt`.
- Output root differs from the planned async production root or writes into an
  unintended chain path.
- External production `latest.pt` guard changes unexpectedly.
- No finite real loss or no positive real tokens/sec after the first completed
  generation.
- Quorum accounting missing or global quorum repeatedly fails with no forward
  progress beyond the first monitored window.
- Repeated stale/timed-out/failed update spikes that prevent generation
  advancement.
- Missing recovery checkpoint evidence after the configured recovery cadence.
- Trainer falls back to synthetic/debug metrics.
- Slurm, ROCm, filesystem, or Python/runtime failures that leave no credible
  path to valid checkpoint finalization inside the 12-hour allocation.

## Required next step

Run another intermediate validation ladder, not the 256-node job:

1. Add or restore the real async E97 trainer entrypoint and wire the production
   launcher to reject synthetic/prototype entrypoints for production evidence.
2. Re-run the 1-node and 2-node real-trainer ladder until it produces finite
   real loss, real tokens/sec, quorum distributions, update outcome counts,
   recovery records, and export/finalization records.
3. Re-run the 8-node, 20-minute production-wrapper validation only after the
   1n/2n ladder passes.
4. Prepare a new decision report from passing real metrics before requesting
   user approval for any 256-node submit.

## Validation checklist for this task

- Decision report is written and linked: **yes**, this file.
- Old synthetic `retry-refreshed-e97` is left paused/no-go or superseded:
  **left paused/no-go**; `wg show retry-refreshed-e97` reports
  `open (PAUSED)`.
- No 256-node job submitted without fresh user approval: **yes**, no Slurm
  submission was made by this task.
- If submit is recommended, ready-but-paused submit task is created:
  **not applicable** because recommendation is no-go.
