# Async DiLoCo E97 256n12h Human Approval Gate Review

Task: `async-diloco-submit-256n12h-human-approval`
Date: 2026-07-05
Reviewer: `agent-648`
Conclusion: **NO-GO; no production Slurm job submitted.**

## Readiness Grade

- Calibrated grade: `0.46`
- Confidence: `0.82`
- Rubric underspecified: `false`

Dimension scores:

| Dimension | Score | Rationale |
|---|---:|---|
| Upstream launch package evidence | `0.85` | Dependency `async-diloco-prepare-256n12h` is done and cites launch report `reports/frontier/async-diloco-prepare-256n12h-20260705.md` plus wrapper `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` from upstream commit `6f21700`. |
| Explicit human approval record | `0.00` | No task message or WG log entry was found that explicitly approves this exact 256-node 12-hour production launch after the prepared package was available. |
| Checkpoint/latest policy review | `0.80` | Prepared package reviews measured checkpoint write duration, size, generation duration, overhead, and selects `5` generations or `600s` recovery cadence with hourly exports. Approval itself is absent, so this cannot be accepted as human approval. |
| Queue/allocation and operational state | `0.55` | Current queue check found no active jobs for the user at review time. Recent `sacct` entries show only prior 1/2/8/32/64-node async jobs, not a 256-node submission. |
| Submission and monitoring evidence | `0.00` | No `sbatch` submission was made; therefore no job id, production logs, actual node-hours, or monitor task exists for this production launch. |
| Safety/no-submit compliance | `1.00` | The gate did not set `ASYNC_DILOCO_HUMAN_APPROVED=1` and did not submit the production job. |

## Evidence Reviewed

The upstream prepare task is complete:

- `async-diloco-prepare-256n12h` status: done.
- Prepare commit: `6f21700` on branch `wg/agent-644/async-diloco-prepare-256n12h`.
- Prepared report: `reports/frontier/async-diloco-prepare-256n12h-20260705.md`.
- Prepared wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`.
- Prepare validation log: `bash -n scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` passed and no production job was submitted.

The package target is:

- Job size: `256` Frontier nodes.
- Walltime: `12:00:00`.
- Requested node-hours: `3,072`.
- Training target: `E97_1.3B_step483000_async_diloco_256n12h_20260705`.
- Source checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt`.
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_async_quorum_b4_k40_256n12h/`.

## Checkpoint Policy Review

The prepared package uses the required scale-adaptive checkpoint policy rather
than a fixed 20-30 minute recovery cadence:

- Generation manifests: every DiLoCo generation.
- Recovery checkpoints: whichever fires first, `5` generations or `600s`.
- Export checkpoints: whichever fires first, `45` generations or `3600s`.
- Finalization buffer: initial `1200s`; then adapt after real 256-node write evidence.
- Production chain guard: `CHAIN_UPDATE_ON_FAILURE=0`.

Measured evidence cited by the launch package:

| Evidence | Generation duration | Checkpoint write duration | Checkpoint size | Percent overhead |
|---|---:|---:|---:|---:|
| 32n config job `4944228` | `1.2291s` | `0.000167s` | manifest `8,473B`, recovery `8,291B`, total `16,764B` | `0.01362%` |
| 64n config job `4944237` | `1.2255s` | `0.000170s` | manifest `8,482B`, recovery `8,297B`, total `16,779B` | `0.01384%` |

The prepared package correctly cautions that these are metadata recovery records,
not full 7.7GB model exports, and requires immediate monitoring of real 256-node
write duration, size, percent overhead, and latest advancement.

## Queue And Allocation Snapshot

At review time, `squeue -u "$USER"` returned no active queued or running jobs.
`sacct -u "$USER" -S 2026-07-05T00:00:00 -X` showed prior async debug/config
jobs through `4944237`, including the 32-node and 64-node config jobs used as
evidence, but no 256-node async production job.

Recent relevant jobs:

| Job ID | Name | State | Nodes | Elapsed |
|---|---|---:|---:|---:|
| `4944217` | `async-diloco-e97-32n` | `OUT_OF_MEMORY` | `32` | `00:01:40` |
| `4944228` | `async-diloco-e97-32n-r1` | `COMPLETED` | `32` | `00:00:50` |
| `4944237` | `async-diloco-e97-64n` | `COMPLETED` | `64` | `00:00:55` |

## Blocking Findings

1. Explicit 256-node human approval is absent.

   I found no message or WG log entry approving the exact launch package, job
   size, duration, node-hour estimate/cap, checkpoint/latest policy, monitoring
   plan, run directory, and current queue/allocation state after
   `async-diloco-prepare-256n12h` completed.

2. The prepared wrapper is approval-gated and was not submitted.

   The wrapper exits unless `ASYNC_DILOCO_HUMAN_APPROVED=1`. This task did not
   set that variable and did not run `sbatch`.

3. The async multi-node entrypoint remains missing from this checkout.

   The prepared package expects `scripts/frontier/async_diloco_e97_2n8n_debug.py`.
   The prepare report states this entrypoint is referenced by 32/64-node evidence
   command files but is not present in the checkout. The wrapper will fail after
   approval unless the entrypoint is restored or replaced.

4. No submitted-job evidence exists because launch was not authorized.

   There is no production job id, submitted command, stdout/stderr path, actual
   node-hours, running monitor task, cancel-trigger record, or production
   latest/checkpoint verification for a 256-node job.

## Required Approval Text Before Any Future Submission

Before any production submission, WG must record an explicit human approval that
names all of the following:

- Launch package: upstream prepare report and wrapper from commit `6f21700`, or
  a later replacement with differences listed.
- Job size and duration: `256` nodes for `12:00:00`.
- Node-hour estimate/cap: `3,072` requested node-hours, plus any acceptable cap.
- Run directory/root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_async_quorum_b4_k40_256n12h/`.
- Checkpoint/latest policy: generation manifests every generation; recovery
  checkpoints every `5` generations or `600s`, whichever fires first; export
  checkpoints every `45` generations or `3600s`; `CHAIN_UPDATE_ON_FAILURE=0`;
  first real recovery/export write measured before unattended operation.
- Measured cadence evidence: 32n and 64n generation duration, checkpoint write
  duration, checkpoint size, percent overhead, and the chosen recovery cadence
  in both generations and wall-clock time.
- Monitoring plan: machine-readable effective quorum, accepted/stale/timed-out/
  failed/drop counts, generation/merge/checkpoint timing, checkpoint overhead,
  loss windows, latest advancement behavior, cancel criteria, and monitoring
  task id.
- Queue/allocation state at approval time.
- Resolution for the missing async multi-node entrypoint.

## Follow-Up

No monitoring task was created because there is no production job to monitor.
The appropriate follow-up is a human decision on this gate after the missing
entrypoint and approval text are resolved. A future submitting agent must record
any package differences before launch and must create a monitor task immediately
after a successful `sbatch`.

I attempted to create a WG follow-up task named
`Resolve async DiLoCo 256n entrypoint and approval preflight`, including retries
with `--independent`, `--place-near`, and `--no-place`. WG rejected all attempts
with `Task would be at user-visible depth 11 (configured max_task_depth: 8)`.
Therefore the follow-up is recorded here as an artifact requirement rather than
as a new WG node.
