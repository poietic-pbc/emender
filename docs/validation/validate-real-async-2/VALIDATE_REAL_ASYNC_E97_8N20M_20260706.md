# Validate real async E97 8n20m production wrapper

Task: `validate-real-async-2`  
Date: 2026-07-06  
Checkout submitted from: `853073601859ddf46008d65cdd0c48ee22ff1804`  
Launcher: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`  
Required real entrypoint override: `scripts/frontier/e97_async_diloco_train.py`

## Verdict

**No-go for preparing the 256n12h production run.**

An 8-node, 20-minute production-wrapper validation was attempted with the
production launcher and the required real-trainer entrypoint override, but the
job did not receive an allocation during this evaluation window. The Slurm job
was accepted as job `4948004`, remained pending on `batch/debug` with reason
`Priority`, and was canceled before it could start so that an unattended 8-node
job would not begin after this task report was written.

No training logs, wrapper artifacts, metrics JSON, quorum records, checkpoint
records, or elapsed node-hours were produced by job `4948004` because it never
started. Separately, the checkout still does not contain the required real
trainer path `scripts/frontier/e97_async_diloco_train.py`; the upstream 1n/2n
ladder already proved that the production wrapper exits before training when
this path is required.

## 8-node attempt

Exact submission command:

```bash
sbatch -A bif148 -p batch -q debug -N 8 -t 00:20:00 \
  --job-name async-diloco-e97-real-8n20m \
  --output logs/frontier/async_diloco_e97/%x-%j.out \
  --error logs/frontier/async_diloco_e97/%x-%j.err \
  --export=ALL,WG_TASK_ID=validate-real-async-2,ASYNC_DILOCO_HUMAN_APPROVED=1,ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py,ASYNC_NODE_COUNT=8,ASYNC_LOCAL_QUORUM=6,ASYNC_GLOBAL_QUORUM=6,OUTPUT_ROOT=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-714/docs/validation/validate-real-async-2/slurm,SCALEOUT_VARIANT=real_async_e97_8n20m_validation,TRAIN_MINUTES=20,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=2.666667 \
  scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Safe overrides from the 256-node production defaults:

- Slurm nodes/time/QOS: `-N 8`, `-t 00:20:00`, `-q debug`.
- `ASYNC_NODE_COUNT=8`.
- `ASYNC_LOCAL_QUORUM=6`, matching a 6-of-8 local quorum.
- `ASYNC_GLOBAL_QUORUM=6`, matching `ceil(2/3 * 8)`.
- `TRAIN_MINUTES=20`.
- `REQUESTED_WALLTIME=00:20:00`.
- `REQUESTED_NODE_HOURS=2.666667`.
- Isolated output root and variant under this validation task.
- `ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py` to avoid
  accepting the synthetic `2n8n_debug` harness as evidence.

Settings intentionally left at production wrapper defaults:

- Seed/latest checkpoint.
- Batch size `4`.
- Chunk size `2048`.
- DiLoCo `K=40`.
- Checkpoint/export cadence.
- Finalization buffer.
- Runtime/Python setup.
- Command construction inside the wrapper.

## Scheduler evidence

Submission response:

```text
Submitted batch job 4948004
```

Queue state while pending:

```text
JOBID|NAME|STATE|TIME|NODES|NODELIST(REASON)|QOS|PARTITION|SUBMIT_TIME
4948004|async-diloco-e97-real-8n20m|PENDING|0:00|8|(Priority)|debug|batch|2026-07-06T12:44:55
```

Accounting after cancellation:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|AllocTRES
4948004|async-diloco-e97-real-8n20m|batch|debug|CANCELLED by 19032|0:0|00:00:00|8|
```

Elapsed node-hours: `8 * 0 / 3600 = 0.000000`.  
Requested node-hours: `2.666667`.

No stdout/stderr log files exist for this job because Slurm never started the
batch script:

- Expected stdout: `logs/frontier/async_diloco_e97/async-diloco-e97-real-8n20m-4948004.out`
- Expected stderr: `logs/frontier/async_diloco_e97/async-diloco-e97-real-8n20m-4948004.err`

No run artifact directory exists under:

```text
docs/validation/validate-real-async-2/slurm/real_async_e97_8n20m_validation/
```

## Real trainer availability

The production wrapper's stable default entrypoint is not sufficient for this
task, because it currently delegates to the debug implementation:

```python
from async_diloco_e97_2n8n_debug import main
```

The task required the real async trainer, so the submission forced:

```text
ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py
```

That required file is still absent from this checkout. The upstream
`validate-real-async` 1n/2n ladder submitted the same production wrapper path
with the same required real entrypoint and both jobs failed before training with
wrapper exit code `65`:

```text
Launch cannot start because ASYNC_ENTRYPOINT is missing: scripts/frontier/e97_async_diloco_train.py
```

Because job `4948004` did not start, this task did not reproduce the wrapper
exit in an 8-node allocation. The missing entrypoint remains a hard blocker for
accepting any real-training evidence.

## Metrics and quorum evidence

No `async_diloco_e97_256n_metrics.json` was produced for job `4948004`.

The following required production-validation evidence is therefore absent:

- Finite real loss.
- Real tokens/sec.
- Accepted/deferred/stale/timed_out/failed update counts.
- Local quorum distribution.
- Global quorum distribution.
- Recovery checkpoint records.
- Export/finalization checkpoint records.
- Generation manifest records.

No synthetic `async_diloco_e97_2n8n_debug.py` metrics were accepted as evidence.

## Production latest guard

Guard path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Guard target and metadata after the 8-node attempt:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt' -> '/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt'|231|1783092774|2026-07-03 11:32:54.000000000 -0400
6cdca7edcf208d96c99f7be5f58b996216eba81de553975c58f04a5bdb5563a3  /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

This matches the guard state recorded by the upstream 1n/2n validation. The
production latest guard is unchanged.

## Validation checklist status

- Uses production launcher and real async trainer; no synthetic debug wrapper:
  **partially met**. The production launcher was used and the real-trainer path
  was forced, but the required trainer file is absent and the job never ran.
- Slurm job id, exact sbatch command, queue/QOS, logs, elapsed time, and
  node-hours recorded: **partially met**. Job id `4948004`, exact command,
  queue/QOS, cancellation state, elapsed `00:00:00`, and node-hours `0.000000`
  are recorded. Logs do not exist because the job never started.
- Env/command artifacts show real training branch and refreshed seed
  `latest.pt`: **not met for the 8-node attempt**. No wrapper artifacts were
  produced because the job never started. Upstream 1n/2n artifacts show the
  refreshed seed and required real-trainer path.
- Metrics show finite real loss, real tokens/sec, accepted/deferred/stale/
  timed_out/failed counts, local/global quorum distribution, checkpoint records:
  **not met**. No metrics or checkpoint records were produced.
- Production latest guard unchanged: **met**.
- Clear pass/no-go for preparing 256n12h production run: **met, no-go**.

## Required next steps

Do not prepare or submit the 256n12h production run from this evidence.

Before repeating the 8n20m validation:

1. Add or restore the real async trainer entrypoint
   `scripts/frontier/e97_async_diloco_train.py`.
2. Re-run the 1n/2n real-trainer ladder through the production wrapper until it
   produces finite real loss, real tokens/sec, quorum distributions, and
   checkpoint records.
3. Re-submit the 8n20m production-wrapper validation only after the 1n/2n real
   path passes.
