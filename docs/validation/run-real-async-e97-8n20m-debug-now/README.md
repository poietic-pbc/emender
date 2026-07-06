# Real async E97 8n20m debug run

Task: `run-real-async-e97-8n20m-debug-now`
Date: 2026-07-06
Checkout: `fd5605e9ef511abb98ce7276ded3d0a258a9fff3`
Launcher: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
Trainer: `scripts/frontier/e97_async_diloco_train.py`

## Verdict

**No-go for preparing the larger 256n12h run.**

The required 8-node debug-QOS Slurm job was submitted from merged `main` /
`origin/main` at `fd5605e9ef511abb98ce7276ded3d0a258a9fff3` and ran on the
requested 8 Frontier nodes, but it timed out at the 20-minute debug walltime
before the real trainer returned a metrics summary, run-local `latest.json`, or
checkpoint/finalization artifacts.

Observed failure class: **runtime/time-budget**. This was not an `sbatch`
submission failure, scheduler pending failure, missing trainer file, missing
seed/data preflight failure, or Python traceback in the captured logs. The
production wrapper emitted the runtime environment and command, then Slurm
cancelled the batch step due to the time limit before trainer completion.

## Preflight

- Current branch: `main`
- `HEAD`: `fd5605e9ef511abb98ce7276ded3d0a258a9fff3`
- `origin/main`: `fd5605e9ef511abb98ce7276ded3d0a258a9fff3`
- Required files existed before submission:
  - `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
  - `scripts/frontier/e97_async_diloco_train.py`

## Submission

Exact command:

```bash
sbatch -A bif148 -p batch -q debug -N 8 -t 00:20:00 --job-name async-diloco-e97-real-8n-debug --output logs/frontier/async_diloco_e97/%x-%j.out --error logs/frontier/async_diloco_e97/%x-%j.err --export=ALL,WG_TASK_ID=run-real-async-e97-8n20m-debug-now,TASK_ID=run-real-async-e97-8n20m-debug-now,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_HUMAN_APPROVED=1,ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_NODE_COUNT=8,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=6,ASYNC_GLOBAL_TIMEOUT_S=1800,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now,SCALEOUT_VARIANT=real_async_e97_8n20m_debug,TRAIN_MINUTES=20,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=2.666667,BATCH_SIZE=4,CHUNK_SIZE=2048,DILOCO_K=40 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Submission result:

```text
Submitted batch job 4949665
```

The memory-feasible debug override used the merged fix branch's documented
shape: one real worker per node and local quorum 1. The 8-node global quorum
was set to `ceil(2/3 * 8) = 6`.

## Slurm result

Accounting from `sacct`:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Start|End
4949665|async-diloco-e97-real-8n-debug|batch|debug|TIMEOUT|0:0|00:20:20|00:20:00|8|frontier[08378-08379,08384,08386-08387,08389-08390,08392]|billing=896,cpu=896,energy=4682744,mem=4000G,node=8|2026-07-06T16:41:16|2026-07-06T17:01:36
4949665.batch|batch|||CANCELLED|0:15|00:20:20||1|frontier08378|cpu=56,mem=500G,node=1|2026-07-06T16:41:16|2026-07-06T17:01:36
4949665.extern|extern|||COMPLETED|0:0|00:20:20||8|frontier[08378-08379,08384,08386-08387,08389-08390,08392]|billing=896,cpu=896,mem=4000G,node=8|2026-07-06T16:41:16|2026-07-06T17:01:36
```

- Job id: `4949665`
- Queue/partition/QOS: `batch` / `debug`
- State: `TIMEOUT`
- Exit code: `0:0` at job level, batch step `CANCELLED` with `0:15`
- Elapsed: `00:20:20`
- Requested walltime: `00:20:00`
- Requested node-hours: `2.666667`
- Elapsed node-hours: `8 * 1220 / 3600 = 2.711111`
- Nodes: `frontier[08378-08379,08384,08386-08387,08389-08390,08392]`

Stdout/stderr:

- `logs/frontier/async_diloco_e97/async-diloco-e97-real-8n-debug-4949665.out`
- `logs/frontier/async_diloco_e97/async-diloco-e97-real-8n-debug-4949665.err`

Final stderr line:

```text
[2026-07-06T17:01:36.045] error: *** JOB 4949665 ON frontier08378 CANCELLED AT 2026-07-06T17:01:36 DUE TO TIME LIMIT ***
```

## Wrapper and environment evidence

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z
```

Wrapper artifacts:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/artifacts/command.txt`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/artifacts/env.txt`

Key fields captured in `env.txt`:

```text
seed_latest_path=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
e97_checkpoint=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
data=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
async_entrypoint=scripts/frontier/e97_async_diloco_train.py
python_bin=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
async_node_count=8
async_worker_count_per_node=1
async_worker_count=8
async_local_quorum=1
async_global_quorum=6
batch_size=4
chunk_size=2048
diloco_k=40
non_production_debug=1
synthetic_token_stream=0
git_commit=fd5605e9ef511abb98ce7276ded3d0a258a9fff3
```

The emitted trainer command used:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py ... --node-count 8 --worker-count 8 --local-quorum 1 --global-quorum 6 ... --batch-size 4 --chunk-size 2048 ... --steps 40
```

## Metrics, quorum, and checkpoint behavior

No metrics JSON was produced:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/artifacts/async_diloco_e97_256n_metrics.json
```

does not exist.

No run-local latest file was produced:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/async_run/latest.json
```

does not exist.

Artifact listing at collection time contained only:

```text
2026-07-06T16:41:24.0000000000 1553 /lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/artifacts/command.txt
2026-07-06T16:41:42.0000000000 7236 /lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/run-real-async-e97-8n20m-debug-now/real_async_e97_8n20m_debug/20260706/4949665-20260706T204118Z/artifacts/env.txt
```

Consequences:

- No real training loss or tokens/sec metrics were recorded.
- No local quorum distribution was recorded beyond the configured local quorum.
- No global quorum distribution was recorded beyond the configured global quorum.
- No recovery checkpoint, export checkpoint, finalization checkpoint, or
  generation manifest was recorded.
- The failure happened after wrapper preflight and command emission, before the
  trainer returned its final JSON summary.

## Root cause

The concrete observed root cause is **20-minute debug walltime exhaustion before
real trainer completion**.

The logs do not show a Python exception. The wrapper completed preflight far
enough to record the activated runtime, exact command, seed, data path, and
configuration. The real trainer process did not emit its final JSON summary
before Slurm cancelled the batch step. Since `e97_async_diloco_train.py` writes
the configured metrics JSON only after `run_real_async_diloco(...)` returns,
the timeout explains the absent metrics/checkpoint artifacts.

## Validation checklist

- 8-node debug-QOS Slurm job submitted: **met** (`4949665`).
- Exact `sbatch` command recorded: **met** (`sbatch_command.txt` and above).
- Job id, queue/QOS, elapsed time, node-hours, node list, stdout/stderr paths
  recorded: **met**.
- Artifacts show real trainer on merged main, activated `PYTHON_BIN`, refreshed
  `latest.pt` seed, real data, `batch_size=4`, `chunk_size=2048`, `DILOCO_K=40`:
  **met**.
- Metrics show real training loss/tokens/sec and quorum distributions:
  **not met**. No metrics were produced because the job timed out before
  trainer completion.
- Checkpoint/finalization behavior recorded: **met as negative evidence**. No
  checkpoint/finalization artifacts were produced before timeout.
- Clear pass/no-go for larger 256n12h run: **met, no-go**.

