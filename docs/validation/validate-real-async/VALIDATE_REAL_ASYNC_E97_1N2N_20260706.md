# Validate real async E97 1n/2n debug ladder

Task: `validate-real-async`  
Date: 2026-07-06  
Checkout submitted from: `6aff28e520c81da0c40b97e611545032551505b8`  
Launcher: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`  
Required real entrypoint override: `scripts/frontier/e97_async_diloco_train.py`

## Verdict

**No-go for `validate-real-async-2` / 8n20m production-wrapper validation.**

Both bounded Frontier debug-QOS jobs were submitted through the production
wrapper path, with only scale/time/quorum/output overrides and with
`ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py` to avoid accepting
the synthetic `2n8n_debug` harness as evidence. Both jobs failed before
training with wrapper exit code `65`:

```text
Launch cannot start because ASYNC_ENTRYPOINT is missing: scripts/frontier/e97_async_diloco_train.py
```

Therefore the ladder did not prove real token training, real loss/tokens/sec,
local/global quorum behavior, checkpoint recovery/finalization, or manifest
behavior. The production latest guard was unchanged.

## Slurm attempts

### 1-node debug leg

Exact submission command:

```bash
sbatch -A bif148 -p batch -q debug -N 1 -t 00:20:00 \
  --job-name async-diloco-e97-real-1n-debug \
  --output logs/frontier/async_diloco_e97/%x-%j.out \
  --error logs/frontier/async_diloco_e97/%x-%j.err \
  --export=ALL,WG_TASK_ID=validate-real-async,TASK_ID=validate-real-async-1n,ASYNC_DILOCO_HUMAN_APPROVED=1,ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py,ASYNC_NODE_COUNT=1,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=1,ASYNC_LOCAL_TIMEOUT_S=60,ASYNC_GLOBAL_TIMEOUT_S=60,OUTPUT_ROOT=/lustre/orion/bif148/scratch/erikgarrison/emender/docs/validation/validate-real-async/slurm,SCALEOUT_VARIANT=real_async_e97_1n_debug,TRAIN_MINUTES=20,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.333333,GENERATION_MANIFEST_EVERY=1,RECOVERY_EVERY_GENERATIONS=1,RECOVERY_EVERY_SECONDS=300,EXPORT_EVERY_GENERATIONS=1,EXPORT_EVERY_SECONDS=600 \
  scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|AllocTRES
4947951|async-diloco-e97-real-1n-debug|batch|debug|FAILED|65:0|00:00:18|1|billing=112,cpu=112,energy=9285,mem=500G,node=1
4947951.batch|batch|||FAILED|65:0|00:00:18|1|cpu=56,mem=500G,node=1
4947951.extern|extern|||COMPLETED|0:0|00:00:19|1|billing=112,cpu=112,mem=500G,node=1
```

Elapsed node-hours: `1 * 18 / 3600 = 0.005000`.
Requested node-hours: `0.333333`.

Logs:

- `logs/frontier/async_diloco_e97/async-diloco-e97-real-1n-debug-4947951.out`
- `logs/frontier/async_diloco_e97/async-diloco-e97-real-1n-debug-4947951.err`

Run artifacts:

- `docs/validation/validate-real-async/slurm/real_async_e97_1n_debug/20260706/4947951-20260706T163104Z/artifacts/env.txt`
- `docs/validation/validate-real-async/slurm/real_async_e97_1n_debug/20260706/4947951-20260706T163104Z/artifacts/command.txt`

### 2-node debug leg

Exact submission command:

```bash
sbatch -A bif148 -p batch -q debug -N 2 -t 00:20:00 \
  --job-name async-diloco-e97-real-2n-debug \
  --output logs/frontier/async_diloco_e97/%x-%j.out \
  --error logs/frontier/async_diloco_e97/%x-%j.err \
  --export=ALL,WG_TASK_ID=validate-real-async,TASK_ID=validate-real-async-2n,ASYNC_DILOCO_HUMAN_APPROVED=1,ASYNC_ENTRYPOINT=scripts/frontier/e97_async_diloco_train.py,ASYNC_NODE_COUNT=2,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=1,ASYNC_LOCAL_TIMEOUT_S=60,ASYNC_GLOBAL_TIMEOUT_S=60,OUTPUT_ROOT=/lustre/orion/bif148/scratch/erikgarrison/emender/docs/validation/validate-real-async/slurm,SCALEOUT_VARIANT=real_async_e97_2n_debug,TRAIN_MINUTES=20,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.666667,GENERATION_MANIFEST_EVERY=1,RECOVERY_EVERY_GENERATIONS=1,RECOVERY_EVERY_SECONDS=300,EXPORT_EVERY_GENERATIONS=1,EXPORT_EVERY_SECONDS=600 \
  scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Slurm accounting:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|AllocTRES
4947962|async-diloco-e97-real-2n-debug|batch|debug|FAILED|65:0|00:00:13|2|billing=224,cpu=224,energy=12376,mem=1000G,node=2
4947962.batch|batch|||FAILED|65:0|00:00:13|1|cpu=56,mem=500G,node=1
4947962.extern|extern|||COMPLETED|0:0|00:00:13|2|billing=224,cpu=224,mem=1000G,node=2
```

Elapsed node-hours: `2 * 13 / 3600 = 0.007222`.
Requested node-hours: `0.666667`.

Logs:

- `logs/frontier/async_diloco_e97/async-diloco-e97-real-2n-debug-4947962.out`
- `logs/frontier/async_diloco_e97/async-diloco-e97-real-2n-debug-4947962.err`

Run artifacts:

- `docs/validation/validate-real-async/slurm/real_async_e97_2n_debug/20260706/4947962-20260706T163252Z/artifacts/env.txt`
- `docs/validation/validate-real-async/slurm/real_async_e97_2n_debug/20260706/4947962-20260706T163252Z/artifacts/command.txt`

Total elapsed node-hours consumed by failed ladder attempts:
`0.005000 + 0.007222 = 0.012222`.

## Environment and wrapper evidence

Both jobs recorded wrapper environment before failing. The relevant fields were:

```text
seed_latest_path=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
e97_checkpoint=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
async_entrypoint=scripts/frontier/e97_async_diloco_train.py
env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
emender_conda_env=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
python_bin=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
batch_size=4
chunk_size=2048
diloco_k=40
```

The activated Python/runtime fields were also recorded:

```text
python_executable=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
python_version=3.12.13
torch.__version__=2.10.0+rocm7.1
torch.version.hip=7.1.25424
triton.__version__=3.6.0
```

## Real training metrics

No real training metrics were produced. This is expected for the observed
failure mode: the wrapper rejected the launch before invoking Python training
because `scripts/frontier/e97_async_diloco_train.py` is absent.

No `async_diloco_e97_256n_metrics.json` files exist under either run directory.
This avoids the forbidden synthetic `2n8n_debug` evidence but also means the
required real loss/tokens/sec criteria are unmet.

## Quorum, deferred generations, checkpoint, and finalization

No local quorum metrics, global quorum metrics, deferred generations, recovery
checkpoints, finalization checkpoints, or generation manifests were produced.
The jobs failed before the trainer process could start.

The wrapper did create only the pre-launch artifact files:

```text
docs/validation/validate-real-async/slurm/real_async_e97_1n_debug/20260706/4947951-20260706T163104Z/artifacts/command.txt
docs/validation/validate-real-async/slurm/real_async_e97_1n_debug/20260706/4947951-20260706T163104Z/artifacts/env.txt
docs/validation/validate-real-async/slurm/real_async_e97_2n_debug/20260706/4947962-20260706T163252Z/artifacts/command.txt
docs/validation/validate-real-async/slurm/real_async_e97_2n_debug/20260706/4947962-20260706T163252Z/artifacts/env.txt
```

## Production latest guard

Guard path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Target and metadata after the failed ladder:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt' -> '/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt'|231|1783092774|2026-07-03 11:32:54.000000000 -0400
```

Checkpoint content checksum captured before and after the submissions:

```text
6cdca7edcf208d96c99f7be5f58b996216eba81de553975c58f04a5bdb5563a3
```

The production latest guard is unchanged.

## Validation checklist status

- Slurm job ids, exact sbatch commands, queues/QOS, logs, elapsed time, and
  node-hours recorded: **met**.
- Stdout/env artifacts show refreshed seed, real async trainer script,
  activated python, `batch_size=4`, `chunk_size=2048`, `K=40`: **partially met**.
  The artifacts show the refreshed seed, activated Python, and required geometry,
  and show the intended real trainer path. The path is missing in the checkout.
- Metrics show real training loss/tokens/sec, not synthetic loss: **not met**.
  No metrics were produced because the wrapper failed before training.
- Local/global quorum metrics and deferred generations recorded: **not met**.
  No trainer process started.
- Recovery/finalization checkpoint records exist and production latest guard is
  unchanged: **partially met**. No recovery/finalization records exist; the
  production latest guard is unchanged.
- Clear pass/no-go for 8n20m production-wrapper validation: **met, no-go**.

## Required next step before 8n20m

Do not run the 8-node production-wrapper validation until the current checkout
contains the real async trainer entrypoint and module, and the production wrapper
can start that path without delegating to `async_diloco_e97_2n8n_debug`.
