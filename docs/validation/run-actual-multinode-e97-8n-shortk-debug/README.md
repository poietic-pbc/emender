# Actual Multinode E97 8n Short-K Debug

Task: `run-actual-multinode-e97-8n-shortk-debug`

Date: 2026-07-06

Conclusion: **pass for actual 8-node launcher validation at K1; no-go for restoring K40 yet.**

The newly fixed production E97 launcher was exercised through Slurm debug QOS with real data, refreshed `latest.pt`, synthetic stream disabled, and one trainer process per node. The first K10 attempt failed with a real trainer non-finite-loss path before node payload serialization. A bounded K1 retry completed and produced finite real metrics, quorum artifacts, node heartbeats, node update artifacts, and run-local checkpoint/latest finalization.

## Inputs

- Seed checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Real data: `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- Entrypoint: `scripts/frontier/e97_async_diloco_train.py`
- Wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- Synthetic stream: disabled, `ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0`
- Actual multinode path: enabled, `ASYNC_ACTUAL_MULTINODE_FILE_QUORUM=1`
- One trainer per node: `ASYNC_WORKER_COUNT_PER_NODE=1`, `ASYNC_WORKER_COUNT=8`
- Device: `ASYNC_DILOCO_DEVICE=cuda`

## Attempt 1: K10

Slurm job: `4949872`

Exact submit command:

```bash
sbatch --parsable -A bif148 -p batch -q debug -N 8 -t 00:20:00 --ntasks-per-node=1 --cpus-per-task=56 -J async-e97-8n-k10-debug --export=ALL,WG_TASK_ID=run-actual-multinode-e97-8n-shortk-debug,TASK_ID=run-actual-multinode-e97-8n-shortk-debug,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_ACTUAL_MULTINODE_FILE_QUORUM=1,ASYNC_NODE_COUNT=8,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_WORKER_COUNT=8,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=6,DILOCO_K=10,GENERATIONS=1,BATCH_SIZE=1,CHUNK_SIZE=2048,ASYNC_GLOBAL_TIMEOUT_S=900,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=2.666667,TRAIN_MINUTES=20,FINALIZATION_BUFFER_SECONDS=60,RECOVERY_EVERY_GENERATIONS=1,RECOVERY_EVERY_SECONDS=60,EXPORT_EVERY_GENERATIONS=1,EXPORT_EVERY_SECONDS=60,ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT=1,ASYNC_DILOCO_DEVICE=cuda,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b1_k10_8n20m_debug,TOKENS_PER_LOCAL_STEP=20480,TOKENS_PER_DILOCO_GENERATION=20480 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Slurm accounting:

```text
4949872|async-e97-8n-k10-debug|FAILED|143:0|00:01:05|8||frontier[01910-01916,01918]
4949872.batch|batch|FAILED|143:0|00:01:05|1|1|frontier01910
4949872.extern|extern|COMPLETED|0:0|00:01:08|8|8|frontier[01910-01916,01918]
4949872.0|python|CANCELLED|0:15|00:00:50|8|8|frontier[01910-01916,01918]
```

Recorded launcher `srun` command:

```bash
srun -N 8 -n 8 --ntasks-per-node=1 /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py --run-id run-actual-multinode-e97-8n-shortk-debug-4949872-20260706T212709Z --checkpoint /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt --run-dir /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k10_8n20m_debug/20260706/4949872-20260706T212709Z/async_run --metrics-json /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k10_8n20m_debug/20260706/4949872-20260706T212709Z/artifacts/async_diloco_e97_256n_metrics.json --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --node-count 8 --worker-count 8 --local-quorum 1 --global-quorum 6 --generations 1 --local-steps 10 --timeout-s 900 --level E97 --params 1.3b --tokenizer p50k_base --batch-size 1 --chunk-size 2048 --lr 1e-3 --steps 10 --e97-chunk-size 32 --checkpoint-interval 64 --projection-chunk-size 256 --loss-chunk-size 256 --recovery-every-generations 1 --recovery-every-seconds 60 --export-every-generations 1 --export-every-seconds 60 --finalization-reserve-seconds 60 --bf16 --use-chunked-e97 --gradient-checkpointing --walltime-remaining-s 60 --dim 1792 --depth 11 --n-heads 216 --n-state 32 --n-groups 32 --n-slots 64 --expansion 1.0 --state-expansion 2 --gate-activation silu --linear-state 1 --mlp-ratio 2.2623 --mlp-multiple 64 --actual-multinode-file-quorum
```

Failure root cause:

```text
ValueError: non-finite metric value: nan
```

This occurred while `run_real_async_diloco_file_rank` attempted to serialize a node payload after real local training. Eight per-node heartbeat artifacts existed and reached `checkpoint_loaded`, but no `node_updates`, metrics JSON, or `latest.json` were produced for K10.

K10 artifact root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k10_8n20m_debug/20260706/4949872-20260706T212709Z
```

K10 per-node heartbeat evidence:

```text
node-00000.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00001.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00002.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00003.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00004.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00005.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00006.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
node-00007.heartbeat.json stage=checkpoint_loaded base_state_bytes=5506770496
```

## Attempt 2: K1

Slurm job: `4949891`

Exact submit command:

```bash
sbatch --parsable -A bif148 -p batch -q debug -N 8 -t 00:20:00 --ntasks-per-node=1 --cpus-per-task=56 -J async-e97-8n-k1-debug --export=ALL,WG_TASK_ID=run-actual-multinode-e97-8n-shortk-debug,TASK_ID=run-actual-multinode-e97-8n-shortk-debug,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_ACTUAL_MULTINODE_FILE_QUORUM=1,ASYNC_NODE_COUNT=8,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_WORKER_COUNT=8,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=6,DILOCO_K=1,GENERATIONS=1,BATCH_SIZE=1,CHUNK_SIZE=2048,ASYNC_GLOBAL_TIMEOUT_S=900,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=2.666667,TRAIN_MINUTES=20,FINALIZATION_BUFFER_SECONDS=60,RECOVERY_EVERY_GENERATIONS=1,RECOVERY_EVERY_SECONDS=60,EXPORT_EVERY_GENERATIONS=1,EXPORT_EVERY_SECONDS=60,ASYNC_DILOCO_FORCE_FINALIZATION_CHECKPOINT=1,ASYNC_DILOCO_DEVICE=cuda,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b1_k1_8n20m_debug,TOKENS_PER_LOCAL_STEP=2048,TOKENS_PER_DILOCO_GENERATION=2048 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Slurm accounting:

```text
4949891|async-e97-8n-k1-debug|COMPLETED|0:0|00:01:05|8||frontier[01651,01782,01915,02044,02301,02430,08437,09082]
4949891.batch|batch|COMPLETED|0:0|00:01:05|1|1|frontier01651
4949891.extern|extern|COMPLETED|0:0|00:01:05|8|8|frontier[01651,01782,01915,02044,02301,02430,08437,09082]
4949891.0|python|COMPLETED|0:0|00:00:50|8|8|frontier[01651,01782,01915,02044,02301,02430,08437,09082]
```

Recorded launcher `srun` command:

```bash
srun -N 8 -n 8 --ntasks-per-node=1 /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py --run-id run-actual-multinode-e97-8n-shortk-debug-4949891-20260706T212922Z --checkpoint /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt --run-dir /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k1_8n20m_debug/20260706/4949891-20260706T212922Z/async_run --metrics-json /lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k1_8n20m_debug/20260706/4949891-20260706T212922Z/artifacts/async_diloco_e97_256n_metrics.json --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --node-count 8 --worker-count 8 --local-quorum 1 --global-quorum 6 --generations 1 --local-steps 1 --timeout-s 900 --level E97 --params 1.3b --tokenizer p50k_base --batch-size 1 --chunk-size 2048 --lr 1e-3 --steps 1 --e97-chunk-size 32 --checkpoint-interval 64 --projection-chunk-size 256 --loss-chunk-size 256 --recovery-every-generations 1 --recovery-every-seconds 60 --export-every-generations 1 --export-every-seconds 60 --finalization-reserve-seconds 60 --bf16 --use-chunked-e97 --gradient-checkpointing --walltime-remaining-s 60 --dim 1792 --depth 11 --n-heads 216 --n-state 32 --n-groups 32 --n-slots 64 --expansion 1.0 --state-expansion 2 --gate-activation silu --linear-state 1 --mlp-ratio 2.2623 --mlp-multiple 64 --actual-multinode-file-quorum
```

K1 artifact root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b1_k1_8n20m_debug/20260706/4949891-20260706T212922Z
```

Key artifacts:

```text
artifacts/command.txt
artifacts/env.txt
artifacts/async_diloco_e97_256n_metrics.json
async_run/latest.json
async_run/generations/gen_000000/manifest.json
async_run/recovery_checkpoints/gen_000000/initial.json
async_run/recovery_checkpoints/gen_000000/walltime_finalization.json
async_run/export_checkpoints/gen_000000/initial.json
async_run/progress/node-00000.heartbeat.json ... node-00007.heartbeat.json
async_run/node_updates/node-00000.json ... node-00007.json
```

Actual multinode evidence:

- Slurm allocated eight distinct nodes: `frontier[01651,01782,01915,02044,02301,02430,08437,09082]`.
- Slurm step `4949891.0` completed with `NNodes=8` and `NTasks=8`.
- `command.txt` records `srun -N 8 -n 8 --ntasks-per-node=1`.
- The run wrote eight per-rank heartbeat files, one for each `node_rank` 0 through 7.
- The run wrote eight per-rank node update files, `node-00000.json` through `node-00007.json`.
- Nonzero ranks printed independent completion records with their own node IDs and node update paths.

K1 metrics summary from `artifacts/async_diloco_e97_256n_metrics.json`:

```text
mode=actual_multinode_file_quorum_debug
synthetic_token_stream=false
partial=false
latest_generation=0
requested_workers=8
participating_workers=7
accepted_updates=7
timed_out_updates=1
failed_updates=0
invalid_updates=0
stale_updates=0
quorum_threshold=6
quorum_size=7
quorum_status=advanced
quorum_distribution min=7 p50=7 p90=7 p95=7 p99=7 max=7 average=7.0
loss=13.975506510053362
loss_100=13.975506510053362
tokens_per_generation=14343
tokens_per_sec=421.5141445204644
latest_advanced=true
```

Per-node progress summary:

```text
node-00000 stage=coordinator_finalized accepted_nodes=7 seen_nodes=7 latest_advanced=true
node-00001 stage=node_update_written tokens=2049 loss=14.298737525939941
node-00002 stage=node_update_written tokens=2049 loss=14.509368896484375
node-00003 stage=node_update_written tokens=2049 loss=13.915630340576172
node-00004 stage=node_update_written tokens=2049 loss=13.835739135742188
node-00005 stage=node_update_written tokens=2049 loss=13.52777099609375
node-00006 stage=node_update_written tokens=2049 loss=13.702651977539062
node-00007 stage=node_update_written tokens=2049 loss=14.004501342773438
```

Checkpoint/latest behavior:

- `latest.json` exists and records `generation=0`, `published_by=global_merger`, and `run_id=run-actual-multinode-e97-8n-shortk-debug-4949891-20260706T212922Z`.
- `latest.json` points at the generation manifest plus recovery/export checkpoint manifests.
- The external production `latest.pt` seed path was used as the input checkpoint and was not mutated by this debug wrapper.
- Generation 0 advanced in the run-local chain only.

## Pass/No-Go

Pass:

- The fixed launcher executed via actual `srun`, not one serialized Python process.
- The successful K1 run used real data and the refreshed E97 `latest.pt`, with synthetic stream disabled.
- Separate Slurm node-rank processes contributed per-node heartbeat and node update artifacts.
- Metrics contain finite real loss and throughput, plus quorum distribution and checkpoint/finalization behavior.

No-go:

- Do not restore K40 or prepare a larger production run from this evidence alone.
- K10 failed before node payload serialization with `ValueError: non-finite metric value: nan`.
- The next step should debug why K10 produces no finite worker losses or add durable worker-error payload handling before scaling beyond K1.
