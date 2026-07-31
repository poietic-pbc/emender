# Async Quorum B4/K40 Debug Ladder Through 256n

Date: 2026-07-09
Task: `run-async-b4k40-debug-ladder-256n`

## Verdict

Decision: **PASS: fixed async DiLoCo B4/K40 debug ladder completed through 256 nodes.**

The ladder used only `batch` / `debug` jobs. No production, regular-QOS, extended-QOS,
1h+, 12h, or production/latest publication job was submitted. Rungs were submitted
sequentially: 1n evidence was reused from accepted parity job `4961611`; new 2n, 8n,
64n, 128n, and 256n rungs were submitted only after the previous rung was terminal
and validated.

Current checkout during the ladder was `f693e2c`, matching `origin/main`.

## Fixed Recipe

All attempted rungs used the fixed train.py-backed async quorum DiLoCo smoke recipe:

- Seed checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Training data:
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- `ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0`
- `BATCH_SIZE=4`
- `CHUNK_SIZE=2048`
- `DILOCO_K=40`
- `ASYNC_LOCAL_STEPS=40`
- `LEARNING_RATE=0.001007`
- `OPTIMIZER=schedulefree`
- `WEIGHT_DECAY=0.01`
- `WARMUP_STEPS=0`
- `GRAD_ACCUM=1`
- `GRAD_CLIP=1.0`
- `MODEL_TOKENIZER=p50k_base`
- one train.py-backed rank per GPU, 8 ranks per node
- `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`
- Actual transport recorded by metrics:
  `compiled-cray-mpich-helper-collective-reduce`
- `ALLOW_FRONTIER_TCP_SCALE_DEBUG=0`
- TCP dense data plane disabled
- no per-step DDP

New ladder output root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709
```

The reused 1n parity rung used its original debug output root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_exact_20260709
```

## Attempted Rungs

| Rung | Job | Expected ranks | Rank starts | Accepted updates | Loss | Tokens/sec | Transport | TCP dense data plane | Elapsed | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1n | `4961611` | 8 | 8 | 8 | 2.52576 | 38,868.412746 | compiled-cray-mpich-helper-collective-reduce | false | 00:04:26 | PASS |
| 2n | `4962220` | 16 | 16 | 16 | 2.54632 | 73,877.459617 | compiled-cray-mpich-helper-collective-reduce | false | 00:04:43 | PASS |
| 8n | `4962237` | 64 | 64 | 64 | 2.52829 | 289,594.119165 | compiled-cray-mpich-helper-collective-reduce | false | 00:04:47 | PASS |
| 64n | `4962270` | 512 | 512 | 512 | 2.52206 | 2,293,483.304503 | compiled-cray-mpich-helper-collective-reduce | false | 00:05:16 | PASS |
| 128n | `4962329` | 1024 | 1024 | 1024 | 2.52105 | 4,574,970.870532 | compiled-cray-mpich-helper-collective-reduce | false | 00:06:14 | PASS |
| 256n | `4962400` | 2048 | 2048 | 2048 | 2.52083 | 9,096,993.214681 | compiled-cray-mpich-helper-collective-reduce | false | 00:08:03 | PASS |

Every rung had `stale_updates=0`, `failed_updates=0`, `invalid_updates=0`,
`timed_out_updates=0`, and loss inside the expected healthy range of roughly
2.0-3.5 for this exact smoke.

## Slurm Accounting

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NTasks|Submit|Start|End
4961611|async-b4k40-opt-1n|batch|debug|COMPLETED|0:0|00:04:26|1||2026-07-09T07:40:14|2026-07-09T07:40:15|2026-07-09T07:44:41
4961611.batch|batch|||COMPLETED|0:0|00:04:26|1|1|2026-07-09T07:40:15|2026-07-09T07:40:15|2026-07-09T07:44:41
4961611.extern|extern|||COMPLETED|0:0|00:04:26|1|1|2026-07-09T07:40:15|2026-07-09T07:40:15|2026-07-09T07:44:41
4961611.0|bash|||COMPLETED|0:0|00:04:04|1|8|2026-07-09T07:40:37|2026-07-09T07:40:37|2026-07-09T07:44:41
4962220|async-b4k40-ladder-2n|batch|debug|COMPLETED|0:0|00:04:43|2||2026-07-09T10:00:47|2026-07-09T10:01:56|2026-07-09T10:06:39
4962220.batch|batch|||COMPLETED|0:0|00:04:43|1|1|2026-07-09T10:01:56|2026-07-09T10:01:56|2026-07-09T10:06:39
4962220.extern|extern|||COMPLETED|0:0|00:04:43|2|2|2026-07-09T10:01:56|2026-07-09T10:01:56|2026-07-09T10:06:39
4962220.0|bash|||COMPLETED|0:0|00:04:15|2|16|2026-07-09T10:02:24|2026-07-09T10:02:24|2026-07-09T10:06:39
4962237|async-b4k40-ladder-8n|batch|debug|COMPLETED|0:0|00:04:47|8||2026-07-09T10:07:56|2026-07-09T10:08:15|2026-07-09T10:13:02
4962237.batch|batch|||COMPLETED|0:0|00:04:47|1|1|2026-07-09T10:08:15|2026-07-09T10:08:15|2026-07-09T10:13:02
4962237.extern|extern|||COMPLETED|0:0|00:04:47|8|8|2026-07-09T10:08:15|2026-07-09T10:08:15|2026-07-09T10:13:02
4962237.0|bash|||COMPLETED|0:0|00:04:23|8|64|2026-07-09T10:08:39|2026-07-09T10:08:39|2026-07-09T10:13:02
4962270|async-b4k40-ladder-64n|batch|debug|COMPLETED|0:0|00:05:16|64||2026-07-09T10:14:03|2026-07-09T10:14:04|2026-07-09T10:19:20
4962270.batch|batch|||COMPLETED|0:0|00:05:16|1|1|2026-07-09T10:14:04|2026-07-09T10:14:04|2026-07-09T10:19:20
4962270.extern|extern|||COMPLETED|0:0|00:05:17|64|64|2026-07-09T10:14:04|2026-07-09T10:14:04|2026-07-09T10:19:21
4962270.0|bash|||COMPLETED|0:0|00:04:54|64|512|2026-07-09T10:14:26|2026-07-09T10:14:26|2026-07-09T10:19:20
4962329|async-b4k40-ladder-128n|batch|debug|COMPLETED|0:0|00:06:14|128||2026-07-09T10:20:26|2026-07-09T10:28:39|2026-07-09T10:34:53
4962329.batch|batch|||COMPLETED|0:0|00:06:14|1|1|2026-07-09T10:28:39|2026-07-09T10:28:39|2026-07-09T10:34:53
4962329.extern|extern|||COMPLETED|0:0|00:06:14|128|128|2026-07-09T10:28:39|2026-07-09T10:28:39|2026-07-09T10:34:53
4962329.0|bash|||COMPLETED|0:0|00:05:49|128|1024|2026-07-09T10:29:04|2026-07-09T10:29:04|2026-07-09T10:34:53
4962400|async-b4k40-ladder-256n|batch|debug|COMPLETED|0:0|00:08:03|256||2026-07-09T10:35:54|2026-07-09T10:48:29|2026-07-09T10:56:32
4962400.batch|batch|||COMPLETED|0:0|00:08:03|1|1|2026-07-09T10:48:29|2026-07-09T10:48:29|2026-07-09T10:56:32
4962400.extern|extern|||COMPLETED|0:0|00:08:03|256|256|2026-07-09T10:48:29|2026-07-09T10:48:29|2026-07-09T10:56:32
4962400.0|bash|||COMPLETED|0:0|00:07:36|256|2048|2026-07-09T10:48:56|2026-07-09T10:48:56|2026-07-09T10:56:32
```

## Submission Commands

The 1n rung reused the accepted fixed optimizer-state parity job `4961611`,
documented in `reports/frontier/debug-async-quorum-b4k40-parity-20260709.md`.
Its captured environment and command files are listed below under artifacts.

New rungs were submitted with these exact `sbatch` commands:

```bash
sbatch --parsable -N 2 -J async-b4k40-ladder-2n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-async-b4k40-debug-ladder-256n,TASK_ID=run-async-b4k40-debug-ladder-256n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709,SMOKE_NAME=2n,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_2n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ALLOW_FRONTIER_TCP_SCALE_DEBUG=0,ASYNC_TRAINPY_RANKS=16,ASYNC_EXPECTED_RANKS=16,ASYNC_GLOBAL_QUORUM=16,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=120,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,BATCH_SIZE=4,CHUNK_SIZE=2048,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,HUMAN_APPROVAL_RECORD='WG run-async-b4k40-debug-ladder-256n: 2-node fixed async DiLoCo B4/K40 stability ladder rung; batch/debug 00:20:00; sequential after accepted 1n parity job 4961611; run-local debug latest/checkpoint only; no production or extended job.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch --parsable -N 8 -J async-b4k40-ladder-8n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-async-b4k40-debug-ladder-256n,TASK_ID=run-async-b4k40-debug-ladder-256n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709,SMOKE_NAME=8n,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_8n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ALLOW_FRONTIER_TCP_SCALE_DEBUG=0,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=64,ASYNC_GLOBAL_QUORUM=64,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=300,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,BATCH_SIZE=4,CHUNK_SIZE=2048,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,HUMAN_APPROVAL_RECORD='WG run-async-b4k40-debug-ladder-256n: 8-node fixed async DiLoCo B4/K40 stability ladder rung; batch/debug 00:20:00; sequential after clean 2n job 4962220; run-local debug latest/checkpoint only; no production or extended job.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch --parsable -N 64 -J async-b4k40-ladder-64n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-async-b4k40-debug-ladder-256n,TASK_ID=run-async-b4k40-debug-ladder-256n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709,SMOKE_NAME=64n,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_64n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ALLOW_FRONTIER_TCP_SCALE_DEBUG=0,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=512,ASYNC_GLOBAL_QUORUM=512,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=600,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,BATCH_SIZE=4,CHUNK_SIZE=2048,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,HUMAN_APPROVAL_RECORD='WG run-async-b4k40-debug-ladder-256n: 64-node fixed async DiLoCo B4/K40 stability ladder rung; batch/debug 00:20:00; sequential after clean 8n job 4962237; run-local debug latest/checkpoint only; no production or extended job.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch --parsable -N 128 -J async-b4k40-ladder-128n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-async-b4k40-debug-ladder-256n,TASK_ID=run-async-b4k40-debug-ladder-256n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709,SMOKE_NAME=128n,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_128n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ALLOW_FRONTIER_TCP_SCALE_DEBUG=0,ASYNC_TRAINPY_RANKS=1024,ASYNC_EXPECTED_RANKS=1024,ASYNC_GLOBAL_QUORUM=1024,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,BATCH_SIZE=4,CHUNK_SIZE=2048,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,HUMAN_APPROVAL_RECORD='WG run-async-b4k40-debug-ladder-256n: 128-node fixed async DiLoCo B4/K40 stability ladder rung; batch/debug 00:20:00; sequential after clean 64n job 4962270; run-local debug latest/checkpoint only; no production or extended job.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch --parsable -N 256 -J async-b4k40-ladder-256n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-async-b4k40-debug-ladder-256n,TASK_ID=run-async-b4k40-debug-ladder-256n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709,SMOKE_NAME=256n,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ALLOW_FRONTIER_TCP_SCALE_DEBUG=0,ASYNC_TRAINPY_RANKS=2048,ASYNC_EXPECTED_RANKS=2048,ASYNC_GLOBAL_QUORUM=2048,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=1200,DILOCO_K=40,ASYNC_LOCAL_STEPS=40,BATCH_SIZE=4,CHUNK_SIZE=2048,LEARNING_RATE=0.001007,OPTIMIZER=schedulefree,WEIGHT_DECAY=0.01,WARMUP_STEPS=0,GRAD_ACCUM=1,GRAD_CLIP=1.0,MODEL_TOKENIZER=p50k_base,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,HUMAN_APPROVAL_RECORD='WG run-async-b4k40-debug-ladder-256n: 256-node fixed async DiLoCo B4/K40 stability ladder final rung; batch/debug 00:20:00; sequential after clean 128n job 4962329; run-local debug latest/checkpoint only; no production or extended job.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

The wrapper captured the expanded `srun` command for each rung in
`artifacts/command.txt`, and captured full runtime environment in
`artifacts/env.txt`.

## Artifact Index

| Rung | Job | Run root | Env | Command | Metrics | Manifest | Stdout | Stderr |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 1n | `4961611` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_exact_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4_k40_optstate_1n/4961611-20260709T114016Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-opt-1n-4961611.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-opt-1n-4961611.err` |
| 2n | `4962220` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_2n/4962220-20260709T140158Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-2n-4962220.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-2n-4962220.err` |
| 8n | `4962237` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_8n/4962237-20260709T140818Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-8n-4962237.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-8n-4962237.err` |
| 64n | `4962270` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_64n/4962270-20260709T141405Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-64n-4962270.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-64n-4962270.err` |
| 128n | `4962329` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_128n/4962329-20260709T142840Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-128n-4962329.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-128n-4962329.err` |
| 256n | `4962400` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/4962400-20260709T144830Z` | `artifacts/env.txt` | `artifacts/command.txt` | `artifacts/metrics.json` | `artifacts/manifest.json` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-256n-4962400.out` | `logs/frontier/trainpy_async_quorum/async-b4k40-ladder-256n-4962400.err` |

## Per-Rung Validation Notes

### 1n: job 4961611

Reused the accepted fixed optimizer-state parity rung. The run was documented
in `reports/frontier/debug-async-quorum-b4k40-parity-20260709.md` and had
sufficient artifacts for this ladder:

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:04:26`.
- Rank starts: `8 / 8`.
- Accepted updates: `8 / 8`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.52576`.
- Tokens/sec: `38,868.412746`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_exact_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4_k40_optstate_1n/4961611-20260709T114016Z/async_run/latest.json`.

### 2n: job 4962220

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:04:43`.
- Rank starts: `16 / 16`.
- Accepted updates: `16 / 16`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.54632`.
- Tokens/sec: `73,877.459617`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_2n/4962220-20260709T140158Z/async_run/latest.json`.

### 8n: job 4962237

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:04:47`.
- Rank starts: `64 / 64`.
- Accepted updates: `64 / 64`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.52829`.
- Tokens/sec: `289,594.119165`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_8n/4962237-20260709T140818Z/async_run/latest.json`.

### 64n: job 4962270

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:05:16`.
- Rank starts: `512 / 512`.
- Accepted updates: `512 / 512`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.52206`.
- Tokens/sec: `2,293,483.304503`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_64n/4962270-20260709T141405Z/async_run/latest.json`.

### 128n: job 4962329

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:06:14`.
- Rank starts: `1024 / 1024`.
- Accepted updates: `1024 / 1024`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.52105`.
- Tokens/sec: `4,574,970.870532`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_128n/4962329-20260709T142840Z/async_run/latest.json`.

### 256n: job 4962400

- Slurm: `COMPLETED 0:0`, `batch/debug`, elapsed `00:08:03`.
- Rank starts: `2048 / 2048`.
- Accepted updates: `2048 / 2048`.
- Stale/failed/invalid/timed-out updates: all `0`.
- Loss: `2.52083`.
- Tokens/sec: `9,096,993.214681`.
- Transport: `compiled-cray-mpich-helper-collective-reduce`.
- TCP dense data plane: `false`.
- Latest path stayed under the run-local debug root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260709/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/4962400-20260709T144830Z/async_run/latest.json`.

## Publication Guard

The latest/checkpoint publication guard passed for every rung:

- `latest.json` paths were under each run root's `async_run/` directory.
- Checkpoint manifest paths recorded by metrics were under the same run-local
  debug roots.
- No shared production `latest.pt`, `last.pt`, or chain pointer path was
  passed in the submission environment.
- `ALLOW_FRONTIER_TCP_SCALE_DEBUG=0` was set on all new submissions.
- Metrics recorded `tcp_dense_data_plane=false` on every rung.

## Conclusion

The fixed async DiLoCo B4/K40 debug smoke ladder is stable through 256 nodes
for this bounded 20-minute debug recipe. The final 256n rung launched all
2048 expected train.py-backed GPU ranks, accepted all 2048 updates, advanced
run-local latest/checkpoint state, used the compiled Cray MPICH helper
collective reduce transport, and reported no stale, failed, invalid, or
timed-out updates.

No follow-on production or extended run is implied by this report.
