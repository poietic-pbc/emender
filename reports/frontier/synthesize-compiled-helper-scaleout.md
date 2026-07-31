# Compiled helper scaleout synthesis

Date: 2026-07-08
Task: `synthesize-compiled-helper-scaleout`

## Decision

Recommendation: **request human approval for a 256n x 1h debug continuation**.

Do not submit from this task. Do not run 256n x 12h, production QOS, or any
production latest/last or shared chain-pointer mutation from this evidence
alone.

Why: the available 1n, 2n, 8n, 64n, 128n, and bounded 256n-debug rungs all
passed with full rank start, full quorum, zero stale/failed/timed-out updates,
run-local latest/checkpoint publication, and stable compiled-helper reduce
metrics. The evidence supports one longer 256-node debug stability check. It
does not prove tolerance of non-joining ranks, straggler loss, or a production
duration.

## Implementation truth

The current compiled-helper path is **strict alive-rank collective**, not a
tree-style reducer and not genuinely failure-tolerant async for non-joining
ranks. Metrics identify the reducer as `mpi_reduce_bucketed_weighted_sum` with
`collective=MPI_Reduce`, `filesystem_live_quorum=false`, and
`strict_collective_all_launched_ranks=true` on the compiled-helper rungs. The
legacy launcher label `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`
does not mean peer-to-peer or failure-tolerant quorum; the observed data plane
is the compiled Cray MPICH collective reducer.

Pass criteria used for this synthesis:

- Slurm terminal state `COMPLETED 0:0`.
- `rank-start.tsv` count equals expected ranks.
- accepted updates equal global quorum, with stale/failed/timed-out/invalid
  counts all zero.
- compiled helper reports 80 reduce buckets, stable per-bucket latency, and
  expected aggregate bytes.
- loss window is finite and no NaN/divergence is reported in the one-generation
  debug window.
- latest/checkpoint publication is run-local under the debug run root only.
- no production latest/last or shared chain pointer is submitted or mutated.

## Shared run policy

- Partition/QOS: `batch` / `debug` for every available rung.
- Requested walltime: `00:20:00` for every rung.
- Seed checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Output root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke`
- Data:
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- Launcher for 8n and above:
  `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- 1n/2n command captures show the same train.py compiled-helper path with
  `--actual-multinode-compiled-mpich-quorum`, `--compiled-mpich-helper-bin`,
  `--compiled-mpich-ipc-dir`, `--mpi-dense-bucket-bytes 67108864`, and the
  seed checkpoint above.

The initial obsolete compiled-helper attempt `4953961` failed at 1n before
this ladder because each train.py rank launched its own helper subprocess. That
failure is superseded by the replacement ladder below.

## Ladder evidence

| Rung | Job | State | Nodes/ranks | Est nh | Actual nh | Accepted/quorum | stale/failed/timed-out/invalid | Reduce duration | Bucket latency min/median/max | Aggregate bytes | Payload sent bytes | Rank-start spread | Loss window | Latest/checkpoint |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| 1n | `4954252` | `COMPLETED 0:0` | 1 / 8 | 0.333333 | 0.056389 | 8 / 8 | 0 / 0 / 0 / 0 | 68.1221 s | 0.000384644 / 0.5817225 / 4.68398 s | 5,506,770,496 | 44,054,163,968 | 0 s | loss=13.6335, loss_100=13.6335 | run-local latest advanced; 4 checkpoint/publication records |
| 2n | `4954257` | `COMPLETED 0:0` | 2 / 16 | 0.666667 | 0.118889 | 16 / 16 | 0 / 0 / 0 / 0 | 72.5107 s | 0.0255964 / 0.620623 / 5.14927 s | 5,506,770,496 | 88,108,327,936 | 2 s | loss=13.7583, loss_100=13.7583 | run-local latest advanced; 4 checkpoint/publication records |
| 8n | `4954290` | `COMPLETED 0:0` | 8 / 64 | 2.666667 | 0.482222 | 64 / 64 | 0 / 0 / 0 / 0 | 73.0083 s | 0.0336529 / 0.619605 / 5.36679 s | 5,506,770,496 | 352,433,311,744 | 3 s | loss=13.8359, loss_100=13.8359 | run-local latest advanced; 4 checkpoint/publication records |
| 64n | `4954317` | `COMPLETED 0:0` | 64 / 512 | 21.333333 | 4.426667 | 512 / 512 | 0 / 0 / 0 / 0 | 73.1428 s | 0.036212 / 0.621742 / 5.23154 s | 5,506,770,496 | 2,819,466,493,952 | 5 s | loss=13.8345, loss_100=13.8345 | run-local latest advanced; 4 checkpoint/publication records |
| 128n | `4954539` | `COMPLETED 0:0` | 128 / 1024 | 42.666667 | 11.271111 | 1024 / 1024 | 0 / 0 / 0 / 0 | 73.5780 s | 0.037033 / 0.620935 / 5.39934 s | 5,506,770,496 | 5,638,932,987,904 | 15 s | loss=13.8437, loss_100=13.8437 | run-local latest advanced; 4 checkpoint/publication records |
| 256n-debug | `4954634` | `COMPLETED 0:0` | 256 / 2048 | 85.333333 | 28.942222 | 2048 / 2048 | 0 / 0 / 0 / 0 | 73.3795 s | 0.036637 / 0.619081 / 5.39255 s | 5,506,770,496 | 11,277,865,975,808 | 37 s | loss=13.8485, loss_100=13.8485 | run-local latest advanced; 4 checkpoint/publication records |

`Rank-start spread` is the first-to-last timestamp spread in `rank-start.tsv`;
it is the best available startup/rendezvous proxy in these artifacts. Slurm
queue/startup accounting is recorded below separately and should not be
confused with helper collective latency.

## Rung details

### 1n

- Job/accounting:
  `4954252|trainpy-async-quorum-1n|COMPLETED|0:0|batch|debug|00:03:23|00:20:00|1|2026-07-07T20:56:28|2026-07-07T20:56:57|2026-07-07T21:00:20`
- Command/export evidence: command capture under the run root records
  `srun -N 1 -n 8`, `--worker-count 8`, `--global-quorum 8`,
  `--timeout-s 120`, `--actual-multinode-compiled-mpich-quorum`, compiled
  helper binary/IPC paths, and the seed checkpoint.
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_1n/4954252-20260708T005658Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`, and
  `artifacts/compiled_mpich_dense_helper.so` under the run root.
- Slurm log paths recorded upstream:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4954252.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4954252.err`
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_1n/4954252-20260708T005658Z/async_run/latest.json`

### 2n

- Job/accounting:
  `4954257|trainpy-async-quorum-2n|COMPLETED|0:0|batch|debug|00:03:34|00:20:00|2|2026-07-07T21:01:18|2026-07-07T21:02:03|2026-07-07T21:05:37`
- Command/export evidence: command capture under the run root records
  `srun -N 2 -n 16`, `--worker-count 16`, `--global-quorum 16`,
  `--timeout-s 120`, `--actual-multinode-compiled-mpich-quorum`, compiled
  helper binary/IPC paths, and the seed checkpoint.
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_2n/4954257-20260708T010207Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`, and
  `artifacts/compiled_mpich_dense_helper.so` under the run root.
- Slurm log paths recorded upstream:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4954257.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4954257.err`
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_2n/4954257-20260708T010207Z/async_run/latest.json`

### 8n

- Submission:
  `sbatch --parsable -N 8 -J compiled-helper-trainpy-8n -t 00:20:00 --export=ALL,WG_TASK_ID=run-compiled-helper,SMOKE_NAME=8n,SMOKE_NODE_COUNT=8,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_8n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=64,ASYNC_GLOBAL_QUORUM=64,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=300,... scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- Job/accounting:
  `4954290|compiled-helper-trainpy-8n|COMPLETED|0:0|batch|debug|00:03:37|00:20:00|8|2026-07-07T21:22:12|2026-07-07T21:28:03|2026-07-07T21:31:40`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`,
  `artifacts/compiled_mpich_dense_helper.so`,
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-8n-4954290.out`,
  and `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-8n-4954290.err`.
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/async_run/latest.json`

### 64n

- Submission:
  `sbatch --parsable -N 64 -J compiled-helper-trainpy-64n -t 00:20:00 --export=ALL,WG_TASK_ID=run-compiled-helper,SMOKE_NAME=64n,SMOKE_NODE_COUNT=64,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_64n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=512,ASYNC_GLOBAL_QUORUM=512,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,... scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- Job/accounting:
  `4954317|compiled-helper-trainpy-64n|COMPLETED|0:0|batch|debug|00:04:09|00:20:00|64|2026-07-07T21:33:07|2026-07-07T21:35:24|2026-07-07T21:39:33`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`,
  `artifacts/compiled_mpich_dense_helper.so`,
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-64n-4954317.out`,
  and `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-64n-4954317.err`.
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/async_run/latest.json`

### 128n

- Submission:
  `sbatch --parsable -N 128 -J compiled-helper-trainpy-128n -t 00:20:00 --export=ALL,WG_TASK_ID=run-compiled-helper-128n,SMOKE_NAME=128n,SMOKE_NODE_COUNT=128,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_128n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=1024,ASYNC_EXPECTED_RANKS=1024,ASYNC_GLOBAL_QUORUM=1024,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,... scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- Job/accounting:
  `4954539|compiled-helper-trainpy-128n|COMPLETED|0:0|batch|debug|00:05:17|00:20:00|128|2026-07-07T22:03:16|2026-07-07T22:59:13|2026-07-07T23:04:30`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`,
  `artifacts/compiled_mpich_dense_helper.so`,
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-128n-4954539.out`,
  and `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-128n-4954539.err`.
- Source report:
  `origin/wg/agent-848/run-compiled-helper-128n:reports/frontier/run-compiled-helper-128n-20260708.md`
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/async_run/latest.json`

### 256n-debug

- Submission:
  `sbatch --parsable -N 256 -J compiled-helper-trainpy-256n-debug -t 00:20:00 --export=ALL,WG_TASK_ID=evaluate-compiled-helper,SMOKE_NAME=256n,SMOKE_NODE_COUNT=256,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=2048,ASYNC_EXPECTED_RANKS=2048,ASYNC_GLOBAL_QUORUM=2048,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=1200,... scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- Job/accounting:
  `4954634|compiled-helper-trainpy-256n-debug|COMPLETED|0:0|batch|debug|00:06:47|00:20:00|256|2026-07-07T23:22:17|2026-07-07T23:24:12|2026-07-07T23:30:59`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z`
- Artifacts:
  `summaries/summary.md`, `artifacts/metrics.json`,
  `artifacts/manifest.json`, `artifacts/command.txt`, `artifacts/env.txt`,
  `artifacts/rank-start.tsv`, `logs/trainpy_async_quorum.log`,
  `artifacts/compiled_mpich_dense_helper`,
  `artifacts/compiled_mpich_dense_helper.so`,
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-256n-debug-4954634.out`,
  and `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-256n-debug-4954634.err`.
- Source report:
  `reports/frontier/evaluate-compiled-helper-256n-debug-20260708.md`
- Latest path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/async_run/latest.json`

## Required approval for the recommended 256n x 1h debug continuation

This task does not submit that run. Before submission, a human must explicitly
approve these exact policies:

- Seed policy: read only the verified seed
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`;
  do not consume or advance production chain latest/last.
- Output policy: write to a new run-local debug root under
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/`
  with a distinct `E97_1.3B_step1065000_trainpy_compiled_helper_256n_1h_debug`
  variant; publish only `async_run/latest.json` and checkpoint metadata under
  that run root.
- Requested cost: `256 nodes * 1 hour = 256 node-hours`.
- Monitoring plan: watch Slurm state and elapsed time; tail wrapper stdout,
  stderr, and `logs/trainpy_async_quorum.log`; check `rank-start.tsv` reaches
  2048; verify `metrics.json` exists and records accepted/quorum, zero
  stale/failed/timed-out/invalid updates, reduce latency, aggregate bytes, loss
  window, and run-local latest/checkpoint paths.
- Stop conditions: cancel or fail the run if rank starts stall materially below
  2048, Slurm enters failure state, train log shows helper timeout/abort,
  metrics record any stale/failed/timed-out/invalid updates, loss is NaN/inf,
  latest/checkpoint publication escapes the debug run root, or any production
  latest/last or chain pointer would be touched.
- Approval boundary: this approval would cover only one 256n x 1h debug-QOS
  run. It would not authorize 256n x 12h, production QOS, larger scale, or
  production pointer mutation.

## Validation status

- No Slurm job was submitted by this synthesis task.
- No production latest/last or shared chain pointer was mutated by this
  synthesis task.
- The recommendation is tied to the concrete ladder evidence above: all six
  available rungs passed the full-rank/full-quorum/zero-error/debug-local
  criteria, but all evidence remains strict-collective and short-window.
