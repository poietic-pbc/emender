# Compiled MPICH helper 8n/64n smoke ladder

Date: 2026-07-08
Task: `run-compiled-helper`

## Verdict

Decision: **PASS: proceed to the 128n debug bridge**.

The compiled MPICH dense helper ladder was run after the prerequisite 1n and
2n train.py smokes passed. The task submitted only the owned 8n and 64n rungs:

1. 8n job `4954290`: **pass**.
2. 64n job `4954317`: **pass**, submitted only after the clean 8n pass.

No 128n, 256n, or production job was submitted by this task. Both jobs used
`batch` / `debug` with `00:20:00` requested walltime and wrote only run-local
latest/checkpoint artifacts under the debug smoke output root.

`ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p` is a legacy selector
label for these runs. The actual helper metrics show the new compiled collective
path:

- `transport=compiled-cray-mpich-helper-collective-reduce`
- `reducer=mpi_reduce_bucketed_weighted_sum`
- `collective=MPI_Reduce`
- `strict_collective_all_launched_ranks=true`

## Shared configuration

Seed checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
```

Training data:

```text
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
```

Output root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke
```

Launcher:

```text
scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

The wrapper records `async_compiled_mpich_file_gather=0` for both rungs, so
these runs did not use the same-node file-gather diagnostic path.

## 8n rung

Slurm job: `4954290`

Partition/QOS: `batch` / `debug`

Requested walltime: `00:20:00`

Estimated node-hours: `2.666667`

Slurm accounting:

```text
4954290|compiled-helper-trainpy-8n|COMPLETED|0:0|batch|debug|00:03:37|00:20:00|8|2026-07-07T21:22:12|2026-07-07T21:28:03|2026-07-07T21:31:40
```

Actual node-hours: `8 * 217 / 3600 = 0.482222`

Submission command:

```bash
sbatch --parsable -N 8 -J compiled-helper-trainpy-8n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-compiled-helper,SMOKE_NAME=8n,SMOKE_NODE_COUNT=8,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_8n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=64,ASYNC_GLOBAL_QUORUM=64,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=300,HUMAN_APPROVAL_RECORD='WG run-compiled-helper: 8-node compiled MPICH helper train.py smoke after 1n/2n pass; run-local latest/checkpoint only; no production latest mutation; no 128n/256n submission.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z
```

Primary artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/metrics.json`
- Command capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/command.txt`
- Environment capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/logs/trainpy_async_quorum.log`
- Slurm stdout:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-8n-4954290.out`
- Slurm stderr:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-8n-4954290.err`
- Helper binary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/compiled_mpich_dense_helper`
- Helper shared library:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/compiled_mpich_dense_helper.so`

8n metrics:

| Field | Value |
| --- | ---: |
| Wrapper validation | `pass` |
| Ranks started | `64 / 64` |
| Requested workers | `64` |
| Participating workers | `64` |
| Quorum | `64 / 64` |
| Quorum status | `advanced` |
| Accepted updates | `64` |
| Stale updates | `0` |
| Failed updates | `0` |
| Timed-out updates | `0` |
| Tokens per generation | `8,256` |
| Tokens/sec | `113.083033` |
| Generation/reduce duration | `73.0083 s` |
| Merge duration | `5.262680 s` |
| Reduce bucket count | `80` |
| Per-bucket reduce latency min/median/max | `0.0336529 / 0.619605 / 5.36679 s` |
| Accepted dense delta bytes | `5,506,770,496` |
| MPI reduce aggregate bytes | `5,506,770,496` |
| MPI reduce payload sent bytes | `352,433,311,744` |
| Loss window | `loss=13.8359`, `loss_100=13.8359` |
| Helper MPI world size | `64` |
| Helper reducer | `mpi_reduce_bucketed_weighted_sum` |
| Helper collective | `MPI_Reduce` |

Latest/checkpoint behavior:

- Run-local latest advanced to generation `0`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/async_run/latest.json`
- Checkpoint/publication records were written under the run root only:
  - `async_run/generations/gen_000000/manifest.json`
  - `async_run/recovery_checkpoints/gen_000000/initial.json`
  - `async_run/export_checkpoints/gen_000000/initial.json`
  - `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

Terminal state: **pass**.

## 64n rung

The 64n job was submitted only after 8n job `4954290` completed cleanly with
wrapper validation `pass`.

Slurm job: `4954317`

Partition/QOS: `batch` / `debug`

Requested walltime: `00:20:00`

Estimated node-hours: `21.333333`

Slurm accounting:

```text
4954317|compiled-helper-trainpy-64n|COMPLETED|0:0|batch|debug|00:04:09|00:20:00|64|2026-07-07T21:33:07|2026-07-07T21:35:24|2026-07-07T21:39:33
```

Actual node-hours: `64 * 249 / 3600 = 4.426667`

Submission command:

```bash
sbatch --parsable -N 64 -J compiled-helper-trainpy-64n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-compiled-helper,SMOKE_NAME=64n,SMOKE_NODE_COUNT=64,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_64n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=512,ASYNC_GLOBAL_QUORUM=512,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,HUMAN_APPROVAL_RECORD='WG run-compiled-helper: 64-node compiled MPICH helper train.py smoke after clean 8n job 4954290 pass; run-local latest/checkpoint only; no production latest mutation; no 128n/256n submission.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z
```

Primary artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/metrics.json`
- Command capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/command.txt`
- Environment capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/logs/trainpy_async_quorum.log`
- Slurm stdout:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-64n-4954317.out`
- Slurm stderr:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-64n-4954317.err`
- Helper binary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/compiled_mpich_dense_helper`
- Helper shared library:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/compiled_mpich_dense_helper.so`

64n metrics:

| Field | Value |
| --- | ---: |
| Wrapper validation | `pass` |
| Ranks started | `512 / 512` |
| Requested workers | `512` |
| Participating workers | `512` |
| Quorum | `512 / 512` |
| Quorum status | `advanced` |
| Accepted updates | `512` |
| Stale updates | `0` |
| Failed updates | `0` |
| Timed-out updates | `0` |
| Tokens per generation | `66,048` |
| Tokens/sec | `903.000705` |
| Generation/reduce duration | `73.1428 s` |
| Merge duration | `5.237598 s` |
| Reduce bucket count | `80` |
| Per-bucket reduce latency min/median/max | `0.036212 / 0.621742 / 5.23154 s` |
| Accepted dense delta bytes | `5,506,770,496` |
| MPI reduce aggregate bytes | `5,506,770,496` |
| MPI reduce payload sent bytes | `2,819,466,493,952` |
| Loss window | `loss=13.8345`, `loss_100=13.8345` |
| Helper MPI world size | `512` |
| Helper reducer | `mpi_reduce_bucketed_weighted_sum` |
| Helper collective | `MPI_Reduce` |

Latest/checkpoint behavior:

- Run-local latest advanced to generation `0`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/async_run/latest.json`
- Checkpoint/publication records were written under the run root only:
  - `async_run/generations/gen_000000/manifest.json`
  - `async_run/recovery_checkpoints/gen_000000/initial.json`
  - `async_run/export_checkpoints/gen_000000/initial.json`
  - `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

Terminal state: **pass**.

## Production and live-update safety

No live Lustre update collection was used. The compiled helper metrics report
`filesystem_live_quorum=false` and the environment captures record
`bounded_debug_transport=actual_multinode_compiled_mpich_quorum`.

No production `latest.pt` or `last.pt` path was passed to either job, and no
shared production chain pointer was updated. The only `latest` publications were
the run-local `async_run/latest.json` files listed above. The only checkpoint
publication records were the run-local generation, recovery, export, and
walltime-finalization JSON files under each debug run root.

## Recommendation

Proceed to the **128n debug bridge**. Do not repeat 64n before the bridge unless
additional independent concerns arise, because the 64n rung launched all 512
ranks, completed the compiled collective reduce, advanced quorum/latest, and
reported zero stale, failed, or timed-out updates.

This task did not submit 128n or 256n.
