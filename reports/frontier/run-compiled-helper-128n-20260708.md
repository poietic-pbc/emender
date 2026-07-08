# Compiled MPICH helper 128n debug bridge

Date: 2026-07-08
Task: `run-compiled-helper-128n`

## Verdict

Decision: **PASS: proceed to the bounded 256n debug gate evaluation**.

The 128-node compiled-helper debug bridge was submitted only after confirming
`run-compiled-helper` had clean 8n and 64n train.py evidence:

1. 8n job `4954290`: wrapper validation `pass`, 64 ranks started, 64 accepted,
   zero stale/failed/timed-out updates.
2. 64n job `4954317`: wrapper validation `pass`, 512 ranks started, 512
   accepted, zero stale/failed/timed-out updates.

The 128n rung used the same validated `origin/main` commit, runtime stack,
train.py wrapper path, one-rank-per-GPU shape, and compiled Cray MPICH helper
transport as the passing 64n rung. The job wrote only run-local latest and
checkpoint artifacts under the debug output root.

## Submission

Validated commit:

```text
ea059485bef2c9f36ac45d247b20ab5f443ab111
```

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

Submission command:

```bash
sbatch --parsable -N 128 -J compiled-helper-trainpy-128n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=run-compiled-helper-128n,SMOKE_NAME=128n,SMOKE_NODE_COUNT=128,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_128n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=1024,ASYNC_EXPECTED_RANKS=1024,ASYNC_GLOBAL_QUORUM=1024,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,HUMAN_APPROVAL_RECORD='WG run-compiled-helper-128n: 128-node compiled MPICH helper train.py debug bridge after clean 8n job 4954290 and 64n job 4954317; run-local latest/checkpoint only; no production latest/last or chain-pointer mutation; no 256n submission.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Slurm job: `4954539`

Partition/QOS: `batch` / `debug`

Requested walltime: `00:20:00`

Estimated node-hours: `128 * 20 / 60 = 42.666667`

Slurm accounting:

```text
4954539|compiled-helper-trainpy-128n|COMPLETED|0:0|batch|debug|00:05:17|00:20:00|128|2026-07-07T22:03:16|2026-07-07T22:59:13|2026-07-07T23:04:30
```

Actual node-hours: `128 * 317 / 3600 = 11.271111`

## Artifacts

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z
```

Primary artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/metrics.json`
- Command capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/command.txt`
- Environment capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/logs/trainpy_async_quorum.log`
- Slurm stdout:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-128n-4954539.out`
- Slurm stderr:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-128n-4954539.err`
- Helper binary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/compiled_mpich_dense_helper`
- Helper shared library:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/compiled_mpich_dense_helper.so`

## Metrics

Wrapper validation: `pass`

| Field | Value |
| --- | ---: |
| Ranks started | `1024 / 1024` |
| Requested workers | `1024` |
| Participating workers | `1024` |
| Quorum | `1024 / 1024` |
| Quorum status | `advanced` |
| Accepted updates | `1024` |
| Stale updates | `0` |
| Failed updates | `0` |
| Timed-out updates | `0` |
| Tokens per generation | `132,096` |
| Tokens/sec | `1,795.319253` |
| Generation/reduce duration | `73.578 s` |
| Merge duration | `5.261716 s` |
| Reduce bucket count | `80` |
| Per-bucket reduce latency min/median/max | `0.037033 / 0.620935 / 5.39934 s` |
| Accepted dense delta bytes | `5,506,770,496` |
| MPI reduce aggregate bytes | `5,506,770,496` |
| MPI reduce payload sent bytes | `5,638,932,987,904` |
| Loss window | `loss=13.8437`, `loss_100=13.8437` |
| Helper MPI world size | `1024` |
| Helper reducer | `mpi_reduce_bucketed_weighted_sum` |
| Helper collective | `MPI_Reduce` |
| Strict collective all launched ranks | `true` |

Latest/checkpoint behavior:

- Run-local latest advanced to generation `0`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/async_run/latest.json`
- Checkpoint/publication records were written under the run root only:
  - `async_run/generations/gen_000000/manifest.json`
  - `async_run/recovery_checkpoints/gen_000000/initial.json`
  - `async_run/export_checkpoints/gen_000000/initial.json`
  - `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

Terminal state: **pass**. Slurm reported `COMPLETED` with exit code `0:0`.

## Production and live-update safety

No production `latest.pt` or `last.pt` path was passed to the 128n job. No
shared production chain pointer was updated. The environment capture records
the run-local debug output root and only the seed checkpoint path as a readable
checkpoint input. The latest and checkpoint publications listed above are all
under the task-local debug run root.

No live Lustre update collection was used. The compiled helper metrics report
`filesystem_live_quorum=false`, `transport=compiled-cray-mpich-helper-collective-reduce`,
`reducer=mpi_reduce_bucketed_weighted_sum`, and `collective=MPI_Reduce`.

## Recommendation

Proceed to `evaluate-compiled-helper` for the bounded 256n debug gate decision.
The 128n bridge launched all 1024 expected ranks, accepted all 1024 updates,
completed the compiled collective reduce, advanced run-local latest/checkpoint
state, and reported zero stale, failed, or timed-out updates.

Do not repeat 128n unless the evaluator finds an external concern outside this
run's recorded evidence. Do not promote anything to production latest/last from
this debug run.
