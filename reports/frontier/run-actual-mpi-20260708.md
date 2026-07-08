# Actual MPI compiled-helper async quorum ladder

Date: 2026-07-08
Task: `run-actual-mpi`

## Verdict

Decision: **PASS: the task-owned 1n -> 8n -> 64n ladder completed cleanly**.

The ladder was run from WG worktree
`/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-917`
after fast-forwarding to `origin/main` commit
`786a55a3c973d06f5736df7f6a90d422eef696d5`, which includes the
`merge-implement-mpi` transport guard and commit `4ee3630`.

Only the authorized debug rungs were submitted:

1. 1n job `4959329`: **pass**.
2. 8n job `4959340`: **pass**, submitted only after 1n passed.
3. 64n job `4959370`: **pass**, submitted only after 8n passed.

No 128n, 256n, 1h, 12h, or production job was submitted by this task.

All three rungs used `batch` / `debug`, requested `00:20:00`, used
`ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`, and wrote only
run-local latest/checkpoint artifacts under:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708
```

The active metrics transport on all three rungs was:

- `transport.name=compiled-cray-mpich-helper-collective-reduce`
- `transport.selector=compiled-cray-mpich-helper-p2p`
- `transport.helper_result.reducer=mpi_reduce_bucketed_weighted_sum`
- `transport.helper_result.mpi.collective=MPI_Reduce`
- `transport.tcp_dense_data_plane=false`
- `transport.filesystem_live_quorum=false`

Therefore TCP was **not** used as the hot dense aggregation path.

## Shared configuration

Seed checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
```

Training data:

```text
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
```

Launcher scripts:

- 1n: `scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch`
- 8n and 64n: `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch` with
  explicit `SMOKE_NODE_COUNT`, rank, quorum, and transport overrides.

## Slurm summary

| Rung | Job | State | Exit | Partition | QOS | Nodes | Requested walltime | Requested node-hours | Elapsed | Actual node-hours | Run root |
| --- | ---: | --- | --- | --- | --- | ---: | --- | ---: | --- | ---: | --- |
| 1n | `4959329` | `COMPLETED` | `0:0` | `batch` | `debug` | 1 | `00:20:00` | `0.333333` | `00:03:34` | `0.059444` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z` |
| 8n | `4959340` | `COMPLETED` | `0:0` | `batch` | `debug` | 8 | `00:20:00` | `2.666667` | `00:03:35` | `0.477778` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z` |
| 64n | `4959370` | `COMPLETED` | `0:0` | `batch` | `debug` | 64 | `00:20:00` | `21.333333` | `00:05:03` | `5.386667` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z` |

Slurm accounting snapshot:

```text
4959329|actual-mpi-compiled-1n|COMPLETED|0:0|batch|debug|00:03:34|00:20:00|1|2026-07-08T16:10:04|2026-07-08T16:10:52|2026-07-08T16:14:26
4959340|actual-mpi-compiled-8n|COMPLETED|0:0|batch|debug|00:03:35|00:20:00|8|2026-07-08T16:15:56|2026-07-08T16:16:29|2026-07-08T16:20:04
4959370|actual-mpi-compiled-64n|COMPLETED|0:0|batch|debug|00:05:03|00:20:00|64|2026-07-08T16:21:04|2026-07-08T16:22:52|2026-07-08T16:27:55
```

## Submission commands

### 1n

```bash
sbatch --parsable -N 1 -J actual-mpi-compiled-1n -t 00:20:00 -p batch -q debug \
  --export=ALL,WG_TASK_ID=run-actual-mpi,SMOKE_NAME=1n,SMOKE_NODE_COUNT=1,SCALEOUT_VARIANT=E97_1.3B_step1065000_actual_mpi_compiled_helper_1n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=8,ASYNC_EXPECTED_RANKS=8,ASYNC_GLOBAL_QUORUM=8,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=300,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.333333,HUMAN_APPROVAL_RECORD='WG run-actual-mpi: task-owned 1-node MPI compiled-helper async quorum rung after origin/main 786a55a; run-local latest/checkpoint only; no production latest/last mutation; no later rung until clean.' \
  scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

### 8n

```bash
sbatch --parsable -N 8 -J actual-mpi-compiled-8n -t 00:20:00 -p batch -q debug \
  --export=ALL,WG_TASK_ID=run-actual-mpi,SMOKE_NAME=8n,SMOKE_NODE_COUNT=8,SCALEOUT_VARIANT=E97_1.3B_step1065000_actual_mpi_compiled_helper_8n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=64,ASYNC_GLOBAL_QUORUM=64,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=600,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=2.666667,HUMAN_APPROVAL_RECORD='WG run-actual-mpi: task-owned 8-node MPI compiled-helper async quorum rung after clean 1n job 4959329; run-local latest/checkpoint only; no production latest/last mutation; no 64n until clean.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

### 64n

```bash
sbatch --parsable -N 64 -J actual-mpi-compiled-64n -t 00:20:00 -p batch -q debug \
  --export=ALL,WG_TASK_ID=run-actual-mpi,SMOKE_NAME=64n,SMOKE_NODE_COUNT=64,SCALEOUT_VARIANT=E97_1.3B_step1065000_actual_mpi_compiled_helper_64n,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=512,ASYNC_GLOBAL_QUORUM=512,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=900,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=21.333333,HUMAN_APPROVAL_RECORD='WG run-actual-mpi: task-owned 64-node MPI compiled-helper async quorum rung after clean 8n job 4959340 and 1n job 4959329; run-local latest/checkpoint only; no production latest/last mutation; no 128n/256n/1h/12h/production submission.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

## Quorum and transport evidence

| Rung | Wrapper validation | Rank starts | MPI world size | Accepted | Missing | Stale | Late | Timed out | Rejected | Catchup events | Latest advanced |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1n | `pass` | `8 / 8` | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | yes, generation `0` |
| 8n | `pass` | `64 / 64` | 64 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | yes, generation `0` |
| 64n | `pass` | `512 / 512` | 512 | 512 | 0 | 0 | 0 | 0 | 0 | 0 | yes, generation `0` |

All rungs reported:

- `mode=actual_multinode_compiled_mpich_quorum`
- `async_quorum_transport=compiled-cray-mpich-helper-p2p`
- `async_quorum_transport_actual=compiled-cray-mpich-helper-collective-reduce`
- `bounded_debug_transport=actual_multinode_compiled_mpich_quorum`
- `transport.approval_class=frontier-production-candidate`
- `strict_collective_all_launched_ranks=true`
- `tcp_dense_data_plane=false`

The 1n rung used `async_compiled_mpich_file_gather=1` as the same-node helper
diagnostic path. The 8n and 64n scale rungs both used
`async_compiled_mpich_file_gather=0`.

## Merge, bytes, and loss

| Rung | Generation duration | Merge duration | Reduce buckets | Bucket latency min / median / max | Accepted dense bytes | MPI aggregate bytes | MPI payload sent bytes | Loss window |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1n | `69.9201 s` | `5.272404 s` | 80 | `0.000412 / 0.610142 / 4.684710 s` | `5,506,770,496` | `5,506,770,496` | `44,054,163,968` | `loss=13.6335`, `loss_100=13.6335` |
| 8n | `72.4874 s` | `5.219292 s` | 80 | `0.033332 / 0.616027 / 5.178220 s` | `5,506,770,496` | `5,506,770,496` | `352,433,311,744` | `loss=13.8359`, `loss_100=13.8359` |
| 64n | `75.6420 s` | `5.251472 s` | 80 | `0.036416 / 0.659801 / 5.377600 s` | `5,506,770,496` | `5,506,770,496` | `2,819,466,493,952` | `loss=13.8345`, `loss_100=13.8345` |

Tokens per generation:

- 1n: `1,032`, `14.7597 tok/s`
- 8n: `8,256`, `113.8957 tok/s`
- 64n: `66,048`, `873.1657 tok/s`

## Latest and checkpoint behavior

Each rung advanced only its run-local `async_run/latest.json` to generation `0`
and wrote four checkpoint/publication records under the run root:

- `async_run/generations/gen_000000/manifest.json`
- `async_run/recovery_checkpoints/gen_000000/initial.json`
- `async_run/export_checkpoints/gen_000000/initial.json`
- `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

Run-local latest paths:

- 1n:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/async_run/latest.json`
- 8n:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/async_run/latest.json`
- 64n:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/async_run/latest.json`

## Primary artifacts

### 1n job `4959329`

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/artifacts/metrics.json`
- Environment:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_1n/4959329-20260708T201056Z/logs/trainpy_async_quorum.log`
- Slurm logs:
  `logs/frontier/trainpy_async_quorum/actual-mpi-compiled-1n-4959329.{out,err}`

### 8n job `4959340`

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/artifacts/metrics.json`
- Environment:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_8n/4959340-20260708T201633Z/logs/trainpy_async_quorum.log`
- Slurm logs:
  `logs/frontier/trainpy_async_quorum/actual-mpi-compiled-8n-4959340.{out,err}`

### 64n job `4959370`

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/artifacts/metrics.json`
- Environment:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/run_actual_mpi_1n8n64n_ladder_20260708/20260708/E97_1.3B_step1065000_actual_mpi_compiled_helper_64n/4959370-20260708T202255Z/logs/trainpy_async_quorum.log`
- Slurm logs:
  `logs/frontier/trainpy_async_quorum/actual-mpi-compiled-64n-4959370.{out,err}`

## Production latest/last check

The task used a debug output root and did not use the production chain updater.
The production chain pointer snapshot was unchanged before and after the ladder:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|7719679569|1782849877|'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'
```

The same production `last.pt` path was absent both before and after the ladder:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/last.pt|ABSENT
```

The verified seed latest was readable and unchanged across the run:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt|7719679924|1783330191|'/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt'
```

## Recommendation

Proceed with **one bounded 256n MPI compiled-helper debug gate** only if it keeps
the same guardrails:

- `batch` / `debug`
- `00:20:00`
- run-local output root
- `ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p`
- explicit fail-fast rejection if metadata reports TCP as the hot aggregation
  path or `tcp_dense_data_plane=true`
- no production latest/last mutation
- no 1h, 12h, or production launch implied by the debug result

The basis for that recommendation is this task-owned clean 1n/8n/64n ladder:
512/512 rank starts at 64n, 512 accepted updates, zero missing/stale/late/
timed-out/rejected updates, run-local latest advanced, and no TCP dense data
plane use.
