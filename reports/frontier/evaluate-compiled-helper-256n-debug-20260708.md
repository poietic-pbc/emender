# Compiled helper 256n debug gate evaluation

Date: 2026-07-08
Task: `evaluate-compiled-helper`

## Verdict

Decision: **PASS: bounded 256n debug smoke completed; do not promote to any
1h, 12h, production-QOS, or larger run without explicit human approval.**

The prior ladder was clean through the 128n bridge, so this task submitted one
bounded 256-node debug-QOS smoke. No production latest/last pointer or shared
production chain pointer was mutated. `register-refreshed-e97` remains paused
with no newer verified seed, so the smoke used the current verified E97 seed
checkpoint for debug only.

## Prior rung gate evidence

All required prior rungs passed before the 256n submission:

| Rung | Job | Status | Key evidence |
| --- | ---: | --- | --- |
| 1n | `4954252` | pass | `COMPLETED 0:0`, `8/8` ranks, `8` accepted, zero stale/failed/timed-out/invalid updates, run-local latest/checkpoint only. |
| 2n | `4954257` | pass | `COMPLETED 0:0`, `16/16` ranks, `16` accepted, zero stale/failed/timed-out/invalid updates, run-local latest/checkpoint only. |
| 8n | `4954290` | pass | `COMPLETED 0:0`, `64/64` ranks, `64` accepted, zero stale/failed/timed-out updates, run-local latest/checkpoint only. |
| 64n | `4954317` | pass | `COMPLETED 0:0`, `512/512` ranks, `512` accepted, zero stale/failed/timed-out updates, run-local latest/checkpoint only. |
| 128n | `4954539` | pass | `COMPLETED 0:0`, `1024/1024` ranks, `1024` accepted, zero stale/failed/timed-out updates, run-local latest/checkpoint only. |

Exact prior rung artifacts:

- 1n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_1n/4954252-20260708T005658Z/summaries/summary.md`
- 1n Slurm stdout/stderr:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4954252.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4954252.err`
- 2n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_async_quorum_2n/4954257-20260708T010207Z/summaries/summary.md`
- 2n Slurm stdout/stderr:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4954257.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4954257.err`
- 8n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/summaries/summary.md`
- 8n metrics/manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/metrics.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/manifest.json`
- 64n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/summaries/summary.md`
- 64n metrics/manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/metrics.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/manifest.json`
- 128n summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/summaries/summary.md`
- 128n metrics/manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/metrics.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_128n/4954539-20260708T025917Z/artifacts/manifest.json`
- 128n report, currently on the 128n task branch:
  `origin/wg/agent-848/run-compiled-helper-128n:reports/frontier/run-compiled-helper-128n-20260708.md`

The original `validate-compiled-helper` job `4953961` failed at 1n, but the
follow-up `implement-compiled-mpich-2` replacement ladder passed both 1n and
2n before `run-compiled-helper` submitted 8n/64n.

## 256n debug submission

Slurm job: `4954634`

Partition/QOS: `batch` / `debug`

Requested walltime: `00:20:00`

Estimated node-hours: `256 * 20 / 60 = 85.333333`

Slurm accounting:

```text
4954634|compiled-helper-trainpy-256n-debug|COMPLETED|0:0|batch|debug|00:06:47|00:20:00|256|2026-07-07T23:24:12|2026-07-07T23:30:59
```

Actual node-hours: `256 * 407 / 3600 = 28.942222`

Submission command:

```bash
sbatch --parsable -N 256 -J compiled-helper-trainpy-256n-debug -t 00:20:00 \
  --export=ALL,WG_TASK_ID=evaluate-compiled-helper,SMOKE_NAME=256n,SMOKE_NODE_COUNT=256,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=2048,ASYNC_EXPECTED_RANKS=2048,ASYNC_GLOBAL_QUORUM=2048,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=1200,HUMAN_APPROVAL_RECORD='WG evaluate-compiled-helper: bounded 256-node compiled MPICH helper train.py debug smoke after clean 1n/2n/8n/64n/128n; batch/debug 00:20:00; run-local latest/checkpoint only; no production latest/last or shared chain-pointer mutation; no 1h+ approval implied.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Seed checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
```

Output root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z
```

Primary 256n artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/summaries/summary.md`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/metrics.json`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/manifest.json`
- Command capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/command.txt`
- Environment capture:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/env.txt`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/logs/trainpy_async_quorum.log`
- Slurm stdout/stderr:
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-256n-debug-4954634.out`,
  `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-256n-debug-4954634.err`
- Helper binary and shared library:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/compiled_mpich_dense_helper`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/artifacts/compiled_mpich_dense_helper.so`

## 256n metrics

Wrapper validation: `pass`

Terminal state: `COMPLETED`, exit `0:0`.

| Field | Value |
| --- | ---: |
| Ranks started | `2048 / 2048` |
| Requested workers | `2048` |
| Participating workers | `2048` |
| Quorum | `2048 / 2048` |
| Quorum status | `advanced` |
| Accepted updates | `2048` |
| Stale updates | `0` |
| Failed updates | `0` |
| Timed-out updates | `0` |
| Invalid updates | `0` |
| Tokens per generation | `264,192` |
| Tokens/sec | `3,600.351597` |
| Generation/reduce duration | `73.3795 s` |
| Merge duration | `5.224699 s` |
| Reduce bucket count | `80` |
| Per-bucket reduce latency min/median/max | `0.036637 / 0.619422 / 5.39255 s` |
| Accepted dense delta bytes | `5,506,770,496` |
| MPI reduce aggregate bytes | `5,506,770,496` |
| MPI reduce payload sent bytes | `11,277,865,975,808` |
| Loss window | `loss=13.8485`, `loss_100=13.8485` |
| Helper MPI world size | `2048` |
| Helper reducer | `mpi_reduce_bucketed_weighted_sum` |
| Helper collective | `MPI_Reduce` |
| Strict collective all launched ranks | `true` |
| File gather diagnostic path | `0` |

Latest/checkpoint behavior:

- Run-local latest advanced to generation `0`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z/async_run/latest.json`
- Checkpoint/publication records were written under the run root only:
  - `async_run/generations/gen_000000/manifest.json`
  - `async_run/recovery_checkpoints/gen_000000/initial.json`
  - `async_run/export_checkpoints/gen_000000/initial.json`
  - `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

## Production and latest guard

Guard outcome: **pass**.

- The 256n smoke used `batch` / `debug`, `00:20:00`, and did not request a
  1h+ walltime or production QOS.
- The job used run-local output under
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_256n_debug/4954634-20260708T032415Z`.
- `env.txt` records the only checkpoint input as the current verified seed:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`.
- `manifest.json` records `latest_path` under the 256n debug run root.
- The human approval record explicitly states no production latest/last or
  shared chain-pointer mutation and no 1h+ approval implied.
- No production latest/last output path was passed to the job, and the recorded
  latest/checkpoint artifacts are all under the debug run root.
- `filesystem_live_quorum=false`; the dense data plane is the compiled MPICH
  collective reducer, not live Lustre update collection.

## Recommendation

Do **not** submit any further job from this task.

Recommended next action: request explicit human approval for a **256n x 1h**
debug continuation if the project wants a longer stability check at this node
count. The short 256n debug rung passed cleanly, so repeating the same 10-20
minute 256n smoke is not necessary unless a human wants a duplicate sample.

Do not request or submit **256n x 12h**, production-QOS, larger-than-256n, or
any production latest/last mutation until after a human reviews this 256n debug
result and explicitly approves the next rung.
