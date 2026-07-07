# Post-patch MPI dense smoke ladder, 2026-07-07

Task: `run-post-patch`

## Outcome

Stopped after the 2-node rung failed. No 8-node, 64-node, or 256-node job was
submitted.

Failure class: runtime/MPI. The 2-node payload reached real train.py execution
and all 16 launched ranks wrote rank-start/progress evidence, then Cray MPICH
aborted during MPI initialization with:

```text
Fatal error in PMPI_Init_thread
open_fabric(1559)...........: OFI fi_getinfo() failed (ofi_init.c:1559:open_fabric:No data available)
```

Next fix: diagnose Frontier OFI/MPICH initialization for the 2-node
`srun -N 2 -n 16` Python/mpi4py path before retrying 2n. The code path already
uses host-staged MPI dense buckets with `MPICH_GPU_SUPPORT_ENABLED=0`; the next
check should compare a minimal 2-node mpi4py init/send smoke under the same
module stack and launcher options, including any required Frontier network or
MPICH environment knobs.

## Rung 1: adopted 1n job `4953629`

Status:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953629|trainpy-async-quorum-1n|batch|debug|COMPLETED|0:0|00:05:21|1|frontier08198
4953629.0|bash|||COMPLETED|0:0|00:05:00|1|frontier08198
```

Submit provenance:

- Submit worktree: `.wg-worktrees/agent-788`
- Branch/commit: `wg/agent-788/implement-frontier-mpi` /
  `fae3a26c041deaf684ac420111ff3d50793341c3`
- Queue/QOS: `batch` / `debug`
- Command intent: `ASYNC_QUORUM_TRANSPORT=mpi-dense`, run-local latest only,
  no production latest mutation.

Artifacts:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/summaries/summary.md`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/manifest.json`
- Metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/metrics.json`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/artifacts/rank-start.tsv`
- Stdout/stderr:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4953629.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4953629.err`

Gate evidence:

- Wrapper validation: `pass`
- Rank starts: `8 / 8`
- Accepted updates: `8`
- Timed-out updates: `0`
- Tokens: `1032`
- Latest behavior: run-local latest advanced at
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_1n_postpatch/4953629-20260707T202832Z/async_run/latest.json`
- Checkpoint behavior: run-local generation/recovery/export checkpoint
  manifests were written under the same debug run root.
- MPI dense metrics: `transport.dense_data_plane=true`,
  `transport.filesystem_live_quorum=false`,
  `update_bytes.mpi_dense_payload_sent=44054163968`,
  `update_bytes.mpi_dense_payload_received=44054163968`.

Decision: pass. This satisfied the task gate for submitting exactly the next
2-node rung.

## Rung 2: submitted 2n job `4953646`

Exact submit command:

```bash
sbatch --parsable --export=ALL,WG_TASK_ID=run-post-patch,TASK_ID=run-post-patch,SMOKE_NAME=2n-mpi-dense-postpatch,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch,ASYNC_QUORUM_TRANSPORT=mpi-dense,HUMAN_APPROVAL_RECORD='WG run-post-patch: 2-node postpatch MPI dense debug smoke after 1n job 4953629 passed; debug QOS; run-local latest only; no production latest mutation authorized.' scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Submission details:

- Job id: `4953646`
- Queue/QOS: `batch` / `debug`
- Requested nodes/walltime: `2` nodes x `00:20:00`
- Expected node-hours: `0.666667`
- Branch/commit at submit: `wg/agent-792/run-post-patch` /
  `3ec982795f97424e9583cbd459d353f8c9d34ae6`
- Rung conditionality: submitted only after the adopted 1n job `4953629`
  passed wrapper validation.

Status:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|NodeList
4953646|trainpy-async-quorum-2n|batch|debug|FAILED|90:0|00:01:42|2|frontier[07442,07446]
4953646.0|bash|||FAILED|255:0|00:01:19|2|frontier[07442,07446]
```

Artifacts:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/summaries/summary.md`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/manifest.json`
- Env/command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/env.txt`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/command.txt`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/artifacts/rank-start.tsv`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_mpi_dense_2n_postpatch/4953646-20260707T203737Z/logs/trainpy_async_quorum.log`
- Stdout/stderr:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4953646.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4953646.err`

Failure evidence:

- Wrapper validation: `fail`
- Rank starts: `16 / 16`
- Per-rank progress: heartbeat files for `node-00000` through `node-00015`
  show `stage="mpi_dense_send_starting"`, `transport="mpi-dense"`,
  `global_quorum=16`, and `mpi_bucket_bytes=67108864`.
- Metrics: `metrics.json` was missing/empty, so no accepted updates, tokens,
  latest advancement, checkpoint paths, or MPI dense byte counters were
  recorded for 2n.
- Summary validation errors:
  `payload_exit_status=255`, `metrics_json_missing_or_empty`,
  `no_accepted_updates`, `no_training_tokens_recorded`,
  `latest_not_advanced`, `checkpoint_paths_missing`.
- Primary runtime error: repeated Cray MPICH aborts in `PMPI_Init_thread` with
  `MPIDI_OFI_mpi_init_hook` / `OFI fi_getinfo() failed`.

Decision: fail. The ladder stops at 2n. The 8n and 64n rungs were not eligible
and were not submitted.

## Production latest behavior

No production latest path was intentionally mutated. Both submitted rungs used
the debug trainpy async quorum root and run-local latest/checkpoint publication.
The checkpoint input path was read as a seed:

`/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`

At the time of this report that seed latest symlink/file had stat evidence:

```text
size=38
mtime=2026-07-06 05:30:48 -0400
```

The adopted 1n rung advanced only its run-local latest. The failed 2n rung did
not create a run-local `latest.json`.
