# Validate compiled MPICH helper train.py smoke

Date: 2026-07-07

Task: `validate-compiled-helper`

## Verdict

Gate status: **failed at 1n**.

First failing blocker category: **helper IPC / train.py integration**.

The compiled helper built successfully and all 8 train.py ranks started, loaded
real E97 checkpoint/data, completed local train.py work far enough to enter the
compiled-helper send path, and wrote per-rank IPC payload files. The smoke then
failed because the per-rank Python bridge invoked
`compiled_mpich_dense_helper --run-once` as independent subprocesses and each
helper invocation timed out after 150 seconds without producing a quorum result.
No 2n smoke was submitted because the 1n wrapper validation did not pass.

No 8n, 64n, or 256n job was submitted from this task.

## 1n Smoke

Slurm job: `4953961`

State: `FAILED`, job exit `90:0`, payload exit `143`

Partition/QOS: `batch` / `debug`

Nodes: `1`

Elapsed: `00:04:49`

Requested walltime: `00:20:00`

Requested node-hours: `0.333333`

Actual elapsed node-hours: approximately `0.080278`

Node list: `frontier00049`

One rank per GPU: yes, `8` Slurm tasks on one node, `--gpus-per-task=1`,
`--gpu-bind=closest`.

Submission command:

```bash
sbatch --parsable -N 1 -J compiled-helper-trainpy-1n --time=00:20:00 \
  --export=ALL,WG_TASK_ID=validate-compiled-helper,SMOKE_NAME=compiled-helper-1n,SMOKE_NODE_COUNT=1,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_compiled_helper_1n,ASYNC_QUORUM_TRANSPORT=compiled-cray-mpich-helper-p2p,ASYNC_TRAINPY_RANKS=8,ASYNC_EXPECTED_RANKS=8,ASYNC_GLOBAL_QUORUM=8,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=120,HUMAN_APPROVAL_RECORD='WG validate-compiled-helper: 1-node train.py compiled MPICH helper smoke; one train.py rank per GPU; run-local latest/checkpoint only; no production latest mutation.' \
  scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z
```

Primary artifacts:

- Slurm stdout: `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-1n-4953961.out`
- Slurm stderr: `logs/frontier/trainpy_async_quorum/compiled-helper-trainpy-1n-4953961.err`
- Command capture: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/command.txt`
- Environment capture: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/env.txt`
- Wrapper manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/manifest.json`
- Wrapper summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/summaries/summary.md`
- Rank starts: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/rank-start.tsv`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/logs/trainpy_async_quorum.log`
- Helper build log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/logs/compiled_mpich_helper_build.log`
- Helper binary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/compiled_mpich_dense_helper`
- IPC directory: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/async_run/ipc`

Helper build:

```text
CC -O2 -std=c++17 -Wall -Wextra /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-811/scripts/frontier/compiled_mpich_dense_helper.cpp -o /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_compiled_helper_1n/4953961-20260707T224908Z/artifacts/compiled_mpich_dense_helper
```

Rank starts:

```text
8 / 8 ranks started
2026-07-07T22:49:33Z ranks 0-7 on frontier00049, SLURM_NTASKS=8
```

Wrapper validation:

```json
{
  "validation_status": "fail",
  "exit_status": 143,
  "rank_start_count": 8,
  "expected_launched_ranks": 8,
  "accepted_updates": 0,
  "timed_out_updates": 0,
  "tokens_per_generation": 0,
  "validation_errors": [
    "payload_exit_status=143",
    "metrics_json_missing_or_empty",
    "no_accepted_updates",
    "no_training_tokens_recorded",
    "latest_not_advanced",
    "checkpoint_paths_missing"
  ]
}
```

Quorum/helper metrics:

- Configured transport: `compiled-cray-mpich-helper-p2p`
- Configured launched ranks: `8`
- Configured expected ranks: `8`
- Configured global quorum: `8`
- Configured bucket bytes: `67108864`
- Accepted updates: `0`
- Timed-out updates in final metrics: unavailable, because `metrics.json` was
  not produced.
- Helper byte counters: unavailable, because the helper subprocesses timed out
  before writing result JSON.
- Per-rank heartbeat stage: every rank reached
  `stage="compiled_mpich_helper_send_starting"` with
  `transport="compiled-cray-mpich-helper-p2p"`,
  `global_quorum=8`, and `bucket_bytes=67108864`.

Failure excerpt:

```text
subprocess.TimeoutExpired: Command '['.../compiled_mpich_dense_helper',
'--ipc-dir', '.../async_run/ipc',
'--request', '.../async_run/ipc/rank_00000/request.gen000000.json',
'--run-once']' timed out after 150.0 seconds
```

The same timeout was observed for multiple ranks. `srun` then cancelled the
step after rank failures:

```text
srun: error: frontier00049: task 1: Exited with exit code 1
srun: Terminating StepId=4953961.0
STEP 4953961.0 ON frontier00049 CANCELLED ... DUE TO TASK FAILURE
```

Loss window:

- No completed global loss window was produced.
- Local train.py execution reached the training path before helper handoff, but
  the smoke failed before global metrics were written.

Latest/checkpoint behavior:

- Input checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Run-local latest: not advanced; `async_run/latest.json` is absent.
- Checkpoint manifests: none recorded in wrapper manifest.
- Production latest mutation: none was requested or observed. The run used only
  the debug run root above for output.

## 2n Smoke

Not submitted. The task policy requires 2n submission only after a passing 1n
wrapper validation, and job `4953961` failed validation.

## Blocker Classification

This is not a helper build failure: the helper binary built successfully in the
run artifact directory.

This is not a checkpoint/data failure: the real data file and E97 checkpoint
were readable, and train.py reached the local training path.

This is not a scheduler/accounting failure: Slurm allocated the requested debug
QOS node and launched all 8 ranks.

This is not a plain MPI runtime diagnostic failure: the prior helper-only 2n
diagnostic from `integrate-compiled-mpich-main` completed with
`world_size=16`.

The current blocker is the train.py compiled-helper IPC/invocation integration:
each Python rank invokes the helper as a local subprocess, while the helper
expects a coherent MPI world matching the request `world_size`. In the real
train.py smoke this produced no helper quorum result and the subprocesses
timed out after 150 seconds.

## Next Action

Fix the compiled-helper launch/IPC model before retrying this task. A retry
should prove the helper is started as one coherent MPI helper world, or revise
the helper bridge so per-rank helper subprocesses do not wait for unreachable
MPI peers. After that fix, rerun only 1n first; submit 2n only after 1n passes.
