# train.py async quorum 1n/2n smoke launchers

Date: 2026-07-07 UTC  
Task: `add-train-py`

## Scope

This task adds bounded Frontier debug-QOS launchers for the current
train.py-backed async quorum DiLoCo path:

- `scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch`
- `scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch`
- shared runner: `scripts/frontier/trainpy_async_quorum_smoke_common.sh`

Both launchers use one Slurm task per GPU:

- `#SBATCH --ntasks-per-node=8`
- `#SBATCH --gpus-per-task=1`
- `#SBATCH --gpu-bind=closest`
- `srun -N "$SMOKE_NODE_COUNT" -n "$ASYNC_TRAINPY_RANKS" --ntasks-per-node="$RANKS_PER_NODE" --gpus-per-task=1`

The launched payload is `scripts/frontier/e97_async_diloco_train.py` with
`--actual-multinode-file-quorum`. Each Slurm task runs real local train.py
optimizer steps on its assigned GPU (`--device cuda:${SLURM_LOCALID}`), writes a
rank update manifest, and rank 0 publishes authoritative run-local async
checkpoint metadata when quorum is reached.

## Validation Built Into The Launchers

The shared runner fails the smoke if any of these checks fail:

- rank-start count does not match the launched train.py rank count;
- logs contain `[DDP] wrapped model in DistributedDataParallel`;
- logs contain `per-step gradient all-reduce`;
- metrics JSON is missing or empty;
- no accepted updates are recorded;
- no training tokens are recorded;
- `latest.json` is not advanced;
- checkpoint publication paths are missing;
- the 2-node smoke does not record at least one timed-out update.

The 1-node smoke launches 8 train.py-backed ranks and requires 8/8 quorum.

The 2-node smoke launches 16 train.py-backed ranks and configures 17 expected
updates with quorum 16. This intentionally creates one missing/timed-out update
path while still allowing quorum advance without killing the job.

## Data Mode

Default mode is real-token training from:

`/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`

If the data file is unavailable, the runner refuses to start by default. Setting
`ALLOW_SYNTHETIC_TOKEN_FALLBACK=1` enables an explicitly labeled
`synthetic-token-fallback` mode. That fallback is recorded in `env.txt`,
`manifest.json`, and `summary.md`; it is diagnostic only and is not equivalent
to real-token smoke evidence.

## Submission Commands

1-node:

```bash
sbatch scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

2-node:

```bash
sbatch scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Slurm submissions were made from this agent session after launcher fixes.

Initial failed attempts were:

- `4951832`: failed before training because the delegated `srun` shell did not
  receive `ASYNC_ENTRYPOINT`.
- `4951869`: launched 8 ranks but killed the step when failed-rank JSON tried
  to serialize `loss=NaN`.
- `4951912`: launched 8 ranks and wrote durable failed-rank records, but ranks
  1-7 used invalid HIP device ordinals because `--gpus-per-task=1` exposes the
  assigned GPU as `cuda:0` inside each task.

The passing submissions below use the corrected launcher.

## Artifact Ledger

| Arm | Slurm job ID | Status | Rank starts | Accepted/timed-out | Tokens | Metrics JSON | Checkpoint/latest artifact | Logs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1n | `4951952` | `COMPLETED 0:0` | `8/8` | `8/0` | `1032` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4951952-20260707T141231Z/artifacts/metrics.json` | latest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4951952-20260707T141231Z/async_run/latest.json`; checkpoints listed in metrics under `checkpoint_paths` | Slurm: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-1n-4951952.{out,err}`; run log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4951952-20260707T141231Z/logs/trainpy_async_quorum.log`; summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4951952-20260707T141231Z/summaries/summary.md` |
| 2n | `4951983` | `COMPLETED 0:0` | `16/16` | `16/1` | `2064` | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_2n/4951983-20260707T141519Z/artifacts/metrics.json` | latest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_2n/4951983-20260707T141519Z/async_run/latest.json`; checkpoints listed in metrics under `checkpoint_paths` | Slurm: `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-2n-4951983.{out,err}`; run log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_2n/4951983-20260707T141519Z/logs/trainpy_async_quorum.log`; summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_2n/4951983-20260707T141519Z/summaries/summary.md` |

Both passing runs report `ddp_forbidden_line_count=0` in their run manifests;
manual grep for `DDP`, `DistributedDataParallel`, and `per-step gradient
all-reduce` in the run-local logs returned no matches.
