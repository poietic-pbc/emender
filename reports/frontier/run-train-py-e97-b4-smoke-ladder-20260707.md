# run-train-py: train.py real E97 B4 smoke ladder

Task: `run-train-py`
Date: 2026-07-07 UTC
Decision: **PASS for train.py baseline through 8 nodes debug-QOS**. Do not treat this as approval for 64n/256n/production; this task submitted only 1n, 2n, and 8n bounded debug-QOS jobs.

## Guardrails

- Async/quorum harness: **not used**. No submission invoked `scripts/frontier/e97_async_diloco_train.py` or the async file-quorum path.
- Launcher path: `scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch` execs the existing Frontier train.py wrapper style, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`, which runs `python -u train.py`.
- Slurm GPU binding: all submitted jobs used `#SBATCH --ntasks-per-node=8`, `--gpus-per-task=1`, and `--gpu-bind=closest`; the wrapper's recorded `srun` command uses `srun -N <nodes> -n <nodes*8> -c7 --gpus-per-task=1 --gpu-bind=closest ... python -u train.py`.
- Recipe: `BATCH_SIZE=4`, `CHUNK_SIZE=2048`, `grad_accum=1`, `DILOCO_K=40`, `DILOCO_OUTER_OPTIMIZER=avg`, no per-step DDP, model averaging every 40 local optimizer steps.
- Stable E97 flags preserved: `--linear_state 0`, `--use_chunked_e97 0`, `--checkpoint_interval 16`, `--use_triton 1`, `--bf16`.
- Data: `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt` plus validation smoke file `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt`.
- Checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`, resolving to `checkpoint_step_1065000_loss_2.5386.pt`; size `7719679924` bytes; manifest SHA256 `c68ea2d95f2721f1f52664f71c1453e4f30a5520b33eb1cf54974185e5a100a4`.
- No 64n, 256n, or production job was submitted.

## Environment

Common runtime captured in each run's `artifacts/env.txt`:

- Host submitter: `login04.frontier.olcf.ornl.gov`.
- Python executable: `/sw/frontier/miniforge3/23.11.0-0/bin/python`.
- Python: `3.10.13`.
- Torch: `2.8.0.dev20250422+rocm6.4`.
- HIP/ROCm reported by torch: `6.4.43482-0f2d60242`.
- Triton: `3.2.0`.
- `FRONTIER_RCCL_ENV=recommended`.
- `rccl_net_plugin_status=not-found`; `REQUIRE_RCCL_NET_PLUGIN` was not set, matching the existing wrapper's non-fatal behavior.
- Runtime log confirmed E97 path: `backend=hip path=e88-sequential-split-edit-triton use_triton=True use_chunked_e97=False e97_chunk_size=32 linear_state=False raw_write=False`.

## Submitted Commands

1-node:

```bash
sbatch scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch
```

2-node:

```bash
sbatch -N 2 -J e97-s1065-b4-k40-trainpy-2n --export=ALL,WG_TASK_ID=run-train-py,SCALEOUT_VARIANT=E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n,REQUESTED_NODE_HOURS=0.666667,HUMAN_APPROVAL_RECORD='WG run-train-py: bounded 2-node debug-QOS train.py smoke after clean 1n finite-loss/merge evidence; refreshed E97 step1065000 latest.pt; no async/quorum harness; no 64n/256n/production approval.' scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch
```

The first 2-node attempt was made while the 1-node allocation was still active and Slurm rejected it with `QOSMaxSubmitJobPerUserLimit`; it was retried after 1n completion and accepted.

8-node:

```bash
sbatch -N 8 -J e97-s1065-b4-k40-trainpy-8n --export=ALL,WG_TASK_ID=run-train-py,SCALEOUT_VARIANT=E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n,REQUESTED_NODE_HOURS=2.666667,HUMAN_APPROVAL_RECORD='WG run-train-py: bounded 8-node debug-QOS train.py smoke after clean 1n and 2n finite-loss/merge evidence; refreshed E97 step1065000 latest.pt; no async/quorum harness; no 64n/256n/production approval.' scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch
```

## Results

| Arm | Job | Slurm state | Elapsed | Rank-start evidence | Train steps | Loss window | Median global tok/s | DiLoCo merge evidence | Final checkpoint |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| 1n | `4951373` | `COMPLETED 0:0` | `00:10:19` | 8 ranks on `frontier09090` | `1065001`-`1065250` | first: `1.7868, 2.8044, 2.7493, 2.5324, 2.3819`; last: `2.3037, 2.7924, 2.7096, 1.9795, 2.3213`; min/median/max `1.3995/2.4577/3.6639` | `37446` | 6 merges across 8 ranks; durations `1224, 181, 181, 181, 180, 181 ms` | `checkpoint_step_1065250_loss_2.4653.pt`; `latest.pt` updated in run dir |
| 2n | `4951383` | `COMPLETED 0:0` | `00:10:21` | 16 ranks on `frontier07733,frontier07734` | `1065001`-`1065260` | first: `1.7868, 2.5466, 2.2184, 2.5493, 2.7716`; last: `2.2533, 1.8187, 2.9917, 2.7856, 2.2417`; min/median/max `1.3690/2.4882/3.6423` | `86897` | 6 merges across 16 ranks; durations `2714, 3015, 1534, 1826, 2282, 2545 ms` | `checkpoint_step_1065260_loss_2.4711.pt`; `latest.pt` updated in run dir |
| 8n | `4951400` | `COMPLETED 0:0` | `00:10:16` | 64 ranks on `frontier00007,00139,00265,00391,00613,00645,00773,00900` | `1065001`-`1065250` | first: `1.7868, 2.8771, 1.9304, 2.5458, 2.3864`; last: `2.6377, 1.8742, 2.2665, 2.8211, 3.0909`; min/median/max `1.4111/2.5434/3.6841` | `353931` | 6 merges across 64 ranks; durations `3663, 5256, 3626, 3553, 3385, 3020 ms` | `checkpoint_step_1065250_loss_2.5069.pt`; `latest.pt` updated in run dir |

Throughput notes:

- 1n global tok/s median was `37,446`; observed max was `44,621`.
- 2n global tok/s median was `86,897`; observed max was `89,712`.
- 8n global tok/s median was `353,931`; observed max was `362,082`.
- The periodic post-checkpoint step immediately after a merge/checkpoint is slower, as expected; medians above include all logged training steps.

## Artifacts

1-node run root:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z`
- Env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z/artifacts/env.txt`
- Rank start: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z/artifacts/rank-start.tsv`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z/logs/train.log`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z/summaries/summary.md`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke/4951373-20260707T101721Z/artifacts/manifest.json`

2-node run root:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z`
- Env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z/artifacts/env.txt`
- Rank start: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z/artifacts/rank-start.tsv`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z/logs/train.log`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z/summaries/summary.md`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_2n/4951383-20260707T102902Z/artifacts/manifest.json`

8-node run root:

- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z`
- Env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z/artifacts/env.txt`
- Rank start: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z/artifacts/rank-start.tsv`
- Train log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z/logs/train.log`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z/summaries/summary.md`
- Manifest: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n/4951400-20260707T104124Z/artifacts/manifest.json`

Slurm stdout/stderr are under `logs/frontier/scaleout/e97-s1065-b4-k40-trainpy-{1n,2n,8n}-<jobid>.{out,err}` in the submit worktree.

## Recommendation

The train.py DiLoCo path is re-established as a clean real-training baseline through 8 nodes with refreshed E97 step1065000, B4/chunk2048, real commapile data, finite losses, successful K40 periodic weight averaging, and run-local final checkpoints.

For any next production-shape decision:

- Prefer a **32-node train.py debug or short batch pilot** before considering 64n, because the current evidence stops at 8n by task constraint.
- Keep K40 initially; 8n K40 merge timings were acceptable at `3.0`-`5.3 s` per merge across 64 ranks, and throughput scaled to a median `353,931` global tok/s.
- Preserve the same wrapper and flags, but consider setting `REQUIRE_RCCL_NET_PLUGIN=1` only after resolving `librccl-net.so` discovery; this ladder was clean without requiring the plugin, but the env consistently reports `rccl_net_plugin_status=not-found`.
- Do not advance to 64n/256n/production without explicit human approval and a separate allocation/ledger entry.
