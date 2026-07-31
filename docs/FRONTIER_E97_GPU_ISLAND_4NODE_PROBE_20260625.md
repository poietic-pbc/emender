# Frontier E97 GPU-Island No-DDP 4-Node Probe

Task: `run-e97-gpu`
Date: 2026-06-25

## Verdict

Pass. The audited `DILOCO_ISLAND_SIZE=1` GPU-island/no-DDP E97-MLP probe was
submitted exactly once as a bounded 4-node Frontier training job, completed
cleanly, produced valid final/latest checkpoints, and passed the fixed
source-vs-candidate eval gate by a large margin.

GPU-island/no-DDP is operationally clean and promising enough for the dependent
8-node probe (`run-e97-gpu-2`) under the same controlled E97-MLP/no-DDP scope.

No 8-node, 16-node, 32-node, or 64-node training job was submitted from this
task. No GDN2, CMAES, schedule-free outer, momentum outer, partial-average, or
other uncontrolled variant was submitted. `run-64-node-e97` was checked after
the run and remained `open (PAUSED)`.

## Audit Gate

Read: `docs/FRONTIER_E97_GPU_ISLAND_AUDIT_20260625.md`.

The audit confirmed that `--diloco` sets `use_ddp=False`, and that the DiLoCo
hybrid DDP wrapper is only created when `diloco_island_size > 1`. Therefore
`DILOCO_ISLAND_SIZE=1` skips within-island DDP wrapping/all-reduce rather than
creating a singleton DDP no-op. The audit recommended the controlled 4-node
E97-MLP `DILOCO_K=80`, `DILOCO_OUTER_OPTIMIZER=avg` probe from the 16-node avg
source checkpoint, followed by the fixed eval tensor used below.

## Training Job

| Field | Value |
| --- | --- |
| Slurm job id | `4899326` |
| Slurm state | `COMPLETED`, exit `0:0` |
| Nodes | `4` Frontier nodes, 32 ranks/GPU islands |
| Partition/QOS | `batch` / `debug` |
| Requested walltime | `00:30:00` |
| Requested node-hours | `2.000000` |
| Actual elapsed | `00:20:22` |
| Actual node-hours | `1.357778` |
| Variant | `e97-MLP` |
| Git commit | `31f35d441d98662ca402af7fb2fec1eb37444e98` |
| Source checkpoint | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899326-20260625T120458Z` |
| Train log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899326-20260625T120458Z/logs/train.log` |
| Slurm stdout | `logs/frontier/scaleout/e97-gpu-island-k80-probe-4899326.out` |
| Slurm stderr | `logs/frontier/scaleout/e97-gpu-island-k80-probe-4899326.err` |

Launch settings matched the audit recommendation:

| Setting | Value |
| --- | --- |
| `SCALEOUT_VARIANT` | `e97-MLP` |
| `DILOCO_ISLAND_SIZE` | `1` |
| `DILOCO_K` | `80` |
| `DILOCO_OUTER_OPTIMIZER` | `avg` |
| `DILOCO_OUTER_LR` | `1.0` |
| `DILOCO_OUTER_BETA` | `0.0` |
| `DILOCO_EXPORT_BASIS` | `x` |
| `BATCH_SIZE` | `1` |
| `CHUNK_SIZE` | `2048` |
| `SAVE_EVERY` | `80` |
| `KEEP_CHECKPOINTS` | `4` |

The run's `args.json` records `_world_size=32`, `diloco_island_size=1`,
`diloco_k=80`, `diloco_outer_optimizer=avg`, `batch_size=1`, and
`chunk_size=2048`.

## Operational Evidence

Training completed without NCCL/RCCL watchdog timeout, collective mismatch, OOM,
non-finite loss, or Python traceback. The only notable warnings were the known
Triton/Python version warnings and the startup ProcessGroupNCCL rank-to-device
warning; these did not prevent clean completion.

No `[DDP] wrapped model` or `[DiLoCo-hybrid]` marker appeared in the Slurm logs.
The logs did show the intended pure DiLoCo path:

- `[DiLoCo] world_size=32 backend=nccl; this is rank 0 on cuda:0`
- periodic model-weight averaging across 32 ranks every 80 steps
- final merge skipped at step 2560 because the last step was already a K-aligned
  consensus merge

Final training counters:

| Metric | Value |
| --- | --- |
| Final step | `2560` |
| `FINAL_LOSS_LAST100` | `5.2398` |
| `DILOCO_MERGES` | `16` |
| `DILOCO_SYNC_TOTAL_S` | `66.585` |
| `DILOCO_SYNC_AVG_MS` | `4161.5` |
| Peak memory | `13117` MB |
| Reserved memory | `19336` MB |

Loss trend was noisy but productive over the short walltime probe. Early logged
losses after resume were around `5.50` to `5.85`, and the final logged window
included many values near `4.73` to `5.38`, with `FINAL_LOSS_LAST100=5.2398`.

Checkpoint/finalization behavior was clean:

- periodic checkpointing occurred every 80 steps at K-aligned merge points;
- bounded retention kept the last four periodic/final checkpoint files plus
  `latest.pt`;
- all 32 `.final_checkpoint_ready/rank_*.ready` markers were present;
- final checkpoint path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899326-20260625T120458Z/train/emender_E97_1.3B_20260625_080611/checkpoint_step_002560_loss_5.2398.pt`;
- `latest.pt` points to `checkpoint_step_002560_loss_5.2398.pt`.

## Fixed Eval

Fixed eval job:

| Field | Value |
| --- | --- |
| Slurm job id | `4899430` |
| Slurm state | `COMPLETED`, exit `0:0` |
| Nodes | `1` |
| Requested walltime | `01:00:00` |
| Requested node-hours | `1.000000` |
| Actual elapsed | `00:01:29` |
| Actual node-hours | `0.024722` |
| Eval mode | `scripts/eval_checkpoint.py --y-mode saved --batch-size 1` |
| Scoring tensor | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |
| CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899326-20260625T120458Z/artifacts/e97_4n_gpu_island_k80_fixed_eval.csv` |

Results:

| Checkpoint | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| 16-node avg source | 1328 | 10.49609756 | 4.36699062 |
| 4-node GPU-island K80 candidate | 2560 | 9.98569679 | 4.15463404 |

Candidate minus source:

| Delta | Value | Gate |
| --- | ---: | --- |
| CE | `-0.51040077` | pass (`<= +0.025`) |
| BPB | `-0.21235658` | pass (`<= +0.010`) |

The eval wrapper's inherited manifest key names the candidate `k80_32node`, but
the path points to this task's 4-node GPU-island checkpoint and the task-specific
CSV is named `e97_4n_gpu_island_k80_fixed_eval.csv`.

## Scope Confirmation

- Submitted exactly one bounded 4-node E97-MLP GPU-island training job:
  `4899326`.
- Submitted one 1-node forward-only fixed eval job: `4899430`.
- Submitted no 8-node, 16-node, 32-node, or 64-node training job from this task.
- Submitted no GDN2/CMAES/non-avg/noise variant from this task.
- Left `run-64-node-e97` paused.
- No extra uncontrolled variable was introduced relative to the audit recipe.

## 8-Node Decision

Proceed to the dependent 8-node GPU-island/no-DDP probe if its task-specific
gate still agrees with this evidence. The 4-node result is operationally clean,
matches the no-DDP singleton-island semantics, and passes the fixed eval gate
with negative CE/BPB deltas versus the source.
