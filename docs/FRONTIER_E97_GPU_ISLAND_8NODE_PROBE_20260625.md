# Frontier E97 GPU-Island No-DDP 8-Node Probe

Task: `run-e97-gpu-2`
Date: 2026-06-25

## Verdict

Do not proceed to a later 16-node or 32-node GPU-island/no-DDP task from this
row.

The required 4-node gate was read before launch and was clean/promising:
`docs/FRONTIER_E97_GPU_ISLAND_4NODE_PROBE_20260625.md` reported job `4899326`
`COMPLETED 0:0`, valid final/latest checkpoint behavior, `DILOCO_MERGES=16`,
and green fixed eval deltas (`CE -0.51040077`, `BPB -0.21235658`) versus the
same 16-node source checkpoint. That authorized this single bounded 8-node
probe under the same audited E97-MLP GPU-island/no-DDP config family.

The 8-node training job itself completed cleanly and confirmed the intended
64-rank GPU-island/no-DDP path. However, the fixed source-vs-candidate eval
failed the gate by a clear margin:

| Delta | Value | Gate | Result |
| --- | ---: | --- | --- |
| CE, candidate - source | `+0.16329194` | `<= +0.025` | fail |
| BPB, candidate - source | `+0.06793899` | `<= +0.010` | fail |

This is useful negative scale evidence: the 4-node no-DDP row was promising, but
doubling to 64 independent GPU islands from the same source with the same K80
avg cadence did not preserve fixed-eval quality.

## Training Job

| Field | Value |
| --- | --- |
| Slurm job id | `4899453` |
| Slurm state | `COMPLETED`, exit `0:0` |
| Nodes | `8` Frontier nodes, 64 ranks/GPU islands |
| Partition/QOS | `batch` / `debug` |
| Requested walltime | `00:30:00` |
| Requested node-hours | `4.000000` |
| Actual elapsed | `00:20:17` |
| Actual node-hours | `2.704444` |
| Variant | `e97-MLP` |
| Git commit | `af4418acd3510425ae2eb601be759a9d82e9ca89` |
| Source checkpoint | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899453-20260625T123949Z` |
| Train log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899453-20260625T123949Z/logs/train.log` |
| Slurm stdout | `logs/frontier/scaleout/e97-gpu-island-8n-k80-probe-4899453.out` |
| Slurm stderr | `logs/frontier/scaleout/e97-gpu-island-8n-k80-probe-4899453.err` |

Launch settings matched the 4-node audited config family except for node count:

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

The run's `args.json` records `_world_size=64`, `diloco_island_size=1`,
`diloco_k=80`, `diloco_outer_optimizer=avg`, `batch_size=1`, `chunk_size=2048`,
`save_every=80`, `keep_checkpoints=4`, and the source checkpoint above.

## Operational Evidence

Training completed without NCCL/RCCL watchdog timeout, collective mismatch, OOM,
non-finite loss, Python traceback, or runtime error. The only notable stderr
items were module reload messages, a low-power GPU warning, and the known
Triton/Python version warnings during eval; these did not block completion.

The logs show the intended no-DDP path:

- `[DiLoCo] world_size=64 backend=nccl; this is rank 0 on cuda:0`
- `[DiLoCo] periodic model-weight averaging: K=80 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)`
- `[DiLoCo] broadcast rank-0 W_0 to all 64 ranks (identical start)`
- `[DiLoCo] outer optimizer: avg (stateless periodic averaging)`

The matched log search found no `[DDP] wrapped model` or `[DiLoCo-hybrid]`
markers.

Final training counters:

| Metric | Value |
| --- | ---: |
| Final step | `2535` |
| `FINAL_LOSS_LAST100` | `5.5594` |
| `DILOCO_MERGES` | `16` |
| `DILOCO_SYNC_TOTAL_S` | `67.382` |
| `DILOCO_SYNC_AVG_MS` | `4211.4` |
| Peak memory | `13117` MB |
| Reserved memory | `19336` MB |

Loss trend was noisy and not clearly productive at this scale. There were 242
rank-0 logged training losses. The first 20 logged losses averaged `5.3428`, the
last 20 averaged `5.5979`, and the recorded last-100 average was `5.5594`.
Late logged samples included `step 2490 loss 5.7418`, `2500 loss 5.9370`,
`2520 loss 6.1322`, and `2535 loss 5.2335`.

Checkpoint/finalization behavior was clean:

- periodic checkpointing occurred at K-aligned merge points;
- bounded retention kept the last three periodic checkpoints plus final
  checkpoint and `latest.pt`;
- all 64 `.final_checkpoint_ready/rank_*.ready` markers were present;
- finalization ran a final consensus merge at non-K-aligned walltime stop step
  `2535`;
- final checkpoint path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899453-20260625T123949Z/train/emender_E97_1.3B_20260625_084058/checkpoint_step_002535_loss_5.5594.pt`;
- `latest.pt` resolves to `checkpoint_step_002535_loss_5.5594.pt`.

## Fixed Eval

Fixed eval job:

| Field | Value |
| --- | --- |
| Slurm job id | `4899491` |
| Slurm state | `COMPLETED`, exit `0:0` |
| Nodes | `1` |
| Requested walltime | `01:00:00` |
| Requested node-hours | `1.000000` |
| Actual elapsed | `00:01:26` |
| Actual node-hours | `0.023889` |
| Eval mode | `scripts/eval_checkpoint.py --y-mode saved --batch-size 1` |
| Scoring tensor | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |
| CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899453-20260625T123949Z/artifacts/e97_8n_gpu_island_k80_fixed_eval.csv` |
| Eval manifest | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/e97-MLP/4899453-20260625T123949Z/artifacts/e97_gpu_island_fixed_eval_manifest.json` |
| Slurm stdout | `logs/frontier/eval/e97-gpu-island-8n-fixed-eval-4899491.out` |
| Slurm stderr | `logs/frontier/eval/e97-gpu-island-8n-fixed-eval-4899491.err` |

Results:

| Checkpoint | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| 16-node avg source | 1328 | 10.49609756 | 4.36699062 |
| 8-node GPU-island K80 candidate | 2535 | 10.65938950 | 4.43492961 |

Candidate minus source:

| Delta | Value | Gate |
| --- | ---: | --- |
| CE | `+0.16329194` | fail (`<= +0.025`) |
| BPB | `+0.06793899` | fail (`<= +0.010`) |

## Scope Confirmation

- Read the 4-node result before deciding; it explicitly authorized this
  dependent 8-node probe.
- Submitted exactly one bounded 8-node E97-MLP GPU-island/no-DDP training job:
  `4899453`.
- Submitted one 1-node forward-only fixed eval job: `4899491`.
- Slurm accounting for the task window showed only those task-relevant jobs:
  `4899453` and `4899491`.
- Submitted no 16-node, 32-node, or 64-node GPU-island/no-DDP training job from
  this task.
- Submitted no GDN2, CMAES, schedule-free outer, momentum outer,
  partial-average, or other uncontrolled variant from this task.
- `run-64-node-e97` was checked after the run and remained `open (PAUSED)`.

## Scale Decision

GPU-island/no-DDP does not deserve an immediate later 16-node or 32-node task
from this evidence. The topology is operationally viable at 8 nodes, but the
same K80 avg continuation from the same 16-node source regressed on the fixed
source-vs-candidate eval. A later follow-up should first investigate cadence or
quality behavior at small scale rather than spend larger node counts on the
current no-DDP K80 recipe.
