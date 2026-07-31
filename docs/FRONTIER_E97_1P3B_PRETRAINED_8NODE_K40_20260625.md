# Frontier E97 1.3B Pretrained 8-Node K40 Probe - 2026-06-25

WG task: `run-e97-1-3b`

## Verdict

The staged validation artifact explicitly authorized this bounded 8-node
pretrained cadence probe: `docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`
lists `run-e97-1-3b` as the K40 member of the authorized 8-node
GPU-island/no-DDP bracket.

The K40 recipe is a candidate for the downstream 16-node continuation decision
only as a gated bracket row, not as a standalone winner. It completed cleanly,
kept the loss trend finite and slightly improved over the first logged window,
and passed the fixed non-regression gate:

| Delta | Value | Gate | Result |
| --- | ---: | --- | --- |
| CE, candidate - source | `+0.00803614` | `<= +0.025` | pass |
| BPB, candidate - source | `+0.00334350` | `<= +0.010` | pass |

## Training Launch

Submitted exactly one bounded 8-node E97-MLP GPU-island/no-DDP K40 training job:

| Field | Value |
| --- | --- |
| Slurm job id | `4900838` |
| Job name | `e97-1p3b-8n-k40` |
| State / exit | `COMPLETED` / `0:0` |
| Queue / QOS | `batch` / `debug` |
| Nodes | `8` Frontier nodes, `64` single-GPU islands |
| Walltime request | `01:00:00` |
| Train minutes | `50` |
| Requested node-hours | `8.0` |
| Actual elapsed | `00:50:17` |
| Actual node-hours | `6.704444` |
| Git commit at launch | `db34e0ee75d3f47048b85b6a8c836d956e6209c1` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z` |
| Train log | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/logs/train.log` |
| Manifest | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/artifacts/manifest.json` |
| Slurm stdout | `logs/frontier/scaleout/e97-1p3b-8n-k40-4900838.out` |
| Slurm stderr | `logs/frontier/scaleout/e97-1p3b-8n-k40-4900838.err` |

Exact launch environment and paths:

| Setting | Value |
| --- | --- |
| `RESUME_CHECKPOINT` | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt` |
| `DATA` | `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt` |
| `VAL_DATA` | `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt` |
| `DILOCO_ISLAND_SIZE` | `1` |
| `DILOCO_K` | `40` |
| `DILOCO_OUTER_OPTIMIZER` | `avg` |
| `DILOCO_OUTER_LR` | `1.0` |
| `DILOCO_OUTER_BETA` | `0.0` |
| `DILOCO_EXPORT_BASIS` | `x` |
| `SAVE_EVERY` | `40` |
| `BATCH_SIZE` | `1` |
| `CHUNK_SIZE` | `2048` |
| `KEEP_CHECKPOINTS` | `4` |
| `SCORING_TENSOR` for eval | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt` |

Note: the launcher manifest's `requested_walltime` field inherited
`SLURM_TIMELIMIT=00:20:00` from the script default, but Slurm accounting and
the submitted command show the effective allocation was `01:00:00`. The
manifest's requested seconds/node-hours fields were explicitly overridden to
`3600` and `8.0`.

## Training Result

| Metric | Value |
| --- | ---: |
| Source step | `260500` |
| Final step | `263082` |
| New steps from source | `2582` |
| Rank-0 logged loss count | `516` |
| First-20 logged loss mean | `2.7153` |
| Last-20 logged loss mean | `2.6672` |
| `FINAL_LOSS_LAST100` | `2.6645` |
| Mean `global_tok/s` over logged rows | `139802` |
| Median `global_tok/s` after first 10 rows | `164697` |
| `DILOCO_MERGES` | `66` |
| `DILOCO_SYNC_TOTAL_S` | `273.306` |
| `DILOCO_SYNC_AVG_MS` | `4141.0` |
| Peak memory | `12627` MB |
| Reserved memory | `18196` MB |

Representative first logged losses:

```text
260505 2.8328
260510 2.5319
260515 2.9004
260520 2.9652
260525 2.7705
```

Representative late logged losses:

```text
263040 2.5290
263045 2.4665
263050 2.7980
263055 3.1296
263060 2.4913
263065 2.8499
263070 2.5970
263075 3.0031
263080 2.9509
```

Checkpoint/finalization behavior was clean:

- Periodic K-aligned checkpoint saves occurred through step `263080`.
- Retention kept the last three periodic checkpoints plus the final checkpoint:
  `checkpoint_step_263000_loss_2.5673.pt`,
  `checkpoint_step_263040_loss_2.5290.pt`,
  `checkpoint_step_263080_loss_2.9509.pt`, and
  `checkpoint_step_263082_loss_2.6645.pt`.
- Finalization ran at step `263082` for reason `walltime:SLURM_JOB_END_TIME`.
- The final consensus merge was `FINAL merge #66` across all 64 ranks.
- `latest.pt` resolves to
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/train/emender_E97_1.3B_20260625_140940/checkpoint_step_263082_loss_2.6645.pt`.

Warnings/errors:

- Slurm stderr contained only module replacement/reload messages.
- Stdout included the known Frontier stack warnings for Triton `3.2.0` and
  Python `3.10`, plus pinned-autotune notices.
- No OOM, Python traceback, non-finite loss, NCCL/RCCL watchdog timeout, or
  collective mismatch was found in the inspected logs.

## Fixed Eval

Submitted exactly one post-training fixed source-vs-candidate eval job:

| Field | Value |
| --- | --- |
| Slurm job id | `4901316` |
| Job name | `e97-1p3b-8n-k40-fixed-eval` |
| State / exit | `COMPLETED` / `0:0` |
| Queue / QOS | `batch` / `debug` |
| Nodes | `1` |
| Requested walltime | `01:00:00` |
| Requested node-hours | `1.0` |
| Actual elapsed | `00:01:24` |
| Actual node-hours | `0.023333` |
| Eval mode | `scripts/eval_checkpoint.py --y-mode saved --batch-size 1` |
| CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/artifacts/source_vs_8n_k40_fixed_eval.csv` |
| Eval manifest | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/artifacts/source_vs_8n_k40_fixed_eval_manifest.json` |
| Slurm stdout | `logs/frontier/eval/e97-1p3b-8n-k40-fixed-eval-4901316.out` |
| Slurm stderr | `logs/frontier/eval/e97-1p3b-8n-k40-fixed-eval-4901316.err` |

Results:

| Checkpoint | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| Staged pretrained source | `260500` | `26.93483090` | `11.20646537` |
| 8-node K40 candidate | `263082` | `26.94286704` | `11.20980887` |

Candidate minus source:

| Delta | Value | Gate |
| --- | ---: | --- |
| CE | `+0.00803614` | pass (`<= +0.025`) |
| BPB | `+0.00334350` | pass (`<= +0.010`) |

## Scope Confirmation

- Read the validation report first and proceeded only because it explicitly
  authorized the 8-node pretrained K40 scaleout probe.
- Submitted exactly one bounded 8-node K40 training job from this task:
  `4900838`.
- Submitted exactly one one-node fixed eval job from this task: `4901316`.
- Did not submit a 16-node, 32-node, or 64-node job from this task.
- Did not submit GDN2, CMAES, schedule-free outer, momentum outer, or any other
  uncontrolled variant from this task.
- Slurm accounting for the task window also showed sibling K160/K320 jobs
  (`4900869`, `4901367`) from the parallel K-sweep tasks; they were not launched
  by this task.
- `run-64-node-e97` was checked after the run and remains `open (PAUSED)`.
