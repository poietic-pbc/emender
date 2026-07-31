# E97 1.3B Pretrained 8-Node K320 Probe - 2026-06-25

WG task: `run-e97-1-3b-3`

## Status

The single permitted 8-node K320 training job completed cleanly, but the fixed
source-vs-candidate eval failed the non-regression gate by a wide margin.

Verdict: do not promote this K320 recipe to 16-node continuation.

| Delta | Value | Gate | Result |
| --- | ---: | --- | --- |
| CE, candidate - source | `+0.30940366` | `<= +0.025` | fail |
| BPB, candidate - source | `+0.12873002` | `<= +0.010` | fail |

## Authorization

`validate-e97-1-3b` explicitly authorized `run-e97-1-3b-3` as one bounded
8-node pretrained GPU-island/no-DDP K320 probe from the staged high-quality
checkpoint bundle. It did not authorize 16/32/64-node continuation, GDN2/CMAES,
or schedule-free outer work.

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
```

Fixed eval tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

## Launch Envelope

| Field | Value |
| --- | --- |
| Launcher | `scripts/frontier/e97_1p3b_pretrained_8n_k320.sbatch` |
| Nodes | 8 Frontier nodes, 64 singleton GPU islands |
| DDP | none within island; `DILOCO_ISLAND_SIZE=1` |
| `DILOCO_K` | `320` |
| `SAVE_EVERY` | `320` |
| Outer optimizer | `avg`, `outer_lr=1.0`, `outer_beta=0.0`, export basis `x` |
| `BATCH_SIZE` | `1` |
| `CHUNK_SIZE` | `2048` |
| `TRAIN_MINUTES` | `50` |
| Walltime | `01:00:00` |
| Requested node-hours | `8.000000` |
| Training data | `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt` |
| Validation text | `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt` |
| Git commit at launch | `e6d4468446344dbf34309dc439bf46d79b546093` |

## Training Result

| Field | Value |
| --- | --- |
| Job id | `4901367` |
| Queue/QOS | `batch` / `debug` |
| Slurm state | `COMPLETED` |
| Exit code | `0:0` |
| Actual elapsed | `00:50:17` |
| Actual node-hours | `6.704444` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z` |
| Final step | `263947` |
| New steps from source | `3447` |
| Loss trend | 689 rank-0 logged losses; first-20 avg `2.7035`, last-20 avg `2.5451`, final last-100 `2.6770` |
| Throughput/global_tok/s | median `165734`, mean `161693`, last-20 mean `166255` |
| `DILOCO_MERGES` | `11` |
| `DILOCO_SYNC_TOTAL_S` | `49.402` |
| `DILOCO_SYNC_AVG_MS` | `4491.1` |
| Checkpoint/finalization/latest behavior | K-aligned periodic saves at steps `263040`, `263360`, `263680`; final consensus merge and checkpoint at step `263947`; `latest.pt -> checkpoint_step_263947_loss_2.6770.pt`; 64/64 final-ready markers present |
| Warnings/errors | No traceback, non-finite loss, watchdog timeout, OOM, or RCCL/NCCL fatal error. Warnings were module reload messages, Triton/Python version warnings, and the known `ProcessGroupNCCL` rank/device warning. |

Training artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/logs/train.log
```

Slurm logs:

```text
logs/frontier/scaleout/e97-1p3b-8n-k320-4901367.out
logs/frontier/scaleout/e97-1p3b-8n-k320-4901367.err
```

The log confirmed the intended no-DDP singleton-island path:

```text
[DiLoCo] world_size=64 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=320 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
[DiLoCo] broadcast rank-0 W_0 to all 64 ranks (identical start)
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

## Fixed Eval

| Field | Value |
| --- | --- |
| Eval job id | `4901744` |
| Eval state | `COMPLETED`, exit `0:0` |
| Eval elapsed / actual node-hours | `00:01:24` / `0.023333` |
| Eval CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/artifacts/e97_1p3b_8n_k320_fixed_eval.csv` |
| Eval manifest | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k320/4901367-20260625T192329Z/artifacts/e97_1p3b_8n_k320_fixed_eval_manifest.json` |
| Source CE/BPB | `26.93483090` / `11.20646537` |
| Candidate CE/BPB | `27.24423456` / `11.33519539` |
| Candidate - source CE delta | `+0.30940366` |
| Candidate - source BPB delta | `+0.12873002` |
| Non-regression gate | fail |

Gate:

- BPB delta must be `<= +0.010`.
- CE delta must be `<= +0.025`.

## Scale Decision

Do not promote this K320 recipe to 16-node continuation. The job was
operationally healthy and the train-loss window improved mildly, but the
row-matched fixed eval regressed far beyond both non-regression thresholds.
Within the neutral K bracket, this is useful negative evidence for K320 rather
than a scaleout candidate.

## Scope Confirmation

- Read `validate-e97-1-3b`; it explicitly authorized this exact 8-node K320
  pretrained probe.
- Submitted exactly one actual bounded 8-node E97-MLP GPU-island/no-DDP K320
  training job from this task: `4901367`.
- An earlier `sbatch` attempt was rejected before job creation by
  `QOSMaxSubmitJobPerUserLimit`; no Slurm job id was assigned for that attempt.
- Submitted one one-node forward-only fixed eval job from this task: `4901744`.
- Submitted no 16/32/64-node continuation job from this task.
- Submitted no GDN2/CMAES job from this task.
- Submitted no schedule-free outer job from this task.
- `run-64-node-e97` remained `open (PAUSED)` after the run.
