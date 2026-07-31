# E97 1.3B Pretrained Continuation Validation - 2026-06-25

WG task: `validate-e97-1-3b`

## Decision

The staged E97 1.3B continuation checkpoint passed the local bundle,
metadata, fixed-eval, and one-node resume canary gates.

Authorized next jobs: the bounded 8-node pretrained GPU-island/no-DDP cadence
probes `run-e97-1-3b` (K40), `run-e97-1-3b-2` (K160), and
`run-e97-1-3b-3` (K320) may run, subject to their task-local limits and fixed
eval non-regression gates. This is a neutral authorization to probe cadence; it
is not a claim that the more trained checkpoint will scale better.

Do not launch 16/32/64-node continuation, GDN2/CMAES, or schedule-free outer
jobs from this validation. `run-64-node-e97` remains paused.

## Provenance And Local Bundle

Source S3 prefix, recorded only as source provenance:

```text
https://spinozans.s3.us-east-1.amazonaws.com/emender/e97-diloco/levelE97_100m_20260623_103742/step_260500/
```

Renamed local reproducibility bundle:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500
```

Primary local checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
```

Stable alias:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/latest.pt
```

Bundle check:

- S3 ListBucket XML contains 12 objects under the source prefix: `args.json`,
  `bundle_files.sha256`, checkpoint, checkpoint SHA256 sidecar, retention
  report, latest-target snapshot, launch manifest, manifest, metadata checksum
  file, run command/process snapshot, log tail, and verification summary.
- Local directory contains local equivalents for those 12 objects.
- The primary checkpoint is locally renamed; the original S3 checkpoint basename
  is preserved as a local symlink for provenance/checksum compatibility.
- `latest.pt` is a local stable alias and points to the renamed primary
  checkpoint.
- `verification_summary.json` reports `final_s3_object_count=12`,
  `checkpoint_sha256_sidecar_matches_local=true`, and no missing expected
  objects.

Primary checkpoint file validation:

- Size: `7719673482` bytes, matching the required exact size.
- SHA256:
  `46fe086e38b58026910886f3df265d8389bc2b56dc406591e4ea1177645dbfb0`.
- `latest.pt -> checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt`.
- Original basename symlink:
  `checkpoint_step_260500_loss_2.7481.pt -> checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt`.

## Checkpoint Metadata

Loaded under the Frontier module stack:

```text
module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a
torch=2.8.0.dev20250422+rocm6.4
```

Checkpoint keys:

```text
['loss', 'model_state_dict', 'optimizer_state_dict', 'step']
```

Recorded metadata:

- `step`: `260500`
- `loss`: `2.7480917453765867`
- Optimizer state: present, `optimizer_state_dict` with `145` state entries.
- DiLoCo outer state: absent. This is compatible with the requested stateless
  `avg` outer canary.
- Checkpoint does not embed args/config, so validation used the staged sibling
  `args.json`.

Staged architecture args needed for strict compatibility:

- `level=E97`
- `dim=1792`
- `depth=11`
- `n_heads=216`
- `n_state=32`
- `n_groups=32`
- `n_slots=64`
- `state_expansion=2`
- `linear_state=0`
- `use_triton=1`
- `use_chunked_e97=0`
- `e97_chunk_size=32`
- `mlp_ratio=2.2623`
- `mlp_multiple=64`
- `use_gate=1`
- `gate_activation=silu`
- `tokenizer=p50k_base`
- `bf16=true`
- Original run args include `diloco=true`, `diloco_k=250`,
  `diloco_outer_optimizer=avg`, `diloco_export_basis=x`, optimizer
  `schedulefree`, and `lr=0.001007`.

Representative tensor shapes:

```text
embedding.weight: (50281, 1792) bf16
lm_head.weight: (50281, 1792) bf16
layers.0.mixer.qkv_proj.weight: (20736, 1792) bf16
layers.0.mixer.a_proj.weight: (216, 1792) bf16
layers.0.mixer.g_proj.weight: (6912, 1792) bf16
layers.0.mixer.erase_gate_proj.weight: (6912, 1792) bf16
layers.0.mixer.value_write_gate_proj.weight: (6912, 1792) bf16
layers.0.mixer.o_proj.weight: (1792, 6912) bf16
layers.0.mlp.w1.weight: (4032, 1792) bf16
layers.0.mlp.w2.weight: (4032, 1792) bf16
layers.0.mlp.w3.weight: (1792, 4032) bf16
```

Runtime strict-load checks during fixed eval and canary both reported:

```text
level=E97 params=1,286,589,072
model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1286589072,mlp_ratio=2.2623
```

Compatibility adjustment from the older generic Frontier template: the canary
used the staged `dim=1792`, `depth=11`, `n_heads=216`, `mlp_ratio=2.2623`
architecture rather than the older `dim=1536`, `depth=10`, `n_heads=323`,
`mlp_ratio=1.5` shape.

## Fixed Eval Baseline

Scoring tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Baseline job:

- Job ID: `4900725`
- Job name: `e97-1p3b-pretrained-fixed-eval`
- State: `COMPLETED`
- Exit: `0:0`
- Elapsed: `00:00:49`
- Nodes: `1`
- Output CSV:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/E97_1.3B_diloco_20260623_103742_step260500/source_fixed_eval.csv`

Baseline result:

| Checkpoint | Step | CE | BPB |
| --- | ---: | ---: | ---: |
| Source pretrained | 260500 | 26.93483090 | 11.20646537 |

The absolute score is poor on this small smoke tensor; the useful signal here is
row-matched source-vs-candidate delta under the same invocation.

## One-Node Resume Canary

Submitted exactly one training canary:

- Job ID: `4900730`
- Job name: `e97-1p3b-pretrained-canary`
- State: `COMPLETED`
- Exit: `0:0`
- Nodes: `1`
- Walltime request: `00:20:00`
- Elapsed: `00:10:12`
- Requested node-hours: `0.333333`
- Actual node-hours from elapsed time: `0.170000`
- Queue/QOS: batch/debug
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/logs/train.log`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/artifacts/manifest.json`
- Canary launcher committed in this task:
  `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`

Canary envelope:

```text
RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
TRAIN_MINUTES=10
DILOCO_ISLAND_SIZE=1
DILOCO_K=40
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.0
DILOCO_EXPORT_BASIS=x
SAVE_EVERY=40
BATCH_SIZE=1
CHUNK_SIZE=2048
```

Loss and merge behavior:

- Resumed from source step `260500`.
- Warmup losses were finite on all ranks.
- First logged train losses:
  `2.8406`, `2.3990`, `2.9754`, `3.0951`.
- Later logged train losses remained finite; examples near the end:
  `2.6886`, `2.9455`, `2.4475`, `2.9723`, `2.3997`, `2.2972`.
- Final step: `261063`.
- New steps: `563`.
- `FINAL_LOSS_LAST100=2.6699`.
- `DILOCO_MERGES=15`.
- `DILOCO_K=40`.
- `DILOCO_SYNC_TOTAL_S=2.925`.
- `DILOCO_SYNC_AVG_MS=195.0`.
- Representative global throughput after warmup: mostly about
  `20k-22.5k tok/s`, with expected dips immediately after checkpoint saves.

Checkpoint/finalization/latest behavior:

- Periodic K-aligned saves occurred at K40 boundaries, including steps
  `260520`, `260560`, `260600`, ..., `261040`.
- Retention kept the last three periodic checkpoints plus the final checkpoint:
  `checkpoint_step_260960_loss_2.6377.pt`,
  `checkpoint_step_261000_loss_3.1674.pt`,
  `checkpoint_step_261040_loss_2.9455.pt`, and
  `checkpoint_step_261063_loss_2.6699.pt`.
- Final consensus merge occurred at step `261063`.
- Final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/train/emender_E97_1.3B_20260625_131005/checkpoint_step_261063_loss_2.6699.pt`
- `latest.pt` resolves to that final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/train/emender_E97_1.3B_20260625_131005/latest.pt`

Warnings:

- Triton/Python version warnings were emitted by the current Frontier stack.
- NCCL rank/device warning appeared during process group initialization, but the
  run completed with clean exit, finite loss, successful merges, final
  checkpoint, and valid `latest.pt`.

## Source Vs Canary Fixed Eval

Comparison job:

- Job ID: `4900750`
- Job name: `e97-1p3b-canary-fixed-eval`
- State: `COMPLETED`
- Exit: `0:0`
- Elapsed: `00:01:24`
- Nodes: `1`
- Result CSV:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/artifacts/source_vs_canary_fixed_eval.csv`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_canary/4900730-20260625T170908Z/artifacts/source_vs_canary_fixed_eval_manifest.json`

Results on the same fixed tensor:

| Checkpoint | Step | CE | BPB | Delta CE | Delta BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source pretrained | 260500 | 26.93483090 | 11.20646537 | 0.00000000 | 0.00000000 |
| One-node canary final | 261063 | 26.82457709 | 11.16059333 | -0.11025381 | -0.04587204 |

The canary improved on the fixed smoke tensor under the same scoring command.
This is enough to authorize the planned 8-node K-cadence bracket, but it should
not be over-interpreted as evidence for larger scaleout.

## Scope Control

Confirmed from task logs, Slurm accounting, and WG task state:

- Submitted one fixed-eval baseline job: `4900725`.
- Submitted exactly one training job: one-node canary `4900730`.
- Submitted one post-canary fixed-eval comparison job: `4900750`.
- Submitted no 8/16/32/64-node training job from this task.
- Submitted no GDN2 job from this task.
- Submitted no CMAES job from this task.
- Submitted no schedule-free outer job from this task.
- `run-64-node-e97` is still `open (PAUSED)`.

## Downstream Instructions

The downstream 8-node tasks may proceed, one task per cadence:

- `run-e97-1-3b`: K40
- `run-e97-1-3b-2`: K160
- `run-e97-1-3b-3`: K320

Use the same local source checkpoint and staged architecture shape recorded
above. Keep the downstream probes bounded, no-DDP singleton-island, avg outer,
and row-match the source-vs-candidate fixed eval against the staged source.
