# E97 1.3B Pretrained K160 8-Node Probe - 2026-06-25

WG task: `run-e97-1-3b-2`

## Decision

The bounded 8-node pretrained GPU-island/no-DDP K160 probe completed cleanly,
but it is not a candidate for 16-node continuation from this result.

The fixed source-vs-candidate eval regressed beyond the task gate:

- CE delta: `+0.23717952` versus allowed `<= +0.025`.
- BPB delta: `+0.09868055` versus allowed `<= +0.010`.

The training run itself had finite losses, regular K160 DiLoCo merges, clean
finalization, and a valid `latest.pt`, so the negative decision is driven by
fixed-eval non-regression rather than launch or checkpoint failure.

## Authorization

Read and followed `validate-e97-1-3b` via
`docs/FRONTIER_E97_1P3B_PRETRAINED_VALIDATION_20260625.md`.

That validation explicitly authorized the bounded 8-node pretrained
GPU-island/no-DDP cadence probes:

- `run-e97-1-3b`: K40
- `run-e97-1-3b-2`: K160
- `run-e97-1-3b-3`: K320

It also explicitly kept 16/32/64-node continuation, GDN2/CMAES, and
schedule-free outer jobs out of scope. This task launched only the K160
8-node avg-outer probe and its fixed eval.

## Launch

Training launcher:

```text
scripts/frontier/e97_1p3b_pretrained_k160_8node.sbatch
```

Launcher support change:

```text
scripts/frontier/e97_1p3b_pretrained_canary.sbatch
```

The K160 wrapper reuses the validated pretrained canary launcher and changes
only the intended K160 scale/cadence envelope.

Git commit used by the launched training job:

```text
54197f7bb977f7dcdc5e05578cda82435a440e29
```

Source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt
```

Data paths:

```text
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
SCORING_TENSOR=/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Exact training envelope:

```text
nodes=8
tasks=64
gpus_per_task=1
no within-island DDP
DILOCO_ISLAND_SIZE=1
DILOCO_K=160
SAVE_EVERY=160
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.0
DILOCO_EXPORT_BASIS=x
BATCH_SIZE=1
CHUNK_SIZE=2048
TRAIN_MINUTES=50
requested_walltime=01:00:00
requested_node_hours=8.0
```

Inner optimizer remained the staged pretrained checkpoint's compatible
training optimizer:

```text
--optimizer schedulefree
--lr 0.001007
```

This is not a schedule-free outer job; the outer optimizer was `avg`.

Queue note: the first training `sbatch` attempt under `debug` was rejected
before job creation by `QOSMaxSubmitJobPerUserLimit` because the sibling K40
debug job was already pending. The actual K160 training job was submitted once,
under `batch/normal`, with the same 8-node/1-hour training envelope.

## Training Job

- Job ID: `4900869`
- Job name: `e97-1p3b-k160-8n`
- Queue/QOS: `batch/normal`
- State: `COMPLETED`
- Exit: `0:0`
- Submit: `2026-06-25T13:45:59-04:00`
- Start: `2026-06-25T14:34:35-04:00`
- End: `2026-06-25T15:24:50-04:00`
- Elapsed: `00:50:15`
- Nodes: `8`
- Requested node-hours: `8.0`
- Actual node-hours: `6.700`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z`
- Slurm stdout:
  `logs/frontier/scaleout/e97-1p3b-k160-8n-4900869.out`
- Slurm stderr:
  `logs/frontier/scaleout/e97-1p3b-k160-8n-4900869.err`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/logs/train.log`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/artifacts/manifest.json`

Manifest note: the recorded `requested_walltime` in the run manifest is
`00:20:00` because the inherited canary launcher default was used when
`SLURM_TIMELIMIT` was not populated. Slurm accounting and the wrapper are the
source of truth for this run: the job requested `01:00:00`. The launcher has
been updated to record `REQUESTED_WALLTIME=01:00:00` for this wrapper going
forward.

## Training Result

- Source step: `260500`
- Final step: `263840`
- New steps from source: `3340`
- Final loss last100: `2.6368`
- First 100 logged loss average: `2.71014`
- Last 100 logged loss average: `2.636824`
- Logged metric lines: `668`
- Median `global_tok/s` over all logged metric lines: `168103`
- Filtered mean `global_tok/s` excluding save/merge dips below 120k: `167306`
- Filtered median `global_tok/s` excluding save/merge dips below 120k: `168217`
- `DILOCO_MERGES`: `21`
- `DILOCO_K`: `160`
- `DILOCO_SYNC_TOTAL_S`: `85.970`
- `DILOCO_SYNC_AVG_MS`: `4093.8`

Representative early losses:

```text
260505 2.8406
260510 2.3990
260515 2.9754
260520 3.0951
260525 2.4344
```

Representative final losses:

```text
263820 2.9154
263825 2.5627
263830 2.0725
263835 2.6485
263840 2.6898
```

The run reached the time/finalization guard at step `263840`, exactly on a
K160 boundary. The final merge was skipped because the step had just been
merged and the checkpoint was already consensus.

Final checkpoint and `latest.pt`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/checkpoint_step_263840_loss_2.6368.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/latest.pt -> checkpoint_step_263840_loss_2.6368.pt
```

Retained checkpoints after finalization:

```text
checkpoint_step_263520_loss_2.7491.pt
checkpoint_step_263680_loss_2.6772.pt
checkpoint_step_263840_loss_2.6898.pt
checkpoint_step_263840_loss_2.6368.pt
latest.pt -> checkpoint_step_263840_loss_2.6368.pt
```

Warnings:

- Frontier module replacement/reload notices.
- Triton `3.2.0` below recommended `3.3.0`.
- Python `3.10` below recommended `3.11`.
- PyTorch scalar conversion `UserWarning` during warmup logging.
- ProcessGroupNCCL rank/device mapping warnings during initialization.

No OOM, traceback, collective timeout, finalization failure, non-finite loss, or
checkpoint/latest failure was observed.

## Fixed Eval

The first eval `sbatch` attempt under `debug` was rejected before job creation
by `QOSMaxSubmitJobPerUserLimit`. The actual fixed eval job was submitted once
under `batch/normal`.

- Job ID: `4901464`
- Job name: `e97-1p3b-k160-fixed-eval`
- Queue/QOS: `batch/normal`
- State: `COMPLETED`
- Exit: `0:0`
- Elapsed: `00:01:23`
- Result CSV:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/artifacts/source_vs_k160_fixed_eval.csv`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/artifacts/source_vs_k160_fixed_eval_manifest.json`
- Slurm stdout:
  `logs/frontier/eval/e97-1p3b-k160-fixed-eval-4901464.out`
- Slurm stderr:
  `logs/frontier/eval/e97-1p3b-k160-fixed-eval-4901464.err`

Fixed eval results:

| Checkpoint | Step | CE | BPB | Delta CE | Delta BPB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source pretrained | 260500 | 26.93483090 | 11.20646537 | 0.00000000 | 0.00000000 |
| K160 candidate | 263840 | 27.17201042 | 11.30514592 | +0.23717952 | +0.09868055 |

Gate outcome:

```text
BPB gate: candidate <= source + 0.010 -> 11.30514592 <= 11.21646537 is false
CE gate:  candidate <= source + 0.025 -> 27.17201042 <= 26.95983090 is false
```

## Scope Control

Confirmed:

- Submitted exactly one K160 8-node training job from this task: `4900869`.
- Submitted one fixed eval job from this task: `4901464`.
- The rejected debug `sbatch` attempts produced no job IDs and therefore no
  Slurm jobs.
- Submitted no 16/32/64-node continuation job from this task.
- Submitted no GDN2 job from this task.
- Submitted no CMAES job from this task.
- Submitted no schedule-free outer job from this task.
- Submitted no uncontrolled extra training variable from this task.
- `run-64-node-e97` remains `open (PAUSED)`.

Sibling accounting observed during the audit included K40 and K320 jobs, but
those belong to their own WG tasks and are not K160 submissions from this task.
