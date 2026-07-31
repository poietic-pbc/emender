# Frontier E97 schedule-free outer audit

Date: 2026-06-25
Task: `audit-e97-schedule`

## Verdict

Use the existing Frontier scaleout wrapper, but switch the DiLoCo outer arm to
`sfsgd_y`:

```text
--optimizer schedulefree
--diloco
--diloco_k 250
--diloco_outer_optimizer sfsgd
--diloco_export_basis y
--diloco_outer_lr 1.0
--diloco_outer_beta 0.1
```

Run the first 4-node and 8-node schedule-free outer probes from scratch, not
from the current 16-node source checkpoint. This sacrifices direct continuation
comparability, but the current implementation deliberately rejects
`--resume` with `--diloco_outer_optimizer sfsgd` unless the checkpoint already
contains `diloco_outer_state` (`train.py:2121-2125`). The 16-node source is an
`avg` outer checkpoint, so it has no coherent outer ScheduleFree state to
restore.

No Slurm job was submitted for this audit. This audit authorizes no GDN2, no
CMAES, no 16-node schedule-free launch, and no 32/64-node schedule-free launch.
Only the downstream 4-node and 8-node E97-MLP schedule-free probes are in scope.

## Evidence

### Code surface

`train.py` exposes the needed command surface:

- `--optimizer schedulefree` selects inner `AdamWScheduleFree`; warmup is
  passed through as `--warmup_steps` and defaults to `0` (`train.py:307-321`,
  `train.py:2004-2012`).
- `--diloco`, `--diloco_k`, `--diloco_outer_lr`,
  `--diloco_outer_beta`, `--diloco_outer_optimizer`, and
  `--diloco_export_basis` select the outer path (`train.py:327-356`).
- `--diloco_outer_optimizer sfsgd` requires `--optimizer schedulefree`; the
  outer state is a separate manual ScheduleFree-SGD state machine with
  `{mode, x, z, y, k, weight_sum, lr_max}` (`train.py:1044-1061`,
  `train.py:1290-1338`).
- For `sfsgd`, `--diloco_export_basis y` causes the merge to export the inner
  train point instead of the inner averaged/eval point (`train.py:1114-1120`,
  `train.py:1180-1184`, `train.py:1308-1318`).
- On resume, non-`avg` outer optimizers require `diloco_outer_state`; otherwise
  training raises before launch work proceeds (`train.py:2030-2035`,
  `train.py:2121-2127`).

### Prior task evidence

Prior schedule-free outer evidence keeps `sfsgd_y` as the only plausible
schedule-free outer canary:

- P3 added `sfsgd`, proved checkpoint/resume support when the checkpoint
  already contains outer state, and ran a real 2-GPU E97 smoke with
  `OUTER_LR=1.0`, `OUTER_BETA=0.1` (`docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md:595-653`).
- P5 froze the comparison arms as `avg` and `sfsgd_y`, with `sfsgd_y` exactly
  `--diloco_k 250 --diloco_outer_optimizer sfsgd --diloco_export_basis y
  --diloco_outer_lr 1.0 --diloco_outer_beta 0.1`; it explicitly excluded fixed
  momentum, `sfsgd` export `x`, beta sweeps, LR sweeps, and K sweeps
  (`docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md:141-166`).
- P7 found local W<=8 evidence insufficient for large-island claims, kept `avg`
  as the deployment default, and retained `sfsgd_y` only as a scale/stability
  canary for true multi-node island counts (`docs/experiments/SF_DILOCO_P7_ISLAND_SCALING_ANALYSIS.md:29-41`,
  `docs/experiments/SF_DILOCO_P7_ISLAND_SCALING_ANALYSIS.md:174-195`).
- The decision handoff says `sfsgd_y` is not the default, but should be carried
  as the paired stability canary; `sfsgd` export `x` is rejected by local
  evidence (`docs/experiments/SF_DILOCO_DECISION_AND_HANDOFF.md:33-46`,
  `docs/experiments/SF_DILOCO_DECISION_AND_HANDOFF.md:86-99`).

## Checkpoint and eval semantics

Training checkpoints should be saved on merge boundaries and at finalization:

- Use `SAVE_EVERY=250` with `DILOCO_K=250` so periodic checkpoints are
  post-merge consensus checkpoints.
- Keep final walltime checkpointing enabled. `train.py` performs a final DiLoCo
  consensus merge unless the last step already merged, then rank 0 saves the
  final checkpoint and advances `latest.pt` (`train.py:2522-2595`).
- Keep `KEEP_CHECKPOINTS=4` unless the run task explicitly needs more retained
  checkpoints.

For fixed eval, use saved-basis scoring:

```text
python scripts/eval_checkpoint.py \
  --checkpoint <final/latest checkpoint> \
  --scoring-tensor /lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt \
  --y-mode saved \
  --batch-size 1 \
  --out <result csv>
```

Rationale:

- `train.py` saves schedule-free checkpoints after `optimizer.eval()`, so the
  stored `model_state_dict` is the averaged/eval `x` basis (`train.py:2454-2480`,
  `train.py:2580-2591`).
- Existing 32-node fixed-eval gates used `--y-mode saved`, not y-swap, for
  row-matched comparability (`docs/FRONTIER_E97_32NODE_FIXED_EVAL_20260624.md:24-40`).
- `scripts/eval_checkpoint.py --y-mode train` reconstructs train/y weights by
  loading optimizer state and calling `optimizer.train()`; `--y-mode saved`
  evaluates the stored model weights unchanged (`scripts/eval_checkpoint.py:69-78`,
  `scripts/eval_checkpoint.py:539-560`, `scripts/eval_checkpoint.py:683-695`).

Therefore: the required fixed-eval gate for these probes is `--y-mode saved`
with no y-swap. A separate `--y-mode train` row is allowed only as an optional
diagnostic and must not replace the saved-basis gate.

## Source choice

Do not resume these first `sfsgd_y` probes from the 16-node source checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt
```

The 16-node source is the right fixed-eval comparator for the avg ladder, and it
was produced by the `avg` outer path (`docs/FRONTIER_E97_16NODE_AVG_SMOKE_20260624.md:120-150`).
However, `sfsgd` resume requires an existing `diloco_outer_state`; P3 only proved
resume from an `sfsgd` checkpoint that already had `diloco_outer_state.mode=sfsgd`
(`docs/SCHEDULEFREE_DILOCO_FRONTIER_DESIGN.md:644-652`). Current code does not
provide a CLI mode to initialize `sfsgd` outer state from an `avg` checkpoint.

Run a short from-scratch/control window instead. Interpret the 4-node and
8-node jobs as operational and stability probes for the current code path:
finite loss, first merge(s), final consensus checkpoint, `diloco_outer_state`
retention, saved-basis fixed eval, and clean logs. Do not compare their absolute
loss/BPB directly to the 16-node continuation source as a same-trajectory result.

## 4-node launch recipe

Downstream task `run-e97-schedule` may submit one 4-node E97-MLP schedule-free
outer training job with:

```bash
sbatch -N 4 -t 00:30:00 -J e97-4n-sfsgd-y-smoke \
  --export=ALL,WG_TASK_ID=run-e97-schedule,TASK_ID=run-e97-schedule,HUMAN_APPROVAL_RECORD=docs/FRONTIER_E97_SCHEDULEFREE_OUTER_AUDIT_20260625.md,SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=4,SCALEOUT_WALLTIME=00:30:00,TRAIN_MINUTES=20,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout,DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt,TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache,DILOCO_K=250,DILOCO_ISLAND_SIZE=8,DILOCO_OUTER_OPTIMIZER=sfsgd,DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.1,DILOCO_EXPORT_BASIS=y,BATCH_SIZE=1,CHUNK_SIZE=2048,LOG_EVERY=5,VAL_EVERY=10000,SAVE_EVERY=250,KEEP_CHECKPOINTS=4,COMPILE_WARMUP_STEPS=1 \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

Important: leave `RESUME_CHECKPOINT` unset.

After the training job completes, score the final `latest.pt` on the fixed tensor
with a one-node forward-only eval. Use the same eval command shape as the K40
fixed eval wrapper, replacing only the checkpoint/output paths:

```bash
sbatch -N 1 -t 01:00:00 -J e97-4n-sfsgd-y-fixed-eval \
  --wrap='set -euo pipefail; cd /lustre/orion/bif148/scratch/erikgarrison/emender; module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a; export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=7 NDM_PIN_TRITON_AUTOTUNE=1 EVAL_CHECKPOINT_GPU_LEASED=1 HELDOUT_EVAL_BS=1; python scripts/eval_checkpoint.py --checkpoint <4N_RUN_ROOT>/train/<run_label>/latest.pt --scoring-tensor /lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt --y-mode saved --batch-size 1 --out <4N_RUN_ROOT>/artifacts/e97_4n_sfsgd_y_fixed_eval.csv'
```

The worker should also inspect the saved checkpoint for `diloco_outer_state`
presence and record `mode=sfsgd`, `k`, `weight_sum`, and `lr_max`.

## 8-node launch recipe

Downstream task `run-e97-schedule-2` may submit one 8-node E97-MLP
schedule-free outer training job with the same configuration:

```bash
sbatch -N 8 -t 00:30:00 -J e97-8n-sfsgd-y-smoke \
  --export=ALL,WG_TASK_ID=run-e97-schedule-2,TASK_ID=run-e97-schedule-2,HUMAN_APPROVAL_RECORD=docs/FRONTIER_E97_SCHEDULEFREE_OUTER_AUDIT_20260625.md,SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=8,SCALEOUT_WALLTIME=00:30:00,TRAIN_MINUTES=20,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout,DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt,TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache,DILOCO_K=250,DILOCO_ISLAND_SIZE=8,DILOCO_OUTER_OPTIMIZER=sfsgd,DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.1,DILOCO_EXPORT_BASIS=y,BATCH_SIZE=1,CHUNK_SIZE=2048,LOG_EVERY=5,VAL_EVERY=10000,SAVE_EVERY=250,KEEP_CHECKPOINTS=4,COMPILE_WARMUP_STEPS=1 \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

Important: leave `RESUME_CHECKPOINT` unset.

Use the same one-node saved-basis fixed-eval command, replacing the checkpoint
and CSV locations with the 8-node run root:

```bash
sbatch -N 1 -t 01:00:00 -J e97-8n-sfsgd-y-fixed-eval \
  --wrap='set -euo pipefail; cd /lustre/orion/bif148/scratch/erikgarrison/emender; module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a; export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=7 NDM_PIN_TRITON_AUTOTUNE=1 EVAL_CHECKPOINT_GPU_LEASED=1 HELDOUT_EVAL_BS=1; python scripts/eval_checkpoint.py --checkpoint <8N_RUN_ROOT>/train/<run_label>/latest.pt --scoring-tensor /lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt --y-mode saved --batch-size 1 --out <8N_RUN_ROOT>/artifacts/e97_8n_sfsgd_y_fixed_eval.csv'
```

The worker should record the same `diloco_outer_state` fields as the 4-node
probe and compare launch health, merge count, checkpoint integrity, and
saved-basis fixed-eval rows between 4 and 8 nodes.

## Scope guard

- Do not submit jobs from `audit-e97-schedule`; this task submitted none.
- Do not run GDN2 or CMAES as part of this mini-track.
- Do not run `sfsgd` export `x`, fixed momentum, beta sweeps, LR sweeps, or K
  sweeps in the first Frontier 4/8 probes.
- Do not submit a 16-node schedule-free job unless a later synthesis task
  explicitly authorizes it after 4/8 evidence.
- Do not submit any 32-node or 64-node schedule-free job from this mini-track.
