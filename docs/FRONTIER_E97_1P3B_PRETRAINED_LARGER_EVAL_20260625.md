# E97 1.3B Pretrained Larger Deterministic Eval - 2026-06-25

WG task: `evaluate-e97-1-3b`

## Verdict

The larger deterministic eval resolves the K40/K160 choice strongly enough to
select K40, not K160, for any single bounded 16-node selected-recipe follow-up.

On a deterministic 69,632-token tensor, K40 remains inside the prior small
source-regression gate, while K160 repeats a large deterministic regression:

| Row | Step | CE | BPB | Delta CE vs source | Delta BPB vs source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Staged source | `260500` | `26.83747050` | `10.31650033` | `+0.00000000` | `+0.00000000` |
| 8-node K40 final | `263082` | `26.85514759` | `10.32329551` | `+0.01767709` | `+0.00679518` |
| 8-node K160 final | `263840` | `27.08281208` | `10.41081125` | `+0.24534158` | `+0.09431092` |

Interpretation:

- K40 is the selected 16-node candidate if exactly one bounded scale task is
  created from deterministic eval evidence alone.
- K160 remains train-loss-strong at its own endpoint, but this larger fixed eval
  preserves the earlier deterministic warning rather than explaining it away.
- I did not create or submit any training or scaleout job from this task.
- Concurrent graph note: after this eval completed, a separate task
  `run-e97-1p3b-k160-16n` in worktree `agent-319` submitted Slurm job `4902278`
  (`e97-1p3b-k160-16n`). That job was not submitted from this task. I messaged
  that task with the larger-eval K40/K160 deltas before finishing this report.
- Coordinator note received during closeout: the active scale ladder will treat
  this eval as secondary context rather than a blocker for train-loss-window
  scaling.

Because the graph already has a concurrent K160 16-node run, I did not add a
second 16-node task from this eval. I briefly created an adjudication gate before
the K160 32-node successor, then removed that dependency and abandoned the gate
after the coordinator clarified that this eval should remain secondary context
for the active scale ladder.

## Inputs Read

Read the synthesis and probe reports required by the task:

- `docs/FRONTIER_E97_1P3B_PRETRAINED_K_SWEEP_SYNTHESIS_20260625.md`
- `docs/FRONTIER_E97_1P3B_PRETRAINED_8NODE_K40_20260625.md`
- `docs/FRONTIER_E97_1P3B_PRETRAINED_K160_8NODE_20260625.md`
- Also checked the optional negative-control report
  `docs/FRONTIER_E97_1P3B_PRETRAINED_8N_K320_20260625.md`.

K320 was not scored in this task. It was optional, and including it would not
change the K40/K160 selection. The required source/K40/K160 comparison was kept
as the priority.

## Eval Job

Submitted exactly one eval-only Slurm job from this task:

| Field | Value |
| --- | --- |
| Slurm job id | `4902103` |
| Job name | `e97-1p3b-large-eval` |
| State / exit | `COMPLETED` / `0:0` |
| Queue / QOS | `batch` / `debug` |
| Nodes | `1` |
| Initial requested walltime | `02:00:00` |
| Effective requested walltime | `01:00:00` after pending-job `scontrol update` before allocation |
| Effective requested node-hours | `1.0` |
| Actual elapsed | `00:02:39` |
| Actual node-hours | `0.044167` |
| Submit / start / end | `2026-06-25T16:46:13` / `2026-06-25T17:08:35` / `2026-06-25T17:11:14` from `sacct` |
| Launcher | `scripts/frontier/e97_1p3b_pretrained_larger_eval.sbatch` |
| Eval code | `scripts/eval_checkpoint.py` |
| Result CSV | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger/4902103-20260625T210836Z/artifacts/source_k40_k160_larger_eval.csv` |
| Manifest | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger/4902103-20260625T210836Z/artifacts/manifest.json` |
| Slurm stdout | `logs/frontier/eval/e97-1p3b-large-eval-4902103.out` |
| Slurm stderr | `logs/frontier/eval/e97-1p3b-large-eval-4902103.err` |

Manifest caveat: the manifest records the script's original `02:00:00` request
and `2.0` requested node-hours. While the job was still pending and had consumed
zero runtime, I reduced the Slurm time limit in place to `01:00:00`; `sacct`
therefore provides the effective requested walltime/node-hours used in the
table above.

## Tensor

The tensor was built deterministically by the launcher from the same validation
text source and tokenizer family used by the previous fixed-eval harness.

| Field | Value |
| --- | --- |
| Tensor path | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger/4902103-20260625T210836Z/artifacts/commapile_mainmix_val_p50k_2048_x128.pt` |
| Tensor metadata | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger/4902103-20260625T210836Z/artifacts/commapile_mainmix_val_p50k_2048_x128.json` |
| Scored tokens | `69,632` |
| Prior smoke scored tokens | `16,384` |
| Size multiple vs smoke | `4.25x` |
| Chunk shape | `34` chunks, `2048` scored tokens/chunk |
| Tokenizer | `p50k_base` |
| Bytes/token | `3.7530445772058822` |
| Source text | `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt` |
| Source bytes | `262,144` |
| Source SHA256 | `f39aa4af301782bd605fe25a838866759112aa06e35c9e063c83f8465764cc57` |
| Tensor SHA256 | `0f392cb89500b7b76009bf9f456cc9ceb6f343e994de14248f1fe5b505f62509` |
| Construction | Contiguous non-overlapping full 2048-token chunks from the start of the validation file |

The launcher requested up to 128 chunks, but the validation source contained
only 34 full chunks. The actual tensor is still materially larger than the
16,384-token smoke slice and records the source/tensor checksums above.

## Identical Evaluator Settings

All three required rows used identical evaluator settings:

- Evaluator: `scripts/eval_checkpoint.py`
- Scoring tensor:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260625/e97-1p3b-pretrained-larger/4902103-20260625T210836Z/artifacts/commapile_mainmix_val_p50k_2048_x128.pt`
- Tokenizer: `p50k_base`
- Batch size: `1`
- `--y-mode saved`
- `EVAL_CHECKPOINT_GPU_LEASED=1`
- `NDM_PIN_TRITON_AUTOTUNE=1`
- `OMP_NUM_THREADS=7`
- Forward-only checkpoint scoring; no `train.py` invocation.

Checkpoint paths:

| Row | Checkpoint |
| --- | --- |
| Source | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_diloco_20260623_103742_step260500/checkpoint_E97_1.3B_diloco_20260623_103742_step260500_loss_2.7481.pt` |
| K40 final | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_8n_k40/4900838-20260625T180815Z/train/emender_E97_1.3B_20260625_140940/checkpoint_step_263082_loss_2.6645.pt` |
| K160 final | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260625/E97_1.3B_pretrained_k160_8node/4900869-20260625T183437Z/train/emender_E97_1.3B_20260625_143545/checkpoint_step_263840_loss_2.6368.pt` |

The launcher accepted `latest.pt` paths for K40/K160, and
`scripts/eval_checkpoint.py` resolved/logged the final checkpoint targets above.

## Strict Load And Warnings

Strict load behavior:

- Source: `strict load OK step=260500; schedulefree_y_swap=False; level=E97 params=1,286,589,072`
- K40: `strict load OK step=263082; schedulefree_y_swap=False; level=E97 params=1,286,589,072`
- K160: `strict load OK step=263840; schedulefree_y_swap=False; level=E97 params=1,286,589,072`

Warnings seen in stderr:

- Frontier module replacement/reload notices.
- Triton `3.2.0` below recommended `3.3.0`.
- Python `3.10` below recommended `3.11`.
- Pinned autotune notice:
  `18 kernels / 30 (name,key) configs from registry`.

No OOM, traceback, non-finite score, checkpoint load mismatch, scaleout launch,
or training invocation was observed in this task's eval logs.

## Scope Confirmation

Confirmed:

- No training job was submitted from this task.
- No 16-node, 32-node, 64-node, GDN2, CMAES, schedule-free outer, LR sweep, beta
  sweep, or K-sweep job was submitted from this task.
- The only Slurm job submitted from this task was eval-only job `4902103`.
- `run-64-node-e97` remains `open (PAUSED)`.

Concurrent-scheduler caveat:

- `sacct`/`scontrol` showed a separate `e97-1p3b-k160-16n` job `4902278`
  submitted from `/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-319`
  by WG task `run-e97-1p3b-k160-16n`.
- That job is outside this task's worktree and was not submitted by
  `evaluate-e97-1-3b`.
- I sent `run-e97-1p3b-k160-16n` a message with the larger-eval deltas so it
  can include this evidence as secondary context.
