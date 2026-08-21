# GDN2-MLP DiLoCo New-Chat Handoff

Date: 2026-07-11

Purpose: hand a fresh chat/model session enough context to switch the current
workstream from the active E97 DiLoCo run to the matched GDN2-MLP control. This
is a coordination handoff and audit document; it does not launch training.

Operational protocols for refreshing the hosted smoothed loss plot, answering
"how's it going" status requests, and uploading checkpoints to S3 live in
`docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`.

Update 2026-07-22: the E97 run has stopped and written its final consensus
checkpoint:

```text
checkpoint: /mnt/nvme1n1/erikg/diloco_8gpu/emender/runs/emender_E97_1.3B_20260709_084606/checkpoint_step_2300930_loss_2.4365.pt
s3_checkpoint: s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
s3_manifest: s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json
final_step: 2,300,930
tokens_per_step: 65,536
final_tokens: 150,793,748,480
final_loss_last100: 2.4365
sha256: 0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

The intended target was 150B tokens; the final clean checkpoint is 0.79374848B
tokens over target. The E97 restart supervisor was stopped with:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.stop
```

Do not remove that stop file unless intentionally resuming E97. A local GDN2-MLP
launcher has been added at `scripts/launch_gdn2_mlp_8gpu_diloco.sh` using the
exact matched harness below. Run the GDN2 smoke/preflight before committing to a
long GDN2 run.

Update 2026-07-22 remote-main audit: `origin/main`, `poietic/main`, and
`ekg/main` have advanced to `5c4950f16bd9`, while this local checkout is still
at `2129e0512e92` plus untracked GDN2 handoff/launcher files. Do not merge
remote main just to start the local GDN2 comparison. The remote branch adds a
large Frontier async/resilient/native E97 stack (`native_e97_*`,
`resilient_e97_*`, `e97_async_*`) and refactors `train.py`; that is valuable
for cluster E97 scaleout, but it is not the simple local 8-GPU DiLoCo harness
used for the 150B E97 run. The important compatibility facts from the audit:

- `scripts/launch_detached_run.sh` is unchanged between this checkout and
  `origin/main`.
- `scripts/launch_emender_8gpu_diloco.sh` is unchanged between this checkout
  and `origin/main`.
- `ndm/models/external_gdn2.py` is unchanged between this checkout and
  `origin/main`.
- `docs/repro/lb_gdn2_mlp_20260612/best.json` still pins the same CMA-best
  GDN2-MLP geometry.
- Remote `train.py` still accepts the same plain `--level gdn2-mlp`,
  `--use_conv`, `--d_conv`, `--gdn2_mlp_ratio`, and `--diloco` argument shape,
  but it also changes finalization/status logic and adds import-safe helper
  paths for the cluster runtime.

Decision: for the matched local E97-vs-GDN2 comparison, prefer this current
checkout plus `scripts/launch_gdn2_mlp_8gpu_diloco.sh`. For Frontier scaleout,
do not confuse this with "local only": GDN2 should also scale out through the
plain `train.py --diloco` Frontier scaleout pattern. The existing
`scripts/frontier/diloco_scaleout_readiness.sbatch` already has
`SCALEOUT_VARIANT=gdn2-MLP`, and `scripts/frontier/debug_smoke_one_node.slurm`
already has `SMOKE_VARIANT=gdn2-MLP`.

If someone wants to use `origin/main`, do it in an isolated worktree and require
a bounded GDN2 smoke that proves the same args, parameter count, fused GDN2 path,
DiLoCo K/outer settings, checkpoint save/resume, and output metadata before any
long run. The remote E97 native/resilient runtime is not conceptually
incompatible with GDN2, but the checked-in production stack is explicitly
E97-named/hardcoded; use the model-general `train.py --diloco` scaleout path for
GDN2 unless and until that native/resilient stack is generalized and smoked for
GDN2.

## New Chat Instructions

The human is switching chat/model sessions. The next assistant should:

1. Start in `/home/erikg/ndm`.
2. Run `wg quickstart` because this repo requires it.
3. Read this file before proposing or editing any GDN2 launch path.
4. Treat "switch over for GDN2" as "prepare the matched `gdn2-mlp` DiLoCo
   analogue of the current E97 DiLoCo run", not as "resume E97 weights in a
   different architecture".
5. Do not launch a long run until the human explicitly asks. The immediate safe
   work is launcher review, preflight/smoke setup, and documenting/validating the
   exact command shape.
6. Keep generated plots/checkpoints out of git unless explicitly requested.
7. Be aware this checkout may be locally divergent from `poietic/main`; check
   `git status --short --branch` before committing or pushing.
8. Confirm the E97 supervisor is not running and all GPUs are free:
   `ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd | rg 'supervise_emender|torchrun|train.py'`
   and `nvidia-smi`.

The short answer for the new chat:

```text
Use level=gdn2-mlp with the lb-gdn2-mlp CMA best:
dim2176 depth12 n_heads30 expansion1 gdn2_mlp_ratio3.258732449079677
lr4.7431158698290157e-4 batch_size4 bf16 chunk2048 schedulefree.

Wrap it in the same conservative DiLoCo shape as the current E97 run:
8 ranks, per-rank batch 4, diloco_k=250, outer avg with lr=1.0 beta=0.0,
save_every=500, keep_checkpoints=20.
```

## Decision

Use the CMA-best `gdn2-mlp` config from `lb-gdn2-mlp` as the GDN2 analogue of
the current E97 DiLoCo run.

This is the exact-parameter GDN2 control. Do not substitute the older mixer-only
`gdn2`, and do not substitute the dense `fla-gdn` shapes unless the task is
explicitly a dense-GDN comparison rather than the current E97-vs-GDN2-MLP
DiLoCo comparison.

## Source of Truth

Primary result:

- `docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md`
- `docs/repro/lb_gdn2_mlp_20260612/best.json`
- `docs/repro/lb_gdn2_mlp_20260612/gdn2_mlp_results.json`
- Commit carrying the result in this repo history: `f2e88dc feat: lb-gdn2-mlp (agent-1397)`

The best `gdn2-mlp` candidate:

```text
level: gdn2-mlp
dim: 2176
depth: 12
n_heads: 30
expansion: 1
gdn2_mlp_ratio: 3.258732449079677
lr: 0.00047431158698290157
batch_size: 4
actual_params: 1,286,713,448
best_avg_loss: 5.894941176470589
best_eval_id: 86
total_evals: 104
generations: 13
bf16: yes
fused GDN2 kernel: yes, via external GatedDeltaNet-2 / FLA path
data: /home/erikg/elman/data/pile.txt
tokenizer: p50k_base
chunk_size: 2048
```

Current local E97 DiLoCo analogue:

```text
level: E97
dim: 1792
depth: 11
n_heads: 216
n_state: 32
mlp_ratio: 2.2623
lr: 0.001007
batch_size: 4
actual_params: 1,286,589,072
```

The measured parameter delta is:

```text
gdn2-mlp - E97 = 124,376 params = 0.00967% of E97
```

That is close enough to treat this as the matched-parameter GDN2-MLP control for
the current E97 1.3B DiLoCo experiment.

## Harness Switch

The current local E97 run uses `scripts/launch_emender_8gpu_diloco.sh`, which is
hard-coded for E97. It should not be reused unmodified.

The intended transformation is:

- Keep the DiLoCo wrapper and per-replica training regime.
- Keep per-GPU batch size `4`.
- Keep `chunk_size=2048`, `optimizer=schedulefree`, bf16, `pile.txt`,
  `p50k_base`, and constant base LR from the CMA result.
- Replace only the model-specific argument block with the `gdn2-mlp` block.
- Start from scratch or from a GDN2-MLP checkpoint only. Never resume an E97
  checkpoint into `gdn2-mlp`.

The model-specific block is:

```bash
--level gdn2-mlp \
--dim 2176 \
--depth 12 \
--n_heads 30 \
--expansion 1 \
--gdn2_mlp_ratio 3.258732449079677 \
--use_conv 1 \
--d_conv 4
```

Do not pass E97-only structure knobs such as `--n_state 32`,
`--use_gate 1`, `--gate_activation silu`, `--mlp_ratio 2.2623`,
`--mlp_multiple 64`, or `--use_triton 1` as if they controlled GDN2-MLP. The
SwiGLU ratio for this arm is `--gdn2_mlp_ratio`, not `--mlp_ratio`.

## Local 8-GPU Command Shape

The command shape is encoded in `scripts/launch_gdn2_mlp_8gpu_diloco.sh`. The
underlying training command is:

```bash
GDN2_PATH=/home/erikg/GatedDeltaNet-2 \
torchrun --standalone --nproc_per_node=8 train.py \
  --level gdn2-mlp \
  --dim 2176 \
  --depth 12 \
  --n_heads 30 \
  --expansion 1 \
  --gdn2_mlp_ratio 3.258732449079677 \
  --use_conv 1 \
  --d_conv 4 \
  --optimizer schedulefree \
  --lr 0.00047431158698290157 \
  --bf16 \
  --batch_size 4 \
  --chunk_size 2048 \
  --data /home/erikg/elman/data/pile.txt \
  --tokenizer p50k_base \
  --diloco \
  --diloco_k 250 \
  --diloco_outer_lr 1.0 \
  --diloco_outer_beta 0.0 \
  --steps 100000000 \
  --save_every 500 \
  --keep_checkpoints 20 \
  --log_every 25 \
  --output /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs
```

Rationale for the shared DiLoCo settings:

- `--diloco_k 250`, `--diloco_outer_lr 1.0`, `--diloco_outer_beta 0.0` match the
  conservative current E97 local-SGD/periodic-average path.
- `--save_every 500` lands checkpoints on merge boundaries.
- `--keep_checkpoints 20` avoids unbounded checkpoint growth.

## Frontier Notes

Frontier scaleout is a required follow-on, not an optional side quest. The
correct reading is:

1. First prove the matched GDN2 model on the same simple local/single-node
   `train.py --diloco` path used by E97.
2. Then take the same GDN2 model block into the Frontier DiLoCo scaleout ladder.
3. Do not replace this with the new E97-specific native/resilient runtime unless
   a separate task explicitly generalizes and validates it for GDN2.

Frontier already has GDN2-MLP argument plumbing in scaleout/debug harnesses. The
known correct GDN2 model block appears in:

- `scripts/frontier/diloco_scaleout_readiness.sbatch`
- `scripts/frontier/debug_smoke_one_node.slurm`
- `scripts/frontier/gdn2_rocm_preflight.py`

Frontier must set `GDN2_PATH` to the staged external checkout, typically:

```bash
export GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2"
```

Before any long run, run a one-rank and one-node smoke that proves:

- GDN2 imports resolve under ROCm/HIP.
- FLA chunked GDN2 path is active.
- A real fwd/bwd step runs on the target data/tokenizer.
- Checkpoint save and resume work for `gdn2-mlp`.
- The run label and args record the actual measured model, not the stale
  `params=100m` CLI default.

For `scripts/frontier/diloco_scaleout_readiness.sbatch`, do not rely on generic
defaults for the GDN2 scientific recipe. Set at least:

```bash
SCALEOUT_VARIANT=gdn2-MLP
LR=0.00047431158698290157
BATCH_SIZE=4
CHUNK_SIZE=2048
TOKENIZER=p50k_base
OPTIMIZER=schedulefree
DILOCO_K=250
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.0
GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2"
```

Decide and record `DILOCO_ISLAND_SIZE` explicitly. `DILOCO_ISLAND_SIZE=0` or
`1` is the pure local-DiLoCo semantics used by the local 8-GPU run. The Frontier
scaleout pattern often uses `DILOCO_ISLAND_SIZE=8`, meaning per-step DDP within a
node and periodic DiLoCo across nodes. That is compatible with GDN2, but it is a
scaleout design choice rather than the exact same optimizer semantics as the
local pure 8-island run.

## Preflight Checklist

Before launching anything long:

- Confirm `GDN2_PATH` exists and points to the intended external checkout.
- Confirm `/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.stop` exists so the
  old E97 supervisor cannot restart and compete for GPUs.
- Confirm no `supervise_emender`, `torchrun`, or `train.py` process is already
  active.
- Run the GDN2 ROCm/local dependency probe where relevant.
- Run a short `train.py` smoke with exactly the source-of-truth args above.
- Confirm `actual_params` is near `1,286,713,448`.
- Confirm bf16 is enabled.
- Confirm no eager fallback is used for the GDN2 path.
- Confirm first checkpoint can be reloaded.
- Confirm DiLoCo reports `world_size=8`, `K=250`, `outer_lr=1.0`,
  `outer_beta=0.0`, and no per-step DDP gradient all-reduce.
- Confirm the output directory is separate from the E97 run directory.

## What Not To Conclude

This handoff does not decide whether GDN2-MLP will scale better or worse than
E97 under DiLoCo. It only pins the right matched config and the correct harness
translation.

This handoff also does not bless dense `fla-gdn` as the comparison. Dense
`fla-gdn` remains useful for separate Frontier/kernel and dense-GDN questions,
but it is not the exact current-run analogue because its parameterization and
model harness differ.

## Open Work

1. Run a short local GDN2 smoke through `scripts/launch_gdn2_mlp_8gpu_diloco.sh`
   or an equivalent bounded `train.py --steps <small>` command; inspect
   `args.json` and `run_manifest.json` before a long run.
2. Add or update a Frontier launcher path if the target is Frontier rather than
   the local 8-GPU box. Use `SCALEOUT_VARIANT=gdn2-MLP` with explicit GDN2 LR,
   batch size, and island-size settings; do not inherit the generic scaleout
   defaults silently.
3. Only then approve a long GDN2-MLP DiLoCo run.
