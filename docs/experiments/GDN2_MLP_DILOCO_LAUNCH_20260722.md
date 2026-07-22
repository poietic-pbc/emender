# GDN2-MLP 8-GPU DiLoCo Launch Audit - 2026-07-22

Task: `launch-matched-gdn2`

## Result

The matched from-scratch GDN2-MLP 8-rank pure local DiLoCo control was launched
successfully after preflight, checkpoint-save smoke, and checkpoint-resume
smoke gates passed.

Full run:

- Run ID: `gdn2_mlp_full_20260722T083424Z`
- Supervisor / torchrun PID: `3754241`
- Log directory: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp`
- Main log: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log`
- Manifest: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/launch_manifest.json`
- Output directory: `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444`
- Leased GPUs: `0,1,2,3,4,5,6,7`

## Required Inputs Read

- `/home/erikg/ndm/docs/GDN2_MLP_DILOCO_HANDOFF_20260711.md`
- `/home/erikg/ndm/scripts/launch_gdn2_mlp_8gpu_diloco.sh`
- `docs/repro/lb_gdn2_mlp_20260612/best.json`
- `docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md`
- `docs/FRONTIER_E97_SCALEOUT_NO_DDP_POLICY_20260625.md`
- `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`

The handoff document and launcher were untracked in `/home/erikg/ndm`; they
were read and used but not staged or committed.

## Frozen Policy Cross-Check

The launcher command and `best.json` matched the requested frozen policy:

- `level=gdn2-mlp`
- `dim=2176`
- `depth=12`
- `n_heads=30`
- `expansion=1`
- `gdn2_mlp_ratio=3.258732449079677`
- `lr=0.00047431158698290157`
- `actual_params=1286713448`
- per-GPU `batch_size=4`
- `chunk_size=2048`
- `optimizer=schedulefree`
- `bf16`
- tokenizer `p50k_base`
- data `/home/erikg/elman/data/pile.txt`
- `torchrun --standalone --nproc_per_node=8`
- `--diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0`
- no `--diloco_island_size > 1`, so the run is pure local DiLoCo with one
  independent learner per GPU and no per-step DDP gradient all-reduce

Preflight model construction was performed with the repo's actual `LadderLM`
plus external GDN2 wiring on the meta device:

```text
{'total_params': 1286713448, 'trainable_params': 1286713448}
actual_params=1286713448
```

## Preflight Commands And Results

Commands run before any GPU-occupying smoke:

```bash
wg quickstart
wg msg read launch-matched-gdn2 --agent "$WG_AGENT_ID"
git status --short
git log --oneline main..HEAD --max-count=20
sed -n '1,320p' /home/erikg/ndm/docs/GDN2_MLP_DILOCO_HANDOFF_20260711.md
sed -n '1,320p' /home/erikg/ndm/scripts/launch_gdn2_mlp_8gpu_diloco.sh
sed -n '1,220p' docs/repro/lb_gdn2_mlp_20260612/best.json
sed -n '1,260p' docs/repro/lb_gdn2_mlp_20260612/REPRODUCTION.md
sed -n '1,260p' docs/FRONTIER_E97_SCALEOUT_NO_DDP_POLICY_20260625.md
sed -n '1,260p' docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md
git -C /home/erikg/ndm status --short
git -C /home/erikg/ndm rev-parse --abbrev-ref HEAD
git -C /home/erikg/ndm rev-list --left-right --count origin/main...HEAD
test -d /home/erikg/GatedDeltaNet-2
test -e /mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.stop
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd | rg 'supervise_emender|torchrun|train.py' || true
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits
nvidia-smi pmon -c 1
```

Results:

- No WG messages were unread at start.
- Local `/home/erikg/ndm` was on `main`, ahead `8` / behind `622` versus
  `origin/main`; no merge, pull, or rebase was performed.
- `GDN2_PATH` existed at `/home/erikg/GatedDeltaNet-2`.
- E97 stop sentinel existed at
  `/mnt/nvme1n1/erikg/diloco_8gpu/emender/supervisor.stop`.
- No `supervise_emender`, `torchrun`, or `train.py` process was active.
- All eight RTX 6000 Ada GPUs were idle with 2 MiB used and no `pmon` clients.

## Smoke

Initial smoke output root:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/gdn2_mlp_smoke_20260722T083200Z
```

Initial smoke used the frozen model/data/tokenizer/DiLoCo settings with
`--steps 1 --save_every 1 --keep_checkpoints 4 --log_every 1`. Evidence:

```text
[DiLoCo] world_size=8 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=250 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
Run label prefix: gdn2_gdn2-mlp_1.3B (params_arg=100m, total_params=1,286,713,448, trainable_params=1,286,713,448)
Model: Level gdn2-mlp, 1,286,713,448 parameters
step      1 | loss 11.2430 | lr 4.74e-04 | grad 27.00 | tok/s 1963 | global_tok/s 15707
>>> saved checkpoint: checkpoint_step_000001_loss_11.2430.pt
>>> [DiLoCo] FINAL merge #1 at step 1: consensus model averaged across 8 ranks (4262 ms)
```

All eight ranks emitted:

```text
FLA chunked GDN-2 fused kernel import path, NO eager fallback
```

The initial smoke checkpoint was:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/gdn2_mlp_smoke_20260722T083200Z/gdn2_gdn2-mlp_1.3B_20260722_083220/checkpoint_step_000001_loss_11.2430.pt
```

After initial smoke shutdown, `nvidia-smi pmon -c 1` showed no GPU clients and
`ps` showed no stale `torchrun` or `train.py` process.

Resume smoke output root:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/gdn2_mlp_smoke_20260722T083200Z_resume
```

Resume smoke used the same frozen settings with `--resume <initial latest.pt>`,
`--steps 2 --save_every 1 --keep_checkpoints 4 --log_every 1`. Evidence:

```text
Resuming from /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/gdn2_mlp_smoke_20260722T083200Z/gdn2_gdn2-mlp_1.3B_20260722_083220/latest.pt
Resumed at step 1
step      2 | loss 7.7064 | lr 4.74e-04 | grad 8.75 | tok/s 3185 | global_tok/s 25481
>>> saved checkpoint: checkpoint_step_000002_loss_7.7064.pt
>>> [DiLoCo] FINAL merge #1 at step 2: consensus model averaged across 8 ranks (4191 ms)
```

The resume smoke checkpoint was:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/gdn2_mlp_smoke_20260722T083200Z_resume/gdn2_gdn2-mlp_1.3B_20260722_083329/checkpoint_step_000002_loss_7.7064.pt
```

## Full Launch

Full launch command:

```bash
LOGDIR=/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp \
NAME=gdn2_mlp_full_20260722T083424Z \
GDN2_PATH=/home/erikg/GatedDeltaNet-2 \
/home/erikg/ndm/scripts/launch_gdn2_mlp_8gpu_diloco.sh
```

The manifest recorded this exact underlying command:

```text
env GDN2_PATH=/home/erikg/GatedDeltaNet-2 NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train.py --level gdn2-mlp --dim 2176 --depth 12 --n_heads 30 --expansion 1 --gdn2_mlp_ratio 3.258732449079677 --use_conv 1 --d_conv 4 --optimizer schedulefree --lr 0.00047431158698290157 --bf16 --batch_size 4 --chunk_size 2048 --data /home/erikg/elman/data/pile.txt --tokenizer p50k_base --diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --steps 100000000 --save_every 500 --keep_checkpoints 20 --log_every 25 --output /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs
```

The full run log showed:

```text
gpu_lease: granted GPUs: 0,1,2,3,4,5,6,7 (pid 3754241)
[DiLoCo] world_size=8 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=250 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
Output directory: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444
Model: Level gdn2-mlp, 1,286,713,448 parameters
```

All eight ranks emitted the fused GDN2 guard, and all eight bound to separate
CUDA devices. Representative early training evidence:

```text
step     25 | loss 9.1693 | lr 4.74e-04 | grad 3.59 | tok/s 7758 | global_tok/s 62062 | elapsed_h 0.014 | time 2026-07-22T08:35:16+00:00
step     50 | loss 7.3608 | lr 4.74e-04 | grad 2.36 | tok/s 8614 | global_tok/s 68911 | elapsed_h 0.020 | time 2026-07-22T08:35:40+00:00
step     75 | loss 7.0565 | lr 4.74e-04 | grad 2.97 | tok/s 8319 | global_tok/s 66551 | elapsed_h 0.027 | time 2026-07-22T08:36:04+00:00
step    100 | loss 6.7642 | lr 4.74e-04 | grad 1.96 | tok/s 8109 | global_tok/s 64870 | elapsed_h 0.034 | time 2026-07-22T08:36:30+00:00
```

`nvidia-smi pmon -c 1` at the health check showed one active Python worker on
each GPU with 98-99% SM utilization.

## Git / Artifact Audit

Generated logs and checkpoints were outside the repository:

- `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp_smoke/...`
- `/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/...`

`git status --short` in this worktree showed only `.wg` before this report was
added. The base checkout still had its pre-existing untracked files, including
the untracked handoff and launcher, which were preserved and not committed.

Recent reflog was checked for `pull`, `merge`, and `rebase`; none were present
from this task.
