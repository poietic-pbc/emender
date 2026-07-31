# Frontier E97 GPU-Island No-DDP Audit

Task: `audit-e97-gpu`
Date: 2026-06-25

## Verdict

`DILOCO_ISLAND_SIZE=1` with `--diloco` gives the intended GPU-as-island,
no-DDP path. In the current `train.py`, `--diloco` sets `use_ddp=False`, then
the only DiLoCo DDP wrapping branch is guarded by `island_size > 1`. Therefore
`DILOCO_ISLAND_SIZE=1` does not construct a singleton DDP process group and does
not run a per-step DDP gradient all-reduce, no-op or otherwise.

No Slurm jobs were submitted from this audit. No 32-node or 64-node GPU-island
job is authorized by this audit. `run-64-node-e97` was checked and remains
`open (PAUSED)`.

## Exact 4-node GPU-island launch settings

Use the existing Frontier wrapper, but override its default node-island setting:

```bash
sbatch -A bif148 -p batch --qos=debug -N 4 -t 00:30:00 \
  --job-name e97-gpu-island-k80-probe \
  --export=ALL,WG_TASK_ID=run-e97-gpu,TASK_ID=run-e97-gpu,\
SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=4,SCALEOUT_WALLTIME=00:30:00,\
TRAIN_MINUTES=20,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout,\
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,\
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt,\
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache,\
DILOCO_K=80,DILOCO_ISLAND_SIZE=1,DILOCO_OUTER_OPTIMIZER=avg,\
DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.0,DILOCO_EXPORT_BASIS=x,\
BATCH_SIZE=1,CHUNK_SIZE=2048,LOG_EVERY=5,VAL_EVERY=10000,\
SAVE_EVERY=80,KEEP_CHECKPOINTS=4,COMPILE_WARMUP_STEPS=1,\
RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt,\
HUMAN_APPROVAL_RECORD=docs/FRONTIER_E97_GPU_ISLAND_AUDIT_20260625.md \
  scripts/frontier/diloco_scaleout_readiness.sbatch
```

The load-bearing settings are:

- `SCALEOUT_VARIANT=e97-MLP`, which maps to `--level E97 --linear_state 0 --use_triton 1 --n_state 32 --n_heads 323 --dim 1536 --depth 10 --mlp_ratio 1.5 --mlp_multiple 64`.
- `--diloco`, always added by `scripts/frontier/diloco_scaleout_readiness.sbatch`.
- `DILOCO_ISLAND_SIZE=1`, overriding the wrapper default `8`.
- `DILOCO_OUTER_OPTIMIZER=avg`, `DILOCO_OUTER_LR=1.0`, `DILOCO_OUTER_BETA=0.0`, `DILOCO_EXPORT_BASIS=x`.
- `DILOCO_K=80` for the first probe.
- `BATCH_SIZE=1`, `CHUNK_SIZE=2048`, `SAVE_EVERY=80`, `KEEP_CHECKPOINTS=4`.
- Resume from the existing 16-node avg source checkpoint above.

Relevant code references:

- `train.py:327-371` documents that DiLoCo trains ranks independently and that `diloco_island_size 0/1 = pure DiLoCo (no intra-island DDP)`.
- `train.py:1580-1604` sets `use_ddp = dist_enabled and not args.diloco` and `use_diloco = dist_enabled and args.diloco`.
- `train.py:2105-2179` wraps normal DDP only when `use_ddp` is true and wraps DiLoCo hybrid DDP only under `if island_size > 1`.
- `train.py:2595-2609` runs the only DiLoCo synchronization at `step % diloco_k == 0`, via `diloco_merge()`.
- `scripts/frontier/diloco_scaleout_readiness.sbatch:31-36` defines the DiLoCo env defaults, including default `DILOCO_ISLAND_SIZE=8`; lines `97-120` pass these through to `train.py`.

## DDP semantics

`DILOCO_ISLAND_SIZE=1` truly avoids within-island DDP gradient all-reduce. It is
not merely a singleton DDP no-op.

The exact control flow is:

1. `--diloco` makes `use_ddp=False` under distributed launch.
2. The normal DDP wrapping block is skipped.
3. DiLoCo broadcasts rank-0 initial weights to all ranks once, so every rank
   starts from the same `W_0`.
4. `island_size = int(args.diloco_island_size or 0)`.
5. Only `island_size > 1` creates subgroup process groups and wraps the model in
   `DistributedDataParallel(process_group=island_group)`.
6. With `DILOCO_ISLAND_SIZE=1`, the `island_size > 1` block is skipped entirely.
7. Each rank trains independently for K optimizer steps and then participates in
   the global DiLoCo model-weight average.

There is still distributed communication, but it is not per-step DDP gradient
communication. The remaining collectives are initialization/process-group setup,
the rank-0 `W_0` broadcast, barriers, and the periodic global model-weight
all-reduce in `diloco_merge()`.

## 4-node versus 32-node geometry

Frontier nodes expose 8 GPUs in the wrapper shape (`--ntasks-per-node=8`,
`--gpus-per-task=1`).

For the 4-node GPU-island probe:

```text
nodes = 4
world_size = 4 * 8 = 32 ranks
DILOCO_ISLAND_SIZE = 1
islands = 32
shape = 32 islands x 1 GPU
per-step DDP gradient all-reduce = none
periodic DiLoCo average = across all 32 ranks every K local steps
per-step global batch at BATCH_SIZE=1 = 32 chunks
```

For the prior 32-node node-island shape:

```text
nodes = 32
world_size = 32 * 8 = 256 ranks
DILOCO_ISLAND_SIZE = 8
islands = 32
shape = 32 islands x 8 GPUs
per-step DDP gradient all-reduce = within each 8-GPU island
periodic DiLoCo average = across all 256 ranks every K local steps
per-step global batch at BATCH_SIZE=1 = 256 chunks
```

The two shapes have the same island count, 32, but not the same local island
update. GPU-island mode has one independent GPU per island and no within-island
gradient averaging; node-island mode has 8 GPUs per island doing exact per-step
DDP before the cross-island DiLoCo merge. The 4-node probe is therefore a
topology/communication test and a global-batch reduction at the same island
count, not a matched-token throughput replacement for the 32-node ladder.

## K recommendation

Use `DILOCO_K=80` for the first 4-node no-DDP probe.

Rationale:

- The current same-source avg ladder showed monotone improvement from K10 to K40
  to K80 on the fixed smoke gate. K80 is the best observed avg cadence even
  though it did not clear the 32-node node-island gate.
- The GPU-island probe keeps the island count at 32, matching the 32-node
  node-island ladder's cross-island count. Reusing K80 isolates the no-DDP
  topology change better than changing cadence at the same time.
- Per-rank local update frequency is still K local optimizer steps before the
  global model-weight average. `DILOCO_ISLAND_SIZE=1` changes the effective
  per-island batch from 8 GPUs to 1 GPU, so a later K bracket may be warranted,
  but the first probe should preserve the best available avg cadence.

Do not use K10 for this first no-DDP probe. K10 is the reproducible failure
class in the 32-node avg ladder. Do not jump to a much larger K before the
baseline no-DDP topology has one controlled K80 row.

## Resume safety

It is safe to resume from the 16-node avg source checkpoint when the probe also
uses avg outer.

`avg` outer is stateless in this code path: `resolve_diloco_outer_state_for_resume`
immediately initializes avg state, and `initialize_diloco_outer_state()` returns
`None` for `diloco_outer_optimizer == 'avg'`. The non-avg resume guard is the
one that fails closed when a checkpoint lacks compatible `diloco_outer_state`;
that is not used for this probe.

Therefore:

- Allowed: `RESUME_CHECKPOINT=<16-node avg source>`, `DILOCO_OUTER_OPTIMIZER=avg`.
- Not authorized by this audit: changing the first no-DDP probe to `momentum` or
  `sfsgd`, or adding `DILOCO_BOOTSTRAP_OUTER_STATE`, because that would mix the
  topology audit with outer-state provenance changes.

## Fixed eval after the probe

After the 4-node K80 no-DDP training probe, run one forward-only fixed eval that
scores both the 16-node source checkpoint and the 4-node candidate checkpoint on
the existing fixed tensor:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt
```

Use `scripts/eval_checkpoint.py` with `--y-mode saved`, `--batch-size 1`, and the
same module/env setup as the existing K40/K80 fixed-eval wrappers. The command
shape is:

```bash
python scripts/eval_checkpoint.py \
  --checkpoint <CHECKPOINT> \
  --scoring-tensor /lustre/orion/bif148/proj-shared/emender/frontier_runs/fixed_eval/20260624/e97-MLP/4894484-20260624T130601Z/artifacts/commapile_mainmix_val_smoke_p50k_2048.pt \
  --y-mode saved \
  --batch-size 1 \
  --out <RUN_ROOT>/artifacts/e97_4n_gpu_island_k80_fixed_eval.csv
```

Run it for:

1. `source_16node`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893977-20260624T103042Z/train/emender_E97_1.3B_20260624_063731/latest.pt`
2. `gpu_island_4node_k80`: the final `latest.pt` or final checkpoint from the
   4-node probe.

Acceptance gate:

- Training operationally clean: Slurm exit `0:0`, no NCCL/RCCL watchdog timeout,
  no collective mismatch, no OOM/non-finite/traceback, valid final checkpoint,
  valid `latest.pt`, bounded retention, and recorded DiLoCo merge count.
- Logs show `[DiLoCo] periodic model-weight averaging ... (no per-step gradient
  all-reduce)` and do not show `[DDP] wrapped model` or `[DiLoCo-hybrid]`.
- Manifest/logs record `world_size=32`, `DILOCO_ISLAND_SIZE=1`,
  `DILOCO_K=80`, `DILOCO_OUTER_OPTIMIZER=avg`, source checkpoint, requested and
  actual node-hours, and output path.
- Fixed eval source-vs-candidate uses the tensor above and `--y-mode saved`.
- Green fixed-smoke gate: candidate BPB delta versus the 16-node source
  `<= +0.010` and CE delta `<= +0.025`. Negative deltas are green.
- If operationally clean but worse than this gate, treat it as an informative
  no-DDP topology row, not authorization for 8/16/32/64 nodes.

## Authorization boundaries

This audit authorizes only the downstream `run-e97-gpu` decision to submit at
most one bounded 4-node E97-MLP GPU-island/no-DDP probe plus its required
one-node fixed eval.

This audit does not authorize:

- any 8-node job; that remains conditional on the 4-node result and belongs to
  `run-e97-gpu-2`;
- any 16-node, 32-node, or 64-node GPU-island/no-DDP job;
- any GDN2, CMAES, schedule-free outer, momentum outer, partial-average, or
  pretrained-continuation variant;
- resuming `run-64-node-e97`.

Validation confirmations:

- No Slurm jobs were submitted from this audit.
- No 32-node or 64-node job is authorized by this audit.
- `run-64-node-e97` remains paused.
