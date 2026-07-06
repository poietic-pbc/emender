# Validate Merged Real Async E97 1n/2n Debug, 2026-07-06

Task: `validate-merged-real`

Conclusion: **NO-GO for `validate-merged-real-2` / 8n20m production-wrapper validation.**

The required bounded Frontier debug-QOS ladder was attempted with the production
wrapper and real trainer path. The initial `origin/main` run failed before
checkpoint load completed because the real trainer CLI forced tiny model
defaults that did not match the refreshed E97 1.3B `latest.pt` seed. A narrow
production-path fix was applied in this task to expose and record the real E97
geometry and a shared offline tiktoken cache. After that fix, both 1-node and
2-node runs reached the real checkpoint-compatible model path, but real B4 /
chunk 2048 / K40 worker training deferred due HIP OOM on MI250X GPU 0 before
any finite loss or tokens/sec were produced.

## Source State

- Starting source: `origin/main` at `48d06f07a6950d528756a42d986a585dbecbd740`.
- Real trainer entrypoint: `scripts/frontier/e97_async_diloco_train.py`.
- Production wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`.
- Synthetic/protocol debug scripts were not used as training evidence.
- External production latest guard was recorded and unchanged:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`.

## Required Seed And Runtime

- Refreshed seed latest:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Seed symlink target observed before launch:
  `checkpoint_step_1065000_loss_2.5386.pt`
- Real data:
  `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- Activated `PYTHON_BIN` recorded by wrapper:
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`
- Runtime versions recorded by wrapper:
  Python `3.12.13`, Torch `2.10.0+rocm7.1`, HIP `7.1.25424`, Triton `3.6.0`.
- Shared tiktoken cache staged for offline compute nodes:
  `/lustre/orion/bif148/proj-shared/emender/tiktoken-cache`

## Production-Path Fixes Applied

The initial production wrapper command showed that `MODEL_PARAMS=1.3b` was
passed, but the trainer CLI still used parser defaults `dim=8`, `depth=1`,
byte vocab `256`, and tiny E97 knobs. Strict checkpoint load then failed
against the refreshed `latest.pt`, whose tensors require vocab `50281`,
`dim=1792`, and 11 layers.

Changes made:

- `scripts/frontier/e97_async_diloco_train.py` now accepts model geometry
  overrides: tokenizer, heads/state/slots/groups, expansion, E97 runtime
  switches, gate settings, MLP settings, and weight decay.
- `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch` now defaults to
  checkpoint-compatible E97 1.3B geometry:
  `tokenizer=p50k_base`, `params=100m`, `dim=1792`, `depth=11`,
  `n_heads=216`, `n_state=32`, `n_slots=64`, `n_groups=32`,
  `expansion=1.0`, `use_triton=1`, `use_chunked_e97=0`,
  `linear_state=0`, `use_write_gate=0`, `use_gate=1`,
  `gate_activation=silu`, `mlp_ratio=2.2623`, `mlp_multiple=64`,
  `weight_decay=0.01`.
- The wrapper now exports and records `TIKTOKEN_CACHE_DIR` so compute-node
  launches do not attempt external network fetches for `p50k_base.tiktoken`.
- `tests/test_async_diloco_real_trainer.py` includes a CLI checkpoint-load
  regression test using non-default model geometry.

Local validation:

- `bash -n scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m py_compile scripts/frontier/e97_async_diloco_train.py`
- `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest tests/test_async_diloco_real_trainer.py -q`
  passed: `8 passed in 33.62s`.

## Slurm Attempts

### 1n Original Main Attempt

Job ID: `4948625`

Exact command:

```bash
sbatch --parsable -A bif148 -p batch -q debug -N 1 -t 00:20:00 -J async-diloco-real-e97-1n --output=logs/frontier/async_diloco_e97/%x-%j.out --error=logs/frontier/async_diloco_e97/%x-%j.err --export=ALL,WG_TASK_ID=validate-merged-real,TASK_ID=validate-merged-real-1n,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-724,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_NODE_COUNT=1,ASYNC_WORKER_COUNT_PER_NODE=8,ASYNC_WORKER_COUNT=8,ASYNC_LOCAL_QUORUM=6,ASYNC_GLOBAL_QUORUM=1,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.333333,TRAIN_MINUTES=20 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Slurm accounting:

```text
4948625|async-diloco-real-e97-1n|batch|debug|FAILED|1:0|00:00:41|1|frontier07540|2026-07-06T14:33:59|2026-07-06T14:35:09|2026-07-06T14:35:50
```

Artifacts:

- stdout: `logs/frontier/async_diloco_e97/async-diloco-real-e97-1n-4948625.out`
- stderr: `logs/frontier/async_diloco_e97/async-diloco-real-e97-1n-4948625.err`
- env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948625-20260706T183513Z/artifacts/env.txt`
- command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948625-20260706T183513Z/artifacts/command.txt`

Result:

- Failed before real training.
- Root cause: strict `latest.pt` load into tiny CLI-default model.
- No metrics JSON, no loss, no tokens/sec, no checkpoint finalization.

### 1n Patched Geometry, Offline Tokenizer Cache, B4/2048/K40

Final representative job ID: `4948769`

Exact command:

```bash
sbatch --parsable -A bif148 -p batch -q debug -N 1 -t 00:20:00 -J async-diloco-real-e97-1n-r4 --output=logs/frontier/async_diloco_e97/%x-%j.out --error=logs/frontier/async_diloco_e97/%x-%j.err --export=ALL,WG_TASK_ID=validate-merged-real,TASK_ID=validate-merged-real-1n-r4,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-724,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_DILOCO_DEVICE=cuda,PYTORCH_ALLOC_CONF=expandable_segments:True,ASYNC_NODE_COUNT=1,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_WORKER_COUNT=1,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=1,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.333333,TRAIN_MINUTES=20 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Intentional bounded-debug overrides:

- Worker count reduced from production `8/node` to `1/node`.
- Local quorum reduced from production `6` to `1`.
- Global quorum remained one node for 1n.
- `batch_size=4`, `chunk_size=2048`, and `DILOCO_K=40` were not overridden.

Slurm accounting:

```text
4948769|async-diloco-real-e97-1n-r4|batch|debug|COMPLETED|0:0|00:01:02|1|frontier02887|2026-07-06T14:57:16|2026-07-06T14:57:27|2026-07-06T14:58:29
```

Artifacts:

- stdout: `logs/frontier/async_diloco_e97/async-diloco-real-e97-1n-r4-4948769.out`
- stderr: `logs/frontier/async_diloco_e97/async-diloco-real-e97-1n-r4-4948769.err`
- env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948769-20260706T185730Z/artifacts/env.txt`
- command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948769-20260706T185730Z/artifacts/command.txt`
- metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948769-20260706T185730Z/artifacts/async_diloco_e97_256n_metrics.json`

Metrics summary:

- `latest_generation=-1`
- node quorum: `deferred`, `accepted_updates=0`, `failed_updates=1`, `tokens_per_generation=0`, `tokens_per_sec=0.0`
- global quorum: `deferred`, `accepted_updates=0`, `timed_out_updates=1`, `tokens_per_generation=0`, `tokens_per_sec=0.0`
- worker error: HIP OOM after PyTorch allocated `57.58 GiB` and reserved `6.06 GiB`; attempted extra allocation was `20.00 MiB`.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` did not help because HIP reported expandable segments unsupported on this platform.
- No recovery/finalization checkpoint paths were created because no generation advanced.

### 2n Patched Geometry, Offline Tokenizer Cache, B4/2048/K40

Job ID: `4948828`

Exact command:

```bash
sbatch --parsable -A bif148 -p batch -q debug -N 2 -t 00:20:00 -J async-diloco-real-e97-2n-r1 --output=logs/frontier/async_diloco_e97/%x-%j.out --error=logs/frontier/async_diloco_e97/%x-%j.err --export=ALL,WG_TASK_ID=validate-merged-real,TASK_ID=validate-merged-real-2n-r1,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-724,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real,ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1,ASYNC_DILOCO_SYNTHETIC_TOKEN_STREAM=0,ASYNC_DILOCO_DEVICE=cuda,ASYNC_NODE_COUNT=2,ASYNC_WORKER_COUNT_PER_NODE=1,ASYNC_WORKER_COUNT=2,ASYNC_LOCAL_QUORUM=1,ASYNC_GLOBAL_QUORUM=2,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=0.666667,TRAIN_MINUTES=20 scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Intentional bounded-debug overrides:

- Worker count reduced from production `8/node` to `1/node`.
- Local quorum reduced from production `6` to `1`.
- Global quorum set to `2` for the 2-node global-supervisor check.
- `batch_size=4`, `chunk_size=2048`, and `DILOCO_K=40` were not overridden.

Slurm accounting:

```text
4948828|async-diloco-real-e97-2n-r1|batch|debug|COMPLETED|0:0|00:01:18|2|frontier[01143,01151]|2026-07-06T14:59:27|2026-07-06T15:00:11|2026-07-06T15:01:29
```

Artifacts:

- stdout: `logs/frontier/async_diloco_e97/async-diloco-real-e97-2n-r1-4948828.out`
- stderr: `logs/frontier/async_diloco_e97/async-diloco-real-e97-2n-r1-4948828.err`
- env: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948828-20260706T190014Z/artifacts/env.txt`
- command: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948828-20260706T190014Z/artifacts/command.txt`
- metrics: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/validation/validate-merged-real/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4948828-20260706T190014Z/artifacts/async_diloco_e97_256n_metrics.json`

Metrics summary:

- `latest_generation=-1`
- local node-0 quorum: `deferred`, `accepted_updates=0`, `failed_updates=1`
- local node-1 quorum: `deferred`, `accepted_updates=0`, `failed_updates=1`
- global quorum: `deferred`, `quorum_threshold=2`, `accepted_updates=0`, `timed_out_updates=2`
- total real training loss/tokens: no finite loss, `tokens_per_generation=0`, `tokens_per_sec=0.0`
- both worker reports failed with the same HIP OOM pattern as the 1n run.
- No recovery/finalization checkpoint paths were created because no generation advanced.

## Validation Checklist Assessment

- Slurm job ids, commands, queue/QOS, logs, elapsed time, and node-hours:
  **recorded** for all attempts above.
- Artifacts show real trainer path on main, activated `PYTHON_BIN`, refreshed
  `latest.pt`, `batch_size=4`, `chunk_size=2048`, `K=40`: **recorded**.
- Metrics show real training loss/tokens/sec, not synthetic loss: **not met**.
  The real trainer path was used with real data, but all advanced-training
  metrics are zero because workers OOMed before producing loss/tokens.
- Local/global quorum metrics and deferred generations: **recorded**.
  Both 1n and 2n metrics contain deferred local/global generations.
- Recovery/finalization checkpoint records: **not met**.
  Checkpoint path arrays are empty because no global generation advanced.
- Clear pass/no-go for 8n20m production-wrapper validation: **NO-GO**.

## Follow-Up Needed

Before submitting `validate-merged-real-2`, the real async trainer needs a
memory-feasible production-debug execution strategy. The current import-level
orchestrator builds a full 1.286B E97 model and optimizer inside a worker on a
single MI250X GPU and fails before the first B4/chunk2048/K40 update. Candidate
fixes include launching rank-local workers with the existing `train.py`/`srun`
GPU binding path, reducing full-state/dense-delta residency, using a memory
lighter optimizer/debug step path, or explicitly defining a lower-memory debug
geometry if the acceptance criteria allow that override.
