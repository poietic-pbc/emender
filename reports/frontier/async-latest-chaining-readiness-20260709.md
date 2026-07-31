# Async Latest.pt Chaining Readiness: 256n x 12h Production Path

Date: 2026-07-09
Task: `verify-async-latest`
Decision: **READY for a human-approved 256n x 12h chain, with the existing 256n production wrapper approval gates still in force.**

No production, extended, or 12h Slurm job was submitted by this task. All Slurm validation used `batch/debug` 1-node jobs with `#SBATCH -q debug` through `scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch`.

## Scope Boundary

The prior async B4/K40 debug ladder proved scale merge/load stability through 256n with single-generation merges. This audit is narrower and different: it verifies checkpoint chaining semantics, specifically that a future job can start from a `latest.pt` pointer produced by a previous async run, load model plus optimizer state without a schedulefree state mismatch, save a verified checkpoint, and advance only a run-local latest pointer after verification.

## Audited Paths

- 256n production wrapper checkpoint resolution:
  `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:26-28` resolves `E97_CHECKPOINT` at job start from `SEED_LATEST_PATH` or the refreshed project default:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt`.
  The wrapper passes the resolved path to the trainer at `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:192-218`; this avoids a stale checkpoint baked into the submitted command.

- Exact B4/K40 optimizer recipe:
  the wrapper now defaults and passes `LEARNING_RATE=0.001007`, `OPTIMIZER=schedulefree`, `WEIGHT_DECAY=0.01`, `WARMUP_STEPS=0`, `GRAD_ACCUM=1`, and `GRAD_CLIP=1.0` at `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:81-92` and `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch:211-217`.

- Full checkpoint load:
  `ndm/async_diloco_real.py` loads the initial `train.py` checkpoint payload with `return_checkpoint=True`, preserving `optimizer_state_dict`, source step, and source path for continuation metadata. The global merged state is then checkpointed from the actual merged tensor state at `ndm/async_diloco_real.py:724-794`.

- Chain checkpoint save:
  `_write_verified_chain_checkpoint` writes a train.py-compatible payload with `model_state_dict`, `optimizer_state_dict`, `step`, `loss`, and `checkpoint_metadata` at `ndm/async_diloco_real.py:803-870`. The output filename preserves step and loss, for example `checkpoint_step_1282580_loss_2.2741.pt`.

- Schedulefree continuation fix:
  a pre-fix smoke showed hop B loss spiking to `10.022` after loading hop A. Root cause: the checkpoint saved merged model weights while reusing stale schedulefree `z` state from the input checkpoint. The fix aligns schedulefree optimizer `z` to the merged model before saving and marks the optimizer state as eval-mode at `ndm/async_diloco_real.py:888-897`.

- Atomic latest behavior:
  the checkpoint is written to a temp file, loaded back for verification, then atomically renamed; only after that is run-local `latest.pt` atomically replaced at `ndm/async_diloco_real.py:862-885`. `latest.json` only includes verified extra checkpoint paths and raises before latest update if an expected checkpoint is missing at `ndm/async_diloco.py:809-888`.

- Compiled MPICH production-candidate path:
  the compiled helper root now carries the merged state privately in memory for checkpoint publication at `ndm/async_diloco_mpi.py:134-160` and `ndm/async_diloco_compiled_mpich.py:143-146`. The compiled coordinator strips the private tensor payload before JSON metrics and writes the verified chain checkpoint before publishing latest metadata at `ndm/async_diloco_real.py:1611-1722`.

## Code Changes

- Added train.py-compatible async chain checkpoint publication from merged global state.
- Added run-local `latest.pt` atomic symlink advancement after checkpoint verification.
- Added `latest.json` `model_checkpoint_path` and verified extra checkpoint-path support.
- Fixed schedulefree optimizer state alignment so chained `optimizer.train()` does not move parameters away from saved checkpoint weights.
- Propagated private merged state from dense/compiled transport root to the real async coordinator for checkpoint publication.
- Updated the 256n production wrapper to pass the exact B4/K40 optimizer recipe.
- Added tests:
  - two-hop latest.pt chain with optimizer state and schedulefree no-parameter-shift check at `tests/test_async_diloco_real_trainer.py:243-315`;
  - failure-safe latest behavior when a model checkpoint path is missing at `tests/test_async_diloco_checkpoint_manager.py:132-145`;
  - wrapper assertions for production optimizer flags.

## Failed Pre-Fix Smoke

This task intentionally records the failed validation that exposed the bug.

| Hop | Job | Slurm | Input checkpoint | Accepted | Bad updates | Loss | Step | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| A | `4962945` | `batch/debug`, `COMPLETED 0:0`, `00:05:05` | refreshed project `latest.pt` | 8 | 0 | 2.50541 | 1282540 | checkpoint saved |
| B | `4963052` | `batch/debug`, `COMPLETED 0:0`, `00:05:01` | job A run-local `async_run/latest.pt` | 8 | 0 | 10.02200 | 1282580 | **failed health gate** |

Hop B did start from hop A output, not the original S3/project seed, but the high loss showed that schedulefree optimizer state was not chain-safe before the fix.

## Fixed Two-Hop Smoke

Recipe for both fixed hops:

- `BATCH_SIZE=4`
- `CHUNK_SIZE=2048`
- `DILOCO_K=40`
- `ASYNC_LOCAL_STEPS=40`
- `LEARNING_RATE=0.001007`
- `OPTIMIZER=schedulefree`
- `WEIGHT_DECAY=0.01`
- `WARMUP_STEPS=0`
- `GRAD_ACCUM=1`
- `GRAD_CLIP=1.0`
- `MODEL_TOKENIZER=p50k_base`
- real CommaPile data at `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- compiled MPICH helper transport, reported as `compiled-cray-mpich-helper-collective-reduce`
- TCP dense data plane disabled

### Hop A

- Slurm job: `4963230`
- Slurm result: `COMPLETED 0:0`, `batch/debug`, elapsed `00:04:56`, one node
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_a/4963230-20260709T165345Z`
- Input checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1282500/latest.pt`
- Output `latest.pt`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_a/4963230-20260709T165345Z/async_run/latest.pt`
- Resolved checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_a/4963230-20260709T165345Z/async_run/checkpoints/emender_E97_100m_20260709/checkpoint_step_1282540_loss_2.5054.pt`
- SHA256: `0211a25d5165ba4c58da48396d31e0c409c5e6d35e125ade012f13caf708f668`
- Size: `15439252234` bytes
- Step/loss: `1282540`, `2.50541`
- Tokens consumed: `2622720`
- Accepted/stale/failed/invalid/timed-out updates: `8/0/0/0/0`
- Optimizer state present: yes
- Schedulefree no-parameter-shift check after load and `optimizer.train()`: true

### Hop B

- Slurm job: `4963311`
- Slurm result: `COMPLETED 0:0`, `batch/debug`, elapsed `00:05:08`, one node
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_b/4963311-20260709T170017Z`
- Input checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_a/4963230-20260709T165345Z/async_run/latest.pt`
- This differs from the refreshed project seed and demonstrates job B starts from job A output latest.
- Output `latest.pt`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_b/4963311-20260709T170017Z/async_run/latest.pt`
- Resolved checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_latest_chain_20260709/20260709/E97_1.3B_step1282500_async_latest_chain_fix_hop_b/4963311-20260709T170017Z/async_run/checkpoints/emender_E97_100m_20260709/checkpoint_step_1282580_loss_2.2741.pt`
- SHA256: `7390a4fd3dc30d726fae2dee934229f13c84fb055b365189c1f8823a945195f6`
- Size: `15439252362` bytes
- Step/loss: `1282580`, `2.27408`
- Tokens consumed: `2622720`
- Accepted/stale/failed/invalid/timed-out updates: `8/0/0/0/0`
- Optimizer state present: yes
- Schedulefree no-parameter-shift check after load and `optimizer.train()`: true

## Validation

- No production, extended, or 12h job was submitted. Submitted jobs were debug-only: `4962945`, `4963052`, `4963230`, `4963311`.
- The audit distinguishes previous merge-correctness evidence from newly verified `latest.pt` chaining correctness.
- Fixed hop B started from fixed hop A output `latest.pt`, not the project seed.
- Each fixed hop performed one async DiLoCo merge with finite healthy loss and zero stale/failed/invalid/timed-out updates.
- Checkpoints include `optimizer_state_dict` and are loadable by the next hop.
- Atomic/failure-safe latest behavior is directly tested by `test_extra_checkpoint_verification_failure_leaves_previous_latest_intact`.
- Chain accounting recorded start checkpoint, end checkpoint, Slurm job id, run root, and tokens consumed in this report and in run artifacts.

## Test Commands

```bash
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/pytest \
  tests/test_async_diloco_real_trainer.py \
  tests/test_async_diloco_checkpoint_manager.py \
  tests/test_async_diloco_e97_2n8n_debug_runner.py \
  tests/test_frontier_runtime_plumbing.py
```

Result: `42 passed in 43.72s`.

```bash
git diff --check
```

Result: pass.

## Final Decision

**Ready for a human-approved 256n x 12h chain from the latest.pt chaining perspective.**

The wrapper still intentionally enforces the existing production gates, including human approval and the readable passing 64n compiled-helper gate artifact for 256n production. This task did not remove or bypass those gates.
