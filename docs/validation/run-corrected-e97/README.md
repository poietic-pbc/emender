# run-corrected-e97 validation

Date: 2026-07-07
Task: `run-corrected-e97`

## Result

No Slurm job was submitted. The corrected E97 actual-multinode B4 GPU smoke
ladder remains blocked at the hard pre-submit gate.

The wrapper was run directly in preflight/debug mode with B4 token geometry and
real commapile/latest.pt paths so it could record the effective command and
environment, then refuse before `srun`:

```bash
REPO=$PWD OUTPUT_ROOT=$PWD/docs/validation/run-corrected-e97/preflight-output TASK_ID=run-corrected-e97 ASYNC_DILOCO_NON_PRODUCTION_DEBUG=1 ASYNC_ACTUAL_MULTINODE_FILE_QUORUM=1 ASYNC_NODE_COUNT=8 ASYNC_WORKER_COUNT_PER_NODE=8 ASYNC_WORKER_COUNT=64 ASYNC_LOCAL_QUORUM=6 ASYNC_GLOBAL_QUORUM=7 BATCH_SIZE=4 CHUNK_SIZE=2048 DILOCO_K=2 ASYNC_GLOBAL_TIMEOUT_S=60 bash scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

Exit code: `70`, the corrected-ladder pre-submit refusal code.

## Exact Commands

No `sbatch` command was executed. This is intentional: the task required no
Slurm submission unless the hard gates were satisfied and recorded.

The would-be `srun` command recorded by the wrapper was:

```bash
srun -N 8 -n 8 --ntasks-per-node=1 /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -u scripts/frontier/e97_async_diloco_train.py --run-id run-corrected-e97-manual-20260707T082142Z --checkpoint /lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt --run-dir /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-751/docs/validation/run-corrected-e97/preflight-output/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260707/manual-20260707T082142Z/async_run --metrics-json /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-751/docs/validation/run-corrected-e97/preflight-output/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260707/manual-20260707T082142Z/artifacts/async_diloco_e97_256n_metrics.json --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --node-count 8 --worker-count 64 --local-quorum 6 --global-quorum 7 --generations 1 --local-steps 2 --timeout-s 60 --level E97 --params 1.3b --tokenizer p50k_base --batch-size 4 --chunk-size 2048 --lr 1e-3 --steps 2 --e97-chunk-size 32 --checkpoint-interval 64 --projection-chunk-size 256 --loss-chunk-size 256 --recovery-every-generations 5 --recovery-every-seconds 600 --export-every-generations 45 --export-every-seconds 3600 --finalization-reserve-seconds 1200 --bf16 --use-chunked-e97 --gradient-checkpointing --walltime-remaining-s 1200 --dim 1792 --depth 11 --n-heads 216 --n-state 32 --n-groups 32 --n-slots 64 --expansion 1.0 --state-expansion 2 --gate-activation silu --linear-state 1 --mlp-ratio 2.2623 --mlp-multiple 64 --actual-multinode-file-quorum
```

## Slurm State

Job ids: none.

Final Slurm state: no Slurm allocation or payload was submitted.

Elapsed time and nodes: not applicable. The preflight wrapper exited locally
before `srun`.

## Pre-Submit Gate Findings

Recorded artifact:
`docs/validation/run-corrected-e97/preflight-output/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260707/manual-20260707T082142Z/artifacts/env.txt`

Passing inputs recorded:
- `batch_size=4`
- `chunk_size=2048`
- `diloco_k=2` for the bounded K1/K2 first-rung attempt
- `synthetic_token_stream=0`
- checkpoint path: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- data path: `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt`
- Python/Torch/ROCm/Triton: Python 3.12.13, Torch 2.10.0+rocm7.1, HIP 7.1.25424, Triton 3.6.0

Blocking failures recorded:
- `actual_multinode_file_quorum` is metadata-only shared-storage quorum; dense async DiLoCo delta/state exchange is not implemented.
- Slurm topology uses one task per node, not the stable B4 eight GPU tasks per node.
- Slurm GPU request records `--gpus-per-task=0`, not stable B4 `--gpus-per-task=1`.
- Slurm GPU binding is `unset`, not stable B4 `--gpu-bind=closest`.
- `ASYNC_DENSE_UPDATE_STORAGE=0`, so dense update exchange is not recorded.
- Runtime flags drift from stable `train.py` B4 path:
  - `linear_state=1` vs stable `linear_state=0`
  - `use_chunked_e97=1` vs stable `use_chunked_e97=0`
  - `checkpoint_interval=64` vs stable `checkpoint_interval=16`

## Worker And Update Semantics

The current file-rank implementation creates one real training worker per
Slurm-launched rank (`worker-00000` for each node rank). It records
`async_worker_count=64`, but the actual debug `srun` command is `-n 8` with
`--ntasks-per-node=1`, so this is not evidence of 64 GPU workers.

Dense update exchange status: not implemented and not validated for this
actual-multinode file-quorum path. The path records metadata quorum only and
must not be presented as async DiLoCo dense model delta/state exchange.

Per-rank progress artifacts: none expected for this task attempt, because the
pre-submit gate exited before `srun` and before any rank could run. The exact
failure is the pre-submit refusal above.

## Ladder Decision

K1/K2: no-go before submission.

K4: not run, because K1/K2 did not pass the hard pre-submit gates.

K10: not run, because K1/K2 did not pass the hard pre-submit gates.

K40, 64-node, 256-node, and production launches remain blocked. They should not
be considered until a path exists that uses the stable B4 GPU task/binding
topology, proves per-rank GPU-bound execution, creates the intended worker
topology, and implements or explicitly validates dense async DiLoCo update
exchange rather than metadata-only quorum.
