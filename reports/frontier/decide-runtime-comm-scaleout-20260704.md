# Runtime and Communication Decision for Renewed E97 Scaleout

Task: `decide-runtime-comm-scaleout`
Date: 2026-07-04

## Decision

Go for further scaleout, but do not restart at 256 nodes. Use the updated
OLCF-aligned runtime with `rccl-net-plugin/1.0` loaded and with explicit plugin
path verification. The smallest safe next scaleout from the offered choices is
8 nodes.

Use 8 nodes because the updated runtime has only been proven at 2 nodes for
E97 distributed resume/compile/training and at 1 node for GDN2/FLA fwd/bwd.
The current runtime plus official RCCL plugin is a valid rollback path, but the
OLCF-aligned candidate is now the better forward path: it matches the prepared
Python 3.12 / PyTorch 2.10.0+ROCm 7.1 stack, resolves the official plugin, and
successfully resumed from the active production checkpoint under Slurm. Jumping
directly back to 64, 128, or 256 nodes would conflate a runtime migration with a
large-scale rendezvous retry. The previous 4936017 failure was a 256-node
`DistStoreError` during process-group startup, with `2016/2048` clients joined,
so the next job should first prove that the new runtime/plugin path is stable
beyond the 2-node debug shape.

The pending or previously attempted 256-node shape should remain no-go until an
8-node OLCF runtime run completes cleanly, writes an isolated run checkpoint,
and advances the production chain pointer only after success. A clean 8-node
run should be followed by 32 nodes, then 64 nodes, before considering 128 or
256 again.

## Evidence Comparison

| Question | Current runtime + `rccl-net-plugin/1.0` | Updated OLCF runtime + plugin | Decision impact |
| --- | --- | --- | --- |
| Runtime | Current known-good `EMENDER_CONDA_ENV=base`, torch `2.8.0.dev20250422+rocm6.4`, Triton `3.2.0`, Python 3.10-era stack under ROCm 7.1.1 modules | Candidate prefix `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`, torch `2.10.0+rocm7.1`, HIP `7.1.25424`, Python 3.12 | Updated runtime is better aligned with OLCF guidance and removes the old torch/ROCm mismatch. |
| Plugin resolution | Passed. Both jobs resolved `/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so` | Passed. E97 and GDN2 valid jobs resolved the same readable `librccl-net.so` path | Missing plugin defect from 4936017 is fixed in both candidate paths when the module/path is actually loaded. |
| RCCL/c10d diagnostic | Stronger: 2-node, 16-rank allreduce diagnostic completed, including model-sized `1,286,589,072` element allreduce | No separate allreduce diagnostic in the updated report; E97 2-node distributed init/training passed | Current runtime has stronger direct allreduce evidence, but updated runtime has enough 2-node training evidence to proceed cautiously at 8 nodes. |
| E97 active pointer resume | Passed. 2-node smoke resumed from active production `latest.pt`; started at step `489920`; completed to step `489960`; final loss summary finite | Passed. 2-node E97 job resolved active production `latest.pt`, resumed at step `489920`, compiled fused E97 Triton path, and logged finite losses through step `491062` before Slurm timeout | Both paths actually loaded from the active chain pointer. Updated path ran longer and exercised the new torch/Triton stack. |
| Checkpoint behavior | Wrote a final checkpoint because the final-checkpoint margin equaled debug walltime, but only under the debug run tree | Wrote no `.pt` or `latest.pt` under the E97 debug run root; timed out at allocation boundary because wrapper stop condition was too loose | Neither debug path modified production chain pointers. The next job must use a proven bounded stop condition. |
| Production symlink guard | Production `latest.pt` stat/readlink metadata unchanged before vs after; `CHAIN_LATEST_PATH=` and `CHAIN_UPDATE_ON_FAILURE=0` | Production `latest.pt` metadata and resolved target unchanged after job; `CHAIN_LATEST_PATH=`, `CHAIN_MANIFEST_PATH=`, `CHAIN_UPDATE_ON_FAILURE=0` | Production chain symlinks were not modified by debug jobs. |
| Terminal state | Diagnostic and smoke both `COMPLETED` | E97 job `TIMEOUT` after passing intended smoke evidence; valid GDN2 preflight `COMPLETED`; invalid delegated GDN2 attempt intentionally cancelled | Updated runtime is go for the next bounded scaleout, but do not reuse the exact debug wrapper. |

## Answers

Use the updated OLCF-aligned runtime for the next scaleout attempt. Keep the
current runtime plus plugin as rollback if the candidate fails before or during
early initialization for reasons attributable to torch/Triton/runtime rather
than to launch scale.

The debug jobs did load from the active production pointer successfully. The
current-runtime smoke used the active production chain `latest.pt` symlink as
`RESUME_CHECKPOINT`, resolved it to the verified step-489920 checkpoint, and
logged `Starting training from step 489920...`. The updated-runtime E97 smoke
resolved the same production pointer to
`checkpoint_step_489920_loss_2.4894.pt` and logged `Resumed at step 489920`.
The reports did not identify any separate `last.pt` production chain pointer in
use for these debug validations; the active chain input was `latest.pt`.

Checkpoint writes were isolated. The current-runtime debug smoke wrote a final
checkpoint under
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/.../checkpoint_step_489960_loss_2.4912.pt`
because its final-checkpoint walltime margin forced early finalization. It had
`CHAIN_LATEST_PATH=` and `CHAIN_UPDATE_ON_FAILURE=0`, so it had no production
chain path to update. The updated-runtime E97 debug job wrote no `.pt` or
`latest.pt` under its debug run root. In both cases, production chain symlink
metadata and resolved target were unchanged before versus after.

The plugin is now resolving to a real path:

```text
/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
```

This corrects the earlier stack-alignment defect in which both the successful
256-node smoke `4908935` and failed 256-node job `4936017` had
`NCCL_NET_PLUGIN=librccl-net.so` but `rccl_net_plugin_status=not-found`.

Total actual node-hours spent in debug validation were approximately
`1.149733`:

| Source task | Jobs counted | Actual node-hours |
| --- | --- | ---: |
| `debug-current-runtime-rccl-plugin` | `4939731`, `4939734` | `0.083333` |
| `debug-updated-olcf-runtime` | `4939804`, `4939880`, `4939888` | `1.0664` |
| Total | all debug validation jobs | `1.149733` |

## Recommended Next Job

Submit an 8-node, 1-hour, batch-queue chained E97 continuation from the active
production `latest.pt`, using the updated OLCF runtime and official RCCL plugin.
Do not submit it from this task; the command below is the recommended next
operator action.

Run from the repository root on Frontier:

```bash
mkdir -p logs/frontier/scaleout

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load rccl-net-plugin/1.0

export REPO=/lustre/orion/bif148/scratch/erikgarrison/emender
export ENV_PREFIX=${REPO}/.envs/olcf-rocm711-torch210-py312
export GDN2_PATH=${REPO}/src/GatedDeltaNet-2
export PYTHONPATH=${REPO}:${GDN2_PATH}:${PYTHONPATH:-}
export EMENDER_CONDA_ENV=${ENV_PREFIX}
export AWS_OFI_RCCL_PLUGIN_DIR=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1
export LD_LIBRARY_PATH=${AWS_OFI_RCCL_PLUGIN_DIR}/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}

sbatch -A bif148 -p batch -N 8 -t 01:00:00 -J e97-b4-k80-8n-olcf-chain \
  --network=disable_rdzv_get \
  --export=ALL,REPO,EMENDER_CONDA_ENV,GDN2_PATH,PYTHONPATH,AWS_OFI_RCCL_PLUGIN_DIR,LD_LIBRARY_PATH,WG_TASK_ID=scaleout-8n-olcf-after-debug,SCALEOUT_VARIANT=E97_1.3B_step489920_b4_k80_8n_olcf_rccl_chain,CHAIN_NAME=E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg,CHAIN_UPDATE_ON_FAILURE=0,FRONTIER_RCCL_ENV=recommended,FRONTIER_RCCL_ALT_RDZV=1,NCCL_NET_PLUGIN=librccl-net.so,TRAIN_MINUTES=45,REQUESTED_SECONDS=3600,REQUESTED_WALLTIME=01:00:00,REQUESTED_NODE_HOURS=8.0,DILOCO_K=80,DILOCO_ISLAND_SIZE=1,DILOCO_MERGE_TOPOLOGY=hierarchical,DILOCO_MERGE_GROUP_SIZE=4,DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=8,DILOCO_MERGE_COMPLETION_BARRIER=1,DILOCO_MERGE_BUCKET_NUMEL=67108864,BATCH_SIZE=4,CHUNK_SIZE=2048,LOG_EVERY=10,SAVE_EVERY=80,KEEP_CHECKPOINTS=8,COMPILE_WARMUP_STEPS=1,FRONTIER_PER_RANK_TRITON_CACHE=1,WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=900,WALLTIME_CHECK_EVERY=80,DISTRIBUTED_HEALTH_CHECK_EVERY=80,DISABLE_SCALAR_STATUS_COLLECTIVES=1,STATUS_REQUEST_POLL_SECONDS=1.0,HUMAN_APPROVAL_RECORD='WG decide-runtime-comm-scaleout: 8-node OLCF runtime + rccl-net-plugin chained smoke from active production latest.pt; production chain advances only on train_status=0; do not retry 256n until 8n then 32n pass.' \
  scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch
```

This deliberately reuses the production chain launcher so checkpoint/finalization
behavior is the known chained path, while command-line `sbatch` options override
the script's 64-node and 2-hour defaults. `CHAIN_UPDATE_ON_FAILURE=0` is required
for this first candidate-runtime scaleout so the production pointer advances
only on clean training exit.

## Monitor Stop Conditions

The monitor should cancel the 8-node job and mark the scaleout no-go if any of
these occur:

- The job preflight logs `rccl_net_plugin_status=not-found`, lacks
  `NCCL_NET_PLUGIN=librccl-net.so`, or does not show the real
  `/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`
  path in the environment or manifest.
- Python/torch/HIP/Triton versions do not match the candidate runtime:
  `torch==2.10.0+rocm7.1`, HIP `7.1.25424`, and the candidate prefix under
  `.envs/olcf-rocm711-torch210-py312`.
- Distributed initialization fails, stalls beyond 15 minutes from job start, or
  emits a `DistStoreError`, TCPStore timeout, broken pipe cascade, NCCL/RCCL
  watchdog, communicator initialization failure, or `srun` killed-task pattern.
- The run does not log resume from the production step-489920 checkpoint before
  training starts.
- Loss becomes NaN/non-finite, an OOM/segmentation fault/traceback occurs, or
  rank-0 loss is grossly inconsistent with the debug/current baseline before
  the first DiLoCo merge.
- No first DiLoCo merge completes by the expected K80 cadence after training
  starts, unless the job is already finalizing cleanly due to walltime margin.
- A checkpoint/finalization failure occurs, or the production chain pointer is
  advanced when `train_status != 0`.

If the job is clean, the monitor should confirm:

- active `latest.pt` resolved to the step-489920 seed at startup;
- run manifest records the OLCF candidate runtime and real plugin path;
- at least one finite-loss training window and preferably at least one K80
  DiLoCo merge completed;
- final checkpoint and run-local `latest.pt` were written;
- production chain `latest.pt` advanced atomically only after `train_status=0`;
- no severe RCCL/NCCL/Triton/runtime signatures appeared in stdout, stderr, or
  train logs.

## Remaining Risks

The updated runtime E97 debug job ended in Slurm `TIMEOUT`, not because resume
or training failed, but because the debug wrapper did not stop the loop before
the allocation boundary. The recommended next job avoids that wrapper and uses
the established chained launcher with a shorter requested walltime, normal
walltime finalization checks, and `CHAIN_UPDATE_ON_FAILURE=0`.

The current-runtime path still has stronger direct 2-node model-sized allreduce
evidence than the updated runtime. This is why the next updated-runtime run is
8 nodes rather than 64 or 256 nodes. If the 8-node job fails specifically in
RCCL collectives, run an updated-runtime 8-node `rccl_allreduce_diag` before
trying training again.

The 4936017 failure may have involved startup jitter, filesystem pressure, or
rank-local failure before all ranks joined c10d. Loading the official plugin
fixes a real environment defect, but it does not by itself prove 2048-rank
rendezvous readiness. The scale ladder should be 8 -> 32 -> 64 -> 128/256, not
2 -> 256.

The production chain pointer was manually initialized to a verified seed
checkpoint after earlier cancelled jobs. The next chained job must treat that
pointer as the serialization source of truth and must not use stale direct
checkpoint paths unless rolling back intentionally.

## Rollback

If the 8-node OLCF candidate job fails before training for candidate-runtime
reasons, roll back to the current known-good runtime while keeping the official
RCCL plugin loaded. The rollback command should be the same 8-node shape but
without `EMENDER_CONDA_ENV` pointing at the candidate prefix, and with the
submission shell still loading `rccl-net-plugin/1.0` or exporting
`AWS_OFI_RCCL_PLUGIN_DIR`/`LD_LIBRARY_PATH` so
`librccl-net.so` resolves. Keep `CHAIN_UPDATE_ON_FAILURE=0` for the rollback
probe as well.

If the failure is a scale/rendezvous failure independent of Python runtime, do
not roll back immediately. First run the same OLCF runtime at a smaller or more
diagnostic shape: 8-node RCCL allreduce diagnostic, then 8-node training retry
with a fresh per-rank Triton cache root. Only return to 256 nodes after 8, 32,
and 64 nodes have passed with the real plugin path recorded.
