# Debug Current Runtime With OLCF RCCL Plugin

Date: 2026-07-04  
Task: `debug-current-runtime-rccl-plugin`  
Runtime under test: current known-good training Python environment (`EMENDER_CONDA_ENV=base`) with `rccl-net-plugin/1.0` loaded before launch.

## Verdict

Pass. The current runtime can resolve the OLCF RCCL net plugin, complete a 2-node RCCL/c10d allreduce diagnostic, and run a 2-node E97 resume smoke from the active production `latest.pt` pointer without modifying the production chain symlink.

Recommendation for `debug-updated-olcf-runtime`: proceed. Use this result as the current-runtime baseline. The updated-runtime task should reproduce at least the same evidence set: readable `librccl-net.so`, 2-node c10d/RCCL model-sized allreduce, active-pointer resume, at least one training step, and no production chain pointer update.

## Active Pointer Guard

Production pointer checked before and after the smoke:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Before and after `stat` metadata matched:

```text
'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt' -> '/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt'|1783092774|231
```

The E97 smoke wrapper logged the resolved target at job start:

```text
active_chain_latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
resolved_active_chain_target=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

`CHAIN_LATEST_PATH` and `CHAIN_MANIFEST_PATH` were exported empty for the smoke, and `CHAIN_UPDATE_ON_FAILURE=0`, so the delegated canary runner had no production chain path to update. The final checkpoint that the training script wrote landed only under the isolated debug run directory.

## Plugin Resolution

Both jobs resolved a readable plugin path:

```text
/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
```

The smoke wrapper loaded these modules before delegating to the canary runner:

```text
PrgEnv-gnu/8.7.0
cpe/26.03
miniforge3/23.11.0-0
rocm/7.1.1
craype-accel-amd-gfx90a
rccl-net-plugin/1.0
```

## Job Accounting

| Job | Purpose | Nodes | State | Elapsed | Requested node-hours | Actual node-hours | Logs |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `4939731` | 2-node RCCL/c10d allreduce diagnostic | 2 | `COMPLETED` / `0:0` | `00:00:34` | `0.666667` | `0.018889` | `logs/frontier/rccl_diag/rccl-diag-current-plugin-4939731.out` |
| `4939734` | 2-node E97 resume/load/compile/training smoke | 2 | `COMPLETED` / `0:0` | `00:01:56` | `0.666667` | `0.064444` | `logs/frontier/debug/e97-current-rccl-debug-4939734.out` |

Total actual node-hours spent: `0.083333`.  
Total requested node-hours exposed to debug queue: `1.333334`.

## RCCL/c10d Diagnostic Evidence

Command path: existing `scripts/frontier/rccl_allreduce_diag.sbatch`, submitted as a 2-node debug job with `RCCL_DIAG_ENV_MODE=recommended`, `RCCL_DIAG_ALT_RDZV=1`, `--network=disable_rdzv_get`, and the plugin-loaded module environment.

Artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rccl_diag/20260704/recommended/4939731-20260704T145252Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rccl_diag/20260704/recommended/4939731-20260704T145252Z/artifacts/rank0_results.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rccl_diag/20260704/recommended/4939731-20260704T145252Z/logs/allreduce.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rccl_diag/20260704/recommended/4939731-20260704T145252Z/summaries/summary.md
```

Rank/process evidence:

```text
world_size=16
RCCL_DIAG_RANK ... rank 0 ... hostname frontier09403
RCCL_DIAG_RANK ... rank 15 ... hostname frontier09444
NCCL INFO ncclCommInitRank ... rank 0 nranks 16 ... Init COMPLETE
NCCL INFO ncclCommInitRank ... rank 15 nranks 16 ... Init COMPLETE
```

Rank-0 allreduce results:

| Size | Numel | Status | Elapsed seconds | GiB/s per rank |
| --- | ---: | --- | ---: | ---: |
| scalar | 1 | pass | `0.0011185212060809135` | `0.0000033305495490019594` |
| medium | 1,048,576 | pass | `0.003390358993783593` | `1.152164123965138` |
| model | 1,286,589,072 | pass | `1.2201219331473112` | `3.928228530131705` |

The diagnostic emitted NCCL plugin lines:

```text
NCCL_NET_PLUGIN=librccl-net.so
TUNER/Plugin: Plugin name set by env to librccl-net.so
```

The tuner symbol warnings are expected for this plugin path and did not prevent NCCL communicator initialization or allreduce completion.

## E97 Smoke Evidence

Script added for this task:

```text
scripts/frontier/e97_current_runtime_rccl_plugin_debug.sbatch
```

The wrapper logs the active chain pointer and resolved target, loads `rccl-net-plugin/1.0`, verifies `librccl-net.so`, sets isolated output under `frontier_runs/debug`, clears production chain update variables, and delegates to `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`.

Artifacts:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/summaries/summary.md
```

Smoke launch settings:

```text
resume_checkpoint=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
chain_latest_path=
chain_manifest_path=
chain_update_on_failure=0
frontier_rccl_env=recommended
rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
save_every=999999
output=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/train
```

Distributed initialization evidence:

```text
[DiLoCo] world_size=16 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] periodic model-weight averaging: K=999999 outer_lr=1.0 outer_beta=0.0 (no per-step gradient all-reduce)
[DiLoCo] rank 0/16 bound to cuda:0
[DiLoCo] rank 15/16 bound to cuda:0
```

Resume/load evidence:

```text
Starting training from step 489920...
```

Forward/backward/training evidence:

```text
warmup 1/1 | loss 2.0635
step 489921 | loss 2.3118 | lr 1.01e-03 | grad 1.41 | tok/s 2341 | global_tok/s 37456 | elapsed_h 0.000 | time 2026-07-04T14:55:52+00:00
step 489960 | loss 1.9973 | lr 1.01e-03 | grad 1.33 | tok/s 3070 | global_tok/s 49126 | elapsed_h 0.008 | time 2026-07-04T14:56:21+00:00
Training complete! Final step: 489960
FINAL_LOSS_LAST100: 2.4912
DILOCO_MERGES: 1
```

The final checkpoint was unavoidable because the runner observed the large final-checkpoint walltime margin. It stayed in the debug output tree:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_current_runtime_rccl_plugin_debug/4939734-20260704T145446Z/train/emender_E97_1.3B_20260704_105538/checkpoint_step_489960_loss_2.4912.pt
```

## Notes and Risks

- The current training environment reports `torch==2.8.0.dev20250422+rocm6.4`, `torch.version.hip=6.4...`, and Triton `3.2.0` warnings even though the system module stack includes `rocm/7.1.1`. This is the known-good current runtime baseline, not the updated OLCF-aligned candidate.
- c10d printed IPv6 address-family warnings while connecting to the rendezvous hostname. They were non-fatal in both diagnostic and smoke; all ranks initialized and completed.
- Because `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=1200` equaled the 20-minute debug walltime, the smoke finalized early after 40 steps. That was acceptable for this load/compile/rendezvous smoke and kept the checkpoint under the debug run directory only.
