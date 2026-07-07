# E97 train.py updated-runtime 8-node smoke, 2026-07-07

## Summary

- Slurm job: `4951457` (`e97-s1065-b4-k40-trainpy-8n-rt71`)
- State: `COMPLETED`, exit `0:0`
- Queue/QOS: debug
- Nodes/ranks: 8 Frontier nodes, 64 GPU ranks, 8 ranks per node
- Runtime elapsed: `00:10:21`
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n_rocm711_torch210/4951457-20260707T115405Z`
- Resume checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`

## Runtime stack

The smoke used the updated local OLCF ROCm 7.1 runtime environment:

- `EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`
- Python: `3.12.13`
- Torch: `2.10.0+rocm7.1`
- `torch.version.hip=7.1.25424`
- Triton: `3.6.0`
- ROCm module: `rocm/7.1.1`

The delegated `srun` ranks also reported the same Python executable and package versions, so this was not the old system/miniforge runtime.

RCCL environment:

- `FRONTIER_RCCL_ENV=recommended`
- `NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`
- `NCCL_NET_PLUGIN=librccl-net.so`
- External RCCL net plugin was not enabled for this smoke: `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=0`, `librccl_net_path=not-found`, `require_rccl_net_plugin=0`

## DiLoCo mode

This was not quorum DiLoCo and not asynchronous quorum DiLoCo.

It used `train.py` synchronous periodic model averaging:

- `--diloco`
- `--diloco_k 40`
- `--diloco_outer_optimizer avg`
- `--diloco_outer_lr 1.0`
- `--diloco_outer_beta 0.0`
- `--diloco_island_size 1`
- `--diloco_merge_topology global`
- `--diloco_merge_completion_barrier 1`

Operationally: each GPU rank trains independently for 40 local optimizer steps, then all 64 ranks synchronously average model weights. There is no per-step DDP gradient all-reduce, but the K-step merge still requires all 64 ranks for the collective path used here.

The async/quorum harness was not used in this job.

## Training result

- Start step: `1065000`
- Final step: `1065280`
- Local optimizer steps completed: 280
- Global tokens per step: `64 * 4 * 2048 = 524,288`
- Total tokens processed: `146,800,640`
- `FINAL_LOSS_LAST100`: `2.4949`
- Peak memory per rank: `26397 MB`
- Reserved memory per rank: `34152 MB`

Throughput from logged `global_tok/s`:

- Logged step samples: 280
- Mean: `325,180 tok/s`
- Median: `342,584 tok/s`
- Min: `53,073 tok/s` (post-merge/checkpoint stall sample)
- Max: `353,439 tok/s`

## Merge and checkpoint behavior

The run completed seven K=40 DiLoCo merge windows:

- Merge 1: step `1065040`, 2639 ms
- Merge 2: step `1065080`, 2467 ms
- Merge 3: step `1065120`, 2396 ms
- Merge 4: step `1065160`, 3263 ms
- Merge 5: step `1065200`, 2528 ms
- Merge 6: step `1065240`, 2506 ms
- Merge 7: step `1065280`, 2501 ms

Finalization behavior was correct:

- Final merge was skipped because step `1065280` had already merged (`step % K == 0`), so the final checkpoint was already consensus.
- Final checkpoint completed:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_smoke_8n_rocm711_torch210/4951457-20260707T115405Z/train/emender_E97_1.3B_20260707_075518/checkpoint_step_1065280_loss_2.4949.pt`
- `latest.pt` points to `checkpoint_step_1065280_loss_2.4949.pt`.
- Retention kept the last three checkpoint files:
  - `checkpoint_step_1065200_loss_2.6485.pt`
  - `checkpoint_step_1065240_loss_2.0373.pt`
  - `checkpoint_step_1065280_loss_2.4949.pt`

## Validation

- Slurm completion checked with `sacct`: job `4951457` completed with exit `0:0`.
- `rank-start.tsv` has 64 rank records.
- Log search found no `Traceback`, `RuntimeError`, `non-finite`, `NaN`, `nan`, `inf`, `FAILED`, or `ERROR` entries in stdout, stderr, or train log.
- Slurm stdout/stderr captured in:
  - `logs/frontier/scaleout/e97-s1065-b4-k40-trainpy-8n-rt71-4951457.out`
  - `logs/frontier/scaleout/e97-s1065-b4-k40-trainpy-8n-rt71-4951457.err`
