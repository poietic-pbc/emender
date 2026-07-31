# Enable OLCF RCCL Net Plugin for `train.py` Frontier Smoke, 2026-07-07

## Summary

- Slurm job: `4951475` (`e97-s1065-b4-k40-2n-rccl`)
- State: `COMPLETED`, exit `0:0`
- Queue/QOS: `batch` / `debug`
- Nodes/ranks: 2 Frontier nodes, 16 GPU ranks, 8 ranks per node, 1 rank per GPU
- Runtime elapsed: `00:10:25`
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_2n_olcf_rccl/4951475-20260707T121753Z`
- Submit wrapper: `scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch`
- Delegated training path: real `python -u train.py`, not the async/quorum harness and not a reduced RCCL-only diagnostic
- Stdout/stderr:
  - `logs/frontier/scaleout/e97-s1065-b4-k40-2n-rccl-4951475.out`
  - `logs/frontier/scaleout/e97-s1065-b4-k40-2n-rccl-4951475.err`

## Runtime Stack

The delegated `srun` preflight rank reported the intended OLCF ROCm/Torch/Triton environment:

- `delegated_python_version=3.12.13`
- `delegated_torch_version=2.10.0+rocm7.1`
- `delegated_torch_hip=7.1.25424`
- `delegated_triton_version=3.6.0`
- `python_executable=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`
- `frontier_rocm_module=rocm/7.1.1`

The same runtime capture block also printed:

- `python_version=3.12.13`
- `torch.__version__=2.10.0+rocm7.1`
- `torch.version.hip=7.1.25424`
- `triton.__version__=3.6.0`

## RCCL Net Plugin Evidence

`module show rccl-net-plugin/1.0` on Frontier confirms the module provides AWS NCCL/RCCL OFI interfaces `1.19.2`, sets `OLCF_OFI_NCCL_ROOT`, sets the Frontier Slingshot `FI_CXI*`/`NCCL*` recommendations, and requires a compatible ROCm module. It prepends the ROCm-specific plugin directory to `LD_LIBRARY_PATH`.

The smoke requested and required the plugin:

- `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1`
- `FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug`
- `FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0`
- `REQUIRE_RCCL_NET_PLUGIN=1`
- `NCCL_NET_PLUGIN=librccl-net.so`

Submit-side plugin resolution succeeded before training:

- `rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`
- `require_rccl_net_plugin=1`

Delegated `srun` preflight evidence from the GPU-rank environment:

- `frontier_enable_olcf_rccl_plugin=1`
- `OLCF_OFI_NCCL_ROOT=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1`
- `NCCL_NET_PLUGIN=librccl-net.so`
- `librccl_net_path=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`

`rank-start.tsv` contains 16 rows, matching 2 nodes x 8 GPU ranks.

## Wrapper Changes

- `scripts/frontier/frontier_runtime_env.sh` now searches the OLCF plugin root in `lib`, `lib64`, `FRONTIER_ROCM_MODULE/lib`, and `rocm/*/lib` layouts. This matches the observed Frontier module path `/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`.
- `frontier_require_requested_rccl_net_plugin` provides the fail-fast mode. With `REQUIRE_RCCL_NET_PLUGIN=1`, it exits before training if `librccl-net.so` resolves to `not-found`.
- `scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch` now defaults to `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1`, `FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug`, `FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0`, and `REQUIRE_RCCL_NET_PLUGIN=1`.
- `scripts/frontier/e97_1p3b_pretrained_canary.sbatch` uses the shared fail-fast helper both before launch and inside delegated `srun` ranks.
- `train.py` runtime manifests now resolve the same OLCF `rocm/*/lib/librccl-net.so` layout.

## Training Recipe

This smoke preserved the requested `train.py` E97 B4 checkpoint-loading path:

- Resume checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- `--level E97`
- Effective model size: 1.3B, from the existing E97 geometry in the wrapper
- `--batch_size 4`
- `--chunk_size 2048`
- `--resume ...step_1065000/latest.pt`
- `--diloco`
- `--diloco_k 40`
- `--diloco_outer_optimizer avg`
- `--diloco_outer_lr 1.0`
- `--diloco_outer_beta 0.0`
- `--diloco_island_size 1`
- `--diloco_merge_topology global`
- `--diloco_merge_completion_barrier 1`

This is synchronous periodic model averaging. It is not quorum DiLoCo and not the asynchronous quorum harness.

## Training Result

- Start step: `1065000`
- Final step: `1065300`
- Logged training steps: 300
- Global tokens per step: `16 * 4 * 2048 = 131,072`
- Total logged training tokens: `39,321,600`
- Final logged step loss: `2.4397` at step `1065300`
- `FINAL_LOSS_LAST100`: `2.4793`
- Peak memory per rank: `26397 MB`
- Reserved memory per rank: `34152 MB`

Throughput from logged `global_tok/s`:

- Mean: `83,952.6 tok/s`
- Median: `87,417.5 tok/s`
- Min: `13,500 tok/s` (post-merge/checkpoint stall sample)
- Max: `89,187 tok/s`

## DiLoCo Merge Evidence

The run completed seven scheduled K=40 merges plus the final consensus merge:

- Merge 1: step `1065040`, 467 ms
- Merge 2: step `1065080`, 448 ms
- Merge 3: step `1065120`, 458 ms
- Merge 4: step `1065160`, 488 ms
- Merge 5: step `1065200`, 547 ms
- Merge 6: step `1065240`, 522 ms
- Merge 7: step `1065280`, 518 ms
- Final merge 8: step `1065300`, 186 ms

The train log reports:

- `DILOCO_MERGES: 8`
- `DILOCO_K: 40`
- `DILOCO_SYNC_TOTAL_S: 3.633`
- `DILOCO_SYNC_AVG_MS: 454.1`

Final checkpoint:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_e97_step1065000_b4_smoke/20260707/E97_1.3B_step1065000_b4_k40_trainpy_2n_olcf_rccl/4951475-20260707T121753Z/train/emender_E97_1.3B_20260707_081851/checkpoint_step_1065300_loss_2.4793.pt`
- `latest.pt` points to that final checkpoint.

## Validation

- `main` was current with `origin/main` before editing: `git rev-list --left-right --count HEAD...origin/main` returned `0 0`.
- Focused tests passed:
  - `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_frontier_runtime_plumbing.py`
  - Result: `9 passed`
- Shell syntax checks passed:
  - `bash -n scripts/frontier/frontier_runtime_env.sh scripts/frontier/e97_1p3b_pretrained_canary.sbatch scripts/frontier/e97_1p3b_step1065000_b4_trainpy_smoke.sbatch`
- `git diff --check` passed.
- Slurm accounting reports job `4951475` and step `4951475.1` completed with exit `0:0`.
- Log search found no `Traceback`, `RuntimeError`, `non-finite`, `NaN`, `nan`, `FAILED`, `ERROR`, `error`, `timeout`, or `not-found` entries in the committed stdout/stderr or train log. The only match from that grep was the benign blank field `distributed_init_timeout_seconds=`.

## Caveats

- The live smoke used 2 nodes rather than 8 nodes to keep debug-QOS validation bounded. It still exercised real `train.py`, 16 GPU ranks, checkpoint loading, plugin-required startup, and multiple DiLoCo collectives.
- The stdout command-preview line was generated before the preview text was aligned with the new delegated-rank fail-fast check. The actual `srun` command in the wrapper already included the delegated rank check for job `4951475`; the committed wrapper now has the preview and actual command text aligned.
