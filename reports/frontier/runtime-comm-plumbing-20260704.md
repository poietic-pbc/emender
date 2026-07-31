# Frontier Runtime/Communication Plumbing Report

Task: `frontier-runtime-comm-plumbing`
Date: 2026-07-04

## Changed Files

- `scripts/frontier/frontier_runtime_env.sh`
  - Added shared Frontier debug runtime helpers.
  - `frontier_load_default_modules` preserves the existing ROCm-only default path.
  - Setting `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1` or `FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug` loads `rccl-net-plugin/1.0` after `FRONTIER_ROCM_MODULE` (default `rocm/7.1.1`) and exports `NCCL_NET_PLUGIN=librccl-net.so` if the launcher did not already set it.
  - `frontier_resolve_librccl_net` checks `OLCF_OFI_NCCL_ROOT`, optional `AWS_OFI_RCCL_PLUGIN_DIR`, and `LD_LIBRARY_PATH` for a readable `librccl-net.so`.
  - `frontier_capture_runtime_env` records `OLCF_OFI_NCCL_ROOT`, the resolved plugin path, `FI_CXI*`, `NCCL*`, `HSA_FORCE_FINE_GRAIN_PCIE`, Python, torch, HIP, and Triton versions.

- `scripts/frontier/debug_smoke_one_node.slurm`
  - Sources the shared helper and records the runtime capture into `artifacts/env.txt`.
  - Adds manifest fields for runtime profile, selected ROCm/plugin module, `OLCF_OFI_NCCL_ROOT`, `NCCL_NET_PLUGIN`, and resolved `librccl-net.so`.
  - Does not enable the OLCF plugin by default. Downstream debug jobs must opt in explicitly.

- `scripts/frontier/rccl_allreduce_diag.sbatch`
  - Sources the shared helper, uses the shared plugin resolver, and captures the same runtime evidence in the diagnostic env artifact.
  - Existing `RCCL_DIAG_ENV_MODE=current` and `recommended` behavior remains available; the OLCF plugin module is only loaded with the explicit debug runtime flag/profile.

- `train.py`
  - Adds `frontier_runtime_manifest()` to the training run manifest with Python, torch, HIP, Triton, `OLCF_OFI_NCCL_ROOT`, resolved `librccl-net.so`, selected `FI_CXI*`, `NCCL*`, `HSA_FORCE_FINE_GRAIN_PCIE`, and relevant Frontier runtime env flags.
  - Adds opt-in `--distributed_init_timeout_seconds` / `NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS` for `torch.distributed.init_process_group`. Unset preserves PyTorch defaults; use a finite debug value such as `900` to avoid masking rendezvous failures indefinitely.

- `tests/test_frontier_runtime_plumbing.py`
  - Adds static coverage for module ordering, opt-in plugin loading, manifest capture fields, and shared diagnostic plugin resolution.
  - Exercises the Python `librccl-net.so` resolver with a temporary `OLCF_OFI_NCCL_ROOT`.

## Risk

- Default production behavior is intentionally unchanged unless a debug job sets `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1` or `FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug`.
- The helper is only wired into the debug smoke and RCCL diagnostic wrappers in this patch. Active production chain pointer scripts and checkpoint symlinks were not modified.
- The plugin resolver proves whether a readable `librccl-net.so` appears in expected paths. It does not prove RCCL actually uses the plugin; downstream debug validation should inspect NCCL/RCCL INFO logs and allreduce/training behavior.
- `--distributed_init_timeout_seconds` should be used only with bounded debug values. Very large values would delay real rendezvous failures; unset leaves the existing default behavior.

## Downstream Debug Commands

Run from the repository root on Frontier after ensuring logs and data paths exist. These commands submit debug-QOS jobs only.

Current runtime with OLCF RCCL plugin on the one-node smoke:

```bash
mkdir -p logs/frontier/debug
FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1 \
NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=900 \
SMOKE_VARIANT=e97-MLP \
DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt \
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt \
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

Equivalent profile-based selector:

```bash
mkdir -p logs/frontier/debug
FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug \
NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=900 \
SMOKE_VARIANT=gdn2-MLP \
DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt \
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt \
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

RCCL allreduce diagnostic with the OLCF plugin module:

```bash
mkdir -p logs/frontier/rccl_diag
FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1 \
RCCL_DIAG_ENV_MODE=recommended \
RCCL_DIAG_ALT_RDZV=1 \
sbatch -N 2 -J rccl-olcf-plugin-2n \
  --export=ALL,FRONTIER_ENABLE_OLCF_RCCL_PLUGIN,RCCL_DIAG_ENV_MODE,RCCL_DIAG_ALT_RDZV \
  scripts/frontier/rccl_allreduce_diag_alt_rdzv.sbatch
```

After each job, check:

```bash
grep -E 'OLCF_OFI_NCCL_ROOT|NCCL_NET_PLUGIN|librccl_net_path|torch.__version__|torch.version.hip|triton.__version__|python_version' \
  /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/*/*/*/artifacts/env.txt
```

For training jobs, also inspect the rank-0 training `run_manifest.json` under the run's `train/` directory and verify:

- `runtime.librccl_net_path` is a readable path, not `not-found`.
- `runtime.env.NCCL_NET_PLUGIN` is `librccl-net.so`.
- `runtime.torch_version`, `runtime.torch_version_hip`, `runtime.triton_version`, and `runtime.python_version` match the intended debug runtime.

## Validation Notes

- No Slurm jobs were submitted by this task.
- No `latest.pt` or `last.pt` symlink was modified.
- The code changes are limited to Frontier debug launch/runtime plumbing, training runtime metadata, focused tests, and this report.
