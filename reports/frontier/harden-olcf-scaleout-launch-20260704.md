# Harden OLCF Scaleout Launch Before Ladder

Task: `harden-olcf-scaleout-launch`
Date: 2026-07-04
Branch: `main`

## Result

Status: go for launcher-side hardening only. No scaleout or production Slurm jobs were submitted by this task, and no production chain symlink was modified.

I tightened the merged launcher path for the OLCF debug/ladder flow and committed the changes. The goal is fail-fast setup evidence, not elastic training.

## Launch-Hardening Checks Present In `main`

1. OLCF runtime candidate selection for ladder jobs:
   - `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch` now defaults `ENV_PREFIX` and `EMENDER_CONDA_ENV` to `${REPO}/.envs/olcf-rocm711-torch210-py312`.
   - It defaults `FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1`, `FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug`, and `FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0`.
   - Evidence: `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch:32-60`.

2. OLCF RCCL plugin load and mandatory `librccl-net.so` resolution:
   - The delegated canary sources `scripts/frontier/frontier_runtime_env.sh` and calls `frontier_load_default_modules`, which loads `rccl-net-plugin/1.0` when OLCF plugin mode is enabled.
   - The canary resolves `librccl-net.so` through `frontier_resolve_librccl_net`.
   - If `REQUIRE_RCCL_NET_PLUGIN=1` and resolution is `not-found`, the launcher exits before `srun`/training with: `librccl-net.so was not found after runtime setup; refusing to start ladder training`.
   - The ladder wrapper and OLCF debug wrapper both set `REQUIRE_RCCL_NET_PLUGIN=1`.
   - Evidence: `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:101-108`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:154-158`, `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch:57-60`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:72-76`.

3. Active production `latest.pt` resolution/logging at debug job start:
   - `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch` resolves `PRODUCTION_LATEST` with `readlink -f`, verifies the resolved checkpoint is readable, records pre-run symlink metadata, logs both the symlink and resolved target, and passes the resolved checkpoint to the delegated canary.
   - Evidence: `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:24-47`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:63-66`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:112-120`.

4. Debug output isolation and production chain protection:
   - The OLCF debug wrapper defaults `OUTPUT_ROOT` to `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug`.
   - It clears `CHAIN_LATEST_PATH` and `CHAIN_MANIFEST_PATH`, sets `CHAIN_UPDATE_ON_FAILURE=0`, and checks production `latest.pt` metadata after the run, exiting with error if the symlink metadata changed.
   - Evidence: `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:27-28`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:67-69`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:150-156`.

5. Production chain update policy:
   - The delegated canary updates a chain symlink only when `CHAIN_LATEST_PATH` is non-empty, a run `latest.pt` exists, and `TRAIN_STATUS == 0 || CHAIN_UPDATE_ON_FAILURE == 1`.
   - Therefore production may update after clean `train_status=0`; debug wrappers clear the chain path and set `CHAIN_UPDATE_ON_FAILURE=0`, so they cannot advance the production chain pointer.
   - Evidence: `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:400-412`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:67-69`.

6. Explicit distributed init timeout:
   - `train.py` supports `--distributed_init_timeout_seconds` and `NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS`; it passes the timeout into `torch.distributed.init_process_group`.
   - The ladder and OLCF debug wrapper now default `NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=1800`.
   - The delegated canary forwards the value as `--distributed_init_timeout_seconds` and logs/manifests it.
   - Evidence: `train.py:449-457`, `train.py:1934-1938`, `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch:54`, `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch:76`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:60`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:205-207`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:269`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:333`.

7. Unique `MASTER_PORT` default:
   - The canary now derives `MASTER_PORT` from `SLURM_JOB_ID` when available: `20000 + (SLURM_JOB_ID % 40000)`.
   - It retains `3442` only for non-Slurm/manual fallback or explicit `MASTER_PORT` override.
   - Evidence: `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:121-128`.

8. Cheap rank-start evidence:
   - The canary writes one tab-separated line per rank to `${RUN_ROOT}/artifacts/rank-start.tsv` immediately inside the `srun` rank shell, before exporting `RANK/WORLD_SIZE/LOCAL_RANK` and starting `train.py`.
   - Fields: UTC timestamp, `SLURM_PROCID`, `SLURM_LOCALID`, node name, and `SLURM_NTASKS`.
   - The env file and manifest include the rank-start log path.
   - Evidence: `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:93`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:118`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:275`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:285-293`, `scripts/frontier/e97_1p3b_pretrained_canary.sbatch:340`.

9. Runtime manifest evidence inside training:
   - `train.py` records Frontier runtime details including `OLCF_OFI_NCCL_ROOT`, `NCCL_NET_PLUGIN`, selected Frontier runtime env vars, and resolved `librccl-net.so` or `not-found`.
   - Evidence: `train.py:868-919`.

## Required Env Vars For Ladder Jobs

Use these defaults for the 8/64/256 ladder unless there is a deliberate override recorded in the task report:

```bash
ENV_PREFIX=${REPO}/.envs/olcf-rocm711-torch210-py312
EMENDER_CONDA_ENV=${ENV_PREFIX}
FRONTIER_RCCL_ENV=recommended
FRONTIER_RCCL_ALT_RDZV=1
FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug
FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0
REQUIRE_RCCL_NET_PLUGIN=1
NCCL_NET_PLUGIN=librccl-net.so
NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=1800
DISTRIBUTED_HEALTH_CHECK_EVERY=160
WALLTIME_CHECK_EVERY=160
CHAIN_UPDATE_ON_FAILURE=0
```

For debug probes using `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch`, also keep:

```bash
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug
PRODUCTION_LATEST=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
CHAIN_LATEST_PATH=
CHAIN_MANIFEST_PATH=
SAVE_EVERY=999999
KEEP_CHECKPOINTS=1
DILOCO_K=1000000
DISABLE_SCALAR_STATUS_COLLECTIVES=1
```

For production chain jobs, do not use the debug wrapper. Production may set `CHAIN_LATEST_PATH` and `CHAIN_MANIFEST_PATH`, but should keep `CHAIN_UPDATE_ON_FAILURE=0` unless a human explicitly accepts failure-updates. With the current canary logic, a production chain pointer advances after `train_status=0`; with `CHAIN_UPDATE_ON_FAILURE=0`, failed training leaves it unchanged.

`MASTER_PORT` should normally be omitted so the launcher derives it from `SLURM_JOB_ID`. Override only if there is a known collision.

## Fault-Tolerance Limitation

The current PyTorch/RCCL DiLoCo path is not rank-dropout tolerant. It initializes a vanilla `torch.distributed` process group with `WORLD_SIZE` ranks and requires all ranks to join and participate in required collectives. Missing ranks, lost nodes, or a bad rendezvous are job failures, not recoverable elastic events. There is no implemented or validated design here that can ignore missing ranks or continue with a smaller world size.

Operationally: if a debug run dies from missing ranks or transient node failure, retry once only on a fresh allocation if setup evidence is otherwise correct. If it repeats at the same scale, stop the ladder and report no-go.

## Validation

Commands run:

```bash
bash -n scripts/frontier/frontier_runtime_env.sh \
  scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch \
  scripts/frontier/e97_1p3b_pretrained_canary.sbatch \
  scripts/frontier/e97_updated_olcf_runtime_debug.sbatch

python3 -m py_compile train.py tests/test_frontier_runtime_plumbing.py

git diff --check -- \
  scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch \
  scripts/frontier/e97_1p3b_pretrained_canary.sbatch \
  scripts/frontier/e97_updated_olcf_runtime_debug.sbatch \
  tests/test_frontier_runtime_plumbing.py

python3 - <<'PY'
from pathlib import Path
root = Path('.')
ladder = (root / 'scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch').read_text()
canary = (root / 'scripts/frontier/e97_1p3b_pretrained_canary.sbatch').read_text()
debug = (root / 'scripts/frontier/e97_updated_olcf_runtime_debug.sbatch').read_text()
checks = [
    '.envs/olcf-rocm711-torch210-py312' in ladder,
    'FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-1}' in ladder,
    'FRONTIER_RCCL_NET_PLUGIN_MODULE=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}' in ladder,
    'REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}' in ladder,
    'NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}' in ladder,
    'frontier_load_default_modules' in canary,
    'frontier_derive_master_port' in canary,
    'RCCL_NET_PLUGIN_STATUS=$(frontier_resolve_librccl_net)' in canary,
    'refusing to start ladder training' in canary,
    '--distributed_init_timeout_seconds' in canary,
    'RANK_START_LOG' in canary,
    'export CHAIN_UPDATE_ON_FAILURE=0' in debug,
    'REQUIRE_RCCL_NET_PLUGIN=${REQUIRE_RCCL_NET_PLUGIN:-1}' in debug,
    'NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS=${NDM_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-1800}' in debug,
    'production latest.pt metadata changed during debug smoke' in debug,
]
if not all(checks):
    raise SystemExit(f'static check failed at {[i for i, ok in enumerate(checks) if not ok]}')
print('static launcher hardening assertions passed')
PY
```

Results:

- `bash -n`: passed.
- `python3 -m py_compile`: passed.
- `git diff --check`: passed.
- Static launcher hardening assertions: passed.
- `python3 -m pytest tests/test_frontier_runtime_plumbing.py`: not run because this environment has no `pytest` module (`/usr/bin/python3: No module named pytest`), matching the upstream integration task limitation.

No `sbatch`, `srun`, scaleout/prod Slurm submission, or production chain-symlink mutation was performed by this hardening task.

## Files Changed

- `scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch`
- `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`
- `scripts/frontier/e97_updated_olcf_runtime_debug.sbatch`
- `tests/test_frontier_runtime_plumbing.py`
- `reports/frontier/harden-olcf-scaleout-launch-20260704.md`
