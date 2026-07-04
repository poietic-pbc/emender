# OLCF Runtime Candidate for E97/GDN2 Debug Validation

Task: `prepare-olcf-runtime-candidate`
Report time: `2026-07-04T15:20Z`
Host context: Frontier-compatible OLCF login/compute session with one visible HIP device during local GPU smokes.

## Decision

Proceed to 1-2 node debug smokes with this runtime candidate.

The candidate environment was created separately from the production Python
environment at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
```

It matches current OLCF PyTorch-on-Frontier guidance for the preferred path:
`PrgEnv-gnu/8.7.0`, `cpe/26.03`, `miniforge3/23.11.0-0`,
`rocm/7.1.1`, Python 3.12, and PyTorch 2.10.0 ROCm 7.1 wheels. OLCF's
software news also recommends `rccl-net-plugin/1.0` for PyTorch users on
Slingshot. Sources checked:

- https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html
- https://docs.olcf.ornl.gov/software/software-news.html
- https://docs.olcf.ornl.gov/systems/frontier_user_guide.html

No production Slurm jobs were submitted, and no production checkpoint or chain
symlink paths were touched.

## Reproducible Commands

The reusable helper is:

```bash
scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Create the prefix:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
SETUP_MODE=create scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Install the candidate packages:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
SETUP_MODE=install-deps scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Run import/version checks only:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
SETUP_MODE=check scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Run import/version checks plus tiny local GPU fwd/bwd smokes:

```bash
cd /lustre/orion/bif148/scratch/erikgarrison/emender
SETUP_MODE=check RUN_GPU_SMOKES=1 scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Exact activation for downstream debug jobs:

```bash
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load rccl-net-plugin/1.0
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
conda activate /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
export REPO=/lustre/orion/bif148/scratch/erikgarrison/emender
export GDN2_PATH="$REPO/src/GatedDeltaNet-2"
export PYTHONPATH="$REPO:$GDN2_PATH:${PYTHONPATH:-}"
```

Rollback path:

```bash
conda deactivate || true
rm -rf /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
```

This rollback deletes only the candidate prefix created for this task. It does
not alter production Python environments, checkpoints, or Slurm artifacts.

## Installed Runtime

Creation command:

```bash
module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a rccl-net-plugin/1.0
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
conda create -y -p /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 python=3.12 -c conda-forge
```

Result:

```text
environment location: /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
python-3.12.13-hd63d673_0_cpython
```

PyTorch install command:

```bash
python -m pip install --no-cache-dir \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/rocm7.1
```

Result:

```text
Successfully installed torch-2.10.0+rocm7.1 torchaudio-2.10.0+rocm7.1
torchvision-0.25.0+rocm7.1 triton-rocm-3.6.0
```

Additional packages installed:

```bash
python -m pip install --no-cache-dir \
  einops tqdm schedulefree datasets tiktoken transformers tokenizers \
  typing_extensions packaging ninja flash-linear-attention
python -m pip install --no-deps -e .
```

Important packaging note: project `pyproject.toml` currently pins
`torch==2.9.1` and `triton==3.5.1`, which would downgrade the OLCF candidate
runtime. The candidate therefore installs `ndm` editable with `--no-deps`
after installing the OLCF torch wheel stack.

## Module and Plugin State

Loaded module set included:

```text
PrgEnv-gnu/8.7.0
cpe/26.03
miniforge3/23.11.0-0
rocm/7.1.1
craype-accel-amd-gfx90a
rccl-net-plugin/1.0
cray-mpich/9.1.0
gcc-native/14.2
```

ROCm paths:

```text
ROCM_PATH=/opt/rocm-7.1.1
HIP_LIB_PATH=/opt/rocm-7.1.1/lib
```

The `rccl-net-plugin/1.0` module added the ROCm 7.1 AWS OFI NCCL plugin
library path:

```text
/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib
```

Plugin-related environment variables observed:

```text
FI_CXI_ATS=0
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
FI_CXI_DISABLE_HOST_REGISTER=1
FI_CXI_RDZV_EAGER_SIZE=0
FI_CXI_RDZV_GET_MIN=0
FI_CXI_RDZV_PROTO=alt_read
FI_CXI_RDZV_THRESHOLD=0
FI_CXI_RX_MATCH_MODE=hybrid
FI_MR_CACHE_MONITOR=kdreg2
MPICH_OFI_NIC_POLICY=NUMA
NCCL_CROSS_NIC=1
NCCL_NET_GDR_LEVEL=PHB
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
```

## Import and Version Results

Command:

```bash
SETUP_MODE=check scripts/frontier/prepare_olcf_runtime_candidate.sh
```

Key results:

| Component | Result |
| --- | --- |
| Python | `3.12.13` from candidate prefix |
| torch | `2.10.0+rocm7.1` |
| `torch.version.hip` | `7.1.25424` |
| Triton | `3.6.0` from `triton-rocm==3.6.0` |
| FLA | `flash-linear-attention/fla` `0.5.1` |
| `ndm` | imported from repo, version `0.2.0` |
| `ndm.models.e88_fla_hybrid` | import OK |
| `ndm.triton.e97_chunked` | import OK |
| `ndm.triton.e97_chunked_autograd` | import OK |
| `ndm.triton.e97_multiquery_autograd` | import OK |
| `ndm.models.external_gdn2` | import OK |
| External GDN2 path | `src/GatedDeltaNet-2` |
| GDN2 chunk op | `src/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py` |
| Required GDN2 symbol | `chunk_gla_fwd_o_gk` present |

The GDN2 dependency probe returned `ok: true` with:

```text
torch_backend=hip
torch_version=2.10.0+rocm7.1
torch_version_hip=7.1.25424
fla_version=0.5.1
missing_required_symbols=[]
```

## Local GPU Smoke Results

These were local tiny fwd/bwd checks only. They were not Slurm submissions.

GDN2 command:

```bash
python scripts/frontier/gdn2_rocm_preflight.py \
  --run-fwdbwd --bf16 \
  --batch-size 1 --seq-len 8 --dim 64 --n-heads 4 --head-dim 16 \
  --expansion 1.0 --mlp-ratio 1.0
```

Result:

```text
device_name: AMD Instinct MI210
dtype: torch.bfloat16
loss: 0.008105617016553879
finite_output: true
finite_loss: true
finite_input_grad: true
finite_param_grads: true
ok: true
```

E97 command: the script's `RUN_GPU_SMOKES=1` path instantiates
`E88FLAHybrid` with `use_split_edit=True`, `use_triton=True`, bf16, shape
`[1, 8, 64]`.

Result:

```text
[e97-runtime] backend=hip path=e88-sequential-split-edit-triton use_triton=True
device_name: AMD Instinct MI210
loss: 0.0013263248838484287
finite_output: true
finite_input_grad: true
finite_param_grads: true
ok: true
```

## Compatibility Assessment

No import-time incompatibility was found for torch, Triton, FLA, `ndm`, E97
Triton modules, or GDN2 modules under the OLCF ROCm 7.1.1/PyTorch 2.10.0
runtime.

No tiny local GPU incompatibility was found for:

- the external GDN2 bf16 fwd/bwd path through FLA and local GDN2 chunk kernels;
- the E97 split-edit Triton fwd/bwd path.

Known caveat before Slurm smokes: this validates single-process imports and
tiny local GPU execution. It does not validate multi-rank RCCL behavior. The
next gate should be the downstream 1-node debug smoke, followed by a 2-node
debug smoke with `rccl-net-plugin/1.0` loaded to validate cross-node transport.

## Recommendation

Use this runtime for the downstream debug task:

1. Run the 1-node debug smoke with the activation block above and
   `rccl-net-plugin/1.0` loaded.
2. If the 1-node smoke passes, run a 2-node debug smoke to test RCCL/Slingshot
   transport under the official plugin.
3. Keep per-rank Triton cache isolation enabled in debug jobs to avoid compile
   cache races during first backward.
4. Do not update production symlinks or checkpoint chain targets until the
   debug smokes pass.
