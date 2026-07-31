#!/bin/bash
# Build and/or validate the OLCF-aligned Emender runtime candidate on Frontier.
#
# This script intentionally uses a separate conda prefix and refuses to
# overwrite an existing non-empty prefix unless SETUP_MODE=check.

set -euo pipefail

REPO=${REPO:-/lustre/orion/bif148/scratch/erikgarrison/emender}
ENV_PREFIX=${ENV_PREFIX:-"${REPO}/.envs/olcf-rocm711-torch210-py312"}
GDN2_PATH=${GDN2_PATH:-"${REPO}/src/GatedDeltaNet-2"}
SETUP_MODE=${SETUP_MODE:-check}  # create, install-deps, check
RUN_GPU_SMOKES=${RUN_GPU_SMOKES:-0}

if ! command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load rccl-net-plugin/1.0

# OLCF requires this when using a non-default CPE.
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
export REPO
export GDN2_PATH
export PYTHONPATH="${REPO}:${GDN2_PATH}:${PYTHONPATH:-}"

case "${SETUP_MODE}" in
  create)
    if [ -e "${ENV_PREFIX}" ]; then
      echo "ERROR: ${ENV_PREFIX} already exists; refusing to overwrite." >&2
      exit 20
    fi
    conda create -y -p "${ENV_PREFIX}" python=3.12 -c conda-forge
    ;;
  install-deps)
    if [ ! -d "${ENV_PREFIX}" ]; then
      echo "ERROR: ${ENV_PREFIX} does not exist. Run SETUP_MODE=create first." >&2
      exit 21
    fi
    ;;
  check)
    if [ ! -d "${ENV_PREFIX}" ]; then
      echo "ERROR: ${ENV_PREFIX} does not exist. Run SETUP_MODE=create first." >&2
      exit 21
    fi
    ;;
  *)
    echo "ERROR: SETUP_MODE must be create, install-deps, or check." >&2
    exit 22
    ;;
esac

# shellcheck disable=SC1091
source activate "${ENV_PREFIX}"

if [ "${SETUP_MODE}" = "install-deps" ]; then
  python -m pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/rocm7.1
  python -m pip install --no-cache-dir \
    einops tqdm schedulefree datasets tiktoken transformers tokenizers \
    typing_extensions packaging ninja flash-linear-attention
  python -m pip install --no-deps -e "${REPO}"
fi

python - <<'PY'
import importlib
import json
import os
import sys
import traceback


def probe_import(name):
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": repr(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


report = {
    "python": sys.version,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "loaded_modules": os.environ.get("LOADEDMODULES", ""),
    "rocm_path": os.environ.get("ROCM_PATH"),
    "hip_lib_path": os.environ.get("HIP_LIB_PATH"),
    "rccl_plugin_env": {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(("NCCL", "RCCL", "FI_CXI", "FI_MR", "MPICH"))
    },
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
}

for name in [
    "torch",
    "triton",
    "fla",
    "ndm",
    "ndm.models.e88_fla_hybrid",
    "ndm.triton.e97_chunked",
    "ndm.triton.e97_chunked_autograd",
    "ndm.triton.e97_multiquery_autograd",
    "ndm.models.external_gdn2",
]:
    report[name] = probe_import(name)

try:
    import torch

    report["torch"].update(
        {
            "hip": getattr(torch.version, "hip", None),
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_is_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
        }
    )
except Exception:
    pass

try:
    from ndm.models.external_gdn2 import probe_gdn2_external_dependencies

    report["gdn2_probe"] = probe_gdn2_external_dependencies()
except Exception as exc:
    report["gdn2_probe"] = {
        "ok": False,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }

print(json.dumps(report, indent=2, sort_keys=True))
PY

if [ "${RUN_GPU_SMOKES}" = "1" ]; then
  python "${REPO}/scripts/frontier/gdn2_rocm_preflight.py" \
    --run-fwdbwd --bf16 \
    --batch-size 1 --seq-len 8 --dim 64 --n-heads 4 --head-dim 16 \
    --expansion 1.0 --mlp-ratio 1.0
  python - <<'PY'
import json
import traceback

import torch

from ndm.models.e88_fla_hybrid import E88FLAHybrid

report = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
}
try:
    torch.manual_seed(123)
    layer = E88FLAHybrid(
        dim=64,
        n_heads=4,
        n_state=16,
        expansion=1.0,
        use_gate=True,
        gate_activation="silu",
        use_split_edit=True,
        use_triton=True,
        linear_state=False,
        use_chunked_e97=False,
    ).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, state = layer(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    report.update(
        {
            "ok": True,
            "device_name": torch.cuda.get_device_name(0),
            "output_shape": list(y.shape),
            "state_is_none": state is None,
            "loss": float(loss.detach().cpu()),
            "finite_output": bool(torch.isfinite(y.float()).all().item()),
            "finite_input_grad": bool(torch.isfinite(x.grad.float()).all().item()),
            "finite_param_grads": all(
                p.grad is None or bool(torch.isfinite(p.grad.float()).all().item())
                for p in layer.parameters()
            ),
        }
    )
except Exception as exc:
    report.update({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})

print(json.dumps({"e97_fwdbwd": report}, indent=2, sort_keys=True))
raise SystemExit(0 if report.get("ok") else 1)
PY
fi
