#!/bin/bash
# Shared Frontier runtime helpers for debug launchers.
#
# Defaults preserve the historical ROCm-only path.  Set either
# FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1 or FRONTIER_RUNTIME_PROFILE=olcf-rccl-debug
# to load OLCF's rccl-net-plugin module after the selected ROCm module.

frontier_runtime_olcf_rccl_enabled() {
  [[ "${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-0}" == "1" || \
     "${FRONTIER_RUNTIME_PROFILE:-}" == "olcf-rccl-debug" ]]
}

frontier_load_default_modules() {
  local rocm_module="${FRONTIER_ROCM_MODULE:-rocm/7.1.1}"
  local rccl_plugin_module="${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}"

  module load PrgEnv-gnu/8.7.0
  module load cpe/26.03
  module load miniforge3/23.11.0-0
  module load "$rocm_module"
  if frontier_runtime_olcf_rccl_enabled; then
    module load "$rccl_plugin_module"
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-librccl-net.so}"
  fi
  module load craype-accel-amd-gfx90a
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
}

frontier_resolve_rocm_runtime_dir() {
  local root="${ROCM_PATH:-${ROCM_HOME:-}}"
  local candidate
  [[ -n "$root" && "$root" == /* ]] || {
    echo "the reviewed ROCm module did not set an absolute ROCM_PATH/ROCM_HOME" >&2
    return 66
  }
  root=$(readlink -f "$root") || return
  for candidate in "$root/lib" "$root/lib64"; do
    if [[ -r "$candidate/libamdhip64.so.7" ]]; then
      readlink -f "$candidate"
      return 0
    fi
  done
  echo "libamdhip64.so.7 is absent from reviewed ROCm root $root" >&2
  return 66
}

frontier_resolve_libfabric_runtime_dir() {
  local directory
  directory=$(pkg-config --variable=libdir libfabric 2>/dev/null) || {
    echo "reviewed Frontier modules did not expose libfabric through pkg-config" >&2
    return 66
  }
  [[ "$directory" == /* ]] || {
    echo "libfabric pkg-config returned a non-absolute library directory" >&2
    return 66
  }
  directory=$(readlink -f "$directory") || return
  [[ -r "$directory/libfabric.so.1" ]] || {
    echo "libfabric.so.1 is absent from reviewed module directory $directory" >&2
    return 66
  }
  printf '%s\n' "$directory"
}

frontier_assert_emender_conda_env() {
  local expected_prefix="${EMENDER_CONDA_ENV:-}"
  local active_prefix
  local expected_real
  local active_real

  [[ -n "$expected_prefix" ]] || return 0

  case "$expected_prefix" in
    /*|.*|*/*)
      expected_real=$(readlink -f "$expected_prefix")
      active_prefix=$(python - <<'PY'
import sys
print(sys.prefix)
PY
)
      active_real=$(readlink -f "$active_prefix")
      if [[ "$active_real" != "$expected_real" ]]; then
        echo "active Python prefix ${active_real} does not match EMENDER_CONDA_ENV ${expected_real}" >&2
        return 4
      fi
      ;;
  esac
}

frontier_activate_emender_conda_env() {
  local conda_base

  [[ -n "${EMENDER_CONDA_ENV:-}" ]] || return 0

  conda_base=$(conda info --base)
  # shellcheck disable=SC1090
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
  frontier_assert_emender_conda_env
}

frontier_resolve_librccl_net() {
  local candidates=()
  local globbed=()
  local part

  if [[ -n "${OLCF_OFI_NCCL_ROOT:-}" ]]; then
    candidates+=("${OLCF_OFI_NCCL_ROOT%/}/lib/librccl-net.so")
    candidates+=("${OLCF_OFI_NCCL_ROOT%/}/lib64/librccl-net.so")
    if [[ -n "${FRONTIER_ROCM_MODULE:-}" ]]; then
      candidates+=("${OLCF_OFI_NCCL_ROOT%/}/${FRONTIER_ROCM_MODULE%/}/lib/librccl-net.so")
    fi
    globbed=("${OLCF_OFI_NCCL_ROOT%/}"/rocm/*/lib/librccl-net.so)
    candidates+=("${globbed[@]}")
  fi
  if [[ -n "${AWS_OFI_RCCL_PLUGIN_DIR:-}" ]]; then
    candidates+=("${AWS_OFI_RCCL_PLUGIN_DIR%/}/lib/librccl-net.so")
  fi
  IFS=':' read -r -a _frontier_ld_parts <<< "${LD_LIBRARY_PATH:-}"
  for part in "${_frontier_ld_parts[@]}"; do
    [[ -n "$part" ]] && candidates+=("${part%/}/librccl-net.so")
  done

  for part in "${candidates[@]}"; do
    if [[ -r "$part" ]]; then
      printf '%s\n' "$part"
      return 0
    fi
  done
  printf 'not-found\n'
}

frontier_require_requested_rccl_net_plugin() {
  local context="${1:-runtime setup}"
  local librccl_net_path

  librccl_net_path=$(frontier_resolve_librccl_net)
  if [[ "${REQUIRE_RCCL_NET_PLUGIN:-0}" == "1" && "$librccl_net_path" == "not-found" ]]; then
    echo "librccl-net.so was not found after ${context}; refusing to start training" >&2
    return 4
  fi
  printf '%s\n' "$librccl_net_path"
}

frontier_capture_runtime_env() {
  local librccl_net_path="${1:-$(frontier_resolve_librccl_net)}"

  echo "frontier_runtime_profile=${FRONTIER_RUNTIME_PROFILE:-default}"
  echo "frontier_enable_olcf_rccl_plugin=${FRONTIER_ENABLE_OLCF_RCCL_PLUGIN:-0}"
  echo "frontier_rocm_module=${FRONTIER_ROCM_MODULE:-rocm/7.1.1}"
  echo "frontier_rccl_net_plugin_module=${FRONTIER_RCCL_NET_PLUGIN_MODULE:-rccl-net-plugin/1.0}"
  echo "EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-}"
  echo "CONDA_PREFIX=${CONDA_PREFIX:-}"
  echo "OLCF_OFI_NCCL_ROOT=${OLCF_OFI_NCCL_ROOT:-}"
  echo "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-}"
  echo "librccl_net_path=${librccl_net_path}"
  echo
  echo "=== frontier communication env ==="
  env | sort | grep -E '^(FI_|MPICH_|NCCL|RCCL|HSA_FORCE_FINE_GRAIN_PCIE|OLCF_OFI_NCCL_ROOT|LD_LIBRARY_PATH|ROCM|FRONTIER_RUNTIME_PROFILE|FRONTIER_ENABLE_OLCF_RCCL_PLUGIN|FRONTIER_ROCM_MODULE|FRONTIER_RCCL_NET_PLUGIN_MODULE|EMENDER_CONDA_ENV|CONDA_PREFIX)=' || true
  echo
  echo "=== python runtime versions ==="
  python - <<'PY'
import importlib.metadata as md
import platform
import sys

print(f"python_executable={sys.executable}")
print(f"python_version={platform.python_version()}")
try:
    import torch
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
except Exception as exc:
    print(f"torch_import_error={exc!r}")
try:
    import triton
    print(f"triton.__version__={triton.__version__}")
except Exception:
    try:
        print(f"triton.__version__={md.version('triton')}")
    except Exception as exc:
        print(f"triton_import_error={exc!r}")
PY
}
