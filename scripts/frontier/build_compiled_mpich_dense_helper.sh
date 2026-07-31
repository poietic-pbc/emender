#!/bin/bash
# Build the standalone compiled Cray MPICH dense helper on Frontier.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO}/build/frontier}
SRC=${SRC:-${REPO}/scripts/frontier/compiled_mpich_dense_helper.cpp}
OUT=${OUT:-${ARTIFACT_DIR}/compiled_mpich_dense_helper}
OUT_SO=${OUT_SO:-${OUT}.so}
CXX=${CXX:-CC}
CXXFLAGS=${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra}
SHARED_CXXFLAGS=${SHARED_CXXFLAGS:--O2 -std=c++17 -Wall -Wextra -fPIC -shared}

mkdir -p "$ARTIFACT_DIR"
cd "$REPO"

# The helper sends CPU byte buffers read from IPC files.  When Frontier jobs
# have craype-accel-amd-gfx90a loaded for train.py, the Cray wrapper may inject
# ROCm/GTL libraries into this shared object; loading that through ctypes can
# collide with the already-loaded torch ROCm stack.  Unload accelerator target
# modules in this child process and clear GTL add-ons while preserving normal
# Cray MPICH linkage for host buffers.
if command -v module >/dev/null 2>&1; then
  while IFS= read -r loaded_module; do
    case "$loaded_module" in
      craype-accel-*|rocm/*) module unload "$loaded_module" || true ;;
    esac
  done < <(module -t list 2>&1 || true)
fi
for accelerator in amd_gfx906 amd_gfx908 amd_gfx90a amd_gfx940 amd_gfx942 nvidia70 nvidia80 nvidia90 ponteVecchio; do
  export "PE_MPICH_GTL_DIR_${accelerator}="
  export "PE_MPICH_GTL_LIBS_${accelerator}="
done

printf '%q ' "$CXX" $CXXFLAGS "$SRC" -o "$OUT"
printf '\n'
"$CXX" $CXXFLAGS "$SRC" -o "$OUT"
printf '%q ' "$CXX" $SHARED_CXXFLAGS "$SRC" -o "$OUT_SO"
printf '\n'
"$CXX" $SHARED_CXXFLAGS "$SRC" -o "$OUT_SO"
echo "$OUT"
echo "$OUT_SO"
