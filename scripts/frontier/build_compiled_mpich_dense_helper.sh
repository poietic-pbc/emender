#!/bin/bash
# Build the standalone compiled Cray MPICH dense helper on Frontier.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO}/build/frontier}
SRC=${SRC:-${REPO}/scripts/frontier/compiled_mpich_dense_helper.cpp}
OUT=${OUT:-${ARTIFACT_DIR}/compiled_mpich_dense_helper}
CXX=${CXX:-CC}
CXXFLAGS=${CXXFLAGS:--O2 -std=c++17 -Wall -Wextra}

mkdir -p "$ARTIFACT_DIR"
cd "$REPO"

printf '%q ' "$CXX" $CXXFLAGS "$SRC" -o "$OUT"
printf '\n'
"$CXX" $CXXFLAGS "$SRC" -o "$OUT"
echo "$OUT"
