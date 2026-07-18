#!/bin/bash
# Build the model-free node-local native data-plane library on Frontier or CI.

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SOURCE_DIR=${SOURCE_DIR:-${REPO}/src/native_resilient_dataplane}
BUILD_DIR=${BUILD_DIR:-${REPO}/build/native-dataplane}
CMAKE_BIN=${CMAKE_BIN:-cmake}
CXX_COMPILER=${CXX_COMPILER:-CC}
BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}
NDP_ENABLE_XPMEM=${NDP_ENABLE_XPMEM:-ON}

"$CMAKE_BIN" -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DNDP_ENABLE_XPMEM="$NDP_ENABLE_XPMEM"
"$CMAKE_BIN" --build "$BUILD_DIR" --parallel
ctest --test-dir "$BUILD_DIR" --output-on-failure

printf '%s\n' "$BUILD_DIR/libemender_ndp.so"
