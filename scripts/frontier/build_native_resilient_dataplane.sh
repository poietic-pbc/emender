#!/bin/bash
set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
SOURCE_DIR=${SOURCE_DIR:-$REPO/native}
BUILD_DIR=${BUILD_DIR:-$REPO/build/native-resilient-dataplane-build}
INSTALL_DIR=${INSTALL_DIR:-$REPO/build/native-resilient-dataplane}
BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  # shellcheck source=activate_emender_frontier.sh
  source "$REPO/scripts/frontier/activate_emender_frontier.sh"
fi
: "${PYTHON_BIN:?canonical Frontier activation did not select PYTHON_BIN}"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DNDP_ENABLE_XPMEM="${NDP_ENABLE_XPMEM:-ON}" \
  -DNDP_BUILD_TESTS=ON
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-8}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
cmake --install "$BUILD_DIR"
"$PYTHON_BIN" "$REPO/scripts/frontier/attest_native_dataplane.py" record-build \
  --prefix "$INSTALL_DIR" --source-root "$REPO" \
  --cmake-cache "$BUILD_DIR/CMakeCache.txt"
