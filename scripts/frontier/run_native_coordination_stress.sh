#!/bin/bash
set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
BUILD_DIR=${BUILD_DIR:-$REPO/build/native-coordination-stress}
OUTPUT=${OUTPUT:-$REPO/reports/frontier/native-coordination-stress-v1.json}
RANDOM_SCHEDULES=${RANDOM_SCHEDULES:-50000}
MAXIMUM_EVENTS=${MAXIMUM_EVENTS:-32}
DETERMINISM_REPEATS=${DETERMINISM_REPEATS:-2}
BUILD_JOBS=${BUILD_JOBS:-4}

# The project activation is authoritative even though this model-free local
# gate does not invoke Python, a GPU, libfabric, Frontier, or Slurm.
# shellcheck source=activate_emender_frontier.sh
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
: "${EMENDER_PYTHON:?canonical Frontier activation did not select Python}"

cmake -S "$REPO/src/native_resilient_dataplane" -B "$BUILD_DIR" \
  -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD_DIR" \
  --target ndp_coordination_schedule_stress --parallel "$BUILD_JOBS"

SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD)
"$BUILD_DIR/ndp_coordination_schedule_stress" \
  --source-root "$REPO" \
  --corpus-dir "$REPO/tests/corpus/native_coordination" \
  --failure-dir "$REPO/tests/corpus/native_coordination" \
  --source-commit "$SOURCE_COMMIT" \
  --random-schedules "$RANDOM_SCHEDULES" \
  --maximum-events "$MAXIMUM_EVENTS" \
  --determinism-repeats "$DETERMINISM_REPEATS" \
  --output "$OUTPUT"
