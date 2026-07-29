#!/usr/bin/env bash
set -euo pipefail

repository_root=$(git rev-parse --show-toplevel)
# shellcheck source=../frontier/activate_emender_frontier.sh
source "$repository_root/scripts/frontier/activate_emender_frontier.sh"
: "${EMENDER_PYTHON:?canonical Frontier Python was not selected}"

"$repository_root/formal/resilient/scripts/lake.sh" build \
  resilient-conformance resilient-conformance-corpus
"$EMENDER_PYTHON" \
  "$repository_root/scripts/conformance/generate_native_lean_corpus.py"
PYTHON_BIN="$EMENDER_PYTHON" \
  "$repository_root/scripts/frontier/build_native_resilient_dataplane.sh"
EMENDER_NDP_BUILD_MANIFEST=\
"$repository_root/build/native-resilient-dataplane/native-artifacts.json" \
EMENDER_LEAN_CONFORMANCE_RUNNER=\
"$repository_root/formal/resilient/.lake/build/bin/resilient-conformance" \
  "$EMENDER_PYTHON" -m pytest -q \
  "$repository_root/tests/test_native_lean_conformance.py"
"$EMENDER_PYTHON" \
  "$repository_root/scripts/conformance/generate_native_lean_manifest.py" \
  --build-manifest \
    "$repository_root/build/native-resilient-dataplane/native-artifacts.json" \
  --lean-runner \
    "$repository_root/formal/resilient/.lake/build/bin/resilient-conformance"
