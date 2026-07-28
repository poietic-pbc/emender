#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
trace_file=$(mktemp "${TMPDIR:-/tmp}/emender-resilient-trace.XXXXXX.json")
cleanup_trace() {
  rm -f -- "$trace_file"
}
trap cleanup_trace EXIT

printf '%s  %s\n' \
  cf654525395e63b31b2d76e8109ee2bcc6a652f6273d1c6e4ca5bec9ecb776b4 \
  "$package_root/trace-schema-v1.json" | sha256sum --check -

if rg -n '\b(sorry|admit|native_decide)\b|^\s*(unsafe\s+)?axiom\b|^\s*opaque\b' \
    "$package_root/ResilientProtocol.lean" \
    "$package_root/ResilientProtocol" \
    "$package_root/TraceMain.lean" \
    "$package_root/ExamplesMain.lean" \
    "$package_root/TestsMain.lean"; then
  echo "resilient protocol contains a forbidden trust escape" >&2
  exit 1
fi

"$package_root/scripts/lake.sh" build
"$package_root/scripts/lake.sh" test
"$package_root/scripts/lake.sh" exe resilient-examples
"$package_root/scripts/lake.sh" exe resilient-examples \
  --trace job-5105811-generation-closed-restart-rejoin > "$trace_file"
"$package_root/scripts/lake.sh" exe resilient-trace replay "$trace_file"
"$package_root/scripts/lake.sh" exe resilient-conformance-corpus

for corpus_trace in "$package_root"/corpus/native-v1/native-*.json; do
  scenario=$(basename "$corpus_trace" .json)
  "$package_root/scripts/lake.sh" exe resilient-conformance-corpus \
    --trace "$scenario" >"$trace_file"
  if ! cmp -s "$trace_file" "$corpus_trace"; then
    echo "checked native conformance trace is stale: $scenario" >&2
    exit 1
  fi
  "$package_root/scripts/lake.sh" exe resilient-conformance "$corpus_trace" \
    >/dev/null
done
