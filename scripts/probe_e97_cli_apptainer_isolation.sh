#!/bin/bash
set -euo pipefail
: "${SANDBOX:?}"
: "${HOST_SENTINEL:?}"
: "${IMAGE:?}"
: "${RANK:?}"
mkdir -p "$SANDBOX"
printf 'allowed-%s\n' "$RANK" > "$SANDBOX/allowed.txt"
printf 'forbidden-%s\n' "$RANK" > "$HOST_SENTINEL"
export EMENDER_HOST_SECRET=must-not-cross
output=$(timeout --signal=TERM --kill-after=5 60 apptainer exec \
  --containall --cleanenv --net --network none --no-privs --drop-caps all \
  --no-mount bind-paths,home,cwd,tmp,hostfs,proc,sys \
  --bind "$SANDBOX:/work:rw" --cwd /work "$IMAGE" \
  /bin/sh -s -- "$RANK" <<'INNER'
set -eu
rank=$1
test "$PWD" = /work
grep -q "allowed-${rank}" allowed.txt
test ! -e /lustre && test ! -e /autofs && test ! -e /ccs
test ! -e /proc/self/status && test ! -e /sys/kernel
test -z "${EMENDER_HOST_SECRET:-}"
! wget -T 3 -qO- https://example.com >/dev/null 2>&1
! sh -c 'echo escape > /outside.txt' 2>/dev/null
printf 'written-%s\n' "$rank" > result.txt
printf 'pass\n'
INNER
)
test "$output" = pass
grep -q "written-${RANK}" "$SANDBOX/result.txt"
grep -q "forbidden-${RANK}" "$HOST_SENTINEL"
