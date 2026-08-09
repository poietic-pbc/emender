#!/bin/bash
set -euo pipefail
ARM=${1:?usage: $0 ARM NODES}
NODES=${2:?usage: $0 ARM NODES}
[[ "$ARM" =~ ^(e97-mlp|e97-linear-mlp|gdn2-mlp)$ ]] || exit 64
[[ "$NODES" =~ ^(8|32|256)$ ]] || { echo "NODES must be 8, 32, or 256" >&2; exit 64; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
source scripts/frontier/activate_emender_frontier.sh >/dev/null
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "tracked source must be clean" >&2; exit 64; }
git fetch -q origin main
SOURCE_COMMIT=$(git rev-parse HEAD)
[[ "$SOURCE_COMMIT" == $(git rev-parse origin/main) ]] || { echo "HEAD != origin/main" >&2; exit 64; }
if squeue -u "$USER" -h -o '%q' | grep -qx debug; then
  echo "a debug-QOS job is already queued/running" >&2; exit 65
fi
RUN_BASE=${RUN_BASE:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-gdn2-paper/qualification}
if [[ "$NODES" != 8 ]]; then
  [[ "$NODES" == 32 ]] && PREVIOUS=8 || PREVIOUS=32
  predecessor=
  while IFS= read -r verdict; do
    receipts=$(dirname "$verdict")
    if [[ $(<"$verdict") == PASS \
       && -r "$receipts/source-commit.txt" \
       && $(<"$receipts/source-commit.txt") == "$SOURCE_COMMIT" ]]; then
      predecessor=$receipts; break
    fi
  done < <(find "$RUN_BASE/${PREVIOUS}node/$ARM" -path '*/receipts/verdict.txt' \
            -type f -print 2>/dev/null | sort -r)
  [[ -n "$predecessor" ]] || {
    echo "no exact-source PASS from ${PREVIOUS}-node predecessor" >&2; exit 67; }
  echo "predecessor_receipts=$predecessor"
fi
case "$NODES" in
  8) TIME_LIMIT=00:30:00 ;;
  32) TIME_LIMIT=00:40:00 ;;
  256) TIME_LIMIT=01:00:00 ;;
esac
mkdir -p logs/frontier/e97_paper
export_args="ALL,ARM=$ARM,SOURCE_COMMIT=$SOURCE_COMMIT,REPO=$ROOT,EXPECTED_NODES=$NODES,EXPECTED_QOS=debug,MAX_STEPS=160,SAVE_EVERY=80"
job=$(sbatch --parsable --nodes="$NODES" --ntasks=$((NODES * 8)) --qos=debug \
  --time="$TIME_LIMIT" --job-name="paper-${ARM}-${NODES}n" \
  --export="$export_args" scripts/frontier/e97_gdn2_paper_scale.sbatch)
echo "$job"
for _ in $(seq 1 30); do
  binding=$(squeue -h -j "$job" -o '%i|%P|%q|%T|%D|%l' || true)
  if [[ -n "$binding" ]]; then
    echo "$binding"
    [[ "$binding" == "$job|batch|debug|"*"|$NODES|$TIME_LIMIT" ]] || {
      echo "scheduler binding mismatch; cancelling $job" >&2; scancel "$job"; exit 66; }
    exit 0
  fi
  sleep 1
done
scancel "$job"; exit 66
