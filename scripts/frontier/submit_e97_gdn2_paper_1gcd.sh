#!/bin/bash
set -euo pipefail
ARM=${1:?usage: $0 e97-mlp|e97-linear-mlp|gdn2-mlp}
[[ "$ARM" =~ ^(e97-mlp|e97-linear-mlp|gdn2-mlp)$ ]] || { echo "invalid arm" >&2; exit 64; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
source scripts/frontier/activate_emender_frontier.sh >/dev/null
[[ -z $(git status --porcelain --untracked-files=no) ]] || {
  echo "tracked source must be clean" >&2; exit 64; }
git fetch -q origin main
SOURCE_COMMIT=$(git rev-parse HEAD)
[[ "$SOURCE_COMMIT" == $(git rev-parse origin/main) ]] || {
  echo "submission requires HEAD == origin/main" >&2; exit 64; }
if squeue -u "$USER" -h -o '%q' | grep -qx debug; then
  echo "a debug-QOS job is already queued/running; serialize qualification" >&2
  exit 65
fi
mkdir -p logs/frontier/e97_paper
job=$(sbatch --parsable --export="ALL,ARM=$ARM,SOURCE_COMMIT=$SOURCE_COMMIT,REPO=$ROOT" \
  --job-name="paper-${ARM}-1g" scripts/frontier/e97_gdn2_paper_1gcd.sbatch)
echo "$job"
for _ in $(seq 1 30); do
  binding=$(squeue -h -j "$job" -o '%i|%P|%q|%T|%D|%l' || true)
  if [[ -n "$binding" ]]; then
    echo "$binding"
    [[ "$binding" == "$job|batch|debug|"*"|1|"* ]] || {
      echo "scheduler binding mismatch; cancelling $job" >&2; scancel "$job"; exit 66; }
    exit 0
  fi
  sleep 1
done
echo "could not verify scheduler binding; cancelling $job" >&2
scancel "$job"
exit 66
