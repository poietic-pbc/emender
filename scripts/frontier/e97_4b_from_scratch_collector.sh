#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-4b-collector
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --no-requeue
set -euo pipefail
: "${PAYLOAD_JOB_ID:?payload job id is required}"
: "${RUN_DIR:?run directory is required}"
: "${EXPECTED_PARTITION:?expected partition is required}"
: "${EXPECTED_QOS:?expected QoS is required}"
mkdir -p "$RUN_DIR/terminal"
out="$RUN_DIR/terminal/payload-${PAYLOAD_JOB_ID}.sacct"
for _ in $(seq 1 30); do
  sacct -j "$PAYLOAD_JOB_ID" -n -P \
    --format=JobIDRaw,Partition,QOS,State,Elapsed,NNodes,ExitCode > "$out" || true
  grep -E "^${PAYLOAD_JOB_ID}\|${EXPECTED_PARTITION}\|${EXPECTED_QOS}\|" "$out" >/dev/null && break
  sleep 2
done
grep -E "^${PAYLOAD_JOB_ID}\|${EXPECTED_PARTITION}\|${EXPECTED_QOS}\|" "$out" >/dev/null || {
  echo "terminal accounting lacks exact Partition/QOS evidence" >&2; exit 66;
}
sha256sum "$out" > "$out.sha256"
latest="$RUN_DIR/train/latest.pt"
if [[ -L "$latest" && -r "$latest" ]]; then
  checkpoint=$(readlink -f "$latest")
  checkpoint_record="$RUN_DIR/terminal/checkpoint-${PAYLOAD_JOB_ID}.sha256"
  sha256sum "$checkpoint" > "$checkpoint_record.tmp"
  mv "$checkpoint_record.tmp" "$checkpoint_record"
  stat -c 'path=%n bytes=%s mtime=%Y' "$checkpoint" \
    > "$RUN_DIR/terminal/checkpoint-${PAYLOAD_JOB_ID}.stat"
fi
cat "$out"
