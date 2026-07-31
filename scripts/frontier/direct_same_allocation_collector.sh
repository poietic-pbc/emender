#!/bin/bash
#SBATCH -A bif148
#SBATCH -J collect-direct-samealloc
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o /lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/collectors/%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/emender/frontier_runs/direct-same-allocation/collectors/%j.err
set -euo pipefail
: "${PAYLOAD_JOB_ID:?}" "${DIRECT_RESTART_ROOT:?}" "${SUBMISSION_RECORD:?}"
ROOT=$DIRECT_RESTART_ROOT
COLLECTOR_ROOT="$(dirname "$ROOT")/collectors/${SLURM_JOB_ID}/payload-${PAYLOAD_JOB_ID}"
mkdir -p "$COLLECTOR_ROOT"
cp --reflink=auto "$SUBMISSION_RECORD" "$COLLECTOR_ROOT/submission.json"
sacct -X -j "$PAYLOAD_JOB_ID,$SLURM_JOB_ID" -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed \
  > "$COLLECTOR_ROOT/sacct.txt"
scontrol show job -dd "$PAYLOAD_JOB_ID" > "$COLLECTOR_ROOT/scontrol-payload.txt" || true
find "$ROOT" -type f -o -type l | sort > "$COLLECTOR_ROOT/artifact-paths.txt" || true
if [[ -r "$ROOT/verdict.json" ]]; then
  cp --reflink=auto "$ROOT/verdict.json" "$COLLECTOR_ROOT/verdict.json"
else
  printf '%s\n' '{"allocation_survived":false,"failed_step_bounded":false,"fresh_srun_launched":false,"world_size_changed":false,"checkpoint_reloaded":false,"post_relaunch_merge_passed":false,"full_pass":false,"reason":"payload verdict missing"}' > "$COLLECTOR_ROOT/verdict.json"
fi
sha256sum "$COLLECTOR_ROOT"/* > "$COLLECTOR_ROOT/SHA256SUMS"
