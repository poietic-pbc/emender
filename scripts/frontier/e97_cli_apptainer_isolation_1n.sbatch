#!/bin/bash
#SBATCH -J e97-cli-isolate
#SBATCH -A bif148
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 00:20:00
#SBATCH --no-requeue
#SBATCH -o logs/frontier/e97_dense_agent/%x-%j.out
#SBATCH -e logs/frontier/e97_dense_agent/%x-%j.err
set -euo pipefail

: "${SOURCE_COMMIT:?}"
: "${RUN_ROOT:?}"
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
IMAGE=${IMAGE:-/lustre/orion/bif148/proj-shared/emender/agent_sandbox/images/alpine-3.20.sif}
IMAGE_SHA256=${IMAGE_SHA256:-ff23e743832f57c91c9c9c0f3e1296cde5d8c8a9299f9a751247e63a318a00f0}

cd "$REPO"
git cat-file -e "${SOURCE_COMMIT}^{commit}"
record=$(scontrol show job "$SLURM_JOB_ID" -o)
[[ "$RUN_ROOT" == /* && "$record" == *"NumNodes=1"* && "$record" == *"NumTasks=8"* && "$record" == *"Partition=batch"* && "$record" == *"QOS=debug"* && "$record" == *"Requeue=0"* ]] || exit 64
mkdir -p "$RUN_ROOT"/{identity,results,sandboxes,sources,terminal} logs/frontier/e97_dense_agent
printf '%s\n' "$record" > "$RUN_ROOT/identity/scontrol-$SLURM_JOB_ID.txt"
squeue -j "$SLURM_JOB_ID" -h -o 'JobID=%i|Partition=%P|QOS=%q|State=%T|NodeList=%N' > "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
grep -q 'Partition=batch|QOS=debug' "$RUN_ROOT/identity/squeue-$SLURM_JOB_ID.txt"
printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE" | sha256sum -c -
SNAPSHOT="$RUN_ROOT/sources/$SOURCE_COMMIT"
mkdir -p "$SNAPSHOT"
git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"

set +e
srun --label --kill-on-bad-exit=1 --wait=120 bash -lc '
  set -euo pipefail
  rank=$SLURM_PROCID
  SANDBOX="'$RUN_ROOT'/sandboxes/rank-${rank}" \
  HOST_SENTINEL="'$RUN_ROOT'/sandboxes/host-secret-${rank}.txt" \
  IMAGE="'$IMAGE'" RANK="$rank" \
    "'$SNAPSHOT'/scripts/probe_e97_cli_apptainer_isolation.sh"
  printf '\''{"rank":%s,"status":"pass","host_paths_hidden":true,"network_isolated":true,"root_read_only":true,"proc_sys_hidden":true,"clean_environment":true}\n'\'' "$rank" > "'$RUN_ROOT'/results/rank-${rank}.json"
'
rc=$?
set -e

if (( rc == 0 )); then
  python3 - "$RUN_ROOT" <<'PY' || rc=$?
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); rows=[json.loads(p.read_text()) for p in sorted((root/'results').glob('rank-*.json'))]
if len(rows)!=8 or not all(row['status']=='pass' for row in rows): raise SystemExit(1)
summary={'schema':'emender-e97-cli-apptainer-isolation-v1','status':'pass','ranks':8,'checks':{key:all(row[key] for row in rows) for key in ('host_paths_hidden','network_isolated','root_read_only','proc_sys_hidden','clean_environment')}}
(root/'results/summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,sort_keys=True))
PY
fi
sacct -j "$SLURM_JOB_ID" --format=JobIDRaw,Partition,QOS,State,Elapsed,ExitCode -P > "$RUN_ROOT/terminal/sacct-$SLURM_JOB_ID.txt" || true
printf '%s\n' "$rc" > "$RUN_ROOT/terminal/job-$SLURM_JOB_ID-rc.txt"
exit "$rc"
