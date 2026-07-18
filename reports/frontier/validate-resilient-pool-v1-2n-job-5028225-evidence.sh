#!/bin/bash
set -euo pipefail

JOB_ID=5028225
RUN_ID=validate-resilient-pool-v1-2n-startup-20260718T143538Z-8735527
RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/$RUN_ID

[[ $(squeue -h -j "$JOB_ID" -o '%T') == RUNNING ]] || {
  echo "job $JOB_ID is not running; refusing a misleading node-local snapshot" >&2
  exit 69
}

# This is post-failure evidence retention, not a training data path.  Copy only
# small control-plane JSON/JSONL artifacts; mailbox, tensor, checkpoint, and
# kernel-cache payloads remain node-local and are deliberately excluded.
srun --jobid="$JOB_ID" --overlap --no-kill --exact \
  -N2 -n2 --ntasks-per-node=1 -c1 \
  env RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" bash -c '
    set -euo pipefail
    node_rank=${SLURM_NODEID:?}
    source_root=/tmp/resilient-e97/$RUN_ID/node-$node_rank
    destination=$RUN_DIR/retained-evidence/failed-job-5028225/node-$node_rank
    temporary=$destination.tmp-$SLURM_STEP_ID
    mkdir -p "$temporary"
    for tree in supervision telemetry control; do
      [[ ! -d $source_root/$tree ]] || cp -a "$source_root/$tree" "$temporary/"
    done
    find "$temporary" -type f ! -name "*.json" ! -name "*.jsonl" -delete
    [[ ! -e $destination ]] || {
      echo "refusing to replace existing retained evidence: $destination" >&2
      exit 73
    }
    mv "$temporary" "$destination"
  '
