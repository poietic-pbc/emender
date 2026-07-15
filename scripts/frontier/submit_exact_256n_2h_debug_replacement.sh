#!/usr/bin/env bash
set -euo pipefail

readonly PROD_JOB=4980157
readonly CANCELLED_JOB=5000354
readonly ATTESTED_TREE=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1023/build/e97-256/repeated-smoke-4979704/render/production/attested-tree-9fff689c9f9252b6a264773c207f8f8ca8509666-jj6etgng
readonly ATTEMPT="$PWD/build/e97-256/exact-debug-256n-2h-20260715"
readonly WRAPPER="$ATTEMPT/durable-debug-wrapper.sbatch"
readonly SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt

[[ ${1:-} == --submit ]] || { echo "preflight only; pass --submit for the one authorized replacement"; submit=0; }
[[ ${1:-} == --submit ]] && submit=1
test ! -e "$ATTEMPT/replacement-job-id.txt" || { echo "refusing duplicate replacement" >&2; exit 70; }
cancelled_state=$(sacct -j "$CANCELLED_JOB" -X -n -o State | xargs | cut -d' ' -f1)
[[ "$cancelled_state" == CANCELLED* ]] || { echo "job $CANCELLED_JOB is not cancelled: $cancelled_state" >&2; exit 71; }
scontrol show job -dd "$PROD_JOB" > "$ATTEMPT/replacement-production-scontrol.txt"
prod=$(sed -n 's/^   SubmitLine=//p' "$ATTEMPT/replacement-production-scontrol.txt")
test -n "$prod"
debug=${prod/ -t 12:00:00 / -t 02:00:00 }
debug=${debug/ -q normal / -q debug }
debug=${debug% scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch}
debug="$debug $WRAPPER"
printf '%s\n' "$debug" > "$ATTEMPT/replacement-debug-command.txt"

PROD="$prod" DEBUG="$debug" WRAPPER_PATH="$WRAPPER" python3 - "$ATTEMPT/replacement-structured-diff.json" <<'PY'
import json, os, shlex, sys
def parse(value):
    prefix, rest = value.split(" --export ", 1)
    export, script = rest.rsplit(" ", 1)
    return shlex.split(prefix) + ["--export", export, script]
p, d = parse(os.environ["PROD"]), parse(os.environ["DEBUG"])
assert len(p) == len(d), (len(p), len(d))
diff = [{"index": i, "production": a, "debug": b} for i,(a,b) in enumerate(zip(p,d)) if a != b]
expected = {
    ("12:00:00", "02:00:00"),
    ("normal", "debug"),
    ("scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch", os.environ["WRAPPER_PATH"]),
}
assert {(x["production"], x["debug"]) for x in diff} == expected, diff
exports = os.environ["PROD"]
for value in ("E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt", "ASYNC_TRAINPY_RANKS=2048", "ASYNC_EXPECTED_RANKS=2048", "ASYNC_GLOBAL_QUORUM=2048", "ASYNC_TIMEOUT_S=1200", "DILOCO_K=40", "ASYNC_LOCAL_STEPS=40", "ASYNC_GENERATIONS=1000000", "ASYNC_STEPS=40000000"):
    assert value in exports, value
json.dump({"production_argv": p, "replacement_argv": d, "allowed_differences": diff,
           "wrapper_change_scope": "scheduler-exit immutable continuation handoff only"},
          open(sys.argv[1], "w"), indent=2, sort_keys=True)
PY

sha256sum "$WRAPPER" > "$ATTEMPT/durable-wrapper.sha256"
printf '106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30  %s\n' "$ATTESTED_TREE/scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch" | sha256sum -c - > "$ATTEMPT/replacement-launcher-hash-check.txt"
printf '1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9  %s\n' "$SEED" | sha256sum -c - > "$ATTEMPT/replacement-seed-hash-check.txt"
bash -n "$WRAPPER"
[[ $submit == 1 ]] || exit 0
: > "$ATTEMPT/replacement-submission-attempted.marker"
python3 - "$ATTEMPT/replacement-structured-diff.json" "$ATTESTED_TREE" <<'PY' | tee "$ATTEMPT/replacement-submission-output.txt"
import json, subprocess, sys
argv=json.load(open(sys.argv[1]))["replacement_argv"]
print(subprocess.check_output(argv, cwd=sys.argv[2], universal_newlines=True).strip())
PY
job_id=$(tr -dc '0-9' < "$ATTEMPT/replacement-submission-output.txt")
test -n "$job_id"
printf '%s\n' "$job_id" > "$ATTEMPT/replacement-job-id.txt"
scontrol show job -dd "$job_id" > "$ATTEMPT/replacement-scontrol-initial.txt"
squeue --start -j "$job_id" -o '%i|%T|%S|%R' > "$ATTEMPT/replacement-start-estimate.txt"
echo "$job_id"
