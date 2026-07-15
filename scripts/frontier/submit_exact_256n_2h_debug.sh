#!/usr/bin/env bash
set -euo pipefail

readonly PROD_JOB=4980157
readonly ATTESTED_TREE=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1023/build/e97-256/repeated-smoke-4979704/render/production/attested-tree-9fff689c9f9252b6a264773c207f8f8ca8509666-jj6etgng
readonly HASH_MANIFEST="$PWD/build/e97-256/repeated-smoke-4979704/repaired-source-hashes.sha256"
readonly SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt
readonly ATTEMPT="$PWD/build/e97-256/exact-debug-256n-2h-20260715"

mkdir -p "$ATTEMPT"
test ! -e "$ATTEMPT/job-id.txt" || { echo "refusing duplicate submission" >&2; exit 70; }
scontrol show job -dd "$PROD_JOB" > "$ATTEMPT/production-scontrol-preflight.txt"
prod=$(sed -n 's/^   SubmitLine=//p' "$ATTEMPT/production-scontrol-preflight.txt")
test -n "$prod"
debug=${prod/ -t 12:00:00 / -t 02:00:00 }
debug=${debug/ -q normal / -q debug }
printf '%s\n' "$prod" > "$ATTEMPT/production-command.txt"
printf '%s\n' "$debug" > "$ATTEMPT/debug-command.txt"

PROD="$prod" DEBUG="$debug" python3 - "$ATTEMPT" <<'PY'
import json, os, shlex, sys
def parse_submitline(value):
    prefix, rest = value.split(' --export ', 1)
    export, script = rest.rsplit(' ', 1)
    return shlex.split(prefix) + ['--export', export, script]
p, d = map(parse_submitline, (os.environ['PROD'], os.environ['DEBUG']))
diff = [{'index': i, 'production': a, 'debug': b} for i, (a, b) in enumerate(zip(p, d)) if a != b]
assert len(p) == len(d), (len(p), len(d))
assert diff == [
    {'index': p.index('12:00:00'), 'production': '12:00:00', 'debug': '02:00:00'},
    {'index': p.index('normal'), 'production': 'normal', 'debug': 'debug'},
], diff
assert p[p.index('-N') + 1] == '256'
exports = os.environ['PROD']
for required in ('ASYNC_TRAINPY_RANKS=2048', 'ASYNC_EXPECTED_RANKS=2048',
                 'ASYNC_GLOBAL_QUORUM=2048', 'ASYNC_TIMEOUT_S=1200',
                 'DILOCO_K=40', 'ASYNC_LOCAL_STEPS=40',
                 'ASYNC_GENERATIONS=1000000', 'ASYNC_STEPS=40000000'):
    assert required in exports, required
with open(os.path.join(sys.argv[1], 'structured-command-diff.json'), 'w') as f:
    json.dump({'production_argv': p, 'debug_argv': d, 'allowed_differences': diff}, f, indent=2, sort_keys=True)
    f.write('\n')
PY

(cd "$ATTESTED_TREE" && sha256sum -c "$HASH_MANIFEST") > "$ATTEMPT/source-hash-check.txt"
printf '106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30  %s\n' \
  "$ATTESTED_TREE/scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch" | sha256sum -c - > "$ATTEMPT/launcher-hash-check.txt"
printf '1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9  %s\n' "$SEED" | sha256sum -c - > "$ATTEMPT/seed-hash-check.txt"

printf 'approved=true\nproduction_job=%s\nsemantic_allowlist=walltime,qos\n' "$PROD_JOB" > "$ATTEMPT/preflight-approval.txt"
: > "$ATTEMPT/submission-attempted.marker"
python3 - "$ATTEMPT/structured-command-diff.json" "$ATTESTED_TREE" <<'PY' | tee "$ATTEMPT/submission-output.txt"
import json, subprocess, sys
with open(sys.argv[1]) as f:
    argv = json.load(f)['debug_argv']
print(subprocess.check_output(argv, cwd=sys.argv[2]).decode().strip())
PY
job_id=$(tr -dc '0-9' < "$ATTEMPT/submission-output.txt")
test -n "$job_id"
printf '%s\n' "$job_id" > "$ATTEMPT/job-id.txt"
scontrol show job -dd "$job_id" > "$ATTEMPT/scontrol-initial.txt"
squeue -j "$job_id" -o '%i|%j|%a|%q|%l|%D|%T|%M|%S|%R' > "$ATTEMPT/squeue-initial.txt"
sacct -j "$job_id" -X -n -P -o JobIDRaw,JobName,Account,QOS,Partition,Timelimit,NNodes,NTasks,State,ExitCode,Submit,Start,End > "$ATTEMPT/sacct-initial.txt"
echo "$job_id"
