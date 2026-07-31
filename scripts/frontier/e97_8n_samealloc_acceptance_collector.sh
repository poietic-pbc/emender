#!/bin/bash
#SBATCH -A bif148
#SBATCH -J collect-e97-8n-samealloc
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:10:00
set -euo pipefail
: "${PAYLOAD_JOB_ID:?}" "${RUN_DIR:?}" "${PAYLOAD_ROOT:?}" "${REPO:?}" "${SUBMISSION_RECORD:?}"
COLLECTOR_ROOT=${COLLECTOR_ROOT:-$(dirname "$(dirname "$RUN_DIR")")/collectors/${SLURM_JOB_ID}/payload-${PAYLOAD_JOB_ID}}
mkdir -p "$COLLECTOR_ROOT" "$RUN_DIR/terminal"
cp --reflink=auto "$SUBMISSION_RECORD" "$COLLECTOR_ROOT/submission.json"
sacct -j "$PAYLOAD_JOB_ID,$SLURM_JOB_ID" -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed \
  > "$COLLECTOR_ROOT/sacct.txt"
cp "$COLLECTOR_ROOT/sacct.txt" "$RUN_DIR/terminal/sacct.txt"
scontrol show job -dd "$PAYLOAD_JOB_ID" > "$COLLECTOR_ROOT/scontrol-payload.txt" || true
find "$RUN_DIR" -type f -o -type l | sort > "$COLLECTOR_ROOT/artifact-paths.txt" || true

export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
"$EMENDER_PYTHON" - "$RUN_DIR" "$COLLECTOR_ROOT" "$PAYLOAD_JOB_ID" <<'PY'
import json, math, os, pathlib, re, sys, time
run = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); job = sys.argv[3]
state = run / 'supervisor' / 'acceptance'
reasons = []
def need(ok, message):
    if not ok: reasons.append(message)
def text(path):
    try: return pathlib.Path(path).read_text(errors='replace')
    except OSError: return ''
def integer(path):
    try: return int(text(path).strip())
    except ValueError: return 0

held = text(run/'identity'/'squeue-held.txt')
live = text(state/'squeue-live.txt')
after = text(state/'squeue-allocation-after-fault.txt')
sacct = text(out/'sacct.txt')
need(re.search(rf'^{re.escape(job)}\|[^|]+\|8\|[^|]*\|batch\|normal\|', held, re.M) is not None,
     'held squeue lacks Nodes=8 Partition=batch QOS=normal')
need(re.search(rf'^{re.escape(job)}\|RUNNING\|8\|[^|]*\|batch\|normal\|', live, re.M) is not None,
     'live squeue lacks RUNNING Nodes=8 Partition=batch QOS=normal')
need(re.search(rf'^{re.escape(job)}\|RUNNING\|8\|[^|]*\|batch\|normal\|', after, re.M) is not None,
     'allocation did not remain live after fault')
# JobIDRaw|JobName|State|ExitCode|DerivedExitCode|NNodes|NTasks|NodeList|Partition|QOS|...
top = None
step_rows = []
for line in sacct.splitlines()[1:]:
    p = line.split('|')
    if len(p) < 10: continue
    if p[0] == job: top = p
    elif p[0].startswith(job + '.'): step_rows.append(p)
need(top is not None, 'terminal payload accounting missing')
if top:
    need(top[2] == 'COMPLETED' and top[3] == '0:0', 'payload did not complete 0:0')
    need(top[5] == '8' and top[8] == 'batch' and top[9] == 'normal',
         'terminal accounting lacks NNodes=8 Partition=batch QOS=normal')

rows=[]
for line in text(run/'supervisor'/'execution-epochs.tsv').splitlines():
    p=line.split('|')
    if len(p)==6:
        rows.append({'epoch':int(p[0]),'rc':int(p[1]),'nodes':int(p[2]),
                     'tasks':int(p[3]),'port':int(p[4]),'promoted':int(p[5])})
need(len(rows)==2, f'expected exactly two execution epochs, found {len(rows)}')
if len(rows)==2:
    need(rows[0]['rc'] != 0 and rows[1]['rc'] == 0, 'damaged/relaunch return codes invalid')
    need((rows[0]['nodes'],rows[0]['tasks'])==(8,64), 'baseline/fault world is not 8 nodes/64 ranks')
    need((rows[1]['nodes'],rows[1]['tasks'])==(7,56), 'relaunch world is not 7 nodes/56 ranks')
    need(rows[0]['port'] != rows[1]['port'], 'MASTER_PORT was reused')
    need(rows[0]['promoted']==1 and rows[1]['promoted']==1, 'committed latest was not promoted')

first = text(run/'epochs'/'epoch-000001'/'train.out')
second = text(run/'epochs'/'epoch-000002'/'train.out')
markers = re.findall(r'DILOCO_FAULT_INJECTION (\{[^\n]+\})', first)
need(len(markers)==1, f'expected exactly one injection marker, found {len(markers)}')
injection={}
if len(markers)==1:
    try: injection=json.loads(markers[0])
    except json.JSONDecodeError: reasons.append('injection marker is invalid JSON')
    need(injection.get('rank')==1 and injection.get('merge_index')==6 and
         injection.get('bucket_index')==1 and injection.get('label') in ('sf_x','sf_z'),
         'injection identity is not rank 1, merge 6, later sf_x/sf_z bucket 1')
marker_pos=first.find('DILOCO_FAULT_INJECTION')
prefix=first[:marker_pos] if marker_pos >= 0 else first
baseline_merges=len(re.findall(r'>>> \[DiLoCo\] merge #\d+ at step', prefix))
need(baseline_merges==5, f'expected five completed pre-fault K40 merges, found {baseline_merges}')
need('merge #5 at step 1065200' in prefix, 'fifth K40 merge at step 1065200 missing')
need('saved checkpoint: checkpoint_step_1065200_' in prefix, 'periodic step-1065200 checkpoint missing')
need('diloco_k=40' in text(run/'epochs'/'epoch-000001'/'launch.env'), 'launcher did not bind K40')
need('save_every=200' in text(run/'epochs'/'epoch-000001'/'launch.env'), 'launcher did not bind save_every=200')
need('diloco_merge_bucket_numel=67108864' in text(run/'epochs'/'epoch-000001'/'launch.env'), '64M bucket binding missing')
need('topology=hierarchical' in first or 'hierarchical' in first, 'hierarchical merge evidence missing')

cp_start=integer(state/'checkpoint-start-epoch-ns.txt')
cp_end=integer(state/'checkpoint-published-epoch-ns.txt')
checkpoint_duration=(cp_end-cp_start)/1e9 if cp_start and cp_end and cp_end>=cp_start else math.nan
need(math.isfinite(checkpoint_duration) and checkpoint_duration>0, 'checkpoint duration observation missing')
baseline_ckpt=text(state/'baseline-checkpoint.txt').strip()
latest_after=text(state/'latest-after-fault.txt').strip()
need('checkpoint_step_1065200_' in baseline_ckpt and latest_after==baseline_ckpt,
     'damaged step changed latest away from committed step 1065200')
snapshot=text(state/'train-files-after-fault.tsv')
need('.tmp|' not in snapshot, 'temporary checkpoint remained after damaged step')
steps_after=[int(x) for x in re.findall(r'checkpoint_step_(\d+)_loss_', snapshot)]
need(bool(steps_after) and max(steps_after)==1065200, 'damaged step published post-1065200 checkpoint state')
checkpoint_size=0
for line in snapshot.splitlines():
    if line.startswith('checkpoint_step_1065200_'):
        try: checkpoint_size=int(line.split('|')[1])
        except (IndexError, ValueError): pass
need(7_000_000_000 <= checkpoint_size <= 9_000_000_000, f'checkpoint size is not ~7.7GB: {checkpoint_size}')
need(bool(text(state/'baseline-checkpoint.sha256').strip()), 'baseline checkpoint SHA-256 missing')

fault_at=integer(state/'fault-injection-epoch-ns.txt')
fault_exit=integer(state/'fault-step-exit-epoch-ns.txt')
failure_exit_s=(fault_exit-fault_at)/1e9 if fault_at and fault_exit>=fault_at else math.nan
need(math.isfinite(failure_exit_s) and 0 < failure_exit_s <= 180, f'damaged step exit not bounded: {failure_exit_s}')
need('physical_node_crash=false' in text(state/'deliberate-node-exclusion.txt'), 'physical-node non-crash evidence missing')
need('communicator_shrink=false' in text(state/'deliberate-node-exclusion.txt'), 'no-shrink evidence missing')
need('fault_environment_removed=true' in text(state/'relaunch-environment.txt'), 'one-shot injection was not removed')

need('Resumed at step 1065200' in second, 'fresh child did not reload committed step 1065200')
post_merges=len(re.findall(r'>>> \[DiLoCo\] merge #\d+ at step', second))
need(post_merges>=2, f'fresh 56-rank child completed only {post_merges} merges')
metrics=[]
for m in re.finditer(r'step\s+(\d+) \| loss ([^ ]+) .*?global_tok/s ([^ ]+)', second):
    try: metrics.append((int(m.group(1)),float(m.group(2)),float(m.group(3))))
    except ValueError: pass
finite=[m for m in metrics if math.isfinite(m[1]) and math.isfinite(m[2]) and m[2]>0]
need(bool(finite), 'post-restart finite loss/throughput missing')
need('DILOCO_FAULT_INJECTION' not in second, 'fault injection replayed in fresh child')
need(len(step_rows)>=2, 'child step accounting missing')

relaunch_ns=integer(state/'relaunch-start-epoch-ns.txt')
recovery_downtime=(relaunch_ns-fault_exit)/1e9 if relaunch_ns and fault_exit and relaunch_ns>=fault_exit else math.nan
fields={
 'schema':'e97-same-allocation-8n-acceptance-v1', 'job_id':job,
 'source_sha':text(run/'identity'/f'git-commit-{job}.txt').strip(),
 'payload_digest':json.load(open(run/'identity'/'submission.json'))['payload_digest'],
 'nodes_requested':8, 'partition':'batch', 'qos':'normal',
 'baseline_world_size':64, 'relaunch_world_size':56,
 'baseline_k40_merges':baseline_merges, 'post_relaunch_merges':post_merges,
 'checkpoint_path_at_fault':baseline_ckpt, 'checkpoint_size_bytes':checkpoint_size,
 'checkpoint_duration_seconds':checkpoint_duration,
 'fault_injection':injection, 'failed_step_exit_seconds':failure_exit_s,
 'recovery_downtime_seconds':recovery_downtime,
 'post_restart_last_step':finite[-1][0] if finite else None,
 'post_restart_last_loss':finite[-1][1] if finite else None,
 'post_restart_last_global_tokens_per_second':finite[-1][2] if finite else None,
 'allocation_survived':not any('allocation did not remain' in r for r in reasons),
 'failed_step_bounded':math.isfinite(failure_exit_s) and 0 < failure_exit_s <= 180,
 'fresh_srun_launched':len(rows)==2 and len(step_rows)>=2,
 'world_size_changed':len(rows)==2 and rows[0]['tasks']==64 and rows[1]['tasks']==56,
 'checkpoint_reloaded':'Resumed at step 1065200' in second,
 'post_relaunch_merge_passed':post_merges>=2 and bool(finite),
 'no_failed_state_published':bool(steps_after) and max(steps_after)==1065200 and latest_after==baseline_ckpt,
 'unchanged_failed_payload_retried':False,
 'requirements':['R07','R12','R14','R16','NDP02-retired','NDP13','NDP15-atomic-only','NDP17-retired'],
 'errors':reasons,
}
fields['full_pass']=not reasons and all(fields[k] for k in (
 'allocation_survived','failed_step_bounded','fresh_srun_launched','world_size_changed',
 'checkpoint_reloaded','post_relaunch_merge_passed','no_failed_state_published'))
for path in (run/'verdict.json', out/'verdict.json'):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(fields,sort_keys=True,indent=2)+'\n'); os.replace(tmp,path)
print(json.dumps(fields,sort_keys=True))
if not fields['full_pass']: raise SystemExit(1)
PY
rc=$?
sha256sum "$COLLECTOR_ROOT"/* > "$COLLECTOR_ROOT/SHA256SUMS" || true
exit "$rc"
