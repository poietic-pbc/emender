#!/bin/bash
#SBATCH -A bif148
#SBATCH -J collect-e97-32n-clean
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:12:00
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
find "$RUN_DIR" \( -type f -o -type l \) -print | sort > "$COLLECTOR_ROOT/artifact-paths.txt" || true

export FRONTIER_RCCL_ENV=recommended FRONTIER_ENABLE_OLCF_RCCL_PLUGIN=1
source "$REPO/scripts/frontier/activate_emender_frontier.sh"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
"$EMENDER_PYTHON" - "$RUN_DIR" "$COLLECTOR_ROOT" "$PAYLOAD_JOB_ID" "$PAYLOAD_ROOT" <<'PY'
import datetime as dt
import json, math, os, pathlib, re, statistics, sys
import torch

run=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); job=sys.argv[3]; payload=pathlib.Path(sys.argv[4])
reasons=[]
def need(ok,message):
    if not ok: reasons.append(message)
def text(path):
    try: return pathlib.Path(path).read_text(errors='replace')
    except OSError: return ''
def load_json(path):
    try: return json.loads(text(path))
    except (json.JSONDecodeError, TypeError): return {}

submission=load_json(run/'identity'/'submission.json')
config={}
for line in text(run/'identity'/'config.env').splitlines():
    if '=' in line:
        k,v=line.split('=',1); config[k]=v
held=text(run/'identity'/'squeue-held.txt')
live=text(run/'identity'/'squeue-live.txt')
sacct=text(out/'sacct.txt')
need(re.search(rf'^{re.escape(job)}\|[^|]+\|32\|[^|]*\|batch\|normal\|',held,re.M) is not None,
     'held squeue lacks Nodes=32 Partition=batch QOS=normal')
need(re.search(rf'^{re.escape(job)}\|RUNNING\|32\|[^|]*\|batch\|normal\|',live,re.M) is not None,
     'live squeue lacks RUNNING Nodes=32 Partition=batch QOS=normal')
need(submission.get('collector_registered_before_release') is True and submission.get('released') is True,
     'durable afterany collector was not recorded before release')
need(submission.get('dependency') == 'afterany:'+job, 'collector dependency is not scheduler-owned afterany')
need(submission.get('source_sha') == 'ac0c90a91c4c8e68265e573cea9cb808e00987ac', 'source identity mismatch')
need(submission.get('predecessor_job') == '5126609', 'immediate 8-node predecessor mismatch')
need(config.get('NODES')=='32' and config.get('WORLD_SIZE')=='256', 'rendered 32-node/256-rank config mismatch')
need(config.get('PARTITION')=='batch' and config.get('QOS')=='normal', 'rendered queue binding mismatch')
need(config.get('DILOCO_K')=='40' and config.get('SAVE_EVERY')=='200' and config.get('KEEP_CHECKPOINTS')=='2',
     'K40/save-200/keep-2 binding mismatch')
need(config.get('DILOCO_MERGE_TOPOLOGY')=='hierarchical' and config.get('DILOCO_MERGE_BUCKET_NUMEL')=='67108864',
     'hierarchical 64M-bucket binding mismatch')
need(config.get('FAULT_INJECTION')=='none', 'clean payload contains fault injection')

# JobIDRaw|JobName|State|ExitCode|DerivedExitCode|NNodes|NTasks|NodeList|Partition|QOS|...
top=None; step_rows=[]
for line in sacct.splitlines()[1:]:
    p=line.split('|')
    if len(p)<10: continue
    if p[0]==job: top=p
    elif re.fullmatch(re.escape(job)+r'\.\d+',p[0]): step_rows.append(p)
need(top is not None,'terminal payload accounting missing')
if top:
    need(top[2]=='COMPLETED' and top[3]=='0:0','payload did not complete 0:0')
    need(top[5]=='32' and top[8]=='batch' and top[9]=='normal',
         'terminal accounting lacks NNodes=32 Partition=batch QOS=normal')
train_steps=[p for p in step_rows if p[6]=='256']
need(len(train_steps)==1 and train_steps[0][2]=='COMPLETED' and train_steps[0][3]=='0:0',
     'clean child step is not one COMPLETED 256-task execution epoch')

rows=[]
for line in text(run/'supervisor'/'execution-epochs.tsv').splitlines():
    p=line.split('|')
    if len(p)==6:
        try: rows.append({'epoch':int(p[0]),'rc':int(p[1]),'nodes':int(p[2]),'tasks':int(p[3]),'port':int(p[4]),'promoted':int(p[5])})
        except ValueError: pass
need(len(rows)==1 and rows[0]['rc']==0 and (rows[0]['nodes'],rows[0]['tasks'])==(32,256),
     'launcher did not retain one clean 32-node/256-rank epoch')
need(len(rows)==1 and rows[0]['promoted']==1, 'clean epoch latest checkpoint was not promoted')

stdout=text(run/'epochs'/'epoch-000001'/'train.out')
stderr=text(run/'epochs'/'epoch-000001'/'train.err')
need('DILOCO_FAULT_INJECTION' not in stdout+stderr, 'fault marker appeared in clean run')
need('Traceback (most recent call last)' not in stdout+stderr, 'Python traceback appeared')
need('Training complete! Final step:' in stdout, 'clean training completion marker missing')
need(text(run/'monitor'/'launcher-rc.txt').strip()=='0', 'production launcher return code was nonzero')

# Every rank reaches this post-init train.py line only after the exact launcher
# has required librccl-net.so in that child shell.
init_ranks={int(x) for x in re.findall(r'\[(?:DiLoCo|DDP)\] rank (\d+)/256 bound to',stdout)}
world_ranks={int(x) for x in re.findall(r'world_size=256 backend=nccl; this is rank (\d+)',stdout)}
need(init_ranks==set(range(256)),f'post-RCCL initialization rank set is incomplete: {len(init_ranks)}/256')
# train.py deliberately emits the world-size announcement only from rank 0;
# the per-rank bound line is the post-init proof for all 256 processes.
need(world_ranks=={0},f'rank-0 NCCL world-size announcement is invalid: {sorted(world_ranks)}')
manifest=load_json(run/'train'/'run_manifest.json')
runtime=manifest.get('runtime') or {}; renv=runtime.get('env') or {}
plugin_path=str(runtime.get('librccl_net_path','not-found'))
need(plugin_path!='not-found' and pathlib.Path(plugin_path).is_file(), 'requested librccl-net.so is unresolved')
need(renv.get('NCCL_NET_PLUGIN')=='librccl-net.so', 'NCCL_NET_PLUGIN request missing from runtime manifest')
need(renv.get('FRONTIER_ENABLE_OLCF_RCCL_PLUGIN')=='1', 'OLCF RCCL plugin opt-in missing')
args=load_json(run/'train'/'args.json')
need(str(args.get('level'))=='E97' and args.get('dim')==1792 and args.get('depth')==11,
     'E97 1.3B model configuration mismatch')

merge_matches=list(re.finditer(r'>>> \[DiLoCo\] merge #(\d+) at step (\d+): .*? across 256 ranks in (\d+) ms',stdout))
merges=[{'index':int(m.group(1)),'step':int(m.group(2)),'duration_ms':int(m.group(3))} for m in merge_matches]
need(len(merges)>=5,f'only {len(merges)} real K40 merges completed')
need(all(math.isfinite(m['duration_ms']) and m['duration_ms']>0 for m in merges), 'collective duration is nonfinite/nonpositive')
need('topology=hierarchical' in stdout and 'bucket_numel=67108864' in stdout, 'runtime hierarchical 64M-bucket evidence missing')

metrics=[]
metric_re=re.compile(r'step\s+(\d+) \| loss ([^ ]+) .*?global_tok/s ([^ ]+) \| elapsed_h [^ ]+ \| time ([^\n ]+)')
for m in metric_re.finditer(stdout):
    try:
        metrics.append({'step':int(m.group(1)),'loss':float(m.group(2)),'global_tps':float(m.group(3)),
                        'time':dt.datetime.fromisoformat(m.group(4))})
    except (ValueError,TypeError): pass
finite=[m for m in metrics if math.isfinite(m['loss']) and math.isfinite(m['global_tps']) and m['global_tps']>0]
need(len(finite)>=5,'finite loss/global throughput observations missing')
metric_by_step={m['step']:m for m in finite}
merge_times=[metric_by_step[m['step']]['time'] for m in merges if m['step'] in metric_by_step]
k40_cadences=[(b-a).total_seconds() for a,b in zip(merge_times,merge_times[1:])]
need(len(k40_cadences)>=4 and all(math.isfinite(x) and x>0 for x in k40_cadences), 'steady K40 cadence observations missing')
steady_k40=statistics.median(k40_cadences[1:] if len(k40_cadences)>1 else k40_cadences) if k40_cadences else math.nan
collective_ms=statistics.median([m['duration_ms'] for m in merges]) if merges else math.nan
steady_tps=statistics.median([m['global_tps'] for m in finite[-min(10,len(finite)):]]) if finite else math.nan

checkpoints=[]
for cp in run.joinpath('train').glob('checkpoint_step_*_loss_*.pt'):
    mm=re.search(r'checkpoint_step_(\d+)_loss_([^/]+)\.pt$',cp.name)
    if mm:
        try: checkpoints.append({'path':str(cp.resolve()),'step':int(mm.group(1)),'loss':float(mm.group(2)),'size':cp.stat().st_size})
        except (ValueError,OSError): pass
checkpoints.sort(key=lambda x:x['step'])
need(1<=len(checkpoints)<=2,f'checkpoint retention is not one/two files: {len(checkpoints)}')
large=[c for c in checkpoints if 7_000_000_000<=c['size']<=9_000_000_000]
need(bool(large),'no atomically published ~7.7GB checkpoint retained')
latest=run/'train'/'latest.pt'
need(latest.is_symlink() and latest.exists(),'latest.pt is not a readable atomic symlink')
need(not list(run.joinpath('train').glob('.*.tmp')),'temporary checkpoint/latest files remain')
latest_resolved=str(latest.resolve()) if latest.exists() else ''
need(any(c['path']==latest_resolved for c in checkpoints),'latest.pt does not name a retained checkpoint')

# Independent mmap reload uses the same torch serialization boundary as train.py.
reload_ok=False; reload_step=None; reload_loss=None; reload_keys=[]
if latest.exists():
    try:
        cp=torch.load(latest,map_location='cpu',mmap=True,weights_only=False)
        reload_step=int(cp['step']); reload_loss=float(cp['loss']); reload_keys=sorted(cp)
        # The accepted production profile uses the stateless avg outer update
        # (outer beta 0), for which train.py intentionally serializes no outer
        # tensor. Model, inner optimizer, step and finite loss are the complete
        # reload boundary; stateful outer modes would require their own field.
        reload_ok=('model_state_dict' in cp and 'optimizer_state_dict' in cp and
                   math.isfinite(reload_loss) and args.get('diloco_outer_optimizer')=='avg' and
                   float(args.get('diloco_outer_beta',1.0))==0.0)
        del cp
    except Exception as exc:
        reasons.append(f'checkpoint reload failed: {exc!r}')
need(reload_ok,'atomic latest checkpoint is not independently reloadable')

# Monitor observations: duration is tmp-file first sight to atomic latest switch.
events=[]
for line in text(run/'monitor'/'checkpoint-events.tsv').splitlines():
    p=line.split('\t',2)
    if len(p)==3:
        try: events.append((p[0],int(p[1]),p[2]))
        except ValueError: pass
starts={}
durations=[]
for kind,ns,path in events:
    sm=re.search(r'checkpoint_step_(\d+)_loss_',path)
    if not sm: continue
    step=int(sm.group(1))
    if kind=='tmp_seen': starts.setdefault(step,ns)
    elif kind=='latest_published' and step in starts and ns>=starts[step]: durations.append((step,(ns-starts[step])/1e9))
periodic=[x for x in durations if x[0]%200==0 and x[0]>1065000]
need(bool(periodic) and all(math.isfinite(x[1]) and x[1]>0 for x in periodic), 'checkpoint duration observation missing')
checkpoint_step,checkpoint_duration=(periodic[0] if periodic else (None,math.nan))

# Honest synchronous-pause observation: compare the first post-checkpoint log
# interval with the median same-sized non-checkpoint intervals. This is not an
# asynchronous/overlap claim and retains both raw interval and excess estimate.
post_interval=math.nan; baseline_interval=math.nan; observed_pause=math.nan
if checkpoint_step in metric_by_step:
    ordered=sorted(finite,key=lambda x:x['step'])
    intervals=[]
    for a,b in zip(ordered,ordered[1:]):
        if b['step']-a['step']==10:
            seconds=(b['time']-a['time']).total_seconds()
            if a['step']==checkpoint_step: post_interval=seconds
            elif a['step']%200!=0 and seconds>0: intervals.append(seconds)
    if intervals: baseline_interval=statistics.median(intervals)
    if math.isfinite(post_interval) and math.isfinite(baseline_interval): observed_pause=max(0.0,post_interval-baseline_interval)
need(math.isfinite(post_interval) and math.isfinite(baseline_interval) and math.isfinite(observed_pause),
     'observed synchronous training-pause interval is missing')

fields={
 'schema':'e97-same-allocation-32n-clean-v2','job_id':job,
 'source_sha':submission.get('source_sha'),'payload_digest':submission.get('payload_digest'),
 'predecessor_job':submission.get('predecessor_job'),'predecessor_payload_digest':submission.get('predecessor_payload_digest'),
 'nodes_requested':32,'world_size':256,'partition':'batch','qos':'normal',
 'initialized_ranks':len(init_ranks),'rccl_world_announcement_ranks':sorted(world_ranks),
 'rccl_plugin_path':plugin_path,'k40_merges':len(merges),'merge_steps':[m['step'] for m in merges],
 'k40_cadence_seconds':k40_cadences,'steady_k40_cadence_seconds':steady_k40,
 'collective_duration_ms':[m['duration_ms'] for m in merges],
 'median_collective_duration_ms':collective_ms,
 'steady_global_tokens_per_second':steady_tps,
 'last_step':finite[-1]['step'] if finite else None,'last_loss':finite[-1]['loss'] if finite else None,
 'last_global_tokens_per_second':finite[-1]['global_tps'] if finite else None,
 'checkpoint_path':latest_resolved,'checkpoint_size_bytes':next((c['size'] for c in checkpoints if c['path']==latest_resolved),0),
 'checkpoint_duration_step':checkpoint_step,'checkpoint_duration_seconds':checkpoint_duration,
 'post_checkpoint_log_interval_seconds':post_interval,'steady_log_interval_seconds':baseline_interval,
 'observed_synchronous_training_pause_seconds':observed_pause,
 'checkpoint_reloadable':reload_ok,'checkpoint_reload_step':reload_step,'checkpoint_reload_loss':reload_loss,
 'checkpoint_reload_keys':reload_keys,'retained_checkpoints':checkpoints,
 'clean_shutdown':top is not None and top[2]=='COMPLETED' and 'Training complete!' in stdout,
 'fault_injection':False,'unchanged_failed_payload_retried':False,
 'requirements':['R07','R12','R14','R16','NDP02-retired','NDP13','NDP15-atomic-only','NDP17-retired'],
 'errors':reasons,
}
fields['full_pass']=not reasons and all((fields['initialized_ranks']==256,
 fields['rccl_world_announcement_ranks']==[0],fields['k40_merges']>=5,
 fields['checkpoint_reloadable'],fields['clean_shutdown']))
for path in (run/'verdict.json',out/'verdict.json'):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(fields,sort_keys=True,indent=2)+'\n'); os.replace(tmp,path)
print(json.dumps(fields,sort_keys=True))
if not fields['full_pass']: raise SystemExit(1)
PY
rc=$?
sha256sum "$COLLECTOR_ROOT"/* > "$COLLECTOR_ROOT/SHA256SUMS" || true
exit "$rc"
