#!/usr/bin/env python3
"""Strict smoke/production parity and promotion/submission gate."""
import argparse, hashlib, json, math, os, re, subprocess, sys
from pathlib import Path

ARTIFACTS=("runtime.json","inputs.json","code.json","runtime-manifest.json","helper-manifest.json")
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def fail(kind,detail):
    print(json.dumps({"ok":False,"kind":kind,"detail":detail},sort_keys=True),file=sys.stderr); raise SystemExit(1)
def directives(text):
    out={}
    for line in text.splitlines():
        if not line.startswith("#SBATCH "): continue
        item=line[8:].strip()
        if not item.startswith("--") or "=" not in item: fail("directive_syntax",line)
        k,v=item[2:].split("=",1)
        if k in out: fail("duplicate_directive",k)
        out[k]=v
    return out
def unresolved(x): return bool(re.search(r"\$\{[^}]+\}",canon(x)))
def normalized(bundle,policy):
    script=(bundle/"rendered.sbatch").read_text(); d=directives(script)
    req=set(policy["required_directives"])
    if set(d)!=req: fail("directives",{"missing":sorted(req-set(d)),"unknown":sorted(set(d)-req)})
    for k in policy["allowed_scheduler_differences"]: script=script.replace(f"#SBATCH --{k}={d[k]}",f"#SBATCH --{k}=@{k.upper()}@")
    for k in ("job-name","output","error"):
        if not re.fullmatch(r"e97-async-256-(smoke|production)(-%j\.(out|err))?",d[k]): fail("derived_scheduler_metadata",{k:d[k]})
        d[k]=re.sub(r"e97-async-256-(smoke|production)","e97-async-256-@PROFILE@",d[k])
    script=re.sub(r"e97-async-256-(smoke|production)","e97-async-256-@PROFILE@",script)
    vals={n:json.loads((bundle/n).read_text()) for n in ARTIFACTS}
    r=vals["runtime.json"]; duration=r["trainer_duration_seconds"]
    script=re.sub(r"(?m)^BUNDLE_DIR=.*$","BUNDLE_DIR=@BUNDLE_DIR@",script)
    script=re.sub(r"BUNDLE_FINGERPRINT='[0-9a-f]{64}'","BUNDLE_FINGERPRINT='@FINGERPRINT@'",script)
    script=re.sub(r"BUNDLE_MANIFEST_SHA256='[0-9a-f]{64}'","BUNDLE_MANIFEST_SHA256='@BUNDLE_MANIFEST@'",script)
    script=script.replace("--walltime-remaining-s "+str(duration),"--walltime-remaining-s @DURATION@")
    r["profile"]="@PROFILE@"; r["trainer_duration_seconds"]="@DURATION@"
    av=r["trainer_argv"]
    try: av[av.index("--walltime-remaining-s")+1]="@DURATION@"
    except (ValueError,IndexError): fail("duration","missing deterministic stop budget")
    if unresolved(vals): fail("unresolved_variable","manifest contains ${...}")
    allowed_vars={"BASH_SOURCE[0]","SLURM_SUBMIT_DIR:?","SLURM_SUBMIT_DIR","SLURM_JOB_ID:?","SLURM_JOB_ID","TMPDIR:-/tmp","SLURM_NODELIST","SLURM_NODELIST:?","RUN_ID","RUN_DIR","METRICS","HELPER","IPC","SLURM_PROCID:?missing SLURM_PROCID","COORDINATOR_HOST","COORDINATOR_PORT"}
    unknown=set(re.findall(r"\$\{([^}]+)\}",script))-allowed_vars
    if unknown: fail("unresolved_variable",sorted(unknown))
    return {"script":script,"artifacts":vals}, d
def invariants(n):
    r=n["artifacts"]["runtime.json"]; t=r["topology"]; s=r["srun_argv"]; a=r["trainer_argv"]
    if [t[k] for k in ("launched_ranks","participant_ranks","worker_ranks","global_quorum")]!=[2048]*4 or t["rank_ids"]!={"start":0,"stop":2047}: fail("topology",t)
    expected=["srun","-N","256","-n","2048","--ntasks-per-node","8","-c","7","--gpus-per-task","1","--gpu-bind","closest"]
    if s!=expected: fail("srun_argv",s)
    needed={"--node-count":"2048","--worker-count":"2048","--global-quorum":"2048","--node-rank":"@SLURM_PROCID@","--device":"cuda:0"}
    for k,v in needed.items():
        if k not in a or a[a.index(k)+1]!=v: fail("trainer_argv",{k:v})
def verify_promotion(path,fingerprint):
    try: p=json.loads(path.read_text())
    except Exception as e: fail("promotion",f"missing/invalid atomic promotion.json: {e}")
    required={"schema_version","fingerprint","job_id","slurm_state","exit_code","rank_ids","rank_starts","accepted_updates","finite_loss","completed_merges","metrics","checkpoint_finalized","checkpoint_reloaded","external_pointer_unchanged","artifact_paths"}
    if set(p)!=required: fail("promotion_schema",{"missing":sorted(required-set(p)),"unknown":sorted(set(p)-required)})
    ok=(p["schema_version"]==1 and p["fingerprint"]==fingerprint and p["slurm_state"]=="COMPLETED" and p["exit_code"]=="0:0" and p["rank_starts"]==2048 and p["accepted_updates"]==2048 and p["rank_ids"]==list(range(2048)) and p["finite_loss"] is True and p["completed_merges"]>=1 and p["checkpoint_finalized"] is True and p["checkpoint_reloaded"] is True and p["external_pointer_unchanged"] is True and p["metrics"])
    if not ok: fail("promotion_evidence","successful exact-topology evidence not satisfied")
def verify_live_inputs(bundle):
    for name,v in json.loads((bundle/"inputs.json").read_text()).items():
        p=Path(v["realpath"])
        if not p.is_file() or p.stat().st_size!=v["size"] or os.path.realpath(v["path"])!=v["realpath"]: fail("changed_input",name)
        if v["hash_kind"]=="sha256-stat-v1":
            st=p.stat(); ident={"realpath":os.path.realpath(str(p)),"size":st.st_size,"mtime_ns":st.st_mtime_ns,"inode":st.st_ino}; observed=hashlib.sha256(canon(ident).encode()).hexdigest()
        elif v["hash_kind"]=="sha256":
            h=hashlib.sha256()
            with p.open("rb") as f:
                for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
            observed=h.hexdigest()
        else: fail("input_hash_kind",v["hash_kind"])
        if observed!=v["sha256"]: fail("changed_input_hash",name)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke",required=True,type=Path); ap.add_argument("--production",required=True,type=Path); ap.add_argument("--policy",required=True,type=Path); ap.add_argument("--require-promotion",action="store_true"); ap.add_argument("--write-promotion",type=Path,metavar="EVIDENCE_JSON"); ap.add_argument("--submit",action="store_true"); ap.add_argument("--approval",type=Path); a=ap.parse_args()
    policy=json.loads(a.policy.read_text()); sn,sd=normalized(a.smoke,policy); pn,pd=normalized(a.production,policy); invariants(sn); invariants(pn)
    for k in set(sd)|set(pd):
        if sd.get(k)!=pd.get(k) and k not in policy["allowed_scheduler_differences"]: fail("scheduler_drift",{k:[sd.get(k),pd.get(k)]})
    if canon(sn)!=canon(pn): fail("forbidden_drift",{"smoke_sha256":hashlib.sha256(canon(sn).encode()).hexdigest(),"production_sha256":hashlib.sha256(canon(pn).encode()).hexdigest()})
    sf=(a.smoke/"fingerprint.sha256").read_text().strip(); pf=(a.production/"fingerprint.sha256").read_text().strip()
    if sf!=pf: fail("fingerprint",[sf,pf])
    if a.write_promotion:
        try: evidence=json.loads(a.write_promotion.read_text())
        except Exception as e: fail("promotion_evidence",str(e))
        evidence["fingerprint"]=sf
        tmp=a.smoke/(".promotion.json.tmp.%d"%os.getpid()); tmp.write_text(json.dumps(evidence,sort_keys=True,indent=2)+"\n")
        verify_promotion(tmp,sf); os.replace(str(tmp),str(a.smoke/"promotion.json"))
    if a.require_promotion or a.submit: verify_promotion(a.smoke/"promotion.json",sf)
    if a.submit:
        verify_live_inputs(a.production)
        if not a.approval: fail("approval","current exact-fingerprint human approval required")
        try: approval=json.loads(a.approval.read_text())
        except Exception as e: fail("approval",str(e))
        if approval!={"approved":True,"fingerprint":sf}: fail("approval","must contain only approved=true and exact fingerprint")
        os.execvp("sbatch",["sbatch",str(a.production/"rendered.sbatch")])
    print(json.dumps({"ok":True,"fingerprint":sf,"allowed_differences":{"time":[sd["time"],pd["time"]],"partition":[sd["partition"],pd["partition"]],"qos":[sd["qos"],pd["qos"]]}}))
if __name__=="__main__": main()
