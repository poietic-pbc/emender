#!/usr/bin/env python3
"""Fail-closed parity and submission gate for the job-4962400 launcher."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path

ALLOWED={"profile","walltime","queue"}
EXPECTED_POLICY={"schema_version":2,"allowed_profile_keys":["walltime","queue"],"allowed_queue_keys":["partition","qos"],"allowed_sbatch_argv_flags":["-t","-p","-q"],"forbidden_differences":"all launch inputs not explicitly allowlisted","training_stop_budget_must_match":True,"launcher_must_match_job_4962400_byte_for_byte":True}
def fail(kind,detail): print(json.dumps({"ok":False,"kind":kind,"detail":detail},sort_keys=True),file=sys.stderr); raise SystemExit(1)
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"))
def read(bundle):
    try: launch=json.loads((bundle/"launch-inputs.json").read_text()); golden=json.loads((bundle/"golden-manifest.json").read_text())
    except Exception as e: fail("bundle",str(e))
    actual=hashlib.sha256((bundle/"rendered.sbatch").read_bytes()).hexdigest()
    if actual!=golden["files"][golden["script"]] or actual!=launch["launcher_sha256"]: fail("launcher_drift",actual)
    return launch,golden
def normalized(value):
    value=json.loads(json.dumps(value)); argv=value["sbatch_argv"]
    for flag,token in (("-t","@WALLTIME@"),("-p","@PARTITION@"),("-q","@QOS@")):
        if argv.count(flag)!=1: fail("argv",f"missing/duplicate {flag}")
        argv[argv.index(flag)+1]=token
    for key in ALLOWED: value[key]="@ALLOWED@"
    return value
def repository_commits():
    root=Path(__file__).resolve().parents[2]
    try:
        head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(root),universal_newlines=True).strip()
        origin=subprocess.check_output(["git","rev-parse","origin/main"],cwd=str(root),universal_newlines=True).strip()
    except (OSError,subprocess.CalledProcessError) as e:
        fail("origin_commit",str(e))
    return head,origin
def validate_promotion(promotion, launch, bundle):
    fingerprint=(bundle/"fingerprint.sha256").read_text().strip()
    seed=launch["resolved"]["seed"]
    job_id=promotion.get("job_id")
    required={
        "fingerprint":fingerprint,"slurm_state":"COMPLETED","exit_code":"0:0",
        "nodes":launch["resolved"]["nodes"],"ranks":launch["resolved"]["launched_ranks"],
        "seed":seed,
    }
    if not isinstance(job_id,int) or isinstance(job_id,bool) or job_id <= 0:
        fail("promotion",promotion)
    if not isinstance(promotion.get("origin_commit"),str) or len(promotion["origin_commit"])!=40 or any(c not in "0123456789abcdef" for c in promotion["origin_commit"]):
        fail("promotion",promotion)
    head,origin=repository_commits()
    if promotion["origin_commit"]!=head or promotion["origin_commit"]!=origin:
        fail("origin_commit",{"promotion":promotion["origin_commit"],"head":head,"origin_main":origin})
    if any(promotion.get(key)!=value for key,value in required.items()):
        fail("promotion",promotion)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke",type=Path,required=True); p.add_argument("--production",type=Path,required=True); p.add_argument("--policy",type=Path,required=True); p.add_argument("--submit",action="store_true"); p.add_argument("--approval",type=Path); p.add_argument("--require-promotion",action="store_true"); a=p.parse_args()
    try: policy=json.loads(a.policy.read_text())
    except Exception as e: fail("policy",str(e))
    if policy!=EXPECTED_POLICY: fail("policy",policy)
    s,sg=read(a.smoke); prod,pg=read(a.production)
    if canon(sg)!=canon(pg): fail("golden_manifest_drift",None)
    if normalized(s)!=normalized(prod): fail("forbidden_drift",{"smoke":s,"production":prod})
    if s["training_stop_budget"]!=prod["training_stop_budget"]: fail("training_stop_budget_drift",None)
    if (a.submit or a.require_promotion):
        try: promotion=json.loads((a.smoke/"promotion.json").read_text())
        except Exception as e: fail("promotion",str(e))
        validate_promotion(promotion,s,a.smoke)
    if a.submit:
        if not a.approval: fail("approval","required")
        approval=json.loads(a.approval.read_text())
        if approval!={"approved":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip()}: fail("approval",approval)
        os.execvp(prod["sbatch_argv"][0],prod["sbatch_argv"])
    print(json.dumps({"ok":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip(),"allowed_differences":{"walltime":[s["walltime"],prod["walltime"]],"partition":[s["queue"]["partition"],prod["queue"]["partition"]],"qos":[s["queue"]["qos"],prod["queue"]["qos"]]},"training_stop_budget":s["training_stop_budget"]},sort_keys=True))
if __name__=="__main__": main()
