#!/usr/bin/env python3
"""Fail-closed parity and submission gate for the job-4962400 launcher."""
import argparse, hashlib, json, os, sys
from pathlib import Path

ALLOWED={"profile","walltime","queue"}
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
def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke",type=Path,required=True); p.add_argument("--production",type=Path,required=True); p.add_argument("--policy",type=Path,required=True); p.add_argument("--submit",action="store_true"); p.add_argument("--approval",type=Path); p.add_argument("--require-promotion",action="store_true"); a=p.parse_args()
    s,sg=read(a.smoke); prod,pg=read(a.production)
    if canon(sg)!=canon(pg): fail("golden_manifest_drift",None)
    if normalized(s)!=normalized(prod): fail("forbidden_drift",{"smoke":s,"production":prod})
    if s["training_stop_budget"]!=prod["training_stop_budget"]: fail("training_stop_budget_drift",None)
    if (a.submit or a.require_promotion):
        try: promotion=json.loads((a.smoke/"promotion.json").read_text())
        except Exception as e: fail("promotion",str(e))
        if promotion.get("fingerprint")!=(a.smoke/"fingerprint.sha256").read_text().strip() or promotion.get("job_id")!=4962400 or promotion.get("slurm_state")!="COMPLETED": fail("promotion",promotion)
    if a.submit:
        if not a.approval: fail("approval","required")
        approval=json.loads(a.approval.read_text())
        if approval!={"approved":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip()}: fail("approval",approval)
        os.execvp(prod["sbatch_argv"][0],prod["sbatch_argv"])
    print(json.dumps({"ok":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip(),"allowed_differences":{"walltime":[s["walltime"],prod["walltime"]],"partition":[s["queue"]["partition"],prod["queue"]["partition"]],"qos":[s["queue"]["qos"],prod["queue"]["qos"]]},"training_stop_budget":s["training_stop_budget"]},sort_keys=True))
if __name__=="__main__": main()
