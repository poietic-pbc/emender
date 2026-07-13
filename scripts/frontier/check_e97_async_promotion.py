#!/usr/bin/env python3
"""Fail-closed parity and submission gate for the job-4962400 launcher."""
import argparse, hashlib, io, json, os, subprocess, sys, tarfile, tempfile
from pathlib import Path

ALLOWED={"profile","nodes","walltime","queue"}
EXPECTED_POLICY={"schema_version":2,"allowed_profile_keys":["nodes","walltime","queue"],"allowed_queue_keys":["partition","qos"],"allowed_sbatch_argv_flags":["-N","-t","-p","-q"],"forbidden_differences":"all launch inputs not explicitly allowlisted","training_stop_budget_must_match":True,"launcher_must_match_job_4962400_byte_for_byte":True}
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
    nodes=value["nodes"]; ranks=nodes*8
    if any(value["resolved"][key]!=want for key,want in {
        "nodes":nodes,"launched_ranks":ranks,"participant_ranks":ranks,
        "worker_ranks":ranks,"global_quorum":ranks}.items()): fail("derived_scale",value["resolved"])
    if argv[argv.index("-N")+1]!=str(nodes): fail("derived_scale",argv)
    export=argv[argv.index("--export")+1]
    if any(f"{key}={ranks}" not in export for key in ("ASYNC_TRAINPY_RANKS","ASYNC_EXPECTED_RANKS","ASYNC_GLOBAL_QUORUM")): fail("derived_scale",export)
    for flag,token in (("-N","@NODES@"),("-t","@WALLTIME@"),("-p","@PARTITION@"),("-q","@QOS@")):
        if argv.count(flag)!=1: fail("argv",f"missing/duplicate {flag}")
        argv[argv.index(flag)+1]=token
    export_index=argv.index("--export")+1
    argv[export_index]=__import__("re").sub(r"(ASYNC_(?:TRAINPY|EXPECTED)_RANKS|ASYNC_GLOBAL_QUORUM)=\d+",r"\1=@DERIVED_RANKS@",argv[export_index])
    for key in ("nodes","launched_ranks","participant_ranks","worker_ranks","global_quorum"):
        value["resolved"][key]="@DERIVED@"
    for key in ALLOWED: value[key]="@ALLOWED@"
    return value
def git(root,*args):
    return subprocess.check_output(["git",*args],cwd=str(root),universal_newlines=True,stderr=subprocess.DEVNULL).strip()
def remote_contains(root,commit):
    return [r.strip() for r in git(root,"branch","-r","--contains",commit).splitlines() if r.strip() and " -> " not in r]
def repository_commits(attested):
    root=Path(__file__).resolve().parents[2]
    try:
        head=git(root,"rev-parse","HEAD")
        origin=git(root,"rev-parse","origin/main")
        git(root,"cat-file","-e",attested+"^{commit}")
        subprocess.check_call(["git","merge-base","--is-ancestor",attested,"origin/main"],cwd=str(root),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # The exact gate/submission commit must be retained on an origin ref;
        # evidence-only main may advance independently of a review branch.
        if not remote_contains(root,attested):
            raise subprocess.CalledProcessError(1,"git branch -r --contains attested")
        if not remote_contains(root,head):
            raise subprocess.CalledProcessError(1,"git branch -r --contains HEAD")
    except (OSError,subprocess.CalledProcessError) as e:
        fail("origin_commit",str(e))
    return head,origin
def validate_attested_files(commit,golden):
    root=Path(__file__).resolve().parents[2]
    for path,want in golden["files"].items():
        try: content=subprocess.check_output(["git","show",commit+":"+path],cwd=str(root),stderr=subprocess.DEVNULL)
        except (OSError,subprocess.CalledProcessError) as e: fail("attested_tree_drift",{"path":path,"error":str(e)})
        got=hashlib.sha256(content).hexdigest()
        if got!=want: fail("attested_tree_drift",{"path":path,"got":got,"want":want})
def duration_seconds(value):
    try:
        parts=[int(part) for part in value.split(":")]
    except (AttributeError,ValueError):
        fail("smoke_terminal_evidence",{"invalid_duration":value})
    if len(parts)!=3 or parts[1]>=60 or parts[2]>=60:
        fail("smoke_terminal_evidence",{"invalid_duration":value})
    return parts[0]*3600+parts[1]*60+parts[2]
def validate_terminal_evidence(promotion, launch, bundle):
    relative=promotion.get("terminal_evidence")
    if not isinstance(relative,str) or Path(relative).is_absolute() or Path(relative).parts!=(relative,):
        fail("smoke_terminal_evidence",{"path":relative})
    path=bundle/relative
    try:
        payload=path.read_bytes()
        evidence=json.loads(payload)
    except (OSError,ValueError) as error:
        fail("smoke_terminal_evidence",str(error))
    if hashlib.sha256(payload).hexdigest()!=promotion.get("terminal_evidence_sha256"):
        fail("smoke_terminal_evidence",{"sha256":"mismatch"})
    required={
        "schema_version":1,"result":"pass","job_id":promotion["job_id"],
        "submission_count":1,"nodes":launch["resolved"]["nodes"],
        "ranks":launch["resolved"]["launched_ranks"],"ranks_per_node":8,
        "partition":launch["queue"]["partition"],"qos":launch["queue"]["qos"],
        "walltime":launch["walltime"],"terminal_state":"TIMEOUT",
        "scheduler_controlled_finalization":True,"accepted_updates_per_generation":16,
        "host_oom":False,"cross_node_path_failure":False,
        "latest_checkpoint_reload":"pass","stable_seed_pointer_unchanged":True,
        "production_submitted":False,"larger_smoke_submitted":False,
    }
    if any(evidence.get(key)!=value for key,value in required.items()):
        fail("smoke_terminal_evidence",evidence)
    generations=evidence.get("generations_completed")
    if not isinstance(generations,list) or len(generations)<2 or generations[:2]!=[0,1]:
        fail("smoke_terminal_evidence",evidence)
    if evidence.get("merges_completed",0)<2:
        fail("smoke_terminal_evidence",evidence)
    seed_step=launch["resolved"]["seed"]["step"]
    if evidence.get("latest_checkpoint_step",0)<=seed_step+launch["resolved"]["local_steps"]:
        fail("smoke_terminal_evidence",evidence)
    requested=duration_seconds(evidence["walltime"])
    elapsed=duration_seconds(evidence.get("terminal_elapsed"))
    if elapsed<requested or elapsed>requested+60:
        fail("smoke_terminal_evidence",evidence)
    return evidence
def validate_promotion(promotion, launch, bundle, golden):
    fingerprint=(bundle/"fingerprint.sha256").read_text().strip()
    seed=launch["resolved"]["seed"]
    job_id=promotion.get("job_id")
    required={
        "fingerprint":fingerprint,"slurm_state":"TIMEOUT","exit_code":"0:0",
        "nodes":launch["resolved"]["nodes"],"ranks":launch["resolved"]["launched_ranks"],
        "seed":seed,
    }
    if not isinstance(job_id,int) or isinstance(job_id,bool) or job_id <= 0:
        fail("promotion",promotion)
    if not isinstance(promotion.get("origin_commit"),str) or len(promotion["origin_commit"])!=40 or any(c not in "0123456789abcdef" for c in promotion["origin_commit"]):
        fail("promotion",promotion)
    head,origin=repository_commits(promotion["origin_commit"])
    if any(promotion.get(key)!=value for key,value in required.items()):
        fail("promotion",promotion)
    terminal=validate_terminal_evidence(promotion,launch,bundle)
    validate_attested_files(promotion["origin_commit"],golden)
    return {"attested_commit":promotion["origin_commit"],"submission_commit":head,"origin_main":origin,"smoke_job_id":terminal["job_id"]}
def materialize_attested_tree(commit, bundle, golden):
    """Create the execution tree from the immutable commit, never current HEAD."""
    root=Path(__file__).resolve().parents[2]
    bundle.mkdir(parents=True,exist_ok=True)
    payload=Path(tempfile.mkdtemp(prefix="attested-tree-"+commit+"-",dir=str(bundle)))
    try:
        archive=subprocess.check_output(["git","archive","--format=tar",commit],cwd=str(root))
        with tarfile.open(fileobj=io.BytesIO(archive),mode="r:") as tf:
            for member in tf.getmembers():
                target=(payload/member.name).resolve()
                if payload.resolve() not in target.parents and target!=payload.resolve():
                    fail("attested_tree",member.name)
                if member.issym() or member.islnk(): fail("attested_tree",member.name)
            tf.extractall(payload)
        for path,want in golden["files"].items():
            candidate=payload/path
            got=hashlib.sha256(candidate.read_bytes()).hexdigest()
            if got!=want: fail("attested_tree",{"path":path,"got":got,"want":want})
    except (OSError,subprocess.CalledProcessError,tarfile.TarError,KeyError) as e:
        fail("attested_tree",str(e))
    return payload
def main():
    p=argparse.ArgumentParser(); p.add_argument("--smoke",type=Path,required=True); p.add_argument("--production",type=Path,required=True); p.add_argument("--policy",type=Path,required=True); p.add_argument("--submit",action="store_true"); p.add_argument("--approval",type=Path); p.add_argument("--attempt-marker",type=Path); p.add_argument("--require-promotion",action="store_true"); a=p.parse_args()
    try: policy=json.loads(a.policy.read_text())
    except Exception as e: fail("policy",str(e))
    if policy!=EXPECTED_POLICY: fail("policy",policy)
    s,sg=read(a.smoke); prod,pg=read(a.production)
    if canon(sg)!=canon(pg): fail("golden_manifest_drift",None)
    if normalized(s)!=normalized(prod): fail("forbidden_drift",{"smoke":s,"production":prod})
    if s["training_stop_budget"]!=prod["training_stop_budget"]: fail("training_stop_budget_drift",None)
    identity=None
    if (a.submit or a.require_promotion):
        try: promotion=json.loads((a.smoke/"promotion.json").read_text())
        except Exception as e: fail("promotion",str(e))
        identity=validate_promotion(promotion,s,a.smoke,sg)
    if a.submit:
        if not a.approval: fail("approval","required")
        if not a.attempt_marker: fail("attempt_marker","required")
        approval=json.loads(a.approval.read_text())
        if approval!={"approved":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip()}: fail("approval",approval)
        try:
            a.attempt_marker.parent.mkdir(parents=True,exist_ok=True)
            with a.attempt_marker.open("x") as marker:
                json.dump({"status":"sbatch_exec_started","fingerprint":approval["fingerprint"],"attested_commit":identity["attested_commit"]},marker,sort_keys=True)
                marker.write("\n")
        except FileExistsError:
            fail("duplicate_submission_attempt",str(a.attempt_marker))
        payload=materialize_attested_tree(identity["attested_commit"],a.production,pg)
        os.chdir(payload)
        os.execvp(prod["sbatch_argv"][0],prod["sbatch_argv"])
    print(json.dumps({"ok":True,"fingerprint":(a.smoke/"fingerprint.sha256").read_text().strip(),"launch_identity":identity,"allowed_differences":{"nodes":[s["nodes"],prod["nodes"]],"ranks":[s["resolved"]["launched_ranks"],prod["resolved"]["launched_ranks"]],"walltime":[s["walltime"],prod["walltime"]],"partition":[s["queue"]["partition"],prod["queue"]["partition"]],"qos":[s["queue"]["qos"],prod["queue"]["qos"]]},"training_stop_budget":s["training_stop_budget"]},sort_keys=True))
if __name__=="__main__": main()
