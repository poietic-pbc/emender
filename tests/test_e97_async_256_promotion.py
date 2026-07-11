import copy, importlib.util, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
RENDER=module("e97render",ROOT/"scripts/frontier/render_e97_async_256.py")
CHECK=[sys.executable,str(ROOT/"scripts/frontier/check_e97_async_promotion.py")]
POLICY=str(ROOT/"configs/frontier/e97_async_256_parity_policy.json")

def bundles(tmp_path):
    s=tmp_path/"smoke"; p=tmp_path/"production"; RENDER.render("smoke",s); RENDER.render("production",p); return s,p
def check(s,p,*extra):
    return subprocess.run(CHECK+["--smoke",str(s),"--production",str(p),"--policy",POLICY]+list(extra),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
def mutate_json(path,parts,value):
    x=json.loads(path.read_text()); cur=x
    for p in parts[:-1]: cur=cur[p]
    cur[parts[-1]]=value; path.write_text(json.dumps(x))

def test_golden_exact_commands_and_only_typed_differences(tmp_path):
    s,p=bundles(tmp_path); r=check(s,p)
    assert r.returncode==0,r.stderr
    result=json.loads(r.stdout); assert result["allowed_differences"]=={"time":["00:20:00","12:00:00"],"partition":["batch","batch"],"qos":["debug","normal"]}
    assert (s/"fingerprint.sha256").read_text()==(p/"fingerprint.sha256").read_text()

def test_regression_4962400_vs_4963853_forbidden_drift(tmp_path):
    s,p=bundles(tmp_path)
    mutate_json(p/"runtime.json",["topology","participant_ranks"],256)
    mutate_json(p/"runtime.json",["topology","global_quorum"],171)
    mutate_json(p/"runtime.json",["trainer","params"],"1.3b")
    assert check(s,p).returncode!=0

MUTATIONS=[
 ("runtime.json",["environment","MPICH_GPU_SUPPORT_ENABLED"],"1"),
 ("runtime.json",["modules",0],"PrgEnv-cray"),
 ("runtime.json",["topology","launched_ranks"],2047),
 ("runtime.json",["topology","worker_ranks"],256),
 ("runtime.json",["topology","global_quorum"],171),
 ("runtime.json",["trainer","optimizer"],"adamw"),
 ("runtime.json",["trainer","timeout_s"],240),
 ("runtime.json",["trainer","params"],"1.3b"),
 ("runtime.json",["checkpoint","publication"],"external-latest"),
 ("runtime.json",["transport","bucket_bytes"],1),
 ("inputs.json",["seed","sha256"],"1"*64),
 ("inputs.json",["data","sha256"],"2"*64),
 ("inputs.json",["tokenizer","sha256"],"0"*40),
 ("code.json",["git_commit"],"0"*40),
 ("code.json",["entrypoint","sha256"],"0"*64),
 ("helper-manifest.json",[0,"sha256"],"0"*64),
 ("runtime-manifest.json",[0,"sha256"],"0"*64),
]
import pytest
@pytest.mark.parametrize("name,parts,value",MUTATIONS)
def test_forbidden_json_mutations_fail_closed(tmp_path,name,parts,value):
    s,p=bundles(tmp_path); mutate_json(p/name,parts,value); assert check(s,p).returncode!=0

@pytest.mark.parametrize("directive,value",[("account","other"),("nodes","255"),("ntasks-per-node","7"),("gpus-per-task","2"),("export","ALL")])
def test_forbidden_scheduler_mutations(tmp_path,directive,value):
    s,p=bundles(tmp_path); q=p/"rendered.sbatch"; q.write_text(q.read_text().replace("#SBATCH --"+directive+"=", "#SBATCH --"+directive+"="+value+" #")); assert check(s,p).returncode!=0

def test_missing_unknown_directive_arbitrary_path_and_unresolved_variable(tmp_path):
    for transform in (lambda x:x.replace("#SBATCH --account=bif148\n",""),lambda x:x.replace("#SBATCH --account=bif148","#SBATCH --account=bif148\n#SBATCH --reservation=x"),lambda x:x.replace("e97-async-256-production-%j.out","arbitrary.out"),lambda x:x+"\necho ${UNRESOLVED}\n"):
        s,p=bundles(tmp_path); f=p/"rendered.sbatch"; f.write_text(transform(f.read_text())); assert check(s,p).returncode!=0

def test_unknown_profile_keys_and_nondeterministic_duration_fail(tmp_path,monkeypatch):
    original=RENDER.CONFIG
    c=json.loads(original.read_text()); c["profiles"]["smoke"]["account"]="bad"
    bad=tmp_path/"bad.yaml"; bad.write_text(json.dumps(c)); monkeypatch.setattr(RENDER,"CONFIG",bad)
    with pytest.raises(ValueError): RENDER.load()
    c["profiles"]["smoke"].pop("account"); c["profiles"]["smoke"]["trainer_duration_seconds"]=1; bad.write_text(json.dumps(c))
    with pytest.raises(ValueError): RENDER.load()

def test_dirty_tracked_worktree_is_rejected(monkeypatch):
    monkeypatch.setattr(RENDER.subprocess,"check_output",lambda *a,**k:" M train.py\n")
    with pytest.raises(ValueError,match="dirty"): RENDER.ensure_clean_tree()

def test_submission_stops_before_sbatch_without_atomic_promotion(tmp_path):
    s,p=bundles(tmp_path); marker=tmp_path/"sbatch-called"; fake=tmp_path/"sbatch"; fake.write_text("#!/bin/sh\ntouch '%s'\n"%marker); fake.chmod(0o755)
    r=check(s,p,"--submit","--approval",str(tmp_path/"missing.json")); assert r.returncode!=0; assert not marker.exists()
