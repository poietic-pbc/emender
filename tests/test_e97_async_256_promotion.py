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

def test_slurm_spooled_bash_source_does_not_select_bundle(tmp_path):
    smoke,_=bundles(tmp_path)
    script=smoke/"rendered.sbatch"
    spool=tmp_path/"var/spool/slurmd/job4972201"
    spool.mkdir(parents=True)
    spooled=spool/"slurm_script"
    spooled.write_text(script.read_text())
    assignment=next(line for line in spooled.read_text().splitlines() if line.startswith("BUNDLE_DIR="))
    result=subprocess.run(
        ["bash","-c",assignment+'\nprintf "%s\\n" "$BUNDLE_DIR"'],
        cwd=spool, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode==0,result.stderr
    assert Path(result.stdout.strip())==smoke.resolve()
    assert str(spool) not in result.stdout
    assert "BASH_SOURCE" not in assignment

def test_rendered_on_node_preflight_is_before_srun_and_restores_compiler(tmp_path):
    smoke,_=bundles(tmp_path)
    text=(smoke/"rendered.sbatch").read_text()
    before_helper=text.index("--phase before-helper")
    build=text.index("build_compiled_mpich_dense_helper.sh",before_helper)
    before_srun=text.index("--phase before-srun",build)
    launch=text.index("exec srun",before_srun)
    assert before_helper < build < before_srun < launch
    assert "source /etc/profile" in text
    assert "module load PrgEnv-gnu" in text
    assert "module load cray-mpich" in text
    assert "module load rocm/6.3.1" in text
    assert "command -v CC" in text
    assert "bundle-files.sha256" in text
    assert "export RUN_ID RUN_DIR METRICS HELPER IPC COORDINATOR_HOST COORDINATOR_PORT" in text

def test_preflight_verifies_all_immutable_launch_surfaces():
    text=(ROOT/"scripts/frontier/e97_async_256_preflight.py").read_text()
    for kind in ("fingerprint", "code_commit", "code_tree", "entrypoint", "input_hash", "environment", "modules", "helper", "launcher", "trainer_argv", "srun_argv", "topology"):
        assert kind in text

def test_silent_presrun_failure_names_exact_command_and_phase(tmp_path):
    # This is the failure mode from job 4972494: plain `set -e` preserves the
    # nonzero status but a quiet module/helper command leaves both logs empty.
    old=subprocess.run(
        ["bash","-c","set -e; silent_module_load() { return 19; }; silent_module_load cray-mpich"],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    )
    assert old.returncode==19
    assert old.stdout==old.stderr==""

    smoke,_=bundles(tmp_path)
    text=(smoke/"rendered.sbatch").read_text()
    start=text.index("E97_PHASE=bootstrap")
    end=text.index("BUNDLE_MANIFEST_SHA256=",start)
    diagnostics=text[start:end]
    result=subprocess.run(
        ["bash","-c","set -Eeuo pipefail\nBUNDLE_FINGERPRINT=test-fingerprint\n"+diagnostics+"e97_phase module-bootstrap\nsilent_module_load() { return 19; }\nsilent_module_load cray-mpich\n"],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    )
    assert result.returncode==19
    assert "phase=module-bootstrap status=error rc=19" in result.stderr
    assert "command=return\\ 19" in result.stderr

def test_rendered_script_has_actionable_phases_through_srun(tmp_path):
    smoke,_=bundles(tmp_path)
    text=(smoke/"rendered.sbatch").read_text()
    phases=[
        "bundle-binding", "module-bootstrap", "runtime-paths",
        "preflight-before-helper", "helper-build", "preflight-before-srun", "srun",
    ]
    positions=[text.index("e97_phase "+phase) for phase in phases]
    assert positions==sorted(positions)
    assert "set -Eeuo pipefail" in text
    assert "trap e97_error ERR" in text
    assert "status=error rc=%s line=%s command=%q" in text
    assert "phase=srun status=exec ranks=2048 nodes=256" in text
    assert "source /etc/profile.d/olcf-env.sh" in text
    assert "source /opt/cray/pe/lmod/lmod/init/bash" in text
    launcher=RENDER.load()["launcher"]["python"]
    assert Path(launcher).is_absolute()
    assert Path(launcher).is_file()
    assert f"LAUNCHER={launcher}" in text

def test_regression_4974391_clean_environment_uses_frontier_lmod(tmp_path):
    # Job 4974391 had export=NONE: no inherited module function, and Frontier
    # intentionally has no /etc/profile.d/modules.sh compatibility file.
    assert not Path("/etc/profile.d/modules.sh").exists()
    smoke,_=bundles(tmp_path)
    text=(smoke/"rendered.sbatch").read_text()
    bootstrap=next(line for line in text.splitlines() if line.startswith("if ! command -v module"))+"\n"
    result=subprocess.run(
        ["env","-i",f"HOME={Path.home()}",f"USER={Path.home().name}",
         "SHELL=/bin/bash","PATH=/usr/bin:/bin","bash","--noprofile","--norc","-c",
         "set -euo pipefail\n"+bootstrap+"command -v module\n"],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
    )
    assert result.returncode==0,result.stderr
    assert "module" in result.stdout

def test_allocation_free_gate_stops_after_full_presrun_preflight(tmp_path):
    smoke,_=bundles(tmp_path)
    text=(smoke/"rendered.sbatch").read_text()
    before=text.index("--phase before-srun --helper")
    stop=text.index("E97_STOP_BEFORE_SRUN",before)
    launch=text.index("exec srun",stop)
    assert before < stop < launch
    assert "phase=preflight-only status=complete ranks=2048 nodes=256" in text
