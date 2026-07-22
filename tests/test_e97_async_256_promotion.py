import hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SEED=json.loads((ROOT/"configs/frontier/e97_async_256.yaml").read_text())["seed"]
def module(path):
    spec=importlib.util.spec_from_file_location("e97_render",path); result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result
RENDER=module(ROOT/"scripts/frontier/render_e97_async_256.py")
CHECK=[sys.executable,str(ROOT/"scripts/frontier/check_e97_async_promotion.py")]
POLICY=str(ROOT/"configs/frontier/e97_async_256_parity_policy.json")
def bundles(tmp_path):
    s=tmp_path/"smoke"; p=tmp_path/"production"; RENDER.render("smoke",s); RENDER.render("production",p); return s,p
def captured_run(argv, **kwargs):
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True, **kwargs)
def check(s,p): return captured_run(CHECK+["--smoke",str(s),"--production",str(p),"--policy",POLICY])

def test_job_4962400_launcher_is_byte_exact_and_only_scale_queue_time_differ(tmp_path):
    s,p=bundles(tmp_path); result=check(s,p)
    assert result.returncode==0,result.stderr
    evidence=json.loads(result.stdout)
    assert evidence["allowed_differences"]=={"nodes":[2,256],"ranks":[16,2048],"walltime":["00:20:00","12:00:00"],"partition":["batch","batch"],"qos":["debug","normal"]}
    assert evidence["training_stop_budget"]=={"generations":1000000,"local_steps":40,"steps":40000000,"timeout_s":1200,"walltime_remaining_s":1200}
    assert (s/"rendered.sbatch").read_bytes()==(ROOT/"scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch").read_bytes()==(p/"rendered.sbatch").read_bytes()
    assert json.loads((s/"launch-inputs.json").read_text())["launcher_sha256"]=="106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30"
    for bundle in (s,p):
        launch=json.loads((bundle/"launch-inputs.json").read_text())
        assert launch["resolved"]["seed"]==SEED
        assert launch["resolved"]["generations"]==1000000
        assert launch["resolved"]["steps"]==40000000
        assert launch["resolved"]["local_steps"]==40
        assert launch["resolved"]["steps"]==launch["resolved"]["generations"]*launch["resolved"]["local_steps"]
        assert "E97_SEED_B64=" in launch["sbatch_argv"][launch["sbatch_argv"].index("--export")+1]
        serialized=json.dumps(launch).lower()
        assert "latest.pt" not in serialized
        assert "latest" not in launch["resolved"]["seed"]["uri"].lower()
        assert launch["resolved"]["seed"]["latest_pointer_uri"].endswith("latest_emender_E97_1.3B.json")

MUTATIONS=[
 ("source_commit","0"*40),("launcher_sha256","0"*64),("resolved.account","other"),("resolved.reservation","x"),
 ("sbatch_argv",["sbatch","changed-launcher.sbatch"]),
 ("resolved.nodes",255),("resolved.ranks_per_node",7),("resolved.launched_ranks",256),("resolved.participant_ranks",256),
 ("resolved.worker_ranks",256),("resolved.global_quorum",171),("resolved.local_steps",41),("resolved.steps",40000),
 ("resolved.timeout_s",1),("resolved.walltime_remaining_s",43200),("resolved.checkpoint_interval",1),
 ("resolved.seed.uri","s3://bucket/new.pt"),("resolved.seed.sha256","0"*64),("resolved.seed.step",1),("resolved.seed.loss",1.0),("resolved.seed.tokens",1),("resolved.seed.size",1),("resolved.data","other"),("resolved.model","E97/1.3b"),("resolved.optimizer","adamw"),
 ("resolved.learning_rate","1"),("resolved.batch_size",8),("resolved.chunk_size",1),("resolved.transport","tcp"),
 ("resolved.mpich_gpu_support_enabled","1"),("resolved.signal","USR1"),("resolved.requeue",True),
 ("training_stop_budget.steps",40000),("training_stop_budget.walltime_remaining_s",43200),
]
def mutate(path,dotted,value):
    data=json.loads(path.read_text()); target=data; parts=dotted.split(".")
    for key in parts[:-1]: target=target[key]
    target[parts[-1]]=value; path.write_text(json.dumps(data))
@pytest.mark.parametrize("field,value",MUTATIONS)
def test_every_non_allowlisted_field_fails_closed(tmp_path,field,value):
    s,p=bundles(tmp_path); mutate(p/"launch-inputs.json",field,value); assert check(s,p).returncode!=0

def test_recent_wrapper_drift_reproduced_and_rejected(tmp_path):
    s,p=bundles(tmp_path); launch=p/"launch-inputs.json"
    for field,value in (("resolved.global_quorum",171),("resolved.participant_ranks",256),("resolved.seed.uri","s3://bucket/latest.pt"),("resolved.signal","B:USR1@1200")):
        mutate(launch,field,value)
    # Jobs 4972201/4972494/4974389/4974391/4974444 also introduced export=NONE,
    # spool/helper bootstrap. Any launcher byte drift is independently fatal.
    (p/"rendered.sbatch").write_text((p/"rendered.sbatch").read_text()+"\n#SBATCH --export=NONE\nmodule load rocm\n")
    assert check(s,p).returncode!=0

def test_unknown_profile_key_and_golden_source_drift_fail(tmp_path,monkeypatch):
    config=json.loads(RENDER.CONFIG.read_text()); config["profiles"]["production"]["steps"]=99
    bad=tmp_path/"config.json"; bad.write_text(json.dumps(config)); monkeypatch.setattr(RENDER,"CONFIG",bad)
    with pytest.raises(ValueError,match="only nodes, walltime, and queue"): RENDER.load()

def test_dynamic_seed_reference_is_rejected(tmp_path,monkeypatch):
    config=json.loads(RENDER.CONFIG.read_text()); config["seed"]["uri"]="s3://bucket/latest.pt"
    bad=tmp_path/"config.json"; bad.write_text(json.dumps(config)); monkeypatch.setattr(RENDER,"CONFIG",bad)
    with pytest.raises(ValueError,match="dynamic or invalid seed"): RENDER.load()

def test_modified_parity_policy_fails_closed(tmp_path):
    s,p=bundles(tmp_path); policy=tmp_path/"policy.json"
    data=json.loads(Path(POLICY).read_text()); data["allowed_profile_keys"].append("seed"); policy.write_text(json.dumps(data))
    result=captured_run(CHECK+["--smoke",str(s),"--production",str(p),"--policy",str(policy)])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="policy"

def promotion(s, **overrides):
    # Model a smoke attested to the implementation under test.  Historical
    # evidence remains fail-closed when a pinned runtime source has changed.
    commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,universal_newlines=True
    ).strip()
    terminal={
        "schema_version":1,"task_id":"fix-e97-per","result":"pass",
        "job_id":4979704,"submission_count":1,"nodes":2,"ranks":16,
        "ranks_per_node":8,"partition":"batch","qos":"debug",
        "walltime":"00:20:00","terminal_state":"TIMEOUT",
        "terminal_elapsed":"00:20:01","scheduler_controlled_finalization":True,
        "generations_completed":[0,1,2],"merges_completed":3,
        "accepted_updates_per_generation":16,"aggregate_bytes_per_generation":5506770496,
        "aggregate_bucket_count":80,"max_rss_kib":61159724,"host_oom":False,
        "cross_node_path_failure":False,"latest_checkpoint_step":SEED["step"]+120,
        "latest_checkpoint_generation":2,"latest_checkpoint_reload":"pass",
        "stable_seed_pointer_unchanged":True,"production_submitted":False,
        "larger_smoke_submitted":False,"production_paused":True,
    }
    evidence=s/"terminal-validation.json"
    evidence.write_text(json.dumps(terminal,sort_keys=True)+"\n")
    value={"job_id":4979704,"slurm_state":"TIMEOUT","exit_code":"0:0",
           "origin_commit":commit,"fingerprint":(s/"fingerprint.sha256").read_text().strip(),
           "nodes":2,"ranks":16,"seed":SEED,
           "terminal_evidence":"terminal-validation.json",
           "terminal_evidence_sha256":hashlib.sha256(evidence.read_bytes()).hexdigest()}
    value.update(overrides)
    (s/"promotion.json").write_text(json.dumps(value))

def test_scheduler_controlled_repeated_smoke_is_accepted_for_promotion(tmp_path):
    s,p=bundles(tmp_path); promotion(s)
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    on_main=subprocess.run(
        ["git","merge-base","--is-ancestor","HEAD","origin/main"],cwd=ROOT
    ).returncode==0
    if on_main:
        assert result.returncode==0,result.stderr
    else:
        assert result.returncode!=0
        assert json.loads(result.stderr)["kind"]=="origin_commit"

def test_unknown_or_unpushed_attested_commit_fails(tmp_path):
    s,p=bundles(tmp_path); promotion(s,origin_commit="5"*40)
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="origin_commit"

def test_historical_evidence_is_rejected_after_pinned_helper_changes(tmp_path):
    s,p=bundles(tmp_path)
    promotion(s,origin_commit="d554965461428bd9ff040329812b09d413e9b723")
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="attested_tree_drift"

def test_submit_executes_from_attested_tree_not_current_main(tmp_path,monkeypatch):
    s,p=bundles(tmp_path); promotion(s)
    checker=module(ROOT/"scripts/frontier/check_e97_async_promotion.py")
    approval=tmp_path/"approval.json"
    attempt=tmp_path/"submission-attempt.json"
    approval.write_text(json.dumps({"approved":True,"fingerprint":(s/"fingerprint.sha256").read_text().strip()}))
    attested=tmp_path/"attested"
    monkeypatch.setattr(checker,"validate_promotion",lambda *args:{"attested_commit":"a"*40})
    monkeypatch.setattr(checker,"materialize_attested_tree",lambda *args:attested)
    calls=[]
    monkeypatch.setattr(checker.os,"chdir",lambda path:calls.append(("chdir",path)))
    monkeypatch.setattr(checker.os,"execvp",lambda exe,argv:(_ for _ in ()).throw(RuntimeError((exe,argv))))
    monkeypatch.setattr(sys,"argv",["check","--smoke",str(s),"--production",str(p),"--policy",POLICY,"--submit","--approval",str(approval),"--attempt-marker",str(attempt)])
    with pytest.raises(RuntimeError): checker.main()
    assert calls==[("chdir",attested)]
    assert json.loads(attempt.read_text())["status"]=="sbatch_exec_started"

def test_submit_attempt_marker_rejects_duplicate_before_exec(tmp_path,monkeypatch):
    s,p=bundles(tmp_path); promotion(s)
    checker=module(ROOT/"scripts/frontier/check_e97_async_promotion.py")
    approval=tmp_path/"approval.json"
    approval.write_text(json.dumps({"approved":True,"fingerprint":(s/"fingerprint.sha256").read_text().strip()}))
    attempt=tmp_path/"submission-attempt.json"
    attempt.write_text('{"status":"already-attempted"}\n')
    monkeypatch.setattr(checker,"validate_promotion",lambda *args:{"attested_commit":"a"*40})
    monkeypatch.setattr(checker,"materialize_attested_tree",lambda *args:(_ for _ in ()).throw(AssertionError("must not materialize")))
    monkeypatch.setattr(sys,"argv",["check","--smoke",str(s),"--production",str(p),"--policy",POLICY,"--submit","--approval",str(approval),"--attempt-marker",str(attempt)])
    with pytest.raises(SystemExit): checker.main()

def test_attested_tree_rejects_unverifiable_training_code(tmp_path,monkeypatch):
    checker=module(ROOT/"scripts/frontier/check_e97_async_promotion.py")
    golden={"files":{"train.py":"0"*64}}
    monkeypatch.setattr(checker.subprocess,"check_output",lambda *args,**kwargs:b"train.py\0")
    # Invalid archive (and therefore any unverifiable tree) fails closed.
    with pytest.raises(SystemExit): checker.materialize_attested_tree("a"*40,tmp_path,golden)

@pytest.mark.parametrize("field,value",[
    ("job_id",0),("job_id","4979704"),("slurm_state","COMPLETED"),("exit_code","1:0"),
    ("origin_commit","bad"),("fingerprint","0"*64),("nodes",255),("ranks",2047),
    ("seed",{**SEED,"step":1}),
])
def test_incomplete_or_mismatched_promotion_fails_closed(tmp_path,field,value):
    s,p=bundles(tmp_path); promotion(s,**{field:value})
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    # A review branch is rejected by ancestry before promotion fields are
    # inspected; once merged, the field-level fail-closed check applies.
    assert json.loads(result.stderr)["kind"] in {"origin_commit","promotion"}

@pytest.mark.parametrize("field,value",[
    ("terminal_state","COMPLETED"),("terminal_elapsed","00:01:00"),
    ("scheduler_controlled_finalization",False),("generations_completed",[0]),
    ("merges_completed",1),("latest_checkpoint_step",1525040),
    ("latest_checkpoint_reload","fail"),("stable_seed_pointer_unchanged",False),
    ("host_oom",True),("production_submitted",True),
])
def test_inadequate_terminal_smoke_evidence_fails_closed(tmp_path,field,value):
    s,p=bundles(tmp_path); promotion(s)
    evidence=s/"terminal-validation.json"
    data=json.loads(evidence.read_text()); data[field]=value
    evidence.write_text(json.dumps(data,sort_keys=True)+"\n")
    promo=json.loads((s/"promotion.json").read_text())
    promo["terminal_evidence_sha256"]=hashlib.sha256(evidence.read_bytes()).hexdigest()
    (s/"promotion.json").write_text(json.dumps(promo))
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"] in {"origin_commit","smoke_terminal_evidence"}

def test_actual_complete_proven_batch_prologue_executes_cleanly(tmp_path):
    repo=tmp_path/"repo"; common=repo/"scripts/frontier/trainpy_async_quorum_smoke_common.sh"; common.parent.mkdir(parents=True)
    common.write_text("#!/bin/bash\nset -euo pipefail\nprintf '%s\\n' \"$SMOKE_NAME|$SMOKE_NODE_COUNT|$ASYNC_TRAINPY_RANKS|$ASYNC_GLOBAL_QUORUM|$SCALEOUT_VARIANT\"\n"); common.chmod(0o755)
    env={**os.environ,"REPO":str(repo),"SLURM_JOB_NUM_NODES":"256","SMOKE_NAME":"256n","ASYNC_TRAINPY_RANKS":"2048","ASYNC_EXPECTED_RANKS":"2048","ASYNC_GLOBAL_QUORUM":"2048","SCALEOUT_VARIANT":"E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n"}
    r=captured_run(["bash",str(ROOT/"scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch")],env=env)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=="256n|256|2048|2048|E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n"

def test_proven_sources_match_success_commit():
    golden=json.loads((ROOT/"configs/frontier/e97_async_256_job4962400_golden.json").read_text())
    for path,want in golden["files"].items(): assert RENDER.digest(ROOT/path)==want
    historical=subprocess.check_output(["git","show",golden["source_commit"]+":"+golden["script"]])
    assert historical==(ROOT/golden["script"]).read_bytes()

def test_exact_durable_job_4962400_artifact_paths_and_hashes():
    golden=json.loads((ROOT/"configs/frontier/e97_async_256_job4962400_golden.json").read_text())
    artifacts=golden["source_artifacts"]
    files=artifacts["files"]
    assert len(files)==4
    assert all(Path(path).parent==Path(artifacts["root"])/"artifacts" for path in files)
    assert {Path(path).name for path in files}=={"command.txt","env.txt","manifest.json","metrics.json"}
    if not Path(artifacts["root"]).exists():
        pytest.skip("Frontier durable job evidence mount is unavailable")
    for path,want in files.items():
        assert RENDER.digest(Path(path))==want
