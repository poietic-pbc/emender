import importlib.util, json, os, subprocess, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SEED={"uri":"s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1525000/checkpoint_step_1525000_loss_2.4378.pt","path":"/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt","step":1525000,"loss":2.4378,"tokens":"99.9424B","size":7719679924,"sha256":"1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9"}
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

def test_job_4962400_launcher_is_byte_exact_and_only_queue_time_differ(tmp_path):
    s,p=bundles(tmp_path); result=check(s,p)
    assert result.returncode==0,result.stderr
    evidence=json.loads(result.stdout)
    assert evidence["allowed_differences"]=={"walltime":["00:20:00","12:00:00"],"partition":["batch","batch"],"qos":["debug","normal"]}
    assert evidence["training_stop_budget"]=={"generations":1,"local_steps":40,"steps":40,"timeout_s":1200,"walltime_remaining_s":1200}
    assert (s/"rendered.sbatch").read_bytes()==(ROOT/"scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch").read_bytes()==(p/"rendered.sbatch").read_bytes()
    assert json.loads((s/"launch-inputs.json").read_text())["launcher_sha256"]=="106a4dde6b966b0af66a1ac92ea0f459c7a435f81f6e322d92e08f30a2cfad30"
    for bundle in (s,p):
        launch=json.loads((bundle/"launch-inputs.json").read_text())
        assert launch["resolved"]["seed"]==SEED
        assert "E97_CHECKPOINT="+SEED["path"] in launch["sbatch_argv"][launch["sbatch_argv"].index("--export")+1]
        serialized=json.dumps(launch).lower()
        assert "latest.pt" not in serialized
        assert "latest_emender" not in serialized

MUTATIONS=[
 ("source_commit","0"*40),("launcher_sha256","0"*64),("resolved.account","other"),("resolved.reservation","x"),
 ("resolved.nodes",255),("resolved.ranks_per_node",7),("resolved.launched_ranks",256),("resolved.participant_ranks",256),
 ("resolved.worker_ranks",256),("resolved.global_quorum",171),("resolved.local_steps",41),("resolved.steps",40000),
 ("resolved.timeout_s",1),("resolved.walltime_remaining_s",43200),("resolved.checkpoint_interval",1),
 ("resolved.seed.path","new.pt"),("resolved.seed.sha256","0"*64),("resolved.seed.step",1),("resolved.seed.loss",1.0),("resolved.seed.tokens","1B"),("resolved.seed.size",1),("resolved.data","other"),("resolved.model","E97/1.3b"),("resolved.optimizer","adamw"),
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
    for field,value in (("resolved.global_quorum",171),("resolved.participant_ranks",256),("resolved.seed.path","step1282500/latest.pt"),("resolved.signal","B:USR1@1200")):
        mutate(launch,field,value)
    # Jobs 4972201/4972494/4974389/4974391/4974444 also introduced export=NONE,
    # spool/helper bootstrap. Any launcher byte drift is independently fatal.
    (p/"rendered.sbatch").write_text((p/"rendered.sbatch").read_text()+"\n#SBATCH --export=NONE\nmodule load rocm\n")
    assert check(s,p).returncode!=0

def test_unknown_profile_key_and_golden_source_drift_fail(tmp_path,monkeypatch):
    config=json.loads(RENDER.CONFIG.read_text()); config["profiles"]["production"]["steps"]=99
    bad=tmp_path/"config.json"; bad.write_text(json.dumps(config)); monkeypatch.setattr(RENDER,"CONFIG",bad)
    with pytest.raises(ValueError,match="only walltime and queue"): RENDER.load()

def test_dynamic_seed_reference_is_rejected(tmp_path,monkeypatch):
    config=json.loads(RENDER.CONFIG.read_text()); config["seed"]["path"]="/tmp/latest.pt"
    bad=tmp_path/"config.json"; bad.write_text(json.dumps(config)); monkeypatch.setattr(RENDER,"CONFIG",bad)
    with pytest.raises(ValueError,match="dynamic or invalid seed"): RENDER.load()

def test_modified_parity_policy_fails_closed(tmp_path):
    s,p=bundles(tmp_path); policy=tmp_path/"policy.json"
    data=json.loads(Path(POLICY).read_text()); data["allowed_profile_keys"].append("seed"); policy.write_text(json.dumps(data))
    result=captured_run(CHECK+["--smoke",str(s),"--production",str(p),"--policy",str(policy)])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="policy"

def promotion(s, **overrides):
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),universal_newlines=True).strip()
    value={"job_id":4975667,"slurm_state":"COMPLETED","exit_code":"0:0",
           "origin_commit":commit,"fingerprint":(s/"fingerprint.sha256").read_text().strip(),
           "nodes":256,"ranks":2048,"seed":SEED}
    value.update(overrides)
    (s/"promotion.json").write_text(json.dumps(value))

def test_new_successful_smoke_job_is_accepted_for_promotion(tmp_path):
    s,p=bundles(tmp_path); promotion(s)
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode==0,result.stderr

def test_promotion_commit_must_equal_head_and_origin_main(tmp_path):
    s,p=bundles(tmp_path); promotion(s,origin_commit="5"*40)
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="origin_commit"

@pytest.mark.parametrize("field,value",[
    ("job_id",0),("job_id","4975667"),("slurm_state","RUNNING"),("exit_code","1:0"),
    ("origin_commit","bad"),("fingerprint","0"*64),("nodes",255),("ranks",2047),
    ("seed",{**SEED,"step":1}),
])
def test_incomplete_or_mismatched_promotion_fails_closed(tmp_path,field,value):
    s,p=bundles(tmp_path); promotion(s,**{field:value})
    result=captured_run(CHECK+['--smoke',str(s),'--production',str(p),'--policy',POLICY,'--require-promotion'])
    assert result.returncode!=0
    assert json.loads(result.stderr)["kind"]=="promotion"

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
