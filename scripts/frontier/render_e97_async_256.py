#!/usr/bin/env python3
"""Render job 4962400's launcher without reimplementing its batch body."""
import argparse, base64, hashlib, json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / "configs/frontier/e97_async_256.yaml"

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"))

def load():
    config=json.loads(CONFIG.read_text())
    if set(config)!={"schema_version","golden_manifest","seed","profiles"} or config["schema_version"]!=3:
        raise ValueError("unknown or missing configuration key")
    seed=config["seed"]
    if set(seed)!={"uri","manifest_uri","latest_pointer_uri","step","loss","tokens","size","sha256","provenance"}:
        raise ValueError("unknown or missing seed key")
    if not all(seed[key].startswith("s3://") for key in ("uri","manifest_uri","latest_pointer_uri")):
        raise ValueError("seed authorities must be S3 URIs")
    if "latest" in seed["uri"].lower() or not re.fullmatch(r"[0-9a-f]{64}",seed["sha256"]):
        raise ValueError("dynamic or invalid seed reference")
    from scripts.frontier.materialize_e97_s3_seed import verify_authorities
    verify_authorities(seed)
    if set(config["profiles"])!={"smoke","production"}: raise ValueError("profiles must be smoke and production")
    for name,p in config["profiles"].items():
        if set(p)!={"nodes","walltime","queue"} or set(p["queue"])!={"partition","qos"}: raise ValueError(f"only nodes, walltime, and queue are permitted in {name}")
        if not isinstance(p["nodes"],int) or p["nodes"]<=0: raise ValueError("nodes must be positive")
        if not re.fullmatch(r"\d\d:\d\d:\d\d",p["walltime"]): raise ValueError("invalid walltime")
    golden=json.loads((ROOT/config["golden_manifest"]).read_text())
    for path,want in golden["files"].items():
        got=digest(ROOT/path)
        if got!=want: raise ValueError(f"golden source drift: {path}: {got} != {want}")
    return config,golden

def ensure_clean_tree():
    # Rendering from committed provenance prevents an unreviewed launcher edit.
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=str(ROOT),universal_newlines=True)
    if dirty: raise ValueError("tracked worktree is dirty; commit before rendering")

def render(profile,out):
    config,golden=load(); selected=config["profiles"][profile]; out.mkdir(parents=True,exist_ok=True)
    launcher=(ROOT/golden["script"]).read_bytes()
    # The batch file is byte-for-byte job 4962400. Slurm CLI overrides are the
    # sole parameterization point and do not mutate its proven prologue/body.
    (out/"rendered.sbatch").write_bytes(launcher)
    nodes=selected["nodes"]; ranks=nodes*8
    seed_b64=base64.urlsafe_b64encode(canonical(config["seed"]).encode()).decode()
    export=golden["export"]+",E97_SEED_B64="+seed_b64+",ASYNC_GENERATIONS=1000000,ASYNC_STEPS=40000000"
    export=re.sub(r"ASYNC_TRAINPY_RANKS=\d+",f"ASYNC_TRAINPY_RANKS={ranks}",export)
    export=re.sub(r"ASYNC_EXPECTED_RANKS=\d+",f"ASYNC_EXPECTED_RANKS={ranks}",export)
    export=re.sub(r"ASYNC_GLOBAL_QUORUM=\d+",f"ASYNC_GLOBAL_QUORUM={ranks}",export)
    argv=list(golden["sbatch_fixed_argv"]); argv[argv.index("-N")+1]=str(nodes)
    argv += ["-t",selected["walltime"],"-p",selected["queue"]["partition"],"-q",selected["queue"]["qos"],"--export",export,golden["script"]]
    resolved={**golden["resolved"],"nodes":nodes,"launched_ranks":ranks,"participant_ranks":ranks,"worker_ranks":ranks,"global_quorum":ranks,"local_steps":40,"steps":40000000,"generations":1000000,"seed":config["seed"]}
    launch={"schema_version":2,"profile":profile,"source_job":golden["source_job"],"source_commit":golden["source_commit"],"nodes":nodes,"walltime":selected["walltime"],"queue":selected["queue"],"sbatch_argv":argv,"training_stop_budget":{"local_steps":40,"steps":40000000,"generations":1000000,"timeout_s":1200,"walltime_remaining_s":1200},"resolved":resolved,"seed_materialization":{"scope":"job","atomic_promotion":True,"reject_existing":True,"verify_before_load":["size","sha256"],"legacy_seed_mutation":False},"launcher_sha256":digest(out/"rendered.sbatch")}
    (out/"launch-inputs.json").write_text(json.dumps(launch,sort_keys=True,indent=2)+"\n")
    (out/"golden-manifest.json").write_text(json.dumps(golden,sort_keys=True,indent=2)+"\n")
    normalized={**launch,"profile":"@PROFILE@","nodes":"@NODES@","walltime":"@WALLTIME@","queue":{"partition":"@PARTITION@","qos":"@QOS@"}}
    a=normalized["sbatch_argv"]
    for flag,value in (("-N","@NODES@"),("-t","@WALLTIME@"),("-p","@PARTITION@"),("-q","@QOS@")): a[a.index(flag)+1]=value
    a[a.index("--export")+1]=re.sub(r"(ASYNC_(?:TRAINPY|EXPECTED)_RANKS|ASYNC_GLOBAL_QUORUM)=\d+",r"\1=@DERIVED_RANKS@",a[a.index("--export")+1])
    for key in ("nodes","launched_ranks","participant_ranks","worker_ranks","global_quorum"): normalized["resolved"][key]="@DERIVED@"
    fp=hashlib.sha256((canonical(normalized)+"\n"+digest(out/"golden-manifest.json")).encode()).hexdigest()
    (out/"fingerprint.sha256").write_text(fp+"\n")
    return fp

def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=("smoke","production"),required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--render-only",action="store_true"); a=p.parse_args()
    if not a.render_only: ensure_clean_tree()
    print(render(a.profile,a.out))
if __name__=="__main__": main()
