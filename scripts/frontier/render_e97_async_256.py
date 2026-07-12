#!/usr/bin/env python3
"""Render job 4962400's launcher without reimplementing its batch body."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/frontier/e97_async_256.yaml"

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"))

def load():
    config=json.loads(CONFIG.read_text())
    if set(config)!={"schema_version","golden_manifest","profiles"} or config["schema_version"]!=2:
        raise ValueError("unknown or missing configuration key")
    if set(config["profiles"])!={"smoke","production"}: raise ValueError("profiles must be smoke and production")
    for name,p in config["profiles"].items():
        if set(p)!={"walltime","queue"} or set(p["queue"])!={"partition","qos"}: raise ValueError(f"only walltime and queue are permitted in {name}")
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
    argv=golden["sbatch_fixed_argv"] + ["-t",selected["walltime"],"-p",selected["queue"]["partition"],"-q",selected["queue"]["qos"],"--export",golden["export"],golden["script"]]
    launch={"schema_version":1,"profile":profile,"source_job":golden["source_job"],"source_commit":golden["source_commit"],"walltime":selected["walltime"],"queue":selected["queue"],"sbatch_argv":argv,"training_stop_budget":{"local_steps":40,"steps":40,"generations":1,"timeout_s":1200,"walltime_remaining_s":1200},"resolved":golden["resolved"],"launcher_sha256":digest(out/"rendered.sbatch")}
    (out/"launch-inputs.json").write_text(json.dumps(launch,sort_keys=True,indent=2)+"\n")
    (out/"golden-manifest.json").write_text(json.dumps(golden,sort_keys=True,indent=2)+"\n")
    normalized={**launch,"profile":"@PROFILE@","walltime":"@WALLTIME@","queue":{"partition":"@PARTITION@","qos":"@QOS@"}}
    a=normalized["sbatch_argv"]
    for flag,value in (("-t","@WALLTIME@"),("-p","@PARTITION@"),("-q","@QOS@")): a[a.index(flag)+1]=value
    fp=hashlib.sha256((canonical(normalized)+"\n"+digest(out/"golden-manifest.json")).encode()).hexdigest()
    (out/"fingerprint.sha256").write_text(fp+"\n")
    return fp

def main():
    p=argparse.ArgumentParser(); p.add_argument("--profile",choices=("smoke","production"),required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--render-only",action="store_true"); a=p.parse_args()
    if not a.render_only: ensure_clean_tree()
    print(render(a.profile,a.out))
if __name__=="__main__": main()
