#!/usr/bin/env python3
"""Render the sole canonical E97 256-node launch bundle."""
import argparse, hashlib, json, os, re, shlex, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/frontier/e97_async_256.yaml"
FILES = ("runtime.json", "inputs.json", "code.json", "runtime-manifest.json", "helper-manifest.json")

def canonical(x): return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
def digest_file(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def manifest(paths):
    return [{"path":p,"sha256":digest_file(ROOT/p),"size":(ROOT/p).stat().st_size} for p in paths]
def seconds(w):
    a=[int(x) for x in w.split(":")]
    if len(a)!=3: raise ValueError("walltime must be HH:MM:SS")
    return a[0]*3600+a[1]*60+a[2]
def load():
    data=json.loads(CONFIG.read_text())
    allowed={"walltime","trainer_duration_seconds","queue"}; qallowed={"partition","qos"}
    if set(data)!= {"schema_version","provenance","scheduler","launcher","topology","environment","modules","runtime_files","helper_files","inputs","trainer","transport","checkpoint","profiles"}: raise ValueError("unknown or missing base key")
    if set(data["profiles"])!={"smoke","production"}: raise ValueError("profiles must be smoke and production")
    for name,p in data["profiles"].items():
        if set(p)!=allowed or set(p["queue"])!=qallowed: raise ValueError(f"unknown or missing profile key in {name}")
        if p["trainer_duration_seconds"]!=seconds(p["walltime"]): raise ValueError("trainer duration is not derived from walltime")
    if re.search(r"\$\{[^}]+\}", canonical(data)): raise ValueError("unresolved variable")
    s=data["scheduler"]; t=data["topology"]
    if s["reservation"] is not None or (s["nodes"],s["ranks_per_node"],s["gpus_per_rank"])!=(256,8,1): raise ValueError("invalid scheduler topology/reservation")
    if [t[x] for x in ("launched_ranks","participant_ranks","worker_ranks","global_quorum")] != [2048]*4 or t["rank_ids"]!={"start":0,"stop":2047}: raise ValueError("invalid exact-rank topology")
    return data
def ensure_clean_tree():
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=str(ROOT),universal_newlines=True)
    if dirty: raise ValueError("tracked worktree is dirty; commit or restore it before rendering")
def trainer_argv(c, duration):
    t=c["trainer"]; z=c["topology"]; i=c["inputs"]; ck=c["checkpoint"]; x=c["transport"]
    a=[c["launcher"]["python"],"-u",c["launcher"]["entrypoint"],"--run-id","@RUN_ID@","--run-dir","@RUN_DIR@","--metrics-json","@METRICS@","--data",i["data"]["path"],"--checkpoint",i["seed"]["path"],"--tokenizer",t["tokenizer"],"--worker-count",str(z["worker_ranks"]),"--node-count",str(z["participant_ranks"]),"--node-rank","@SLURM_PROCID@","--global-quorum",str(z["global_quorum"]),"--generations",str(t["generations"]),"--local-steps",str(t["local_steps"]),"--steps",str(t["local_steps"]),"--timeout-s",str(t["timeout_s"]),"--diloco-quorum-mode",t["quorum_mode"]]
    flags={"level":"--level","params":"--params","batch_size":"--batch-size","chunk_size":"--chunk-size","lr":"--lr","optimizer":"--optimizer","weight_decay":"--weight-decay","warmup_steps":"--warmup-steps","min_lr_frac":"--min-lr-frac","grad_accum":"--grad-accum","grad_clip":"--grad-clip","dim":"--dim","depth":"--depth","n_heads":"--n-heads","n_state":"--n-state","n_groups":"--n-groups","n_slots":"--n-slots","expansion":"--expansion","state_expansion":"--state-expansion","gate_activation":"--gate-activation","linear_state":"--linear-state","mlp_ratio":"--mlp-ratio","mlp_multiple":"--mlp-multiple","e97_chunk_size":"--e97-chunk-size","checkpoint_interval":"--checkpoint-interval","projection_chunk_size":"--projection-chunk-size","loss_chunk_size":"--loss-chunk-size"}
    for k,f in flags.items(): a += [f,str(t[k]).lower()]
    for k,f in (("recovery_every_generations","--recovery-every-generations"),("recovery_every_seconds","--recovery-every-seconds"),("export_every_generations","--export-every-generations"),("export_every_seconds","--export-every-seconds"),("finalization_reserve_seconds","--finalization-reserve-seconds")): a += [f,str(ck[k])]
    a += ["--walltime-remaining-s",str(duration),"--coordinator-host","@HOST@","--coordinator-bind-host","0.0.0.0","--coordinator-port","@PORT@","--mpi-dense-bucket-bytes",str(x["bucket_bytes"]),"--compiled-mpich-helper-bin","@HELPER@","--compiled-mpich-ipc-dir","@IPC@","--actual-multinode-compiled-mpich-quorum","--device","cuda:0"]
    if t["bf16"]: a.append("--bf16")
    return a
def render(profile,out):
    c=load(); p=c["profiles"][profile]; out=out.resolve(); out.mkdir(parents=True,exist_ok=True)
    rtman=manifest(c["runtime_files"]); hpman=manifest(c["helper_files"])
    entry=ROOT/c["launcher"]["entrypoint"]
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=str(ROOT),universal_newlines=True).strip()
    tree=subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=str(ROOT),universal_newlines=True).strip()
    code={"git_commit":head,"git_tree":tree,"clean_tree_required":True,"entrypoint":{"path":c["launcher"]["entrypoint"],"sha256":digest_file(entry),"size":entry.stat().st_size}}
    inputs={}
    for k,v in c["inputs"].items():
        st=Path(v["path"]).stat(); inputs[k]={**v,"realpath":os.path.realpath(v["path"]),"size":st.st_size}
        if v["hash_kind"]=="sha256-stat-v1": inputs[k].update({"mtime_ns":st.st_mtime_ns,"inode":st.st_ino})
    if any(not re.fullmatch(r"[0-9a-f]{64}",v["sha256"]) for v in inputs.values()): raise ValueError("every input requires a SHA256 content hash")
    argv=trainer_argv(c,p["trainer_duration_seconds"])
    s=c["scheduler"]; total=s["nodes"]*s["ranks_per_node"]
    srun=["srun","-N",str(s["nodes"]),"-n",str(total),"--ntasks-per-node",str(s["ranks_per_node"]),"-c",str(s["cpus_per_rank"]),"--gpus-per-task",str(s["gpus_per_rank"]),"--gpu-bind",s["gpu_bind"]]
    runtime={"schema_version":1,"profile":profile,"trainer_duration_seconds":p["trainer_duration_seconds"],"trainer_argv":argv,"srun_argv":srun,"environment":dict(sorted(c["environment"].items())),"modules":c["modules"],"topology":c["topology"],"transport":c["transport"],"trainer":c["trainer"],"checkpoint":c["checkpoint"],"derived_metadata":["profile","trainer_duration_seconds","run_id","run_dir","metrics","host","port","helper","ipc"]}
    payload="exec "+" ".join(shlex.quote(x) for x in argv)
    replacements={"@SLURM_PROCID@":"${SLURM_PROCID:?missing SLURM_PROCID}","@RUN_ID@":"${RUN_ID}","@RUN_DIR@":"${RUN_DIR}","@METRICS@":"${METRICS}","@HOST@":"${COORDINATOR_HOST}","@PORT@":"${COORDINATOR_PORT}","@HELPER@":"${HELPER}","@IPC@":"${IPC}"}
    for old,new in replacements.items(): payload=payload.replace(old,new)
    module_commands=['module purge']+["module load "+shlex.quote(x) for x in c["modules"]]
    environment_commands=["export "+name+"="+shlex.quote(value) for name,value in sorted(c["environment"].items())]
    preflight='"$LAUNCHER" "$REPO/scripts/frontier/e97_async_256_preflight.py" --bundle "$BUNDLE_DIR" --repo "$REPO" --fingerprint "$BUNDLE_FINGERPRINT"'
    diagnostics=['E97_PHASE=bootstrap','e97_phase() { E97_PHASE="$1"; printf \'e97-presrun phase=%s status=begin fingerprint=%s\\n\' "$E97_PHASE" "$BUNDLE_FINGERPRINT" >&2; }','e97_error() { local rc=$?; trap - ERR; printf \'e97-presrun phase=%s status=error rc=%s line=%s command=%q\\n\' "$E97_PHASE" "$rc" "${BASH_LINENO[0]}" "$BASH_COMMAND" >&2; exit "$rc"; }',"trap e97_error ERR",'printf \'e97-presrun phase=bootstrap status=begin job_id=%s host=%s export=NONE\\n\' "${SLURM_JOB_ID:-unset}" "$(hostname)" >&2']
    module_bootstrap='if ! command -v module >/dev/null 2>&1; then test -r /etc/bash.bashrc.local || { echo "Cray PE initialization is unavailable" >&2; exit 72; }; set +u; source /etc/bash.bashrc.local; set -u; test -r /etc/profile.d/olcf-env.sh || { echo "OLCF module site initialization is unavailable" >&2; exit 72; }; source /etc/profile.d/olcf-env.sh || command -v module >/dev/null 2>&1; fi'
    lines=["#!/bin/bash",f"#SBATCH --account={s['account']}","#SBATCH --job-name=e97-async-256-@PROFILE@",f"#SBATCH --partition={p['queue']['partition']}",f"#SBATCH --qos={p['queue']['qos']}",f"#SBATCH --nodes={s['nodes']}",f"#SBATCH --time={p['walltime']}",f"#SBATCH --ntasks-per-node={s['ranks_per_node']}",f"#SBATCH --cpus-per-task={s['cpus_per_rank']}",f"#SBATCH --gpus-per-task={s['gpus_per_rank']}",f"#SBATCH --gpu-bind={s['gpu_bind']}",f"#SBATCH --export={s['export']}",f"#SBATCH --signal={s['signal']}","#SBATCH --output=e97-async-256-@PROFILE@-%j.out","#SBATCH --error=e97-async-256-@PROFILE@-%j.err","","set -Eeuo pipefail",f"BUNDLE_FINGERPRINT='{('@FINGERPRINT@')}'"]+diagnostics+["BUNDLE_MANIFEST_SHA256='@BUNDLE_MANIFEST@'",f"BUNDLE_DIR={shlex.quote(str(out))}",f"REPO={shlex.quote(str(ROOT))}",f"LAUNCHER={shlex.quote(str(ROOT/c['launcher']['python']))}",'e97_phase bundle-binding','test -d "$BUNDLE_DIR" || { echo "immutable bundle is unavailable: $BUNDLE_DIR" >&2; exit 72; }','test -r "$BUNDLE_DIR/bundle-files.sha256" || { echo "bundle manifest is unavailable" >&2; exit 72; }','test "$(sha256sum "$BUNDLE_DIR/bundle-files.sha256" | cut -d\' \' -f1)" = "$BUNDLE_MANIFEST_SHA256" || { echo "bundle manifest hash mismatch" >&2; exit 72; }','e97_phase module-bootstrap',module_bootstrap,'command -v module >/dev/null 2>&1 || { echo "module command is unavailable after initialization" >&2; exit 72; }']+module_commands+environment_commands+['command -v CC >/dev/null 2>&1 || { echo "declared modules did not provide CC" >&2; exit 72; }','test -x "$LAUNCHER" || { echo "declared launcher is unavailable: $LAUNCHER" >&2; exit 72; }','e97_phase runtime-paths','RUN_ID="e97-256-${SLURM_JOB_ID:?}"','RUN_DIR="${SLURM_SUBMIT_DIR:?}/e97-256-runs/${SLURM_JOB_ID}"','METRICS="${RUN_DIR}/metrics.json"','HELPER="${RUN_DIR}/compiled_mpich_dense_helper"','IPC="${TMPDIR:-/tmp}/e97-256-${SLURM_JOB_ID}/ipc"','COORDINATOR_HOST=$(scontrol show hostnames "${SLURM_NODELIST:?}" | head -1)','COORDINATOR_PORT=29497','export RUN_ID RUN_DIR METRICS HELPER IPC COORDINATOR_HOST COORDINATOR_PORT','mkdir -p "$RUN_DIR" "$IPC"','cd "$REPO"','e97_phase preflight-before-helper',preflight+' --phase before-helper','e97_phase helper-build','ARTIFACT_DIR="$RUN_DIR" OUT="$HELPER" scripts/frontier/build_compiled_mpich_dense_helper.sh','e97_phase preflight-before-srun',preflight+' --phase before-srun --helper "$HELPER"','if [[ "${E97_STOP_BEFORE_SRUN:-0}" == 1 ]]; then printf \'e97-presrun phase=preflight-only status=complete ranks=2048 nodes=256\\n\' >&2; exit 0; fi','e97_phase srun','printf \'e97-presrun phase=srun status=exec ranks=2048 nodes=256\\n\' >&2','exec '+" ".join(shlex.quote(x) for x in srun+["bash","-lc",payload]),""]
    artifacts={"runtime.json":runtime,"inputs.json":inputs,"code.json":code,"runtime-manifest.json":rtman,"helper-manifest.json":hpman}
    for n,v in artifacts.items(): (out/n).write_text(json.dumps(v,sort_keys=True,indent=2)+"\n")
    (out/"rendered.sbatch").write_text("\n".join(lines).replace("@PROFILE@",profile))
    normscript="\n".join(lines).replace(f"BUNDLE_DIR={shlex.quote(str(out))}","BUNDLE_DIR=@BUNDLE_DIR@").replace(p["walltime"],"@WALLTIME@").replace(p["queue"]["partition"],"@PARTITION@").replace(p["queue"]["qos"],"@QOS@").replace("--walltime-remaining-s "+str(p["trainer_duration_seconds"]),"--walltime-remaining-s @DURATION@")
    norm={"script":normscript,"artifacts":artifacts}
    norm["artifacts"]["runtime.json"]={**runtime,"profile":"@PROFILE@","trainer_duration_seconds":"@DURATION@","trainer_argv":["@DURATION@" if x==str(p["trainer_duration_seconds"]) and argv[i-1]=="--walltime-remaining-s" else x for i,x in enumerate(argv)]}
    fp=hashlib.sha256(canonical(norm).encode()).hexdigest(); (out/"fingerprint.sha256").write_text(fp+"\n")
    (out/"fingerprint-file.sha256").write_text(digest_file(out/"fingerprint.sha256")+"\n")
    bundle_names=FILES+("fingerprint.sha256","fingerprint-file.sha256")
    (out/"bundle-files.sha256").write_text("".join(f"{digest_file(out/name)}  {name}\n" for name in bundle_names))
    (out/"rendered.sbatch").write_text((out/"rendered.sbatch").read_text().replace("@FINGERPRINT@",fp).replace("@BUNDLE_MANIFEST@",digest_file(out/"bundle-files.sha256")))
    return fp
def main():
    a=argparse.ArgumentParser(); a.add_argument("--profile",required=True,choices=["smoke","production"]); a.add_argument("--out",required=True,type=Path); a.add_argument("--render-only",action="store_true"); x=a.parse_args(); ensure_clean_tree(); print(render(x.profile,x.out))
if __name__=="__main__": main()
