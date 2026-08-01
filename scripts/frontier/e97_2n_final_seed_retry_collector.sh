#!/bin/bash
#SBATCH -A bif148
#SBATCH -J collect-e97-final-seed-2n
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -t 00:10:00
set -euo pipefail
: "${PAYLOAD_JOB_ID:?}" "${RUN_DIR:?}" "${PAYLOAD_ROOT:?}" "${REPO:?}" "${SUBMISSION_RECORD:?}"
COLLECTOR_ROOT=${COLLECTOR_ROOT:-$(dirname "$(dirname "$RUN_DIR")")/collectors/${SLURM_JOB_ID}/payload-${PAYLOAD_JOB_ID}}
mkdir -p "$COLLECTOR_ROOT" "$RUN_DIR/terminal"
cp --reflink=auto "$SUBMISSION_RECORD" "$COLLECTOR_ROOT/submission.json"
sacct -j "$PAYLOAD_JOB_ID,$SLURM_JOB_ID" -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed \
  > "$COLLECTOR_ROOT/sacct.txt"
cp "$COLLECTOR_ROOT/sacct.txt" "$RUN_DIR/terminal/sacct.txt"
scontrol show job -dd "$PAYLOAD_JOB_ID" > "$COLLECTOR_ROOT/scontrol-payload.txt" || true
find "$RUN_DIR" \( -type f -o -type l \) | sort > "$COLLECTOR_ROOT/artifact-paths.txt" || true

source "$REPO/scripts/frontier/activate_emender_frontier.sh"
PYTHON_BIN=$EMENDER_PYTHON
export PYTHON_BIN
"$EMENDER_PYTHON" - "$RUN_DIR" "$COLLECTOR_ROOT" "$PAYLOAD_JOB_ID" <<'PY'
import json, math, os, pathlib, re, sys
run = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2]); job = sys.argv[3]
state = run / "supervisor" / "final-seed-retry"
errors = []
def need(condition, message):
    if not condition: errors.append(message)
def text(path):
    try: return pathlib.Path(path).read_text(errors="replace")
    except OSError: return ""
def checkpoint_step(path):
    m = re.search(r"checkpoint_step_(\d+)_", path)
    return int(m.group(1)) if m else None

submission = json.load(open(run / "identity" / "submission.json"))
held = text(run / "identity" / "squeue-held.txt")
live = text(state / "squeue-live.txt")
after = text(state / "squeue-allocation-after-fault.txt")
for label, value, running in (("held", held, False), ("live", live, True), ("after-fault", after, True)):
    pattern = rf"^{re.escape(job)}\|" + ("RUNNING" if running else "[^|]+") + r"\|2\|[^|]*\|batch\|debug\|"
    need(re.search(pattern, value, re.M) is not None,
         f"{label} scheduler evidence lacks Nodes=2 Partition=batch QOS=debug")

sacct = text(out / "sacct.txt")
top = None
for line in sacct.splitlines()[1:]:
    fields = line.split("|")
    if len(fields) >= 10 and fields[0] == job: top = fields
need(top is not None, "terminal payload accounting missing")
if top:
    need(top[2] == "COMPLETED" and top[3] == "0:0", "payload did not complete 0:0")
    need(top[5] == "2" and top[8] == "batch" and top[9] == "debug",
         "terminal accounting lacks Nodes=2 Partition=batch QOS=debug")

rows = []
for line in text(run / "supervisor" / "execution-epochs.tsv").splitlines():
    fields = line.split("|")
    if len(fields) == 7:
        rows.append({"epoch": int(fields[0]), "rc": int(fields[1]),
                     "nodes": int(fields[2]), "tasks": int(fields[3]),
                     "port": int(fields[4]), "promoted": int(fields[5]),
                     "step": int(fields[6])})
need(len(rows) == 2, f"expected two execution epochs, found {len(rows)}")
if len(rows) == 2:
    need(rows[0]["rc"] != 0 and rows[1]["rc"] == 0, "fault/retry return codes invalid")
    need(all((r["nodes"], r["tasks"]) == (2, 16) for r in rows),
         "retry did not use the same two-node 16-rank world")
    need(rows[0]["port"] != rows[1]["port"], "fresh child reused MASTER_PORT")
    need(rows[0]["step"] > 2300930 and rows[1]["step"] > rows[0]["step"],
         "post-retry checkpoint did not advance")

first_env = text(run / "epochs" / "epoch-000001" / "launch.env")
second_env = text(run / "epochs" / "epoch-000002" / "launch.env")
first = text(run / "epochs" / "epoch-000001" / "train.out")
second = text(run / "epochs" / "epoch-000002" / "train.out")
for env in (first_env, second_env):
    need("nodes=2\ntasks=16\n" in env, "execution epoch was not two nodes/16 ranks")
    need("diloco_k=40" in env and "diloco_merge_bucket_numel=67108864" in env,
         "K40/64M production settings missing")
    need("final_seed_step=2300930" in env and "final_seed_tokens=150793748480" in env
         and "final_seed_size=7719680116" in env
         and "final_seed_sha256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2" in env,
         "exact final seed identity missing")
need(re.search(r"resume_target=/tmp/emender-e97-seed-\d+/checkpoint-step-2300930.pt", first_env) is not None,
     "first child did not use the verified job-local final seed")
need(first_env.split("nodelist=",1)[1].splitlines()[0] == second_env.split("nodelist=",1)[1].splitlines()[0],
     "same allocation node set was not retried")
markers = re.findall(r"DILOCO_FAULT_INJECTION (\{[^\n]+\})", first)
need(len(markers) == 1, f"expected one exact fault marker, found {len(markers)}")
need("DILOCO_FAULT_INJECTION" not in second, "fault injection repeated")
need("fault_environment_removed=true" in text(state / "retry-environment.txt")
     and "unchanged_failed_payload_retried=false" in text(state / "retry-environment.txt"),
     "retry payload-change evidence missing")

latest_after = text(state / "latest-after-fault.txt").strip()
baseline_step = checkpoint_step(latest_after)
need(baseline_step is not None and baseline_step > 2300930,
     "no synchronous run checkpoint was authoritative before fault")
files_after = text(state / "train-files-after-fault.tsv")
need(".tmp|" not in files_after, "failed child left a partial checkpoint")
need(max([int(v) for v in re.findall(r"checkpoint_step_(\d+)_", files_after)] or [0]) == baseline_step,
     "failed child published a newer checkpoint")
policy = text(run / "supervisor" / "node-policy.tsv")
need(re.search(r"^1\|-\|ambiguous\|no-strike\|-", policy, re.M) is not None,
     "ambiguous collective failure recorded a node strike")
health = text(state / "node-health-after-fault.txt")
need(len(text(state / "allocation-hostnames-after-fault.txt").splitlines()) == 2,
     "both allocation nodes were not observed after failure")
need(not re.search(r"State=[^ ]*(DOWN|DRAIN|FAIL|NO_RESPOND)", health),
     "Slurm reported an unhealthy node after ambiguous failure")

need(first.count("Resumed at step 2300930") >= 16,
     "all 16 ranks did not report loading step 2300930")
need(baseline_step is not None and second.count(f"Resumed at step {baseline_step}") >= 16,
     "fresh 16-rank child did not reload the committed run checkpoint")
need(first.count(">>> [DiLoCo] merge #") >= 5, "pre-fault K40 merges missing")
need(second.count(">>> [DiLoCo] merge #") >= 2, "post-retry K40 progress missing")
metrics = []
for match in re.finditer(r"step\s+(\d+) \| loss ([^ ]+) .*?global_tok/s ([^ ]+)", first + "\n" + second):
    try: metrics.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    except ValueError: pass
finite = [m for m in metrics if math.isfinite(m[1]) and math.isfinite(m[2]) and m[2] > 0]
need(bool(finite), "finite loss/throughput evidence missing")

seed_manifests = list((run / "seed-materialization" / f"job-{job}").glob("*.json"))
need(len(seed_manifests) == 2, f"expected two independent node seed manifests, found {len(seed_manifests)}")
for manifest in seed_manifests:
    value = json.load(open(manifest))
    need(value.get("node_checkpoint_size") == 7719680116
         and value.get("node_checkpoint_sha256") == "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
         and value.get("network_fetches") == 0,
         f"offline seed manifest mismatch: {manifest.name}")

fields = {
    "schema": "e97-final-seed-same-set-retry-2n-v1",
    "job_id": job, "source_sha": submission.get("source_sha"),
    "payload_digest": submission.get("payload_digest"),
    "nodes": 2, "world_size": 16, "partition": "batch", "qos": "debug",
    "seed_step": 2300930, "seed_accepted_tokens": 150793748480,
    "seed_bytes": 7719680116,
    "seed_sha256": "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2",
    "same_node_set_retried": len(rows) == 2 and all((r["nodes"],r["tasks"])==(2,16) for r in rows)
        and not any("node set was not retried" in e for e in errors),
    "checkpoint_reloaded": baseline_step is not None and second.count(f"Resumed at step {baseline_step}") >= 16,
    "post_retry_checkpoint_advanced": len(rows) == 2 and rows[1]["step"] > rows[0]["step"],
    "ambiguous_failure_no_node_strike": "ambiguous|no-strike" in policy,
    "unchanged_failed_payload_retried": False,
    "requirements": ["R07", "R12", "R14", "R16", "NDP13", "NDP15-atomic-only", "NDP17-retired"],
    "errors": errors,
}
fields["full_pass"] = not errors and all(fields[k] is True for k in (
    "same_node_set_retried", "checkpoint_reloaded", "post_retry_checkpoint_advanced",
    "ambiguous_failure_no_node_strike"))
for path in (run / "verdict.json", out / "verdict.json"):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(fields, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
print(json.dumps(fields, sort_keys=True))
if not fields["full_pass"]: raise SystemExit(1)
PY
rc=$?
sha256sum "$COLLECTOR_ROOT"/* > "$COLLECTOR_ROOT/SHA256SUMS" || true
exit "$rc"
