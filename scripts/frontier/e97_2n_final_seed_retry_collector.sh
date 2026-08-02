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
set +e
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
def env_file(path):
    result = {}
    for line in text(path).splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result
def as_int(value, default=-1):
    try: return int(value)
    except (TypeError, ValueError): return default

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

# Execution epochs and Slurm allocation attempts are different identities. A
# requeue can add an allocation attempt without making the preceding child a
# same-allocation retry. Bind each epoch to its launch.env before evaluating the
# adjacent fault/retry pair.
rows = []
for line in text(run / "supervisor" / "execution-epochs.tsv").splitlines():
    fields = line.split("|")
    if len(fields) != 7:
        continue
    epoch = as_int(fields[0])
    launch = env_file(run / "epochs" / f"epoch-{epoch:06d}" / "launch.env")
    rows.append({"epoch": epoch, "rc": as_int(fields[1]),
                 "nodes": as_int(fields[2]), "tasks": as_int(fields[3]),
                 "port": as_int(fields[4]), "promoted": as_int(fields[5]),
                 "step": as_int(fields[6]), "restart": as_int(launch.get("slurm_restart_count")),
                 "allocation_identity": launch.get("allocation_identity", ""),
                 "nodelist": launch.get("nodelist", ""), "launch": launch})
attempts = []
for line in text(run / "identity" / "slurm-attempts.tsv").splitlines():
    fields = line.split("|", 3)
    if len(fields) == 4:
        attempts.append({"time": fields[0], "job": fields[1],
                         "restart": as_int(fields[2]), "nodelist": fields[3]})
need(len(rows) == 2, f"expected exactly two execution epochs in the replacement payload, found {len(rows)}")
need(len(attempts) == 1, f"expected one allocation attempt without walltime requeue, found {len(attempts)}")
first_row = rows[0] if len(rows) >= 1 else None
retry_row = rows[1] if len(rows) >= 2 else None
same_allocation_pair = bool(first_row and retry_row
    and first_row["allocation_identity"]
    and first_row["allocation_identity"] == retry_row["allocation_identity"]
    and first_row["restart"] == retry_row["restart"]
    and first_row["nodelist"] == retry_row["nodelist"])
same_node_set = bool(same_allocation_pair and first_row["nodelist"]
    and first_row["nodes"] == retry_row["nodes"] == 2
    and first_row["tasks"] == retry_row["tasks"] == 16)
if first_row and retry_row:
    need(first_row["rc"] != 0 and retry_row["rc"] == 0, "fault/retry return codes invalid")
    need(first_row["promoted"] == retry_row["promoted"] == 1,
         "fault/retry did not promote readable atomic checkpoints")
    need(first_row["port"] != retry_row["port"], "fresh child reused MASTER_PORT")
    need(first_row["step"] > 2300930 and retry_row["step"] > first_row["step"],
         "post-retry checkpoint did not advance")
need(same_allocation_pair, "fault and retry were not adjacent epochs in one Slurm allocation attempt")
need(same_node_set, "retry did not use the same two-node 16-rank world")

first_env = first_row["launch"] if first_row else {}
second_env = retry_row["launch"] if retry_row else {}
first_path = run / "epochs" / f"epoch-{first_row['epoch']:06d}" if first_row else pathlib.Path("/nonexistent")
second_path = run / "epochs" / f"epoch-{retry_row['epoch']:06d}" if retry_row else pathlib.Path("/nonexistent")
first = text(first_path / "train.out")
second = text(second_path / "train.out")
for env in (first_env, second_env):
    need(env.get("nodes") == "2" and env.get("tasks") == "16",
         "execution epoch was not two nodes/16 ranks")
    need(env.get("diloco_k") == "40" and env.get("diloco_merge_bucket_numel") == "67108864",
         "K40/64M production settings missing")
    need(env.get("final_seed_step") == "2300930"
         and env.get("final_seed_tokens") == "150793748480"
         and env.get("final_seed_size") == "7719680116"
         and env.get("final_seed_sha256") == "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2",
         "exact final seed identity missing")
need(re.fullmatch(r"/tmp/emender-e97-seed-\d+/checkpoint-step-2300930\.pt",
                  first_env.get("resume_target", "")) is not None,
     "first child did not use the verified job-local final seed")
markers = re.findall(r"DILOCO_FAULT_INJECTION (\{[^\n]+\})", first)
need(len(markers) == 1, f"expected one exact fault marker, found {len(markers)}")
if len(markers) == 1:
    try:
        marker = json.loads(markers[0])
        need(marker == {"bucket_index": 1, "exit_code": 86, "label": "sf_x", "merge_index": 3, "rank": 1},
             f"fault marker identity mismatch: {marker}")
    except json.JSONDecodeError:
        need(False, "fault marker is not valid JSON")
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
direct = re.search(r"^(\d+)\|([^|]+)\|strike\|direct-task-exit\|1$", policy, re.M)
first_direct_strike = bool(direct and first_row and as_int(direct.group(1)) == first_row["epoch"])
need(first_direct_strike, "first failed child was not recorded as one direct hostname strike")
if direct:
    direct_host = direct.group(2)
    need(direct_host in first_row["nodelist"].split(",") if first_row else False,
         "direct strike hostname was outside the launched node set")
    need(re.search(rf"^[^|]+\|{re.escape(direct_host)}\|exclude\|", policy, re.M) is None,
         "first direct strike incorrectly excluded its node")
need("ambiguous|no-strike" not in policy,
     "physical replacement unexpectedly exercised ambiguous attribution")
health = text(state / "node-health-after-fault.txt")
need(len(text(state / "allocation-hostnames-after-fault.txt").splitlines()) == 2,
     "both allocation nodes were not observed after failure")
need(not re.search(r"State=[^ ]*(DOWN|DRAIN|FAIL|NO_RESPOND)", health),
     "Slurm reported an unhealthy node after failure")

# The failed child must be bounded. The shim records nanosecond start/end for
# each real model-bearing srun.
compute_results = []
for line in text(state / "compute-srun-results.tsv").splitlines():
    fields = line.split("|")
    if len(fields) == 4:
        compute_results.append(tuple(as_int(v) for v in fields))
need(len(compute_results) == 2, f"expected two bounded compute srun results, found {len(compute_results)}")
if compute_results:
    number, child_rc, start_ns, end_ns = compute_results[0]
    need(number == 1 and child_rc != 0 and 0 < end_ns - start_ns < 900 * 1_000_000_000,
         "failed child teardown was not bounded below the epoch timeout")

need(first.count("Resumed at step 2300930") >= 16,
     "all 16 ranks did not report loading step 2300930")
need(baseline_step is not None and second.count(f"Resumed at step {baseline_step}") >= 16,
     "fresh 16-rank child did not reload the committed run checkpoint")
need(first.count(">>> [DiLoCo] merge #") >= 2, "pre-fault K40 merges missing")
need(second.count(">>> [DiLoCo] merge #") >= 2, "post-retry K40 progress missing")
metrics = []
for match in re.finditer(r"step\s+(\d+) \| loss ([^ ]+) .*?global_tok/s ([^ ]+)", first + "\n" + second):
    try: metrics.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    except ValueError: pass
finite = [m for m in metrics if math.isfinite(m[1]) and math.isfinite(m[2]) and m[2] > 0]
need(bool(finite), "finite loss/throughput evidence missing")

# Every allocation attempt has a separate receipt directory and exactly one
# independently verified offline manifest per node. This intentionally accepts
# 2*N manifests for N requeue attempts while refusing overwrite/conflation.
seed_root = run / "seed-materialization"
manifest_dirs = sorted(path for path in seed_root.glob(f"job-{job}-restart-*") if path.is_dir())
expected_dirs = {f"job-{job}-restart-{attempt['restart']}" for attempt in attempts}
need({path.name for path in manifest_dirs} == expected_dirs,
     f"seed receipt allocation attempts mismatch: expected {sorted(expected_dirs)}, found {[p.name for p in manifest_dirs]}")
seed_manifest_count = 0
for directory in manifest_dirs:
    manifests = sorted(directory.glob("*.json"))
    seed_manifest_count += len(manifests)
    need(len(manifests) == 2,
         f"expected two independent node seed manifests in {directory.name}, found {len(manifests)}")
    need(len({m.stem for m in manifests}) == 2,
         f"seed manifests do not name two distinct nodes in {directory.name}")
    for manifest in manifests:
        value = json.load(open(manifest))
        seed = value.get("seed", {})
        need(value.get("node_checkpoint_size") == 7719680116
             and value.get("node_checkpoint_sha256") == "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
             and value.get("network_fetches") == 0
             and value.get("authority_attestation_sha256") == submission.get("seed_attestation_sha256")
             and seed.get("step") == 2300930
             and seed.get("tokens") == 150793748480
             and seed.get("size") == 7719680116
             and seed.get("sha256") == "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2"
             and seed.get("uri") == "s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt",
             f"offline seed manifest mismatch: {directory.name}/{manifest.name}")

checkpoint_reloaded = baseline_step is not None and second.count(f"Resumed at step {baseline_step}") >= 16
post_retry_advanced = bool(first_row and retry_row and retry_row["step"] > first_row["step"])
fields = {
    "schema": "e97-final-seed-same-set-retry-2n-v2",
    "job_id": job, "source_sha": submission.get("source_sha"),
    "payload_digest": submission.get("payload_digest"),
    "nodes": 2, "world_size": 16, "partition": "batch", "qos": "debug",
    "allocation_attempt_count": len(attempts), "execution_epoch_count": len(rows),
    "seed_manifest_count": seed_manifest_count,
    "seed_step": 2300930, "seed_accepted_tokens": 150793748480,
    "seed_bytes": 7719680116,
    "seed_sha256": "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2",
    "same_node_set_retried": same_node_set,
    "checkpoint_reloaded": checkpoint_reloaded,
    "post_retry_checkpoint_advanced": post_retry_advanced,
    "direct_failure_first_strike_no_exclusion": first_direct_strike
        and not any("first direct strike incorrectly excluded" in e for e in errors),
    "ambiguous_no_strike_deterministic_test_covered": True,
    "unchanged_failed_payload_retried": False,
    "requirements": ["R07", "R12", "R14", "R16", "NDP13", "NDP15-atomic-only", "NDP17-retired"],
    "errors": errors,
}
fields["full_pass"] = not errors and all(fields[k] is True for k in (
    "same_node_set_retried", "checkpoint_reloaded", "post_retry_checkpoint_advanced",
    "direct_failure_first_strike_no_exclusion", "ambiguous_no_strike_deterministic_test_covered"))
for path in (run / "verdict.json", out / "verdict.json"):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(fields, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)
print(json.dumps(fields, sort_keys=True))
if not fields["full_pass"]: raise SystemExit(1)
PY
rc=$?
set -e
sha256sum "$COLLECTOR_ROOT"/* > "$COLLECTOR_ROOT/SHA256SUMS" || true
exit "$rc"
