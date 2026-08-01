import hashlib
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/frontier/e97_same_allocation_restart.sbatch"


def _bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", "-c", f"source {LAUNCHER!s}; {script}"],
        cwd=REPO,
        env=full_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_production_defaults_and_fixed_world_data_plane_are_explicit():
    text = LAUNCHER.read_text()

    assert "#SBATCH --no-kill" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:TERM@300" in text
    assert "DILOCO_K=${DILOCO_K:-40}" in text
    assert "SAVE_EVERY=${SAVE_EVERY:-200}" in text
    assert "KEEP_CHECKPOINTS=${KEEP_CHECKPOINTS:-2}" in text
    assert "DILOCO_MERGE_BUCKET_NUMEL=${DILOCO_MERGE_BUCKET_NUMEL:-67108864}" in text
    assert "--diloco_merge_topology \"$DILOCO_MERGE_TOPOLOGY\"" in text
    assert "--diloco_merge_bucket_numel \"$DILOCO_MERGE_BUCKET_NUMEL\"" in text
    assert "--kill-on-bad-exit=1" in text
    assert "--wait=\"$FAILED_STEP_WAIT_SECONDS\"" in text
    assert "timeout --signal=TERM" in text
    assert "--kill-after=\"${FAILED_STEP_KILL_GRACE_SECONDS}s\"" in text
    assert "--resume \"$stable_latest\"" in text
    assert '--exact_output_dir "$epoch_output"' in text
    assert 'local stable_latest="$RUN_DIR/train/latest.pt"' in text
    assert 'RUN_DIR=${RUN_DIR:-${RUN_ROOT%/}/$RUN_ID}' in text
    assert 'samealloc_next_epoch "$RUN_DIR/supervisor/execution-epoch.txt"' in text
    assert "MAX_CONSECUTIVE_NO_PROGRESS_FAILURES" in text
    assert "samealloc_direct_failure_host" in text
    assert "samealloc_record_direct_failure" in text
    assert "samealloc_update_no_progress" in text
    assert 'tasks=$((current_nodes * TASKS_PER_NODE))' in text
    assert "source \"$REPO/scripts/frontier/activate_emender_frontier.sh\"" in text
    assert "PYTHON_BIN=$EMENDER_PYTHON" in text
    assert "final_seed_step=2300930" in text
    assert "final_seed_tokens=150793748480" in text
    assert "final_seed_size=7719680116" in text
    assert "0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2" in text
    assert "sbcast" in text
    assert "--verify-local" in text
    assert "samealloc_bind_restart_authority" in text
    assert "INITIAL_CHECKPOINT" not in text


def test_exact_two_node_final_seed_runner_is_held_collected_and_debug_bound():
    submitter = REPO / "scripts/frontier/submit_e97_2n_final_seed_retry.sh"
    collector = REPO / "scripts/frontier/e97_2n_final_seed_retry_collector.sh"
    shim = REPO / "scripts/frontier/e97_2n_final_seed_retry_srun_shim.sh"
    submitted = submitter.read_text()
    collected = collector.read_text()
    observed = shim.read_text()

    assert "CANONICAL_BASE=c625cede2b97ad43af6e1e47a5fd4d58e1dbafcb" in submitted
    assert "--parsable --hold" in submitted
    assert '-p batch -q debug' in submitted
    assert ' -N2 -t 00:20:00' in submitted
    assert 'dependency="afterany:$payload_id"' in submitted
    assert "scontrol release \"$payload_id\"" in submitted
    assert "materialize_e97_s3_seed.py" in submitted and "--prefetch" in submitted
    assert "unchanged payload bytes already attempted" in submitted
    assert '"same_node_set_retried"' in collected
    assert '"checkpoint_reloaded"' in collected
    assert '"post_retry_checkpoint_advanced"' in collected
    assert 'fields["full_pass"]' in collected
    assert "ambiguous|no-strike" in collected
    assert "fault_environment_removed=true" in observed
    assert "unchanged_failed_payload_retried=false" in observed


def test_atomic_promotion_selects_only_complete_epoch_latest(tmp_path: Path):
    epoch = tmp_path / "epoch" / "train" / "run-name"
    epoch.mkdir(parents=True)
    committed = epoch / "checkpoint_step_000200_loss_1.0000.pt"
    committed.write_bytes(b"complete")
    (epoch / ".checkpoint_step_000400_loss_0.9000.pt.partial.tmp").write_bytes(b"partial")
    (epoch / "latest.pt").symlink_to(committed.name)
    stable = tmp_path / "stable" / "latest.pt"

    result = _bash(
        'samealloc_promote_epoch_latest "$EPOCH_OUTPUT" "$STABLE"; '
        'printf "%s\\n" "$(readlink -f "$STABLE")"',
        env={"EPOCH_OUTPUT": str(epoch.parent), "STABLE": str(stable), "SLURM_JOB_ID": "99"},
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == committed
    assert stable.is_symlink()
    assert stable.resolve() == committed


def test_partial_checkpoint_without_latest_is_never_promoted(tmp_path: Path):
    epoch = tmp_path / "epoch" / "train" / "run-name"
    epoch.mkdir(parents=True)
    (epoch / ".checkpoint_step_000200_loss_1.0.pt.partial.tmp").write_bytes(b"partial")
    stable = tmp_path / "stable" / "latest.pt"

    result = _bash(
        'samealloc_promote_epoch_latest "$EPOCH_OUTPUT" "$STABLE"',
        env={"EPOCH_OUTPUT": str(epoch.parent), "STABLE": str(stable)},
    )

    assert result.returncode != 0
    assert not stable.exists()


def test_first_attempt_binds_verified_final_seed_without_legacy_pointer(tmp_path: Path):
    seed = tmp_path / "emender-e97-seed-7001" / "checkpoint-step-2300930.pt"
    seed.parent.mkdir()
    seed.write_bytes(b"verified-final-seed")
    stable = tmp_path / "run" / "train" / "latest.pt"

    result = _bash(
        'samealloc_bind_restart_authority "$STABLE" "$SEED"; readlink "$STABLE"',
        env={"STABLE": str(stable), "SEED": str(seed), "SLURM_JOB_ID": "7001"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(seed.resolve())
    assert stable.resolve() == seed


def test_pre_first_checkpoint_requeue_rebinds_prior_job_tmp_seed(tmp_path: Path):
    seed = tmp_path / "emender-e97-seed-7002" / "checkpoint-step-2300930.pt"
    seed.parent.mkdir()
    seed.write_bytes(b"same-verified-final-seed")
    stable = tmp_path / "run" / "train" / "latest.pt"
    stable.parent.mkdir(parents=True)
    stable.symlink_to(
        "/tmp/emender-e97-seed-7001/checkpoint-step-2300930.pt")

    result = _bash(
        'samealloc_bind_restart_authority "$STABLE" "$SEED"; readlink "$STABLE"',
        env={"STABLE": str(stable), "SEED": str(seed), "SLURM_JOB_ID": "7002"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(seed.resolve())
    assert stable.resolve() == seed


def test_requeue_never_replaces_newer_committed_run_checkpoint(tmp_path: Path):
    seed = tmp_path / "emender-e97-seed-7002" / "checkpoint-step-2300930.pt"
    seed.parent.mkdir()
    seed.write_bytes(b"verified-final-seed")
    committed = tmp_path / "run" / "train" / "checkpoint_step_2301130_loss_2.4.pt"
    committed.parent.mkdir(parents=True)
    committed.write_bytes(b"newer-run-checkpoint")
    stable = committed.parent / "latest.pt"
    stable.symlink_to(committed.name)

    result = _bash(
        'samealloc_bind_restart_authority "$STABLE" "$SEED"; readlink -f "$STABLE"',
        env={"STABLE": str(stable), "SEED": str(seed), "SLURM_JOB_ID": "7002"},
    )

    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == committed


def test_execution_epoch_and_master_port_change_across_slurm_job_ids(tmp_path: Path):
    state = tmp_path / "execution-epoch.txt"
    first = _bash(
        'epoch=$(samealloc_next_epoch "$STATE"); samealloc_master_port "$epoch"',
        env={"STATE": str(state), "SLURM_JOB_ID": "5125415"},
    )
    second = _bash(
        'epoch=$(samealloc_next_epoch "$STATE"); samealloc_master_port "$epoch"',
        env={"STATE": str(state), "SLURM_JOB_ID": "6125415"},
    )

    assert first.returncode == second.returncode == 0
    assert state.read_text().strip() == "2"
    assert first.stdout.strip() != second.stdout.strip()
    assert 20000 <= int(first.stdout) < 60000
    assert 20000 <= int(second.stdout) < 60000


def test_usable_node_selection_drops_only_unhealthy_whole_nodes(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scontrol = bindir / "scontrol"
    scontrol.write_text(
        """#!/bin/bash
if [[ $1 == show && $2 == hostnames ]]; then
  printf '%s\\n' n0 n1 n2 n3
elif [[ $1 == show && $2 == node ]]; then
  case $3 in
    n0) echo 'NodeName=n0 State=ALLOCATED' ;;
    n1) echo 'NodeName=n1 State=DOWN*+NOT_RESPONDING' ;;
    n2) echo 'NodeName=n2 State=MIXED' ;;
    n3) echo 'NodeName=n3 State=DRAINING' ;;
  esac
else
  exit 2
fi
"""
    )
    scontrol.chmod(0o755)

    result = _bash(
        """
declare -a usable=()
declare -A excluded=() hard_bad=()
samealloc_select_usable_nodes fake-nodelist excluded usable hard_bad
printf 'usable=%s\\n' "$(IFS=,; echo "${usable[*]}")"
printf 'n1=%s|%s\\n' "${hard_bad[n1]}" "${excluded[n1]}"
printf 'n3=%s|%s\\n' "${hard_bad[n3]}" "${excluded[n3]}"
""",
        env={"PATH": f"{bindir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "usable=n0,n2",
        "n1=DOWN*+NOT_RESPONDING|slurm-DOWN*+NOT_RESPONDING",
        "n3=DRAINING|slurm-DRAINING",
    ]


def test_direct_hostname_attribution_is_strict_and_never_uses_collective_reporter(tmp_path: Path):
    ambiguous = tmp_path / "ambiguous.err"
    ambiguous.write_text(
        "[rank3]: RuntimeError: collective failed on n0\n"
        "srun: error: n0: task 3: Exited with exit code 86\n"
        "srun: error: n1: task 8: Exited with exit code 1\n"
    )
    one = tmp_path / "one.err"
    one.write_text(
        "[rank7]: NCCL collective reporter n1\n"
        "srun: error: n2: task 17: Exited with exit code 86\n"
    )

    ambiguous_result = _bash(
        'samealloc_direct_failure_host "$LOG" n0 n1 n2', env={"LOG": str(ambiguous)}
    )
    one_result = _bash(
        'samealloc_direct_failure_host "$LOG" n0 n1 n2', env={"LOG": str(one)}
    )

    assert ambiguous_result.returncode != 0
    assert ambiguous_result.stdout == ""
    assert one_result.returncode == 0, one_result.stderr
    assert one_result.stdout.strip() == "n2"


def test_direct_failure_strikes_exclude_only_on_second_same_host():
    result = _bash(
        """
declare -A strikes=() excluded=()
samealloc_record_direct_failure n1 strikes excluded
printf 'first=%s excluded_first=%s\\n' "${strikes[n1]}" "${excluded[n1]-}"
samealloc_record_direct_failure n2 strikes excluded
printf 'other=%s excluded_other=%s\\n' "${strikes[n2]}" "${excluded[n2]-}"
samealloc_record_direct_failure n1 strikes excluded
printf 'second=%s excluded_second=%s\\n' "${strikes[n1]}" "${excluded[n1]-}"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "first=1 excluded_first=",
        "other=1 excluded_other=",
        "second=2 excluded_second=direct-failure-strike-2",
    ]


def test_ambiguous_failure_does_not_record_any_strike(tmp_path: Path):
    log = tmp_path / "collective.err"
    log.write_text(
        "[rank0] n0 first collective reporter\n"
        "srun: error: n0: task 0: Exited with exit code 1\n"
        "srun: error: n1: task 8: Exited with exit code 1\n"
    )
    result = _bash(
        """
declare -A strikes=() excluded=()
if host=$(samealloc_direct_failure_host "$LOG" n0 n1); then
  samealloc_record_direct_failure "$host" strikes excluded
fi
printf 'strike_count=%s exclusion_count=%s\\n' "${#strikes[@]}" "${#excluded[@]}"
""",
        env={"LOG": str(log)},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "strike_count=0 exclusion_count=0"


def test_new_checkpoint_resets_bounded_no_progress_counter():
    result = _bash(
        """
count=2
step=200
samealloc_update_no_progress 200 200 count step
printf 'stalled=%s:%s\\n' "$count" "$step"
samealloc_update_no_progress 200 400 count step
printf 'advanced=%s:%s\\n' "$count" "$step"
samealloc_update_no_progress 400 400 count step
printf 'again=%s:%s\\n' "$count" "$step"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["stalled=3:200", "advanced=0:400", "again=1:400"]


def test_allocation_identity_change_clears_strikes_exclusions_and_no_progress():
    result = _bash(
        """
declare identity='job1:0:n[0-1]' count=2
declare -A strikes=([n1]=2) excluded=([n1]=direct-failure-strike-2)
samealloc_reset_allocation_policy 'job1:0:n[0-1]' identity strikes excluded count
printf 'same=%s:%s:%s\\n' "$count" "${strikes[n1]}" "${excluded[n1]}"
samealloc_reset_allocation_policy 'job1:1:n[2-3]' identity strikes excluded count
printf 'new=%s:%s:%s:%s\\n' "$identity" "$count" "${#strikes[@]}" "${#excluded[@]}"
"""
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "same=2:2:direct-failure-strike-2",
        "new=job1:1:n[2-3]:0:0:0",
    ]


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def test_supervisor_retries_exact_checkpoint_with_fresh_epoch_port_and_reduced_world(tmp_path: Path):
    repo = tmp_path / "repo"
    bindir = tmp_path / "bin"
    run_dir = tmp_path / "run"
    (repo / "scripts/frontier").mkdir(parents=True)
    bindir.mkdir()
    data = tmp_path / "data.txt"
    val_data = tmp_path / "val.txt"
    data.write_text("x")
    val_data.write_text("x")
    initial_dir = tmp_path / "seed"
    initial_dir.mkdir()
    initial_checkpoint = initial_dir / "checkpoint_step_000100_loss_1.0.pt"
    initial_checkpoint.write_bytes(b"step100")
    initial_latest = initial_dir / "latest.pt"
    initial_latest.symlink_to(initial_checkpoint.name)

    _write_executable(
        repo / "scripts/frontier/activate_emender_frontier.sh",
        """#!/bin/bash
export EMENDER_PYTHON=${TEST_EMENDER_PYTHON:?}
frontier_require_requested_rccl_net_plugin() { return 0; }
""",
    )
    _write_executable(repo / "scripts/frontier/frontier_runtime_env.sh", "#!/bin/bash\n")
    (repo / "configs/frontier").mkdir(parents=True)
    (repo / "configs/frontier/e97_async_256.yaml").write_text(
        (REPO / "configs/frontier/e97_async_256.yaml").read_text()
    )
    (repo / "scripts/frontier/materialize_e97_s3_seed.py").write_text("# mocked by srun\n")
    cache = tmp_path / (
        "sha256-0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2.pt"
    )
    cache.write_bytes(b"fixture")
    attestation = tmp_path / "seed-attestation.json"
    attestation.write_text("{}\n")
    _write_executable(bindir / "git", "#!/bin/bash\necho fake-source-sha\n")
    _write_executable(
        bindir / "sbcast",
        "#!/bin/bash\nset -e\nmkdir -p \"$(dirname \"$3\")\"\ncp \"$2\" \"$3\"\n",
    )
    _write_executable(
        bindir / "scontrol",
        """#!/bin/bash
if [[ $1 == show && $2 == hostnames ]]; then printf '%s\\n' n0 n1; exit 0; fi
if [[ $1 == show && $2 == node ]]; then echo "NodeName=$3 State=ALLOCATED"; exit 0; fi
if [[ $1 == requeue ]]; then printf 'requeue=%s\\n' "$2" >> "$TEST_STATE/requeue.log"; exit 0; fi
exit 2
""",
    )
    _write_executable(
        bindir / "srun",
        """#!/bin/bash
set -euo pipefail
[[ $* == *--gpus-per-task=0* ]] && exit 0
count_file=$TEST_STATE/count
count=0; [[ ! -r $count_file ]] || read -r count < "$count_file"; count=$((count+1)); echo "$count" > "$count_file"
printf '%s|%s|%s|%s|%s\\n' "$count" "$EMENDER_EXECUTION_EPOCH" "$MASTER_PORT" "$*" "$(readlink -f "$RUN_DIR/train/latest.pt")" >> "$TEST_STATE/launches.tsv"
if (( count == 1 )); then
  printf step2301130 > "$RUN_DIR/train/checkpoint_step_2301130_loss_2.4.pt"
  ln -sfn checkpoint_step_2301130_loss_2.4.pt "$RUN_DIR/train/latest.pt"
  echo 'srun: error: n1: task 8: Exited with exit code 86' >&2
  exit 86
fi
if (( count == 2 )); then
  echo 'srun: error: n1: task 8: Exited with exit code 86' >&2
  exit 86
fi
[[ $* == *--nodelist=n0* && $* != *--nodelist=n0,n1* ]] || exit 91
printf step2301330 > "$RUN_DIR/train/checkpoint_step_2301330_loss_2.3.pt"
ln -sfn checkpoint_step_2301330_loss_2.3.pt "$RUN_DIR/train/latest.pt"
exit 0
""",
    )

    integration_env = {
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "REPO": str(repo),
        "RUN_ID": "integration",
        "RUN_DIR": str(run_dir),
        "RUN_ROOT": str(tmp_path),
        "INITIAL_CHECKPOINT": str(initial_latest),
        "SLURM_JOB_ID": "7001",
        "SLURM_JOB_NODELIST": "n[0-1]",
        "SLURM_JOB_NUM_NODES": "2",
        "TARGET_NODES": "2",
        "MIN_NODES": "1",
        "TASKS_PER_NODE": "1",
        "CPUS_PER_TASK": "1",
        "MAX_CONSECUTIVE_NO_PROGRESS_FAILURES": "3",
        "EXECUTION_EPOCH_TIMEOUT_SECONDS": "30",
        "FAILED_STEP_WAIT_SECONDS": "1",
        "FAILED_STEP_KILL_GRACE_SECONDS": "1",
        "DATA": str(data),
        "VAL_DATA": str(val_data),
        "TEST_STATE": str(tmp_path),
        "TEST_EMENDER_PYTHON": os.environ.get("EMENDER_PYTHON", os.sys.executable),
        "E97_SEED_CONFIG": str(repo / "configs/frontier/e97_async_256.yaml"),
        "E97_SEED_CACHE": str(cache),
        "E97_SEED_ATTESTATION": str(attestation),
        "E97_SEED_ATTESTATION_SHA256": hashlib.sha256(
            attestation.read_bytes()
        ).hexdigest(),
    }
    result = _bash('samealloc_main', env=integration_env)

    assert result.returncode == 0, result.stderr
    launches = (tmp_path / "launches.tsv").read_text().splitlines()
    assert len(launches) == 3
    records = [line.split("|", 4) for line in launches]
    assert [record[1] for record in records] == ["1", "2", "3"]
    assert len({record[2] for record in records}) == 3
    assert "--nodelist=n0,n1" in records[0][3]
    assert "--nodelist=n0,n1" in records[1][3]
    assert "--nodelist=n0" in records[2][3]
    assert records[0][4].endswith("checkpoint-step-2300930.pt")
    assert records[1][4].endswith("checkpoint_step_2301130_loss_2.4.pt")
    assert records[2][4].endswith("checkpoint_step_2301130_loss_2.4.pt")
    ports = [int(record[2]) for record in records]
    assert all(20000 <= port < 60000 for port in ports)
    assert not (tmp_path / "requeue.log").exists()
    for epoch in (1, 2, 3):
        epoch_dir = run_dir / "epochs" / f"epoch-{epoch:06d}"
        assert (epoch_dir / "launch.env").is_file()
        assert (epoch_dir / "train-command.txt").is_file()

    # A second deterministic allocation has only ambiguous collective fallout,
    # never advances latest.pt, retries the identical healthy set without a
    # strike, and reaches the bounded requeue path after exactly two failures.
    bounded_state = tmp_path / "bounded-state"
    bounded_state.mkdir()
    bounded_run = tmp_path / "bounded-run"
    _write_executable(
        bindir / "srun",
        """#!/bin/bash
set -euo pipefail
[[ $* == *--gpus-per-task=0* ]] && exit 0
count_file=$TEST_STATE/count
count=0; [[ ! -r $count_file ]] || read -r count < "$count_file"; count=$((count+1)); echo "$count" > "$count_file"
printf '%s|%s|%s|%s\\n' "$count" "$EMENDER_EXECUTION_EPOCH" "$MASTER_PORT" "$*" >> "$TEST_STATE/launches.tsv"
echo 'srun: error: n0: task 0: Exited with exit code 1' >&2
echo 'srun: error: n1: task 1: Exited with exit code 1' >&2
exit 1
""",
    )
    bounded_env = integration_env | {
        "RUN_ID": "bounded",
        "RUN_DIR": str(bounded_run),
        "SLURM_JOB_ID": "7002",
        "MAX_CONSECUTIVE_NO_PROGRESS_FAILURES": "2",
        "TEST_STATE": str(bounded_state),
    }
    bounded = _bash("samealloc_main", env=bounded_env)

    assert bounded.returncode == 0, bounded.stderr
    bounded_launches = (bounded_state / "launches.tsv").read_text().splitlines()
    assert len(bounded_launches) == 2
    assert all("--nodelist=n0,n1" in line for line in bounded_launches)
    assert (bounded_state / "requeue.log").read_text().strip() == "requeue=7002"
    policy = (bounded_run / "supervisor/node-policy.tsv").read_text().splitlines()
    assert policy == [
        "1|-|ambiguous|no-strike|-",
        "2|-|ambiguous|no-strike|-",
    ]


def test_repeated_no_progress_failures_request_bounded_requeue():
    # Source-level guard plus deterministic transition coverage: the integration
    # above covers progress reset, while this verifies the exact bounded branch.
    text = LAUNCHER.read_text()
    assert re.search(
        r"consecutive_no_progress\s*>=\s*MAX_CONSECUTIVE_NO_PROGRESS_FAILURES", text
    )
    assert 'samealloc_request_requeue "bounded consecutive failures made no checkpoint progress' in text


def test_acceptance_wrappers_do_not_precreate_finite_epoch_directories():
    wrappers = [
        REPO / "scripts/frontier/submit_e97_8n_samealloc_acceptance.sh",
        REPO / "scripts/frontier/submit_e97_32n_clean.sh",
        REPO / "scripts/frontier/submit_e97_128n_clean.sh",
    ]
    for wrapper in wrappers:
        text = wrapper.read_text()
        assert "precreate" not in text.lower()
        assert not re.search(r'mkdir -p .*epochs/epoch-00000[12]', text)


def test_launcher_has_no_async_or_communicator_shrink_protocol():
    text = LAUNCHER.read_text().lower()

    forbidden = (
        "sqlite",
        "ncclcommshrink",
        "shrink_communicator",
        "owner-tree",
        "owner_tree",
        "background checkpoint",
    )
    for token in forbidden:
        assert token not in text
