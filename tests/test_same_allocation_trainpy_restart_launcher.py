import os
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
    assert 'if (( ${#usable[@]} < current_nodes )); then' in text
    assert 'current_nodes=${#usable[@]}' in text
    assert 'tasks=$((current_nodes * TASKS_PER_NODE))' in text
    assert "source \"$REPO/scripts/frontier/activate_emender_frontier.sh\"" in text
    assert "PYTHON_BIN=$EMENDER_PYTHON" in text


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
    n1) echo 'NodeName=n1 State=DOWN+NOT_RESPONDING' ;;
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
        'samealloc_usable_nodes fake-nodelist',
        env={"PATH": f"{bindir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["n0", "n2"]


def test_launcher_has_no_async_or_communicator_shrink_protocol():
    text = LAUNCHER.read_text().lower()

    forbidden = (
        "sqlite",
        "sha256sum",
        "ncclcommshrink",
        "shrink_communicator",
        "owner-tree",
        "owner_tree",
        "background checkpoint",
    )
    for token in forbidden:
        assert token not in text
