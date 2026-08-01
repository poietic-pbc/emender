import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts/frontier/submit_e97_256n_final_seed_2h.sh"
PAYLOAD = REPO / "scripts/frontier/e97_256n_final_seed_payload.sh"
LAUNCHER = REPO / "scripts/frontier/e97_same_allocation_restart.sbatch"


def test_256n_submit_is_exactly_one_unheld_payload_and_no_collector_job():
    text = SUBMIT.read_text()

    assert text.count("payload_id=$(sbatch") == 1
    assert text.count("$(sbatch") == 1
    assert "collector_id=$(sbatch" not in text
    assert "--dependency=" not in text
    assert "scontrol release" not in text
    assert "--hold" not in text
    assert "--parsable --no-kill --requeue" in text
    assert "-p batch -q debug -J e97-final-seed-256n -N256 -t 02:00:00" in text
    assert "--ntasks-per-node=8" in text
    assert "NumNodes=256" in text
    assert "NumTasks=2048" in text
    assert "Partition=batch" in text
    assert "QOS=debug" in text
    assert "TimeLimit=02:00:00" in text
    assert "unchanged payload bytes already attempted" in text
    assert "scancel" not in text
    assert "collector_job_id=none" in text


def test_256n_snapshot_binds_clean_final_seed_and_all_immutable_assets():
    text = SUBMIT.read_text()

    assert "RUN_ID=${E97_256N_RUN_ID:-e97-final-seed-production-256n}" in text
    assert "RUN_DIR=${E97_256N_RUN_DIR:-$BASE/runs/$RUN_ID}" in text
    assert "EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312}" in text
    assert "SOURCE_SHA=$(git rev-parse HEAD)" in text
    assert '"$SOURCE_SHA" == "$ORIGIN_MAIN_SHA"' in text
    assert '"$SOURCE_SHA" == "$LOCAL_MAIN_SHA"' in text
    assert 'git cat-file -e "$SOURCE_SHA:.final_checkpoint_request"' in text
    assert "SEED_STEP=2300930" in text
    assert "SEED_ACCEPTED_TOKENS=150793748480" in text
    assert "SEED_BYTES=7719680116" in text
    assert "SEED_SHA256=0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2" in text
    assert "--prefetch" in text
    assert "verified_seed_bytes" in text and "verified_seed_sha" in text
    for asset in (
        "e97_256n_final_seed_payload.sh",
        "e97_same_allocation_restart.sbatch",
        "materialize_e97_s3_seed.py",
        "e97_async_256.yaml",
        "train.py",
        "seed_attestation",
    ):
        assert asset in text
    assert "e97_256n_final_seed_collector" not in text
    assert "payload-SHA256SUMS" in text
    assert "source-refs.txt" in text
    assert "FAULT_INJECTION=none" in text
    assert "ACCEPTANCE_SHIM=none" in text


def test_256n_payload_is_clean_envelope_not_a_fault_or_acceptance_shim():
    payload = PAYLOAD.read_text()

    assert 'source "$launcher"' in payload
    assert "samealloc_main" in payload
    assert "NumNodes=256" in payload
    assert "NumTasks=2048" in payload
    assert "Partition=batch" in payload
    assert "QOS=debug" in payload
    assert "TimeLimit=02:00:00" in payload
    assert "e97_256n_final_seed_retry_srun_shim" not in payload
    assert "FINAL_SEED_RETRY_STATE_DIR" not in payload
    assert "EMENDER_DILOCO_EXIT_RANK" in payload  # fail-closed absence check
    assert "export EMENDER_DILOCO_EXIT" not in payload
    assert "fault injection variable is set" in payload


def test_256n_runtime_preserves_atomic_continuing_authority_and_seed_broadcast():
    submit = SUBMIT.read_text()
    launcher = LAUNCHER.read_text()

    assert "TARGET_NODES=256 MIN_NODES=256 TASKS_PER_NODE=8" in submit
    assert "MAX_CONSECUTIVE_NO_PROGRESS_FAILURES=2 REQUEUE_ON_EXHAUSTION=1" in submit
    assert "DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2" in submit
    assert "DILOCO_MERGE_BUCKET_NUMEL=67108864" in submit
    assert "TRAIN_MINUTES=180" in submit
    assert "WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=900" in submit
    assert re.search(r"unset EMENDER_DILOCO_EXIT_RANK.*EMENDER_DILOCO_EXIT_MERGE", submit)
    assert 'local stable_latest="$RUN_DIR/train/latest.pt"' in launcher
    assert 'samealloc_bind_restart_authority "$stable_latest" "$job_seed"' in launcher
    assert 'samealloc_promote_epoch_latest "$epoch_output" "$stable_latest"' in launcher
    assert "sbcast" in launcher
    assert "--verify-local" in launcher
    assert 'job-${SLURM_JOB_ID}-restart-${SLURM_RESTART_COUNT:-0}' in launcher
    assert "samealloc_update_no_progress" in launcher


def test_256n_batch_writes_durable_live_and_terminal_state_for_interactive_monitoring():
    payload = PAYLOAD.read_text()
    submit = SUBMIT.read_text()

    assert '"$RUN_DIR/identity/squeue-live.txt"' in payload
    assert '"$RUN_DIR/identity/scontrol-live.txt"' in payload
    assert '"$RUN_DIR/monitor/launcher-rc.txt"' in payload
    assert '"$RUN_DIR/logs/batch-%j.out"' in submit
    assert '"$RUN_DIR/logs/batch-%j.err"' in submit
    assert '"$RUN_DIR/identity/squeue-submitted.txt"' in submit
    assert '"$RUN_DIR/identity/scontrol-submitted.txt"' in submit
    assert "terminal_evidence_command=" in submit
    assert "sacct -j $payload_id" in submit
