import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts/frontier/submit_e97_256n_final_seed_2h.sh"
SUBMIT_4H = REPO / "scripts/frontier/submit_e97_256n_final_seed_4h_normal.sh"
SUBMIT_7H = REPO / "scripts/frontier/submit_e97_256n_final_seed_7h_normal.sh"
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
    assert "--parsable --no-requeue" in text
    assert "--no-kill" not in text
    assert "Requeue=0" in text
    assert "-p batch -q debug -J e97-final-seed-256n -N256 -t 02:00:00" in text
    assert "--ntasks-per-node=8" in text
    assert "NumNodes=256" in text
    assert "NumTasks=2048" in text
    assert "Partition=batch" in text
    assert "QOS=debug" in text
    assert "TimeLimit=02:00:00" in text
    assert "unchanged payload bytes already attempted" in text
    assert "E97_EXPECTED_RESUME_STEP" in text
    assert "E97_EXPECTED_RESUME_TOTAL_TOKENS" in text
    assert "explicitly approved resume authority" in text
    assert "mmap=True" in text
    assert "scancel" not in text
    assert "collector_job_id=none" in text


def test_256n_four_hour_normal_submit_is_fail_stop_and_resume_bound():
    text = SUBMIT_4H.read_text()

    assert text.count("payload_id=$(sbatch") == 1
    assert text.count("$(sbatch") == 1
    assert "--dependency=" not in text
    assert "--hold" not in text
    assert "scancel" not in text
    assert "-p batch -q normal -J e97-final-seed-256n-4h -N256 -t 04:00:00" in text
    assert "QOS=normal" in text
    assert "TIME_LIMIT=04:00:00" in text
    assert "EXPECTED_QOS=normal EXPECTED_TIME_LIMIT=04:00:00" in text
    assert "EXECUTION_EPOCH_TIMEOUT_SECONDS=18000" in text
    assert "TRAIN_MINUTES=300" in text
    assert "FAIL_STOP_SINGLE_EPOCH=1 ENABLE_VALIDATION=0 REQUEUE_ON_EXHAUSTION=0" in text
    assert "unset VAL_DATA VAL_EVERY" in text
    assert "--parsable --no-requeue" in text
    assert "Requeue=0" in text
    assert "E97_EXPECTED_RESUME_STEP" in text
    assert "E97_EXPECTED_RESUME_TOTAL_TOKENS" in text
    assert "explicitly approved resume authority" in text
    assert "DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2" in text
    assert "DILOCO_MERGE_BUCKET_NUMEL=67108864" in text
    assert "collector_job_id=none" in text


def test_256n_seven_hour_normal_submit_preserves_production_recipe():
    text = SUBMIT_7H.read_text()

    assert text.count("payload_id=$(sbatch") == 1
    assert text.count("$(sbatch") == 1
    assert "--dependency=" not in text
    assert "--hold" not in text
    assert "scancel" not in text
    assert "-p batch -q normal -J e97-final-seed-256n-7h -N256 -t 07:00:00" in text
    assert "QOS=normal" in text
    assert "TimeLimit=07:00:00" in text
    assert "EXECUTION_EPOCH_TIMEOUT_SECONDS=28800" in text
    assert "--parsable --no-requeue" in text
    assert "FAIL_STOP_SINGLE_EPOCH=1 ENABLE_VALIDATION=0 REQUEUE_ON_EXHAUSTION=0" in text
    assert "unset VAL_DATA VAL_EVERY" in text
    assert "Requeue=0" in text
    assert "TRAIN_MINUTES=480" in text
    assert "DILOCO_K=40 SAVE_EVERY=200 KEEP_CHECKPOINTS=2" in text
    assert "DILOCO_MERGE_BUCKET_NUMEL=67108864" in text
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
    assert "TOKEN_MIGRATION_RECEIPT_REL=docs/validation/e97-total-token-migration-step2303840.json" in text
    assert "TOKEN_MIGRATION_STEP=2303840" in text
    assert "TOKEN_MIGRATION_TOTAL_TOKENS=199615447040" in text
    assert "TOKEN_MIGRATION_SOURCE_JOB=5134243" in text
    assert "TOKEN_MIGRATION_CHECKPOINT_BYTES=7719680116" in text
    assert "EXPECTED_RESUME_STEP=${EXPECTED_RESUME_STEP:-none}" in text
    assert "EXPECTED_RESUME_TOTAL_TOKENS=${EXPECTED_RESUME_TOTAL_TOKENS:-none}" in text
    assert "--expected-step 2303840" in text
    assert "--expected-total-tokens 199615447040" in text
    assert "--expected-source-job-id 5134243" in text
    assert "--expected-size-bytes 7719680116" in text
    assert "--prefetch" in text
    assert "verified_seed_bytes" in text and "verified_seed_sha" in text
    for asset in (
        "e97_256n_final_seed_payload.sh",
        "e97_same_allocation_restart.sbatch",
        "materialize_e97_s3_seed.py",
        "validate_total_token_migration_receipt.py",
        "e97-total-token-migration-step2303840.json",
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
    assert "EXPECTED_PARTITION=${EXPECTED_PARTITION:-batch}" in payload
    assert "EXPECTED_QOS=${EXPECTED_QOS:-debug}" in payload
    assert "EXPECTED_TIME_LIMIT=${EXPECTED_TIME_LIMIT:-02:00:00}" in payload
    assert 'Partition=$EXPECTED_PARTITION' in payload
    assert 'QOS=$EXPECTED_QOS' in payload
    assert 'TimeLimit=$EXPECTED_TIME_LIMIT' in payload
    assert '"Requeue=0"' in payload
    assert "production requires one epoch, validation disabled, and no requeue" in payload
    assert "e97_256n_final_seed_retry_srun_shim" not in payload
    assert "FINAL_SEED_RETRY_STATE_DIR" not in payload
    assert "EMENDER_DILOCO_EXIT_RANK" in payload  # fail-closed absence check
    assert "export EMENDER_DILOCO_EXIT" not in payload
    assert "fault injection variable is set" in payload


def test_256n_runtime_preserves_atomic_continuing_authority_and_seed_broadcast():
    submit = SUBMIT.read_text()
    launcher = LAUNCHER.read_text()

    assert "TARGET_NODES=256 MIN_NODES=256 TASKS_PER_NODE=8" in submit
    assert "FAIL_STOP_SINGLE_EPOCH=1 ENABLE_VALIDATION=0 REQUEUE_ON_EXHAUSTION=0" in submit
    assert "VAL_EVERY=" not in submit
    assert "--no-requeue" in submit
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
    assert 'single execution epoch $epoch failed rc=$rc; no retry or requeue' in launcher
    assert 'if [[ "$ENABLE_VALIDATION" == 1 ]]; then' in launcher
    assert "samealloc_resume_token_bootstrap" in launcher
    assert 'token_args=(--total_tokens "$resume_total_tokens")' in launcher


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
