from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_trainpy_async_quorum_smokes_launch_one_rank_per_gpu():
    common = _read("scripts/frontier/trainpy_async_quorum_smoke_common.sh")
    one = _read("scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch")
    two = _read("scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch")
    combined = common + "\n" + one + "\n" + two

    for wrapper in (one, two):
        assert "#SBATCH --ntasks-per-node=8" in wrapper
        assert "#SBATCH --gpus-per-task=1" in wrapper
        assert "#SBATCH --gpu-bind=closest" in wrapper

    assert 'srun\n  -N "$SMOKE_NODE_COUNT"\n  -n "$ASYNC_TRAINPY_RANKS"' in common
    assert '--ntasks-per-node="$RANKS_PER_NODE"' in common
    assert "--gpus-per-task=1" in common
    assert '--gpu-bind="$GPU_BIND"' in common
    assert 'one_trainpy_rank_per_gpu=1' in common
    assert 'printf "%s\\t%s\\t%s\\t%s\\t%s\\n"' in common
    assert '--node-rank "${SLURM_PROCID:?missing SLURM_PROCID}"' in common
    assert '--device "cuda:${ASYNC_VISIBLE_DEVICE_ORDINAL:-0}"' in common
    assert "--diloco" not in combined
    assert "[DDP] wrapped model in DistributedDataParallel" in common
    assert 'exec "$PYTHON_BIN" -u "$ASYNC_ENTRYPOINT"' in common


def test_trainpy_async_quorum_smokes_record_metrics_checkpoint_and_no_ddp_validation():
    common = _read("scripts/frontier/trainpy_async_quorum_smoke_common.sh")

    for token in (
        'METRICS_JSON="${ARTIFACT_DIR}/metrics.json"',
        'SUMMARY_FILE="${SUMMARY_DIR}/summary.md"',
        'MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"',
        'COMMAND_FILE="${ARTIFACT_DIR}/command.txt"',
        'RANK_START_LOG="${ARTIFACT_DIR}/rank-start.tsv"',
        'DEFAULT_ENV_PREFIX="${REPO}/.envs/olcf-rocm711-torch210-py312"',
        "DEFAULT_ENV_PREFIX=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312",
        'export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-$ENV_PREFIX}',
        "frontier_assert_emender_conda_env",
        "emender_conda_env=$EMENDER_CONDA_ENV",
        "python_bin=$PYTHON_BIN",
        "ASYNC_QUORUM_TRANSPORT=${ASYNC_QUORUM_TRANSPORT:-compiled-cray-mpich-helper-p2p}",
        'ASYNC_COMPILED_MPICH_IPC_BASE=${ASYNC_COMPILED_MPICH_IPC_BASE:-${TMPDIR:-/tmp}/emender-${USER:-unknown}/trainpy_async_quorum}',
        'ASYNC_COMPILED_MPICH_IPC_DIR=${ASYNC_COMPILED_MPICH_IPC_DIR:-${ASYNC_COMPILED_MPICH_IPC_BASE}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}/ipc}',
        'ASYNC_COMPILED_MPICH_TRACE_DIR=${ASYNC_COMPILED_MPICH_TRACE_DIR:-${ARTIFACT_DIR}/compiled_mpich_trace}',
        'ASYNC_COMPILED_MPICH_FILE_GATHER=${ASYNC_COMPILED_MPICH_FILE_GATHER:-0}',
        'if [[ "$SMOKE_NODE_COUNT" -eq 1 && "$ASYNC_QUORUM_TRANSPORT" == "compiled-cray-mpich-helper-p2p" ]]; then',
        "--actual-multinode-compiled-mpich-quorum",
        "--compiled-mpich-helper-bin",
        "--compiled-mpich-ipc-dir",
        "--mpi-dense-bucket-bytes",
        "--coordinator-host",
        "async_quorum_transport=$ASYNC_QUORUM_TRANSPORT",
        "async_mpi_dense_bucket_bytes=$ASYNC_MPI_DENSE_BUCKET_BYTES",
        "async_compiled_mpich_helper_bin=$ASYNC_COMPILED_MPICH_HELPER_BIN",
        "async_compiled_mpich_ipc_dir=$ASYNC_COMPILED_MPICH_IPC_DIR",
        "async_compiled_mpich_trace_dir=$ASYNC_COMPILED_MPICH_TRACE_DIR",
        "async_compiled_mpich_file_gather=$ASYNC_COMPILED_MPICH_FILE_GATHER",
        '! -r "${ASYNC_COMPILED_MPICH_HELPER_BIN}.so"',
        "scripts/frontier/build_compiled_mpich_dense_helper.sh",
        "mpich_gpu_support_enabled=$MPICH_GPU_SUPPORT_ENABLED",
        "export CRAY_MPI4PY_SITE=${CRAY_MPI4PY_SITE:-/opt/cray/pe/python/3.10.10/lib/python3.10/site-packages}",
        "frontier_capture_runtime_env",
        "actual_multinode_tcp_quorum",
        "--recovery-every-generations",
        "--export-every-generations",
        "checkpoint_paths_missing",
        "latest_not_advanced",
        "ddp_or_per_step_all_reduce_log_detected",
        "no_training_tokens_recorded",
        '"ddp_forbidden_line_count"',
    ):
        assert token in common


def test_trainpy_async_quorum_2n_smoke_forces_missing_update_recovery_path():
    two = _read("scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch")
    common = _read("scripts/frontier/trainpy_async_quorum_smoke_common.sh")

    assert "#SBATCH -N 2" in two
    assert "ASYNC_TRAINPY_RANKS=${ASYNC_TRAINPY_RANKS:-$((SMOKE_NODE_COUNT * 8))}" in two
    assert "ASYNC_EXPECTED_MISSING_UPDATES=${ASYNC_EXPECTED_MISSING_UPDATES:-1}" in two
    assert "ASYNC_EXPECTED_RANKS=${ASYNC_EXPECTED_RANKS:-$((ASYNC_TRAINPY_RANKS + ASYNC_EXPECTED_MISSING_UPDATES))}" in two
    assert "ASYNC_GLOBAL_QUORUM=${ASYNC_GLOBAL_QUORUM:-$ASYNC_TRAINPY_RANKS}" in two
    assert 'if [[ "$ASYNC_QUORUM_TRANSPORT" != "tcp" && "$ASYNC_EXPECTED_RANKS" -gt "$ASYNC_TRAINPY_RANKS" ]]; then' in common
    assert "timed_out_updates" in common
    assert "expected_at_least" in common


def test_trainpy_async_quorum_smoke_keeps_mpi4py_path_explicit_only():
    common = _read("scripts/frontier/trainpy_async_quorum_smoke_common.sh")
    entrypoint = _read("scripts/frontier/e97_async_diloco_train.py")

    assert "compiled-cray-mpich-helper-p2p)" in common
    assert "mpi-dense)" in common
    assert "--actual-multinode-mpi-dense-quorum" in entrypoint
    assert "Explicit legacy comparison path" in entrypoint
    assert "choose only one actual multinode quorum transport" in entrypoint


def test_trainpy_async_quorum_report_tracks_pending_slurm_artifacts():
    report = _read("reports/frontier/trainpy-async-quorum-1n2n-smokes-20260707.md")

    for token in (
        "scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch",
        "scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch",
        "Slurm job ID",
        "Metrics JSON",
        "Checkpoint/latest artifact",
        "4951952",
        "4951983",
        "ddp_forbidden_line_count=0",
        "synthetic-token-fallback",
    ):
        assert token in report
