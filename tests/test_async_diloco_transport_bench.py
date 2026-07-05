import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_async_diloco_transport_benchmark_uses_mpi_p2p_not_lustre_payloads():
    bench = (ROOT / "scripts/frontier/async_diloco_transport_bench.cpp").read_text()

    assert "MPI_Isend" in bench
    assert "MPI_Irecv" in bench
    assert "MPI_BYTE" in bench
    assert "dense_data_plane" in bench
    assert "mpi_memory_buffers_no_lustre_payload_files" in bench
    assert "torch.distributed" not in bench
    assert "all_reduce" not in bench
    assert "MPI_Allreduce(&failed" in bench


def test_async_diloco_transport_benchmark_has_xz_payload_and_gpu_staging_metadata():
    bench = (ROOT / "scripts/frontier/async_diloco_transport_bench.cpp").read_text()

    assert "Buffer x(payload_bytes" in bench
    assert "Buffer z(payload_bytes" in bench
    assert "xz_payload_bytes_per_rank" in bench
    assert "hip_device_buffer_gpu_aware_mpi_required" in bench
    assert "MPICH_GPU_SUPPORT_ENABLED" in bench
    assert "FI_CXI_RX_MATCH_MODE" in bench
    assert "failure_modes" in bench


def test_async_diloco_transport_frontier_wrapper_is_debug_scale_and_gpu_aware():
    wrapper = (ROOT / "scripts/frontier/async_diloco_transport_bench.sbatch").read_text()

    assert "#SBATCH -q debug" in wrapper
    assert "#SBATCH -N 1" in wrapper
    assert "#SBATCH -t 00:10:00" in wrapper
    assert "frontier_load_default_modules" in wrapper
    assert "MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}" in wrapper
    assert "FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-kdreg2}" in wrapper
    assert "FI_CXI_RX_MATCH_MODE=${FI_CXI_RX_MATCH_MODE:-hybrid}" in wrapper
    assert "--gpus-per-task=1" in wrapper
    assert "transport_metrics.json" in wrapper
    assert "node_hours=" in wrapper


def test_async_diloco_transport_report_documents_no_frontier_submission_and_env():
    report = (ROOT / "reports/frontier/async-diloco-transport-prototype-20260705.md").read_text()

    for required in [
        "module load PrgEnv-gnu/8.7.0",
        "module load cpe/26.03",
        "module load rocm/7.1.1",
        "module load craype-accel-amd-gfx90a",
        "MPICH_GPU_SUPPORT_ENABLED=1",
        "FI_CXI_RX_MATCH_MODE=hybrid",
        "No Frontier benchmark job was submitted",
        "no-go-not-run",
        "does not use `torch.distributed` all-rank collectives",
    ]:
        assert required in report


def test_async_diloco_transport_local_metrics_schema():
    metrics = json.loads(
        (ROOT / "reports/frontier/async-diloco-transport-local-metrics-20260705.json").read_text()
    )

    assert metrics["benchmark"] == "async_diloco_transport_bench"
    assert metrics["transport"] == "cray_mpich_point_to_point"
    assert metrics["dense_data_plane"] == "mpi_memory_buffers_no_lustre_payload_files"
    assert metrics["merge_mechanism"] == "mpi_p2p_no_torch_distributed_collectives"
    assert metrics["world_size"] == 1
    assert metrics["device"] == "cpu"
    assert metrics["payload_mib_per_state"] == 1
    assert "failure_modes" in metrics
