from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SERVICE_TEST_MODULES = {
    "test_native_dataplane_abi.py",
    "test_native_dataplane_failure.py",
    "test_native_dataplane_reference.py",
    "test_native_pool_integration.py",
}
_TOKEN_HEX = "7c" * 32


def _service_binary() -> Path:
    configured = os.environ.get("EMENDER_NDP_SERVICE")
    library = os.environ.get("EMENDER_NDP_LIBRARY")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    if library:
        lib = Path(library).resolve()
        candidates.extend([
            lib.parent.parent / "bin" / "ndp_cxi_service",
            lib.parent.parent / "transport" / "ndp_cxi_service",
        ])
    candidates.extend([
        ROOT / "build/native-rpc-v1/transport/ndp_cxi_service",
        ROOT / "build/native-resilient-dataplane/bin/ndp_cxi_service",
        ROOT / "build/native-resilient-dataplane-build/transport/ndp_cxi_service",
        ROOT / "build/native-dataplane/ndp_cxi_service",
    ])
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "ndp_cxi_service was not found; set EMENDER_NDP_SERVICE or build the native bundle"
    )


@pytest.fixture(autouse=True)
def compiled_native_service(request: pytest.FixtureRequest, monkeypatch, tmp_path,
                            tmp_path_factory):
    """Give local-ABI tests a fresh authoritative service process per test."""
    if Path(str(request.node.path)).name not in _SERVICE_TEST_MODULES:
        yield
        return

    service_dir = tmp_path_factory.mktemp("native-service")
    socket_path = service_dir / "ndp-service.sock"
    monkeypatch.setenv("EMENDER_NDP_SOCKET", str(socket_path))
    monkeypatch.setenv("EMENDER_NDP_ADMISSION_TOKEN_HEX", _TOKEN_HEX)
    service_environment = dict(os.environ)
    if request.node.name == "test_shared_byte_exhaustion_is_fixed_at_service_start":
        service_environment["EMENDER_NDP_MAX_SHARED_BYTES"] = "31"
    if request.node.name == "test_optional_fallback_materializes_only_one_reduced_numerator":
        service_environment["EMENDER_NDP_FALLBACK_SPOOL_DIR"] = str(tmp_path)
    if request.node.name == "test_persistent_service_preserves_exact_global_numerator":
        service_environment["EMENDER_NDP_INTERMEDIATE_F64"] = "1"
    process = subprocess.Popen(
        [
            str(_service_binary()),
            "--provider", "tcp;ofi_rxm",
            "--test-only", "--serve",
            "--bind-node", "127.0.0.1",
            "--socket", str(socket_path),
            "--admission-token-hex", _TOKEN_HEX,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=service_environment,
    )
    try:
        for _ in range(300):
            if socket_path.is_socket():
                break
            if process.poll() is not None:
                raise RuntimeError(
                    f"ndp_cxi_service exited during startup with {process.returncode}"
                )
            time.sleep(0.05)
        else:
            raise TimeoutError("ndp_cxi_service did not bind its control socket")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if socket_path.exists():
            raise AssertionError("ndp_cxi_service leaked its AF_UNIX socket on shutdown")
