"""Reproducible native bundle records and fail-closed launch attestation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


BUILD_SCHEMA = "emender-native-dataplane-build-v1"
GATE_SCHEMA = "emender-native-dataplane-gate-v1"
NATIVE_CXI = "native-cxi"
NATIVE_TEST = "native-test"
PYTHON_TCP_DEBUG = "python-tcp-debug"
BACKENDS = frozenset((NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG))

_REQUIRED_ARTIFACT_NAMES = {
    "local_library": "lib/libemender_ndp.so.1",
    "transport_library": "lib/libemender_ndp_transport.so.1",
    "service_binary": "bin/ndp_cxi_service",
}
_GATE_ARTIFACT_NAMES = {
    "synthetic_gate_binary": "bin/ndp_frontier_2n_gate",
}
_ARTIFACT_NAMES = {**_REQUIRED_ARTIFACT_NAMES, **_GATE_ARTIFACT_NAMES}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True).strip()


def _cache_value(cache: Path, name: str) -> str:
    if not cache.is_file():
        raise ValueError(f"CMake cache is missing: {cache}")
    prefix = f"{name}:"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    raise ValueError(f"CMake cache does not contain {name}")


def _artifact_path(prefix: Path, relative: str) -> Path:
    direct = prefix / relative
    if direct.is_file():
        return direct
    # GNUInstallDirs may select lib64 on some Frontier configurations.
    if relative.startswith("lib/"):
        alternate = prefix / "lib64" / relative.split("/", 1)[1]
        if alternate.is_file():
            return alternate
    raise ValueError(f"installed native artifact is missing: {relative}")


def _bundle_digest(artifacts: Mapping[str, Mapping[str, object]]) -> str:
    bound = [(name, str(value["sha256"]))
             for name, value in sorted(artifacts.items())]
    return hashlib.sha256(_canonical(bound)).hexdigest()


def record_build_manifest(*, prefix: str | Path, source_root: str | Path,
                          cmake_cache: str | Path,
                          output: str | Path | None = None) -> Path:
    """Write the deterministic identity of one unified native installation."""
    prefix, source_root = Path(prefix).resolve(), Path(source_root).resolve()
    cache = Path(cmake_cache).resolve()
    artifacts: dict[str, dict[str, object]] = {}
    for name, expected in _ARTIFACT_NAMES.items():
        path = _artifact_path(prefix, expected)
        artifacts[name] = {
            "path": path.relative_to(prefix).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    status = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--quiet", "--ignore-submodules", "--"],
        check=False,
    )
    cached_build_type = _cache_value(cache, "CMAKE_BUILD_TYPE")
    manifest = {
        "schema": BUILD_SCHEMA,
        "source_commit": _git(source_root, "rev-parse", "HEAD"),
        "source_tree_dirty": status.returncode != 0,
        "protocol_version": "1.0",
        "local_abi": 0x00010000,
        "transport_abi": 0x00010000,
        "build": {
            "cmake_version": subprocess.check_output(
                ["cmake", "--version"], text=True).splitlines()[0],
            "build_type": cached_build_type,
            "c_compiler": _cache_value(cache, "CMAKE_C_COMPILER"),
            "cxx_compiler": _cache_value(cache, "CMAKE_CXX_COMPILER"),
            "xpmem_enabled": _cache_value(cache, "NDP_ENABLE_XPMEM"),
            "tests_enabled": _cache_value(cache, "NDP_BUILD_TESTS"),
        },
        "artifacts": artifacts,
    }
    manifest["bundle_sha256"] = _bundle_digest(artifacts)
    target = Path(output).resolve() if output else prefix / "native-artifacts.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical(manifest) + b"\n")
    os.replace(temporary, target)
    return target


@dataclass(frozen=True)
class BuildAttestation:
    path: Path
    source_commit: str
    source_tree_dirty: bool
    bundle_sha256: str
    artifacts: Mapping[str, Mapping[str, object]]


def _forbidden_symbols(path: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["nm", "-D", "--defined-only", str(path)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot inspect native symbols for {path}: {result.stderr.strip()}")
    forbidden = []
    for line in result.stdout.splitlines():
        symbol = line.rsplit(maxsplit=1)[-1] if line.split() else ""
        if symbol.startswith(("MPI_", "PMPI_")):
            forbidden.append(symbol)
    return tuple(sorted(set(forbidden)))


def validate_build_manifest(path: str | Path, *,
                            source_root: str | Path | None = None,
                            require_clean: bool = False) -> BuildAttestation:
    path = Path(path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"native build manifest cannot be read: {path}") from error
    if value.get("schema") != BUILD_SCHEMA:
        raise ValueError("native build manifest schema mismatch")
    if value.get("local_abi") != 0x00010000 or value.get("transport_abi") != 0x00010000:
        raise ValueError("native local/transport ABI major mismatch")
    if require_clean and value.get("source_tree_dirty") is not False:
        raise ValueError("production native build was recorded from a dirty source tree")
    artifacts = value.get("artifacts")
    artifact_names = set(artifacts) if isinstance(artifacts, dict) else set()
    accepted_artifact_sets = (
        set(_REQUIRED_ARTIFACT_NAMES),
        set(_ARTIFACT_NAMES),
    )
    if not isinstance(artifacts, dict) or artifact_names not in accepted_artifact_sets:
        raise ValueError("native build manifest artifact set is incomplete")
    for name, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid native artifact record: {name}")
        artifact = (path.parent / str(record.get("path", ""))).resolve()
        try:
            artifact.relative_to(path.parent)
        except ValueError as error:
            raise ValueError("native artifact path escapes its install prefix") from error
        if (not artifact.is_file() or artifact.stat().st_size != int(record.get("bytes", -1))
                or sha256_file(artifact) != record.get("sha256")):
            raise ValueError(f"native artifact digest/size mismatch: {name}")
        forbidden = _forbidden_symbols(artifact)
        if forbidden:
            raise ValueError(f"elastic native artifact exports collective symbols: {forbidden}")
    bundle = _bundle_digest(artifacts)
    if bundle != value.get("bundle_sha256"):
        raise ValueError("native bundle digest mismatch")
    source_commit = str(value.get("source_commit", ""))
    if len(source_commit) != 40:
        raise ValueError("native build source commit is invalid")
    if source_root is not None:
        current = _git(Path(source_root).resolve(), "rev-parse", "HEAD")
        if source_commit != current:
            raise ValueError("native build does not match the launched source commit")
    return BuildAttestation(
        path, source_commit, bool(value.get("source_tree_dirty")), bundle, artifacts)


def validate_backend(backend: str, *, production: bool, full_layout: bool) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported resilient data-plane backend: {backend}")
    if backend == PYTHON_TCP_DEBUG and (production or full_layout):
        raise ValueError("Python TCP is a small debug fixture and is forbidden for production/full-layout launchers")
    if backend == NATIVE_TEST and (production or full_layout):
        raise ValueError("test native providers cannot be promoted to production/full-layout")
    if production and backend != NATIVE_CXI:
        raise ValueError("production resilient launch requires native-cxi")
    return backend


def validate_g2_gate(
    path: str | Path,
    build: BuildAttestation,
    *,
    required_gate: str = "G2",
) -> Mapping[str, object]:
    if required_gate not in {"G2", "G2-fault-rejoin-replay"}:
        raise ValueError(f"unsupported native G2 gate kind: {required_gate}")
    path = Path(path).resolve()
    try:
        gate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"native G2 gate cannot be read: {path}") from error
    required = {
        "schema": GATE_SCHEMA, "gate": required_gate, "status": "passed",
        "source_commit": build.source_commit,
        "bundle_sha256": build.bundle_sha256,
        "provider": "cxi", "endpoint_type": "FI_EP_RDM",
        "production_provider": True,
        "layout_bytes": 5_506_770_496, "shard_count": 83,
        "trainers_per_node": 8, "nodes": 2,
        "logical_contribution_bytes": 11_013_540_992,
        "logical_redistribution_bytes": 11_013_540_992,
        "python_dense_socket_bytes": 0, "trainer_spool_bytes": 0,
        "disk_replay_bytes": 0, "handoff_full_copy_bytes": 0,
    }
    mismatches = {name: (gate.get(name), expected) for name, expected in required.items()
                  if gate.get(name) != expected}
    if mismatches:
        raise ValueError(f"native G2 gate identity/metrics mismatch: {mismatches}")
    if required_gate == "G2-fault-rejoin-replay":
        fault = gate.get("fault")
        required_fault = {
            "peer_loss": True,
            "new_incarnation": True,
            "old_epoch_rejected": True,
            "partial_commit": False,
            "reassignment_count": 1,
            "replay_bytes": 134_217_728,
        }
        if not isinstance(fault, Mapping):
            raise ValueError("native G2 fault identity is missing")
        fault_mismatches = {
            name: (fault.get(name), expected)
            for name, expected in required_fault.items()
            if fault.get(name) != expected
        }
        if fault_mismatches:
            raise ValueError(
                f"native G2 fault identity/metrics mismatch: "
                f"{fault_mismatches}")
    if set(build.artifacts) != set(_ARTIFACT_NAMES):
        raise ValueError("native G2 build does not attest the synthetic gate executable")
    artifacts = gate.get("artifacts")
    expected_artifacts = {name: record["sha256"]
                          for name, record in build.artifacts.items()}
    if artifacts != expected_artifacts:
        raise ValueError("native G2 gate does not attest every launched artifact digest")
    return gate


def attest_launch(*, backend: str, production: bool, full_layout: bool,
                  build_manifest: str | Path | None,
                  gate_json: str | Path | None,
                  source_root: str | Path | None = None,
                  required_gate: str = "G2") -> dict[str, object]:
    backend = validate_backend(backend, production=production, full_layout=full_layout)
    if backend == PYTHON_TCP_DEBUG:
        if build_manifest or gate_json:
            raise ValueError("Python TCP debug fixture cannot consume native promotion evidence")
        return {"backend": backend, "production": False, "full_layout": False,
                "status": "debug-fixture"}
    if build_manifest is None:
        raise ValueError("native backend requires an attested build manifest")
    build = validate_build_manifest(
        build_manifest, source_root=source_root,
        require_clean=production or full_layout,
    )
    gate = None
    if production or full_layout:
        if gate_json is None:
            raise ValueError("production/full-layout native backend requires the exact G2 gate")
        gate = validate_g2_gate(
            gate_json, build, required_gate=required_gate)
    elif gate_json is not None:
        gate = validate_g2_gate(
            gate_json, build, required_gate=required_gate)
    return {
        "backend": backend, "production": production,
        "full_layout": full_layout, "status": "attested",
        "source_commit": build.source_commit,
        "bundle_sha256": build.bundle_sha256,
        "build_manifest": str(build.path),
        "gate": None if gate is None else str(Path(gate_json).resolve()),
        "artifacts": {name: record["sha256"]
                      for name, record in build.artifacts.items()},
    }
