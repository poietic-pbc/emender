"""Native service lifecycle bound to the resilient Python control plane.

The allocation holder continues to own leases, READY membership, generation
freeze, checkpoint policy, and atomic publication.  This module owns the two
compiled ABI handles as one manager-scoped resource: the local exact reducer
and the libfabric endpoint start before READY, native routes are installed
from leased endpoint records, and TERM/normal cleanup drains both without a
peer rendezvous.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from ndm.native_artifacts import (
    NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG, attest_launch,
)
from ndm.native_dataplane import Client, Command, NativeLibrary, Role
from ndm.native_transport import (
    NativeTransport, NativeTransportLibrary, decode_endpoint_record,
)
from ndm.resilient_pool_runtime import OwnerEndpoint


@dataclass(frozen=True)
class NativeServiceTelemetry:
    backend: str
    source_commit: str
    artifact_bundle_sha256: str
    provider: str
    endpoint_epoch: int
    live_peers: int
    local: Mapping[str, int]
    transport: Mapping[str, object]
    terminal_reason: str


def _artifact_paths(manifest_path: str | Path) -> dict[str, Path]:
    path = Path(manifest_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, record in value["artifacts"].items():
        artifact = (path.parent / record["path"]).resolve()
        artifact.relative_to(path.parent)
        result[name] = artifact
    return result


class NativeManagerSession:
    """One model-free manager's joined local/fabric ABI lifecycle."""

    def __init__(self, *, backend: str, attestation: Mapping[str, object],
                 local: Client, transport: NativeTransport,
                 endpoint: OwnerEndpoint, telemetry_path: Path | None,
                 telemetry_fd: int):
        self.backend, self.attestation = backend, dict(attestation)
        self.local, self.transport, self.owner_endpoint = local, transport, endpoint
        self.telemetry_path, self.telemetry_fd = telemetry_path, telemetry_fd
        self.routes: dict[str, int] = {}
        self.closed = False

    @classmethod
    def start(cls, *, backend: str, run_id: str, fence_epoch: int,
              worker_id: str, incarnation: str, host: str,
              build_manifest: str | Path, gate_json: str | Path | None,
              source_root: str | Path, production: bool, full_layout: bool,
              deadline_s: float, telemetry_path: str | Path | None = None,
              payload_max: int = 64 << 20,
              resident_limit_bytes: int = 16 << 30) -> "NativeManagerSession":
        if backend not in {NATIVE_CXI, NATIVE_TEST}:
            raise ValueError("NativeManagerSession cannot start a Python TCP fixture")
        attestation = attest_launch(
            backend=backend, production=production, full_layout=full_layout,
            build_manifest=build_manifest, gate_json=gate_json,
            source_root=source_root,
        )
        artifacts = _artifact_paths(build_manifest)
        telemetry = Path(telemetry_path) if telemetry_path else None
        telemetry_fd = -1
        if telemetry is not None:
            telemetry.parent.mkdir(parents=True, exist_ok=True)
            telemetry_fd = os.open(
                telemetry, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
                0o600)
        local = None
        transport = None
        try:
            local = Client.open(
                library=NativeLibrary(artifacts["local_library"]), role=Role.CONTROLLER,
                run_key=run_id, fence_epoch=fence_epoch, worker_key=worker_id,
                incarnation=incarnation, deadline_s=min(10.0, deadline_s),
            )
            provider = "cxi" if backend == NATIVE_CXI else os.environ.get(
                "NDP_TEST_PROVIDER", "tcp;ofi_rxm")
            transport = NativeTransport.open(
                library=NativeTransportLibrary(artifacts["transport_library"]),
                provider=provider, production=production, bind_node=host,
                deadline_s=deadline_s, payload_max=payload_max,
                resident_limit_bytes=resident_limit_bytes,
                telemetry_fd=telemetry_fd,
            )
            endpoint_epoch = max(1, time.time_ns())
            expires_unix_ns = min(
                transport.deadline_unix_ns,
                time.time_ns() + int(deadline_s * 1e9),
            )
            record = transport.bind(
                run_key=run_id, fence_epoch=fence_epoch, worker_key=worker_id,
                incarnation=incarnation, endpoint_epoch=endpoint_epoch,
                expires_unix_ns=expires_unix_ns,
            )
            endpoint = OwnerEndpoint(
                worker_id, incarnation, host, 0, backend,
                record.encoded.hex(), record.provider, record.endpoint_epoch,
                record.expires_unix_ns, str(attestation["bundle_sha256"]),
            )
            return cls(
                backend=backend, attestation=attestation, local=local,
                transport=transport, endpoint=endpoint,
                telemetry_path=telemetry, telemetry_fd=telemetry_fd,
            )
        except BaseException:
            if transport is not None:
                transport.close()
            if local is not None:
                local.close()
            if telemetry_fd >= 0:
                os.close(telemetry_fd)
            raise

    def install_routes(self, peers: Sequence[OwnerEndpoint]) -> Mapping[str, int]:
        """Install only leased, same-backend endpoint records from Python."""
        expected_bundle = str(self.attestation["bundle_sha256"])
        desired = {}
        for peer in sorted(peers, key=lambda item: item.worker_id):
            if peer.worker_id == self.owner_endpoint.worker_id:
                continue
            if (peer.backend != self.backend
                    or peer.artifact_bundle_sha256 != expected_bundle):
                raise ValueError("native route backend/artifact differs from frozen membership")
            record = decode_endpoint_record(bytes.fromhex(peer.endpoint_record))
            desired[peer.worker_id] = self.transport.upsert(record)
        for worker_id, peer_id in tuple(self.routes.items()):
            if worker_id not in desired:
                self.transport.remove(peer_id)
        self.routes = desired
        return dict(self.routes)

    def telemetry(self, terminal_reason: str = "running") -> NativeServiceTelemetry:
        local = asdict(self.local.metrics)
        transport = asdict(self.transport.metrics)
        return NativeServiceTelemetry(
            self.backend, str(self.attestation["source_commit"]),
            str(self.attestation["bundle_sha256"]),
            self.owner_endpoint.provider, self.owner_endpoint.endpoint_epoch,
            int(transport["live_peers"]), local, transport, terminal_reason,
        )

    def write_readiness(self, path: str | Path) -> Path:
        """Publish the bounded node-local pre-trainer READY attestation."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        value = asdict(self.telemetry("native_service_ready"))
        value.update({
            "schema": "emender-native-manager-ready-v1",
            "endpoint_record_sha256": hashlib.sha256(
                bytes.fromhex(self.owner_endpoint.endpoint_record)).hexdigest(),
            "python_dense_socket_bytes": 0,
            "trainer_spool_bytes": 0,
        })
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target

    def close(self, terminal_reason: str = "normal") -> NativeServiceTelemetry:
        if self.closed:
            raise RuntimeError("native manager session is already closed")
        # Local DRAIN is bounded by the caller's absolute process deadline. It
        # has no peer rendezvous; transport route removal/cancel is likewise
        # independent for each current endpoint.
        try:
            try:
                self.local.control(Command.DRAIN, deadline_s=30.0).close()
            except Exception:
                try:
                    self.local.control(Command.ABORT, deadline_s=1.0).close()
                except Exception:
                    pass
            for peer_id in tuple(self.routes.values()):
                try:
                    self.transport.library.check(
                        self.transport.library.lib.ndp_transport_cancel_v1(
                            self.transport.handle, peer_id), "transport_cancel")
                except Exception:
                    pass
            final = self.telemetry(terminal_reason)
        finally:
            self.transport.close()
            self.local.close()
            if self.telemetry_fd >= 0:
                os.close(self.telemetry_fd)
                self.telemetry_fd = -1
            self.closed = True
        return final

    def __enter__(self) -> "NativeManagerSession":
        return self

    def __exit__(self, type_, _value, _traceback) -> None:
        self.close("exception" if type_ is not None else "normal")


__all__ = ["NativeManagerSession", "NativeServiceTelemetry", "PYTHON_TCP_DEBUG"]
