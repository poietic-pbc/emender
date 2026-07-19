"""Backend-neutral, process-local adapter for resilient Compute Pool v1.

The adapter deliberately supplies no scheduler semantics.  It maps local host
agents onto the same durable allocation fence, leased READY membership,
generation admission, contribution receipts, deterministic shard-owner plan,
native v1 ABI, and fenced checkpoint publication used by the Frontier adapter.
``native-test`` is the only accepted backend here; the production ``native-cxi``
provider gate remains unchanged and cannot be selected through this fixture.

This module is a qualification adapter, not a new protocol implementation.
Dense shard reduction goes through producer-direct native memfd buffers.  TCP
is used only by the existing bounded metadata control protocol, and every
generation is closed from a live READY snapshot rather than a launched world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import mmap
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from ndm.fenced_admission import (
    AllocationLease,
    FenceRejected,
    SQLiteFencedControlStore,
)
from ndm.native_artifacts import NATIVE_TEST, BuildAttestation, validate_build_manifest
from ndm.native_dataplane import DType
from ndm.native_pool_runtime import NativeManagerSession, NativeServiceTelemetry
from ndm.resilient_e97_reducer import TensorLayout
from ndm.resilient_pool_runtime import (
    OwnerEndpoint,
    PoolControlClient,
    PoolControlConfig,
    PoolControlServer,
    PoolStageSLO,
)


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_SessionFactory = Callable[..., NativeManagerSession]


class OwnerUnavailable(RuntimeError):
    """A frozen snapshot owner disappeared before native execution."""


@dataclass(frozen=True)
class HyperscaleLocalConfig:
    """Scheduler-free adapter configuration with explicit finite bounds."""

    run_id: str
    allocation_id: str
    allocation_incarnation: str
    protocol_id: str
    config_id: str
    control_db: Path
    evidence_root: Path
    build_manifest: Path
    source_root: Path
    lease_ttl_s: float = 120.0
    q_min: int = 2
    t_min: int = 2
    ready_fraction: float | None = None
    payload_max: int = 1 << 20
    resident_limit_bytes: int = 64 << 20
    session_deadline_s: float = 120.0
    base_digest: str = hashlib.sha256(b"hyperscale-local-base-v1").hexdigest()
    policy_digest: str = hashlib.sha256(b"hyperscale-local-policy-v1").hexdigest()
    layout_digest: str = hashlib.sha256(b"hyperscale-local-layout-v1").hexdigest()
    code_digest: str = hashlib.sha256(b"hyperscale-local-code-v1").hexdigest()
    # A spawned local host agent imports the full Python/native bridge before
    # READY.  These are finite qualification bounds, not performance targets;
    # they remain below the Frontier production generation bound.
    slo: PoolStageSLO = PoolStageSLO(
        30.0, 180.0, 60.0, 60.0, 30.0, 60.0, 60.0, 30.0)

    def __post_init__(self) -> None:
        identities = (
            self.run_id,
            self.allocation_id,
            self.allocation_incarnation,
            self.protocol_id,
            self.config_id,
        )
        if not all(identities):
            raise ValueError("complete local allocation identities are required")
        if not all(_SAFE_COMPONENT.fullmatch(value) for value in identities[:3]):
            raise ValueError("run/allocation identities must be safe path components")
        if self.lease_ttl_s <= 0 or self.session_deadline_s <= 0:
            raise ValueError("lease and session deadlines must be positive")
        if self.payload_max < 64 or self.resident_limit_bytes < self.payload_max:
            raise ValueError("native payload/resident bounds are invalid")
        for name in ("base_digest", "policy_digest", "layout_digest", "code_digest"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                bytes.fromhex(value)
            except ValueError as error:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from error


@dataclass(frozen=True)
class OpenedGeneration:
    run_id: str
    fence: int
    generation: int
    attempt: int
    peers: tuple[OwnerEndpoint, ...]
    observed_at: float

    @property
    def identities(self) -> tuple[tuple[str, str], ...]:
        return tuple((peer.worker_id, peer.incarnation) for peer in self.peers)


@dataclass(frozen=True)
class LocalContribution:
    worker_id: str
    incarnation: str
    contribution_seq: int
    accepted_tokens: int
    values: tuple[float, ...]

    @classmethod
    def create(cls, worker_id: str, incarnation: str, contribution_seq: int,
               accepted_tokens: int, values: Sequence[float]) -> "LocalContribution":
        array = np.asarray(values, dtype=np.float32)
        if (not worker_id or not incarnation or contribution_seq < 0
                or accepted_tokens <= 0 or array.ndim != 1 or array.size == 0
                or not np.isfinite(array).all()):
            raise ValueError("fresh finite contribution identity/values are required")
        return cls(worker_id, incarnation, int(contribution_seq),
                   int(accepted_tokens), tuple(float(value) for value in array))

    def array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)

    def digest(self) -> str:
        return hashlib.sha256(self.array().astype("<f4", copy=False).tobytes()).hexdigest()


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    attempt: int
    fence: int
    ready_snapshot: tuple[tuple[str, str], ...]
    accepted_identities: tuple[tuple[str, str, int], ...]
    accepted_tokens: int
    receipts: tuple[Mapping[str, object], ...]
    owner_by_shard: Mapping[int, str]
    aggregate: tuple[float, ...]
    checkpoint_path: Path
    checkpoint_sha256: str
    publication_manifest: Path
    manifest_sha256: str


@dataclass
class _Worker:
    worker_id: str
    incarnation: str
    generation: int
    session: "_NativeProcessSession"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _atomic_immutable(path: Path, payload: bytes) -> None:
    """Create immutable-by-name evidence, permitting only identical replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FenceRejected(f"immutable evidence already differs: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _endpoint_from_snapshot(value: Mapping[str, object]) -> OwnerEndpoint:
    return OwnerEndpoint(
        worker_id=str(value["worker_id"]), incarnation=str(value["incarnation"]),
        host=str(value["host"]), port=int(value["port"]),
        backend=str(value["backend"]), endpoint_record=str(value["endpoint_record"]),
        provider=str(value["provider"]), endpoint_epoch=int(value["endpoint_epoch"]),
        expires_unix_ns=int(value["expires_unix_ns"]),
        artifact_bundle_sha256=str(value["artifact_bundle_sha256"]),
    )


def _native_process_main(connection: Connection, start_kwargs: Mapping[str, object],
                         session_factory: _SessionFactory) -> None:
    """Own exactly one native controller in one host-agent process."""
    session: NativeManagerSession | None = None
    buffers: dict[int, object] = {}
    views: dict[int, tuple[object, object]] = {}
    next_token = 1
    try:
        session = session_factory(**dict(start_kwargs))
        connection.send({"ok": True, "endpoint": asdict(session.owner_endpoint)})
        while True:
            request = connection.recv()
            operation = str(request["operation"])
            try:
                if operation == "install_routes":
                    peers = tuple(_endpoint_from_snapshot(value)
                                  for value in request["peers"])
                    connection.send({"ok": True, "routes": dict(session.install_routes(peers))})
                elif operation == "write_readiness":
                    path = session.write_readiness(str(request["path"]))
                    connection.send({"ok": True, "path": str(path)})
                elif operation == "install_generation":
                    session.install_generation(
                        total_elements=int(request["total_elements"]),
                        generation=int(request["generation"]),
                        attempt=int(request["attempt"]),
                        owner_epoch=int(request["owner_epoch"]),
                        source_dtype=DType(int(request["source_dtype"])),
                        payload_max=int(request["payload_max"]),
                        base_digest=bytes(request["base_digest"]),
                        plan_digest=bytes(request["plan_digest"]),
                        deadline_s=float(request["deadline_s"]))
                    connection.send({"ok": True})
                elif operation == "allocate":
                    buffer = session.allocate_trainer_buffer(
                        deadline_s=float(request["deadline_s"]))
                    token = next_token
                    next_token += 1
                    buffers[token] = buffer
                    connection.send({
                        "ok": True, "token": token, "fd": DupFd(buffer.fd),
                        "length": buffer.length,
                    })
                elif operation == "submit":
                    token = int(request["token"])
                    buffer = buffers.pop(token)
                    try:
                        buffer.seal()
                        with session.submit_local(
                                buffer, trainer_key=str(request["trainer_key"]),
                                trainer_incarnation=str(request["trainer_incarnation"]),
                                submission_seq=int(request["submission_seq"]),
                                weight=int(request["weight"]),
                                deadline_s=float(request["deadline_s"])):
                            pass
                    finally:
                        buffer.close()
                    connection.send({"ok": True})
                elif operation == "freeze":
                    with session.freeze(deadline_s=float(request["deadline_s"])):
                        pass
                    connection.send({"ok": True})
                elif operation == "finalize":
                    native_operation, result = session.finalize_redistribution(
                        deadline_s=float(request["deadline_s"]))
                    token = next_token
                    next_token += 1
                    views[token] = (native_operation, result)
                    connection.send({
                        "ok": True, "token": token, "fd": DupFd(result.fd),
                        "length": result.length, "dtype": int(result.dtype),
                    })
                elif operation == "checkpoint_proposal":
                    token = int(request["token"])
                    native_operation, result = views.pop(token)
                    try:
                        path = session.checkpoint_proposal(
                            str(request["path"]), result,
                            publisher=str(request["publisher"]),
                            metadata=dict(request["metadata"]))
                        value = json.loads(path.read_text(encoding="utf-8"))
                    finally:
                        result.close()
                        native_operation.close()
                    connection.send({"ok": True, "proposal": value})
                elif operation == "commit":
                    session.commit(
                        publication_manifest=str(request["publication_manifest"]),
                        authoritative_latest=dict(request["authoritative_latest"]),
                        deadline_s=float(request["deadline_s"]))
                    connection.send({"ok": True})
                elif operation == "abort":
                    session.abort(deadline_s=float(request["deadline_s"]))
                    connection.send({"ok": True})
                elif operation == "telemetry":
                    connection.send({
                        "ok": True,
                        "telemetry": asdict(session.telemetry(str(request["reason"])))})
                elif operation == "close":
                    value = session.close(str(request["reason"]))
                    session = None
                    connection.send({"ok": True, "telemetry": asdict(value)})
                    break
                else:
                    raise ValueError(f"unsupported native host-agent operation: {operation}")
            except BaseException as error:
                connection.send({
                    "ok": False, "error": f"{type(error).__name__}: {error}"})
    except BaseException as error:
        try:
            connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
        except Exception:
            pass
    finally:
        for native_operation, result in views.values():
            try:
                result.close()
                native_operation.close()
            except Exception:
                pass
        for buffer in buffers.values():
            try:
                buffer.close()
            except Exception:
                pass
        if session is not None:
            try:
                session.close("host_agent_process_exit")
            except Exception:
                pass
        connection.close()


class _NativeProcessSession:
    """Metadata/FD-only proxy for one isolated native host-agent process."""

    def __init__(self, process: multiprocessing.Process, connection: Connection,
                 endpoint: OwnerEndpoint, timeout_s: float):
        self.process, self.connection = process, connection
        self.owner_endpoint, self.timeout_s = endpoint, float(timeout_s)
        self.closed = False

    @classmethod
    def start(cls, *, session_factory: _SessionFactory, timeout_s: float,
              **start_kwargs: object) -> "_NativeProcessSession":
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(
            target=_native_process_main,
            args=(child, dict(start_kwargs), session_factory),
            name=f"native-host-{start_kwargs['worker_id']}", daemon=True)
        process.start()
        child.close()
        try:
            if not parent.poll(timeout_s):
                raise TimeoutError("native host-agent startup deadline expired")
            response = parent.recv()
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error")))
            endpoint = _endpoint_from_snapshot(response["endpoint"])
            return cls(process, parent, endpoint, timeout_s)
        except BaseException:
            parent.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5.0)
            process.close()
            raise

    @property
    def pid(self) -> int:
        if self.process.pid is None:
            raise RuntimeError("native host-agent has no process id")
        return self.process.pid

    def _request(self, operation: str, **fields: object) -> Mapping[str, object]:
        if self.closed:
            raise RuntimeError("native host-agent proxy is closed")
        if not self.process.is_alive():
            raise RuntimeError(
                f"native host-agent exited before {operation}: {self.process.exitcode}")
        self.connection.send({"operation": operation, **fields})
        if not self.connection.poll(self.timeout_s):
            raise TimeoutError(f"native host-agent {operation} deadline expired")
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error")))
        return response

    def install_routes(self, peers: Sequence[OwnerEndpoint]) -> Mapping[str, int]:
        response = self._request(
            "install_routes", peers=tuple(asdict(peer) for peer in peers))
        return dict(response["routes"])

    def write_readiness(self, path: str | Path) -> Path:
        response = self._request("write_readiness", path=str(path))
        return Path(str(response["path"]))

    def install_generation(self, *, total_elements: int, generation: int,
                           attempt: int, owner_epoch: int, source_dtype: DType,
                           payload_max: int, base_digest: bytes, plan_digest: bytes,
                           deadline_s: float) -> None:
        self._request(
            "install_generation", total_elements=total_elements,
            generation=generation, attempt=attempt, owner_epoch=owner_epoch,
            source_dtype=int(source_dtype), payload_max=payload_max,
            base_digest=base_digest, plan_digest=plan_digest,
            deadline_s=deadline_s)

    def submit_values(self, values: np.ndarray, *, trainer_key: str,
                      trainer_incarnation: str, submission_seq: int,
                      weight: int, deadline_s: float) -> None:
        allocation = self._request("allocate", deadline_s=deadline_s)
        descriptor = allocation["fd"].detach()
        try:
            length = int(allocation["length"])
            if length != values.size * np.dtype(np.float32).itemsize:
                raise RuntimeError("native allocated buffer differs from owner shard")
            with mmap.mmap(descriptor, length, access=mmap.ACCESS_WRITE) as mapping:
                target = np.ndarray((values.size,), dtype=np.float32, buffer=mapping)
                target[:] = values
                del target
                mapping.flush()
        finally:
            os.close(descriptor)
        self._request(
            "submit", token=int(allocation["token"]), trainer_key=trainer_key,
            trainer_incarnation=trainer_incarnation,
            submission_seq=submission_seq, weight=weight,
            deadline_s=deadline_s)

    def freeze(self, *, deadline_s: float) -> None:
        self._request("freeze", deadline_s=deadline_s)

    def finalize_result(self, *, deadline_s: float) -> tuple[int, int, int]:
        response = self._request("finalize", deadline_s=deadline_s)
        return (int(response["token"]), response["fd"].detach(),
                int(response["length"]))

    def checkpoint_proposal(self, token: int, path: str | Path, *, publisher: str,
                            metadata: Mapping[str, object]) -> Mapping[str, object]:
        response = self._request(
            "checkpoint_proposal", token=token, path=str(path),
            publisher=publisher, metadata=dict(metadata))
        return dict(response["proposal"])

    def commit(self, *, publication_manifest: str | Path,
               authoritative_latest: Mapping[str, object], deadline_s: float) -> None:
        self._request(
            "commit", publication_manifest=str(publication_manifest),
            authoritative_latest=dict(authoritative_latest), deadline_s=deadline_s)

    def abort(self, *, deadline_s: float) -> None:
        self._request("abort", deadline_s=deadline_s)

    def telemetry(self, reason: str = "running") -> NativeServiceTelemetry:
        response = self._request("telemetry", reason=reason)
        return NativeServiceTelemetry(**response["telemetry"])

    def close(self, reason: str) -> NativeServiceTelemetry:
        if self.closed:
            raise RuntimeError("native host-agent proxy is already closed")
        response = self._request("close", reason=reason)
        self.closed = True
        self.connection.close()
        self.process.join(timeout=min(self.timeout_s, 10.0))
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
            self.process.close()
            raise TimeoutError("native host-agent did not exit after bounded drain")
        exitcode = self.process.exitcode
        self.process.close()  # release the parent-side sentinel descriptor
        if exitcode != 0:
            raise RuntimeError(f"native host-agent exited nonzero: {exitcode}")
        return NativeServiceTelemetry(**response["telemetry"])


class HyperscaleLocalAdapter:
    """One admitted local allocation using the production pool contracts."""

    def __init__(self, *, config: HyperscaleLocalConfig,
                 store: SQLiteFencedControlStore, lease: AllocationLease,
                 build: BuildAttestation, session_factory: _SessionFactory):
        self.config, self.store, self.lease, self.build = config, store, lease, build
        self._session_factory = session_factory
        self._workers: dict[str, _Worker] = {}
        self._departures: list[NativeServiceTelemetry] = []
        self._closed = False
        pool_config = PoolControlConfig(
            run_id=config.run_id, fence=lease.fence, q_min=config.q_min,
            t_min=config.t_min, ready_fraction=config.ready_fraction,
            base_digest=config.base_digest, policy_digest=config.policy_digest,
            layout_digest=config.layout_digest, code_digest=config.code_digest,
            slo=config.slo, dataplane_backend=NATIVE_TEST, production=False,
            full_layout=False, artifact_bundle_sha256=build.bundle_sha256,
        )
        self._server = PoolControlServer(
            ("127.0.0.1", 0), pool_config,
            evidence_root=config.evidence_root / "control-evidence")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"hyperscale-local-{config.run_id}", daemon=True)
        self._thread.start()
        self._client = PoolControlClient(self._server.server_address, timeout_s=5.0).bind(
            config.run_id, lease.fence)

    @classmethod
    def try_start(cls, config: HyperscaleLocalConfig, *,
                  session_factory: _SessionFactory = NativeManagerSession.start
                  ) -> "HyperscaleLocalAdapter | None":
        """Acquire before artifact inspection, server start, or native work.

        Returning ``None`` is the successful zero-work loser path.  In
        particular, a losing caller may supply a nonexistent manifest and it
        still performs only the lease CAS required to discover the winner.
        """
        store = SQLiteFencedControlStore(config.control_db)
        lease = store.acquire(
            run_id=config.run_id, allocation_id=config.allocation_id,
            incarnation=config.allocation_incarnation,
            protocol_id=config.protocol_id, config_id=config.config_id,
            ttl_s=config.lease_ttl_s)
        if lease is None:
            return None
        try:
            build = validate_build_manifest(
                config.build_manifest, source_root=config.source_root,
                require_clean=False)
            return cls(config=config, store=store, lease=lease, build=build,
                       session_factory=session_factory)
        except BaseException:
            store.release(lease)
            raise

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address

    @property
    def workers(self) -> Mapping[str, tuple[str, int]]:
        return {key: (value.incarnation, value.generation)
                for key, value in self._workers.items()}

    @property
    def departures(self) -> tuple[NativeServiceTelemetry, ...]:
        return tuple(self._departures)

    @property
    def native_process_ids(self) -> tuple[int, ...]:
        """Live host-agent PIDs for external bounded-resource attestation."""
        return tuple(sorted(worker.session.pid for worker in self._workers.values()))

    def _assert_live(self) -> None:
        if self._closed:
            raise RuntimeError("hyperscale-local adapter is closed")
        self.store.assert_current(self.lease)

    def renew_lease(self) -> AllocationLease:
        self._assert_live()
        self.lease = self.store.renew(self.lease, ttl_s=self.config.lease_ttl_s)
        return self.lease

    def join_worker(self, worker_id: str, incarnation: str, *, generation: int
                    ) -> Mapping[str, object]:
        """Start one native host agent and advertise its leased endpoint."""
        self._assert_live()
        if (not _SAFE_COMPONENT.fullmatch(worker_id)
                or not _SAFE_COMPONENT.fullmatch(incarnation) or generation < 0):
            raise ValueError("worker identity/incarnation/generation is invalid")
        if worker_id in self._workers:
            raise ValueError("worker is already active; remove it before rejoining")
        telemetry_path = (
            self.config.evidence_root / "telemetry" /
            f"{worker_id}-{incarnation}.jsonl")
        session = _NativeProcessSession.start(
            session_factory=self._session_factory,
            timeout_s=self.config.session_deadline_s,
            backend=NATIVE_TEST, run_id=self.config.run_id,
            fence_epoch=self.lease.fence, worker_id=worker_id,
            incarnation=incarnation, host="127.0.0.1",
            build_manifest=self.config.build_manifest, gate_json=None,
            source_root=self.config.source_root, production=False,
            full_layout=False, deadline_s=self.config.session_deadline_s,
            telemetry_path=telemetry_path, payload_max=self.config.payload_max,
            resident_limit_bytes=self.config.resident_limit_bytes,
        )
        worker = _Worker(worker_id, incarnation, generation, session)
        try:
            ready = self._client.ready(
                session.owner_endpoint, generation,
                run_id=self.config.run_id, fence=self.lease.fence)
            self._workers[worker_id] = worker
            self._refresh_routes()
            session.write_readiness(
                self.config.evidence_root / "readiness" /
                f"{worker_id}-{incarnation}.json")
            return ready
        except BaseException:
            self._workers.pop(worker_id, None)
            try:
                self._client.drain(worker_id, incarnation)
            except Exception:
                pass
            session.close("join_failed")
            raise

    def sync_worker(self, worker_id: str, *, generation: int) -> Mapping[str, object]:
        """Catch up and renew an existing incarnation for a later generation."""
        self._assert_live()
        worker = self._workers[worker_id]
        ready = self._client.ready(
            worker.session.owner_endpoint, generation,
            run_id=self.config.run_id, fence=self.lease.fence)
        worker.generation = generation
        return ready

    def _refresh_routes(self, peers: Sequence[OwnerEndpoint] | None = None) -> None:
        desired = tuple(peers) if peers is not None else tuple(
            worker.session.owner_endpoint for worker in self._workers.values())
        for worker in tuple(self._workers.values()):
            worker.session.install_routes(desired)

    def remove_worker(self, worker_id: str, *, reason: str = "host_agent_disappeared"
                      ) -> NativeServiceTelemetry:
        """Expire one observed worker without waiting for a launched world."""
        self._assert_live()
        worker = self._workers.pop(worker_id)
        self._client.drain(worker.worker_id, worker.incarnation)
        final = worker.session.close(reason)
        self._departures.append(final)
        self._refresh_routes()
        return final

    def open_generation(self, generation: int, attempt: int = 1) -> OpenedGeneration:
        self._assert_live()
        response = self._client.open_generation(
            generation, attempt, deadline=time.monotonic() + self.config.slo.sync_s)
        peers = tuple(_endpoint_from_snapshot(value) for value in response["peers"])
        return OpenedGeneration(
            run_id=str(response["run_id"]), fence=int(response["fence"]),
            generation=int(response["generation"]), attempt=int(response["attempt"]),
            peers=peers, observed_at=float(response["observed_at"]),
        )

    def _validate_opened(self, opened: OpenedGeneration) -> tuple[_Worker, ...]:
        self._assert_live()
        if (opened.run_id != self.config.run_id or opened.fence != self.lease.fence
                or opened.generation < 0 or opened.attempt <= 0):
            raise FenceRejected("opened generation differs from current allocation fence")
        owners = []
        for endpoint in opened.peers:
            worker = self._workers.get(endpoint.worker_id)
            if worker is None or worker.incarnation != endpoint.incarnation:
                raise OwnerUnavailable(
                    f"frozen owner disappeared: {endpoint.worker_id}/{endpoint.incarnation}")
            owners.append(worker)
        self._refresh_routes(opened.peers)
        return tuple(owners)

    def commit_generation(self, opened: OpenedGeneration,
                          contributions: Sequence[LocalContribution]) -> GenerationResult:
        """Freeze metadata, execute deterministic native shard owners, and commit."""
        owners = self._validate_opened(opened)
        if not owners:
            raise OwnerUnavailable("generation has no native shard owners")
        keys = [(item.worker_id, item.incarnation, item.contribution_seq)
                for item in contributions]
        if len(set(keys)) != len(keys):
            raise ValueError("generation contributions must have unique fenced identities")
        arrays: dict[tuple[str, str, int], np.ndarray] = {}
        receipts: list[Mapping[str, object]] = []
        for item in contributions:
            array = item.array()
            arrays[(item.worker_id, item.incarnation, item.contribution_seq)] = array
            receipts.append(self._client.contribute(
                opened.generation, opened.attempt, item.worker_id, item.incarnation,
                item.contribution_seq, item.accepted_tokens, item.digest()))
        close = self._client.close_generation(
            opened.generation, opened.attempt,
            deadline=time.monotonic() + self.config.slo.freeze_s)
        if close["status"] != "commit_ready":
            raise RuntimeError(f"generation cannot commit: {close['status']}")
        frozen = tuple(
            (str(value["worker_id"]), str(value["incarnation"]),
             int(value["contribution_seq"]))
            for value in close["frozen_identities"])
        by_key = {key: item for key, item in zip(keys, contributions)}
        missing = [identity for identity in frozen if identity not in arrays]
        if missing:
            raise RuntimeError(f"frozen native contribution values are missing: {missing}")
        accepted = tuple(by_key[identity] for identity in frozen)
        total_elements = arrays[frozen[0]].size
        if total_elements < len(owners) or any(
                arrays[identity].size != total_elements for identity in frozen):
            raise ValueError("accepted vectors must share a layout large enough for all owners")

        chunk_elements = math.ceil(total_elements / len(owners))
        layout = TensorLayout.from_flat_stream(
            total_elements, max_chunk_bytes=chunk_elements * 8,
            dtype="torch.float32")
        owner_ids = tuple(worker.worker_id for worker in owners)
        owner_by_shard = {
            shard: layout.owner(
                shard, owner_ids, run_id=opened.run_id,
                generation=opened.generation, attempt=opened.attempt)
            for shard in range(layout.shard_count)
        }
        owned_shards: dict[str, list[int]] = {worker_id: [] for worker_id in owner_ids}
        for shard, worker_id in owner_by_shard.items():
            owned_shards[worker_id].append(shard)
        if any(len(shards) != 1 for shards in owned_shards.values()):
            raise RuntimeError("local qualification requires exactly one bounded shard per owner")

        plan = {
            "run_id": opened.run_id, "fence": opened.fence,
            "generation": opened.generation, "attempt": opened.attempt,
            "frozen": frozen, "layout": layout.digest,
            "owner_by_shard": owner_by_shard,
        }
        plan_digest = hashlib.sha256(_canonical_json(plan)).digest()
        owner_results: dict[str, dict[str, object]] = {}
        aggregate = np.empty(total_elements, dtype=np.float32)
        installed: list[_NativeProcessSession] = []
        try:
            for worker in owners:
                shard = owned_shards[worker.worker_id][0]
                offset = shard * layout.chunk_elements
                elements = min(layout.chunk_elements, total_elements - offset)
                session = worker.session
                session.install_generation(
                    total_elements=elements, generation=opened.generation,
                    attempt=opened.attempt, owner_epoch=1, source_dtype=DType.F32,
                    payload_max=self.config.payload_max,
                    base_digest=bytes.fromhex(self.config.base_digest),
                    plan_digest=plan_digest, deadline_s=self.config.slo.transport_s)
                installed.append(session)
                for item in accepted:
                    values = item.array()[offset:offset + elements]
                    session.submit_values(
                        values, trainer_key=item.worker_id,
                        trainer_incarnation=item.incarnation,
                        submission_seq=item.contribution_seq,
                        weight=item.accepted_tokens,
                        deadline_s=self.config.slo.transport_s)
                session.freeze(deadline_s=self.config.slo.freeze_s)
                token, descriptor, length = session.finalize_result(
                    deadline_s=self.config.slo.apply_s)
                try:
                    if length != elements * np.dtype(np.float32).itemsize:
                        raise RuntimeError("native owner result byte extent is invalid")
                    with mmap.mmap(descriptor, length, flags=mmap.MAP_PRIVATE,
                                   prot=mmap.PROT_READ) as mapping:
                        values = np.ndarray((elements,), dtype=np.float32, buffer=mapping)
                        aggregate[offset:offset + elements] = values
                        del values
                finally:
                    os.close(descriptor)
                proposal_value = session.checkpoint_proposal(
                    token,
                    self.config.evidence_root / "proposals" /
                    f"g{opened.generation:08d}-a{opened.attempt}-{worker.worker_id}.json",
                    publisher=worker.worker_id,
                    metadata={
                        "shard_id": shard, "element_offset": offset,
                        "elements": elements, "frozen_identities": frozen,
                    })
                owner_results[worker.worker_id] = {
                    key: proposal_value[key] for key in (
                        "attempt", "layout_digest", "base_digest",
                        "result_root", "global_weight", "result_bytes")
                }
                owner_results[worker.worker_id].update({
                    "shard_id": shard, "element_offset": offset,
                    "elements": elements,
                })

            weights = np.asarray([item.accepted_tokens for item in accepted], dtype=np.float64)
            reference = np.zeros(total_elements, dtype=np.float64)
            for item, weight in zip(accepted, weights):
                reference += item.array().astype(np.float64) * weight
            reference = (reference / weights.sum()).astype(np.float32)
            if not np.array_equal(aggregate, reference):
                raise RuntimeError("native sharded aggregate differs from exact reference")

            checkpoint_payload = aggregate.astype("<f4", copy=False).tobytes()
            checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
            checkpoint_path = (
                self.config.evidence_root / "checkpoints" /
                f"generation-{opened.generation:08d}-fence-{opened.fence:08d}.f32")
            _atomic_immutable(checkpoint_path, checkpoint_payload)
            if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_sha256:
                raise RuntimeError("checkpoint reload digest mismatch")

            manifest_path = (
                self.config.evidence_root / "manifests" /
                f"generation-{opened.generation:08d}-fence-{opened.fence:08d}.json")
            manifest = {
                "schema": "emender-hyperscale-local-commit-v1",
                "finalized": True, "run_id": opened.run_id,
                "generation": opened.generation, "attempt": opened.attempt,
                "fence": {"coordinator_epoch": opened.fence},
                "ready_snapshot": opened.identities,
                "frozen_identities": frozen,
                "accepted_tokens": int(close["accepted_tokens"]),
                "layout": {
                    "digest": layout.digest, "total_elements": total_elements,
                    "shard_count": layout.shard_count,
                },
                "owner_by_shard": owner_by_shard,
                "owner_results": owner_results,
                "checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha256,
                    "bytes": len(checkpoint_payload), "dtype": "float32-le",
                },
                "artifact_bundle_sha256": self.build.bundle_sha256,
                "backend": NATIVE_TEST,
                "python_dense_socket_bytes": 0,
                "trainer_spool_bytes": 0,
            }
            manifest_payload = _canonical_json(manifest)
            _atomic_immutable(manifest_path, manifest_payload)
            manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
            latest = {
                "generation": opened.generation, "fence": opened.fence,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": manifest_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
            name = f"generation-{opened.generation:08d}"
            self.store.publish_bundle(self.lease, (
                ("commit", name, {
                    "generation": opened.generation, "fence": opened.fence,
                    "manifest_sha256": manifest_sha256,
                    "accepted_tokens": int(close["accepted_tokens"]),
                }),
                ("checkpoint", name, {
                    "generation": opened.generation, "fence": opened.fence,
                    "path": str(checkpoint_path.resolve()),
                    "sha256": checkpoint_sha256,
                }),
                ("latest", "authoritative", latest),
            ))
            authoritative = self.store.read_publication(
                opened.run_id, "latest", "authoritative")
            if authoritative != latest:
                raise RuntimeError("authoritative latest readback differs after atomic CAS")
            for session in installed:
                session.commit(
                    publication_manifest=manifest_path,
                    authoritative_latest=authoritative,
                    deadline_s=self.config.slo.apply_s)
            installed.clear()
            return GenerationResult(
                generation=opened.generation, attempt=opened.attempt,
                fence=opened.fence, ready_snapshot=opened.identities,
                accepted_identities=frozen,
                accepted_tokens=int(close["accepted_tokens"]),
                receipts=tuple(receipts), owner_by_shard=owner_by_shard,
                aggregate=tuple(float(value) for value in aggregate),
                checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256,
                publication_manifest=manifest_path, manifest_sha256=manifest_sha256,
            )
        except BaseException:
            for session in installed:
                try:
                    session.abort(deadline_s=1.0)
                except Exception:
                    pass
            raise

    def close(self, *, release_lease: bool = True) -> tuple[NativeServiceTelemetry, ...]:
        if self._closed:
            return tuple(self._departures)
        for worker_id in tuple(self._workers):
            try:
                self.remove_worker(worker_id, reason="adapter_shutdown")
            except FenceRejected:
                worker = self._workers.pop(worker_id)
                self._departures.append(worker.session.close("lease_lost_shutdown"))
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=self.config.slo.shutdown_s)
        if self._thread.is_alive():
            raise TimeoutError("pool control server did not stop within shutdown bound")
        if release_lease:
            self.store.release(self.lease)
        self._closed = True
        return tuple(self._departures)

    def __enter__(self) -> "HyperscaleLocalAdapter":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = [
    "GenerationResult", "HyperscaleLocalAdapter", "HyperscaleLocalConfig",
    "LocalContribution", "OpenedGeneration", "OwnerUnavailable",
]
