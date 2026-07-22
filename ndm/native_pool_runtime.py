"""Component native-service lifecycle for control-plane integration tests.

The allocation holder continues to own leases, READY membership, generation
freeze, checkpoint policy, and atomic publication.  This module owns the two
compiled ABI handles as one manager-scoped resource: the local exact reducer
and the libfabric endpoint start together, native routes are installed from
leased endpoint records, and cleanup drains both without a peer rendezvous.
Dense owner frames are transferred and replayed through the same session, so
callers cannot accidentally fall back to a Python dense socket after freezing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import time
from typing import Iterator, Mapping, Sequence

from ndm.native_artifacts import (
    NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG, attest_launch,
)
from ndm.native_dataplane import (
    Buffer, Client, Command, DType, NativeLibrary, Operation, ResultView, Role,
)
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


@dataclass(frozen=True)
class NativeTrainerContribution:
    """Producer-owned dense memfd received without copying its payload."""
    trainer_id: str
    incarnation: str
    submission_seq: int
    generation: int
    attempt: int
    weight: int
    length: int
    layout_digest: str
    sha256: str
    fd: int

    def close(self) -> None:
        os.close(self.fd)


class NativeTrainerHandoff:
    """Metadata-only seqpacket boundary for producer-direct sealed memfds."""
    MAX_METADATA = 4096

    def __init__(self, sock: socket.socket, *, run_id: str, fence_epoch: int,
                 generation: int, attempt: int, expected_bytes: int,
                 layout_digest: str, accepted_incarnations: Mapping[str, str],
                 replay_ledger: str | Path | None = None):
        if expected_bytes <= 0 or expected_bytes % 4:
            raise ValueError("expected_bytes must be positive f32 extent")
        if attempt <= 0 or len(layout_digest) != 64:
            raise ValueError("attempt/layout identity is invalid")
        self.sock, self.run_id = sock, run_id
        self.fence_epoch, self.generation, self.attempt = fence_epoch, generation, attempt
        self.expected_bytes = expected_bytes
        self.layout_digest = layout_digest.lower()
        self.accepted_incarnations = dict(accepted_incarnations)
        self.replay_ledger = Path(replay_ledger) if replay_ledger else None
        self._seen: dict[str, str] = {}
        if self.replay_ledger is not None and self.replay_ledger.exists():
            value = json.loads(self.replay_ledger.read_text(encoding="utf-8"))
            if value.get("identity") != self._ledger_identity():
                raise ValueError("trainer replay ledger identity mismatch")
            self._seen = dict(value.get("seen", {}))

    def _ledger_identity(self) -> dict[str, object]:
        return {"run_id": self.run_id, "fence_epoch": self.fence_epoch,
                "generation": self.generation, "attempt": self.attempt,
                "layout_digest": self.layout_digest,
                "expected_bytes": self.expected_bytes}

    def _record(self, key: str, digest: str) -> None:
        self._seen[key] = digest
        if self.replay_ledger is None:
            return
        self.replay_ledger.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.replay_ledger.with_name(
            f".{self.replay_ledger.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({"identity": self._ledger_identity(),
                                         "seen": self._seen}, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.replay_ledger)

    @classmethod
    def listen(cls, path: str | Path, **identity) -> "NativeTrainerHandoff":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.bind(str(target)); os.chmod(target, 0o600); sock.listen(16)
        return cls(sock, **identity)

    @classmethod
    def connect(cls, path: str | Path, **identity) -> "NativeTrainerHandoff":
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        sock.connect(str(path))
        return cls(sock, **identity)

    def send_memfd(self, fd: int, *, trainer_id: str, incarnation: str,
                   submission_seq: int, weight: int, sha256: str) -> None:
        import fcntl
        required = 0x0004 | 0x0002 | 0x0008  # F_SEAL_GROW|SHRINK|WRITE
        if (os.fstat(fd).st_size != self.expected_bytes
                or fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034))
                & required != required):
            raise ValueError("trainer memfd must have exact extent and immutable seals")
        value = {"run_id": self.run_id, "fence_epoch": self.fence_epoch,
                 "generation": self.generation, "attempt": self.attempt,
                 "layout_digest": self.layout_digest, "trainer_id": trainer_id,
                 "incarnation": incarnation, "submission_seq": submission_seq,
                 "weight": weight, "length": self.expected_bytes, "sha256": sha256}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        if (not trainer_id or not incarnation or submission_seq < 0 or weight <= 0
                or len(sha256) != 64 or len(encoded) > self.MAX_METADATA):
            raise ValueError("invalid trainer handoff identity")
        self.sock.sendmsg([encoded], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                      struct.pack("i", fd))])

    def receive_memfd(self) -> NativeTrainerContribution:
        conn, _ = self.sock.accept()
        try:
            encoded, ancillary, flags, _ = conn.recvmsg(
                self.MAX_METADATA, socket.CMSG_SPACE(2 * struct.calcsize("i")),
                getattr(socket, "MSG_CMSG_CLOEXEC", 0))
            if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                raise ValueError("truncated trainer handoff")
            fds = []
            for level, kind, data in ancillary:
                if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                    continue
                if len(data) % struct.calcsize("i"):
                    raise ValueError("malformed trainer descriptor control message")
                fds.extend(item[0] for item in struct.iter_unpack("i", data))
            if len(fds) != 1:
                for item in fds: os.close(item)
                raise ValueError("trainer handoff requires exactly one memfd")
            fd = fds[0]
            try:
                value = json.loads(encoded)
                if not isinstance(value, dict):
                    raise ValueError("trainer handoff metadata must be an object")
                integer_fields = ("fence_epoch", "generation", "attempt", "length",
                                  "weight", "submission_seq")
                if any(type(value.get(field)) is not int for field in integer_fields):
                    raise ValueError("trainer handoff integer metadata is malformed")
                if (value.get("run_id") != self.run_id
                        or value["fence_epoch"] != self.fence_epoch
                        or value["generation"] != self.generation
                        or value["attempt"] != self.attempt
                        or value.get("layout_digest") != self.layout_digest
                        or value["length"] != self.expected_bytes
                        or os.fstat(fd).st_size != self.expected_bytes
                        or value["weight"] <= 0
                        or value["submission_seq"] < 0):
                    raise ValueError("trainer handoff identity/extent mismatch")
                trainer_id = str(value.get("trainer_id", ""))
                incarnation = str(value.get("incarnation", ""))
                sequence = value["submission_seq"]
                if self.accepted_incarnations.get(trainer_id) != incarnation:
                    raise ValueError("trainer handoff incarnation mismatch")
                import fcntl
                required = 0x0004 | 0x0002 | 0x0008
                if (fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034))
                        & required != required):
                    raise ValueError("received trainer memfd is not immutable")
                claimed = str(value.get("sha256", "")).lower()
                actual_hash = hashlib.sha256()
                finite = True
                offset = 0
                while offset < self.expected_bytes:
                    chunk = os.pread(fd, min(1 << 20, self.expected_bytes - offset), offset)
                    if not chunk:
                        raise ValueError("trainer memfd ended before its admitted extent")
                    actual_hash.update(chunk)
                    if len(chunk) % 4:
                        raise ValueError("trainer f32 extent is misaligned")
                    finite = finite and all(math.isfinite(item[0]) for item in
                        struct.iter_unpack("<f", chunk))
                    offset += len(chunk)
                actual = actual_hash.hexdigest()
                if claimed != actual:
                    raise ValueError("trainer handoff digest mismatch")
                if not finite:
                    raise ValueError("trainer handoff contains nonfinite input")
                key = f"{trainer_id}\0{incarnation}\0{sequence}"
                prior = self._seen.get(key)
                if prior is not None:
                    kind = "duplicate" if prior == actual else "conflicting"
                    raise ValueError(f"{kind} trainer submission sequence")
                self._record(key, actual)
                return NativeTrainerContribution(
                    trainer_id, incarnation, sequence, self.generation, self.attempt,
                    int(value["weight"]), self.expected_bytes, self.layout_digest,
                    actual, fd)
            except BaseException:
                os.close(fd); raise
        finally:
            conn.close()

    def close(self) -> None:
        self.sock.close()


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
                 telemetry_fd: int, run_id: str, fence_epoch: int):
        self.backend, self.attestation = backend, dict(attestation)
        self.run_id, self.fence_epoch = run_id, fence_epoch
        self.local, self.transport, self.owner_endpoint = local, transport, endpoint
        self.telemetry_path, self.telemetry_fd = telemetry_path, telemetry_fd
        self.routes: dict[str, int] = {}
        self._generation_installed = False
        self._frozen = False
        self._checkpoint_proposed = False
        self._checkpoint_identity: dict[str, object] | None = None
        self._owner_replays: dict[tuple[str, bytes], int] = {}
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
                run_id=run_id, fence_epoch=fence_epoch,
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

    def install_generation(self, *, total_elements: int, generation: int,
                           attempt: int = 1, owner_epoch: int = 1,
                           source_dtype: DType = DType.F32,
                           payload_max: int = 64 << 20,
                           base_digest: bytes | None = None,
                           plan_digest: bytes | None = None,
                           deadline_s: float = 30.0,
                           generation_deadline_s: float | None = None
                           ) -> bytes:
        """Install a bounded native local generation after Python opens it."""
        if self._generation_installed:
            raise RuntimeError("native generation is already installed")
        digest = self.local.install_flat_layout(
            total_elements, source_dtype=source_dtype, payload_max=payload_max)
        self.local.install_generation(
            generation, attempt=attempt, owner_epoch=owner_epoch,
            base_digest=base_digest, plan_digest=plan_digest,
            deadline_s=deadline_s,
            generation_deadline_s=generation_deadline_s).close()
        self._generation_installed = True
        self._frozen = self._checkpoint_proposed = False
        return digest

    def install_reduction_attempt(self, *, generation: int, attempt: int,
                                  owner_epoch: int, source_dtype: DType,
                                  base_digest: bytes, plan_digest: bytes,
                                  deadline_s: float,
                                  generation_deadline_s: float | None = None
                                  ) -> None:
        """Reuse the installed flat layout for the post-transfer f64 attempt."""
        if self._generation_installed:
            raise RuntimeError("native generation is already installed")
        self.local.source_dtype = source_dtype
        self.local.install_generation(
            generation, attempt=attempt, owner_epoch=owner_epoch,
            base_digest=base_digest, plan_digest=plan_digest,
            deadline_s=deadline_s,
            generation_deadline_s=generation_deadline_s).close()
        self._generation_installed = True
        self._frozen = self._checkpoint_proposed = False

    @contextmanager
    def import_reduction_sources(
            self, sources: Sequence[tuple[int, str, str, int, int, bytes]], *,
            source_dtype: DType, deadline_s: float
            ) -> Iterator[tuple[Operation, ...]]:
        """Admit independent sealed owner results through parallel RPC clients.

        Each source tuple is ``(fd, worker_id, incarnation, sequence, weight,
        sha256)``. The raw digest is accumulated while the authenticated owner
        transfer is in progress, so admission does not add a Python full-buffer
        pass before the authoritative native checksum and finite scan.
        A real E97 binary64 node numerator is several GiB.  Importing two of
        them through the controller's one RPC session serializes the caller
        checksum and service checksum/finite scans even though the service is
        explicitly able to validate independent client sessions concurrently.
        Give every immutable source its own short-lived native client so those
        scans overlap without weakening either validation boundary.

        The controller still owns the installed attempt, freeze, exact
        reduction, and result.  Source clients remain live through the yielded
        freeze/finalize window and are closed on every exit path.
        """
        if (not self._generation_installed or self._frozen
                or not sources or len(sources) > 16 or deadline_s <= 0):
            raise RuntimeError("imported reduction sources are outside LOCAL_COLLECT")
        if len({(worker, incarnation, sequence)
                for _fd, worker, incarnation, sequence, _weight, _digest in sources}) \
                != len(sources):
            raise ValueError("imported reduction source identities must be unique")
        if any(fd < 0 or not worker or not incarnation or sequence < 0 or weight <= 0
                   or len(bytes(digest)) != 32
               for fd, worker, incarnation, sequence, weight, digest in sources):
            raise ValueError("imported reduction source metadata is invalid")

        def admit(source: tuple[int, str, str, int, int, bytes]
                  ) -> tuple[Client, Operation]:
            fd, worker, incarnation, sequence, weight, source_sha256 = source
            client = Client.open(
                library=self.local.native, role=Role.TRAINER,
                run_key=self.run_id, fence_epoch=self.fence_epoch,
                worker_key=f"owner-import:{worker}:{sequence}",
                incarnation=incarnation,
                deadline_s=min(10.0, deadline_s))
            try:
                client.attach_generation(
                    total_elements=self.local.total_elements,
                    layout_digest=self.local.layout_digest,
                    generation=self.local.generation,
                    attempt=self.local.attempt,
                    owner_epoch=self.local.owner_epoch,
                    source_dtype=source_dtype,
                    deadline_s=deadline_s,
                    deadline_unix_ns=self.local.generation_deadline_ns,
                    base_digest=self.local.base_digest,
                    plan_digest=self.local.plan_digest)
                with client.register_memfd(
                        fd, length=self.local.total_elements
                        * source_dtype.numpy_dtype.itemsize,
                        handle_generation=self.local.generation) as buffer:
                    operation = client.submit(
                        buffer, trainer_key=worker,
                        trainer_incarnation=incarnation,
                        submission_seq=sequence, weight=weight,
                        source_dtype=source_dtype,
                        source_sha256=source_sha256, deadline_s=deadline_s)
                return client, operation
            except BaseException:
                client.close()
                raise

        admitted: list[tuple[Client, Operation]] = []
        futures = []
        try:
            with ThreadPoolExecutor(
                    max_workers=len(sources),
                    thread_name_prefix="native-owner-import") as executor:
                futures = [executor.submit(admit, source) for source in sources]
                admitted = [future.result() for future in futures]
            yield tuple(operation for _client, operation in admitted)
        finally:
            # A future after the first failed one may still have admitted a
            # source.  Collect every successful result before releasing so an
            # exception cannot strand a native client or descriptor.
            known_clients = {id(client) for client, _operation in admitted}
            for future in futures:
                if not future.done() or future.cancelled() or future.exception() is not None:
                    continue
                pair = future.result()
                if id(pair[0]) not in known_clients:
                    admitted.append(pair); known_clients.add(id(pair[0]))
            for client, operation in admitted:
                operation.close()
                client.close()

    def allocate_trainer_buffer(self, *, deadline_s: float = 30.0) -> Buffer:
        if not self._generation_installed or self._frozen:
            raise RuntimeError("trainer buffer admission is outside LOCAL_COLLECT")
        return self.local.allocate(deadline_s=deadline_s)

    def submit_local(self, buffer: Buffer, *, trainer_key: bytes | str,
                     trainer_incarnation: bytes | str, submission_seq: int,
                     weight: int, deadline_s: float = 30.0) -> Operation:
        if not self._generation_installed or self._frozen:
            raise RuntimeError("local contribution is outside LOCAL_COLLECT")
        return self.local.submit(
            buffer, trainer_key=trainer_key,
            trainer_incarnation=trainer_incarnation,
            submission_seq=submission_seq, weight=weight,
            deadline_s=deadline_s)

    def freeze(self, *, deadline_s: float = 30.0) -> Operation:
        """Execute Python's immutable accepted-set decision in native memory."""
        if not self._generation_installed:
            raise RuntimeError("native contribution set cannot be frozen")
        # Native FREEZE is identity-fenced and idempotent, including after the
        # lifecycle has advanced to RESULT_READY/COMMITTED.  Always let the
        # service return its established operation for a retry; a Python-only
        # `_frozen` rejection previously turned harmless reordered control
        # traffic into asymmetric manager failure.
        operation = self.local.control(Command.FREEZE, deadline_s=deadline_s)
        self._frozen = True
        return operation

    def transfer_frozen_frame(self, worker_id: str, frame: bytes, *,
                              result_root: bytes,
                              deadline_unix_ns: int | None = None) -> None:
        """Transfer a frozen owner frame with bounded, identity-stable replay.

        Python supplies only an already encoded native frame and its expected
        result root.  The bytes themselves cross the compiled fabric ABI.  A
        given ``(worker, result_root)`` may be sent initially and replayed at
        most twice after owner loss (NDP11); conflicting or unfrozen traffic is
        rejected before provider submission.
        """
        if not self._frozen:
            raise RuntimeError("owner transfer requires a frozen accepted set")
        if worker_id not in self.routes:
            raise KeyError(f"no current-fence native route for {worker_id}")
        root = bytes(result_root)
        if len(root) != 32 or root == bytes(32):
            raise ValueError("owner transfer requires a full nonzero result root")
        key = (worker_id, root)
        sends = self._owner_replays.get(key, 0)
        if sends >= 3:
            raise RuntimeError("native owner replay limit exceeded")
        deadline = deadline_unix_ns or self.transport.deadline_unix_ns
        self.transport.send(self.routes[worker_id], frame,
                            deadline_unix_ns=deadline)
        self._owner_replays[key] = sends + 1

    def transfer_frozen_fd(self, worker_id: str, fd: int, *, frame_bytes: int,
                           result_root: bytes, replay_identity: bytes,
                           deadline_unix_ns: int | None = None) -> None:
        """Transfer a sealed owner frame without materializing it in Python bytes."""
        if not self._frozen:
            raise RuntimeError("owner transfer requires a frozen accepted set")
        if worker_id not in self.routes:
            raise KeyError(f"no current-fence native route for {worker_id}")
        root = bytes(result_root)
        if len(root) != 32 or root == bytes(32):
            raise ValueError("owner transfer requires a full nonzero result root")
        identity = bytes(replay_identity)
        if not identity or len(identity) > 64:
            raise ValueError("native owner frame replay identity is invalid")
        key = (worker_id, root + identity)
        sends = self._owner_replays.get(key, 0)
        if sends >= 3:
            raise RuntimeError("native owner replay limit exceeded")
        self.transport.send_fd(
            self.routes[worker_id], fd, frame_bytes=frame_bytes,
            deadline_unix_ns=deadline_unix_ns or self.transport.deadline_unix_ns)
        self._owner_replays[key] = sends + 1

    def receive_owner_frame(self, *, capacity: int | None = None
                            ) -> tuple[str, bytes] | None:
        """Receive one compiled-ABI frame and authenticate its installed route."""
        received = self.transport.receive(capacity=capacity)
        if received is None:
            return None
        peer_id, frame = received
        worker = next((name for name, value in self.routes.items()
                       if value == peer_id), None)
        if worker is None:
            raise RuntimeError("native frame arrived from an unfenced route")
        return worker, frame

    def receive_owner_fd(self, fd: int, *, capacity: int
                         ) -> tuple[str, int] | None:
        received = self.transport.receive_into_fd(fd, capacity=capacity)
        if received is None:
            return None
        peer_id, frame_bytes = received
        worker = next((name for name, value in self.routes.items()
                       if value == peer_id), None)
        if worker is None:
            raise RuntimeError("native frame arrived from an unfenced route")
        return worker, frame_bytes

    def finalize_redistribution(self, *, deadline_s: float = 30.0
                                ) -> tuple[Operation, ResultView]:
        """Expose the one shared read-only trainer-apply view after owner readiness."""
        if not self._frozen:
            raise RuntimeError("native owners cannot finalize before freeze")
        operation = self.local.control(Command.FINALIZE_OWNERS, deadline_s=deadline_s)
        return operation, self.local.result_view(operation)

    def checkpoint_proposal(self, path: str | Path, result: ResultView,
                            *, publisher: str,
                            metadata: Mapping[str, object] | None = None) -> Path:
        """Emit metadata only; Python remains the checkpoint/publication owner."""
        if not self._frozen or not publisher:
            raise RuntimeError("checkpoint proposal requires a frozen native result")
        expected_run = self.local.run_key
        if (result.client is not self.local or result.closed
                or result.run_key != expected_run
                or result.fence_epoch != self.fence_epoch
                or result.generation != self.local.generation
                or result.attempt != self.local.attempt
                or result.layout_digest != self.local.layout_digest
                or result.base_digest != self.local.base_digest
                or result.global_weight <= 0
                or result.result_root == bytes(32)):
            raise RuntimeError("native checkpoint result identity/fence mismatch")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema": "emender-native-checkpoint-proposal-v1",
            "publisher": publisher, "run_key": result.run_key.hex(),
            "fence_epoch": result.fence_epoch, "generation": result.generation,
            "attempt": result.attempt, "layout_digest": result.layout_digest.hex(),
            "base_digest": result.base_digest.hex(),
            "result_root": result.result_root.hex(),
            "global_weight": result.global_weight,
            "result_bytes": result.length,
            "metadata": dict(metadata or {}),
        }
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self._checkpoint_proposed = True
        self._proposal_generation = int(
            (metadata or {}).get("publication_generation", result.generation))
        self._result_generation = result.generation
        self._checkpoint_identity = {
            "attempt": result.attempt,
            "layout_digest": result.layout_digest.hex(),
            "base_digest": result.base_digest.hex(),
            "result_root": result.result_root.hex(),
            "global_weight": result.global_weight,
            "result_bytes": result.length,
        }
        return target

    def commit(self, *, publication_manifest: str | Path,
               authoritative_latest: Mapping[str, object],
               deadline_s: float = 30.0) -> None:
        """Release native state only after Python supplies fenced CAS evidence."""
        if not self._checkpoint_proposed or self._checkpoint_identity is None:
            raise RuntimeError("native commit requires durable Python publication approval")
        publication_path = Path(publication_manifest).resolve()
        publication_bytes = publication_path.read_bytes()
        publication = json.loads(publication_bytes)
        digest = hashlib.sha256(publication_bytes).hexdigest()
        expected = {
            "generation": self._proposal_generation,
            "fence": self.fence_epoch,
            "manifest": str(publication_path),
            "manifest_sha256": digest,
        }
        if any(authoritative_latest.get(key) != value
               for key, value in expected.items()):
            raise RuntimeError("native commit lacks matching authoritative latest CAS")
        fence = publication.get("fence", {})
        if (publication.get("finalized") is not True
                or publication.get("run_id") != self.run_id
                or int(publication.get("generation", -1)) != self._proposal_generation
                or int(fence.get("coordinator_epoch", -1)) != self.fence_epoch):
            raise RuntimeError("native commit publication identity/fence mismatch")
        publication_identity = publication.get("digests", {}).get(
            "native_result", publication)
        if any(publication_identity.get(key) != value
               for key, value in self._checkpoint_identity.items()):
            raise RuntimeError("native commit publication result identity mismatch")
        # This is a state transition, not an idempotent transport receipt.
        # Issue it exactly once after the durable CAS evidence has passed every
        # run/fence/generation/attempt/layout/base/result/weight check above.
        self.local.control(Command.COMMIT, deadline_s=deadline_s).close()
        self._generation_installed = self._frozen = self._checkpoint_proposed = False
        self._checkpoint_identity = None
        self._owner_replays.clear()

    def abort(self, *, deadline_s: float = 1.0) -> None:
        if self._generation_installed:
            self.local.control(Command.ABORT, deadline_s=deadline_s).close()
        self._generation_installed = self._frozen = self._checkpoint_proposed = False
        self._checkpoint_identity = None
        self._owner_replays.clear()

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
        # A process-local allocation handoff detaches this disposable manager
        # incarnation without changing the persistent service's global state.
        # The node supervisor owns the eventual service-wide TERM/drain.  A
        # normal terminal manager close still drains explicitly after the last
        # generation.  Transport route removal/cancel is independent for each
        # current endpoint in either case.
        try:
            if terminal_reason != "allocation_term_handoff":
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


__all__ = ["NativeManagerSession", "NativeServiceTelemetry",
           "NativeTrainerContribution", "NativeTrainerHandoff", "PYTHON_TCP_DEBUG"]
