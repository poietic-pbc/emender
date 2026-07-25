"""Immutable allocation fencing and checkpoint authority for resilient E97.

Live membership, generation admission, result agreement, and recovery
handshakes belong to :mod:`ndm.resilient_pool_runtime` and the persistent
native services.  This module has one deliberately small durable role:

* publish an immutable scheduler-fence claim before model load;
* publish one content-attested commit receipt after a checkpoint and manifest
  have been written and reload-verified; and
* recover only the receipt chain selected by the newest scheduler fence.

There is no database, lock file, mutable lease row, or heartbeat here.  A
fresh allocation fence records the exact receipt from which it starts.
Subsequent commits must extend that digest, so an older allocation can neither
become an ancestor of the new fence nor regain authority by writing a later
generation.  The compatibility ``handoff/latest.json`` pointer may be emitted
by callers for operators, but no method in this module reads it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence


CLAIM_SCHEMA = "emender-native-peer-allocation-claim-v1"
COMMIT_SCHEMA = "emender-native-peer-commit-receipt-v1"
APPLY_SCHEMA = "emender-native-peer-node-apply-receipt-v1"
MAX_AUTHORITY_BYTES = 1024 * 1024


class FenceRejected(RuntimeError):
    """A stale/conflicting allocation or immutable publication was rejected."""


def _canonical(value: Mapping[str, object]) -> bytes:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    if len(encoded) > MAX_AUTHORITY_BYTES:
        raise ValueError("immutable peer-control record exceeds bounded metadata")
    return encoded


def _digest(domain: str, value: Mapping[str, object]) -> str:
    return hashlib.sha256(domain.encode() + b"\0" + _canonical(value)).hexdigest()


def _bounded_string(value: object, field: str, *, maximum: int = 4096) -> str:
    result = str(value)
    if not result or len(result.encode()) > maximum:
        raise ValueError(f"{field} is missing or exceeds its bound")
    return result


@dataclass(frozen=True)
class AllocationClaim:
    run_id: str
    allocation_id: str
    incarnation: str
    fence: int
    protocol_id: str
    config_id: str
    base_generation: int
    base_commit_digest: str
    previous_claim_digest: str
    created_at_ns: int
    claim_digest: str

    def __post_init__(self) -> None:
        for field in (
            "run_id", "allocation_id", "incarnation", "protocol_id",
            "config_id", "claim_digest",
        ):
            _bounded_string(getattr(self, field), field)
        if self.fence <= 0 or self.base_generation < 0 or self.created_at_ns <= 0:
            raise ValueError("allocation claim has an invalid fence/base/time")
        for field in (
            "base_commit_digest", "previous_claim_digest", "claim_digest",
        ):
            value = getattr(self, field)
            if value and len(value) != 64:
                raise ValueError(f"{field} is not a SHA-256 digest")

    def encode(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def decode(cls, value: str) -> "AllocationClaim":
        return cls(**json.loads(value))


@dataclass(frozen=True)
class CommitReceipt:
    run_id: str
    allocation_fence: int
    allocation_incarnation: str
    allocation_claim_digest: str
    generation: int
    accepted_tokens: int
    outer_step: int
    manifest_path: Path
    manifest_sha256: str
    checkpoint_sha256: str
    result_root: str
    previous_result_root: str
    membership_digest: str
    previous_receipt_digest: str
    published_at_ns: int
    receipt_path: Path
    receipt_digest: str

    def pointer(self) -> dict[str, object]:
        """Return a non-authoritative compatibility/operator pointer."""
        return {
            "schema": "emender-native-peer-latest-compatibility-v1",
            "generation": self.generation,
            "fence": self.allocation_fence,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "accepted_tokens": self.accepted_tokens,
            "commit_receipt": str(self.receipt_path),
            "commit_receipt_sha256": self.receipt_digest,
            "authoritative_source": "immutable_commit_receipt_chain",
        }


@dataclass(frozen=True)
class NodeApplyReceipt:
    run_id: str
    allocation_fence: int
    allocation_claim_digest: str
    generation: int
    commit_receipt_digest: str
    node_id: str
    node_incarnation: str
    result_root: str
    trainer_receipts: tuple[tuple[int, str, str], ...]
    applied_at_ns: int
    receipt_path: Path
    receipt_digest: str


class ManifestPeerAuthority:
    """Content-attested allocation/commit authority rooted in one run tree."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "handoff" / "authority"
        self.allocations = self.root / "allocations"
        self.commits = self.root / "commits"
        self.applies = self.root / "applies"
        for path in (self.allocations, self.commits, self.applies):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _unsigned(value: Mapping[str, object], digest_field: str) -> dict[str, object]:
        return {key: item for key, item in value.items() if key != digest_field}

    def _within(self, path: str | Path, root: Path, field: str) -> Path:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{field} escapes its immutable authority root") from error
        return resolved

    def _relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.run_dir))

    def _resolve_relative(self, value: object, field: str) -> Path:
        relative = Path(_bounded_string(value, field))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{field} must be run-relative")
        return self._within(self.run_dir / relative, self.run_dir, field)

    def _write_immutable(self, path: Path, value: Mapping[str, object]) -> None:
        """Link a fully fsynced temporary record into its final name once."""
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical(value) + b"\n"
        if path.exists():
            if path.read_bytes() != encoded:
                raise FenceRejected(f"immutable authority record conflicts: {path.name}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise FenceRejected(
                        f"immutable authority record conflicts: {path.name}")
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _read_records(self, directory: Path) -> list[tuple[Path, dict[str, object]]]:
        records: list[tuple[Path, dict[str, object]]] = []
        paths = sorted(directory.glob("*.json"))
        if len(paths) > 100_000:
            raise FenceRejected("immutable authority record count exceeds bound")
        for path in paths:
            try:
                raw = path.read_bytes()
                if len(raw) > MAX_AUTHORITY_BYTES:
                    raise ValueError("record exceeds bounded metadata")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("authority record is not an object")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise FenceRejected(
                    f"malformed immutable authority record: {path.name}") from error
            records.append((path.resolve(), value))
        return records

    def _claims(self) -> list[tuple[Path, AllocationClaim]]:
        values: list[tuple[Path, AllocationClaim]] = []
        for path, value in self._read_records(self.allocations):
            if value.get("schema") != CLAIM_SCHEMA:
                raise FenceRejected(f"unknown allocation-claim schema: {path.name}")
            unsigned = self._unsigned(value, "claim_digest")
            actual = _digest("emender-native-peer-allocation-claim-v1", unsigned)
            if value.get("claim_digest") != actual:
                raise FenceRejected(f"allocation-claim digest mismatch: {path.name}")
            claim = AllocationClaim(
                run_id=_bounded_string(value.get("run_id"), "run_id"),
                allocation_id=_bounded_string(
                    value.get("allocation_id"), "allocation_id"),
                incarnation=_bounded_string(value.get("incarnation"), "incarnation"),
                fence=int(value.get("fence", 0)),
                protocol_id=_bounded_string(value.get("protocol_id"), "protocol_id"),
                config_id=_bounded_string(value.get("config_id"), "config_id"),
                base_generation=int(value.get("base_generation", -1)),
                base_commit_digest=str(value.get("base_commit_digest", "")),
                previous_claim_digest=str(value.get("previous_claim_digest", "")),
                created_at_ns=int(value.get("created_at_ns", 0)),
                claim_digest=actual,
            )
            values.append((path, claim))
        return values

    def current_claim(self, run_id: str | None = None) -> AllocationClaim | None:
        claims = self._claims()
        if run_id is not None:
            claims = [item for item in claims if item[1].run_id == run_id]
        if not claims:
            return None
        maximum = max(claim.fence for _, claim in claims)
        newest = [claim for _, claim in claims if claim.fence == maximum]
        identities = {claim.claim_digest for claim in newest}
        if len(identities) != 1:
            raise FenceRejected(
                "conflicting allocation incarnation at the newest scheduler fence")
        return newest[0]

    def claim(self, *, run_id: str, allocation_id: str, incarnation: str,
              fence: int, protocol_id: str, config_id: str,
              now_ns: int | None = None) -> AllocationClaim | None:
        """Claim a strictly newer scheduler fence before any model is loaded."""
        if fence <= 0:
            raise ValueError("scheduler allocation fence must be positive")
        for name, value in {
            "run_id": run_id,
            "allocation_id": allocation_id,
            "incarnation": incarnation,
            "protocol_id": protocol_id,
            "config_id": config_id,
        }.items():
            _bounded_string(value, name)
        current = self.current_claim(run_id)
        if current is not None and current.fence > fence:
            return None
        if current is not None and current.fence == fence:
            if (
                current.allocation_id == allocation_id
                and current.incarnation == incarnation
                and current.protocol_id == protocol_id
                and current.config_id == config_id
            ):
                return current
            raise FenceRejected(
                "conflicting allocation incarnation at the same scheduler fence")

        base = self.current_commit(current) if current is not None else None
        unsigned: dict[str, object] = {
            "schema": CLAIM_SCHEMA,
            "run_id": run_id,
            "allocation_id": allocation_id,
            "incarnation": incarnation,
            "fence": int(fence),
            "protocol_id": protocol_id,
            "config_id": config_id,
            "base_generation": 0 if base is None else base.generation,
            "base_commit_digest": "" if base is None else base.receipt_digest,
            "previous_claim_digest": (
                "" if current is None else current.claim_digest),
            "created_at_ns": int(now_ns if now_ns is not None else time.time_ns()),
        }
        claim_digest = _digest(
            "emender-native-peer-allocation-claim-v1", unsigned)
        value = {**unsigned, "claim_digest": claim_digest}
        path = self.allocations / (
            f"allocation-fence-{fence:020d}-{claim_digest[:20]}.json")
        self._write_immutable(path, value)
        admitted = self.current_claim(run_id)
        if admitted is None or admitted.claim_digest != claim_digest:
            return None
        return AllocationClaim(
            run_id=run_id,
            allocation_id=allocation_id,
            incarnation=incarnation,
            fence=fence,
            protocol_id=protocol_id,
            config_id=config_id,
            base_generation=int(unsigned["base_generation"]),
            base_commit_digest=str(unsigned["base_commit_digest"]),
            previous_claim_digest=str(unsigned["previous_claim_digest"]),
            created_at_ns=int(unsigned["created_at_ns"]),
            claim_digest=claim_digest,
        )

    def assert_current(self, claim: AllocationClaim) -> None:
        current = self.current_claim(claim.run_id)
        if current is None or current.claim_digest != claim.claim_digest:
            raise FenceRejected("operation rejected by newer allocation fence")

    def _commit_records(self) -> dict[str, CommitReceipt]:
        result: dict[str, CommitReceipt] = {}
        for path, value in self._read_records(self.commits):
            if value.get("schema") != COMMIT_SCHEMA:
                raise FenceRejected(f"unknown commit-receipt schema: {path.name}")
            unsigned = self._unsigned(value, "receipt_digest")
            actual = _digest("emender-native-peer-commit-receipt-v1", unsigned)
            if value.get("receipt_digest") != actual:
                raise FenceRejected(f"commit-receipt digest mismatch: {path.name}")
            manifest = self._resolve_relative(value.get("manifest"), "manifest")
            receipt = CommitReceipt(
                run_id=_bounded_string(value.get("run_id"), "run_id"),
                allocation_fence=int(value.get("allocation_fence", 0)),
                allocation_incarnation=_bounded_string(
                    value.get("allocation_incarnation"), "allocation_incarnation"),
                allocation_claim_digest=_bounded_string(
                    value.get("allocation_claim_digest"),
                    "allocation_claim_digest"),
                generation=int(value.get("generation", -1)),
                accepted_tokens=int(value.get("accepted_tokens", -1)),
                outer_step=int(value.get("outer_step", -1)),
                manifest_path=manifest,
                manifest_sha256=_bounded_string(
                    value.get("manifest_sha256"), "manifest_sha256"),
                checkpoint_sha256=_bounded_string(
                    value.get("checkpoint_sha256"), "checkpoint_sha256"),
                result_root=_bounded_string(value.get("result_root"), "result_root"),
                previous_result_root=str(value.get("previous_result_root", "")),
                membership_digest=_bounded_string(
                    value.get("membership_digest"), "membership_digest"),
                previous_receipt_digest=str(
                    value.get("previous_receipt_digest", "")),
                published_at_ns=int(value.get("published_at_ns", 0)),
                receipt_path=path,
                receipt_digest=actual,
            )
            if (
                receipt.allocation_fence <= 0
                or receipt.generation <= 0
                or receipt.accepted_tokens < 0
                or receipt.outer_step < 0
                or receipt.published_at_ns <= 0
                or any(
                    len(item) != 64 for item in (
                        receipt.allocation_claim_digest,
                        receipt.manifest_sha256,
                        receipt.checkpoint_sha256,
                        receipt.result_root,
                        receipt.membership_digest,
                    )
                )
                or (
                    receipt.previous_result_root
                    and len(receipt.previous_result_root) != 64
                )
                or (
                    receipt.previous_receipt_digest
                    and len(receipt.previous_receipt_digest) != 64
                )
            ):
                raise FenceRejected(f"invalid commit-receipt fields: {path.name}")
            prior = result.get(actual)
            if prior is not None and prior != receipt:
                raise FenceRejected("commit receipt digest collision/conflict")
            result[actual] = receipt
        return result

    def _validate_commit(self, receipt: CommitReceipt, *,
                         verify_checkpoint: bool) -> dict[str, object]:
        try:
            encoded = receipt.manifest_path.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != receipt.manifest_sha256:
                raise ValueError("manifest digest mismatch")
            manifest = json.loads(encoded)
            if not isinstance(manifest, dict):
                raise ValueError("manifest is not an object")
            membership = manifest.get("membership")
            if not isinstance(membership, Sequence) or isinstance(
                    membership, (str, bytes)) or not membership:
                raise ValueError("manifest frozen membership is invalid")
            membership_digest = self._membership_digest(membership)
            manifest_result_root = self._manifest_result_root(manifest)
            declared_previous = str(
                dict(manifest.get("digests", {})).get(
                    "previous_result_root", ""))
            checkpoint = self._within(
                manifest.get("checkpoint", ""), self.run_dir,
                "checkpoint")
            if (
                not manifest.get("finalized")
                or manifest.get("run_id") != receipt.run_id
                or int(manifest.get("generation", -1)) != receipt.generation
                or int(dict(manifest.get("fence", {})).get(
                    "coordinator_epoch", -1)) != receipt.allocation_fence
                or manifest.get("checkpoint_sha256")
                != receipt.checkpoint_sha256
                or int(manifest.get("accepted_tokens", -1))
                != receipt.accepted_tokens
                or int(dict(manifest.get("outer_update_state", {})).get(
                    "step", -1)) != receipt.outer_step
                or manifest_result_root != receipt.result_root
                or membership_digest != receipt.membership_digest
                or (
                    declared_previous
                    and declared_previous != (
                        receipt.previous_result_root or "00" * 32)
                )
            ):
                raise ValueError("manifest/receipt identity mismatch")
            if verify_checkpoint:
                if not checkpoint.is_file():
                    raise ValueError("checkpoint is missing")
                if int(manifest.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size:
                    raise ValueError("checkpoint size mismatch")
                digest = hashlib.sha256()
                with checkpoint.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != receipt.checkpoint_sha256:
                    raise ValueError("checkpoint digest mismatch")
            return manifest
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise FenceRejected(
                f"invalid immutable commit authority: {receipt.receipt_path.name}") from error

    def current_commit(self, claim: AllocationClaim | None = None, *,
                       verify_checkpoint: bool = False) -> CommitReceipt | None:
        """Resolve the only commit chain selected by the newest fence claim."""
        current = self.current_claim(None if claim is None else claim.run_id)
        if current is None:
            return None
        if claim is not None and current.claim_digest != claim.claim_digest:
            raise FenceRejected("operation rejected by newer allocation fence")
        claim = current
        records = self._commit_records()
        cursor = (
            None if not claim.base_commit_digest
            else records.get(claim.base_commit_digest)
        )
        if claim.base_commit_digest and cursor is None:
            raise FenceRejected("allocation claim base commit is missing")
        if cursor is not None and cursor.generation != claim.base_generation:
            raise FenceRejected("allocation claim base generation/digest mismatch")
        previous_digest = "" if cursor is None else cursor.receipt_digest
        expected_generation = claim.base_generation + 1
        while True:
            candidates = [
                value for value in records.values()
                if (
                    value.allocation_claim_digest == claim.claim_digest
                    and value.allocation_fence == claim.fence
                    and value.previous_receipt_digest == previous_digest
                    and value.generation == expected_generation
                )
            ]
            unique = {item.receipt_digest: item for item in candidates}
            if len(unique) > 1:
                raise FenceRejected(
                    "conflicting exact-once commits extend one receipt")
            if not unique:
                break
            cursor = next(iter(unique.values()))
            self._validate_commit(cursor, verify_checkpoint=False)
            previous_digest = cursor.receipt_digest
            expected_generation += 1
        # Any current-fence receipt not on the single chain is evidence of a
        # conflicting publication, not an ignorable orphan.
        off_chain = [
            value for value in records.values()
            if value.allocation_claim_digest == claim.claim_digest
            and value.allocation_fence == claim.fence
            and value.generation >= claim.base_generation + 1
            and (
                cursor is None
                or value.generation > cursor.generation
                or (
                    value.generation == cursor.generation
                    and value.receipt_digest != cursor.receipt_digest
                )
            )
        ]
        if off_chain:
            raise FenceRejected("conflicting or discontinuous commit receipt")
        if cursor is not None:
            self._validate_commit(cursor, verify_checkpoint=verify_checkpoint)
        return cursor

    @staticmethod
    def _manifest_result_root(manifest: Mapping[str, object]) -> str:
        digests = dict(manifest.get("digests", {}))
        native = digests.get("native_result")
        result = (
            native.get("result_root")
            if isinstance(native, Mapping)
            else digests.get("result_root")
        )
        # Explicit native result roots are mandatory on the production path.
        # The small Python debug fixture has no native result object; its
        # reload-verified checkpoint digest is the complete-state result root.
        result = str(result or manifest.get("checkpoint_sha256", ""))
        if len(result) != 64 or result == "00" * 32:
            raise ValueError("checkpoint manifest lacks a native result root")
        return result

    @staticmethod
    def _membership_digest(membership: Sequence[object]) -> str:
        return hashlib.sha256(
            b"emender-native-peer-membership-v1\0"
            + json.dumps(
                list(membership), sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def publish_checkpoint(self, claim: AllocationClaim,
                           manifest_path: str | Path,
                           *, now_ns: int | None = None,
                           verified_checkpoint_sha256: str | None = None,
                           ) -> CommitReceipt:
        """Publish one exact-once immutable receipt for a verified checkpoint."""
        self.assert_current(claim)
        manifest_path = self._within(manifest_path, self.run_dir / "handoff",
                                     "manifest")
        encoded = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        manifest = json.loads(encoded)
        if not isinstance(manifest, dict):
            raise ValueError("checkpoint manifest is not an object")
        generation = int(manifest.get("generation", -1))
        if (
            not manifest.get("finalized")
            or manifest.get("run_id") != claim.run_id
            or generation <= 0
            or int(dict(manifest.get("fence", {})).get(
                "coordinator_epoch", -1)) != claim.fence
        ):
            raise ValueError("checkpoint manifest allocation identity is invalid")
        checkpoint = self._within(
            manifest.get("checkpoint", ""), self.run_dir, "checkpoint")
        if (
            not checkpoint.is_file()
            or int(manifest.get("checkpoint_bytes", -1)) != checkpoint.stat().st_size
        ):
            raise ValueError("checkpoint manifest size/path is invalid")
        checkpoint_sha256 = str(
            verified_checkpoint_sha256
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        if len(checkpoint_sha256) != 64:
            raise ValueError("verified checkpoint digest is not SHA-256")
        if checkpoint_sha256 != manifest.get("checkpoint_sha256"):
            raise ValueError("checkpoint manifest digest is invalid")
        accepted_tokens = int(manifest.get("accepted_tokens", -1))
        outer_step = int(dict(
            manifest.get("outer_update_state", {})).get("step", -1))
        if accepted_tokens < 0 or outer_step < 0:
            raise ValueError("checkpoint manifest lacks outer/token recovery state")
        result_root = self._manifest_result_root(manifest)
        membership = manifest.get("membership")
        if not isinstance(membership, Sequence) or isinstance(
                membership, (str, bytes)) or not membership:
            raise ValueError("checkpoint manifest lacks frozen membership")
        membership_digest = self._membership_digest(membership)

        current = self.current_commit(claim)
        if current is not None and generation == current.generation:
            if (
                current.manifest_sha256 == manifest_sha256
                and current.result_root == result_root
            ):
                return current
            raise FenceRejected("generation is already committed with different bytes")
        expected = claim.base_generation + 1 if current is None else current.generation + 1
        if generation != expected:
            raise FenceRejected(
                f"commit generation must extend authority exactly: expected {expected}")
        previous_receipt = "" if current is None else current.receipt_digest
        previous_result = "" if current is None else current.result_root
        declared_previous = str(
            dict(manifest.get("digests", {})).get("previous_result_root", ""))
        if declared_previous and declared_previous not in (
                previous_result, "00" * 32 if current is None else ""):
            raise FenceRejected("checkpoint result lineage does not extend prior root")
        unsigned: dict[str, object] = {
            "schema": COMMIT_SCHEMA,
            "run_id": claim.run_id,
            "allocation_fence": claim.fence,
            "allocation_incarnation": claim.incarnation,
            "allocation_claim_digest": claim.claim_digest,
            "generation": generation,
            "accepted_tokens": accepted_tokens,
            "outer_step": outer_step,
            "manifest": self._relative(manifest_path),
            "manifest_sha256": manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "result_root": result_root,
            "previous_result_root": previous_result,
            "membership_digest": membership_digest,
            "previous_receipt_digest": previous_receipt,
            "published_at_ns": int(now_ns if now_ns is not None else time.time_ns()),
        }
        receipt_digest = _digest(
            "emender-native-peer-commit-receipt-v1", unsigned)
        value = {**unsigned, "receipt_digest": receipt_digest}
        path = self.commits / (
            f"commit-generation-{generation:08d}-"
            f"fence-{claim.fence:020d}-{receipt_digest[:20]}.json")
        self._write_immutable(path, value)
        self.assert_current(claim)
        committed = self.current_commit(claim)
        if committed is None or committed.receipt_digest != receipt_digest:
            raise FenceRejected("published receipt did not become peer authority")
        return committed

    def wait_for_generation(self, claim: AllocationClaim, generation: int, *,
                            deadline: float,
                            verify_checkpoint: bool = False) -> CommitReceipt:
        if generation <= 0:
            raise ValueError("wait generation must be positive")
        last: CommitReceipt | None = None
        while time.monotonic() < deadline:
            last = self.current_commit(
                claim, verify_checkpoint=verify_checkpoint)
            if last is not None and last.generation == generation:
                return last
            if last is not None and last.generation > generation:
                raise FenceRejected("peer authority advanced past requested generation")
            time.sleep(min(.02, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            "immutable peer commit deadline expired"
            + ("" if last is None else f" at generation {last.generation}"))

    def record_node_apply(
            self, claim: AllocationClaim, commit: CommitReceipt, *,
            node_id: str, node_incarnation: str,
            trainer_receipts: Sequence[tuple[int, str, str]],
            now_ns: int | None = None) -> NodeApplyReceipt:
        """Retain one exact all-eight-trainer recovery transaction receipt."""
        self.assert_current(claim)
        current = self.current_commit(claim)
        if current is None or current.receipt_digest != commit.receipt_digest:
            raise FenceRejected("node apply does not target authoritative commit")
        ordered = tuple(sorted(
            (int(rank), _bounded_string(incarnation, "trainer_incarnation"),
             _bounded_string(digest, "recovery_digest"))
            for rank, incarnation, digest in trainer_receipts))
        if (
            tuple(rank for rank, _, _ in ordered) != tuple(range(8))
            or len({incarnation for _, incarnation, _ in ordered}) != 8
            or any(len(digest) != 64 for _, _, digest in ordered)
        ):
            raise ValueError("node apply requires eight unique fenced trainer receipts")
        unsigned: dict[str, object] = {
            "schema": APPLY_SCHEMA,
            "run_id": claim.run_id,
            "allocation_fence": claim.fence,
            "allocation_claim_digest": claim.claim_digest,
            "generation": commit.generation,
            "commit_receipt_digest": commit.receipt_digest,
            "node_id": _bounded_string(node_id, "node_id"),
            "node_incarnation": _bounded_string(
                node_incarnation, "node_incarnation"),
            "result_root": commit.result_root,
            "trainer_receipts": [
                {"rank": rank, "trainer_incarnation": incarnation,
                 "recovery_digest": digest}
                for rank, incarnation, digest in ordered
            ],
            "applied_at_ns": int(now_ns if now_ns is not None else time.time_ns()),
        }
        receipt_digest = _digest(
            "emender-native-peer-node-apply-receipt-v1", unsigned)
        value = {**unsigned, "receipt_digest": receipt_digest}
        node_key = hashlib.sha256(str(node_id).encode()).hexdigest()[:16]
        path = self.applies / (
            f"apply-generation-{commit.generation:08d}-"
            f"fence-{claim.fence:020d}-node-{node_key}.json")
        self._write_immutable(path, value)
        # A newer allocation may have claimed the run while this immutable
        # write was already open.  The orphan remains harmless evidence, but
        # the superseded caller must never receive an acknowledgement that
        # could advance its live peer state.
        self.assert_current(claim)
        return NodeApplyReceipt(
            run_id=claim.run_id,
            allocation_fence=claim.fence,
            allocation_claim_digest=claim.claim_digest,
            generation=commit.generation,
            commit_receipt_digest=commit.receipt_digest,
            node_id=str(node_id),
            node_incarnation=str(node_incarnation),
            result_root=commit.result_root,
            trainer_receipts=ordered,
            applied_at_ns=int(unsigned["applied_at_ns"]),
            receipt_path=path,
            receipt_digest=receipt_digest,
        )

    def node_apply_receipts(self, commit: CommitReceipt) -> tuple[NodeApplyReceipt, ...]:
        values: list[NodeApplyReceipt] = []
        seen_nodes: dict[str, str] = {}
        for path, value in self._read_records(self.applies):
            if value.get("schema") != APPLY_SCHEMA:
                raise FenceRejected(f"unknown node-apply schema: {path.name}")
            unsigned = self._unsigned(value, "receipt_digest")
            actual = _digest(
                "emender-native-peer-node-apply-receipt-v1", unsigned)
            if value.get("receipt_digest") != actual:
                raise FenceRejected(f"node-apply digest mismatch: {path.name}")
            if value.get("commit_receipt_digest") != commit.receipt_digest:
                continue
            trainers = tuple(
                (int(item["rank"]), str(item["trainer_incarnation"]),
                 str(item["recovery_digest"]))
                for item in value.get("trainer_receipts", []))
            receipt = NodeApplyReceipt(
                run_id=str(value.get("run_id")),
                allocation_fence=int(value.get("allocation_fence", 0)),
                allocation_claim_digest=str(
                    value.get("allocation_claim_digest")),
                generation=int(value.get("generation", -1)),
                commit_receipt_digest=str(
                    value.get("commit_receipt_digest")),
                node_id=str(value.get("node_id")),
                node_incarnation=str(value.get("node_incarnation")),
                result_root=str(value.get("result_root")),
                trainer_receipts=trainers,
                applied_at_ns=int(value.get("applied_at_ns", 0)),
                receipt_path=path,
                receipt_digest=actual,
            )
            if (
                receipt.run_id != commit.run_id
                or receipt.allocation_fence != commit.allocation_fence
                or receipt.allocation_claim_digest
                != commit.allocation_claim_digest
                or receipt.generation != commit.generation
                or receipt.result_root != commit.result_root
                or tuple(rank for rank, _, _ in trainers) != tuple(range(8))
                or len({item for _, item, _ in trainers}) != 8
                or any(len(digest) != 64 for _, _, digest in trainers)
                or len(receipt.node_incarnation) == 0
                or receipt.applied_at_ns <= 0
            ):
                raise FenceRejected("node-apply receipt identity mismatch")
            prior = seen_nodes.get(receipt.node_id)
            if prior is not None and prior != receipt.receipt_digest:
                raise FenceRejected("conflicting node-apply receipt")
            seen_nodes[receipt.node_id] = receipt.receipt_digest
            values.append(receipt)
        return tuple(sorted(values, key=lambda item: item.node_id))
