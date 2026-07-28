"""Narrow Python adapter for the production native coordination kernel.

This module deliberately contains no transition policy.  It converts stable
runtime identities to the fixed C ABI, submits one event at a time to the
persistent native service, decodes the authoritative result, and executes the
canonical trace-write effect.  Networking, clocks, storage, and process
supervision remain callers of this adapter rather than hidden kernel inputs.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import threading
from typing import Mapping, Sequence

from ndm.native_dataplane import (
    ABI_V1,
    Client,
    CoordinationDisposition,
    CoordinationEffectKind,
    CoordinationEventKind,
    CoordinationEventV1,
    CoordinationPhase,
    CoordinationResultV1,
    Role,
)


_DISPOSITION_NAMES = {
    value: value.name.lower().replace("_", "-")
    for value in CoordinationDisposition
}
_EFFECT_NAMES = {
    value: value.name.lower().replace("_", "-")
    for value in CoordinationEffectKind
}
_PHASE_NAMES = {
    value: value.name.lower().replace("_", "-")
    for value in CoordinationPhase
}

MEMBER_LIVE = 1 << 0
MEMBER_READY = 1 << 1
MEMBER_RECOVERING = 1 << 2
MEMBER_COHORT = 1 << 3
MEMBER_CONTRIBUTED = 1 << 4
MEMBER_RESULT_RECEIPT = 1 << 5
MEMBER_NODE_APPLIED = 1 << 6

FINITE_CLOSE = 1 << 0
DEADLINE_EXPIRED = 1 << 1


def identity_key(value: bytes | str, *, field: str) -> bytes:
    """Return the service's stable 128-bit opaque identity."""
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{field} must not be empty")
        return hashlib.sha256(value.encode("utf-8")).digest()[:16]
    raw = bytes(value)
    if len(raw) != 16:
        raise ValueError(f"{field} must be exactly 16 bytes")
    return raw


def digest_bytes(value: bytes | str | None, *, field: str,
                 allow_zero: bool = True) -> bytes:
    """Decode one canonical SHA-256 value, using all-zero for absent identity."""
    if value in (None, "", b""):
        if allow_zero:
            return bytes(32)
        raise ValueError(f"{field} must not be empty")
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError(f"{field} must be hexadecimal") from error
    else:
        raw = bytes(value)
    if len(raw) != 32 or (not allow_zero and raw == bytes(32)):
        raise ValueError(f"{field} must be one nonzero SHA-256 digest")
    return raw


class NativeCoordinationAuthority:
    """Serialized owner of one fenced native authority.

    ``Client.coordination_step`` reaches the live service over the same
    metadata-only RPC used by the data plane.  This lock makes the single
    writer explicit at the process boundary; the service independently
    serializes the authority state around the pure C++ ``step`` call.
    """

    def __init__(
        self,
        client: Client,
        *,
        run_id: str,
        fence: int,
        q_min: int,
        t_min: int,
        policy_digest: str,
        committed_generation: int = 0,
        committed_receipt_digest: str = "",
        committed_accepted_tokens: int = 0,
        committed_manifest_digest: str = "",
        committed_result_root: str = "",
        committed_apply_receipts: Sequence[tuple[str, ...]] = (),
        trace_path: str | Path | None = None,
    ):
        if client.role is not Role.CONTROLLER or client.closed:
            raise ValueError("native coordination authority requires a live controller")
        if fence <= 0 or q_min <= 0 or t_min <= 0:
            raise ValueError("native coordination authority bounds are invalid")
        self.client = client
        self.run_id = run_id
        self.run_key = identity_key(run_id, field="run_id")
        self.fence = int(fence)
        self.q_min = int(q_min)
        self.t_min = int(t_min)
        self.policy_digest = digest_bytes(
            policy_digest, field="policy_digest", allow_zero=False)
        self._lock = threading.RLock()
        self._node_names: dict[bytes, str] = {}
        self._incarnation_names: dict[bytes, str] = {}
        self._trace_fd = -1
        if trace_path is not None:
            target = Path(trace_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fd = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
                0o600)

        configured = self.step(
            CoordinationEventKind.RECOVER_AUTHORITY,
            generation=committed_generation,
            exact_tokens=committed_accepted_tokens,
            receipt_digest=committed_receipt_digest,
            previous_receipt_digest=committed_receipt_digest,
            manifest_digest=committed_manifest_digest,
            result_digest=committed_result_root,
            minimum_nodes=q_min,
            minimum_tokens=t_min,
        )
        if configured["disposition"] not in {"accepted", "identical-duplicate"}:
            self.close()
            raise RuntimeError(
                "native authority recovery was rejected: "
                f"{configured['disposition']}")
        for record in committed_apply_receipts:
            if len(record) != 3:
                self.close()
                raise ValueError(
                    "native recovered node apply requires "
                    "node/incarnation/receipt identity")
            node_id, incarnation, receipt = record
            recovered = self.step(
                CoordinationEventKind.RECOVER_NODE_APPLY,
                generation=committed_generation,
                node_id=node_id,
                incarnation=incarnation,
                trainer_count=8,
                receipt_digest=receipt,
            )
            if recovered["disposition"] not in {
                "accepted", "identical-duplicate",
            }:
                self.close()
                raise RuntimeError(
                    "native node-apply recovery was rejected: "
                    f"{node_id}: {recovered['disposition']}")

    def _new_event(
        self, kind: CoordinationEventKind, fields: Mapping[str, object]
    ) -> CoordinationEventV1:
        event = CoordinationEventV1()
        event.struct_size = ctypes.sizeof(CoordinationEventV1)
        event.abi_version = ABI_V1
        event.kind = int(kind)
        event.flags = int(fields.get("flags", 0))
        run_id = fields.get("run_id", self.run_id)
        run_key = identity_key(run_id, field="run_id")  # type: ignore[arg-type]
        event.run_key[:] = run_key
        event.fence_epoch = int(fields.get("fence", self.fence))
        event.generation = int(fields.get("generation", 0))
        event.attempt = int(fields.get("attempt", 0))
        event.trainer_count = int(fields.get("trainer_count", 0))

        node = fields.get("node_id", "")
        if node not in ("", b"", None):
            node_key = identity_key(node, field="node_id")  # type: ignore[arg-type]
            event.node_key[:] = node_key
            if isinstance(node, str):
                self._node_names[node_key] = node
        incarnation = fields.get("incarnation", "")
        if incarnation not in ("", b"", None):
            incarnation_key = identity_key(
                incarnation, field="incarnation")  # type: ignore[arg-type]
            event.incarnation[:] = incarnation_key
            if isinstance(incarnation, str):
                self._incarnation_names[incarnation_key] = incarnation

        event.sequence = int(fields.get("sequence", 0))
        event.exact_tokens = int(fields.get("exact_tokens", 0))
        event.minimum_nodes = int(fields.get("minimum_nodes", self.q_min))
        event.minimum_tokens = int(fields.get("minimum_tokens", self.t_min))
        event.policy_digest[:] = digest_bytes(
            fields.get("policy_digest", self.policy_digest),  # type: ignore[arg-type]
            field="policy_digest", allow_zero=False)
        for name in (
            "payload_digest", "result_digest", "receipt_digest",
            "previous_receipt_digest", "manifest_digest",
        ):
            getattr(event, name)[:] = digest_bytes(
                fields.get(name), field=name)  # type: ignore[arg-type]
        return event

    def _decode(self, result: CoordinationResultV1, trace: str
                ) -> dict[str, object]:
        disposition = CoordinationDisposition(int(result.disposition))
        phase = CoordinationPhase(int(result.phase))
        members = []
        for item in result.members[:result.member_count]:
            node_key = bytes(item.node_key)
            incarnation_key = bytes(item.current_incarnation)
            cohort_key = bytes(item.cohort_incarnation)
            flags = int(item.flags)
            members.append({
                "worker_id": self._node_names.get(node_key, node_key.hex()),
                "node_key": node_key.hex(),
                "incarnation": self._incarnation_names.get(
                    incarnation_key, incarnation_key.hex()),
                "incarnation_key": incarnation_key.hex(),
                "cohort_incarnation": self._incarnation_names.get(
                    cohort_key, cohort_key.hex()) if cohort_key != bytes(16) else "",
                "control_sequence": int(item.control_sequence),
                "exact_tokens": int(item.exact_tokens),
                "live": bool(flags & MEMBER_LIVE),
                "ready": bool(flags & MEMBER_READY),
                "recovering": bool(flags & MEMBER_RECOVERING),
                "cohort": bool(flags & MEMBER_COHORT),
                "contributed": bool(flags & MEMBER_CONTRIBUTED),
                "result_receipt": bool(flags & MEMBER_RESULT_RECEIPT),
                "node_applied": bool(flags & MEMBER_NODE_APPLIED),
                "payload_digest": bytes(item.payload_digest).hex(),
                "result_digest": bytes(item.result_digest).hex(),
                "contribution_receipt":
                    bytes(item.contribution_receipt).hex(),
                "apply_receipt": bytes(item.apply_receipt).hex(),
            })
        effects = []
        for item in result.effects[:result.effect_count]:
            kind = CoordinationEffectKind(int(item.kind))
            node_key = bytes(item.node_key)
            effects.append({
                "kind": _EFFECT_NAMES[kind],
                "generation": int(item.generation),
                "worker_id": self._node_names.get(node_key, node_key.hex()),
                "node_key": node_key.hex(),
                "digest": bytes(item.digest).hex(),
            })
        return {
            "schema": "emender-native-coordination-result-v1",
            "disposition": _DISPOSITION_NAMES[disposition],
            "phase": _PHASE_NAMES[phase],
            "fence": int(result.fence_epoch),
            "committed_generation": int(result.committed_generation),
            "accepted_token_clock": int(result.accepted_token_clock),
            "active_generation": int(result.active_generation),
            "active_attempt": int(result.active_attempt),
            "owner_epoch": int(result.owner_epoch),
            "owner_reassignments": int(result.owner_reassignments),
            "live_count": int(result.live_count),
            "ready_count": int(result.ready_count),
            "cohort_count": int(result.cohort_count),
            "contribution_count": int(result.contribution_count),
            "result_receipt_count": int(result.result_receipt_count),
            "policy_digest": bytes(result.policy_digest).hex(),
            "commit_receipt": bytes(result.commit_receipt).hex(),
            "commit_manifest": bytes(result.commit_manifest).hex(),
            "committed_result": bytes(result.committed_result).hex(),
            "pre_state_digest": bytes(result.pre_state_digest).hex(),
            "post_state_digest": bytes(result.post_state_digest).hex(),
            "members": members,
            "effects": effects,
            "trace": trace,
        }

    def step(self, kind: CoordinationEventKind, **fields: object
             ) -> dict[str, object]:
        """Submit one total event and return its typed disposition/effects."""
        with self._lock:
            event = self._new_event(kind, fields)
            result, trace = self.client.coordination_step(event)
            decoded = self._decode(result, trace)
            if self._trace_fd >= 0:
                encoded = (trace + "\n").encode("utf-8")
                written = 0
                while written != len(encoded):
                    written += os.write(self._trace_fd, encoded[written:])
            return decoded

    def close(self) -> None:
        if self._trace_fd >= 0:
            os.close(self._trace_fd)
            self._trace_fd = -1


__all__ = [
    "DEADLINE_EXPIRED", "FINITE_CLOSE", "MEMBER_COHORT",
    "MEMBER_CONTRIBUTED", "MEMBER_LIVE", "MEMBER_NODE_APPLIED",
    "MEMBER_READY", "MEMBER_RECOVERING", "MEMBER_RESULT_RECEIPT",
    "NativeCoordinationAuthority", "digest_bytes", "identity_key",
]
