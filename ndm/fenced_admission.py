"""Durable, fenced allocation admission and small publication metadata.

This is the R01/R07 control-plane adapter from version 1 of
``docs/RESILIENT_DILOCO_COMPUTE_POOL.md``.  SQLite supplies a transactional
compare-and-swap implementation; deployments must place the database on an
approved control store whose locking semantics have been validated.  No model,
tensor, membership, or heartbeat payload belongs in this database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Callable, Mapping, TypeVar


MAX_CONTROL_BYTES = 16 * 1024
PUBLICATION_KINDS = frozenset({"commit", "checkpoint", "latest"})


class FenceRejected(RuntimeError):
    """The caller no longer owns the current, unexpired allocation fence."""


@dataclass(frozen=True)
class AllocationLease:
    run_id: str
    allocation_id: str
    incarnation: str
    fence: int
    acquired_at: float
    renewed_at: float
    expires_at: float
    protocol_id: str
    config_id: str


T = TypeVar("T")


class SQLiteFencedControlStore:
    """Linearizable lease/publication CAS using bounded SQLite transactions.

    ``timeout_s`` bounds lock acquisition.  Lease time is supplied by callers
    in tests and defaults to wall-clock time in production.  Fence counters are
    never reused, including after an explicit release.
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 5.0,
                 clock: Callable[[], float] = time.time):
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.path, self.timeout_s, self.clock = Path(path), float(timeout_s), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS lease_epochs (
                    run_id TEXT PRIMARY KEY, last_fence INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS leases (
                    run_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS publications (
                    run_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                    fence INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, kind, name));
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=self.timeout_s,
                             isolation_level=None)
        db.execute(f"PRAGMA busy_timeout={max(1, int(self.timeout_s * 1000))}")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @staticmethod
    def _lease(payload: str) -> AllocationLease:
        return AllocationLease(**json.loads(payload))

    @staticmethod
    def _encoded(value: Mapping[str, object]) -> str:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        if len(encoded.encode()) > MAX_CONTROL_BYTES:
            raise ValueError("control payload exceeds small-metadata limit")
        return encoded

    def _transaction(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                result = operation(db)
                db.execute("COMMIT")
                return result
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def acquire(self, *, run_id: str, allocation_id: str, incarnation: str,
                protocol_id: str, config_id: str, ttl_s: float) -> AllocationLease | None:
        """Acquire an absent/expired lease, or return ``None`` without mutation."""
        if ttl_s <= 0 or not all((run_id, allocation_id, incarnation,
                                  protocol_id, config_id)):
            raise ValueError("lease identities and positive ttl_s are required")
        now = self.clock()

        def cas(db: sqlite3.Connection) -> AllocationLease | None:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?", (run_id,)).fetchone()
            if row is not None and self._lease(row[0]).expires_at > now:
                return None
            epoch = db.execute("SELECT last_fence FROM lease_epochs WHERE run_id=?",
                               (run_id,)).fetchone()
            fence = (epoch[0] if epoch else 0) + 1
            lease = AllocationLease(run_id, allocation_id, incarnation, fence,
                                    now, now, now + ttl_s, protocol_id, config_id)
            payload = self._encoded(asdict(lease))
            db.execute("INSERT INTO lease_epochs VALUES (?,?) ON CONFLICT(run_id) "
                       "DO UPDATE SET last_fence=excluded.last_fence", (run_id, fence))
            db.execute("INSERT INTO leases VALUES (?,?) ON CONFLICT(run_id) "
                       "DO UPDATE SET payload=excluded.payload", (run_id, payload))
            return lease
        return self._transaction(cas)

    def current(self, run_id: str) -> AllocationLease | None:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?", (run_id,)).fetchone()
        return self._lease(row[0]) if row else None

    @staticmethod
    def _same_owner(current: AllocationLease, lease: AllocationLease) -> bool:
        return (current.run_id, current.allocation_id, current.incarnation, current.fence) == \
               (lease.run_id, lease.allocation_id, lease.incarnation, lease.fence)

    def renew(self, lease: AllocationLease, *, ttl_s: float) -> AllocationLease:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        now = self.clock()

        def cas(db: sqlite3.Connection) -> AllocationLease:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?",
                             (lease.run_id,)).fetchone()
            current = self._lease(row[0]) if row else None
            if current is None or not self._same_owner(current, lease) or current.expires_at <= now:
                raise FenceRejected("stale or expired allocation lease renewal")
            renewed = AllocationLease(current.run_id, current.allocation_id,
                                      current.incarnation, current.fence,
                                      current.acquired_at, now, now + ttl_s,
                                      current.protocol_id, current.config_id)
            db.execute("UPDATE leases SET payload=? WHERE run_id=?",
                       (self._encoded(asdict(renewed)), lease.run_id))
            return renewed
        return self._transaction(cas)

    def release(self, lease: AllocationLease) -> None:
        now = self.clock()

        def cas(db: sqlite3.Connection) -> None:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?",
                             (lease.run_id,)).fetchone()
            current = self._lease(row[0]) if row else None
            if (current is None or not self._same_owner(current, lease)
                    or current.expires_at <= now):
                raise FenceRejected("stale or expired allocation lease release")
            db.execute("DELETE FROM leases WHERE run_id=?", (lease.run_id,))
        self._transaction(cas)

    def assert_current(self, lease: AllocationLease) -> None:
        """Check a fence for non-mutating work.

        This is intentionally not a publication primitive: callers MUST use
        :meth:`publish` for commits, checkpoints, and latest pointers so the
        fence check and metadata write occur in one transaction.
        """
        now = self.clock()

        def check(db: sqlite3.Connection) -> None:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?",
                             (lease.run_id,)).fetchone()
            current = self._lease(row[0]) if row else None
            if (current is None or not self._same_owner(current, lease)
                    or current.expires_at <= now):
                raise FenceRejected("operation rejected by newer or expired fence")
        self._transaction(check)

    def publish(self, lease: AllocationLease, *, kind: str, name: str,
                payload: Mapping[str, object]) -> None:
        """Atomically guard a commit, checkpoint, or latest-pointer metadata write."""
        if kind not in PUBLICATION_KINDS or not name:
            raise ValueError("publication kind/name is invalid")
        encoded, now = self._encoded(payload), self.clock()

        def cas(db: sqlite3.Connection) -> None:
            row = db.execute("SELECT payload FROM leases WHERE run_id=?",
                             (lease.run_id,)).fetchone()
            current = self._lease(row[0]) if row else None
            if current is None or not self._same_owner(current, lease) or current.expires_at <= now:
                raise FenceRejected(f"stale {kind} publication")
            existing = db.execute(
                "SELECT fence,payload FROM publications WHERE run_id=? AND kind=? AND name=?",
                (lease.run_id, kind, name)).fetchone()
            if existing is not None:
                if existing == (lease.fence, encoded):
                    return
                raise FenceRejected("immutable publication already exists")
            db.execute("INSERT INTO publications VALUES (?,?,?,?,?)",
                       (lease.run_id, kind, name, lease.fence, encoded))
        self._transaction(cas)

    def read_publication(self, run_id: str, kind: str, name: str) -> dict[str, object] | None:
        with self._connect() as db:
            row = db.execute("SELECT payload FROM publications WHERE run_id=? AND kind=? AND name=?",
                             (run_id, kind, name)).fetchone()
        return json.loads(row[0]) if row else None


def run_if_admitted(store: SQLiteFencedControlStore, *, load_and_run: Callable[[AllocationLease], T],
                    **acquire: object) -> tuple[int, T | None]:
    """Acquire before any model/run callback; a losing allocation exits zero."""
    lease = store.acquire(**acquire)  # type: ignore[arg-type]
    if lease is None:
        return 0, None
    return 0, load_and_run(lease)
