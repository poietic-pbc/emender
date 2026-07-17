import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


# Load this dependency-free control module without importing ndm's optional
# model stack.  This keeps the lease gate testable on model-free controllers.
SOURCE = Path(__file__).parents[1] / "ndm/fenced_admission.py"
SPEC = importlib.util.spec_from_file_location("fenced_admission_under_test", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
FenceRejected = MODULE.FenceRejected
MAX_CONTROL_BYTES = MODULE.MAX_CONTROL_BYTES
SQLiteFencedControlStore = MODULE.SQLiteFencedControlStore
run_if_admitted = MODULE.run_if_admitted


class Clock:
    def __init__(self): self.now = 100.0
    def __call__(self): return self.now


def _acquire(store, allocation, incarnation="boot"):
    return store.acquire(run_id="run", allocation_id=allocation,
                         incarnation=incarnation, protocol_id="pool-v1",
                         config_id="config-sha", ttl_s=10)


def test_loser_exits_zero_before_model_load_or_run_state_mutation(tmp_path):
    clock = Clock(); store = SQLiteFencedControlStore(tmp_path / "control.db", clock=clock)
    assert _acquire(store, "job-1") is not None
    effects = []
    status, result = run_if_admitted(
        store, run_id="run", allocation_id="job-2", incarnation="boot-2",
        protocol_id="pool-v1", config_id="config-sha", ttl_s=10,
        load_and_run=lambda lease: effects.append(("model-loaded", lease.fence)))
    assert (status, result, effects) == (0, None, [])
    assert store.current("run").allocation_id == "job-1"


def test_expiry_takeover_has_newer_fence_and_rejects_stale_renewal(tmp_path):
    clock = Clock(); store = SQLiteFencedControlStore(tmp_path / "control.db", clock=clock)
    old = _acquire(store, "job-1")
    assert _acquire(store, "job-2") is None
    clock.now = old.expires_at
    new = _acquire(store, "job-2", "boot-2")
    assert new.fence == old.fence + 1
    try:
        store.renew(old, ttl_s=10)
    except FenceRejected as error:
        assert "stale or expired" in str(error)
    else:
        raise AssertionError("stale renewal succeeded")
    assert store.renew(new, ttl_s=5).expires_at == clock.now + 5


def test_newer_fence_rejects_every_stale_publication(tmp_path):
    clock = Clock(); store = SQLiteFencedControlStore(tmp_path / "control.db", clock=clock)
    old = _acquire(store, "job-1")
    clock.now = old.expires_at
    new = _acquire(store, "job-2")
    for kind in ("commit", "checkpoint", "latest"):
        try:
            store.publish(old, kind=kind, name="generation-7", payload={"generation": 7})
        except FenceRejected as error:
            assert f"stale {kind}" in str(error)
        else:
            raise AssertionError(f"stale {kind} succeeded")
        store.publish(new, kind=kind, name="generation-7", payload={"generation": 7})
        assert store.read_publication("run", kind, "generation-7") == {"generation": 7}


def test_publications_are_immutable_idempotent_and_small(tmp_path):
    store = SQLiteFencedControlStore(tmp_path / "control.db")
    lease = _acquire(store, "job-1")
    store.publish(lease, kind="commit", name="g1", payload={"digest": "abc"})
    store.publish(lease, kind="commit", name="g1", payload={"digest": "abc"})
    try:
        store.publish(lease, kind="commit", name="g1", payload={"digest": "changed"})
    except FenceRejected as error:
        assert "immutable" in str(error)
    else:
        raise AssertionError("immutable publication changed")
    try:
        store.publish(lease, kind="checkpoint", name="tensor",
                      payload={"dense": "x" * MAX_CONTROL_BYTES})
    except ValueError as error:
        assert "small-metadata" in str(error)
    else:
        raise AssertionError("dense payload was admitted")


def test_release_preserves_monotonic_fence(tmp_path):
    store = SQLiteFencedControlStore(tmp_path / "control.db")
    first = _acquire(store, "job-1")
    store.release(first)
    assert _acquire(store, "job-2").fence == first.fence + 1


def test_expired_owner_cannot_release_or_pass_guard(tmp_path):
    clock = Clock(); store = SQLiteFencedControlStore(tmp_path / "control.db", clock=clock)
    expired = _acquire(store, "job-1")
    clock.now = expired.expires_at
    for operation in (lambda: store.release(expired),
                      lambda: store.assert_current(expired)):
        try:
            operation()
        except FenceRejected:
            pass
        else:
            raise AssertionError("expired owner retained mutation authority")


def test_store_contains_control_metadata_only_not_ready_or_rank_membership(tmp_path):
    store = SQLiteFencedControlStore(tmp_path / "control.db")
    _acquire(store, "job-1")
    with store._connect() as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"lease_epochs", "leases", "publications"}
    assert not ({"ranks", "workers", "ready", "membership"} & tables)


class FencedAdmissionTests(unittest.TestCase):
    def invoke(self, function):
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_loser(self): self.invoke(test_loser_exits_zero_before_model_load_or_run_state_mutation)
    def test_takeover(self): self.invoke(test_expiry_takeover_has_newer_fence_and_rejects_stale_renewal)
    def test_publication_fences(self): self.invoke(test_newer_fence_rejects_every_stale_publication)
    def test_publication_limits(self): self.invoke(test_publications_are_immutable_idempotent_and_small)
    def test_release(self): self.invoke(test_release_preserves_monotonic_fence)
    def test_expired_guard(self): self.invoke(test_expired_owner_cannot_release_or_pass_guard)
    def test_control_plane_only(self): self.invoke(test_store_contains_control_metadata_only_not_ready_or_rank_membership)


if __name__ == "__main__":
    unittest.main()
