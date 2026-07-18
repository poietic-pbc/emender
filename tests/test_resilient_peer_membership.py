import pytest

from ndm.resilient_peer_membership import PeerMembership, PeerState, StageDeadlines


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def controller():
    clock = Clock()
    policy = StageDeadlines(first_heartbeat=2, boot=3, sync=4, lease=5, drain=2)
    return PeerMembership(policy, clock=clock), clock


def admit(pool, worker, incarnation, generation):
    pool.discover(worker, incarnation=incarnation)
    pool.begin_boot(worker, incarnation)
    pool.begin_sync(worker, incarnation)
    return pool.ready(worker, incarnation, base_generation=generation)


def test_slow_boot_and_late_admission_have_bounded_stage_deadlines():
    pool, clock = controller()
    slow = pool.discover("stable-a", incarnation="boot-1")
    assert slow.state is PeerState.DISCOVER
    pool.begin_boot("stable-a", "boot-1")
    clock.advance(3)
    with pytest.raises(TimeoutError, match="BOOTING stage deadline"):
        pool.begin_sync("stable-a", "boot-1")
    assert pool.records["stable-a"].state is PeerState.EXPIRED

    # A peer arriving after generation 7 was snapshotted synchronizes to the
    # current commit and is eligible for the next generation, without blocking 7.
    admit(pool, "stable-b", "late-1", 8)
    assert pool.active_snapshot(7).size == 0
    assert [p.worker_id for p in pool.active_snapshot(8).peers] == ["stable-b"]


def test_active_membership_is_synchronized_live_ready_not_launch_size():
    pool, _ = controller()
    admit(pool, "ready", "r1", 4)
    pool.discover("booting", incarnation="b1")
    pool.begin_boot("booting", "b1")
    admit(pool, "behind", "h1", 3)
    snapshot = pool.active_snapshot(4)
    assert snapshot.size == 1
    assert [(p.worker_id, p.base_generation) for p in snapshot.peers] == [("ready", 4)]


def test_lease_expiry_and_new_incarnation_rejoin_are_bounded():
    pool, clock = controller()
    admit(pool, "node-7", "old", 2)
    clock.advance(5)
    assert pool.active_snapshot(2).size == 0
    assert pool.records["node-7"].state is PeerState.EXPIRED

    admit(pool, "node-7", "new", 3)
    assert pool.active_snapshot(3).peers[0].incarnation == "new"
    with pytest.raises(ValueError, match="superseded"):
        pool.renew("node-7", "old", base_generation=3)
    with pytest.raises(ValueError, match="new incarnation"):
        pool.discover("node-7", incarnation="new")


def test_catch_up_excludes_peer_until_latest_base_is_validated():
    pool, clock = controller()
    admit(pool, "catcher", "c1", 9)
    pool.catch_up("catcher", "c1", committed_generation=10)
    assert pool.active_snapshot(10).size == 0
    clock.advance(3.5)
    pool.ready("catcher", "c1", base_generation=10)
    assert pool.active_snapshot(10).size == 1
    pool.drain("catcher", "c1")
    assert pool.active_snapshot(10).size == 0
    clock.advance(2)
    assert pool.expire_due()[0].state is PeerState.EXPIRED


def test_sync_and_lease_renewal_deadlines_fail_closed():
    pool, clock = controller()
    record = pool.discover("peer", incarnation="i1")
    pool.begin_boot("peer", record.incarnation)
    pool.begin_sync("peer", record.incarnation)
    clock.advance(4)
    with pytest.raises(TimeoutError, match="SYNCING stage deadline"):
        pool.ready("peer", record.incarnation, base_generation=0)

    admit(pool, "leased", "i2", 0)
    clock.advance(4)
    pool.renew("leased", "i2", base_generation=0)
    clock.advance(4.9)
    assert pool.active_snapshot(0).size == 1
    clock.advance(.1)
    assert pool.active_snapshot(0).size == 0
