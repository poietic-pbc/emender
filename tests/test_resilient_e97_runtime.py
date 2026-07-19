import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from ndm.resilient_e97_roles import LocalFence, LocalTrainerSpool
from ndm.resilient_e97_runtime import (apply_delta, finalize_checkpoint,
                                       assert_node_local_path, flatten_tensors,
                                       outer_state_migration)
from ndm.fenced_admission import FenceRejected, SQLiteFencedControlStore
from ndm.native_e97_runtime import state_digest


ROLE = Path(__file__).parents[1] / "scripts/frontier/resilient_e97_role.py"


def test_native_state_digest_hashes_exact_bfloat16_storage_bits():
    state = {"bf": torch.tensor([1.0, -2.0], dtype=torch.bfloat16)}
    expected = hashlib.sha256(b"emender-native-e97-base-v1\0")
    expected.update((2).to_bytes(4, "little")); expected.update(b"bf")
    expected.update(b"torch.bfloat16\0")
    expected.update((1).to_bytes(4, "little"))
    expected.update((2).to_bytes(8, "little"))
    expected.update(b"\x80\x3f\x00\xc0")

    assert state_digest(state) == expected.digest()


def test_manager_publishes_heartbeat_before_heavy_runtime_imports():
    text = ROLE.read_text()
    bootstrap = text.index("_IMPORT_HEARTBEAT = _role_import_heartbeat()")
    assert bootstrap < text.index("import torch")
    assert bootstrap < text.index("from ndm.resilient_e97_runtime import")
    assert '"stage": "runtime_import"' in text
    assert "os.replace(temporary, state)" in text


def test_live_native_selection_is_wired_and_python_debug_remains_explicit():
    from ndm.native_artifacts import NATIVE_CXI, NATIVE_TEST, PYTHON_TCP_DEBUG
    from scripts.frontier import resilient_e97_role as role

    for backend in (PYTHON_TCP_DEBUG, NATIVE_TEST, NATIVE_CXI):
        role._require_wired_dense_runtime(backend)

    source = ROLE.read_text()
    native_manager = source[source.index("def _native_manager(args)"):
                            source.index("def manager(args)")]
    assert "LocalTrainerSpool(" not in native_manager
    assert "DistributedOwnerServer(" not in native_manager
    assert "NativeManagerSession.start(" in native_manager
    assert "spool = (LocalTrainerSpool" in source
    assert "if not native else None" in source
    assert "manager/trainer native runtime digest mismatch" in source
    assert "resume checkpoint native runtime digest mismatch" in source
    assert "role recovery native runtime digest mismatch" in source


def test_native_manager_endpoint_lifetime_spans_all_configured_generations():
    from scripts.frontier import resilient_e97_role as role

    args = SimpleNamespace(deadline_s=600.0, generations=3)
    assert role._native_manager_session_lifetime_s(args) == 1800.0
    manager = ROLE.read_text()[ROLE.read_text().index("def _native_manager(args)"):]
    assert "deadline_s=_native_manager_session_lifetime_s(args)" in manager


def test_owner_endpoint_snapshot_filters_control_only_lease_metadata():
    from scripts.frontier import resilient_e97_role as role

    endpoint = role._owner_endpoint_from_snapshot({
        "worker_id": "node-0", "incarnation": "node-0-boot",
        "host": "127.0.0.1", "port": 29571,
        "backend": "python-tcp-debug", "lease_expiry": 1234.5,
    })

    assert endpoint.worker_id == "node-0"
    assert endpoint.incarnation == "node-0-boot"
    assert not hasattr(endpoint, "lease_expiry")


def test_native_owner_credits_follow_reciprocal_pair_route_readiness():
    source = ROLE.read_text()
    manager = source[source.index("def _native_manager(args)"):
                     source.index("def manager(args)")]
    install = manager.index("session.install_routes(endpoints)")
    reciprocal_ready = manager.index("pool_client.await_peer_route_ready(")
    exchange = manager.index("_native_peer_exchange(")

    assert install < reciprocal_ready < exchange
    assert '"native_route_readiness"' in manager
    assert "pairwise=True" in manager


def test_terminal_native_follower_reuses_fenced_authoritative_checkpoint(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    checkpoint = tmp_path / "checkpoints/generation-00000001-fence-00000001.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authoritative-generation-one")
    manifest = tmp_path / "handoff/generation-00000001-fence-00000001.json"
    manifest.parent.mkdir(parents=True)
    value = {
        "schema": 1, "finalized": True, "run_id": "run-a",
        "payload_id": "payload-a", "source_id": "source-a",
        "generation": 1, "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "fence": {"coordinator_epoch": 1},
    }
    manifest.write_text(json.dumps(value, sort_keys=True))
    latest = tmp_path / "handoff/latest.json"
    latest.write_text(json.dumps({
        "generation": 1, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    args = SimpleNamespace(
        run_id="run-a", payload_id="payload-a", source_id="source-a",
        coordinator_epoch=1)

    assert role._terminal_native_checkpoint(
        tmp_path, args, completed=1, deadline=time.monotonic() + 1) == checkpoint
    value["payload_id"] = "stale-payload"
    manifest.write_text(json.dumps(value, sort_keys=True))
    latest.write_text(json.dumps({
        "generation": 1, "fence": 1, "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }))
    with pytest.raises(ValueError, match="identity"):
        role._terminal_native_checkpoint(
            tmp_path, args, completed=1, deadline=time.monotonic() + 1)


def test_native_trainer_apply_lanes_are_serialized_by_local_rank(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    control = tmp_path / "control"
    control.mkdir()
    args = SimpleNamespace(run_id="run-a", coordinator_epoch=4)
    observed = {}

    def wait_for_rank_one():
        observed["marker"] = role._wait_for_native_apply_lane(
            control, args, generation=2, rank=1,
            result_root="ab" * 32, deadline=time.monotonic() + 2)

    waiter = threading.Thread(target=wait_for_rank_one)
    waiter.start()
    time.sleep(.05)
    assert waiter.is_alive(), "rank one must not contend with rank zero's result view"
    role.atomic_metadata(control / "native-result-applied-00000002-00.json", {
        "run_id": "run-a", "fence_epoch": 4, "generation": 2,
        "result_root": "ab" * 32, "rank": 0,
    })
    waiter.join(2)

    assert not waiter.is_alive()
    assert observed["marker"]["rank"] == 0
    assert role._wait_for_native_apply_lane(
        control, args, generation=2, rank=0,
        result_root="ab" * 32, deadline=time.monotonic() + .1) is None
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args)"):]
    visible = trainer.index("manifest, aggregate = native_context.__enter__()")
    lane = trainer.index("_wait_for_native_apply_lane(", visible)
    apply = trainer.index("state = apply_delta(", lane)
    assert visible < lane < apply
    assert "deadline=exchange_deadline" in trainer[lane:apply]


def test_native_apply_lane_excludes_durable_recovery_checkpoint_io():
    """A slow local checkpoint must not hold the shared-result read lane.

    Intermediate generations persist one trainer recovery checkpoint per GPU.
    Charging that disk write to the next rank's native result-view lane makes
    eight otherwise bounded applies exceed the 60 second APPLY budget.
    """
    trainer = ROLE.read_text()[ROLE.read_text().index("def trainer(args)"):]
    wait = trainer.index("_wait_for_native_apply_lane(")
    apply = trainer.index("state = apply_delta(", wait)
    lane_credit = trainer.index("native-result-applied-", apply)
    recovery_save = trainer.index("torch.save(", lane_credit)
    durable_receipt = trainer.index("native-applied-", recovery_save)

    assert wait < apply < lane_credit < recovery_save < durable_receipt
    timer_reset = trainer.rfind("trainer_apply_started = time.monotonic()", wait, apply)
    assert timer_reset > wait, "the APPLY SLO must measure apply, not lane waiting"


def test_manager_uses_exchange_commit_bound_for_all_recovery_receipts():
    """Eight durable checkpoints are an aggregate commit phase, not one apply."""
    source = ROLE.read_text()
    manager = source[source.index("def _native_manager(args)"):
                     source.index("def manager(args)")]
    receipt_loop = manager.index("for rank in range(args.local_quorum):")
    commit = manager.index("session.commit(", receipt_loop)
    window = manager[receipt_loop - 300:commit]

    assert "recovery_deadline" in window
    assert "min(args.deadline_s, 180.0)" in window
    assert "deadline=recovery_deadline" in window
    assert "apply_deadline" not in window


def test_import_liveness_does_not_refresh_runtime_import_progress_deadline():
    text = ROLE.read_text()
    bootstrap = text[text.index("def _role_import_heartbeat"):
                     text.index("_IMPORT_HEARTBEAT = _role_import_heartbeat()")]
    assert "progress_started = time.time()" in bootstrap
    assert '"progress_time": progress_started' in bootstrap
    assert '"progress_time": now' not in bootstrap


def _control_processes(run, bulk, *, run_id, generations, initial=0, resume="", epoch=1):
    common = ["--run-dir", str(run), "--run-id", run_id, "--generations", str(generations),
              "--initial-generation", str(initial), "--local-steps", "40", "--deadline-s", "15",
              "--source-id", "seed", "--payload-id", "layout", "--code-id", "code",
              "--coordinator-epoch", str(epoch), "--control", "--bulk-root", str(bulk)]
    if resume:
        common += ["--resume-handoff", str(resume)]
    processes = [subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                                  env={**os.environ, "RESILIENT_E97_NODE_RANK": "0"})]
    for rank in range(6):
        processes.append(subprocess.Popen(
            [sys.executable, str(ROLE), "trainer", *common],
            env={**os.environ, "RESILIENT_E97_NODE_RANK": "0",
                 "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    assert [item.wait(timeout=40) for item in processes] == [0] * len(processes)


def test_eight_independent_trainers_advance_three_exact_generations(tmp_path):
    bulk_root = tmp_path.with_name(tmp_path.name + "-bulk")
    common = ["--run-dir", str(tmp_path), "--run-id", "control", "--generations", "3",
              "--local-steps", "40", "--deadline-s", "30", "--source-id", "seed-sha",
              "--payload-id", "layout-sha", "--code-id", "code-sha", "--control",
              "--bulk-root", str(bulk_root), "--local-quorum", "8"]
    manager = subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                               env={**os.environ, "RESILIENT_E97_NODE_RANK": "0"})
    trainers = []
    for rank in range(8):
        trainers.append(subprocess.Popen(
            [sys.executable, str(ROLE), "trainer", *common],
            env={**os.environ, "RESILIENT_E97_NODE_RANK": "0",
                 "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    assert manager.wait(timeout=60) == 0
    assert [item.wait(timeout=60) for item in trainers] == [0] * 8
    for rank in range(8):
        state = json.loads((bulk_root / "control/node-0/supervision" /
                            f"node-0-trainer-{rank}.json").read_text())
        assert state["generation"] == 3 and state["step"] == 120
        assert state["loss"] > 0
    checkpoint = torch.load(tmp_path / "checkpoints/generation-00000003.pt",
                            weights_only=True)
    reference = 0.0
    mailbox = bulk_root / "control/node-0/mailbox"
    for manifest_path in sorted((bulk_root / "control/node-0/control").glob(
            "node-0-generation-*.json")):
        members = json.loads(manifest_path.read_text())["members"]
        reference += sum((rank + 1) ** 2 for rank in members) / sum(rank + 1 for rank in members)
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(reference)
    handoff = json.loads((tmp_path / "handoff/generation-00000003.json").read_text())
    assert handoff["membership"] == ["node-0"]
    assert handoff["checkpoint_sha256"] == hashlib.sha256(
        Path(handoff["checkpoint"]).read_bytes()).hexdigest()
    assert not list(mailbox.glob("control-g*/trainer-*"))
    ownership = json.loads((bulk_root / "control/node-0/control/node-0-bulk-ownership.json").read_text())
    assert ownership["shared_run_dir_is_bulk_path"] is False
    assert 0 < ownership["high_water_bytes"] <= ownership["max_bytes"]
    assert ownership["post_release_bytes"] < ownership["high_water_bytes"]
    assert ownership["published_files"] <= 2 * 3
    assert len(list((mailbox / "aggregates").glob("*/manifest.json"))) <= 2
    assert {path.relative_to(tmp_path).parts[0] for path in tmp_path.rglob("*") if path.is_file()} \
        <= {"checkpoints", "handoff"}


def test_two_model_free_managers_exchange_without_collective(tmp_path):
    bulk_root = tmp_path.with_name(tmp_path.name + "-bulk")
    # The live runtime derives two owner endpoints at coordinator_port+1/+2.
    # Reserving only the coordinator port is racy on shared login nodes: a
    # healthy manager can fail before READY because either adjacent port is in
    # use. Verify the complete local fixture block before launching processes.
    port = None
    # Do not allocate the fixture from Linux's ephemeral client-port range.
    # Closing a port-0 probe lets any of the many concurrent subprocesses on a
    # shared login node immediately reuse it for an outbound connection before
    # the manager binds. A PID-distributed scan below 32768 avoids that kernel
    # allocator race while retaining the complete three-port probe.
    for attempt in range(256):
        probes = []
        try:
            candidate = 20_000 + ((os.getpid() * 3 + attempt * 3) % 10_000)
            first = socket.socket()
            first.bind(("127.0.0.1", candidate))
            probes.append(first)
            for offset in (1, 2):
                probe = socket.socket()
                probe.bind(("127.0.0.1", candidate + offset))
                probes.append(probe)
            port = candidate
            break
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
    assert port is not None, "no free contiguous local three-port block"
    common = ["--run-dir", str(tmp_path), "--run-id", "network", "--generations", "1",
              "--local-steps", "40", "--deadline-s", "60", "--source-id", "seed",
              "--payload-id", "layout", "--code-id", "code", "--control",
              "--node-count", "2", "--global-quorum", "2", "--coordinator-host", "127.0.0.1",
              "--coordinator-port", str(port), "--bulk-root", str(bulk_root)]
    processes = []
    for node in range(2):
        processes.append(subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                                          env={**os.environ, "RESILIENT_E97_NODE_RANK": str(node)}))
    for node in range(2):
        for rank in range(6):
            processes.append(subprocess.Popen(
                [sys.executable, str(ROLE), "trainer", *common],
                env={**os.environ, "RESILIENT_E97_NODE_RANK": str(node),
                     "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    # Fourteen cold Python/torch processes contend on shared login nodes. Keep
    # this local fixture bounded without conflating login load with the live
    # Frontier READY/K40 stage SLOs exercised by the launcher tests and jobs.
    assert [item.wait(timeout=120) for item in processes] == [0] * len(processes)
    manifests = list((tmp_path / "retained-evidence/pool-control").glob("*.jsonl"))
    assert len(manifests) == 1
    closes = [json.loads(line) for line in manifests[0].read_text().splitlines()]
    assert {item["worker_id"] for item in closes[-1]["frozen_identities"]} \
        == {"node-0", "node-1"}
    role_source = ROLE.read_text()
    assert all(word not in role_source for word in ("mpi4py", "TCPStore", "RCCL", "all_reduce"))
    assert "QuorumTransportServer" not in role_source
    assert "central_full_model_broker\": False" in role_source
    for node in range(2):
        generation = json.loads((bulk_root / f"network/node-{node}/control" /
                                 f"node-{node}-generation-00000000.json").read_text())
        assert generation["central_full_model_broker"] is False
        assert generation["p2p_bytes_sent"] > 0
        assert generation["redistribution_bytes"] > 0
        telemetry = [json.loads(line) for line in (
            bulk_root / f"network/node-{node}/telemetry/node-{node}-manager-pool.jsonl"
        ).read_text().splitlines()]
        stages = {item["stage"] for item in telemetry}
        assert {"ready", "k40_and_local_reduce", "freeze",
                "owner_transport_redistribution"} <= stages
        transport = next(item for item in telemetry
                         if item["stage"] == "owner_transport_redistribution")
        assert transport["within_slo"] is True
        assert transport["bytes_per_second"] > 0
        assert transport["released_bytes"] > 0


def test_fresh_process_restart_matches_uninterrupted_continuation(tmp_path):
    uninterrupted, restarted = tmp_path / "uninterrupted", tmp_path / "restarted"
    bulk_a = tmp_path.with_name(tmp_path.name + "-bulk-a")
    bulk_b = tmp_path.with_name(tmp_path.name + "-bulk-b")
    _control_processes(uninterrupted, bulk_a, run_id="control-a", generations=3)
    _control_processes(restarted, bulk_b, run_id="control-b", generations=2)
    resume = restarted / "handoff/generation-00000002.json"
    saved = json.loads(resume.read_text())
    loaded = torch.load(saved["checkpoint"], weights_only=True)
    assert loaded["generation"] == saved["generation"] == 2
    assert loaded["outer_update_state"] == saved["outer_update_state"]
    _control_processes(restarted, bulk_b, run_id="control-b", generations=1,
                       initial=2, resume=resume, epoch=2)
    expected = torch.load(uninterrupted / "checkpoints/generation-00000003.pt",
                          weights_only=True)
    actual = torch.load(restarted / "checkpoints/generation-00000003.pt", weights_only=True)
    assert torch.equal(actual["model_state_dict"]["weight"],
                       expected["model_state_dict"]["weight"])
    assert actual["optimizer_state_dict"] == expected["optimizer_state_dict"]
    assert actual["outer_update_state"] == expected["outer_update_state"]
    assert actual["step"] == expected["step"] == 120
    assert actual["coordinator_epoch"] == 2
    recovery_record = json.loads((
        bulk_b / "control-b/node-0/control/recovery/node-0-trainer-0.json"
    ).read_text())
    assert Path(recovery_record["checkpoint"]) == (
        restarted / "checkpoints/generation-00000003.pt").resolve()
    assert not (bulk_b /
        "control-b/node-0/recovery/node-0-trainer-0/generation-00000003.pt").exists()
    recovery = torch.load(recovery_record["checkpoint"], weights_only=True)
    assert recovery["identity"] == "node-0-trainer-0"
    assert recovery["generation"] == 3 and recovery["step"] == 120
    assert {"model_state_dict", "optimizer_state_dict", "outer_update_state",
            "membership", "fence", "async_chain"} <= recovery.keys()


def test_apply_identity_deadline_and_corruption_fail_closed(tmp_path):
    spool = LocalTrainerSpool(tmp_path, 4096)
    fence = LocalFence("run", 0, 0, 1, "layout")
    with pytest.raises(TimeoutError, match="deadline"):
        spool.wait_aggregate(fence, deadline=0, expected_source_id="source")
    spool.publish_aggregate(fence, [0], [torch.tensor([2.])], weight=1, source_id="source")
    with pytest.raises(ValueError, match="identity"):
        spool.wait_aggregate(fence, deadline=1e20, expected_source_id="other")
    shard = next((tmp_path / "aggregates").rglob("*.data")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        spool.wait_aggregate(fence, deadline=1e20, expected_source_id="source")
    with pytest.raises(ValueError, match="count"):
        apply_delta({"x": torch.ones(1)}, (), eta_outer=1)
    with pytest.raises(ValueError, match="shared run"):
        assert_node_local_path(tmp_path / "live", tmp_path)


def test_delta_publication_and_apply_are_bounded_across_parameter_boundaries():
    delta = {"z": torch.arange(11, dtype=torch.float32),
             "a": torch.arange(5, dtype=torch.float32) + 20}
    shards = tuple(flatten_tensors(delta, chunk_elements=6))
    assert [shard.numel() for shard in shards] == [6, 6, 4]
    assert all(shard.dtype == torch.float32 and shard.device.type == "cpu" for shard in shards)
    base = {name: torch.zeros_like(value) for name, value in delta.items()}
    applied = apply_delta(base, shards, eta_outer=.5)
    for name in delta:
        assert torch.equal(applied[name], delta[name] * .5)


def test_pinned_cold_start_outer_policy_and_immutable_reloadable_handoff(tmp_path):
    from ndm.resilient_e97_runtime import PINNED_STEP_1525000_SHA256
    seed = {"step": 1525000, "sha256": PINNED_STEP_1525000_SHA256}
    with pytest.raises(ValueError, match="approved initialization"):
        outer_state_migration(seed, policy="")
    migration = outer_state_migration(
        seed, policy="initialize-from-approved-config",
        approved_config={"algorithm": "weighted-mean", "eta_outer": 1.0})
    assert migration["status"] == "initialized_not_restored"
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": {"x": torch.ones(1)},
                "optimizer_state_dict": {"state": {}},
                "outer_update_state": migration["state"], "step": 40,
                "generation": 10, "run_id": "run", "source_id": "source",
                "payload_id": "payload", "coordinator_epoch": 2}, checkpoint)
    fence = LocalFence("run", 9, 0, 2, "payload")
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=10, step=40,
        async_chain=["pinned-step-1525000"], membership=range(6), fence=fence,
        source_id="source", code_id="code", outer_update_state=migration["state"],
        migration=migration)
    payload = json.loads(manifest.read_text())
    assert payload["contains"] == ["model", "inner_optimizer"]
    assert payload["outer_state_migration"]["status"] == "initialized_not_restored"
    assert torch.load(payload["checkpoint"], weights_only=True)["step"] == 40
    with pytest.raises(FileExistsError, match="immutable"):
        finalize_checkpoint(tmp_path, checkpoint, run_id="run", generation=10, step=40,
                            async_chain=[], membership=range(6), fence=fence,
                            source_id="source", code_id="code",
                            outer_update_state=migration["state"],
                            migration=migration)


def test_fenced_atomic_global_commit_and_newer_allocation_restart(tmp_path):
    now = [100.0]
    store = SQLiteFencedControlStore(tmp_path / "pool.sqlite", clock=lambda: now[0])
    old = store.acquire(run_id="run", allocation_id="job-a", incarnation="a",
                        protocol_id="pool-v1", config_id="cfg", ttl_s=10)
    checkpoint = tmp_path / "generation-1.pt"
    torch.save({"model_state_dict": {"x": torch.tensor([3.0])},
                "optimizer_state_dict": {"state": {}},
                "outer_update_state": {"algorithm": "weighted-mean"},
                "step": 40, "generation": 1, "run_id": "run", "source_id": "seed",
                "payload_id": "layout", "coordinator_epoch": old.fence,
                "accepted_tokens": 17}, checkpoint)
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={"layout": "layout", "code": "code"},
        control_store=store, allocation_lease=old)
    latest = store.read_publication("run", "latest", "authoritative")
    assert latest["generation"] == 1 and latest["accepted_tokens"] == 17
    assert manifest.name == "generation-00000001-fence-00000001.json"
    assert json.loads(manifest.read_text())["membership"] == ["node-a:inc-a"]
    assert finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=1, step=40,
        async_chain=["seed"], membership=["node-a:inc-a"],
        fence=LocalFence("run", 0, 0, old.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=17,
        generation_identity={"run_id": "run", "generation": 0,
                             "attempt": 0, "fence": old.fence},
        digests={"layout": "layout", "code": "code"},
        control_store=store, allocation_lease=old) == manifest

    now[0] = old.expires_at
    new = store.acquire(run_id="run", allocation_id="job-b", incarnation="b",
                        protocol_id="pool-v1", config_id="cfg", ttl_s=10)
    assert new.fence == old.fence + 1
    loaded = torch.load(json.loads(manifest.read_text())["checkpoint"], weights_only=True)
    assert loaded["accepted_tokens"] == 17
    stale_checkpoint = tmp_path / "generation-2.pt"
    torch.save({**loaded, "generation": 2, "step": 80}, stale_checkpoint)
    with pytest.raises(FenceRejected):
        finalize_checkpoint(
            tmp_path, stale_checkpoint, run_id="run", generation=2, step=80,
            async_chain=[], membership=[], fence=LocalFence("run", 1, 0, old.fence, "layout"),
            source_id="seed", code_id="code",
            outer_update_state={"algorithm": "weighted-mean"}, migration={},
            accepted_tokens=17, control_store=store, allocation_lease=old)
    assert not (tmp_path / "handoff/generation-00000002.json").exists()

    fresh_checkpoint = tmp_path / "generation-2-fresh.pt"
    torch.save({**loaded, "generation": 2, "step": 80,
                "coordinator_epoch": new.fence, "accepted_tokens": 25},
               fresh_checkpoint)
    continued = finalize_checkpoint(
        tmp_path, fresh_checkpoint, run_id="run", generation=2, step=80,
        async_chain=[str(manifest)], membership=["node-a:inc-b"],
        fence=LocalFence("run", 1, 0, new.fence, "layout"), source_id="seed",
        code_id="code", outer_update_state={"algorithm": "weighted-mean"},
        migration={"status": "restored"}, accepted_tokens=25,
        generation_identity={"run_id": "run", "generation": 1,
                             "attempt": 0, "fence": new.fence},
        control_store=store, allocation_lease=new)
    assert continued.name == "generation-00000002-fence-00000002.json"
    assert store.read_publication("run", "latest", "authoritative") == {
        "accepted_tokens": 25, "fence": new.fence, "generation": 2,
        "manifest": str(continued.resolve()),
        "manifest_sha256": hashlib.sha256(continued.read_bytes()).hexdigest(),
    }
