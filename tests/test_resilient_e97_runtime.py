import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import socket

import pytest
import torch

from ndm.resilient_e97_roles import LocalFence, LocalTrainerSpool
from ndm.resilient_e97_runtime import (apply_delta, finalize_checkpoint,
                                       assert_node_local_path, outer_state_migration)


ROLE = Path(__file__).parents[1] / "scripts/frontier/resilient_e97_role.py"


def test_manager_publishes_heartbeat_before_heavy_runtime_imports():
    text = ROLE.read_text()
    bootstrap = text.index("_IMPORT_HEARTBEAT = _manager_import_heartbeat()")
    assert bootstrap < text.index("import torch")
    assert bootstrap < text.index("from ndm.resilient_e97_runtime import")
    assert '"stage": "runtime_import"' in text
    assert "os.replace(temporary, state)" in text


def _control_processes(run, bulk, *, run_id, generations, initial=0, resume=""):
    common = ["--run-dir", str(run), "--run-id", run_id, "--generations", str(generations),
              "--initial-generation", str(initial), "--local-steps", "40", "--deadline-s", "15",
              "--source-id", "seed", "--payload-id", "layout", "--code-id", "code",
              "--control", "--bulk-root", str(bulk)]
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
              "--local-steps", "40", "--deadline-s", "15", "--source-id", "seed-sha",
              "--payload-id", "layout-sha", "--code-id", "code-sha", "--control",
              "--bulk-root", str(bulk_root)]
    manager = subprocess.Popen([sys.executable, str(ROLE), "manager", *common],
                               env={**os.environ, "RESILIENT_E97_NODE_RANK": "0"})
    trainers = []
    for rank in range(8):
        trainers.append(subprocess.Popen(
            [sys.executable, str(ROLE), "trainer", *common],
            env={**os.environ, "RESILIENT_E97_NODE_RANK": "0",
                 "RESILIENT_E97_LOCAL_RANK": str(rank)}))
    assert manager.wait(timeout=30) == 0
    assert [item.wait(timeout=30) for item in trainers] == [0] * 8
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
    assert len(handoff["membership"]) == 6
    assert handoff["checkpoint_sha256"] == hashlib.sha256(
        Path(handoff["checkpoint"]).read_bytes()).hexdigest()
    assert not list(mailbox.glob("control-g*/trainer-*"))
    ownership = json.loads((bulk_root / "control/node-0/control/node-0-bulk-ownership.json").read_text())
    assert ownership["shared_run_dir_is_bulk_path"] is False
    assert 0 < ownership["high_water_bytes"] <= ownership["max_bytes"]
    assert ownership["post_release_bytes"] < ownership["high_water_bytes"]
    assert len(list((mailbox / "aggregates").glob("*/manifest.json"))) <= 2
    assert {path.relative_to(tmp_path).parts[0] for path in tmp_path.rglob("*") if path.is_file()} \
        <= {"checkpoints", "handoff"}


def test_two_model_free_managers_exchange_without_collective(tmp_path):
    bulk_root = tmp_path.with_name(tmp_path.name + "-bulk")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
    common = ["--run-dir", str(tmp_path), "--run-id", "network", "--generations", "1",
              "--local-steps", "40", "--deadline-s", "20", "--source-id", "seed",
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
    assert [item.wait(timeout=45) for item in processes] == [0] * len(processes)
    manifests = list((tmp_path / "manager-network/network-generations").glob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["accepted_nodes"] == ["node-0", "node-1"]
    role_source = ROLE.read_text()
    assert all(word not in role_source for word in ("mpi4py", "TCPStore", "RCCL", "all_reduce"))


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
                       initial=2, resume=resume)
    expected = torch.load(uninterrupted / "checkpoints/generation-00000003.pt",
                          weights_only=True)
    actual = torch.load(restarted / "checkpoints/generation-00000003.pt", weights_only=True)
    assert torch.equal(actual["model_state_dict"]["weight"],
                       expected["model_state_dict"]["weight"])
    assert actual["optimizer_state_dict"] == expected["optimizer_state_dict"]
    assert actual["outer_update_state"] == expected["outer_update_state"]
    assert actual["step"] == expected["step"] == 120
    recovery = torch.load(
        bulk_b / "control-b/node-0/recovery/node-0-trainer-0/generation-00000003.pt",
        weights_only=True)
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
    shard = next((tmp_path / "aggregates").rglob("*.f64")); shard.write_bytes(b"bad")
    with pytest.raises(ValueError, match="corrupt"):
        spool.wait_aggregate(fence, deadline=1e20, expected_source_id="source")
    with pytest.raises(ValueError, match="count"):
        apply_delta({"x": torch.ones(1)}, (), eta_outer=1)
    with pytest.raises(ValueError, match="shared run"):
        assert_node_local_path(tmp_path / "live", tmp_path)


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
