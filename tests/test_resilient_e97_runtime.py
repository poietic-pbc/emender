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
                                       outer_state_migration)


ROLE = Path(__file__).parents[1] / "scripts/frontier/resilient_e97_role.py"


def test_eight_independent_trainers_advance_three_exact_generations(tmp_path):
    common = ["--run-dir", str(tmp_path), "--run-id", "control", "--generations", "3",
              "--local-steps", "40", "--deadline-s", "15", "--source-id", "seed-sha",
              "--payload-id", "layout-sha", "--code-id", "code-sha", "--control"]
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
        state = json.loads((tmp_path / "supervision" / f"node-0-trainer-{rank}.json").read_text())
        assert state["generation"] == 3 and state["step"] == 120
        assert state["loss"] > 0
    checkpoint = torch.load(tmp_path / "node-0/trainer-checkpoints/g3.pt", weights_only=True)
    reference = 0.0
    for manifest_path in sorted((tmp_path / "node-0/mailbox/aggregates").glob("*/manifest.json")):
        members = json.loads(manifest_path.read_text())["members"]
        reference += sum((rank + 1) ** 2 for rank in members) / sum(rank + 1 for rank in members)
    assert checkpoint["model_state_dict"]["weight"].item() == pytest.approx(reference)
    handoff = json.loads((tmp_path / "handoff/generation-00000003.json").read_text())
    assert len(handoff["membership"]) == 6
    assert handoff["checkpoint_sha256"] == hashlib.sha256(
        Path(handoff["checkpoint"]).read_bytes()).hexdigest()
    assert not list((tmp_path / "node-0/mailbox").glob("control-g*/trainer-*"))


def test_two_model_free_managers_exchange_without_collective(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
    common = ["--run-dir", str(tmp_path), "--run-id", "network", "--generations", "1",
              "--local-steps", "40", "--deadline-s", "20", "--source-id", "seed",
              "--payload-id", "layout", "--code-id", "code", "--control",
              "--node-count", "2", "--global-quorum", "2", "--coordinator-host", "127.0.0.1",
              "--coordinator-port", str(port)]
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


def test_generation9_missing_outer_policy_and_immutable_reloadable_handoff(tmp_path):
    with pytest.raises(ValueError, match="explicit generation-9"):
        outer_state_migration({"generation": 9, "verified": True}, policy="")
    migration = outer_state_migration(
        {"generation": 9, "verified": True},
        policy="initialize-zero-from-verified-generation-9")
    assert migration["status"] == "initialized_not_restored"
    checkpoint = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": {"x": torch.ones(1)},
                "optimizer_state_dict": {"state": {}}, "step": 40}, checkpoint)
    fence = LocalFence("run", 9, 0, 2, "payload")
    manifest = finalize_checkpoint(
        tmp_path, checkpoint, run_id="run", generation=10, step=40,
        async_chain=["seed-g9"], membership=range(6), fence=fence,
        source_id="source", code_id="code", outer_update_state={}, migration=migration)
    payload = json.loads(manifest.read_text())
    assert payload["contains"] == ["model", "inner_optimizer"]
    assert payload["outer_state_migration"]["status"] == "initialized_not_restored"
    assert torch.load(payload["checkpoint"], weights_only=True)["step"] == 40
    with pytest.raises(FileExistsError, match="immutable"):
        finalize_checkpoint(tmp_path, checkpoint, run_id="run", generation=10, step=40,
                            async_chain=[], membership=range(6), fence=fence,
                            source_id="source", code_id="code", outer_update_state={},
                            migration=migration)
