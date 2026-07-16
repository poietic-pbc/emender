import json
import subprocess
import sys
import time

import pytest

from ndm.resilient_e97_topology import (
    ChildSpec, IndependentProcessSupervisor, true_frontier_topology,
    validate_true_topology,
)


def test_two_nodes_are_two_cpu_managers_and_sixteen_real_gpu_trainers():
    specs = true_frontier_topology(2, ["manager"], ["approved-e97-trainer", "--local-steps=40"])
    managers = [item for item in specs if item.role == "manager"]
    trainers = [item for item in specs if item.role == "trainer"]
    assert len(managers) == 2
    assert len(trainers) == 16
    assert all(item.env["CUDA_VISIBLE_DEVICES"] == "" for item in managers)
    assert {item.env["CUDA_VISIBLE_DEVICES"] for item in trainers} == {str(i) for i in range(8)}
    assert all("--local-steps=40" in item.command for item in trainers)


def test_sentinel_workaround_is_rejected():
    specs = list(true_frontier_topology(1, ["manager"], ["trainer"]))
    specs[-1] = ChildSpec("sentinel", 0, 7, ("sleep", "1"), {})
    with pytest.raises(ValueError, match="sentinel"):
        validate_true_topology(specs, node_count=1)


def test_independent_supervision_evicts_failed_trainer_without_manager(tmp_path):
    supervisor = IndependentProcessSupervisor(tmp_path, heartbeat_deadline_s=2,
                                               progress_deadline_s=2)
    healthy = ChildSpec("manager", 0, None,
                        (sys.executable, "-c", "import time; time.sleep(10)"),
                        {"CUDA_VISIBLE_DEVICES": ""})
    failed = ChildSpec("trainer", 0, 3,
                       (sys.executable, "-c", "raise SystemExit(17)"),
                       {"CUDA_VISIBLE_DEVICES": "3"})
    manager = supervisor.start(healthy); trainer = supervisor.start(failed)
    state = tmp_path / "supervision"; state.mkdir()
    now = time.time()
    (state / "node-0-manager.json").write_text(json.dumps({"heartbeat_time": now, "progress_time": now}))
    (state / "node-0-trainer-3.json").write_text(json.dumps({"heartbeat_time": now, "progress_time": now}))
    trainer.wait(2)
    assert supervisor.check() == ("node-0/trainer-3",)
    assert manager.poll() is None
    manager.terminate(); manager.wait(2)


def test_progress_deadline_evicts_only_stalled_manager(tmp_path):
    supervisor = IndependentProcessSupervisor(tmp_path, heartbeat_deadline_s=5,
                                               progress_deadline_s=.1)
    spec = ChildSpec("manager", 0, None,
                     (sys.executable, "-c", "import time; time.sleep(10)"),
                     {"CUDA_VISIBLE_DEVICES": ""})
    process = supervisor.start(spec)
    state = tmp_path / "supervision"; state.mkdir()
    now = time.time()
    (state / "node-0-manager.json").write_text(json.dumps({"heartbeat_time": now, "progress_time": now - 1}))
    assert supervisor.check(now) == ("node-0/manager",)
    process.wait(2)
