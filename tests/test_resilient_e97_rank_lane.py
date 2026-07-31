import json
import subprocess
import sys
import time
from pathlib import Path


LANE = Path(__file__).parents[1] / "scripts/frontier/resilient_e97_rank_lane.py"


def test_trainer_lane_publishes_atomic_terminal(tmp_path):
    result = subprocess.run([sys.executable, str(LANE), "--run-dir", str(tmp_path),
                             "--local-rank", "0", "--node-rank", "1", "--timeout-s", "2",
                             "--", sys.executable, "-c", "raise SystemExit(7)"], check=False)
    assert result.returncode == 7
    payload = json.loads((tmp_path / "rank_lanes/node-00001.terminal.json").read_text())
    assert payload["exit_code"] == 7


def test_sentinel_never_executes_trainer_and_follows_terminal(tmp_path):
    sentinel = subprocess.Popen([sys.executable, str(LANE), "--run-dir", str(tmp_path),
                                 "--local-rank", "3", "--node-rank", "0", "--timeout-s", "2",
                                 "--", "command-must-not-run"])
    time.sleep(0.1)
    terminal = tmp_path / "rank_lanes/node-00000.terminal.json"
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_text('{"exit_code": 0}\n')
    assert sentinel.wait(timeout=3) == 0
    heartbeat = json.loads((tmp_path / "rank_lanes/node-00000-local-03.json").read_text())
    assert heartbeat["role"] == "sentinel"
    assert heartbeat["state"] == "trainer_terminal"


def test_sentinel_deadline_is_bounded(tmp_path):
    result = subprocess.run([sys.executable, str(LANE), "--run-dir", str(tmp_path),
                             "--local-rank", "7", "--node-rank", "0", "--timeout-s", "0.05"],
                            check=False, timeout=2)
    assert result.returncode == 124
