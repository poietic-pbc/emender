from pathlib import Path

import pytest

from ndm.e97_moe_agent_server import TorchE97MoEAgentEngine


class _FakeLoaded:
    config = {"tokenizer": "p50k_base"}
    checkpoint_path = Path("generation")


def test_moe_agent_engine_rejects_invalid_ingest_mode(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 8)
    with pytest.raises(ValueError, match="ingest_mode"):
        TorchE97MoEAgentEngine(_FakeLoaded(), node_group=object(), ingest_mode="full")


def test_moe_agent_engine_requires_eight_rank_group(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    with pytest.raises(RuntimeError, match="eight-rank"):
        TorchE97MoEAgentEngine(_FakeLoaded(), node_group=object())


def test_moe_agent_server_uses_structured_completion_not_rs_as_success():
    source = Path("ndm/e97_moe_agent_server.py").read_text()
    assert "generated_turn_is_complete" in source
    assert 'token == 218' in source
    assert "_broadcast_finished" in source


def test_frontier_launcher_binds_partition_and_qos_explicitly():
    source = Path("scripts/frontier/e97_moe_agent_pi_cli_eval_1n.sbatch").read_text()
    assert "#SBATCH -p batch" in source
    assert "#SBATCH -q debug" in source
    assert "Partition=batch" in source
    assert "QOS=debug" in source
    assert "--no-requeue" in source
    assert "serve_e97_moe_agent_openai.py" in source
