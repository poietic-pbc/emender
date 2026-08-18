from pathlib import Path

import torch

from scripts.frontier.e97_35b_moe_train import (
    _sft_optimizer_parameter_groups, _sft_optimizer_split_policy,
    _sft_transition_has_policy,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_moe_sft_optimizer_state_factorial_3n.sbatch"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"
OPTIMIZER = ROOT / "ndm/e97_moe_optimizer.py"
CHECKPOINT = ROOT / "ndm/e97_moe_checkpoint.py"


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 1
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.router = torch.nn.Linear(3, 4, bias=False)
        layer.other = torch.nn.Linear(3, 3, bias=False)
        self.layers = torch.nn.ModuleList([layer])
        self.head = torch.nn.Linear(3, 3, bias=False)


def test_optimizer_state_split_partitions_all_parameters_with_preserved_first():
    model = _ToyModel()
    router = model.layers[0].mlp.router.weight
    all_parameters = set(model.parameters())
    router_first = _sft_optimizer_parameter_groups(model, "router-preserved")
    nonrouter_first = _sft_optimizer_parameter_groups(model, "nonrouter-preserved")
    assert len(router_first[0]["params"]) == 1
    assert router_first[0]["params"][0] is router
    assert all(parameter is not router for parameter in router_first[1]["params"])
    assert set(router_first[0]["params"] + router_first[1]["params"]) == all_parameters
    assert any(parameter is router for parameter in nonrouter_first[1]["params"])
    assert set(nonrouter_first[0]["params"] + nonrouter_first[1]["params"]) == all_parameters


def test_optimizer_split_policy_survives_record_reset_transition():
    original = {"optimizer_state": "router-preserved"}
    reset = {
        "optimizer_state": "preserved-exact",
        "previous_sampler_transition": original,
    }
    assert _sft_optimizer_split_policy(original) == "router-preserved"
    assert _sft_optimizer_split_policy(reset) == "router-preserved"
    assert _sft_optimizer_split_policy({"optimizer_state": "preserved-exact"}) is None
    assert _sft_transition_has_policy(
        reset, "optimizer_state", "router-preserved")
    assert not _sft_transition_has_policy(
        reset, "optimizer_sync_policy", "corresponding-lane-gradient-sum-v1")


def test_state_factorial_launcher_has_three_isolated_matched_worlds():
    text = LAUNCHER.read_text()
    assert "#SBATCH -N 3" in text and "#SBATCH --no-requeue" in text
    assert '"NumNodes=3"' in text and '"NumTasks=24"' in text
    assert "Partition=batch" in text and "QOS=$EXPECTED_QOS" in text
    assert "srun --exclusive --exact --nodes=1 --ntasks=8" in text
    assert "WORLD_SIZE=8" in text
    assert "states=(fresh router-preserved nonrouter-preserved)" in text
    assert "lrs=(0.0001 0.0001 0.0001)" in text
    assert '--sft-parent-optimizer-split "$ARM_STATE"' in text
    assert "--sft-validation-exhaustive" in text
    assert "--checkpoint-root" not in text
    assert "scontrol requeue" not in text
    assert "sqlite" not in text.lower()


def test_multigroup_state_paths_fail_closed_and_offload_every_group():
    runner = RUNNER.read_text()
    optimizer = OPTIMIZER.read_text()
    checkpoint = CHECKPOINT.read_text()
    assert '"SFT split optimizer policy mismatch on resume"' in runner
    assert 'optimizer.state[parameter].clear()' in runner
    assert 'for group in self.param_groups:' in optimizer
    assert 'checkpoint optimizer group count mismatch' in checkpoint
    assert 'SFT parent optimizer transition group layout mismatch' in checkpoint
