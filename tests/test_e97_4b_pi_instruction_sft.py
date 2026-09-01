import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import tiktoken
import torch

from ndm.data.masked_sft_dataset import RECORD_INDEX, sha256
from ndm.schedulefree_offload import CPUOffloadAdamWScheduleFree
from scripts import build_e97_pi_instruction_sft as builder
from scripts import build_e97_pi_finalization_repair_sft as repair_builder
from scripts import eval_e97_4b_pi_core as evaluator
from scripts import train_e97_4b_pi_sft as trainer
from scripts import verify_e97_4b_pi_sft_checkpoint as verifier


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], check=True, text=True, capture_output=True)


def test_pi_trace_serialization_is_rs_free_and_assistant_only():
    encoding = tiktoken.get_encoding("p50k_base")
    user, turns, _ = builder.trace("edit", 3, __import__("random").Random(7))
    tokens, masks, text = builder.serialize([("system", builder.SYSTEM), ("user", user), *turns], encoding)
    assert "\x1e" not in text
    assert "Action: read" in text and "Action: edit" in text and "Action: bash" in text
    assert "Final:" in text
    targeted = b"".join(
        encoding.decode_single_token_bytes(token)
        for token, mask in zip(tokens, masks) if mask
    ).decode(errors="replace")
    assert "Action:" in targeted and "Final:" in targeted
    assert "Successfully replaced" not in targeted
    assert user not in targeted


def test_finalization_repair_matches_live_empty_tool_context_and_targets_only_final():
    encoding = tiktoken.get_encoding("p50k_base")
    user, turns, _ = builder.trace("recover-test", 3, __import__("random").Random(7))
    messages = [("system", builder.SYSTEM), ("user", user), *turns]
    tokens, masks, text = repair_builder.serialize_final_only(messages, encoding)
    assert "Tool:\n(no tool output)\n\nAssistant:\nFinal:" in text
    targeted = b"".join(
        encoding.decode_single_token_bytes(token)
        for token, mask in zip(tokens, masks) if mask
    ).decode(errors="replace")
    assert targeted == turns[-1][1] + "\n"
    assert "Action:" not in targeted
    assert masks[-1] == 1


def test_build_and_mix_authorities_are_deterministic_and_target_weighted(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run("scripts/build_e97_pi_instruction_sft.py", "--output-root", first,
        "--records", 24, "--seed", 11)
    run("scripts/build_e97_pi_instruction_sft.py", "--output-root", second,
        "--records", 24, "--seed", 12)
    first_sha, second_sha = sha256(first / "manifest.json"), sha256(second / "manifest.json")
    repair = tmp_path / "repair"
    run("scripts/build_e97_pi_finalization_repair_sft.py", "--source-root", first,
        "--source-sha256", first_sha, "--output-root", repair)
    repair_manifest = json.loads((repair / "manifest.json").read_text())
    assert repair_manifest["source_manifest_sha256"] == first_sha
    assert repair_manifest["counts"]["records"] == 24
    assert repair_manifest["counts"]["assistant_target_tokens"] > 0
    mixed = tmp_path / "mixed"
    run("scripts/build_e97_masked_sft_mix.py", "--output-root", mixed,
        "--source", f"pi={first},{first_sha},2000",
        "--source", f"replay={second},{second_sha},1000", "--seed", 13)
    manifest = json.loads((mixed / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sources"]["pi"]["assistant_target_fraction"] > 0.60
    assert manifest["sources"]["replay"]["assistant_target_fraction"] < 0.40
    index_bytes = (mixed / "records.idx").read_bytes()
    assert len(index_bytes) == manifest["counts"]["records"] * RECORD_INDEX.size
    masks = np.fromfile(mixed / "assistant_mask.uint8.bin", dtype=np.uint8)
    assert int(masks.sum()) == manifest["counts"]["assistant_target_tokens"]
    for key in ("tokens", "mask", "index", "metadata"):
        entry = manifest["outputs"][key]
        assert sha256(mixed / __import__("pathlib").Path(entry["path"]).name) == entry["sha256"]


def test_pi_evaluator_reconstructs_exact_bash_contract():
    user, turns, task = builder.trace("bash", 397, __import__("random").Random(4))
    row = {"id": "pi-native-bash-00000397", "kind": "bash", "user": user, "task": task}
    expected = evaluator.expected_calls(row)
    assert expected == [
        ("bash", {"command": "python -c 'import json; print(json.load(open(\"configs/service-000397.json\"))[\"service\"][\"port\"])'"})
    ]


def test_pi_evaluator_rejects_repetitive_or_ungrounded_final():
    user, turns, task = builder.trace("edit", 9, __import__("random").Random(4))
    row = {"id": "pi-native-edit-00000009", "kind": "edit", "user": user, "task": task}
    good = turns[-1][1]
    assert evaluator.grounded_final(row, good)
    assert not evaluator.grounded_final(row, good + "\n\nAction:\n(no tool output)")
    assert not evaluator.grounded_final(row, "Final: done")


def test_pi_evaluator_reconstructs_exact_recovery_contract(tmp_path):
    user, turns, task = builder.trace("recover-test", 9, __import__("random").Random(4))
    row = {"id": "pi-native-recover-test-00000009", "kind": "recover-test", "user": user, "task": task}
    expected = evaluator.expected_calls(row)
    assert [name for name, _ in expected] == ["bash", "read", "edit", "bash"]
    sandbox = evaluator.make_sandbox(tmp_path, row)
    # Applying the expected edit yields the mechanically checked terminal state.
    _, edit_args = expected[2]
    path = sandbox / edit_args["path"]
    path.write_text(path.read_text().replace(edit_args["oldText"], edit_args["newText"]))
    assert evaluator.verify_sandbox(sandbox, row)


def test_trainer_supports_hash_bound_fresh_optimizer_repair_stages():
    text = open("scripts/train_e97_4b_pi_sft.py").read()
    assert 'lineage.add_argument("--new-stage-from", type=Path)' in text
    assert "new-stage-from must equal the hash-bound parent checkpoint" in text
    assert '"new-stage-saved-x" if args.new_stage_from' in text


def test_checkpoint_recipe_is_k_aligned_and_bounded():
    args = trainer.merge_args(67_108_864)
    assert args.diloco_merge_bucket_numel == 67_108_864
    assert args.diloco_merge_topology == "global"
    assert args.diloco_outer_optimizer == "avg"
    assert trainer.EXPECTED_PARAMETERS == 4_045_972_080


def test_checkpoint_parameter_count_deduplicates_tied_embedding():
    embedding = torch.ones(5, 3)
    state = {"embedding.weight": embedding, "lm_head.weight": embedding,
             "other.weight": torch.ones(2, 3)}
    assert verifier.unique_state_numel(state) == 21


def test_local_sft_optimizer_can_offload_schedulefree_state():
    parameter = torch.nn.Parameter(torch.ones(4))
    args = SimpleNamespace(
        lr=1e-5, weight_decay=0.01, warmup_steps=8,
        offload_schedulefree_state=True,
        schedulefree_offload_pin_memory=0,
        schedulefree_offload_release_gradients=1,
        schedulefree_offload_bucket_numel=16,
    )
    optimizer = trainer.build_optimizer([parameter], args)
    assert isinstance(optimizer, CPUOffloadAdamWScheduleFree)
    optimizer.initialize_state_()
    optimizer.assert_state_offloaded()
    assert optimizer.state[parameter]["z"].device.type == "cpu"


def test_local_finalization_repair_launcher_is_hash_bound_and_fresh_optimizer():
    text = open("scripts/launch_e97_4b_pi_finalization_repair_local.sh").read()
    assert "repair requires CONFIRM_REPAIR=1" in text
    assert '--new-stage-from "$PARENT_CHECKPOINT"' in text
    assert 'sha256sum "$PARENT_CHECKPOINT"' in text
    assert "gpu_lease.sh acquire 8 --no-wait" in text
    assert "--sampler-key 974103" in text
    assert "--offload-schedulefree-state" in text
    assert "LOCAL_PI_FINALIZATION_REPAIR_COMPLETE" in text


def test_local_core_eval_launcher_is_real_pi_and_fail_closed():
    text = open("scripts/launch_e97_4b_pi_core_eval_local.sh").read()
    assert "gpu_lease.sh acquire 8 --no-wait" in text
    assert "numa_local_rank_exec.py" in text
    assert "eval_e97_4b_pi_core.py" in text
    assert "configs/pi/e97-core-tools.ts" in text
    assert "aggregate_e97_4b_pi_core.py" in text
    assert "LOCAL_PI_CORE_EVAL_COMPLETE" in text
    assert 'sha256sum "$CHECKPOINT"' in text
    assert 'sha256sum "$CLI_IMAGE"' in text


def test_local_launcher_uses_ddp_numa_and_cpu_offload():
    text = open("scripts/launch_e97_4b_pi_sft_local.sh").read()
    assert "torchrun --standalone --nproc_per_node=\"$WORLD_SIZE\"" in text
    assert "scripts/numa_local_rank_exec.py" in text
    assert "--offload-schedulefree-state" in text
    assert "--island-size 8" in text
    assert "NCCL_P2P_DISABLE=1" in text
    assert 'export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"' in text
    assert "NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX" in text
    assert "gpu_lease.sh acquire 8 --no-wait" in text
    assert 'canary requires RESUME naming the qualification checkpoint' in text
    assert 'RESUME_ARGS=(--resume "$RESUME")' in text
    assert "verify_e97_4b_pi_sft_checkpoint.py" in text


def test_frontier_launcher_has_required_scheduler_and_fail_stop_contracts():
    text = open("scripts/frontier/e97_4b_pi_sft.sbatch").read()
    assert "#SBATCH -p batch" in text
    assert "#SBATCH -q debug" in text
    assert "#SBATCH --no-requeue" in text
    assert "Partition=${EXPECTED_PARTITION}" in text
    assert "QOS=${EXPECTED_QOS}" in text
    assert "|${EXPECTED_PARTITION}|${EXPECTED_QOS}|" in text
    assert "LOCAL_RANK=0" in text
    assert "TRITON_CACHE_DIR=/tmp/e97-4b-pi-sft-${SLURM_JOB_ID}-${SLURM_PROCID}" in text
    assert "--kill-on-bad-exit=1" in text
    assert 'git cat-file -e "${SOURCE_COMMIT}^{commit}"' in text
    assert 'git rev-parse HEAD' not in text


def test_real_pi_eval_uses_hash_pinned_sandbox_extension():
    extension = open("configs/pi/e97-core-tools.ts").read()
    assert 'name: "read"' in extension
    assert 'name: "bash"' in extension
    assert 'name: "edit"' in extension
    assert 'name: "write"' in extension
    assert 'runner, "--image", image, "--image-sha256", imageSha256' in extension
    launcher = open("scripts/frontier/e97_4b_pi_core_eval_1n.sbatch").read()
    assert "--no-requeue" in launcher
    assert "|batch|debug|" in launcher
    assert "LOCAL_RANK=0" in launcher
    assert "--pi-core-canonical-system" in launcher
    assert "configs/pi/e97-core-tools.ts" in launcher
    assert "--kill-on-bad-exit=1" in launcher
    assert 'git rev-parse HEAD' not in launcher
