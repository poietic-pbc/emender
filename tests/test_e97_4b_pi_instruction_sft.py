import json
import subprocess
import sys

import numpy as np
import tiktoken

from ndm.data.masked_sft_dataset import RECORD_INDEX, sha256
from scripts import build_e97_pi_instruction_sft as builder
from scripts import eval_e97_4b_pi_core as evaluator
from scripts import train_e97_4b_pi_sft as trainer


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


def test_build_and_mix_authorities_are_deterministic_and_target_weighted(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    run("scripts/build_e97_pi_instruction_sft.py", "--output-root", first,
        "--records", 24, "--seed", 11)
    run("scripts/build_e97_pi_instruction_sft.py", "--output-root", second,
        "--records", 24, "--seed", 12)
    first_sha, second_sha = sha256(first / "manifest.json"), sha256(second / "manifest.json")
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


def test_checkpoint_recipe_is_k_aligned_and_bounded():
    args = trainer.merge_args(67_108_864)
    assert args.diloco_merge_bucket_numel == 67_108_864
    assert args.diloco_merge_topology == "global"
    assert args.diloco_outer_optimizer == "avg"
    assert trainer.EXPECTED_PARAMETERS == 4_045_972_080


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
