import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frontier/e97_4b_from_scratch.json"
PAYLOAD = ROOT / "scripts/frontier/e97_4b_from_scratch.sbatch"
SUBMIT = ROOT / "scripts/frontier/submit_e97_4b_from_scratch.sh"
COLLECTOR = ROOT / "scripts/frontier/e97_4b_from_scratch_collector.sh"


def test_frontier_4b_config_preserves_shape_budget_and_local_merge_work():
    cfg = json.loads(CONFIG.read_text())
    assert cfg["schema"] == "emender-e97-4b-frontier-from-scratch-v1"
    assert cfg["target_parameters"] == 4_045_972_080
    assert cfg["production_nodes"] == 256
    assert cfg["production_tasks_per_node"] == 8
    assert cfg["production_walltime"] == "06:00:00"
    assert cfg["production_train_minutes"] == 345
    assert cfg["bootstrap_smoke_walltime"] == "00:20:00"
    assert cfg["bootstrap_smoke_steps"] == 256
    train = cfg["training"]
    assert train["batch_size_per_rank"] == 1
    assert train["global_tokens_per_step"] == 256 * 8 * 1 * 2048
    # K128/B1 is slightly more conservative than the proven 256-node
    # E97-1.3B K40/B4 authority (128 versus 160 local samples per merge).
    assert train["batch_size_per_rank"] * train["diloco_k"] == 128
    assert train["save_every"] == 2 * train["diloco_k"]
    assert cfg["target_steps"] * train["global_tokens_per_step"] >= cfg["target_tokens"]
    assert cfg["target_tokens"] == 20 * cfg["target_parameters"]


def test_frontier_4b_payload_is_fixed_world_counter_sampled_and_fail_stop():
    text = PAYLOAD.read_text()
    assert "source scripts/frontier/activate_emender_frontier.sh" in text
    assert "--sampler_schema emender-byte-window-counter-v1" in text
    assert "--sampler_data_world_size \"$EXPECTED_WORLD_SIZE\"" in text
    assert "--kill-on-bad-exit=1" in text
    assert "--offload_schedulefree_state" not in text
    assert "rung authority is fixed at 4 nodes / 32 ranks" in text
    assert "batch probes are fixed at 2 nodes / 16 ranks" in text
    assert 'BATCH_SIZE=${RUN_MODE#probe_b}' in text
    assert 'DILOCO_K=$((128 / BATCH_SIZE))' in text
    assert 'SAVE_EVERY=$((2 * DILOCO_K))' in text
    assert "bootstrap authority is fixed at 256 nodes / 2048 ranks" in text
    assert "production authority is fixed at 256 nodes / 2048 ranks" in text
    assert "EXPECTED_CORPUS_SHA" in text
    assert "sha256sum \"$DATA\"" not in text
    assert 'TRITON_CACHE_DIR=/tmp/e97-4b-${SLURM_JOB_ID}-${SLURM_PROCID}' in text
    assert 'rm -rf "$TRITON_CACHE_DIR"' in text
    cfg = json.loads(CONFIG.read_text())
    assert cfg["training"]["gradient_checkpointing"] is False
    assert "--gradient_checkpointing" not in text
    assert "--diloco_outer_optimizer avg" in text
    assert "--no-requeue" in text
    assert "squeue-${SLURM_JOB_ID}-running.txt" in text
    assert "|$EXPECTED_PARTITION|$EXPECTED_QOS|" in text
    train_text = (ROOT / "train.py").read_text()
    assert "last_periodic_checkpoint_step == step" in train_text
    assert "[final-checkpoint] REUSE periodic checkpoint" in train_text


def test_frontier_4b_submitter_is_immutable_attended_and_records_both_queue_fields():
    text = SUBMIT.read_text()
    assert 'CONFIRM_RUNG:-0' in text
    assert 'CONFIRM_BATCH_PROBE:-0' in text
    assert 'probe_b2|probe_b4|probe_b5|probe_b6|probe_b8)' in text
    assert 'NODES=2; QOS=debug; TIME_LIMIT=00:20:00' in text
    assert 'CONFIRM_BOOTSTRAP:-0' in text
    assert 'CONFIRM_PRODUCTION:-0' in text
    assert 'NODES=4; QOS=debug; TIME_LIMIT=00:20:00' in text
    assert 'NODES=256; QOS=debug; TIME_LIMIT=00:20:00' in text
    assert 'NODES=256; QOS=normal; TIME_LIMIT=06:00:00' in text
    assert 'CONFIRM_RESUME:-0' in text
    assert 'SOURCE_SHA=$(git rev-parse HEAD)' in text
    assert 'ORIGIN_MAIN_SHA=$(git rev-parse origin/main)' in text
    assert 'git clone --shared --no-checkout' in text
    assert '-p "$PARTITION" -q "$QOS"' in text
    assert "'%i|%T|%N|%P|%q|%R'" in text
    assert '--dependency="afterany:$payload_id"' in text


def test_frontier_4b_collector_records_terminal_partition_and_qos():
    text = COLLECTOR.read_text()
    assert "JobIDRaw,Partition,QOS,State" in text
    assert "${EXPECTED_PARTITION}\\|${EXPECTED_QOS}" in text
    assert 'checkpoint-${PAYLOAD_JOB_ID}.sha256' in text
    assert 'readlink -f "$latest"' in text
