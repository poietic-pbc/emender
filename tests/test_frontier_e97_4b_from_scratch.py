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
    assert cfg["production_nodes"] == 32
    assert cfg["production_tasks_per_node"] == 8
    train = cfg["training"]
    assert train["batch_size_per_rank"] == 4
    assert train["global_tokens_per_step"] == 32 * 8 * 4 * 2048
    assert train["batch_size_per_rank"] * train["diloco_k"] == 32 * 32
    assert train["save_every"] % train["diloco_k"] == 0
    assert cfg["target_steps"] * train["global_tokens_per_step"] >= cfg["target_tokens"]
    assert cfg["target_tokens"] == 20 * cfg["target_parameters"]


def test_frontier_4b_payload_is_fixed_world_counter_sampled_and_fail_stop():
    text = PAYLOAD.read_text()
    assert "source scripts/frontier/activate_emender_frontier.sh" in text
    assert "--sampler_schema emender-byte-window-counter-v1" in text
    assert "--sampler_data_world_size \"$EXPECTED_WORLD_SIZE\"" in text
    assert "--kill-on-bad-exit=1" in text
    assert "--offload_schedulefree_state" not in text
    assert "--gradient_checkpoint_group_size 2" in text
    assert "--diloco_outer_optimizer avg" in text
    assert "--no-requeue" in text
    assert "squeue-${SLURM_JOB_ID}-running.txt" in text
    assert "|$EXPECTED_PARTITION|$EXPECTED_QOS|" in text


def test_frontier_4b_submitter_is_immutable_attended_and_records_both_queue_fields():
    text = SUBMIT.read_text()
    assert 'CONFIRM_PRODUCTION:-0' in text
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
