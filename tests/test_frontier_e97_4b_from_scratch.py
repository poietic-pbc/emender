import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/frontier/e97_4b_from_scratch.json"
MATCHED_CLOCK_CONFIG = ROOT / "configs/frontier/e97_4b_matched_clock_32n.json"
SEED_IMPORT_CONFIG = ROOT / "configs/frontier/e97_4b_seed_import_32n.json"
SEED_CONTINUATION_CONFIG = ROOT / "configs/frontier/e97_4b_seed_continuation_32n.json"
SEED_SCALE_CONFIG = ROOT / "configs/frontier/e97_4b_seed_scale_64n.json"
HYBRID_DDP_CONFIG = ROOT / "configs/frontier/e97_4b_hybrid_ddp_8n.json"
HYBRID_96N_CONFIG = ROOT / "configs/frontier/e97_4b_hybrid_ddp_96n_campaign.json"
HYBRID_256N_CONFIG = ROOT / "configs/frontier/e97_4b_hybrid_ddp_256n_campaign.json"
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
    assert cfg["debug_continuation_walltime"] == "02:00:00"
    assert cfg["debug_continuation_train_minutes"] == 105
    assert cfg["bootstrap_smoke_walltime"] == "00:30:00"
    assert cfg["bootstrap_smoke_steps"] == 50
    train = cfg["training"]
    assert train["batch_size_per_rank"] == 5
    assert train["global_tokens_per_step"] == 256 * 8 * 5 * 2048
    # The attended B5 decision accepts 125 rather than 128 local samples per
    # merge to retain nearly all B6 throughput with more live-memory margin.
    assert train["batch_size_per_rank"] * train["diloco_k"] == 125
    assert train["save_every"] == 2 * train["diloco_k"]
    assert cfg["target_steps"] * train["global_tokens_per_step"] >= cfg["target_tokens"]
    assert (cfg["target_steps"] - 1) * train["global_tokens_per_step"] < cfg["target_tokens"]
    assert cfg["target_tokens"] == 100_000_000_000


def test_frontier_4b_matched_clock_config_matches_lambda_aggregate_clock():
    cfg = json.loads(MATCHED_CLOCK_CONFIG.read_text())
    train = cfg["training"]
    assert cfg["schema"] == "emender-e97-4b-frontier-matched-clock-32n-v1"
    assert cfg["production_nodes"] == 32
    assert cfg["production_tasks_per_node"] == 8
    assert train["batch_size_per_rank"] == 1
    assert train["global_tokens_per_step"] == 8 * 32 * 1 * 2048 == 8 * 32 * 2048
    assert train["diloco_k"] == 32
    assert train["global_tokens_per_step"] * train["diloco_k"] == 16_777_216
    assert train["save_every"] == 256
    assert cfg["target_steps"] == 2048
    assert cfg["target_tokens"] == cfg["target_steps"] * train["global_tokens_per_step"]
    assert cfg["claims"]["qualification_only"] is True


def test_frontier_4b_seed_import_config_binds_checkpoint_and_phase_clock():
    cfg = json.loads(SEED_IMPORT_CONFIG.read_text())
    train = cfg["training"]
    seed = cfg["seed"]
    assert seed["sha256"] == "81fcc932e93df59a478e43b31afc5f0b310f58b8a5deab91a73e5be1a4925ed9"
    assert seed["source_step"] == 12800
    assert seed["source_total_tokens"] == 6_710_886_400
    assert cfg["data"]["sampler_schema"] == "emender-byte-window-counter-v2"
    assert cfg["data"]["sampler_stream_origin_accepted_tokens"] == seed["source_total_tokens"]
    assert cfg["target_steps"] == seed["source_step"] + 512
    assert cfg["target_tokens"] == seed["source_total_tokens"] + 512 * train["global_tokens_per_step"]
    assert train["batch_size_per_rank"] == 1
    assert train["diloco_k"] == 32
    assert train["save_every"] == 256


def test_frontier_4b_seed_continuation_preserves_world_and_adds_2048_updates():
    cfg = json.loads(SEED_CONTINUATION_CONFIG.read_text())
    train = cfg["training"]
    parent = cfg["parent"]
    assert cfg["production_nodes"] == 32
    assert parent["step"] == 13312
    assert parent["sha256"] == "fa7b53f8ea31ca177aac0bba6b1fd174a970d8d8db68314c96333efd80a50ade"
    assert cfg["target_steps"] == parent["step"] + 2048 == 15360
    assert cfg["target_tokens"] == parent["total_tokens"] + 2048 * train["global_tokens_per_step"]
    assert train["batch_size_per_rank"] == 1
    assert train["diloco_k"] == 32
    assert cfg["claims"]["same_world_exact_continuation_from_parent"] is True


def test_frontier_4b_seed_scale_64n_is_equal_token_cost_comparison():
    cfg = json.loads(SEED_SCALE_CONFIG.read_text())
    train = cfg["training"]
    parent = cfg["parent"]
    assert cfg["production_nodes"] == 64
    assert parent["step"] == 15360
    assert parent["sha256"] == "9da49a274934135de4b5ac4f9265c0e6eff5ec7672fb7d9c3fb6388b0da18f16"
    assert train["global_tokens_per_step"] == 512 * 2048
    assert cfg["target_steps"] == parent["step"] + 1024
    assert cfg["target_tokens"] == parent["total_tokens"] + 1024 * train["global_tokens_per_step"]
    assert cfg["data"]["sampler_stream_origin_accepted_tokens"] == parent["total_tokens"]
    assert train["diloco_k"] == 32
    assert cfg["claims"]["equal_new_tokens_vs_32n_parent_block"] is True


def test_frontier_4b_hybrid_ddp_matches_lambda_optimizer_topology():
    cfg = json.loads(HYBRID_DDP_CONFIG.read_text())
    train = cfg["training"]
    parent = cfg["parent"]
    assert cfg["production_nodes"] == 8
    assert train["batch_size_per_rank"] == 4
    assert train["diloco_island_size"] == 8
    assert (cfg["production_nodes"] * 8) // train["diloco_island_size"] == 8
    assert train["batch_size_per_rank"] * train["diloco_island_size"] == 32
    assert train["global_tokens_per_step"] == 8 * 32 * 2048 == 524288
    assert train["diloco_k"] == 32
    assert cfg["target_steps"] == parent["step"] + 768
    assert cfg["target_tokens"] == parent["total_tokens"] + 768 * train["global_tokens_per_step"]
    assert cfg["claims"]["lambda_optimizer_topology_match"] is True


def test_frontier_4b_96n_campaign_reaches_approximately_100b():
    cfg = json.loads(HYBRID_96N_CONFIG.read_text())
    train = cfg["training"]
    campaign = cfg["campaign"]
    assert cfg["production_nodes"] == 96
    assert cfg["production_walltime"] == "06:00:00"
    assert train["diloco_island_size"] == 8
    assert train["batch_size_per_rank"] == 4
    assert train["global_tokens_per_step"] == 96 * 8 * 4 * 2048
    assert campaign["debug_target_step"] == 18176
    assert campaign["production_target_steps"] == [21504, 24832, 28160, 31488]
    assert campaign["short_production_walltime"] == "02:00:00"
    assert campaign["short_debug_qos_operator_override"] is True
    assert campaign["short_production_target_steps"] == list(range(19200, 31489, 1024))
    assert campaign["short_production_new_tokens_per_phase"] == 1024 * train["global_tokens_per_step"]
    assert campaign["debug_new_tokens"] == 1024 * train["global_tokens_per_step"]
    assert campaign["production_new_tokens_per_phase"] == 3328 * train["global_tokens_per_step"]
    assert campaign["final_total_tokens"] == cfg["target_tokens"] == 99_723_771_904
    assert cfg["target_steps"] == 31488
    assert campaign["autonomous_continuation_authorized"] is True


def test_frontier_4b_256n_config_satisfies_payload_loader_schema():
    cfg = json.loads(HYBRID_256N_CONFIG.read_text())
    # Keep this synchronized with the unconditional extraction in the sbatch
    # payload; missing legacy mode fields must fail in pytest, not allocation.
    assert isinstance(cfg["bootstrap_smoke_steps"], int)
    assert isinstance(cfg["production_train_minutes"], int)
    assert isinstance(cfg["debug_continuation_train_minutes"], int)
    assert cfg["bootstrap_smoke_steps"] > 0
    assert cfg["production_train_minutes"] == 0
    assert cfg["debug_continuation_train_minutes"] == 0


def test_frontier_4b_256n_campaign_reaches_approximately_100b():
    cfg = json.loads(HYBRID_256N_CONFIG.read_text())
    train = cfg["training"]
    campaign = cfg["campaign"]
    assert cfg["production_nodes"] == 256
    assert cfg["production_qos"] == "debug"
    assert cfg["production_walltime"] == "02:00:00"
    assert train["diloco_island_size"] == 8
    assert train["batch_size_per_rank"] == 4
    assert train["global_tokens_per_step"] == 256 * 8 * 4 * 2048
    assert campaign["target_steps"] == [20992, 21760, 22528, 23296, 24064, 24448]
    expected_tokens = [
        cfg["seed"]["source_total_tokens"]
        + (step - cfg["seed"]["source_step"]) * train["global_tokens_per_step"]
        for step in campaign["target_steps"]
    ]
    assert campaign["target_total_tokens"] == expected_tokens
    assert expected_tokens[-1] == cfg["target_tokens"] == 99_723_771_904
    assert campaign["full_phase_updates"] == 768
    assert campaign["final_phase_updates"] == 384
    assert campaign["autonomous_continuation_authorized"] is True
    assert campaign["final_normal_qos_operator_override"] is True
    assert campaign["final_normal_walltime"] == "08:00:00"
    assert campaign["final_normal_target_step"] == 24448


def test_frontier_4b_payload_is_fixed_world_counter_sampled_and_fail_stop():
    text = PAYLOAD.read_text()
    assert "source scripts/frontier/activate_emender_frontier.sh" in text
    assert '--sampler_schema "$SAMPLER_SCHEMA"' in text
    assert "--sampler_data_world_size \"$EXPECTED_WORLD_SIZE\"" in text
    assert "--kill-on-bad-exit=1" in text
    assert "--offload_schedulefree_state" not in text
    assert "rung authority is fixed at 4 nodes / 32 ranks" in text
    assert "batch probes are fixed at 2 nodes / 16 ranks" in text
    assert 'BATCH_SIZE=${RUN_MODE#probe_b}' in text
    assert 'DILOCO_K=$((128 / BATCH_SIZE))' in text
    assert 'SAVE_EVERY=$((2 * DILOCO_K))' in text
    assert "seed-import canary is fixed at 32 nodes / 256 ranks" in text
    assert "seed continuation is fixed at 32 nodes / 256 ranks" in text
    assert "seed continuation config must be B1/K32/save256/counter-v2/step15360" in text
    assert "seed-import canary requires an exact 512-step counter-v2 phase" in text
    assert "--sampler_transition_from_counter" in text
    assert "seed-scale canary is fixed at 64 nodes / 512 ranks" in text
    assert "seed-scale canary requires B1/K32 and an exact 1024-step counter-v2 phase" in text
    assert "hybrid-DDP canary is fixed at 8 nodes / 64 ranks" in text
    assert "hybrid-DDP canary requires 8 islands / B32 effective / K32 / 768 steps" in text
    assert '--diloco_island_size "$DILOCO_ISLAND_SIZE"' in text
    assert "96-node campaign debug is fixed at 96 nodes / 768 ranks" in text
    assert "96-node production is fixed at 96 nodes / 768 ranks" in text
    assert "invalid 96-node production campaign target" in text
    assert "invalid 96-node short-production campaign target" in text
    assert "256-node campaign is fixed at 256 nodes / 2048 ranks" in text
    assert "invalid 256-node campaign target" in text
    assert "256-node final target must be 24448" in text
    assert '"$RUN_MODE" == hybrid_ddp_256n_final' in text
    assert '( "$RUN_MODE" == hybrid_ddp_256n_debug && "$EXPECTED_TARGET_STEPS" == 20992 )' in text
    assert "matched-clock qualification is fixed at 32 nodes / 256 ranks" in text
    assert "matched-clock config must be B1/K32/save256/2048 steps" in text
    assert "bootstrap authority is fixed at 256 nodes / 2048 ranks" in text
    assert "production authority is fixed at 256 nodes / 2048 ranks" in text
    assert "debug continuation authority is fixed at 256 nodes / 2048 ranks" in text
    assert 'identity_source="$RUN_DIR/identity/source.sha256"' in text
    assert "EXPECTED_CORPUS_SHA" in text
    assert "sha256sum \"$DATA\"" not in text
    assert 'TRITON_CACHE_DIR=/tmp/e97-4b-${SLURM_JOB_ID}-${SLURM_PROCID}' in text
    assert 'rm -rf "$TRITON_CACHE_DIR"' in text
    assert '--walltime_check_every 8' in text
    assert 'requesting graceful final checkpoint' in text
    assert '.final_checkpoint_request' in text
    assert 'kill -TERM -- "-$CHILD_PID"' not in text
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
    assert 'CONFIRM_SEED_IMPORT:-0' in text
    assert 'CONFIRM_SEED_CONTINUATION:-0' in text
    assert 'configs/frontier/e97_4b_seed_continuation_32n.json' in text
    assert 'configs/frontier/e97_4b_seed_import_32n.json' in text
    assert 'imported seed SHA-256 mismatch' in text
    assert 'CONFIRM_SEED_SCALE:-0' in text
    assert 'configs/frontier/e97_4b_seed_scale_64n.json' in text
    assert 'CONFIRM_HYBRID_DDP:-0' in text
    assert 'configs/frontier/e97_4b_hybrid_ddp_8n.json' in text
    assert 'CONFIRM_96N_CAMPAIGN:-0' in text
    assert 'CAMPAIGN_PHASE=1..4' in text
    assert 'CAMPAIGN_PHASE=1..13' in text
    assert 'CONFIRM_DEBUG_QOS_CAMPAIGN:-0' in text
    assert 'QOS=debug' in text
    assert 'TIME_LIMIT=02:00:00' in text
    assert 'EXPECTED_TARGET_STEPS=$((18176 + CAMPAIGN_PHASE * 1024))' in text
    assert 'configs/frontier/e97_4b_hybrid_ddp_96n_campaign.json' in text
    assert 'EXPECTED_TARGET_STEPS=18176' in text
    assert 'EXPECTED_TARGET_STEPS=21504' in text
    assert 'EXPECTED_TARGET_STEPS=31488' in text
    assert 'CONFIRM_256N_CAMPAIGN:-0' in text
    assert 'CONFIRM_256N_FINAL:-0' in text
    assert 'QOS=normal; TIME_LIMIT=08:00:00' in text
    assert 'CAMPAIGN_PHASE=1..6' in text
    assert 'configs/frontier/e97_4b_hybrid_ddp_256n_campaign.json' in text
    assert 'EXPECTED_TARGET_STEPS=$((20224 + CAMPAIGN_PHASE * 768))' in text
    assert 'EXPECTED_TARGET_STEPS=24448' in text
    assert 'CONFIRM_MATCHED_CLOCK:-0' in text
    assert 'NODES=32; QOS=debug; TIME_LIMIT=02:00:00' in text
    assert 'configs/frontier/e97_4b_matched_clock_32n.json' in text
    assert 'CONFIRM_BOOTSTRAP:-0' in text
    assert 'CONFIRM_PRODUCTION:-0' in text
    assert 'CONFIRM_DEBUG_CONTINUATION:-0' in text
    assert 'NODES=4; QOS=debug; TIME_LIMIT=00:20:00' in text
    assert 'NODES=256; QOS=debug; TIME_LIMIT=00:30:00' in text
    assert 'e97-4b-fresh-w2048-r3' in text
    assert 'NODES=256; QOS=debug; TIME_LIMIT=02:00:00; TRAIN_MINUTES=105' in text
    assert 'NODES=256; QOS=normal; TIME_LIMIT=04:00:00; TRAIN_MINUTES=225' in text
    assert 'NODES=256; QOS=normal; TIME_LIMIT=06:00:00; TRAIN_MINUTES=345' in text
    assert 'NODES=256; QOS=normal; TIME_LIMIT=08:00:00; TRAIN_MINUTES=465' in text
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
    assert 'checkpoint-${PAYLOAD_JOB_ID}.reload.json' in text
    assert "mmap=True" in text
    assert "checkpoint token clock mismatch" in text
    assert "checkpoint step disagrees with phase-relative sampler clock" in text
    assert "missing or invalid sampler transition provenance" in text
    assert 'CONFIG=${CONFIG:-$REPO/configs/frontier/e97_4b_from_scratch.json}' in text
