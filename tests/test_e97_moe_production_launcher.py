import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/frontier/e97_35b_moe_production.sbatch"
SUBMITTER = ROOT / "scripts/frontier/submit_e97_35b_moe_scale.sh"
RUNNER = ROOT / "scripts/frontier/e97_35b_moe_train.py"
EP_TRITON = ROOT / "ndm/triton/e97_moe_ep.py"
E97_MIXER = ROOT / "ndm/models/e88_fla_hybrid.py"
LONG_CONTEXT_LAUNCHER = ROOT / "scripts/frontier/e97_35b_moe_long_context_debug.sbatch"
LONG_CONTEXT_CONFIGS = ROOT / "configs/frontier/e97_moe_long_context"


def test_production_launcher_is_fixed_world_fail_stop_and_canonical():
    text = LAUNCHER.read_text()
    assert "#SBATCH --no-requeue" in text
    assert "--kill-on-bad-exit=1" in text
    assert "Requeue=0" in text
    assert "Partition=$EXPECTED_PARTITION" in text
    assert "QOS=$EXPECTED_QOS" in text
    assert "sbcast -f \"$SEED_CHECKPOINT\" \"$JOB_SEED\"" in text
    assert "multinode production requires positive MAX_STEPS and TRAIN_MINUTES=0" in text
    assert '--max-steps "$MAX_STEPS" --minutes "$TRAIN_MINUTES"' in text
    assert "--save-every \"$SAVE_EVERY\"" in text
    assert "--keep-checkpoints \"$KEEP_CHECKPOINTS\"" in text
    assert 'RESUME_ROOT="$CHECKPOINT_ROOT"' in text
    assert '"schema=${SAMPLER_SCHEMA:-legacy-mutable-rng-v0}"' in text
    assert '--sampler-data-world-size "$SAMPLER_DATA_WORLD_SIZE"' in text
    assert '--sampler-stream-origin-accepted-tokens "$SAMPLER_STREAM_ORIGIN_ACCEPTED_TOKENS"' in text
    assert "counter-v2 requires a nonnegative stream origin" in text
    assert "--sampler-transition-from-legacy" in text
    assert "partial sampler identity or transition without schema" in text
    assert "scontrol requeue" not in text


def test_submitter_verifies_partition_and_qos_separately():
    text = SUBMITTER.read_text()
    assert "'%i|%P|%q|%T|%D|%l'" in text
    assert '--nodes="$NODES"' in text
    assert '--qos="$QOS"' in text
    assert "TRAIN_MINUTES=${TRAIN_MINUTES:-0}" in text
    assert "MAX_STEPS=${MAX_STEPS:-200}" in text
    assert "HEAD == origin/main" in text
    assert "SAMPLER_CORPUS_SHA256" in text
    assert "SAMPLER_DATA_WORLD_SIZE" in text
    assert "SAMPLER_STREAM_ORIGIN_ACCEPTED_TOKENS" in text
    assert "SAMPLER_TRANSITION_FROM_LEGACY" in text
    assert "scancel" in text


def test_long_context_launcher_is_debug_fail_stop_and_immutable():
    text = LONG_CONTEXT_LAUNCHER.read_text()
    assert "#SBATCH -q debug" in text
    assert "#SBATCH --no-requeue" in text
    assert 'SOURCE_COMMIT:?immutable source commit required' in text
    assert 'git archive "$SOURCE_COMMIT"' in text
    assert 'sbcast -f "$SEED_CHECKPOINT" "$JOB_SEED"' in text
    assert '--seed-checkpoint "$JOB_SEED"' in text
    assert 'resume=(--resume-root "$RESUME_ROOT")' in text
    assert 'resume+=(--resume-lr-override "$RESUME_LR_OVERRIDE")' in text
    assert 'sampler+=(--sampler-transition-from-counter)' in text
    assert '--sampler-data-world-size "$WORLD_SIZE"' in text
    assert '--gradient-checkpointing' in text
    assert '--loss-chunk-size "$LOSS_CHUNK_SIZE"' in text
    assert 'LOSS_CHECKPOINT_FLAG=--no-checkpoint-loss-chunks' in text
    assert '--checkpoint-interval "$CHECKPOINT_INTERVAL"' in text
    assert '--projection-chunk-size "$PROJECTION_CHUNK_SIZE"' in text
    assert '--sequence-chunk-size "$SEQUENCE_CHUNK_SIZE"' in text
    assert 'full_bptt=(--full-bptt-segments)' in text
    assert '--checkpoint-group-size "$CHECKPOINT_GROUP_SIZE"' in text
    assert '--moe-token-chunk-size "$MOE_TOKEN_CHUNK_SIZE"' in text
    assert '--empty-cache-interval "$EMPTY_CACHE_INTERVAL"' in text
    assert 'cache_before_backward=(--empty-cache-before-backward)' in text
    assert 'z_offload=(--offload-schedulefree-z)' in text
    assert 'z_offload=(--offload-schedulefree-state)' in text
    assert '--profile-phases' in text
    assert '--kill-on-bad-exit=1' in text


def test_long_context_study_configs_freeze_qualified_execution_not_architecture():
    short = json.loads((LONG_CONTEXT_CONFIGS / "32k.json").read_text())
    long = json.loads((LONG_CONTEXT_CONFIGS / "128k.json").read_text())
    fallback = json.loads(
        (LONG_CONTEXT_CONFIGS / "128k_tbptt_fallback.json").read_text())
    assert short["schema"] == long["schema"] == fallback["schema"] == (
        "emender-e97-moe-long-context-study-v1")
    assert short["context_size"] == short["gradient_horizon"] == 32768
    assert short["sequence_chunk_size"] == 0
    assert long["context_size"] == long["gradient_horizon"] == 131072
    assert long["sequence_chunk_size"] == long["moe_token_chunk_size"] == 32768
    assert long["full_bptt_segments"] is True
    assert fallback["context_size"] == 131072
    assert fallback["sequence_chunk_size"] == fallback["gradient_horizon"] == 32768
    assert fallback["full_bptt_segments"] is False
    for recipe in (short, long, fallback):
        assert recipe["parent_step"] == 2338080
        assert recipe["parent_accepted_tokens"] == 250797359104
        assert recipe["projection_chunk_size"] == recipe["loss_chunk_size"] == 2048
        assert recipe["record_separator_token_id"] == 218
        assert recipe["reset_state_at_record_separator"] is False
        assert not ({"dim", "depth", "n_heads", "n_state"} & set(recipe))


def test_split_edit_projection_recomputation_uses_existing_e97_kernel():
    text = E97_MIXER.read_text()
    assert "(not self.use_split_edit or self.use_triton)" in text
    assert "not self.use_output_norm" in text
    assert "if self.use_triton and self.use_split_edit:" in text
    assert "e97_split_edit_triton_apply(" in text
    assert "erase_gate=erase_gate" in text
    assert "value_write_gate=value_write_gate" in text


def test_ragged_ep_rows_do_not_create_unbounded_triton_specializations():
    text = EP_TRITON.read_text()
    # Received assignment counts vary by rank, layer, and step. Specializing
    # ROWS loaded thousands of HIP modules and eventually produced HIP 209.
    assert "ROWS: tl.constexpr" not in text
    assert "row < ROWS" not in text


def test_runner_uses_restored_step_for_data_and_one_canonical_island():
    text = RUNNER.read_text()
    assert 'parser.add_argument("--max-steps", type=int, default=0)' in text
    assert 'args.max_steps == 0 and args.minutes == 0' in text
    assert "groups.node_count > 1 and args.minutes > 0" in text
    assert "data_seed_base = 42 + starting_step" in text
    assert 'rank_seed=data_seed_base + dist.get_rank()' in text
    assert "sampler_identity=sampler_identity" in text
    assert "accepted_tokens if sampler_identity is not None else None" in text
    assert '"legacy-to-counter"' in text
    assert '"--sampler-transition-from-counter"' in text
    assert '"counter-to-counter"' in text
    assert "allow_counter_sampler_transition=args.sampler_transition_from_counter" in text
    assert "if groups.node_index == 0:" in text
    assert "dist.broadcast(authority, src=0)" in text
    assert 'if dist.get_rank() == 0:' in text
    assert 'args.checkpoint_root / f"node-' not in text
    assert "step % args.save_every == 0" in text


def test_runner_exposes_existing_long_context_memory_controls():
    text = RUNNER.read_text()
    assert '"--gradient-checkpointing", action=argparse.BooleanOptionalAction' in text
    assert 'parser.add_argument("--loss-chunk-size", type=int, default=0)' in text
    assert '"--checkpoint-loss-chunks", action=argparse.BooleanOptionalAction' in text
    assert 'model.checkpoint_loss_chunks = bool(args.checkpoint_loss_chunks)' in text
    assert 'parser.add_argument("--checkpoint-interval", type=int, default=16)' in text
    assert 'parser.add_argument("--projection-chunk-size", type=int, default=0)' in text
    assert 'parser.add_argument("--sequence-chunk-size", type=int, default=0)' in text
    assert 'parser.add_argument("--full-bptt-segments", action="store_true")' in text
    assert "def _segmented_full_bptt_objective(" in text
    assert "smaller chunks change the router balance objective" in text
    assert 'parser.add_argument("--checkpoint-group-size", type=int, default=1)' in text
    assert 'parser.add_argument("--moe-token-chunk-size", type=int, default=0)' in text
    assert 'parser.add_argument("--resume-lr-override", type=float)' in text
    assert 'optimizer_group["lr"] = float(args.resume_lr_override)' in text
    assert "model.gradient_checkpointing = bool(args.gradient_checkpointing)" in text
    assert "model.loss_chunk_size = int(args.loss_chunk_size)" in text
    assert "module.checkpoint_interval = int(args.checkpoint_interval)" in text
    assert "module.projection_chunk_size = int(args.projection_chunk_size)" in text
    assert "def _tbptt_objective_backward(model, chunks: torch.Tensor" in text
    assert "prev_hiddens=previous_hiddens" in text
    assert "_detach_recurrent_hiddens(new_hiddens)" in text
    assert "args.sequence_chunk_size > 0 and not args.full_bptt_segments" in text
    assert 'torch.cuda.reset_peak_memory_stats()' in text
    assert 'forward_max_hbm_allocated=forward_max_hbm_allocated' in text
    assert 'backward_max_hbm_allocated=backward_max_hbm_allocated' in text
