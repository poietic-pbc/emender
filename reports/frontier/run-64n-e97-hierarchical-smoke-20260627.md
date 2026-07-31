# 64-Node E97 Hierarchical Smoke

Task: `run-64n-e97`
Date: 2026-06-27
Base commit: `6fed7a9c182f9459764b1192c3470ce887448003`

## Result

Accepted as a 64-node hierarchical scale smoke. Slurm job `4908602`
completed `0:0` in `00:10:15`, resumed from the verified step-383500
checkpoint, built the 512-rank hierarchical process groups, ran 12 K20
hierarchical merges, and wrote a final consensus checkpoint.

Recommendation: run the next 128-node hierarchical safety gate with the same
conservative topology settings before considering 256 nodes. Do not fall back
to bucketed global for the next step on the basis of this smoke; hierarchical
worked at 64 nodes and its average merge time was lower than the passing
32-node hierarchical job. The 128/256 tasks were not unpaused by this task.

## Slurm

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|Start|End
4908602|e97-64n-hier-smoke|COMPLETED|0:0|00:10:15|64|2026-06-27T09:59:18|2026-06-27T10:09:33
4908602.batch|batch|COMPLETED|0:0|00:10:15|1|2026-06-27T09:59:18|2026-06-27T10:09:33
4908602.extern|extern|COMPLETED|0:0|00:10:15|64|2026-06-27T09:59:18|2026-06-27T10:09:33
4908602.0|bash|COMPLETED|0:0|00:10:05|64|2026-06-27T09:59:27|2026-06-27T10:09:32
```

Stdout/stderr:

```text
stdout: logs/frontier/scaleout/e97-64n-hier-smoke-4908602.out
stderr: logs/frontier/scaleout/e97-64n-hier-smoke-4908602.err
```

Run artifacts:

```text
run_root:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z
env:       /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/artifacts/env.txt
manifest:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/artifacts/manifest.json
summary:   /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/summaries/summary.md
train_log: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/logs/train.log
```

## Exact Config

Submitted command:

```bash
env \
  WG_TASK_ID=run-64n-e97 \
  SCALEOUT_VARIANT=E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke \
  RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt \
  TRAIN_MINUTES=12 \
  DILOCO_K=20 \
  DILOCO_ISLAND_SIZE=1 \
  DILOCO_OUTER_OPTIMIZER=avg \
  DILOCO_OUTER_LR=1.0 \
  DILOCO_OUTER_BETA=0.0 \
  DILOCO_EXPORT_BASIS=x \
  DILOCO_MERGE_TOPOLOGY=hierarchical \
  DILOCO_MERGE_GROUP_SIZE=4 \
  DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY=8 \
  DILOCO_MERGE_BUCKET_NUMEL=67108864 \
  DILOCO_MERGE_DEBUG=0 \
  DILOCO_MERGE_DEBUG_RANKS=0 \
  FRONTIER_RCCL_ENV=recommended \
  FRONTIER_RCCL_ALT_RDZV=1 \
  BATCH_SIZE=1 \
  CHUNK_SIZE=2048 \
  SAVE_EVERY=20 \
  KEEP_CHECKPOINTS=4 \
  LOG_EVERY=5 \
  VAL_EVERY=10000 \
  WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=600 \
  WALLTIME_CHECK_EVERY=20 \
  DISTRIBUTED_HEALTH_CHECK_EVERY=20 \
  REQUESTED_SECONDS=1200 \
  REQUESTED_WALLTIME=00:20:00 \
  REQUESTED_NODE_HOURS=21.333333 \
  HUMAN_APPROVAL_RECORD='WG run-64n-e97 64-node E97 step383500 hierarchical K20 smoke, 2026-06-27; bounded scale gate only' \
  sbatch -N 64 -t 00:20:00 -J e97-64n-hier-smoke \
    --network=disable_rdzv_get \
    --export=ALL \
    scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Launcher-recorded command:

```text
srun -N 64 -n 512 -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc export\ RANK=\$SLURM_PROCID\ WORLD_SIZE=\$SLURM_NTASKS\ LOCAL_RANK=0\;\ exec\ python\ -u\ train.py\ \"\$@\" bash --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --val_data /lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt --bf16 --batch_size 1 --chunk_size 2048 --tokenizer p50k_base --optimizer schedulefree --seed 42 --lr 0.001007 --train_minutes 12 --timer_after_compile_warmup --compile_warmup_steps 1 --log_every 5 --val_every 10000 --save_every 20 --keep_checkpoints 4 --walltime_final_checkpoint_margin_seconds 600 --walltime_check_every 20 --distributed_health_check_every 20 --output /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/train --resume /lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt --diloco --diloco_k 20 --diloco_outer_optimizer avg --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x --diloco_island_size 1 --diloco_merge_topology hierarchical --diloco_merge_group_size 4 --diloco_merge_group_create_barrier_every 8 --diloco_merge_bucket_numel 67108864 --diloco_merge_debug 0 --diloco_merge_debug_ranks 0 --level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32 --n_groups 32 --n_slots 64 --state_expansion 2 --expansion 1.0 --use_gate 1 --use_permutation 1 --gate_activation silu --linear_state 0 --use_write_gate 0 --use_triton 1 --use_chunked_e97 0 --e97_chunk_size 32 --mlp_ratio 2.2623 --mlp_multiple 64 --checkpoint_interval 16 --weight_decay 0.01 --grad_accum 1 --grad_clip 1.0
```

RCCL/rendezvous environment confirmed in `env.txt`:

```text
FRONTIER_RCCL_ENV=recommended
FRONTIER_RCCL_ALT_RDZV=1
FI_CXI_RDZV_PROTO=alt_read
FI_MR_CACHE_MONITOR=kdreg2
FI_CXI_DEFAULT_CQ_SIZE=131072
FI_CXI_DEFAULT_TX_SIZE=2048
FI_CXI_RX_MATCH_MODE=hybrid
NCCL_NET_GDR_LEVEL=3
NCCL_CROSS_NIC=1
NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
NCCL_NET_PLUGIN=librccl-net.so
SLURM_NETWORK=disable_rdzv_get
MPICH_GPU_SUPPORT_ENABLED=1
DILOCO_ISLAND_SIZE=1
```

The launcher's `rccl_net_plugin_status` probe reported `not-found`, matching
the environment path visibility in this shell, but the job still completed
successfully with the requested recommended RCCL variables exported.

## Runtime Evidence

Hierarchical setup completed at the 64-node shape:

```text
[DiLoCo-merge] building hierarchical process groups: group_size=4 n_groups=128 create_barrier_every=8
[DiLoCo-merge] hierarchical process groups built; warming communicators
[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, ... 4] roots=[0, 4, ... 508] (exact weighted SUM/world_size; bucket_numel=67108864)
```

The job resumed and trained:

```text
Starting training from step 383500...
step 383505 | loss 2.7781 | lr 1.01e-03 | grad 1.80 | tok/s 2493 | global_tok/s 1276161 | elapsed_h 0.001 | time 2026-06-27T14:03:35+00:00
```

Merge/checkpoint cadence:

```text
>>> [DiLoCo] merge #1 at step 383520: averaged model weights across 512 ranks in 6397 ms
>>> [DiLoCo] merge #2 at step 383540: averaged model weights across 512 ranks in 6749 ms
...
>>> [DiLoCo] merge #12 at step 383740: averaged model weights across 512 ranks in 6402 ms
>>> [DiLoCo] final merge SKIPPED at step 383740: last step already merged (step % K == 0); checkpoint is already consensus
Training complete! Final step: 383740
FINAL_LOSS_LAST100: 2.5720
DILOCO_MERGES: 12
DILOCO_K: 20
DILOCO_SYNC_TOTAL_S: 77.234
DILOCO_SYNC_AVG_MS: 6436.2
```

Final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/train/emender_E97_1.3B_20260627_100213/checkpoint_step_383740_loss_2.5720.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_64n_hier_g4_bucket64m_smoke/4908602-20260627T135923Z/train/emender_E97_1.3B_20260627_100213/latest.pt -> checkpoint_step_383740_loss_2.5720.pt
```

Retained checkpoints at completion:

```text
checkpoint_step_383700_loss_2.4101.pt
checkpoint_step_383720_loss_2.8948.pt
checkpoint_step_383740_loss_2.5720.pt
checkpoint_step_383740_loss_2.8796.pt
latest.pt -> checkpoint_step_383740_loss_2.5720.pt
```

There are two step-383740 checkpoint files because the normal save-at-merge
checkpoint used the instantaneous logged loss `2.8796`, then the finalization
checkpoint used `FINAL_LOSS_LAST100=2.5720` and updated `latest.pt` to that
final consensus checkpoint.

## Metrics

Rank-0 metric rows parsed from `train.log`:

| Run | Nodes/ranks | Metrics | Final step | Merges | Loss avg | First 10 loss | Last 10 loss | Last 20 loss | Median global tok/s | Avg global tok/s | Avg merge ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64n hierarchical `4908602` | 64 / 512 | 48 | 383740 | 12 | 2.5720 | 2.5924 | 2.5713 | 2.6576 | 1,382,453 | 1,012,810 | 6,436.2 |
| 32n hierarchical `4908477` | 32 / 256 | 64 | 383820 | 16 | 2.5905 | 2.6208 | 2.5010 | 2.5395 | 651,330 | 495,265 | 8,013.8 |
| 32n bucketed global `4908457` | 32 / 256 | 72 | 383860 | 18 | 2.5894 | 2.6765 | 2.5292 | 2.5267 | 582,546 | 538,775 | 2,861.5 |

Interpretation:

- 64-node hierarchical worked at 512 ranks with the conservative group-size-4,
  64M bucket, and group-create barrier cadence from the 32-node fix.
- 64-node median global throughput was about `2.12x` the 32-node hierarchical
  median (`1,382,453 / 651,330`), and average global throughput was about
  `2.04x` (`1,012,810 / 495,265`).
- Hierarchical merge time improved from `8,013.8 ms` at 32 nodes to `6,436.2 ms`
  at 64 nodes for this short smoke. It remains slower than 32-node bucketed
  global (`2,861.5 ms`) but is acceptable for the safety gate because the goal
  was hierarchical viability and final checkpoint behavior, not merge-speed
  optimization.
- Loss was finite throughout the 64-node run. The reported `FINAL_LOSS_LAST100`
  was `2.5720`, slightly better than both 32-node references in this short
  step-383500 continuation window.

## Validation

- 64-node Slurm job ID, run root, stdout/stderr paths, manifest, env, train log,
  and exact config are recorded above.
- Job `4908602` was monitored to terminal `COMPLETED 0:0`.
- Hierarchical worked at 64 nodes: setup completed, 12 K20 merges completed,
  final checkpoint completed, and `latest.pt` points at the final checkpoint.
- Compared against 32-node hierarchical job `4908477` and 32-node bucketed
  global job `4908457`.
- `run-e97-1p3b-k160-128n` and `run-e97-1p3b-k160-256n` were still listed as
  `[PAUSED]` after this report was written; this task did not resume them.
