# 128-Node E97 Hierarchical Smoke

Task: `run-128n-e97`
Date: 2026-06-27
Base commit: `2729e9603d99d76f3630b317f7e8a4264ec787a6`

## Result

Accepted as a 128-node hierarchical scale smoke. Slurm job `4908849`
completed `0:0` in `00:10:42`, resumed from the verified step-383500
checkpoint, built the 1024-rank hierarchical process groups, completed 4 K20
hierarchical merges, and wrote a final consensus checkpoint.

Recommendation: unblock the conditional 256-node safety gate, but keep it a
bounded smoke. Preserve the same DiLoCo and RCCL settings (`K=20`,
hierarchical merge, group size 4, 64M bucket, recommended RCCL env, alt
rendezvous, `--network=disable_rdzv_get`). Use a slightly larger bounded
allocation for the 256-node smoke, for example `00:30:00` with
`TRAIN_MINUTES=18` and the same 600 second final-checkpoint margin, because the
128-node job spent about 8.6 minutes in startup/process-group construction and
therefore reached only 4 merges before the walltime finalization path fired.
Do not launch any extended K160/science run until the 256-node smoke also
completes cleanly with multiple merges and a final checkpoint.

## Slurm

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|Start|End
4908849|e97-128n-hier-smoke|COMPLETED|0:0|00:10:42|128|2026-06-27T11:37:38|2026-06-27T11:48:20
4908849.batch|batch|COMPLETED|0:0|00:10:42|1|2026-06-27T11:37:38|2026-06-27T11:48:20
4908849.extern|extern|COMPLETED|0:0|00:10:42|128|2026-06-27T11:37:38|2026-06-27T11:48:20
4908849.0|bash|COMPLETED|0:0|00:10:36|128|2026-06-27T11:37:44|2026-06-27T11:48:20
```

Stdout/stderr:

```text
stdout: logs/frontier/scaleout/e97-128n-hier-smoke-4908849.out
stderr: logs/frontier/scaleout/e97-128n-hier-smoke-4908849.err
```

Run artifacts:

```text
run_root:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z
env:       /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/artifacts/env.txt
manifest:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/artifacts/manifest.json
summary:   /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/summaries/summary.md
train_log: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/logs/train.log
```

## Exact Config

Submitted command:

```bash
env \
  WG_TASK_ID=run-128n-e97 \
  SCALEOUT_VARIANT=E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke \
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
  REQUESTED_NODE_HOURS=42.666667 \
  HUMAN_APPROVAL_RECORD='WG run-128n-e97 128-node E97 step383500 hierarchical K20 smoke, 2026-06-27; bounded scale gate only' \
  sbatch -N 128 -t 00:20:00 -J e97-128n-hier-smoke \
    --network=disable_rdzv_get \
    --export=ALL \
    scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Launcher-recorded command:

```text
srun -N 128 -n 1024 -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc export\ RANK=\$SLURM_PROCID\ WORLD_SIZE=\$SLURM_NTASKS\ LOCAL_RANK=0\;\ exec\ python\ -u\ train.py\ \"\$@\" bash --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --val_data /lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt --bf16 --batch_size 1 --chunk_size 2048 --tokenizer p50k_base --optimizer schedulefree --seed 42 --lr 0.001007 --train_minutes 12 --timer_after_compile_warmup --compile_warmup_steps 1 --log_every 5 --val_every 10000 --save_every 20 --keep_checkpoints 4 --walltime_final_checkpoint_margin_seconds 600 --walltime_check_every 20 --distributed_health_check_every 20 --output /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/train --resume /lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt --diloco --diloco_k 20 --diloco_outer_optimizer avg --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x --diloco_island_size 1 --diloco_merge_topology hierarchical --diloco_merge_group_size 4 --diloco_merge_group_create_barrier_every 8 --diloco_merge_bucket_numel 67108864 --diloco_merge_debug 0 --diloco_merge_debug_ranks 0 --level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32 --n_groups 32 --n_slots 64 --state_expansion 2 --expansion 1.0 --use_gate 1 --use_permutation 1 --gate_activation silu --linear_state 0 --use_write_gate 0 --use_triton 1 --use_chunked_e97 0 --e97_chunk_size 32 --mlp_ratio 2.2623 --mlp_multiple 64 --checkpoint_interval 16 --weight_decay 0.01 --grad_accum 1 --grad_clip 1.0
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

The launcher `rccl_net_plugin_status` probe again reported `not-found`, matching
the 64-node report. The job still completed successfully with the requested
recommended RCCL variables exported.

## Runtime Evidence

Hierarchical setup completed at the 128-node shape:

```text
[DiLoCo-merge] building hierarchical process groups: group_size=4 n_groups=256 create_barrier_every=8
[DiLoCo-merge] hierarchical process groups built; warming communicators
[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, ... 4] roots=[0, 4, ... 1020] (exact weighted SUM/world_size; bucket_numel=67108864)
```

The job resumed and trained:

```text
Starting training from step 383500...
step 383505 | loss 2.7781 | lr 1.01e-03 | grad 1.80 | tok/s 2249 | global_tok/s 2302672 | elapsed_h 0.001 | time 2026-06-27T15:46:14+00:00
```

Merge/checkpoint cadence:

```text
>>> [DiLoCo] merge #1 at step 383520: averaged model weights across 1024 ranks in 6380 ms
>>> [DiLoCo] merge #2 at step 383540: averaged model weights across 1024 ranks in 6629 ms
>>> [DiLoCo] merge #3 at step 383560: averaged model weights across 1024 ranks in 6528 ms
>>> [DiLoCo] merge #4 at step 383580: averaged model weights across 1024 ranks in 6582 ms
>>> [DiLoCo] final merge SKIPPED at step 383580: last step already merged (step % K == 0); checkpoint is already consensus
Training complete! Final step: 383580
FINAL_LOSS_LAST100: 2.5608
DILOCO_MERGES: 4
DILOCO_K: 20
DILOCO_SYNC_TOTAL_S: 26.118
DILOCO_SYNC_AVG_MS: 6529.6
```

Final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/train/emender_E97_1.3B_20260627_114328/checkpoint_step_383580_loss_2.5608.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_128n_hier_g4_bucket64m_smoke/4908849-20260627T153740Z/train/emender_E97_1.3B_20260627_114328/latest.pt -> checkpoint_step_383580_loss_2.5608.pt
```

Retained checkpoints at completion:

```text
checkpoint_step_383540_loss_2.4210.pt
checkpoint_step_383560_loss_2.2824.pt
checkpoint_step_383580_loss_2.5608.pt
checkpoint_step_383580_loss_2.8197.pt
latest.pt -> checkpoint_step_383580_loss_2.5608.pt
```

There are two step-383580 checkpoint files because the normal save-at-merge
checkpoint used the instantaneous logged loss `2.8197`, then the finalization
checkpoint used `FINAL_LOSS_LAST100=2.5608` and updated `latest.pt` to that
final consensus checkpoint.

## Metrics

Rank-0 metric rows parsed from `train.log`:

| Run | Nodes/ranks | Metrics | Final step | Merges | Loss avg | First 4 loss | Last 4 loss | First 10 loss | Last 10 loss | Median global tok/s | Avg global tok/s | Avg merge ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128n hierarchical `4908849` | 128 / 1024 | 16 | 383580 | 4 | 2.5608 | 2.7537 | 2.6190 | 2.5982 | 2.4949 | 2,644,776 | 2,087,777 | 6,529.6 |
| 64n hierarchical `4908602` | 64 / 512 | 48 | 383740 | 12 | 2.5720 | 2.5924* | 2.5713* | 2.5924 | 2.5713 | 1,382,453 | 1,012,810 | 6,436.2 |
| 32n hierarchical `4908477` | 32 / 256 | 64 | 383820 | 16 | 2.5905 | 2.6208* | 2.5010* | 2.6208 | 2.5010 | 651,330 | 495,265 | 8,013.8 |

`*` Prior reports used first/last 10 loss windows for the 64-node and 32-node
rows; those values are repeated where only four-window numbers are not available.

Interpretation:

- 128-node hierarchical worked at 1024 ranks with the conservative group-size-4,
  64M bucket, and group-create barrier cadence from the passing 32-node and
  64-node runs.
- 128-node median global throughput was about `1.91x` the 64-node hierarchical
  median (`2,644,776 / 1,382,453`) and about `4.06x` the 32-node hierarchical
  median (`2,644,776 / 651,330`). Average global throughput was about `2.06x`
  the 64-node average (`2,087,777 / 1,012,810`).
- Average hierarchical merge time was essentially flat versus 64 nodes:
  `6,529.6 ms` at 128 nodes versus `6,436.2 ms` at 64 nodes. This is acceptable
  for the safety gate and does not show a severe merge-time anomaly.
- Loss was finite throughout the 128-node run. The reported
  `FINAL_LOSS_LAST100` was `2.5608`; rank-0 first-10 average was `2.5982` and
  last-10 average was `2.4949`.
- The 20-minute allocation was conservative but tight at 128 nodes. Startup and
  hierarchical setup consumed most of the walltime before rank-0 metrics began;
  final-checkpoint margin triggered after four merges. This is still a valid
  smoke because the job completed multiple K20 merges and final consensus
  checkpointing, but the 256-node smoke should use a bounded longer walltime if
  the goal is to observe more than a handful of merges.

## Gate Decision For 256

Pass for conditional 256-node smoke.

Gate criteria:

- Slurm `COMPLETED 0:0`: yes, job `4908849`.
- Multiple completed K20 hierarchical merges: yes, 4 merges at 1024 ranks.
- Final consensus checkpoint/latest written: yes, `latest.pt` points to
  `checkpoint_step_383580_loss_2.5608.pt`; final merge was skipped only because
  step `383580` had already merged at K20 cadence.
- No NCCL/RCCL watchdogs, segfaults, communicator-construction failures, or
  severe throughput/merge-time anomaly observed in the parsed evidence: yes.
  The log contains expected c10d address-family warnings and Triton/Python
  version warnings, but the job completed cleanly.
- Recommendation to proceed: yes, unblock `conditional-256n-e97`.

Concrete 256 recommendation:

- Launch only the conditional short smoke, not an extended run.
- Use the verified step-383500 checkpoint again unless the 256 task owner has a
  reason to test continuation from the 128 final checkpoint.
- Preserve `DILOCO_K=20`, `DILOCO_MERGE_TOPOLOGY=hierarchical`,
  `DILOCO_MERGE_GROUP_SIZE=4`, `DILOCO_MERGE_BUCKET_NUMEL=67108864`,
  `FRONTIER_RCCL_ENV=recommended`, `FRONTIER_RCCL_ALT_RDZV=1`, and
  `--network=disable_rdzv_get`.
- Use a bounded but slightly longer 256 allocation, for example `00:30:00`,
  `TRAIN_MINUTES=18`, `REQUESTED_SECONDS=1800`, and
  `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=600`, to preserve final-checkpoint
  behavior while allowing startup headroom.
- Pass criteria for 256 should mirror this gate: Slurm `COMPLETED 0:0`,
  topology line printed for 2048 ranks, at least two K20 merges, finite loss
  metrics, no severe watchdog/segfault/communicator failure, and final
  checkpoint/latest written.

## Validation

- 128-node Slurm job ID, run root, stdout/stderr paths, manifest, env, train
  log, and exact config are recorded above.
- Job `4908849` was monitored to terminal `COMPLETED 0:0`.
- Hierarchical worked at 128 nodes: setup completed, 4 K20 merges completed,
  final checkpoint completed, and `latest.pt` points at the final checkpoint.
- Report says pass for 256 with concrete criteria and bounded settings.
- No 256-node run was launched by this task.
