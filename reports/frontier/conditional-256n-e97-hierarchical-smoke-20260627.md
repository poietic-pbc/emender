# Conditional 256-Node E97 Hierarchical Smoke

Task: `conditional-256n-e97`
Date: 2026-06-27
Base commit: `4ac5c628a773aca6327995e92d2841a3d1e6e969`

## Result

Accepted as a 256-node hierarchical DiLoCo scale smoke. Slurm job `4908935`
completed `0:0` in `00:20:53`, resumed from the verified step-383500
checkpoint, built the 2048-rank hierarchical process groups, completed 7 K20
hierarchical merges, and wrote a final consensus checkpoint.

Recommendation: do not launch an extended K160/science run directly from this
task. The 256-node safety gate passed, but startup still consumed most of the
bounded allocation and the 600 second final-checkpoint margin stopped the run
after 7 merges. The next larger/longer run should remain paused until a
separate task explicitly scopes it. A reasonable next validation is either a
bounded 256-node longer K20 run to collect more steady-state merge/throughput
samples, or a carefully bounded K160 run with the same conservative
GPU-island/no-DDP, hierarchical, group-size-4, 64M bucket, recommended RCCL,
alt rendezvous, and `--network=disable_rdzv_get` settings.

## Gate From 128 Nodes

The predecessor report `reports/frontier/run-128n-e97-hierarchical-smoke-20260627.md`
was read before submission. Its gate decision was:

```text
Pass for conditional 256-node smoke.
```

It also recommended:

```text
Launch only the conditional short smoke, not an extended run.
```

The 128-node report satisfied the task gate criteria: Slurm job `4908849`
completed `0:0`, completed 4 K20 hierarchical merges at 1024 ranks, wrote
`latest.pt` to the final consensus checkpoint, reported no NCCL/RCCL
watchdogs, segfaults, communicator-construction failures, or severe
throughput/merge-time anomaly, and explicitly recommended unblocking
`conditional-256n-e97`.

## Slurm

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|Start|End
4908935|e97-256n-hier-smoke|COMPLETED|0:0|00:20:53|256|2026-06-27T12:14:05|2026-06-27T12:34:58
4908935.batch|batch|COMPLETED|0:0|00:20:53|1|2026-06-27T12:14:05|2026-06-27T12:34:58
4908935.extern|extern|COMPLETED|0:0|00:20:53|256|2026-06-27T12:14:05|2026-06-27T12:34:58
4908935.0|bash|COMPLETED|0:0|00:20:45|256|2026-06-27T12:14:13|2026-06-27T12:34:58
```

Stdout/stderr:

```text
stdout: logs/frontier/scaleout/e97-256n-hier-smoke-4908935.out
stderr: logs/frontier/scaleout/e97-256n-hier-smoke-4908935.err
```

Run artifacts:

```text
run_root:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z
env:       /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/artifacts/env.txt
manifest:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/artifacts/manifest.json
summary:   /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/summaries/summary.md
train_log: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/logs/train.log
```

## Exact Config

Submitted command:

```bash
env \
  WG_TASK_ID=conditional-256n-e97 \
  SCALEOUT_VARIANT=E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke \
  RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt \
  TRAIN_MINUTES=18 \
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
  REQUESTED_SECONDS=1800 \
  REQUESTED_WALLTIME=00:30:00 \
  REQUESTED_NODE_HOURS=128.000000 \
  HUMAN_APPROVAL_RECORD='WG conditional-256n-e97 256-node E97 step383500 hierarchical K20 smoke, 2026-06-27; bounded scale gate only after passing 128-node report' \
  sbatch -N 256 -t 00:30:00 -J e97-256n-hier-smoke \
    --network=disable_rdzv_get \
    --export=ALL \
    scripts/frontier/e97_1p3b_pretrained_k160_scale_ladder.sbatch
```

Launcher-recorded command:

```text
srun -N 256 -n 2048 -c7 --gpus-per-task=1 --gpu-bind=closest bash -lc export\ RANK=\$SLURM_PROCID\ WORLD_SIZE=\$SLURM_NTASKS\ LOCAL_RANK=0\;\ exec\ python\ -u\ train.py\ \"\$@\" bash --data /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt --val_data /lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt --bf16 --batch_size 1 --chunk_size 2048 --tokenizer p50k_base --optimizer schedulefree --seed 42 --lr 0.001007 --train_minutes 18 --timer_after_compile_warmup --compile_warmup_steps 1 --log_every 5 --val_every 10000 --save_every 20 --keep_checkpoints 4 --walltime_final_checkpoint_margin_seconds 600 --walltime_check_every 20 --distributed_health_check_every 20 --output /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/train --resume /lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt --diloco --diloco_k 20 --diloco_outer_optimizer avg --diloco_outer_lr 1.0 --diloco_outer_beta 0.0 --diloco_export_basis x --diloco_island_size 1 --diloco_merge_topology hierarchical --diloco_merge_group_size 4 --diloco_merge_group_create_barrier_every 8 --diloco_merge_bucket_numel 67108864 --diloco_merge_debug 0 --diloco_merge_debug_ranks 0 --level E97 --params 100m --dim 1792 --depth 11 --n_heads 216 --n_state 32 --n_groups 32 --n_slots 64 --state_expansion 2 --expansion 1.0 --use_gate 1 --use_permutation 1 --gate_activation silu --linear_state 0 --use_write_gate 0 --use_triton 1 --use_chunked_e97 0 --e97_chunk_size 32 --mlp_ratio 2.2623 --mlp_multiple 64 --checkpoint_interval 16 --weight_decay 0.01 --grad_accum 1 --grad_clip 1.0
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
the passing 64-node and 128-node reports. The job still completed cleanly with
the requested recommended RCCL variables exported.

## Runtime Evidence

Hierarchical setup completed at the 256-node shape:

```text
[DiLoCo-merge] building hierarchical process groups: group_size=4 n_groups=512 create_barrier_every=8
[DiLoCo-merge] hierarchical process groups built; warming communicators
[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, ... 4] roots=[0, 4, ... 2044] (exact weighted SUM/world_size; bucket_numel=67108864)
```

The job resumed and trained:

```text
Starting training from step 383500...
step 383505 | loss 2.9239 | lr 1.01e-03 | grad 1.77 | tok/s 1296 | global_tok/s 2653986 | elapsed_h 0.002 | time 2026-06-27T16:30:48+00:00
```

Merge/checkpoint cadence:

```text
>>> [DiLoCo] merge #1 at step 383520: averaged model weights across 2048 ranks in 8851 ms
>>> [DiLoCo] merge #2 at step 383540: averaged model weights across 2048 ranks in 8822 ms
>>> [DiLoCo] merge #3 at step 383560: averaged model weights across 2048 ranks in 8853 ms
>>> [DiLoCo] merge #4 at step 383580: averaged model weights across 2048 ranks in 8912 ms
>>> [DiLoCo] merge #5 at step 383600: averaged model weights across 2048 ranks in 8902 ms
>>> [DiLoCo] merge #6 at step 383620: averaged model weights across 2048 ranks in 8843 ms
>>> [DiLoCo] merge #7 at step 383640: averaged model weights across 2048 ranks in 8885 ms
>>> [DiLoCo] final merge SKIPPED at step 383640: last step already merged (step % K == 0); checkpoint is already consensus
Training complete! Final step: 383640
FINAL_LOSS_LAST100: 2.6910
DILOCO_MERGES: 7
DILOCO_K: 20
DILOCO_SYNC_TOTAL_S: 62.068
DILOCO_SYNC_AVG_MS: 8866.9
```

Final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/train/emender_E97_1.3B_20260627_122441/checkpoint_step_383640_loss_2.6910.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_256n_hier_g4_bucket64m_smoke/4908935-20260627T161408Z/train/emender_E97_1.3B_20260627_122441/latest.pt -> checkpoint_step_383640_loss_2.6910.pt
```

Retained checkpoints at completion:

```text
checkpoint_step_383600_loss_3.1029.pt
checkpoint_step_383620_loss_2.2929.pt
checkpoint_step_383640_loss_2.3618.pt
checkpoint_step_383640_loss_2.6910.pt
latest.pt -> checkpoint_step_383640_loss_2.6910.pt
```

There are two step-383640 checkpoint files because the normal save-at-merge
checkpoint used the instantaneous logged loss `2.3618`, then the finalization
checkpoint used `FINAL_LOSS_LAST100=2.6910` and updated `latest.pt` to that
final consensus checkpoint.

## Metrics

Rank-0 metric rows parsed from `train.log`:

| Run | Nodes/ranks | Metrics | Final step | Merges | Loss avg | First 4 loss | Last 4 loss | First 10 loss | Last 10 loss | Median global tok/s | Avg global tok/s | Avg merge ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256n hierarchical `4908935` | 256 / 2048 | 28 | 383640 | 7 | 2.6910 | 2.7815 | 2.7584 | 2.6750 | 2.7407 | 4,273,004 | 3,876,147 | 8,866.9 |
| 128n hierarchical `4908849` | 128 / 1024 | 16 | 383580 | 4 | 2.5608 | 2.7537 | 2.6190 | 2.5982 | 2.4949 | 2,644,776 | 2,087,777 | 6,529.6 |
| 64n hierarchical `4908602` | 64 / 512 | 48 | 383740 | 12 | 2.5720 | 2.5924* | 2.5713* | 2.5924 | 2.5713 | 1,382,453 | 1,012,810 | 6,436.2 |

`*` Prior reports used first/last 10 loss windows for the 64-node row; those
values are repeated where only four-window numbers are not available.

Interpretation:

- 256-node hierarchical worked at 2048 ranks with the conservative group-size-4,
  64M bucket, and group-create barrier cadence from the passing lower rungs.
- 256-node median global throughput was about `1.62x` the 128-node
  hierarchical median (`4,273,004 / 2,644,776`) and about `3.09x` the 64-node
  hierarchical median (`4,273,004 / 1,382,453`). Average global throughput was
  about `1.86x` the 128-node average (`3,876,147 / 2,087,777`).
- Average hierarchical merge time increased to `8,866.9 ms` at 256 nodes from
  `6,529.6 ms` at 128 nodes. This is a noticeable cost increase, but not a
  severe anomaly for this safety gate because the job completed 7 merges and
  checkpointed cleanly.
- Loss was finite throughout the 256-node run. The reported
  `FINAL_LOSS_LAST100` was `2.6910`; rank-0 first-10 average was `2.6750` and
  last-10 average was `2.7407`.
- Startup/process-group/resume overhead still dominated the bounded allocation.
  The job started at `16:14:05Z`; the first rank-0 training metric appeared at
  `16:30:48Z`. The final-checkpoint margin triggered at step `383640`, after 7
  K20 merges.

## Failure-Signature Scan

Scanned stdout, stderr, and `train.log` for watchdogs, segfaults, communicator
construction failures, NCCL/RCCL failures, tracebacks, exceptions, OOM, and
timeouts. No severe failure signature was found. The only match in the broad
scan was the benign exported environment variable:

```text
SLURM_OOM_KILL_STEP=0
```

The summary's "Error-Like Lines" section contains expected Triton version
warnings, matching prior successful rungs.

## Gate Decision For Longer Runs

Pass for this conditional 256-node smoke.

Gate criteria:

- Slurm `COMPLETED 0:0`: yes, job `4908935`.
- 2048-rank hierarchical topology line printed: yes, `n_groups=512`,
  `group_size=4`, `bucket_numel=67108864`.
- Multiple completed K20 hierarchical merges: yes, 7 merges at 2048 ranks.
- Final consensus checkpoint/latest written: yes, `latest.pt` points to
  `checkpoint_step_383640_loss_2.6910.pt`; final merge was skipped only because
  step `383640` had already merged at K20 cadence.
- No NCCL/RCCL watchdogs, segfaults, communicator-construction failures, or
  severe throughput/merge-time anomaly observed in the parsed evidence: yes.
- Loss finite: yes, `FINAL_LOSS_LAST100=2.6910`.

Recommendation:

- Keep larger extended K160/science runs paused until a follow-up task scopes
  walltime, pass/fail criteria, and operator approval.
- The 256 smoke supports proceeding to a bounded longer validation, but the
  run should budget for the observed startup overhead and preserve the exact
  conservative settings used here unless a later analysis task recommends a
  safer change.

## Validation

- Gate decision from the 128-node report is quoted above.
- 256-node job ID, run root, stdout/stderr paths, manifest, env, train log,
  exact config, and launcher command are recorded above.
- Job `4908935` was monitored to terminal `COMPLETED 0:0`.
- Hierarchical worked at 256 nodes: setup completed, 7 K20 merges completed,
  final checkpoint completed, and `latest.pt` points at the final checkpoint.
- Larger extended runs remain paused; this report recommends only a separately
  scoped bounded longer validation or K160 run, not an immediate extended run.
