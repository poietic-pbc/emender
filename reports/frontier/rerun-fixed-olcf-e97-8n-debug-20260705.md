# Fixed OLCF E97 8-Node Debug Rerun

Task: `rerun-fixed-olcf-e97-8n-debug`
Date: 2026-07-05
Code under test: `origin/main` / `HEAD` `ed2acc8`

## Recommendation

No-go for `rerun-fixed-olcf-e97-64n-debug` yet.

The post-fix rerun did validate the main target of `fix-olcf-debug`: the actual
`train.py` process, not just the wrapper preflight, ran from
`.envs/olcf-rocm711-torch210-py312` with torch `2.10.0+rocm7.1`, HIP
`7.1.25424`, and Triton `3.6.0`. It also loaded the real OLCF RCCL net plugin,
started all 64 ranks across 8 nodes, loaded the active production checkpoint,
compiled the fused E97 path, and logged finite training metrics.

However, the corrected job timed out before the first K80 DiLoCo merge and
before the outer wrapper's final `production_latest_after` guard could print.
The production symlink metadata was checked manually after the job and matches
the recorded pre-run metadata, but the task's requested evidence includes a
merge if time allows and a before/after guard. Treat this as a partial technical
pass for the runtime fix, but not a full 8-node gate pass for the 64-node rerun.

## Jobs

| Job | Purpose | Nodes | State | Exit | Elapsed | Actual node-hours | Notes |
| --- | --- | ---: | --- | --- | ---: | ---: | --- |
| `4942247` | Initial submission | 8 | `FAILED` | `1:0` | `00:00:10` | `0.0222` | Setup error: `REPO` pointed at WG worktree without the local conda env |
| `4942249` | Corrected rerun | 8 | `TIMEOUT` | `0:0` | `00:30:12` | `4.0267` | Runtime fixed; ranks/checkpoint/training passed; no merge before walltime |

Total actual node-hours: about `4.0489`.

Accounting evidence:

```text
4942247|e97-olcf-8n-debug|batch|debug|FAILED|1:0|00:00:10|8|8|frontier[07868-07872,07877-07878,07933]
4942249|e97-olcf-8n-debug|batch|debug|TIMEOUT|0:0|00:30:12|8|8|frontier[07835,07844,07847,07867-07871]
4942249.1|bash|||TIMEOUT|0:0|00:29:44|8|8|frontier[07835,07844,07847,07867-07871]
```

The failed first submission did not reach training:

```text
EnvironmentLocationNotFound: Not a conda environment:
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-594/.envs/olcf-rocm711-torch210-py312
```

The corrected job used the shared repo path containing the candidate prefix:

```bash
sbatch -N 8 -t 00:30:00 -J e97-olcf-8n-debug \
  --export=ALL,WG_TASK_ID=rerun-fixed-olcf-e97-8n-debug,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender,BATCH_SIZE=4,DILOCO_K=80,TRAIN_MINUTES=18,LOG_EVERY=1,SAVE_EVERY=999999,KEEP_CHECKPOINTS=1,REQUESTED_NODE_HOURS=4.0,REQUESTED_SECONDS=1800,REQUESTED_WALLTIME=00:30:00,HUMAN_APPROVAL_RECORD='WG rerun-fixed-olcf-e97-8n-debug: fixed 8-node E97 B4 K80 OLCF runtime debug probe, 2026-07-05; isolated output root; production chain pointer updates disabled.' \
  scripts/frontier/e97_updated_olcf_runtime_debug.sbatch
```

## Artifacts

- Slurm stdout: `logs/frontier/debug/e97-olcf-8n-debug-4942249.out`
- Slurm stderr: `logs/frontier/debug/e97-olcf-8n-debug-4942249.err`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260705/E97_1.3B_active_latest_olcf_runtime_debug/4942249-20260705T033243Z`
- Env file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260705/E97_1.3B_active_latest_olcf_runtime_debug/4942249-20260705T033243Z/artifacts/env.txt`
- Rank-start evidence:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260705/E97_1.3B_active_latest_olcf_runtime_debug/4942249-20260705T033243Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260705/E97_1.3B_active_latest_olcf_runtime_debug/4942249-20260705T033243Z/logs/train.log`
- Actual train manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260705/E97_1.3B_active_latest_olcf_runtime_debug/4942249-20260705T033243Z/train/emender_E97_1.3B_20260704_233355/run_manifest.json`

No wrapper `artifacts/manifest.json`, wrapper summary, or
`production_latest_after` line was written because Slurm cancelled the job at
the debug walltime.

## Runtime And Plugin Evidence

Wrapper preflight:

```text
env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
loaded_modules=...:rocm/7.1.1:craype-accel-amd-gfx90a:rccl-net-plugin/1.0
"torch_version": "2.10.0+rocm7.1"
"torch_hip": "7.1.25424"
```

Delegated `srun` preflight:

```text
python_executable=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
python_version=3.12.13
torch.__version__=2.10.0+rocm7.1
torch.version.hip=7.1.25424
triton.__version__=3.6.0
librccl_net_path=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
NCCL_NET_PLUGIN=librccl-net.so
```

Actual `train.py` run manifest:

```text
"python_executable": "/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python"
"python_version": "3.12.13"
"torch_version": "2.10.0+rocm7.1"
"torch_version_hip": "7.1.25424"
"triton_version": "3.6.0"
"olcf_ofi_nccl_root": "/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1"
"librccl_net_path": "/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so"
```

This directly resolves the prior no-go reason from job `4940985`: the actual
training process kept the requested OLCF candidate runtime active.

## Rank And Distributed Init Evidence

- `rank-start.tsv` lines: `64`.
- Unique rank ids: `64`.
- Unique nodes: `8`.
- `MASTER_PORT=42249`, derived from job id `4942249`.
- Train log recorded explicit init timeout: `[distributed-init] timeout=1800.0s`.
- Train log recorded `[DiLoCo] world_size=64 backend=nccl`.
- No `non-finite`, `NaN`, `nan`, `Traceback`, `RuntimeError`,
  `DistStoreError`, `TCPStore`, or `watchdog` strings were found in the train
  log.

Rank-start sample:

```text
2026-07-05T03:32:56Z	4	4	frontier07835	64
2026-07-05T03:32:57Z	5	5	frontier07835	64
2026-07-05T03:32:57Z	3	3	frontier07835	64
...
2026-07-05T03:33:00Z	17	1	frontier07847	64
2026-07-05T03:33:00Z	23	7	frontier07847	64
2026-07-05T03:33:00Z	22	6	frontier07847	64
```

## Checkpoint And Chain Guard

Active production pointer at wrapper start:

```text
production_latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
resolved_production_latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
production_latest_before='/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
chain_latest_path_after_guard=''
chain_update_on_failure=0
```

Manual post-run check after Slurm timeout:

```text
'/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

The production symlink metadata is unchanged from the recorded pre-run value.
Because the wrapper was killed at walltime, this is a manual after-guard rather
than the wrapper's own final after-guard.

Checkpoint load evidence:

```text
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
Starting training from step 489920...
```

## E97 Training Evidence

The fused E97 guard appeared for all 64 ranks:

```text
[fused-guard] rank 0/64: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback
```

Count: `64` fused-guard lines.

The E97 runtime path appeared under HIP:

```text
[e97-runtime] backend=hip path=e88-sequential-split-edit-triton use_triton=True use_chunked_e97=False e97_chunk_size=32 linear_state=False raw_write=False use_split_edit=True log_decay=True
```

Finite metrics were logged from step `489921` through `489999`:

```text
step 489921 | loss 1.7467 | lr 1.01e-03 | grad 0.83 | tok/s 4126 | global_tok/s 264091 | elapsed_h 0.001 | time 2026-07-05T03:34:12+00:00
step 489999 | loss 2.6478 | lr 1.01e-03 | grad 0.77 | tok/s 5271 | global_tok/s 337350 | elapsed_h 0.035 | time 2026-07-05T03:36:16+00:00
```

The job did not log a DiLoCo merge:

```text
grep -c '>>> \[DiLoCo\]' train.log
0
```

It reached exactly 79 post-resume training steps; the first K80 merge would
have been at step `490000`. The job then remained alive until Slurm walltime
and was cancelled:

```text
[2026-07-05T00:02:38.032] error: *** STEP 4942249.1 ON frontier07835 CANCELLED AT 2026-07-05T00:02:38 DUE TO TIME LIMIT ***
```

No debug-run checkpoint or `latest.pt` was found under the run root.

## Validation Checklist

- [x] Job ids, terminal states, elapsed, and node-hours recorded.
- [x] Actual training runtime/version evidence recorded from
  `run_manifest.json`.
- [x] Plugin, checkpoint, rank-start, explicit init timeout, and unique port
  evidence recorded.
- [x] Finite training evidence recorded.
- [ ] DiLoCo merge evidence recorded. No merge occurred before timeout.
- [x] Production symlink before guard recorded; manual after guard recorded.
  Wrapper after guard did not run because of Slurm timeout.
- [x] Clear recommendation recorded: no-go for fixed 64-node rerun until an
  8-node rerun captures at least one merge and exits or final-guards cleanly.

## Bottom Line

The OLCF conda activation fix worked for actual training. The corrected 8-node
job confirms torch `2.10.0+rocm7.1` / HIP `7.1.25424` / Triton `3.6.0` in the
training manifest, real `librccl-net.so` plugin resolution, all-rank startup,
active checkpoint load, fused E97 execution, and finite loss.

Do not launch the fixed 64-node rerun from this evidence alone. Submit a short
follow-up 8-node debug run with enough walltime margin or a lower temporary
`DILOCO_K`/step cap so it records at least one merge and the final production
symlink after-guard before advancing the 64-node gate.
