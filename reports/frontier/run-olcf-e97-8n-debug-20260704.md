# OLCF E97 8-Node Updated-Runtime Debug Probe

Task: `run-olcf-e97-8n-debug`
Date: 2026-07-04
Job: `4940985`

## Recommendation

No-go for the 64-node updated-runtime ladder as-is.

The 8-node job proved that the hardened launcher can load the real OLCF RCCL
plugin, start all 64 ranks, load the active production checkpoint, compile/run
the fused E97 path, train with finite loss, and complete DiLoCo merges. However,
the actual `train.py` runtime did not match the requested updated runtime. The
wrapper preflight saw the intended conda prefix and torch `2.10.0+rocm7.1`, but
the training run manifest reports torch `2.8.0.dev20250422+rocm6.4` and Triton
`3.2.0`. That is a setup/config failure for the stated purpose of this probe,
so do not unblock the 64-node updated-runtime run until the delegated canary
keeps the intended environment active inside `srun`.

## Submission

- Submit command:

```bash
sbatch -N 8 -t 00:30:00 -J e97-olcf-8n-debug \
  --export=ALL,WG_TASK_ID=run-olcf-e97-8n-debug,BATCH_SIZE=4,DILOCO_K=80,TRAIN_MINUTES=20,LOG_EVERY=1,SAVE_EVERY=999999,KEEP_CHECKPOINTS=1,REQUESTED_NODE_HOURS=4.0,REQUESTED_SECONDS=1800,REQUESTED_WALLTIME=00:30:00,HUMAN_APPROVAL_RECORD='WG run-olcf-e97-8n-debug: 8-node E97 B4 K80 OLCF runtime debug probe, 2026-07-04; isolated output root; production chain pointer updates disabled.' \
  scripts/frontier/e97_updated_olcf_runtime_debug.sbatch
```

- Slurm job id: `4940985`.
- Partition/QOS: `batch` / `debug`.
- Nodes/ranks: 8 nodes, 64 ranks, 8 ranks per node.
- Walltime requested: `00:30:00`.
- Requested node-hours: `4.0`.
- Actual elapsed: `00:30:25`.
- Actual node-hours: about `4.056` node-hours (`8 * 30.4167 / 60`).
- Terminal state: `TIMEOUT`, exit code `0:0` at the job level; job step
  `4940985.0` was cancelled by Slurm at walltime.
- Nodes allocated:
  `frontier[00605,00712,00847,00993,01124,01251,01379,01496]`.

`sacct` evidence:

```text
4940985|e97-olcf-8n-debug|batch|debug|TIMEOUT|0:0|00:30:25|8|8
4940985.0|bash|||CANCELLED|0:9|00:30:34|8|8
```

The walltime timeout happened after productive training. It is not a rendezvous
or setup timeout, but it also means the wrapper did not exit cleanly or write its
final manifest/summary.

## Artifacts

- Slurm stdout: `logs/frontier/debug/e97-olcf-8n-debug-4940985.out`
- Slurm stderr: `logs/frontier/debug/e97-olcf-8n-debug-4940985.err`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4940985-20260704T221617Z`
- Env file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4940985-20260704T221617Z/artifacts/env.txt`
- Rank-start evidence:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4940985-20260704T221617Z/artifacts/rank-start.tsv`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4940985-20260704T221617Z/logs/train.log`
- Train run manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260704/E97_1.3B_active_latest_olcf_runtime_debug/4940985-20260704T221617Z/train/emender_E97_1.3B_20260704_181725/run_manifest.json`

No canary manifest or summary was written under `artifacts/manifest.json` or
`summaries/summary.md` because the outer wrapper was killed by Slurm at the
30-minute limit before its post-`srun` summary block could run.

## Checkpoint And Chain Guard

Active production pointer:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

Resolved target recorded by the debug wrapper:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

Production symlink guard:

```text
before: '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
after:  '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt'|7719679569|1782849877
```

The debug env recorded:

```text
chain_latest_path=
chain_manifest_path=
chain_update_on_failure=0
```

The production chain pointer did not change.

## Runtime And Plugin Evidence

Wrapper preflight reported the requested candidate prefix and correct torch/HIP:

```text
env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
torch_version=2.10.0+rocm7.1
torch_hip=7.1.25424
```

The real OLCF RCCL plugin resolved successfully:

```text
rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
NCCL_NET_PLUGIN=librccl-net.so
FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0
require_rccl_net_plugin=1
```

But the actual training manifest contradicts the requested updated runtime:

```text
torch_version=2.8.0.dev20250422+rocm6.4
triton_version=3.2.0
olcf_ofi_nccl_root=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1
librccl_net_path=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
```

The Slurm stdout also contains repeated warnings from the training ranks:

```text
Current Python version 3.10 is below the recommended 3.11 version.
Current Triton version 3.2.0 is below the recommended 3.3.0 version.
```

This is the primary failure: the probe did not validate training under
torch `2.10.0+rocm7.1` / HIP `7.1.25424` / Triton `3.6.0`.

## Distributed Init And Rank Evidence

- `rank-start.tsv` lines: `64`.
- Unique rank ids: `64`.
- Unique nodes: `8`.
- Train log recorded explicit init timeout: `[distributed-init] timeout=1800.0s`.
- Train log recorded `world_size=64 backend=nccl`.
- No `DistStoreError`, `TCPStore`, `watchdog`, `Traceback`, `RuntimeError`,
  `non-finite`, `NaN`, or `nan` strings were found in the train log.

Rank-start sample:

```text
2026-07-04T22:16:24Z	7	7	frontier00605	64
2026-07-04T22:16:24Z	5	5	frontier00605	64
2026-07-04T22:16:24Z	1	1	frontier00605	64
...
2026-07-04T22:16:25Z	10	2	frontier00712	64
2026-07-04T22:16:25Z	61	5	frontier01496	64
2026-07-04T22:16:25Z	62	6	frontier01496	64
```

## E97 Compile And Training Evidence

The fused E97 guard appeared for all 64 ranks:

```text
[fused-guard] rank 0/64: level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback
```

Count: `64` fused-guard lines.

Resume and batch/chunk evidence:

```text
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
Starting training from step 489920...
Batch size: 4, Chunk size: 2048
```

Finite metrics were logged through walltime. Final visible metric:

```text
step 490392 | loss 3.1583 | lr 1.01e-03 | grad 0.84 | tok/s 5514 | global_tok/s 352923 | elapsed_h 0.479 | time 2026-07-04T22:46:25+00:00
```

At least five DiLoCo merges completed:

```text
>>> [DiLoCo] merge #1 at step 490000: averaged model weights across 64 ranks in 5894 ms (amortized over 80 steps)
>>> [DiLoCo] merge #2 at step 490080: averaged model weights across 64 ranks in 2334 ms (amortized over 80 steps)
>>> [DiLoCo] merge #3 at step 490160: averaged model weights across 64 ranks in 7549 ms (amortized over 80 steps)
>>> [DiLoCo] merge #4 at step 490240: averaged model weights across 64 ranks in 7272 ms (amortized over 80 steps)
>>> [DiLoCo] merge #5 at step 490320: averaged model weights across 64 ranks in 6941 ms (amortized over 80 steps)
```

Validation also ran once:

```text
>>> validation loss: 14.0281
```

## Validation Checklist

- [x] Job id, partition, elapsed, requested/actual node-hours recorded.
- [x] Active checkpoint pointer and resolved target recorded.
- [x] Runtime/plugin evidence recorded.
- [x] Training/merge evidence recorded.
- [x] Production symlink before/after guard recorded.
- [x] Clear pass/no-go recommendation for 64n recorded.

## Bottom Line

The 8-node probe is a strong positive signal for the OLCF RCCL plugin path,
distributed init, checkpoint load, fused E97 execution, finite training, and
DiLoCo merge behavior. It is not sufficient to advance the updated-runtime
ladder because the actual training process ran the old torch/ROCm/Triton stack.

Fix the delegated environment activation first, then rerun this 8-node probe
with a shorter `TRAIN_MINUTES` or a real walltime drain margin so it can exit
cleanly before the 30-minute debug limit.
