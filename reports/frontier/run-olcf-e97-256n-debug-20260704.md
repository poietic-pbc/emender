# OLCF E97 256-Node Updated-Runtime Debug Probe

Task: `run-olcf-e97-256n-debug`
Date: 2026-07-04
Status: gated no-submit

## Recommendation

No-go for the 256-node 30-minute debug probe and no-go for the downstream
256-node x 12-hour production run.

This task was explicitly gated on the 64-node predecessor reporting pass/stable.
The predecessor task `run-olcf-e97-64n-debug` completed as a gated no-submit and
recommended no-go for this 256-node probe. Therefore no 256-node Slurm job was
submitted.

The relevant predecessor chain is:

```text
run-olcf-e97-8n-debug  -> no-go for 64n updated-runtime ladder
run-olcf-e97-64n-debug -> no 64n Slurm submission; no-go for 256n debug
run-olcf-e97-256n-debug -> no 256n Slurm submission; no-go for 256n x 12h production
```

The 8-node probe did start all 64 ranks, load the active checkpoint, train with
finite losses, and complete DiLoCo merges, but it did not validate the requested
updated OLCF runtime inside `train.py`. Its wrapper preflight saw the intended
candidate prefix and torch/HIP stack, while the actual training manifest
recorded the older runtime:

```text
wrapper preflight env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
wrapper preflight torch_version=2.10.0+rocm7.1
wrapper preflight torch_hip=7.1.25424
training manifest torch_version=2.8.0.dev20250422+rocm6.4
training manifest triton_version=3.2.0
```

That mismatch means the 8-node pass/stable gate was not met for the stated
updated-runtime ladder, and the 64-node predecessor correctly did not submit.
Because the 64-node predecessor produced no pass/stable evidence, this 256-node
task must also stop before scheduler submission.

## Submission

- 256-node Slurm job id: none.
- Partition/QOS: not requested.
- Nodes/ranks: not allocated; no 2048-rank job launched.
- Walltime requested: `00:00:00` for this task instance because no job was
  submitted.
- Requested node-hours: `0`.
- Actual elapsed: `00:00:00`.
- Actual node-hours: `0`.
- Retry count: `0`; retry policy was not invoked because this was a prerequisite
  gate failure, not a transient rank-start, rendezvous, or node failure.

No `sbatch`, `salloc`, or `srun` command was issued for the 256-node probe in
this task instance. The debug partition 256-node admission check was not
performed because the prerequisite gate failed before scheduler interaction.

## Gate Evidence

64-node predecessor task:

```text
task: run-olcf-e97-64n-debug
status: done
artifact: reports/frontier/run-olcf-e97-64n-debug-20260704.md
64-node job id: none
64-node requested node-hours: 0
64-node actual node-hours: 0
64-node recommendation: no-go for the 256-node debug probe
```

The 64-node predecessor report states that it did not submit because the
8-node predecessor failed the updated-runtime gate:

```text
No-go for the 64-node updated-runtime ladder as-is.
```

and:

```text
No-go for the 256-node debug probe.
```

The WG task log for `run-olcf-e97-64n-debug` records the same gate decision:

```text
8n predecessor gate failed: report recommends no-go for 64n because actual
training runtime mismatched requested updated OLCF runtime. No 64n Slurm job
will be submitted.
```

The predecessor was evaluated and marked done, so this is the current validated
upstream state, not an ambiguous in-progress result.

## Runtime, Plugin, And Checkpoint Evidence

No new 256-node runtime, plugin, or checkpoint evidence exists because the
64-node pass/stable gate failed before submission.

The available inherited evidence is:

```text
intended OLCF candidate prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
intended torch_version=2.10.0+rocm7.1
intended torch_hip=7.1.25424
real librccl-net.so path=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0
actual 8n training torch_version=2.8.0.dev20250422+rocm6.4
actual 8n training triton_version=3.2.0
active checkpoint symlink=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
active checkpoint target=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

The real plugin path was proven reachable in the 8-node wrapper preflight, but
the actual training runtime mismatch blocks using that run as updated-runtime
scaleout evidence.

## Distributed Init Evidence

No 2048-rank distributed initialization occurred because no 256-node job was
submitted.

There is no 256-node `DistStoreError`, TCPStore timeout, `2016/2048` partial
join, watchdog, or collective failure signature from this task instance. This is
an intentional prerequisite stop, not a failed 2048-rank launch.

The previous known 256-node failure described in
`reports/frontier/decide-runtime-comm-scaleout-20260704.md` remains the relevant
scale-risk backdrop: job `4936017` failed during process-group startup with
`2016/2048` clients joined. This task adds no new evidence that current
PyTorch/RCCL can safely drop missing ranks, and it does not change that no-go
interpretation.

## Training Evidence

No 256-node E97 training evidence exists because no 256-node job was submitted.

The inherited 8-node evidence showed finite training and DiLoCo merges under the
wrong actual training runtime. It is insufficient to justify either a 256-node
debug probe or a 256-node x 12-hour production launch under the updated-runtime
criteria.

## Production Symlink Guard

This task performed no scheduler submission and requested no production chain
update. The active production pointer was inspected before completing this
report:

```text
path: /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
resolved: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
before: '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt' -> '/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt'|594487587176043644|1783092774
after:  '/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt' -> '/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt'|594487587176043644|1783092774
```

No production symlink change was expected or performed by this no-submit task.

## Go/No-Go For 256n x 12h Production

No-go.

Do not submit `submit-olcf-e97-256n12h-production` from the current evidence.
The 256-node production precondition requires the 256-node 30-minute debug probe
to report explicit go/stable. This task reports no-submit/no-go because the
64-node predecessor did not pass.

Before revisiting 256-node debug or production:

1. Repair the delegated `srun` training environment activation so `train.py`
   runs under the same OLCF candidate runtime reported by wrapper preflight.
2. Rerun the updated-runtime 8-node debug probe and require matching wrapper and
   training manifest evidence for torch `2.10.0+rocm7.1`, HIP `7.1.25424`, and
   the expected Triton stack.
3. Only after that replacement 8-node probe passes, rerun the 64-node debug
   probe and require correct runtime/plugin/checkpoint evidence plus finite
   training and merge evidence.
4. Submit a 256-node 30-minute debug probe only after the replacement 64-node
   probe reports pass/stable.

If repeated 256-node attempts later fail from missing ranks or rendezvous,
mark scaleout blocked rather than treating current PyTorch/RCCL as able to drop
missing ranks. The alternative design should be explicit island training with
offline or hierarchical merge semantics, not implicit elastic recovery inside a
fixed-size c10d world.

## Validation Checklist

- [x] Confirms 64n predecessor passed before submission: predecessor did not
  pass/stable; it reported gated no-submit/no-go, so 256n submission was
  correctly blocked.
- [x] Job id(s), partition, elapsed, requested/actual node-hours recorded: no
  256-node job; all values recorded as none/zero.
- [x] Runtime/plugin/checkpoint evidence recorded: inherited 8-node/64-node
  gate evidence and the blocking runtime mismatch are recorded.
- [x] 2048-rank init evidence or precise failure signature recorded: no
  2048-rank init occurred due to prerequisite gate failure; no failure signature
  exists for this task instance.
- [x] Training evidence recorded if init succeeds: init did not occur; no
  256-node training evidence exists.
- [x] Production symlink before/after guard recorded: active pointer metadata is
  recorded; no production update path was invoked.
- [x] Explicit go/no-go for 256n x 12h production: no-go.

## Bottom Line

The scaleout ladder remains stopped before 64 nodes, so 256 nodes is not
eligible for either debug or production submission. Fix the runtime activation
mismatch, re-establish a clean 8-node updated-runtime pass, then rebuild the
64-node evidence before considering 256 nodes again.
