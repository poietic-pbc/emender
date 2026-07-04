# OLCF E97 64-Node Updated-Runtime Debug Probe

Task: `run-olcf-e97-64n-debug`
Date: 2026-07-04
Status: gated no-submit

## Recommendation

No-go for the 256-node debug probe.

The 64-node probe was not submitted. The task explicitly required this step to
proceed only if the 8-node predecessor reported pass/stable. The predecessor
artifact `reports/frontier/run-olcf-e97-8n-debug-20260704.md` reports:

```text
No-go for the 64-node updated-runtime ladder as-is.
```

The 8-node job started all 64 ranks, loaded the active checkpoint, trained with
finite metrics, and completed DiLoCo merges, but it did not validate the stated
updated runtime. Its wrapper preflight saw the intended conda prefix with torch
`2.10.0+rocm7.1` and HIP `7.1.25424`, while the actual `train.py`
`run_manifest.json` recorded torch `2.8.0.dev20250422+rocm6.4` and Triton
`3.2.0`. That mismatch means the required 8-node pass/stable gate was not met.

Do not advance to `run-olcf-e97-256n-debug` until `fix-olcf-debug` has repaired
the `srun` training environment activation and a replacement 8-node
updated-runtime probe passes with matching wrapper and training runtime
evidence.

## Submission

- 64-node Slurm job id: none.
- Partition/QOS: not requested.
- Nodes/ranks: not allocated.
- Walltime requested: `00:00:00` for this task instance because no job was
  submitted.
- Requested node-hours: `0`.
- Actual elapsed: `00:00:00`.
- Actual node-hours: `0`.
- Retry count: `0`; retry policy was not invoked because this was a prerequisite
  gate failure, not a transient allocation or rank-start failure.

No `sbatch`, `salloc`, or `srun` command was issued for the 64-node probe in
this task instance.

## Gate Evidence

Predecessor job:

```text
8-node task: run-olcf-e97-8n-debug
8-node job: 4940985
8-node partition/QOS: batch/debug
8-node terminal state: TIMEOUT, exit code 0:0 at the job level
8-node recommendation: no-go for 64n
```

Relevant predecessor evidence:

```text
wrapper preflight torch_version=2.10.0+rocm7.1
wrapper preflight torch_hip=7.1.25424
training manifest torch_version=2.8.0.dev20250422+rocm6.4
training manifest triton_version=3.2.0
```

The predecessor also created `fix-olcf-debug` to repair the delegated canary
training environment mismatch. At the time of this report, that fix task was
still in progress, so there was no new passing 8-node evidence to supersede the
recorded no-go.

## Runtime, Plugin, And Checkpoint Evidence

No new 64-node runtime, plugin, or checkpoint evidence exists because the
prerequisite gate prevented submission.

The runtime/plugin/checkpoint evidence available for this decision is inherited
from the 8-node predecessor:

```text
intended env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
intended torch_version=2.10.0+rocm7.1
intended torch_hip=7.1.25424
rccl_net_plugin_status=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
FRONTIER_RCCL_NET_PLUGIN_MODULE=rccl-net-plugin/1.0
actual training torch_version=2.8.0.dev20250422+rocm6.4
actual training triton_version=3.2.0
active checkpoint=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
```

The plugin path requirement was satisfied in the 8-node preflight, but the
training runtime mismatch is a blocker for the updated-runtime ladder.

## Training And Merge Evidence

No 64-node training or DiLoCo merge evidence exists because no 64-node job was
submitted.

This is an intentional stop, not a failed 64-node scale result. The 8-node
predecessor had finite training metrics and DiLoCo merges, but under the wrong
actual training runtime for this ladder. That evidence is insufficient to
justify a 256-node updated-runtime debug probe.

## Production Symlink Guard

The task did not submit a job and did not request any production chain update.
The active production pointer was inspected during this task:

```text
path: /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
resolved: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
metadata: /lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|594487543672603835|1782849877
```

No before/after change was expected or observed by this no-submit task.

## Validation Checklist

- [x] Confirms 8n predecessor passed before submission: predecessor did not
  pass the stated updated-runtime gate, so submission was correctly blocked.
- [x] Job id(s), partition, elapsed, requested/actual node-hours recorded:
  no 64-node job; all values recorded as none/zero.
- [x] Runtime/plugin/checkpoint evidence recorded: inherited 8-node evidence and
  the blocking runtime mismatch are recorded.
- [x] Training/merge evidence recorded: no new 64-node evidence; predecessor
  training/merge evidence is explicitly marked insufficient for this ladder.
- [x] Production symlink before/after guard recorded: active pointer metadata is
  recorded and no production chain update was performed.
- [x] Clear pass/no-go recommendation for 256n debug: no-go.

## Bottom Line

The ladder is stopped at the 8-node gate. Submit no 64-node or 256-node
updated-runtime debug jobs until the training environment activation fix lands
and a replacement 8-node probe proves the same updated torch/HIP/Triton stack
inside `train.py` that the wrapper preflight reports.
