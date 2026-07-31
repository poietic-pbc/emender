# OLCF E97 256-Node 12-Hour Production Submission

Task: `submit-olcf-e97-256n12h-production`
Date: 2026-07-04
Status: gated no-submit

## Decision

No-go. No 256-node x 12-hour production job was submitted.

The task requires the predecessor `run-olcf-e97-256n-debug` to report explicit
go/stable before any production submission. The predecessor report at commit
`6280e10` instead states:

```text
No-go for the 256-node 30-minute debug probe and no-go for the downstream
256-node x 12-hour production run.
```

It also states that no 256-node debug Slurm job was submitted because the
64-node predecessor did not report pass/stable. That means this production task
must stop before scheduler interaction.

## Gate Evidence

Predecessor artifact:

```text
task: run-olcf-e97-256n-debug
artifact: reports/frontier/run-olcf-e97-256n-debug-20260704.md
artifact source: /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-579/reports/frontier/run-olcf-e97-256n-debug-20260704.md
artifact commit: 6280e10
predecessor status: done
predecessor recommendation: no-go for 256n x 12h production
```

The validated upstream chain is:

```text
run-olcf-e97-8n-debug  -> no-go for 64n updated-runtime ladder
run-olcf-e97-64n-debug -> no 64n Slurm submission; no-go for 256n debug
run-olcf-e97-256n-debug -> no 256n Slurm submission; no-go for 256n x 12h production
submit-olcf-e97-256n12h-production -> no production submission
```

Blocking cause:

```text
8n wrapper preflight env_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
8n wrapper preflight torch_version=2.10.0+rocm7.1
8n wrapper preflight torch_hip=7.1.25424
8n training manifest torch_version=2.8.0.dev20250422+rocm6.4
8n training manifest triton_version=3.2.0
```

The actual training runtime did not match the requested updated OLCF runtime,
so the 8-node run was not valid evidence for the updated-runtime scaleout
ladder. The 64-node and 256-node debug tasks correctly blocked their Slurm
submissions. Therefore this production task cannot submit.

## Submission Record

- Production Slurm job id: none.
- Slurm submission count: `0`.
- Exact `sbatch` command issued: none; no `sbatch`, `salloc`, or `srun`
  command was issued by this task.
- Exact submission environment issued: none; no production job environment was
  exported to a scheduler command.
- Nodes requested: `0` for this task instance because submission was blocked.
- Walltime requested: `00:00:00` for this task instance because submission was
  blocked.
- Requested node-hours: `0`.
- Actual node-hours: `0`.
- Production partition/QOS: not requested.
- Production K value: not selected from a successful 256-node debug report,
  because no successful 256-node debug report exists.

For audit clarity, the intended production shape was not run:

```text
nodes=256
walltime=12:00:00
batch_size=4
chunk_size=2048
runtime_prefix=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312
rccl_net_plugin_module=rccl-net-plugin/1.0
real_librccl_net=/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
mode=GPU-island/no-DDP hierarchical DiLoCo
chain_update=production pointer advances only after train_status=0 and successful final checkpoint/manifest
finalization_margin=30m expected if a future submission becomes eligible
```

No candidate `sbatch` command is recorded as executed, because recording an
executable production command in a no-go report can be mistaken for an issued
submission. The scheduler interaction count is zero.

## Active Checkpoint Pointer

The active production checkpoint inherited from the predecessor report is:

```text
active checkpoint symlink=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt
active checkpoint target=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

No production continuation consumed this checkpoint in this task. No new
production output root, checkpoint, manifest, `latest.pt`, or `last.pt` was
created.

Expected production paths for this no-submit instance:

```text
production run root: none
production stdout/stderr: none
production final checkpoint: none
production final manifest: none
production chain pointer update: none
```

The existing chain path remains the only active production pointer context:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/
```

## Chain Update Protection

No production chain update path was invoked.

The required future behavior remains:

```text
CHAIN_UPDATE_ON_FAILURE=false
production chain pointer may advance only after train_status=0
production chain pointer may advance only after final checkpoint exists
production chain pointer may advance only after final manifest exists
```

Because no job was submitted, there is no possible failure path from this task
that can advance the production chain pointer. This satisfies the pointer
protection requirement for the no-go path.

## Rendezvous And Missing-Rank Check

This task did not submit despite the lack of a new repeated 256-node
missing-rank/rendezvous failure in the immediate predecessor. The predecessor
explicitly records that no 2048-rank initialization occurred because the
64-node gate failed first.

The predecessor also preserves the relevant scale-risk backdrop from job
`4936017`: a prior 256-node startup failure with `2016/2048` clients joined.
This task does not treat that as stable and does not infer elastic recovery
inside a fixed-size c10d world.

## Downstream Monitor Context

Downstream task:

```text
monitor task: monitor-olcf-e97-256n12h-production
job id: none
handoff: no production job exists to monitor; monitor should verify this no-go artifact and record no-op completion unless a later task supersedes this decision with a valid job id.
```

I also sent the monitor task a WG message with the same handoff: job id `none`,
zero submissions, and this report path.

## Validation Checklist

- [x] Confirms predecessor 256n debug report says go/stable: it does not; it
  explicitly says no-go for downstream 256n x 12h production.
- [x] Submits exactly one 256n x 12h production job, or records explicit no-go:
  explicit no-go recorded; zero scheduler submissions.
- [x] Records job id, exact sbatch command/env, active checkpoint
  pointer/target, requested node-hours, and expected output/chain paths: job id
  none; command/env none; active pointer/target, zero node-hours, and no-output
  paths recorded.
- [x] Confirms `CHAIN_UPDATE_ON_FAILURE`/chain behavior protects production
  pointer on failure: no update path invoked; required future chain semantics
  restated with `CHAIN_UPDATE_ON_FAILURE=false`.
- [x] Does not submit if 256n debug had repeated transient-rank/rendezvous
  failures: no submission occurred. The predecessor was already no-go before a
  256n launch, and the prior `2016/2048` risk is not treated as stable.
- [x] Creates or clearly identifies the downstream monitor task/context with
  job id: `monitor-olcf-e97-256n12h-production`, job id `none`.

## Bottom Line

The production launch remains blocked. Repair the runtime activation mismatch,
rerun the 8-node updated-runtime debug probe with matching wrapper and training
manifest evidence, then rebuild the 64-node and 256-node debug evidence before
attempting any 256-node x 12-hour production submission.
