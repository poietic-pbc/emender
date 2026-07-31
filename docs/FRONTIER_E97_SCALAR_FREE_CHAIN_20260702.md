# Frontier E97 Scalar-Free 2h Chain

Date: 2026-07-02

## Failure Summary

The 64-node B4/K80 24h run `4911454` failed after healthy training at step
`490000`. The DiLoCo merge at that step completed successfully:

```text
[DiLoCo] merge #88 at step 490000: averaged model weights across 512 ranks in 10885 ms
step 490000 | loss 2.5486 | global_tok/s 2124008
```

The subsequent failure was a default-process-group scalar status collective:

```text
WorkNCCL(SeqNum=517, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
```

This was not a model-loss instability and not the hierarchical DiLoCo model
merge. Earlier fixes had reduced these scalar status collectives to every
`WALLTIME_CHECK_EVERY` / `DISTRIBUTED_HEALTH_CHECK_EVERY` steps, but had not
removed them. In the B4/K80 recipe they still occurred every 80 steps, exactly
aligned with the DiLoCo merge cadence.

Follow-up 2-node debug validation showed the scalar status collectives were no
longer present, but exposed a separate hierarchical-merge ordering issue under
short K/save stress: rank 0 could finish and log a hierarchical merge while a
different local subgroup was still in its final 64M-element reduce/broadcast.
That produced timeouts like:

```text
WorkNCCL(... OpType=REDUCE, NumelIn=67108864, NumelOut=67108864 ...)
WorkNCCL(... OpType=BROADCAST, NumelIn=67108864, NumelOut=67108864 ...)
```

Those are model-weight DiLoCo collectives, not the skipped scalar status
all-reduces.

A later 2-node boundary smoke (`4930857`) reproduced the original step-490000
hang without scalar status collectives. At that step rank 0 entered rank-local
validation before checking the final train-budget stop, while ranks 1-15 skipped
validation, observed the filesystem stop request, and entered finalization.
Rank 0 stayed in model compute and never wrote its finalization-ready file. This
explains the original scalar all-reduce as a symptom of rank divergence at a
rank-0-only eval/save boundary.

The latest usable checkpoint from that run is:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
```

## Fix

`train.py` now has an opt-in scalar-free status mode:

```text
--disable_scalar_status_collectives
--status_request_poll_seconds <seconds>
```

When enabled, steady-state health, walltime, and train-budget checks do not use
the default process group for scalar all-reduces. DiLoCo model-weight merges are
unchanged. Stop coordination uses an atomic filesystem request at the same safe
optimizer-step boundaries, with a short peer poll window so ranks converge on
the same finalization decision without a RCCL/NCCL scalar collective.

For hierarchical DiLoCo merges, `train.py` now also enables one completion
barrier after each tree reduce/broadcast:

```text
--diloco_merge_completion_barrier 1
```

This barrier is recorded in the Frontier wrappers and enabled by default for the
64-node chain. It runs at DiLoCo merge cadence, not in the per-step/status path.

The final-stop gate now runs immediately after the synchronized training step
logging and before any rank-0-only held-out curve, validation, or periodic
checkpoint work. If a walltime/train-budget stop is due on the same step as
validation, all ranks enter finalization first and rank 0 writes the final
checkpoint instead of diverging into eval while peers wait.

## Chain Pattern

The 24h extended job is replaced by regular-queue 2h jobs:

```text
scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch
scripts/frontier/submit_e97_1p3b_b4_k80_64n_chain.sh
```

Each job:

- resolves `CHAIN_LATEST_PATH` if present;
- otherwise starts from `CHAIN_SEED_CHECKPOINT`;
- trains with the B4/K80 hierarchical 64-node recipe;
- writes normal per-run checkpoints;
- atomically advances `CHAIN_LATEST_PATH` to the run's newest readable
  `latest.pt` target.

The submit helper chains jobs with `afterany` dependencies, so a job that exits
nonzero after writing a usable checkpoint can still feed the next job. If no new
checkpoint exists, the chain pointer remains unchanged.

## Validation

Static validation:

```text
python3 -m py_compile train.py
bash -n scripts/frontier/e97_1p3b_pretrained_canary.sbatch
bash -n scripts/frontier/e97_1p3b_b4_k80_64n_chain2h.sbatch
bash -n scripts/frontier/submit_e97_1p3b_b4_k80_64n_chain.sh
```

Live validation should use a short multi-node debug run with
`DISABLE_SCALAR_STATUS_COLLECTIVES=1` and a small train budget, then inspect the
log for:

- `[status] scalar default-process-group status collectives disabled`;
- no `NumelIn=1` default-pg all-reduce timeout;
- hierarchical merge lines whose reported time includes the completion barrier;
- a final checkpoint or a valid periodic `latest.pt`;
- chain manifest advancement when `CHAIN_LATEST_PATH` is set.
