# Hierarchical DiLoCo Smoke Fix - 2026-06-27

Task: `fix-failed-hierarchical`

## Summary

Hierarchical DiLoCo is unblocked for the next scaleout gate. The corrected
1-node, 8-rank hierarchical smoke `4908384` completed with Slurm exit `0:0`,
created two 4-rank local groups with roots `[0, 4]`, completed 91 hierarchical
merges at `DILOCO_K=2`, and wrote a final consensus checkpoint.

Fix commits on this task branch:

- `d469654` - remove the default-process-group barrier interleaving from
  hierarchical subgroup warm-up.
- `8b6e43a` - create the overlapping root communicator before the local
  communicators so Frontier RCCL/NCCL does not construct a malformed root group.

The default `global` / bucketed-global path is unchanged. All code changes are
inside the opt-in `--diloco_merge_topology=hierarchical` path and its tests.

## Failed-job evidence

### Job `4908284`

Slurm accounting:

```text
4908284 emender-hier-merge-smoke FAILED 1:0 elapsed=00:00:03 node=frontier05498
workdir=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-356
submit=sbatch -A bif148 -p batch -N 1 -t 00:10:00 -J emender-hier-merge-smoke scripts/frontier/diloco_scaleout_readiness.sbatch
```

Log/report artifacts:

- Existing report summary:
  `reports/frontier/hierarchical-diloco-integrate-test-20260627.md`
- Original stdout/stderr paths recorded by that worktree:
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-356/logs/frontier/scaleout/emender-hier-merge-smoke-4908284.out`
  and `.err`
- Those original files were no longer present when this task ran because
  `.wg-worktrees/agent-356` had been removed.

Observed failing point from the preserved report: failed before Python in
3 seconds with:

```text
mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied
```

Root-cause classification: smoke-script/environment submission bug. It was not
a hierarchical process-group, subgroup membership, launcher rank mapping, or
ROCm/NCCL failure. The corrected submission used explicit bif148
`OUTPUT_ROOT`.

### Job `4908293`

Slurm accounting:

```text
4908293 emender-hier-merge-smoke FAILED 139:0 elapsed=00:01:31 node=frontier04987
step 4908293.0 CANCELLED 0:11 elapsed=00:01:10
workdir=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-356
```

Run artifacts:

- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908293-20260627T072933Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908293-20260627T072933Z/logs/train.log`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908293-20260627T072933Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908293-20260627T072933Z/artifacts/manifest.json`
- Original stdout/stderr paths recorded in the manifest:
  `logs/frontier/scaleout/emender-hier-merge-smoke-4908293.out`
  and `.err` under the removed `agent-356` worktree.

Observed failing point:

- Distributed init succeeded.
- Rank-local GPU binding printed as `cuda:0` on all ranks. This is normal for
  Frontier `--gpus-per-task=1 --gpu-bind=closest`; previous successful global
  DiLoCo logs show the same pattern.
- Rank 0 broadcasted W0:
  `[DiLoCo] broadcast rank-0 W_0 to all 8 ranks (identical start)`.
- The job failed before printing the hierarchical topology line, with several
  ranks segfaulting and rank 0 / rank 4 reporting NCCL remote-process errors.

Initial hypothesis: the old warm-up sequence interleaved subgroup collectives
with default-process-group barriers. Ranks in one subgroup entered subgroup
`all_reduce` while nonmembers entered a default barrier. Patch `d469654`
removed that interleaving.

Follow-up smoke `4908379` after `d469654` progressed further and exposed the
more precise root cause:

```text
rank0/rank4: dist.all_reduce(_w, group=meta['root_group'])
NCCL invalid argument
AllReduce : invalid root 0 (root should be in the 0..0 range)
```

Run artifacts for `4908379`:

- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908379-20260627T080031Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908379-20260627T080031Z/logs/train.log`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908379-20260627T080031Z/summaries/summary.md`

Final root-cause classification: hierarchical process-group creation order.
Creating all first-level local groups before the overlapping, non-contiguous
root group could leave the Frontier RCCL/NCCL root communicator malformed. The
fix creates the root group first on all ranks, then creates local groups.

## Patch

Changed `train.py` only in the hierarchical path:

- `_build_diloco_hierarchical_merge_groups(...)` computes root ranks first and
  creates the root communicator before any local communicator.
- `_warm_diloco_hierarchical_merge_groups(...)` warms each rank's own local
  communicator, then warms the root communicator on root ranks, then uses one
  default barrier after warm-up.

Added tests in `tests/test_diloco_merge.py`:

- `test_hierarchical_warmup_avoids_default_barrier_between_subgroups`
- `test_hierarchical_group_builder_creates_root_group_before_local_groups`

The global and bucketed-global merge code paths were not edited.

## Local validation

Login shell:

```text
bash -n scripts/frontier/diloco_scaleout_readiness.sbatch
python3 -m py_compile train.py tests/test_diloco_merge.py tests/test_diloco_hierarchical_math.py
```

Frontier miniforge environment:

```text
module load miniforge3/23.11.0-0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python -m pytest \
  tests/test_diloco_merge.py::test_hierarchical_group_builder_creates_root_group_before_local_groups \
  tests/test_diloco_merge.py::test_hierarchical_warmup_avoids_default_barrier_between_subgroups \
  tests/test_diloco_merge.py::test_hierarchical_merge_matches_global_average_for_unequal_groups \
  tests/test_diloco_hierarchical_math.py -q
```

Result:

```text
5 passed in 64.25s
```

## Passing smoke

Submission:

```bash
export REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-362
export WG_TASK_ID=fix-failed-hierarchical
export OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
export GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2
export SCALEOUT_VARIANT=gdn2-MLP
export SCALEOUT_NODES=1
export SCALEOUT_WALLTIME=00:15:00
export TRAIN_MINUTES=3
export HUMAN_APPROVAL_RECORD='WG fix-failed-hierarchical minimal hierarchical merge smoke, 2026-06-27; 1-node debug QOS only, validates 8b6e43a root-first communicator fix'
export DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt
export VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
export TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
export EMENDER_CONDA_ENV=base
export DILOCO_K=2
export DILOCO_ISLAND_SIZE=1
export DILOCO_MERGE_TOPOLOGY=hierarchical
export DILOCO_MERGE_GROUP_SIZE=4
export DILOCO_MERGE_BUCKET_NUMEL=1048576
export NDM_DILOCO_MERGE_DEBUG=1
export NDM_DILOCO_MERGE_DEBUG_RANKS=0,4
export LOG_EVERY=1
export VAL_EVERY=1000
export SAVE_EVERY=1000
export KEEP_CHECKPOINTS=1
export COMPILE_WARMUP_STEPS=1
sbatch -A bif148 -p batch --qos=debug -N 1 -t 00:15:00 -J emender-hier-merge-smoke scripts/frontier/diloco_scaleout_readiness.sbatch
```

Slurm result:

```text
4908384 emender-hier-merge-smoke COMPLETED 0:0 elapsed=00:04:22 node=frontier00127
4908384.0 COMPLETED 0:0 elapsed=00:03:56
```

Run artifacts:

- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908384-20260627T080614Z`
- Train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908384-20260627T080614Z/logs/train.log`
- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908384-20260627T080614Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908384-20260627T080614Z/artifacts/manifest.json`
- Final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/gdn2-MLP/4908384-20260627T080614Z/train/gdn2_gdn2-mlp_1.3B_20260627_040712/checkpoint_step_000182_loss_0.6755.pt`

Key pass evidence:

```text
[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, 4] roots=[0, 4] (exact weighted SUM/world_size; bucket_numel=1048576)
step      1 | loss 11.2562 | lr 9.00e-04 | grad 20.25 | tok/s 4208 | global_tok/s 33661
[DiLoCoMergeDebug] rank=0 step=2 merge=1 label=sf_x bucket=0 ... phase=enter
[DiLoCoMergeDebug] rank=0 step=2 merge=1 label=sf_x bucket=0 ... phase=exit
[DiLoCoMergeDebug] rank=4 step=2 merge=1 label=sf_x bucket=0 ... phase=enter
[DiLoCoMergeDebug] rank=4 step=2 merge=1 label=sf_x bucket=0 ... phase=exit
>>> [DiLoCo] merge #91 at step 182: averaged model weights across 8 ranks in 1086 ms (amortized over 2 steps)
step    182 | loss 0.0343 | lr 9.00e-04 | grad 0.71 | tok/s 1365 | global_tok/s 10920
Training complete! Final step: 182
FINAL_LOSS_LAST100: 0.6755
PEAK_MEMORY_MB: 14485
DILOCO_MERGES: 91
DILOCO_K: 2
DILOCO_SYNC_TOTAL_S: 98.207
DILOCO_SYNC_AVG_MS: 1079.2
```

## Recommendation

Hierarchical DiLoCo is ready for the next small scaleout gate, but not yet for
32/64/128/256-node production-scale jobs without an intermediate 2-node or
8-node hierarchical smoke. Use the passing 1-node smoke as the corrected
process-group evidence, then run a minimal multi-node hierarchical smoke before
larger A/B scaleout. If any larger hierarchical run fails before first merge,
fall back to bucketed global for scaleout until the new failure is classified.
