# Hierarchical DiLoCo Merge Integration Test

Task: `integrate-and-test`
Date: 2026-06-27

## Integration status

Accepted. The prototype from `origin/wg/agent-353/design-hierarchical-diloco`
was already present on `origin/main` at the start of this task:

```bash
git fetch origin
git show --stat --oneline --decorate 7b39739
git show --stat --oneline --decorate origin/main
git diff --stat 7b39739..origin/main
git diff --name-status 7b39739..origin/main
git merge-base origin/main 7b39739
```

Result:

- `7b39739` is the prototype commit.
- `origin/main` was `5662da9 feat: design-hierarchical-diloco (agent-353)`.
- `git diff --stat 7b39739..origin/main` and
  `git diff --name-status 7b39739..origin/main` were empty.
- Merge base was `637d5ecc7030700298b49dddad3d8c74381f391c`, so `5662da9`
  is an exact cherry-pick/equivalent integration of the prototype onto main.
- No force push was used. No extra code merge was needed because `origin/main`
  already contained the accepted content.

## Review notes

Reviewed files:

- `train.py`
- `tests/test_diloco_merge.py`
- `tests/test_diloco_hierarchical_math.py`
- `scripts/frontier/diloco_scaleout_readiness.sbatch`

Correctness review:

- Exact averaging: `_diloco_hierarchical_sum_average_flat` uses
  `dist.reduce(... SUM ...)` inside each first-level group, root-only
  `dist.all_reduce(... SUM ...)` across roots, one final `flat.div_(world_size)`,
  then group broadcast. This preserves the same all-rank arithmetic mean as the
  global SUM/divide path, including unequal final groups. It does not average
  local averages.
- Rank topology: `_build_diloco_hierarchical_merge_groups` creates consecutive
  rank groups of `--diloco_merge_group_size`, matching Frontier's 8 ranks per
  node layout when the default group size is 8. The final group can be smaller;
  correctness is preserved because the implementation sums tensors and divides
  by `world_size`.
- Process-group safety: all ranks call `dist.new_group` for every first-level
  group in identical order. All ranks also call `dist.new_group(ranks=root_ranks)`
  for the root group when there is more than one root. In training setup, subgroup
  warmup is serialized by default-process-group barriers between local group
  warmups, then root group warmup, then another barrier before training proceeds.
- Bucket behavior: `_diloco_allreduce_average_flat` applies the same topology
  choice per bucket. The hierarchical path keeps exact SUM/divide semantics for
  monolithic and bucketed tensors. The ScheduleFree path routes both `sf_x` and
  `sf_z` tensor state through the same averaging helper and intentionally leaves
  scalar clocks local, matching the existing DiLoCo design.
- Launcher wiring: `scripts/frontier/diloco_scaleout_readiness.sbatch` now wires
  `DILOCO_MERGE_TOPOLOGY`, `DILOCO_MERGE_GROUP_SIZE`, and
  `DILOCO_MERGE_BUCKET_NUMEL` into `train.py` and records those values in the
  manifest. Note: debug flags are not wired in this template; use the parser envs
  `NDM_DILOCO_MERGE_DEBUG` and `NDM_DILOCO_MERGE_DEBUG_RANKS` when needed.

No blocking correctness issue was found in the hierarchical merge code.

## Local validation

Environment:

```bash
module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import sys
print(sys.executable)
for name in ('torch', 'pytest', 'schedulefree', 'tiktoken'):
    try:
        mod = __import__(name)
        print(name, getattr(mod, '__version__', 'ok'))
    except Exception as e:
        print(name, 'FAILED', repr(e))
PY
```

Result:

- Python: `/autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/bin/python`
- `torch 2.8.0.dev20250422+rocm6.4`
- `pytest 9.1.1`
- `schedulefree ok`
- `tiktoken 0.13.0`

Commands and results:

```bash
python -m py_compile train.py tests/test_diloco_merge.py tests/test_diloco_hierarchical_math.py
```

Passed.

```bash
python tests/test_diloco_hierarchical_math.py
```

Passed: `Ran 2 tests in 0.000s`, `OK`.

```bash
python -m pytest \
  tests/test_diloco_merge.py::test_bucketed_non_schedulefree_merge_preserves_sum_then_divide_semantics \
  tests/test_diloco_merge.py::test_bucketed_schedulefree_merge_avoids_scalar_collectives \
  tests/test_diloco_merge.py::test_hierarchical_merge_matches_global_average_for_unequal_groups \
  -q -s
```

Passed: `3 passed in 59.78s`.

```bash
bash -n scripts/frontier/diloco_scaleout_readiness.sbatch
```

Passed.

```bash
git diff --check
```

Passed.

## Slurm smoke

Goal: smallest useful hierarchical merge smoke, not a 32/64/128/256-node scale
job. The selected shape is 1 node, 8 ranks, `DILOCO_MERGE_GROUP_SIZE=4`, which
creates two first-level rank groups and a two-root root group on one node. This
exercises process group setup plus hierarchical merge entry/exit without using
the step-383500 E97 checkpoint.

Common corrected submission environment:

```bash
export REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-356
export WG_TASK_ID=integrate-and-test
export OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
export GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2
export SCALEOUT_VARIANT=gdn2-MLP
export SCALEOUT_NODES=1
export SCALEOUT_WALLTIME=00:15:00
export TRAIN_MINUTES=3
export HUMAN_APPROVAL_RECORD='WG integrate-and-test minimal hierarchical merge smoke, 2026-06-27; 1-node debug QOS only, no scaleout'
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

Attempts:

- `4908284`: failed in 3 seconds before Python. Cause:
  `mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied`.
  Fix: set explicit bif148 `OUTPUT_ROOT`.
- `4908285`: reached distributed init, then failed on GDN2 checkout path. Cause:
  `GDN-2 checkout not found at /lustre/orion/scratch/erikgarrison/emender/src/GatedDeltaNet-2`.
  Fix: set explicit `GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2`.
- `4908288`: corrected regular-QOS submission; cancelled while pending so the
  debug-QOS submission could be the authoritative smoke.
- `4908293`: corrected debug-QOS submission. Monitored with:

```bash
squeue -j 4908293 -o '%i %T %M %D %R'
sacct -j 4908293 --format=JobID,State,Elapsed,ExitCode,NodeList%30
```

State at report time: `PENDING`, reason `Priority`, no node assigned yet.

The submitted smoke therefore satisfies the submit-and-monitor requirement, but
it has not yet produced pass/fail runtime evidence. The job should be checked
before authorizing the 32-node A/B:

```bash
sacct -j 4908293 --format=JobID,JobName%30,Partition,QOS,State,ExitCode,Elapsed,NNodes,NodeList%30
tail -n 200 logs/frontier/scaleout/emender-hier-merge-smoke-4908293.out
tail -n 200 logs/frontier/scaleout/emender-hier-merge-smoke-4908293.err
```

Expected pass evidence:

- `[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, 4] roots=[0, 4]`
- rank-filtered `[DiLoCoMergeDebug]` enter/exit lines on ranks `0` and `4`
- at least one completed merge at `DILOCO_K=2`
- finite training loss after the merge
- Slurm state `COMPLETED`, exit `0:0`

## Recommendation

Do not start the 32-node hierarchical A/B until job `4908293` completes cleanly
or an equivalent 1-node/2-node hierarchical smoke records merge entry/exit and
finite post-merge training. The code-level integration is acceptable and local
tests passed, but the required Frontier runtime smoke was still pending when this
report was written.

If `4908293` passes, the next 32-node A/B should compare bucketed global versus
hierarchical with identical model/checkpoint/training settings and only the merge
topology changed:

- arm A: `DILOCO_MERGE_TOPOLOGY=global`,
  `DILOCO_MERGE_BUCKET_NUMEL=1048576`
- arm B: `DILOCO_MERGE_TOPOLOGY=hierarchical`,
  `DILOCO_MERGE_GROUP_SIZE=8`,
  `DILOCO_MERGE_BUCKET_NUMEL=1048576`
- keep `DILOCO_OUTER_OPTIMIZER=avg`, `DILOCO_OUTER_LR=1.0`,
  `DILOCO_OUTER_BETA=0.0`, and the existing singleton-island K20/K160 recipe
  unchanged unless the owning scaleout task explicitly chooses otherwise.
- enable merge debug on a sparse rank set that includes global rank 0 and at
  least one group root per node subset, using `NDM_DILOCO_MERGE_DEBUG=1` and
  `NDM_DILOCO_MERGE_DEBUG_RANKS=0,8,16,24` or the corresponding CLI flags in a
  launcher that wires them.

The 32-node A/B should gate on matching loss sanity, no NCCL/RCCL errors, no
stalled merge bucket, and lower or equal merge wall-clock for hierarchical versus
bucketed global.
