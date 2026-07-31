# Fix 32n Hierarchical E97 Merge Failure - 2026-06-27

Task: `fix-32n-hierarchical`

## Summary

The 32-node E97 1.3B hierarchical DiLoCo failure is fixed for the tested
step-383500 smoke shape. The corrected retry, Slurm `4908477`, completed
successfully with exit `0:0`, reached 16 K20 hierarchical merges, and wrote a
final consensus checkpoint.

Root cause: hierarchical communicator construction burst. The failed job
`4908467` died after W0 broadcast and before the topology print while the
hierarchical path was constructing and warming 65 extra RCCL/NCCL process
groups for the 256-rank, group-size-4 shape: one 64-rank root communicator plus
64 first-level four-rank communicators. The bucketed global path does not create
these extra communicators and was not changed.

The fix keeps the existing root-first communicator order, but paces
hierarchical process-group construction with default-group barriers after the
root communicator and after each batch of first-level groups. For the 32-node
E97 `group_size=4` shape, the default pacing is one barrier after the root
group and then one barrier after each eight local groups.

## Failed Reproducer

Source report:
`reports/frontier/e97-1p3b-32n-topology-ab-20260627.md`

Failed job:

```text
Slurm job: 4908467
State: FAILED
Exit: 139:0
Topology: hierarchical
Group size: 4
Bucket numel: 67108864
Nodes/ranks: 32 nodes / 256 ranks
Checkpoint: /lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt
RCCL env: recommended
FRONTIER_RCCL_ALT_RDZV: 1
FI_CXI_RDZV_PROTO: alt_read
SLURM_NETWORK: disable_rdzv_get
```

Failed run artifacts from the source report:

```text
run_root: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_ab/4908467-20260627T112626Z
env:      /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_ab/4908467-20260627T112626Z/artifacts/env.txt
manifest: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_ab/4908467-20260627T112626Z/artifacts/manifest.json
summary:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_ab/4908467-20260627T112626Z/summaries/summary.md
train:    /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_ab/4908467-20260627T112626Z/logs/train.log
```

Observed failing window:

```text
[DiLoCo] broadcast rank-0 W_0 to all 256 ranks (identical start)
```

The failed run did not print `Resumed at step 383500`, `Starting training`, or
`[DiLoCo-merge] topology=hierarchical ...`. The terminal log was dominated by
post-kill `ProcessGroupNCCL heartbeatMonitor` / `TCPStore Broken pipe` fallout,
which is consistent with one or more ranks dying during the hierarchical setup
window and the rest of the job being force-terminated.

## Patch

Changed `train.py` only in the opt-in hierarchical setup path:

- Added `--diloco_merge_group_create_barrier_every`, defaulting to
  `NDM_DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY` or `8`.
- `_build_diloco_hierarchical_merge_groups(...)` now accepts
  `create_barrier_every`.
- With pacing enabled, all ranks enter a default-group barrier after the root
  communicator and after each batch of local communicator creations.
- Added rank-0 progress prints before construction and before warmup, so future
  setup failures isolate whether they occur during construction or warmup.

Changed `scripts/frontier/e97_1p3b_pretrained_canary.sbatch` to pass and record
`DILOCO_MERGE_GROUP_CREATE_BARRIER_EVERY` in the command, `env.txt`, and
`manifest.json`.

Changed `tests/test_diloco_merge.py`:

- Preserved the existing root-first order regression test.
- Added `test_hierarchical_group_builder_paces_32n_g4_construction`, which
  exercises the exact 256-rank, group-size-4 construction shape with mocked
  distributed calls and verifies the barrier cadence.

## Corrected 32-Node Smoke

Corrected retry:

```text
Slurm job: 4908477
State: COMPLETED
Exit: 0:0
Elapsed: 00:10:36
Nodes/ranks: 32 nodes / 256 ranks
Topology: hierarchical
Group size: 4
Group-create barrier every: 8
Bucket numel: 67108864
DILOCO_K: 20
Checkpoint source: step 383500
```

Run artifacts:

```text
run_root: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z
env:      /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/artifacts/env.txt
manifest: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/artifacts/manifest.json
summary:  /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/summaries/summary.md
train:    /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/logs/train.log
```

Setup evidence:

```text
[DiLoCo-merge] building hierarchical process groups: group_size=4 n_groups=64 create_barrier_every=8
[DiLoCo-merge] hierarchical process groups built; warming communicators
[DiLoCo-merge] topology=hierarchical group_size=4 groups=[4, ... 4] roots=[0, 4, ... 252] (exact weighted SUM/world_size; bucket_numel=67108864)
```

Merge/checkpoint evidence:

```text
>>> [DiLoCo] merge #1 at step 383520: averaged model weights across 256 ranks in 9002 ms
>>> saved checkpoint: checkpoint_step_383520_loss_2.9109.pt
>>> [DiLoCo] merge #2 at step 383540: averaged model weights across 256 ranks in 8903 ms
>>> saved checkpoint: checkpoint_step_383540_loss_2.3776.pt
>>> [DiLoCo] merge #3 at step 383560: averaged model weights across 256 ranks in 8895 ms
>>> saved checkpoint: checkpoint_step_383560_loss_2.3844.pt
...
>>> [DiLoCo] merge #16 at step 383820: averaged model weights across 256 ranks in 6123 ms
Training complete! Final step: 383820
FINAL_LOSS_LAST100: 2.5905
DILOCO_MERGES: 16
DILOCO_SYNC_TOTAL_S: 128.220
DILOCO_SYNC_AVG_MS: 8013.8
```

Final checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/train/emender_E97_1.3B_20260627_074730/checkpoint_step_383820_loss_2.5905.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_step383500_k20_32n_hier_g4_bucket64m_fix/4908477-20260627T114602Z/train/emender_E97_1.3B_20260627_074730/latest.pt
```

## Validation

Local validation:

```text
bash -n scripts/frontier/e97_1p3b_pretrained_canary.sbatch
python3 -m py_compile train.py tests/test_diloco_merge.py tests/test_diloco_hierarchical_math.py
python -m pytest tests/test_diloco_merge.py::test_hierarchical_group_builder_creates_root_group_before_local_groups tests/test_diloco_merge.py::test_hierarchical_group_builder_paces_32n_g4_construction -q
```

Focused pytest result in the Frontier miniforge base environment:

```text
2 passed in 19.29s
```

Production smoke validation:

```text
4908477 e97-32n-hier-fix COMPLETED 0:0 elapsed=00:10:36
```

Bucketed global path remains unchanged in merge execution: the patch changes
only the hierarchical group-construction helper and hierarchical setup call
site. The global branch in `_diloco_allreduce_average_flat(...)` still performs
the same `dist.all_reduce(SUM)` followed by `flat.div_(world_size)`.
