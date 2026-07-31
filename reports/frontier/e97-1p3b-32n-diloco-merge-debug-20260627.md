# E97 1.3B 32n DiLoCo Merge Debug Diagnostic

Date: 2026-06-27
Task: `debug-e97-1-3b`

## Code Changes

- Added `train.py` DiLoCo merge controls:
  - `--diloco_merge_bucket_numel` / `NDM_DILOCO_MERGE_BUCKET_NUMEL`
  - `--diloco_merge_debug` / `NDM_DILOCO_MERGE_DEBUG`
  - `--diloco_merge_debug_ranks` / `NDM_DILOCO_MERGE_DEBUG_RANKS`
- Default merge behavior remains monolithic: when bucket size is unset or `0`,
  the code still performs one flat `dist.all_reduce(... SUM)` followed by
  `div_(world_size)` for each merge tensor.
- Bucketed mode applies the same sum-then-divide semantics per flat slice:
  ScheduleFree `sf_x`, `sf_z`, and `sf_y` where applicable; non-ScheduleFree
  parameter averaging uses label `params`.
- Moved CUDA device binding before process-group initialization and then added a
  follow-up `device_id=device` init path with `TypeError` fallback for older
  PyTorch. The diagnostic job below ran before the `device_id` follow-up commit,
  so it still emitted PyTorch's "device used by this process is currently
  unknown" warning; subsequent jobs should exercise the explicit `device_id`
  path.

## Validation

Environment:

```text
module load miniforge3/23.11.0-0
```

Commands:

```text
python -m py_compile train.py tests/test_diloco_merge.py
python -m pytest tests/test_diloco_merge.py -q -s
```

Result:

```text
22 passed in 476.05s (0:07:56)
```

Notes:

- The base `/usr/bin/python3` does not have `torch`, `schedulefree`, or
  `pytest`; validation used the Frontier miniforge module stack.
- One existing bootstrap test compared tensors byte-for-byte across a
  ScheduleFree `eval()`/`train()` round trip. On this ROCm torch stack that
  round trip introduces sub-1e-8 drift. The test now snapshots the live tensor
  state after its intentional mode round trip and compares the independently
  recomputed x-basis anchor with a tight `atol=1e-7`.

## Checkpoint

Preferred S3 checkpoint requested by user:

```text
s3://spinozans/emender/e97-diloco/levelE97_100m_20260623_103742/step_383500/
```

Fetched local directory:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500
```

Files fetched:

```text
E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt
E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt.sha256
args.json
launch_manifest.json
manifest.json
```

Verification:

```text
checkpoint size: 7719673482 bytes
sha256 sidecar: b899489c77093164f92fc35f85c879204d8e1866068b320598a87cf87d99b1ca
local sha256:   b899489c77093164f92fc35f85c879204d8e1866068b320598a87cf87d99b1ca
```

The downloaded `manifest.json` reports five S3 objects and
`remote_sha256_sidecar_matches_local: true`.

## Slurm Jobs

Malformed first submission:

```text
job_id: 4908079
state: CANCELLED by user before merge evidence
reason: Slurm --export split DILOCO_MERGE_DEBUG_RANKS=0,1,255 at commas,
        leaving only rank 0 in the job environment.
stdout: logs/frontier/scaleout/e97-1p3b-k20-merge-debug-32n-4908079.out
stderr: logs/frontier/scaleout/e97-1p3b-k20-merge-debug-32n-4908079.err
```

Corrected diagnostic:

```text
job_id: 4908087
job_name: e97-1p3b-k20-merge-debug-32n
nodes: 32
tasks: 256
state: CANCELLED by user after merge evidence was captured
elapsed: 00:03:58
stdout: logs/frontier/scaleout/e97-1p3b-k20-merge-debug-32n-4908087.out
stderr: logs/frontier/scaleout/e97-1p3b-k20-merge-debug-32n-4908087.err
run_root: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_pretrained_k20_merge_debug_32n_step383500/4908087-20260627T050806Z
train_log: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_pretrained_k20_merge_debug_32n_step383500/4908087-20260627T050806Z/logs/train.log
env_file: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260627/E97_1.3B_pretrained_k20_merge_debug_32n_step383500/4908087-20260627T050806Z/artifacts/env.txt
```

Key settings:

```text
resume_checkpoint=/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_383500/E97_1.3B_20260623_103742_step_383500_checkpoint_step_383500_loss_2.5679.pt
DILOCO_K=20
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_ISLAND_SIZE=1
DILOCO_MERGE_BUCKET_NUMEL=67108864
DILOCO_MERGE_DEBUG=1
DILOCO_MERGE_DEBUG_RANKS=0,1,255
FRONTIER_RCCL_ENV=recommended
FRONTIER_RCCL_ALT_RDZV=1
FI_CXI_RDZV_PROTO=alt_read
```

## Diagnostic Evidence

The corrected diagnostic resumed all ranks from step `383500`. Selected debug
ranks all entered and exited every first-merge bucket:

```text
merge=1 step=383520 rank=0   sf_x: 20 buckets enter+exit; sf_z: 20 buckets enter+exit
merge=1 step=383520 rank=1   sf_x: 20 buckets enter+exit; sf_z: 20 buckets enter+exit
merge=1 step=383520 rank=255 sf_x: 20 buckets enter+exit; sf_z: 20 buckets enter+exit
```

Rank-filtered debug line counts in `train.log`:

```text
rank=0   641 lines
rank=1   640 lines
rank=255 640 lines
```

Merge summaries:

```text
>>> [DiLoCo] merge #1 at step 383520: averaged model weights across 256 ranks in 2662 ms
>>> [DiLoCo] merge #2 at step 383540: averaged model weights across 256 ranks in 2650 ms
>>> [DiLoCo] merge #3 at step 383560: averaged model weights across 256 ranks in 2693 ms
>>> [DiLoCo] merge #4 at step 383580: averaged model weights across 256 ranks in 2713 ms
>>> [DiLoCo] merge #5 at step 383600: averaged model weights across 256 ranks in 2660 ms
>>> [DiLoCo] merge #6 at step 383620: averaged model weights across 256 ranks in 2691 ms
>>> [DiLoCo] merge #7 at step 383640: averaged model weights across 256 ranks in 2729 ms
>>> [DiLoCo] merge #8 at step 383660: averaged model weights across 256 ranks in 2679 ms
```

No watchdog timeout, missing-rank symptom, task-0 kill, traceback, non-finite
guard, or NCCL/RCCL error appeared before the intentional cancellation. The only
error-like line at the end is the expected Slurm cancellation record:

```text
STEP 4908087.0 ... CANCELLED ... DUE to SIGNAL Terminated
```

## Interpretation

This diagnostic supports:

- **Not rank missing before collective** for the bucketed merge path: ranks
  `0`, `1`, and `255` all reached and exited every `sf_x` and `sf_z` bucket at
  the first merge.
- **Not a blanket RCCL/network failure under training context** for bucketed
  model-sized DiLoCo state: the job completed eight 32-node, 256-rank,
  ScheduleFree x/z merge cycles under the same training process context.
- **Monolithic collective-size/context remains the leading hypothesis** for the
  original failure: standalone model-sized allreduce passed, while the failing
  retry timed out on one monolithic `sf_x` allreduce in training; splitting the
  same ScheduleFree state into 64M-element buckets made the 32-node training
  merge progress repeatedly.

This does not prove the monolithic collective is intrinsically invalid; it shows
the failure is avoided by bucketization in the training context. The next
production-scale retry should keep the bucket option enabled until a separate
monolithic control is explicitly requested.

Scale-ladder work should remain paused/blocked until this evidence is reviewed.
