# Fix B4 Triton Compile Race and Rerun 64n Scan

Task: `fix-b4-triton`  
Report time: `2026-06-28T13:29:00Z` (`2026-06-28 09:29 EDT`)

## Decision

Do not launch a 24h replacement from the failed `4910703` evidence. Job
`4910703` failed during first-backward Triton compilation before any rank-0
training metric and before any DiLoCo merge. This is a startup/compile-cache
failure, not a `BATCH_SIZE=4` learning verdict.

The minimal launch fix is env-gated per-rank Triton cache isolation in the
shared E97 canary runner. Existing behavior is unchanged by default
(`FRONTIER_PER_RANK_TRITON_CACHE=0`), while the B4 scan and smoke wrappers opt
in with `FRONTIER_PER_RANK_TRITON_CACHE=1`.

After fixing the launch path, a one-node `BATCH_SIZE=4` resume smoke reached
finite post-warmup training metrics and multiple DiLoCo merges, so the bounded
64-node K10/K20 scan was resubmitted through the fixed wrappers. The 64-node
jobs are pending at report time; no 24h replacement recommendation should be
made until those jobs produce clean runtime evidence.

## Prior Job State

### K20 superseded job 4910704

Validation command:

```bash
squeue -j 4910704 -o '%i %T %M %D %R %j'
sacct -j 4910704 --format=JobID,JobName%28,State,ExitCode,Elapsed,Submit,Start,End,NNodes -P
```

Result:

| Job | State | Exit | Elapsed | Submit | Start | End | Nodes |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `4910704` / `e97-s483-b4-k20-64n` | `CANCELLED by 19032` | `0:0` | `00:00:00` | `2026-06-28T06:00:22` | `None` | `2026-06-28T09:13:00` | 64 |

`squeue` returned no row for `4910704`, confirming it is not pending or running.

### Failed K10 job 4910703

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k10_64n_hier_g4_bucket64m_avg_scan/4910703-20260628T112102Z
```

Logs:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-411/logs/frontier/scaleout/e97-s483-b4-k10-64n-4910703.out
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-411/logs/frontier/scaleout/e97-s483-b4-k10-64n-4910703.err
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k10_64n_hier_g4_bucket64m_avg_scan/4910703-20260628T112102Z/logs/train.log
```

Manifest facts:

| Field | Value |
| --- | --- |
| Job | `4910703` / `e97-s483-b4-k10-64n` |
| Nodes/ranks | 64 nodes / 512 ranks |
| Recipe | `BATCH_SIZE=4`, `DILOCO_K=10`, hierarchical group size 4, bucket `67108864` |
| Checkpoint | `/lustre/orion/bif148/proj-shared/emender/checkpoints/E97_1.3B_20260623_103742_step_483000/checkpoint_step_483000_loss_2.5431.pt` |
| Start/end | `2026-06-28T11:21:05Z` to `2026-06-28T11:25:11Z` |
| Exit | `143` |

Primary failure:

```text
[rank186]: FileNotFoundError: [Errno 2] No such file or directory
```

Rank 186 failed inside `torch/triton` compilation during the first backward:

```text
train.py: loss.backward()
ndm/triton/e88_triton_backward.py: e88_triton_backward(...)
triton/compiler/compiler.py: CompiledKernel(...)
pathlib.py: read_text()
FileNotFoundError
```

Secondary failures:

```text
NCCL error: remote process exited or there was a network error
srun: error: ... tasks ... Killed
```

These are downstream symptoms after one or more ranks exited during compile.
They are not the primary diagnosis.

Training evidence:

| Metric | Observation |
| --- | --- |
| Rank-0 step/loss lines | none |
| `global_tok/s` | none |
| DiLoCo merge rows | none |
| Checkpoint after resume | none |

The summary only contains setup/resume lines and process teardown. Therefore
`4910703` gives no evidence about B4 loss quality or K10 vs K20 learning
behavior.

## Fix

Changed `scripts/frontier/e97_1p3b_pretrained_canary.sbatch`:

- Adds `FRONTIER_PER_RANK_TRITON_CACHE`, default `0`.
- Adds `FRONTIER_TRITON_CACHE_BASE`, defaulting after `RUN_ROOT` is known to
  `${RUN_ROOT}/artifacts/triton-cache`.
- When enabled inside the `srun` worker shell, each rank sets:

```bash
TRITON_CACHE_DIR="${FRONTIER_TRITON_CACHE_BASE%/}/rank-${SLURM_PROCID}"
```

- Creates the rank cache directory before launching `train.py`.
- Records the cache setting in `artifacts/env.txt` and `artifacts/manifest.json`.

Changed B4 wrappers:

```text
scripts/frontier/e97_1p3b_step483000_b4_k10_64n_scan.sbatch
scripts/frontier/e97_1p3b_step483000_b4_k20_64n_scan.sbatch
```

Both wrappers now opt in:

```bash
export FRONTIER_PER_RANK_TRITON_CACHE=${FRONTIER_PER_RANK_TRITON_CACHE:-1}
```

Added cheap validation wrapper:

```text
scripts/frontier/e97_1p3b_step483000_b4_1n_triton_smoke.sbatch
```

This uses the same step-483000 checkpoint and the same B4 settings, but only
one node in debug QoS.

## Validation

Syntax:

```bash
bash -n scripts/frontier/e97_1p3b_pretrained_canary.sbatch \
  scripts/frontier/e97_1p3b_step483000_b4_1n_triton_smoke.sbatch \
  scripts/frontier/e97_1p3b_step483000_b4_k10_64n_scan.sbatch \
  scripts/frontier/e97_1p3b_step483000_b4_k20_64n_scan.sbatch
```

Result: passed.

An initial smoke submission, `4910750`, failed in 3 seconds before training
because the first patch referenced `RUN_ROOT` before assignment. That launch
ordering bug was fixed, syntax checks were rerun, and the smoke was resubmitted.

Passing startup smoke:

| Field | Value |
| --- | --- |
| Job | `4910752` / `e97-s483-b4-1n-triton-smoke` |
| Submit command | `sbatch --network=disable_rdzv_get --export=ALL,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-420 scripts/frontier/e97_1p3b_step483000_b4_1n_triton_smoke.sbatch` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_1n_triton_cache_smoke/4910752-20260628T131846Z` |
| Slurm stdout | `logs/frontier/scaleout/e97-s483-b4-1n-triton-smoke-4910752.out` |
| Slurm stderr | `logs/frontier/scaleout/e97-s483-b4-1n-triton-smoke-4910752.err` |
| Shared train log | `.../4910752-20260628T131846Z/logs/train.log` |
| Final scheduler state | `COMPLETED`, exit `0:0`, elapsed `00:09:22` |
| Cache mode | `FRONTIER_PER_RANK_TRITON_CACHE=1` |
| Cache base | `.../4910752-20260628T131846Z/artifacts/triton-cache` |

Finite startup metrics observed after compile warmup:

```text
warmup 1/1 | loss 2.7914
step 483001 | loss 1.8902 | lr 1.01e-03 | grad 0.89 | tok/s 4727 | global_tok/s 37815
step 483010 | loss 2.9277 | lr 1.01e-03 | grad 0.84 | tok/s 4590 | global_tok/s 36724
>>> [DiLoCo] merge #1 at step 483010: averaged model weights across 8 ranks in 197 ms
step 483020 | loss 1.8644 | lr 1.01e-03 | grad 0.73 | tok/s 4674 | global_tok/s 37392
>>> [DiLoCo] merge #2 at step 483020: averaged model weights across 8 ranks in 187 ms
```

Final smoke summary:

| Metric | Value |
| --- | --- |
| Train status | `0` |
| Last reported step | `483199` |
| Last reported loss | `2.4099` at step `483199`; final checkpoint loss `2.5615` |
| Last steady `global_tok/s` window | mostly `41k-42k` outside checkpoint/merge steps |
| DiLoCo merges | `20` |
| DiLoCo sync total / average | `3.677 s` / `183.9 ms` |
| Final checkpoint | `.../checkpoint_step_483199_loss_2.5615.pt`; `latest.pt` updated |
| Severe errors | no `Traceback`, `FileNotFoundError`, OOM, NaN, or non-finite lines found in the summary |

This satisfies the cheap gate: B4 startup completed with finite training
metrics, DiLoCo merges, and final checkpointing under isolated Triton caches.

## Resubmitted 64-node Scan

Both jobs were submitted only after the one-node B4 smoke reached finite
training metrics.

Submit commands:

```bash
sbatch --network=disable_rdzv_get --export=ALL,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-420 \
  scripts/frontier/e97_1p3b_step483000_b4_k20_64n_scan.sbatch
sbatch --network=disable_rdzv_get --export=ALL,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-420 \
  scripts/frontier/e97_1p3b_step483000_b4_k10_64n_scan.sbatch
```

Scheduler state at report time:

| Arm | Job | State | Reason | Submit | Backfill start estimate | Nodes | Slurm stdout | Slurm stderr |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| B4 K20 | `4910757` / `e97-s483-b4-k20-64n` | `PENDING` | `Priority` | `2026-06-28T09:22:02` | `2026-06-28T16:12:00` | 64 | `logs/frontier/scaleout/e97-s483-b4-k20-64n-4910757.out` | `logs/frontier/scaleout/e97-s483-b4-k20-64n-4910757.err` |
| B4 K10 | `4910758` / `e97-s483-b4-k10-64n` | `PENDING` | `Priority` | `2026-06-28T09:22:03` | `2026-06-28T23:18:00` | 64 | `logs/frontier/scaleout/e97-s483-b4-k10-64n-4910758.out` | `logs/frontier/scaleout/e97-s483-b4-k10-64n-4910758.err` |

Expected run roots after allocation:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k20_64n_hier_g4_bucket64m_avg_scan/4910757-*
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260628/E97_1.3B_step483000_b4_k10_64n_hier_g4_bucket64m_avg_scan/4910758-*
```

No 64-node runtime metrics exist yet because both jobs are still pending.

## Replacement 24h Recommendation

No replacement 24h recommendation yet.

The valid next decision point is after the fixed 64-node jobs complete or fail.
Harvest, at minimum:

- final Slurm state and exit code for `4910757` and `4910758`;
- `artifacts/env.txt` and `manifest.json`, confirming per-rank Triton cache is
  enabled;
- first and last rank-0 loss/throughput windows;
- DiLoCo merge count and merge durations;
- memory/OOM scan;
- severe error scan for `Traceback`, `FileNotFoundError`, `NCCL`, `RCCL`,
  `timeout`, `nan`, and `non-finite`;
- final checkpoint status.

If both fixed jobs complete cleanly, keep the prior provisional preference for
K20 as the less aggressive lower-merge-frequency B4 production candidate. Choose
K10 only if it is materially cleaner or K20 fails for a reason unrelated to B4
safety.
