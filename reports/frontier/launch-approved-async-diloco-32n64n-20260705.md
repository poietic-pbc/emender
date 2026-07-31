# Async DiLoCo E97 32/64-Node Config Launch

Task: `launch-approved-async-diloco-32n64n`
Date: 2026-07-05
Decision: `pass-with-wrapper-fix`

## Approval Reconfirmation

`approve-async-diloco` is `done`. Its WG log at
`2026-07-05T15:02:27Z` records human approval for:

- one 32-node job and one 64-node job;
- each job walltime `<=00:30:00`;
- total primary requested cap `48` node-hours;
- non-production run root
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705`;
- no production latest pointer advancement;
- no 128-node, 256-node, production, GDN2, or model-only job.

The approval artifact is in upstream commit `9ec8e1f` as
`reports/frontier/approve-async-diloco-20260705.md`. That commit is not merged
into this worktree branch, so this report references it directly.

## Runner Pattern And Fix

The launch used the successful async DiLoCo E97 2/8-node wrapper pattern from
commit `603d48e`:

- `scripts/frontier/async_diloco_e97_2n8n_debug.py`
- `scripts/frontier/async_diloco_e97_2n8n_debug.sbatch`

This task imported that pattern and extended it with:

- recovery checkpoint cadence knobs, defaulting to one recovery checkpoint every
  DiLoCo generation;
- explicit checkpoint duration, total size, and percent-overhead metrics;
- explicit loss moving-average export in the job-level JSON;
- `--reuse-representative-node`, which materializes the E97-sized tensor update
  once and clones lightweight per-node quorum/status metrics for 32/64-node
  config tests.

The first 32-node submission found the issue this mode fixes:

- Slurm job `4944217`
- Command shape: 32 nodes, quorum 30, recovery every generation, resume check,
  non-production output root.
- State: `OUT_OF_MEMORY`, exit `0:125`, elapsed `00:01:40`.
- Logs:
  `logs/frontier/async_diloco_e97/async-diloco-e97-32n-4944217.out`,
  `logs/frontier/async_diloco_e97/async-diloco-e97-32n-4944217.err`
- Actual node-hours: about `0.889`.
- Metrics path: none written.
- Diagnosis: wrapper/infrastructure OOM from materializing full E97 node-update
  tensor state per simulated node in one Python process.

The retry stayed within the approved cap. Requested accounting including the
failed 32-node attempt, fixed 32-node retry, and 64-node job is
`10.666667 + 10.666667 + 21.333333 = 42.666667` node-hours, below the
approved `48` node-hour cap.

## 32-Node Result

- Slurm job ID: `4944228`
- State: `COMPLETED`, exit `0:0`
- Slurm elapsed: `00:00:50`
- Requested walltime/node-hours: `00:20:00`, `10.666667`
- Actual node-hours: about `0.444`
- Stdout: `logs/frontier/async_diloco_e97/async-diloco-e97-32n-r1-4944228.out`
- Stderr: `logs/frontier/async_diloco_e97/async-diloco-e97-32n-r1-4944228.err`
- Metrics JSON:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944228-20260705T200009Z/artifacts/async_diloco_e97_32n_metrics.json`
- Command file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944228-20260705T200009Z/artifacts/command.txt`
- Pass/no-go: `pass`

Launch command summary:

```text
REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-640 \
WG_TASK_ID=launch-approved-async-diloco-32n64n \
TRAINING_TARGET=E97_1.3B_step483000_32n64n_config_20260705 \
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705 \
ASYNC_GLOBAL_QUORUM=30 \
ASYNC_GLOBAL_DROP_NODE_IDS=30,31 \
ASYNC_RESUME_CHECK=1 \
ASYNC_RECOVERY_EVERY_GENERATIONS=1 \
ASYNC_REUSE_REPRESENTATIVE_NODE=1 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=10.666667 \
sbatch --parsable -N 32 -J async-diloco-e97-32n-r1 --time=00:20:00 --export=ALL scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Key metrics:

| Field | Value |
| --- | --- |
| Configured quorum | local `8`, global `30` of `32` |
| Effective local quorum distribution | min `8`, p50 `8`, p95 `8`, max `8` |
| Effective global quorum | `30` |
| Local updates | accepted `256`, stale `0`, timed out `0`, failed `0`, invalid `0` |
| Global updates | accepted `30`, stale `0`, timed out `2`, failed `0`, invalid `0` |
| Tokens/sec | local node0 `6665.164846`, global `199954.945384`, aggregate `16548.892425` |
| Loss moving average | local node0 `loss_100=0.990000`, global `loss_100=0.990000` |
| Generation/global merge/rebase/checkpoint | `1.229077s` / `5.958014s` / `0.0s` / `0.000167s` |
| Recovery checkpoint size/overhead | `16764` total bytes, `0.013618%` |
| Resume-from-latest | selected generation `0`, published generation `1`, latest advanced `true` |
| Production latest changed | `false` |

Generation manifests and recovery records:

- `.../async_run/generations/gen_000000/manifest.json`
- `.../async_run/recovery_checkpoints/gen_000000/initial.json`
- resume published `.../async_run/generations/gen_000001/manifest.json`

## 64-Node Result

- Slurm job ID: `4944237`
- State: `COMPLETED`, exit `0:0`
- Slurm elapsed: `00:00:55`
- Requested walltime/node-hours: `00:20:00`, `21.333333`
- Actual node-hours: about `0.978`
- Stdout: `logs/frontier/async_diloco_e97/async-diloco-e97-64n-4944237.out`
- Stderr: `logs/frontier/async_diloco_e97/async-diloco-e97-64n-4944237.err`
- Metrics JSON:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944237-20260705T201032Z/artifacts/async_diloco_e97_64n_metrics.json`
- Command file:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705/20260705/4944237-20260705T201032Z/artifacts/command.txt`
- Pass/no-go: `pass`

Launch command summary:

```text
REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-640 \
WG_TASK_ID=launch-approved-async-diloco-32n64n \
TRAINING_TARGET=E97_1.3B_step483000_32n64n_config_20260705 \
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step483000_32n64n_config_20260705 \
ASYNC_GLOBAL_QUORUM=60 \
ASYNC_GLOBAL_DROP_NODE_IDS=60,61,62,63 \
ASYNC_RESUME_CHECK=1 \
ASYNC_RECOVERY_EVERY_GENERATIONS=1 \
ASYNC_REUSE_REPRESENTATIVE_NODE=1 \
REQUESTED_WALLTIME=00:20:00 \
REQUESTED_NODE_HOURS=21.333333 \
sbatch --parsable -N 64 -J async-diloco-e97-64n --time=00:20:00 --export=ALL scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Key metrics:

| Field | Value |
| --- | --- |
| Configured quorum | local `8`, global `60` of `64` |
| Effective local quorum distribution | min `8`, p50 `8`, p95 `8`, max `8` |
| Effective global quorum | `60` |
| Local updates | accepted `512`, stale `0`, timed out `0`, failed `0`, invalid `0` |
| Global updates | accepted `60`, stale `0`, timed out `4`, failed `0`, invalid `0` |
| Tokens/sec | local node0 `6684.599023`, global `401075.941355`, aggregate `21326.953058` |
| Loss moving average | local node0 `loss_100=0.990000`, global `loss_100=0.990000` |
| Generation/global merge/rebase/checkpoint | `1.225504s` / `11.033335s` / `0.0s` / `0.000170s` |
| Recovery checkpoint size/overhead | `16779` total bytes, `0.013843%` |
| Resume-from-latest | selected generation `0`, published generation `1`, latest advanced `true` |
| Production latest changed | `false` |

Generation manifests and recovery records:

- `.../async_run/generations/gen_000000/manifest.json`
- `.../async_run/recovery_checkpoints/gen_000000/initial.json`
- resume published `.../async_run/generations/gen_000001/manifest.json`

## Production Latest Guard

Both successful jobs used production latest guard path:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

Both metrics artifacts report:

```json
{"production_latest_guard": {"changed": false}}
```

The E97 source checkpoint identity was also unchanged before/after in both
successful metrics artifacts.

## Recommendation

Proceed to 128/256 preparation only as a preparation task, not an immediate
launch. The 32/64-node config jobs prove the async quorum metrics, latest guard,
resume path, generation manifests, induced missing-update accounting, and
scale-shaped global merge timing under the wrapper, but they do not prove a
real multi-process training transport path or full production checkpoint I/O.

Recommended next prep recipe:

- Keep E97 as the main arm.
- Keep local quorum `8/8`.
- Use global quorum around `94%` of nodes for debug and config tests:
  `30/32`, `60/64`, candidate `120/128`, candidate `240/256`.
- Keep timeout conservative at `900s` until real transport timing is measured;
  reduce only after live worker update latency is known.
- Keep recovery cadence expressed as `N generations or wall-clock interval,
  whichever fires first`. Candidate prep value: recovery every `1-5`
  generations and/or `5-10` minutes for 256-node B4/K40, with export checkpoint
  about hourly.
- Do not inherit a fixed 20-30 minute recovery interval. The measured prototype
  metadata overhead is tiny, but production checkpoint payload size and I/O
  duration remain the real risk.
- Before any 128/256 submission, replace representative-node reuse with a true
  multi-process or multi-node worker path, or explicitly label the next job as a
  wrapper-only shape test. The representative mode is correct for avoiding
  single-process OOM in this config launcher, but it is not a substitute for
  transport and per-node memory validation.

No 128-node, 256-node, production, GDN2, or model-only job was submitted by this
task.
