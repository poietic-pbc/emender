# Rerun resilient quorum 256n debug gate

Task: `evaluate-rerun-resilient-quorum-256n-debug-gate`

Date: 2026-07-08

## Decision

Gate decision before submission: **GO for exactly one bounded 256n debug smoke**.

Post-smoke result: **256n debug smoke did not pass**. Do not submit a 256n x 1h,
12h, production-QOS, or production-latest/last-mutating follow-on from this
evidence.

Next fix: diagnose why the 256n TCP resilient-quorum coordinator only accepted
593 of 2048 rank updates before timeout. The next implementation pass should
focus on 256n fan-in/coordinator startup behavior, timeout/catchup semantics, or
a transport shape that preserves run-local latest/checkpoint behavior. Do not
escalate until a bounded 256n debug run advances quorum and writes run-local
latest/checkpoint artifacts.

## Go Criteria Evidence

- Wrapper fix `fix-resilient-quorum-wrapper-cli` is done and its commit
  `579329c` is contained in `origin/main` (`git merge-base --is-ancestor
  579329c origin/main` returned 0).
- Rerun ladder task `rerun-resilient-quorum-1n8n64n-ladder` is done and
  evaluator-passed. Its Slurm jobs completed in `batch/debug`:
  `4956437` 1n, `4956445` 8n, and `4956459` 64n.
- Independent metrics extraction from the ladder artifacts showed resilient
  quorum mode at each rung:

| rung | job | state | accepted / requested | quorum status | catchup | missing | stale | timed out | latest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1n | `4956437` | `COMPLETED 0:0` | 8 / 8 | `advanced` | 0 | 0 | 0 | 0 | advanced |
| 8n | `4956445` | `COMPLETED 0:0` | 64 / 64 | `advanced` | 0 | 0 | 0 | 0 | advanced |
| 64n | `4956459` | `COMPLETED 0:0` | 512 / 512 | `advanced` | 0 | 0 | 0 | 0 | advanced |

- Production latest/last guard remained intact. The ladder task log records
  empty before/after diffs for production latest/last, and a post-smoke chain
  scan still showed only the existing chain pointers under
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains`, including
  `E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt` pointing to
  the 2026-06-30 chain checkpoint. This task used only a debug output root.

## 256n Submission

Exactly one 256-node smoke was submitted:

- Job id: `4956594`
- Job name: `resilient-quorum-256n-debug`
- Partition/QOS: `batch` / `debug`
- Nodes/ranks: 256 nodes, 2048 GPU ranks
- Walltime: `00:20:00`
- Requested node-hours: `85.333333`
- Seed checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Output root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z`
- Command artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/artifacts/command.txt`
- Env artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/artifacts/env.txt`
- Metrics artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/artifacts/metrics.json`
- Manifest artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/artifacts/manifest.json`
- Rank-start artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/artifacts/rank-start.tsv`
- Summary artifact:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708/20260708/E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug/4956594-20260708T112458Z/summaries/summary.md`
- Stdout/stderr:
  `logs/frontier/trainpy_async_quorum/resilient-quorum-256n-debug-4956594.out`,
  `logs/frontier/trainpy_async_quorum/resilient-quorum-256n-debug-4956594.err`

Submit shape:

```text
sbatch --parsable -N 256 -J resilient-quorum-256n-debug -t 00:20:00 -p batch -q debug \
  --export=ALL,WG_TASK_ID=evaluate-rerun-resilient-quorum-256n-debug-gate,...,SMOKE_NAME=256n-resilient-debug,SMOKE_NODE_COUNT=256,SCALEOUT_VARIANT=E97_1.3B_step1065000_resilient_quorum_rerun_256n_debug,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/rerun_resilient_quorum_256n_debug_gate_20260708,ASYNC_QUORUM_TRANSPORT=tcp,ASYNC_TRAINPY_RANKS=2048,ASYNC_EXPECTED_RANKS=2048,ASYNC_GLOBAL_QUORUM=2048,ASYNC_EXPECTED_MISSING_UPDATES=0,ASYNC_TIMEOUT_S=1200,REQUESTED_WALLTIME=00:20:00,REQUESTED_NODE_HOURS=85.333333 \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

## 256n Terminal State

Slurm terminal record:

```text
4956594|resilient-quorum-256n-debug|FAILED|90:0|batch|debug|00:05:48|00:20:00|256|2026-07-08T07:17:50|2026-07-08T07:24:57|2026-07-08T07:30:45
4956594.batch|batch|FAILED|90:0|||00:05:48||1|2026-07-08T07:24:57|2026-07-08T07:24:57|2026-07-08T07:30:45
4956594.extern|extern|COMPLETED|0:0|||00:05:48||256|2026-07-08T07:24:57|2026-07-08T07:24:57|2026-07-08T07:30:45
4956594.0|bash|CANCELLED|0:15|||00:05:30||256|2026-07-08T07:25:14|2026-07-08T07:25:14|2026-07-08T07:30:44
```

The payload manifest reports `exit_status=143`, `validation_status=fail`, and
validation errors:

- `payload_exit_status=143`
- `latest_not_advanced`
- `latest_path_missing=.../async_run/latest.json`
- `checkpoint_paths_missing`

## 256n Resilient Metrics

The 256n run did emit resilient-mode metrics, but did not advance quorum:

| metric | value |
| --- | --- |
| mode | `resilient_quorum` |
| rank starts | 2048 / 2048 |
| accepted updates | 593 |
| requested workers | 2048 |
| participating workers | 593 |
| quorum size / threshold | 593 / 2048 |
| quorum status | `deferred` |
| catchup events | 0 |
| missing / stale / late / timed out / rejected | 0 / 0 / 0 / 1455 / 0 |
| failed / invalid | 0 / 0 |
| loss window | `loss=13.854275003854582`, `loss_100=13.854275003854582` |
| tokens per generation | 76497 |
| TCP bytes | node metadata 5,294,222; payload 5,198,603 |
| run-local latest | not advanced |
| checkpoint paths | 0 |

Metrics excerpt:

```json
{
  "mode": "resilient_quorum",
  "accepted_updates": 593,
  "requested_workers": 2048,
  "participating_workers": 593,
  "quorum_size": 593,
  "quorum_threshold": 2048,
  "quorum_status": "deferred",
  "catchup_events": [],
  "missing_updates": 0,
  "stale_updates": 0,
  "timed_out_updates": 1455,
  "late_updates": 0,
  "rejected_updates": 0,
  "latest_advanced": false,
  "loss_moving_average": {
    "loss": 13.854275003854582,
    "loss_100": 13.854275003854582
  },
  "checkpoint_paths": []
}
```

## Constraint Check

- Exactly one 256n debug job was submitted by this task: `4956594`.
- No 1h, 12h, or production job was submitted.
- No production latest/last mutation was authorized or observed.
- The 256n smoke used run-local output and run-local latest paths only.
- Because the 256n smoke failed quorum advancement and did not write
  run-local latest/checkpoint artifacts, downstream 256n x 1h / 12h /
  production escalation remains **no-go**.
