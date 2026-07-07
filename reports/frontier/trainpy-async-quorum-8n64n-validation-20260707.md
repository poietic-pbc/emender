# train.py async quorum 8n/64n validation ladder, 2026-07-07

Task: `validate-train-py`  
Decision: **NO-GO for larger Frontier E97 1.3B scaleout from this ladder**

This run used the train.py-native async quorum launcher added by
`add-train-py`, after its 1n/2n smokes passed. The ladder did:

1. 8-node debug job `4952095`: **pass**, sufficient to try 64n.
2. 64-node scale check job `4952123`: **no-go**, quorum did not advance.
3. Larger Frontier candidate: **not submitted**, because 64n did not meet the
   documented checkpoint/latest behavior gate.

No production `latest.pt` path was authorized or mutated by this task. Both jobs
used run-local async roots under
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/...`.

## Gate and launch sequence

The 8n launch was submitted from the existing 2n wrapper with explicit Slurm and
async overrides:

```bash
sbatch -N 8 -J trainpy-async-quorum-8n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=validate-train-py,SMOKE_NAME=8n-debug,SMOKE_NODE_COUNT=8,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_async_quorum_8n_debug,ASYNC_EXPECTED_MISSING_UPDATES=1,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=65,ASYNC_GLOBAL_QUORUM=64,ASYNC_TIMEOUT_S=120,HUMAN_APPROVAL_RECORD='WG validate-train-py: 8-node train.py-native async quorum debug after 1n/2n pass; run-local checkpoint publication only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

8n passed the local go gate:

- Slurm state: `COMPLETED`, exit `0:0`, elapsed `00:02:06`.
- Rank starts: `64/64`.
- Quorum: `64/65`, advanced with one intentionally missing expected update.
- Accepted/timed-out updates: `64/1`.
- Latest advanced: `true`.
- Checkpoint/recovery/export paths were written.
- No forbidden DDP or per-step all-reduce log lines were found.

Only after that 8n pass, the 64n scale check was submitted:

```bash
sbatch -N 64 -J trainpy-async-quorum-64n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=validate-train-py,SMOKE_NAME=64n-scale,SMOKE_NODE_COUNT=64,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_async_quorum_64n_scale,ASYNC_EXPECTED_MISSING_UPDATES=4,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=516,ASYNC_GLOBAL_QUORUM=512,ASYNC_TIMEOUT_S=180,HUMAN_APPROVAL_RECORD='WG validate-train-py: 64-node train.py-native async quorum scale check after 8n job 4952095 validation pass; compare against prior train.py synchronous/harness baselines; run-local latest only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

The 64n job ran the payload, but the wrapper correctly failed post-run
validation with exit `90`.

## 8n debug result

Slurm accounting:

```text
4952095|trainpy-async-quorum-8n|COMPLETED|0:0|00:02:06|8|2026-07-07T10:32:58|2026-07-07T10:33:32|2026-07-07T10:35:38
4952095.0|bash|COMPLETED|0:0|00:01:44|8|2026-07-07T10:33:54|2026-07-07T10:33:54|2026-07-07T10:35:38
```

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z
```

Primary artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z/artifacts/metrics.json`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z/artifacts/rank-start.tsv`
- Slurm logs:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-8n-4952095.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-8n-4952095.err`

Quorum and checkpoint metrics:

| Field | Value |
| --- | ---: |
| Launched rank starts | `64/64` |
| Requested workers | `65` |
| Quorum threshold | `64` |
| Accepted updates | `64` |
| Timed-out updates | `1` |
| Quorum status | `advanced` |
| Latest generation | `0` |
| Latest advanced | `true` |
| Tokens per generation | `8,256` |
| Generation duration | `60.5660 s` |
| Tokens/sec | `136.314` |
| Checkpoint duration | `0.000265 s` |
| Forbidden DDP/per-step all-reduce lines | `0` |

Per-rank progress and loss window:

- `rank-start.tsv` has `64` entries, one per launched GPU rank.
- `async_run/node_updates/` has `64` node update JSON files.
- `async_run/progress/` has `64` heartbeat JSON files: `63` rank heartbeats
  at `node_update_written` plus one coordinator heartbeat at
  `coordinator_finalized`.
- Per-rank node update elapsed time: min `41.569 s`, median `43.929 s`, max
  `46.100 s`.
- Per-rank token count: min `129`, max `129`.
- One-step loss window across 64 ranks: min `10.4213`, median `13.9097`, max
  `15.3066`; global moving loss/loss_100 was `13.8358`.
- Example ranks 0-7 losses:
  `13.3013, 14.1870, 14.8774, 12.6921, 13.9219, 12.6809, 13.3225, 14.0806`.

Latest and recovery behavior:

- Run-local latest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_8n_debug/4952095-20260707T143335Z/async_run/latest.json`
- Generation manifest:
  `async_run/generations/gen_000000/manifest.json`
- Recovery checkpoint:
  `async_run/recovery_checkpoints/gen_000000/initial.json`
- Export checkpoint:
  `async_run/export_checkpoints/gen_000000/initial.json`
- Walltime finalization record:
  `async_run/recovery_checkpoints/gen_000000/walltime_finalization.json`

Interpretation: 8n validates the file-quorum control path, rank start coverage,
intentional missing-update recovery, run-local latest advancement, and
checkpoint/recovery record publication.

## 64n scale-check result

Slurm accounting:

```text
4952123|trainpy-async-quorum-64n|FAILED|90:0|00:05:48|64|2026-07-07T10:37:34|2026-07-07T10:37:42|2026-07-07T10:43:30
4952123.0|bash|COMPLETED|0:0|00:05:27|64|2026-07-07T10:38:06|2026-07-07T10:38:06|2026-07-07T10:43:33
```

The payload step completed, but wrapper validation failed because the async
coordinator did not reach quorum and therefore did not publish latest/checkpoint
paths.

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z
```

Primary artifacts:

- Summary:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z/summaries/summary.md`
- Manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z/artifacts/manifest.json`
- Metrics:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z/artifacts/metrics.json`
- Rank starts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z/artifacts/rank-start.tsv`
- Slurm logs:
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-64n-4952123.out`,
  `logs/frontier/trainpy_async_quorum/trainpy-async-quorum-64n-4952123.err`

64n metrics:

| Field | Value |
| --- | ---: |
| Launched rank starts | `512/512` |
| Requested workers | `516` |
| Quorum threshold | `512` |
| Accepted updates before timeout | `400` |
| Timed-out updates | `116` |
| Quorum status | `deferred` |
| Latest generation | `-1` |
| Latest advanced | `false` |
| Checkpoint paths | `0` |
| Tokens per generation | `51,600` |
| Generation duration | `180.2032 s` |
| Tokens/sec | `286.343` |
| Forbidden DDP/per-step all-reduce lines | `0` |

Wrapper validation errors:

```text
latest_not_advanced
latest_path_missing=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_64n_scale/4952123-20260707T143746Z/async_run/latest.json
checkpoint_paths_missing
```

Per-rank progress and loss window:

- `rank-start.tsv` has `512` entries, one per launched GPU rank.
- `async_run/node_updates/` contains `512` node update JSON files, so training
  work completed broadly across ranks.
- `async_run/progress/` contains `512` heartbeat JSON files: `511` rank
  heartbeats at `node_update_written` plus one coordinator heartbeat at
  `coordinator_deferred`.
- Per-rank node update elapsed time: min `41.700 s`, median `46.538 s`, max
  `59.659 s`.
- Per-rank token count: min `129`, max `129`.
- One-step loss window across 512 ranks: min `10.4213`, median `13.8540`, max
  `15.4937`; global moving loss/loss_100 was `13.8319`.

Interpretation: this is not a Slurm allocation failure and not a DDP/per-step
all-reduce regression. It is a quorum/coordinator collection failure at 64n:
the job launched all ranks and wrote all node updates, but the coordinator only
accepted `400/512` required updates before the 180-second timeout and correctly
refused latest/checkpoint advancement.

## Baseline comparison

The closest synchronous train.py DiLoCo baseline in this repository is the
8-node B4/K40 debug-QOS run `4951457`, reported in
`reports/frontier/trainpy-updated-runtime-8n-smoke-20260707.md`:

- 8 nodes, 64 ranks, synchronous periodic model averaging every K=40 local
  optimizer steps.
- Median throughput: `342,584` global tokens/sec.
- Final loss last100: `2.4949`.
- Seven successful K=40 merges, each about `2.4-3.3 s`.
- Final run-local `latest.pt` advanced to a real checkpoint.

The earlier async harness 64n shape test `4944237`, reported in
`reports/frontier/launch-approved-async-diloco-32n64n-20260705.md`, is not a
train.py-native real-token run, but it is the best matched async-quorum shape
baseline:

- 64 nodes, representative-node async harness mode.
- Global quorum `60/64`, with `4` expected timed-out global updates.
- State: `COMPLETED`, exit `0:0`.
- Effective global updates: accepted `60`, timed out `4`.
- Latest advanced: `true`.
- Global merge duration: `11.033 s`.
- Global tokens/sec: `401,075.9`.
- Production latest changed: `false`.

Compared with those baselines, the train.py-native 64n quorum run is not ready:

| Run | Shape | Quorum result | Latest/checkpoint | Throughput/loss comparability |
| --- | --- | --- | --- | --- |
| `4951457` sync train.py | 8n, B4/K40, real train.py | all ranks synchronized | latest advanced | real loss around `2.49`, median `342,584 tok/s` |
| `4944237` async harness | 64n shape, representative-node | `60/64` accepted | latest advanced | wrapper-shape only, `401,075.9 tok/s` |
| `4952095` train.py async quorum | 8n, one-generation real-token debug | `64/65` advanced | latest advanced | loss window `10.42-15.31`, `136.3 tok/s` for bounded debug |
| `4952123` train.py async quorum | 64n, one-generation real-token scale check | `400/516` deferred | no latest/checkpoint | loss window `10.42-15.49`, `286.3 tok/s`; failed quorum gate |

The one-step async quorum debug jobs use `BATCH_SIZE=1`, `CHUNK_SIZE=128`, and
`ASYNC_LOCAL_STEPS=1`, so their loss and throughput are not directly comparable
to the synchronous B4/chunk2048/K40 training baseline. They are still useful for
checking whether the train.py-native worker path can produce rank updates and
whether the async coordinator can advance/latest-checkpoint at a given node
count. The 64n answer is currently no.

## Larger Frontier candidate decision

Decision: **NO-GO. No larger Frontier candidate was submitted.**

Reason:

- The required 64n scale check did not advance latest.
- No 64n generation manifest/checkpoint/recovery paths were published.
- Accepted updates were `400`, below quorum threshold `512`.
- Timeout behavior was too conservative or coordinator collection did not scale
  sufficiently for the 64n shape.

Because the documented go condition was not met, there is no larger job ID,
node-hour charge, token estimate, or live cancel criterion to report for a
submitted larger run. A future larger candidate should be created only after a
new 64n validation demonstrates latest advancement and recovery/checkpoint
publication. Suggested cancel criteria for that future candidate:

- cancel if accepted updates remain below quorum after the configured timeout;
- cancel if latest/checkpoint paths are not published after generation 0;
- cancel if any production latest guard reports mutation before explicit
  authorization;
- cancel if severe runtime errors appear in Slurm stdout/stderr or the
  trainpy async quorum log.

## Validation checklist

- 8n debug report includes quorum metrics, per-rank progress, loss windows,
  merge latency/latest advancement, and recovery behavior: **met**.
- 64n report compares throughput and loss against synchronous train.py DiLoCo
  and prior async harness baselines where possible: **met**.
- Larger-scale launch submitted only after a documented go decision, with job
  details if submitted: **met by no-go**; no larger launch was submitted because
  the 64n gate failed.
- No production latest path mutated before explicit authorization: **met**; all
  artifacts are run-local debug paths, and no production latest authorization was
  granted.
