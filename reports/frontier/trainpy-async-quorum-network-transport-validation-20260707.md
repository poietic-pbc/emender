# train.py async quorum TCP transport validation, 2026-07-07

Task: `implement-train-py-4`

## Local validation

Command:

```bash
micromamba run -p /lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 \
  python -m pytest -q \
  tests/test_async_diloco_real_trainer.py \
  tests/test_trainpy_async_quorum_smoke_launchers.py \
  tests/test_async_diloco_e97_2n8n_debug_runner.py \
  tests/test_async_diloco_checkpoint_manager.py \
  tests/test_async_diloco_worker_supervisor_prototype.py
```

Result: `38 passed in 32.82s`.

Additional checks:

- `python -m py_compile ndm/async_diloco_real.py scripts/frontier/e97_async_diloco_train.py`: passed under the same Python 3.12 environment.
- `git diff --check`: passed.

The default `/usr/bin/python3` on `login04` is Python 3.6.15 and cannot parse
the repository's `from __future__ import annotations` files; validation used
the project Python 3.12 micromamba environment.

## Slurm 1n TCP smoke

Submitted command:

```bash
sbatch --export=ALL,WG_TASK_ID=implement-train-py-4,ASYNC_COORDINATOR_PORT=29511,HUMAN_APPROVAL_RECORD='WG implement-train-py-4: 1-node TCP quorum debug; run-local latest only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_1n_smoke.sbatch
```

Job: `4952278`  
State: `COMPLETED`  
Exit code: `0:0`  
Elapsed: `00:01:51`  
Node: `frontier02071`

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260707/E97_1.3B_step1065000_trainpy_async_quorum_1n/4952278-20260707T150907Z
```

Artifacts:

- Metrics JSON: `.../artifacts/metrics.json`
- Manifest: `.../artifacts/manifest.json`
- Summary: `.../summaries/summary.md`
- Run-local latest: `.../async_run/latest.json`

Key metrics:

```json
{
  "mode": "actual_multinode_tcp_quorum_debug",
  "rank_start_count": 8,
  "accepted_updates": 8,
  "quorum_threshold": 8,
  "timed_out_updates": 0,
  "tokens_per_generation": 1032,
  "latest_advanced": true,
  "checkpoint_paths": 4,
  "transport": {
    "name": "tcp",
    "filesystem_live_quorum": false,
    "coordinator_host": "frontier02071",
    "coordinator_port": 29511,
    "bytes_sent": 65695,
    "submit_latency_s": {
      "count": 8,
      "avg": 0.2542625069618225,
      "max": 0.6113593578338623
    },
    "timed_out_node_ids": []
  }
}
```

## 2n / 8n / 64n submission attempts

The required larger debug ladder could not be submitted from this session
because Slurm rejected additional jobs with the per-user submit limit:

```text
sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
```

Rejected commands attempted:

```bash
sbatch --export=ALL,WG_TASK_ID=implement-train-py-4,ASYNC_COORDINATOR_PORT=29512,HUMAN_APPROVAL_RECORD='WG implement-train-py-4: 2-node TCP quorum debug; run-local latest only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch -N 8 -J trainpy-tcp-quorum-8n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=implement-train-py-4,SMOKE_NAME=8n-tcp,SMOKE_NODE_COUNT=8,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_async_quorum_tcp_8n_debug,ASYNC_EXPECTED_MISSING_UPDATES=1,ASYNC_TRAINPY_RANKS=64,ASYNC_EXPECTED_RANKS=65,ASYNC_GLOBAL_QUORUM=64,ASYNC_TIMEOUT_S=120,ASYNC_COORDINATOR_PORT=29513,HUMAN_APPROVAL_RECORD='WG implement-train-py-4: 8-node train.py TCP async quorum debug; run-local latest only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch

sbatch -N 64 -J trainpy-tcp-quorum-64n -t 00:20:00 \
  --export=ALL,WG_TASK_ID=implement-train-py-4,SMOKE_NAME=64n-tcp,SMOKE_NODE_COUNT=64,SCALEOUT_VARIANT=E97_1.3B_step1065000_trainpy_async_quorum_tcp_64n_scale,ASYNC_EXPECTED_MISSING_UPDATES=4,ASYNC_TRAINPY_RANKS=512,ASYNC_EXPECTED_RANKS=516,ASYNC_GLOBAL_QUORUM=512,ASYNC_TIMEOUT_S=180,ASYNC_COORDINATOR_PORT=29514,HUMAN_APPROVAL_RECORD='WG implement-train-py-4: 64-node train.py TCP async quorum scale check; run-local latest only; no production latest mutation authorized.' \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

This is a scheduler/accounting blocker, not a transport failure. No 2n/8n/64n
job IDs were created by these attempts.

## Latest/checkpoint guard

All submitted/attempted commands used run-local debug roots under
`/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/...`.
No production `latest.pt` path was authorized or mutated by this validation.
