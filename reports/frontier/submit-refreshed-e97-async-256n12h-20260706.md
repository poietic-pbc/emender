# Submit Refreshed E97 Async DiLoCo 256n12h

Task: `submit-refreshed-e97-async-256n12h`

Decision: **submitted**.

## Direct Gates

- `integrate-async-diloco-main`: done; async launch package was pushed to `origin/main`.
- `fix-async-diloco-256-entrypoint`: done; wrappers resolve `scripts/frontier/async_diloco_e97_multinode.py` from main.
- `register-refreshed-e97-seed-runner`: done; manifest registered the user-provided refreshed E97 seed.
- `validate-refreshed-seed-4n20m`: done; report `reports/frontier/validate-refreshed-seed-4n20m-20260706.md` records **PASS for the submit gate**.

## Refreshed Seed

- Manifest: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/seed_manifest.json`
- Latest path: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`
- Resolved checkpoint: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/checkpoint_step_1065000_loss_2.5386.pt`
- SHA256: `c68ea2d95f2721f1f52664f71c1453e4f30a5520b33eb1cf54974185e5a100a4`
- Size: `7719679924` bytes

## Launch Package

- Commit submitted from: `ffc9e0b` on `main`, pushed to `origin/main`.
- Wrapper: `scripts/frontier/async_diloco_e97_256n12h_launch.sbatch`
- Entrypoint: `scripts/frontier/async_diloco_e97_multinode.py`
- Training target: `E97_1.3B_step1065000_async_diloco_256n12h_20260706`
- Scaleout variant: `E97_1.3B_step1065000_async_quorum_b4_k40_256n12h`
- Output root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco`
- Expected run root pattern: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco/E97_1.3B_step1065000_async_quorum_b4_k40_256n12h/20260706/4946500-<UTC stamp>Z`
- Metrics path pattern: `<run-root>/artifacts/async_diloco_e97_256n_metrics.json`
- Command record pattern: `<run-root>/artifacts/command.txt`
- Environment record pattern: `<run-root>/artifacts/env.txt`

## Production Parameters

- Nodes: `256`
- Walltime: `12:00:00`
- Queue/partition: `batch`
- QoS: `normal`
- Requested node-hours: `3072.0`
- Worker count per node: `8`
- Local quorum: `8`
- Global quorum: `240`
- DiLoCo K: `40`
- Tokens per local step: `16777216`
- Tokens per node per generation: `671088640`
- Aggregate tokens per 256-node generation: `171798691840`
- Checkpoint cadence: generation manifests every generation; recovery every `5` generations or `600` seconds, whichever arrives first; export every `45` generations or `3600` seconds, whichever arrives first.
- Finalization buffer: `1200` seconds.
- Production latest policy: run-local `async_run/latest.json` is allowed to advance; external production chain `latest.pt` is guard-only and must not change.
- Production latest guard: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`

## Preflight

- No active jobs were visible for user `erikgarrison` immediately before submission:

```text
JOBID|NAME|STATE|PARTITION|QOS|NODES|TIME_LIMIT|TIME|NODELIST(REASON)
```

- Validation commands run before submission:

```bash
bash -n scripts/frontier/async_diloco_e97_256n12h_launch.sbatch scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m py_compile scripts/frontier/async_diloco_e97_2n8n_debug.py scripts/frontier/async_diloco_e97_multinode.py
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest tests/test_async_diloco_e97_2n8n_debug_runner.py -q
```

Result: focused pytest `3 passed`.

## Submission

- Slurm job ID: `4946500`
- Submit time from Slurm: `2026-07-06T06:11:07` local Frontier time.
- Initial state after submission: `PENDING`, reason `Resources`.
- Account: `bif148`
- Requested TRES: `cpu=14336,mem=125T,node=256,billing=14336`
- Stdout: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.out`
- Stderr: `/lustre/orion/bif148/scratch/erikgarrison/emender/logs/frontier/async_diloco_e97/async-diloco-e97-256n12h-4946500.err`

Exact launch command:

```bash
sbatch -N 256 -p batch -t 12:00:00 --export=ALL,WG_TASK_ID=submit-refreshed-e97-async-256n12h,TASK_ID=submit-refreshed-e97-async-256n12h,ASYNC_DILOCO_HUMAN_APPROVED=1,REFRESHED_E97_SEED_MANIFEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/seed_manifest.json,REFRESHED_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,SEED_LATEST_PATH=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,TRAINING_TARGET=E97_1.3B_step1065000_async_diloco_256n12h_20260706,SCALEOUT_VARIANT=E97_1.3B_step1065000_async_quorum_b4_k40_256n12h,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/async_diloco,ASYNC_NODE_COUNT=256,ASYNC_GLOBAL_QUORUM=240,ASYNC_LOCAL_QUORUM=8,DILOCO_K=40,RECOVERY_EVERY_GENERATIONS=5,RECOVERY_EVERY_SECONDS=600,EXPORT_EVERY_GENERATIONS=45,EXPORT_EVERY_SECONDS=3600,FINALIZATION_BUFFER_SECONDS=1200,REQUESTED_WALLTIME=12:00:00,REQUESTED_NODE_HOURS=3072.0,PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt,PRODUCTION_LATEST_POLICY=run-local-latest-json-with-external-chain-latest-guard scripts/frontier/async_diloco_e97_256n12h_launch.sbatch
```

## Monitor Handoff

Monitor task: `monitor-refreshed-e97-async-256n12h`.

Expected startup signs:

- Slurm state leaves `PENDING` and reaches `RUNNING`.
- Stdout/stderr files for job `4946500` appear under `logs/frontier/async_diloco_e97/`.
- Run root appears under the expected 20260706 scaleout path.
- `<run-root>/artifacts/env.txt` records refreshed seed, `async_node_count=256`, `async_global_quorum=240`, `async_local_quorum=8`, `diloco_k=40`, and `production_latest_policy=run-local-latest-json-with-external-chain-latest-guard`.
- `<run-root>/artifacts/async_diloco_e97_256n_metrics.json` appears with `conclusion=pass` or a concrete no-go reason.

Cancel criteria:

- Wrong seed path or resolved checkpoint is recorded.
- Async entrypoint is missing or not `scripts/frontier/async_diloco_e97_multinode.py`.
- Local quorum is below `8`, global quorum is below `240`, or stale/timeout/failure counts indicate unsafe progress.
- Production latest guard changes.
- Checkpoint finalization or run-local latest advancement fails.
- Job burns material node-hours without creating stdout/stderr, run-root artifacts, or metrics.
