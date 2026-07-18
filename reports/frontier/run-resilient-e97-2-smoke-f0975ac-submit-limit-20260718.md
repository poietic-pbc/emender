# Resilient E97 fail-closed-runtime smoke submission limit — 2026-07-18

## Result

At `2026-07-18T06:56:08Z`, a real `sbatch` attempt for the changed startup-smoke
payload was rejected before Slurm created a job:

```text
sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
```

There is therefore no job ID and no allocation or runtime for this attempt. No
existing job was cancelled. A retry must preserve the exact run and payload
identity below so it is an idempotent continuation of this unadmitted attempt,
not a new unchanged payload.

## Authoritative payload and validation

- Fetched authoritative `HEAD == origin/main ==
  f0975ac8ad088d2c2921c6928156c8a1fccccebd` before the attempt.
- Run: `run-resilient-e97-2-smoke-20260718T065608Z-f0975ac`.
- Payload: `f0975ac-20260718T065608Z-startup-smoke-fail-closed-runtime`.
- Pinned generation-0 seed SHA256:
  `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`.
- Exact approved runtime exported in the immutable payload:
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312`.
- Runtime admission asserts Python 3.12.13, torch 2.10.0+rocm7.1, HIP
  7.1.25424, and Triton 3.6.0 before any role launch and writes
  `runtime-identity.json`.
- Launcher regression suite: `26 passed` under the approved Python.
- Exactly two nodes, debug QoS, `02:00:00`, `TERM@300`, one smoke generation,
  no injection, two managers, sixteen trainers, K=40, and 900-second progress
  and generation maxima.

The trainer command preserves the known-good job-5000436 training identity:
the approved flat E97 dimensions and split-edit Triton path, batch size 4,
chunk size 2048, ScheduleFree optimizer, no gradient checkpointing, CommaPile
data, and 40 local steps. The resilient launcher differs in its required
split-role topology and bounded node-local/network transport; it does not use
synthetic data, a synthetic model, or a sentinel/legacy launcher mode.

## Exact attempted command

```bash
sbatch --parsable -A bif148 -p batch --qos=debug -N 2 --gpus-per-node=8 -t 02:00:00 -J resilient-e97-true-2n --export=ALL,REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1128,RUN_DIR=/lustre/orion/bif148/proj-shared/emender/runs/run-resilient-e97-2-smoke-20260718T065608Z-f0975ac,RESILIENT_E97_RUN_ID=run-resilient-e97-2-smoke-20260718T065608Z-f0975ac,RESILIENT_E97_SOURCE_ID=step-1525000-sha256-1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9,RESILIENT_E97_PAYLOAD_ID=f0975ac-20260718T065608Z-startup-smoke-fail-closed-runtime,RESILIENT_E97_CODE_ID=f0975ac8ad088d2c2921c6928156c8a1fccccebd,RESILIENT_E97_SEED=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260709_084606_step_1525000/checkpoint_step_1525000_loss_2.4378.pt,RESILIENT_E97_TRAIN_ARGS_JSON=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1128/configs/frontier/e97_resilient_split_role_flat.json,RESILIENT_E97_DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt,RESILIENT_E97_TIKTOKEN_CACHE_FILE=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15,RESILIENT_E97_TIKTOKEN_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069,RESILIENT_E97_GENERATIONS=1,RESILIENT_E97_STARTUP_SMOKE=1,RESILIENT_E97_REQUESTED_WALLTIME=02:00:00,RESILIENT_E97_LAUNCH_MODE=node-local,RESILIENT_E97_STARTUP_DEADLINE_S=300,RESILIENT_E97_HEARTBEAT_DEADLINE_S=60,RESILIENT_E97_PROGRESS_DEADLINE_S=900,RESILIENT_E97_GENERATION_DEADLINE_S=900,RESILIENT_E97_BULK_ROOT=/tmp/resilient-e97,RESILIENT_E97_KERNEL_CACHE_ROOT=/tmp/resilient-e97-kernel-cache,RESILIENT_E97_MAX_RESTARTS=2,EMENDER_CONDA_ENV=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312 scripts/frontier/resilient_e97_true_2n.sbatch
```

## Validation and conformance

This runner is checked against version 1 of
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, with applicable requirements R03,
R05, R06, R08, R09, R10, R12, R14, and R16. READY/live membership, bounded
deadlines, fenced identities, deterministic weighted aggregation, bounded
non-Lustre transport, model-free managers, and committed-generation evidence
remain the live acceptance criteria. R07, R11, R12, and complete R16 remain
unclaimed until the survivable-failure allocation, immutable TERM handoff, and
fresh-allocation continuation actually pass.
