# Quality Pass: Async 256n Integration Chain

Date: 2026-07-06
Task: `qp-256n-async-integration`

## Scope

Reviewed the WG metadata for the refreshed async DiLoCo E97 256n12h launch chain. This was a metadata quality pass only; no code implementation or Slurm launch was performed.

User intent preserved: integrate async DiLoCo into main, prepare the 256n12h path, wait for the user-shared refreshed single-node `latest.pt`, validate that seed with a 4-node 20-minute debug run, and only then queue the 256n12h production job.

## Changes Made

- Added direct dependency gates from `submit-refreshed-e97-async-256n12h` to:
  - `integrate-async-diloco-main`
  - `fix-async-diloco-256-entrypoint`
  - `register-refreshed-e97-seed-latest`
  - `validate-refreshed-seed-4n20m`
- Tightened `register-refreshed-e97-seed-latest` so it remains paused/blocked if the refreshed seed path has not been supplied by the user, forbids inferred or stale seed paths, and requires a manifest plus integrity/provenance evidence.
- Tightened `validate-refreshed-seed-4n20m` so it is clearly a runner task, requires integrated main checkout code, a non-production output root, production `latest` before/after evidence, metrics/manifests/checkpoint evidence, and an explicit pass/no-go.
- Tightened `submit-refreshed-e97-async-256n12h` so production submission is conditional on all direct gates being done, requires the 4n pass to be unambiguous, and requires job ID, command, queue, logs, node-hours, token estimate, latest policy, and monitor/cancel criteria.

## Validation Results

- Downstream tasks are runner/implementation tasks where appropriate:
  - `integrate-async-diloco-main`: `exec_mode=full`
  - `fix-async-diloco-256-entrypoint`: `exec_mode=full`
  - `register-refreshed-e97-seed-latest`: `exec_mode=full` but explicitly registration-only and paused
  - `validate-refreshed-seed-4n20m`: `exec_mode=full`
  - `submit-refreshed-e97-async-256n12h`: `exec_mode=full`
  - `monitor-refreshed-e97-async-256n12h`: `exec_mode=full`
- Production submit is directly blocked by integration, entrypoint validation, seed registration, and 4n debug validation.
- The refreshed seed gate remains paused, so downstream validation remains blocked until the user supplies the new seed path.
- Concrete validation evidence now includes logs, metrics, manifests, checkpoints, `latest` behavior, job ID, launch command, node-hours, and monitor/cancel criteria.
- Large-job submission remains bounded by the user's 2026-07-06 conditional instruction and must not occur on missing or ambiguous preflight evidence.
