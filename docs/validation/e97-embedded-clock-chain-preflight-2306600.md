# E97 embedded-clock continuation preflight at step 2306600

Date: 2026-08-03

This receipt authorizes one final 256-node, two-hour `batch`/`debug` continuation using the unchanged accepted production submission path:

- submitter: `scripts/frontier/submit_e97_256n_final_seed_2h.sh`
- launcher: `scripts/frontier/e97_same_allocation_restart.sbatch`
- `RUN_ID=e97-final-seed-production-256n`
- stable `RUN_DIR=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n`
- resolved `latest.pt`: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n/train/checkpoint_step_2306600_loss_2.3050.pt`
- checkpoint bytes: `7719680180`
- checkpoint step: `2306600`
- top-level `total_tokens`: `245920563200`
- metadata `total_tokens`: `245920563200`
- authority: embedded checkpoint; no explicit token-count bootstrap is required
- source job: `5140503`, terminal `COMPLETED`, exit `0:0`

The checkpoint was independently mmap-loaded under the canonical Frontier environment immediately before publication. The user queue was empty. The new commit provides a fresh immutable source and payload digest without changing the proven training, launcher, or submission logic.

## Validation

This fixed-world continuation conforms to the ADR-003 production checklist and the accepted R07/R12/R14/R16 and NDP13/NDP15 boundaries: synchronous atomic checkpoint authority, stable same-run restart authority, bounded fresh-process recovery, explicit 256-node evidence, and no background checkpoint or independently advancing token service. The embedded counter is transactionally co-published with model and optimizer state.
