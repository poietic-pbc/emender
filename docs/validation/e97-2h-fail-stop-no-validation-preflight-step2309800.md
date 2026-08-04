# E97 256-node two-hour fail-stop continuation preflight

Date: 2026-08-04

## Decision

The next continuation is restricted to the existing two-hour debug-QOS shape. It is prepared but not submitted by this change. Production training performs no inline validation, launches exactly one child `srun`, requests `--no-requeue`, and exits after the first failure. A human must inspect before any fresh submission.

Authoritative resume state observed after cancelled job 5148033:

- `RUN_ID=e97-final-seed-production-256n`
- `latest.pt -> checkpoint_step_2309800_loss_2.2665.pt`
- step `2309800`
- embedded `total_tokens=299607654400`
- top-level and checkpoint-metadata token clocks match

## Incident addressed

Job 5148033 crossed absolute step 2310000, where the former `VAL_EVERY=10000` path ran validation only on rank 0. Other ranks advanced to the next K40 collective and timed out while rank 0 was absent. Ten execution epochs reproduced the same no-progress failure; allocation-local failure accounting then permitted four Slurm requeues. This change removes validation from the production command rather than adding a barrier, and removes automatic recovery from the accepted production mode.

## Architecture conformance

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 with the 2026-08-04 operator fail-stop amendment, and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

Applicable safety intent:

- **R07 / NDP15 checkpoint atomicity:** `train.py` continues atomic temporary-file/rename checkpoint publication and atomic `latest.pt` replacement. No failed child state is promoted.
- **R12:** the one child resumes stable `RUN_DIR/train/latest.pt`, including model, optimizer, DiLoCo outer state, and embedded total-token authority.
- **R14 / NDP13:** finite `timeout`, TERM/KILL, `srun --kill-on-bad-exit`, and the batch job's immediate nonzero exit bound failure handling.
- **R16:** this is the already reviewed 256-node fixed-world path; the selected next attempt remains two-hour debug QOS and requires a separate human submission decision.

Explicitly retired/unclaimed for this production path: R02–R06, R08–R11; NDP01–NDP12, NDP14, NDP16–NDP17; V21S01–V21S17; ISP01–ISP07. There is no elastic membership, communicator shrink, background snapshot/apply, native service, database, retry, requeue, or overlap claim.

Rendered compute-role closure retains no SQLite/database/lock/heartbeat path. Dense transport remains the proven fixed-world hierarchical RCCL data plane with 67,108,864-element buckets; there is no central full-model broker or new Lustre hot-path protocol. The minimum progress floor is one complete fixed world of 256 nodes / 2,048 ranks; any failure terminates the job.

## Validation

Run from the canonical Frontier environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_256n_final_seed_production_runner.py \
  tests/test_same_allocation_trainpy_restart_launcher.py
bash -n \
  scripts/frontier/e97_same_allocation_restart.sbatch \
  scripts/frontier/e97_256n_final_seed_payload.sh \
  scripts/frontier/submit_e97_256n_final_seed_2h.sh \
  scripts/frontier/submit_e97_256n_final_seed_7h_normal.sh
```

Required source assertions:

```bash
rg -n -- '--no-requeue|Requeue=0|FAIL_STOP_SINGLE_EPOCH=1|ENABLE_VALIDATION=0|unset VAL_DATA VAL_EVERY' \
  scripts/frontier/submit_e97_256n_final_seed_2h.sh \
  scripts/frontier/e97_256n_final_seed_payload.sh
```

The immutable submitted command must contain no `--val_data`, `--val_every`, held-out, or final-evaluation option. Live scheduler evidence must name `Partition=batch`, `QOS=debug`, and `Requeue=0` separately. This document does not authorize or claim a Slurm submission.
