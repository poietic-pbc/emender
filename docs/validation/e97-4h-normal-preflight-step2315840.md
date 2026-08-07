# E97 four-hour normal-QOS preflight from step 2315840

Date: 2026-08-04

## Decision

Prepare one human-approved four-hour normal-QOS continuation from the completed 400.94B milestone. The longer segment amortizes the approximately 15–20 minute 2,048-rank startup and 15-minute finalization reserve while retaining the fail-stop policy: one child, no validation, no retry, and scheduler `Requeue=0`.

Approved resume authority:

- step `2315840`
- total tokens `400942039040`
- loss `2.2928028006553647`
- checkpoint bytes `7719680180`
- milestone SHA-256 `c1f1fcb98099e7d141b998dda810baaa76208d36e3f1cc749c1b6ddddf370895`

The immutable milestone is stored at:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2315840-tokens-400942039040/checkpoint_step_2315840_loss_2.2928.pt
```

## Scheduler and training binding

- 256 nodes / 2,048 ranks
- `Partition=batch`
- `QOS=normal`
- `TimeLimit=04:00:00`
- `Requeue=0`
- no dependency or collector
- K40; save every 200 steps; retain two rolling checkpoints
- hierarchical 67,108,864-element RCCL buckets
- no validation or held-out evaluation arguments
- exact expected resume step/token pair included in the payload digest and checked by mmap before scheduler mutation

## Architecture conformance

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 with the 2026-08-04 fail-stop amendment, and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

Applicable safety intent: **R07/NDP15** atomic checkpoint authority; **R12** exact model/optimizer/outer/token resume; **R14/NDP13** bounded single-child failure and finalization; **R16** explicitly reviewed fixed-world 256-node continuation. Explicitly retired/unclaimed: R02–R06, R08–R11; NDP01–NDP12, NDP14, NDP16–NDP17; V21S01–V21S17; ISP01–ISP07. No elastic membership, communicator shrink, automatic restart/requeue, database, background snapshot, or overlap claim is made. The minimum progress floor is the complete 256-node / 2,048-rank world; any failure terminates the job.

## Validation

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_256n_final_seed_production_runner.py \
  tests/test_same_allocation_trainpy_restart_launcher.py
bash -n scripts/frontier/submit_e97_256n_final_seed_4h_normal.sh \
  scripts/frontier/e97_256n_final_seed_payload.sh \
  scripts/frontier/e97_same_allocation_restart.sbatch
```

Submission must set:

```bash
export E97_EXPECTED_RESUME_STEP=2315840
export E97_EXPECTED_RESUME_TOTAL_TOKENS=400942039040
bash scripts/frontier/submit_e97_256n_final_seed_4h_normal.sh
```

Retain submitted and live scheduler evidence naming `Partition=batch`, `QOS=normal`, `TimeLimit=04:00:00`, and `Requeue=0` separately.
