# E97 two-hour fail-stop continuation 5153148 and next-stage binding

Date: 2026-08-04

## Result

Job 5153148 completed successfully on the approved fail-stop, no-validation path:

- `State=COMPLETED`, `ExitCode=0:0`, elapsed `01:45:45`
- 256 nodes / 2,048 ranks
- `Partition=batch`, `QOS=debug`, `Requeue=0`, zero restarts
- source `93c419a693d442c616bf0750d02137bdb5bcae90`
- resumed step 2309800 / 299607654400 tokens
- final step 2312400 / 343228416000 tokens
- final loss 2.2960539491176606
- 65 successful K40 merges
- no validation, recovery epoch, error, timeout, abort, or temporary checkpoint debris

The final checkpoint is transactionally authoritative: top-level and `checkpoint_metadata` token fields both equal 343228416000. It was copied outside rolling retention to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2312400-tokens-343228416000/checkpoint_step_2312400_loss_2.2961.pt
```

Size is 7719680180 bytes and SHA-256 is `1d8a6ea45661d5defe0371c8a284148809317c8271a4ab7ffbf3ace13bb8ef78`.

## Next-stage safety binding

The two-hour submitter now accepts paired `E97_EXPECTED_RESUME_STEP` and `E97_EXPECTED_RESUME_TOTAL_TOKENS`. When supplied, it mmap-loads stable `latest.pt`, requires matching filename/embedded step, matching top-level/metadata token clocks, and exact equality to the explicitly approved values before any scheduler mutation. These values enter the immutable payload digest, so each sequential continuation has distinct payload bytes even when the training recipe is unchanged.

The next approved resume authority is step 2312400 / 343228416000 tokens. Production remains one child execution, no inline validation, `--no-requeue`, scheduler `Requeue=0`, and human inspection before each later submission.

## Architecture conformance

Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 with the 2026-08-04 fail-stop amendment, and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

Applicable safety intent: **R07/NDP15** atomic checkpoint authority and milestone copy; **R12** exact model/optimizer/outer/token resume; **R14/NDP13** bounded single-child termination; **R16** explicitly reviewed 256-node debug continuation. Explicitly retired/unclaimed: R02–R06, R08–R11; NDP01–NDP12, NDP14, NDP16–NDP17; V21S01–V21S17; ISP01–ISP07. No elastic membership, communicator shrink, automatic restart/requeue, database, background snapshot, or overlap claim is made. Fixed-world hierarchical RCCL remains the data plane, with no central broker or new Lustre hot-path protocol. The minimum progress floor is the complete 256-node / 2,048-rank world; any failure terminates the job.

## Validation

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_256n_final_seed_production_runner.py \
  tests/test_same_allocation_trainpy_restart_launcher.py
bash -n scripts/frontier/submit_e97_256n_final_seed_2h.sh \
  scripts/frontier/e97_256n_final_seed_payload.sh \
  scripts/frontier/e97_same_allocation_restart.sbatch
```

Before submitting the next stage:

```bash
export E97_EXPECTED_RESUME_STEP=2312400
export E97_EXPECTED_RESUME_TOTAL_TOKENS=343228416000
bash scripts/frontier/submit_e97_256n_final_seed_2h.sh
```

Retain submitted and live scheduler evidence naming `Partition=batch`, `QOS=debug`, and `Requeue=0` separately.
