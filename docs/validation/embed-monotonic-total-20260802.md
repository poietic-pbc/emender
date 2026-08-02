# Embedded monotonic total-token checkpoint clock

**Task:** `embed-monotonic-total`
**Date:** 2026-08-02
**Scope:** production fixed-world `train.py` and its same-allocation / reviewed
256-node continuation wiring. No Slurm job was submitted.

## Authority and ADR-003 checklist

This implementation was checked against **Resilient DiLoCo Compute Pool,
version 1, ADR-003 production same-allocation execution epochs (2026-07-31)**
and the ADR-003 fixed-world crosswalk and required conformance checklist in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** `total_tokens` is in the same `torch.save` payload as model and
  optimizer state, before the existing temporary-file `os.replace` and atomic
  `latest.pt` symlink publication. There is no independently advancing token
  database, service, pointer, or sidecar. The field is also mirrored in
  `checkpoint_metadata`.
- **R12:** resume restores the embedded integer clock independently of current
  world size. A contradictory explicit count or top-level/metadata mismatch
  fails closed. A legacy checkpoint has no usable inferred clock and requires
  an explicit trusted bootstrap. Failed execution-epoch work after the newest
  checkpoint is discarded; the fresh child reloads that checkpoint's count.
- **R14 / NDP13:** the existing finite child `timeout`, TERM/KILL,
  `srun --kill-on-bad-exit --wait`, no-progress bound, final-checkpoint margin,
  and Slurm requeue boundary are unchanged. The clock adds no wait or runtime
  service. Publication logs now include both checkpoint step and total tokens.
- **R16:** this change wires the already human-reviewed single-job 256-node
  continuation path without submitting a job or creating a new promotion rung.
  Exact-source and `main == origin/main` submission guards remain in place.
- **NDP15:** only ADR-003 synchronous checkpoint atomicity is claimed. No
  background checkpointing, hashing, mailbox, later apply, or overlap behavior
  was introduced.

The other production dispositions remain unchanged: **NDP02** and
**NDP17** are retired/replaced for this fixed-world path; R02-R06/R08-R11,
NDP01/NDP03-NDP12/NDP14/NDP16, V21S01-V21S17, and ISP01-ISP07 remain explicitly
retired or unclaimed. The rendered child remains fixed-world hierarchical RCCL
with 67,108,864-element buckets and a fresh process group per execution epoch.
There is no SQLite/database, filesystem lock, metadata heartbeat, membership
service, owner tree, central full-model broker, or new coordination protocol.
The minimum production progress floor remains `MIN_NODES`; below it the parent
fails closed to scheduler requeue.

## Accounting contract

One successfully returned optimizer update advances the in-memory count by

```text
world_size * batch_size * chunk_size * grad_accum
```

and the next periodic/final atomic checkpoint commits that count with model and
optimizer state. No absolute-step derivation occurs on resume. Changed-world
continuations preserve the embedded base count and use only the new fixed
world for subsequent increments.

The immutable final seed is the legacy step `2300930` bootstrap with
`total_tokens=150793748480`. The one-time receipt
`e97-total-token-migration-step2303840.json` binds the current legacy target to:

- stable `RUN_ID=e97-final-seed-production-256n`;
- stable `RUN_DIR` and `train/latest.pt` path;
- resolved `checkpoint_step_2303840_loss_2.3178.pt`;
- step `2303840`, size `7719680116`, and source job `5134243`;
- `150793748480 + 2910 * 2048 * 4 * 2048 * 1 = 199615447040`.

The launcher supplies that receipt count only while the exact bound legacy
checkpoint is current. It supplies the seed count only for the recognized
job-local immutable seed. For every other checkpoint it supplies no duplicate
count, so a newly published checkpoint's embedded field is sole authority and
an unexpected legacy checkpoint fails closed in `train.py`. The existing 7.7-GB checkpoint was not rewritten. A read-only
`torch.load(..., mmap=True)` inspection confirmed `step=2303840`, no top-level
`total_tokens`, and no metadata mirror before issuing the receipt.

## Validation

All Python commands sourced `scripts/frontier/activate_emender_frontier.sh` and
used `$EMENDER_PYTHON` (Python 3.12.13).

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_monotonic_total_tokens.py \
  tests/test_same_allocation_trainpy_restart_launcher.py \
  tests/test_e97_256n_final_seed_production_runner.py \
  tests/test_checkpoint_finalization.py \
  tests/test_e97_checkpoint_retention_guard.py \
  tests/test_diloco_merge.py \
  tests/test_diloco_hierarchical_math.py \
  tests/test_walltime_final_checkpoint.py \
  tests/test_frontier_runtime_plumbing.py
# 92 passed

RUN_ID=e97-final-seed-production-256n
RUN_DIR=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/$RUN_ID
"$EMENDER_PYTHON" scripts/frontier/validate_total_token_migration_receipt.py \
  --receipt docs/validation/e97-total-token-migration-step2303840.json \
  --checkpoint "$RUN_DIR/train/latest.pt" --run-id "$RUN_ID" \
  --run-dir "$RUN_DIR" --latest "$RUN_DIR/train/latest.pt" \
  --expected-step 2303840 --expected-total-tokens 199615447040 \
  --expected-source-job-id 5134243 --expected-size-bytes 7719680116
# 199615447040

bash -n scripts/frontier/e97_same_allocation_restart.sbatch \
  scripts/frontier/submit_e97_256n_final_seed_2h.sh \
  scripts/frontier/e97_256n_final_seed_payload.sh
git diff --check
```

The unit suite covers trusted seed bootstrap, periodic/final and `latest.pt`
round trips, exact per-step arithmetic, gradient accumulation, changed-world
resume, failed-epoch rollback, explicit/embedded and metadata mismatch failure,
invalid count types, and exact legacy-receipt path/step/size/job/run binding.
No `sbatch`, `srun`, `scontrol requeue`, or other scheduler mutation was run by
this task; launcher integration uses deterministic test shims only.
