# Progress-aware same-allocation recovery policy

**Task:** `encode-progress-aware`

**Date:** 2026-07-31
**Scope:** local production-launcher implementation and deterministic tests only; no Slurm job was submitted.

## Authority and production conformance boundary

This change was checked against **ADR-003, production same-allocation execution
epochs (2026-07-31)** and the production conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, using the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** only the readable atomic `train/latest.pt` is promoted or resumed.
  Partial/bare files remain ineligible. The synchronous `train.py` checkpoint
  implementation was not changed.
- **R12:** every retry receives the same stable run-level `latest.pt`. Tests
  prove exact step-100 -> step-200 resume across fresh execution epochs and
  resume of step 200 after whole-node exclusion. Allocation-local strike state
  is reset for a different `(job ID, restart count, allocation nodelist)`.
- **R14 / NDP13:** each child remains a bounded fixed-world `srun` under the
  existing timeout, TERM/KILL, `--kill-on-bad-exit`, and finite `--wait`
  boundary. A failed child is discarded; the next child uses a fresh epoch,
  process group, and port. Slurm hard-bad nodes are omitted as whole nodes.
- **R16:** this is post-acceptance hardening only. It preserves the accepted
  immutable `8 -> 32 -> 128` evidence and does not create, authorize, or run a
  256-node rung.
- **NDP15:** only synchronous checkpoint atomicity is applicable. No background
  checkpointing, mailbox, later apply, hashing-in-training, or overlap claim was
  added.
- **NDP17:** the native G2-G6 chain is retired/replaced for this production
  path. The accepted fixed-world ladder remains the production evidence gate;
  no native-service or communicator-shrink conformance is claimed.
- **NDP02** and R02-R06/R08-R11 dynamic membership, NDP01/NDP03-NDP12/NDP14/
  NDP16, V21S01-V21S17, and ISP01-ISP07 remain explicitly retired or unclaimed
  for ADR-003 production.

The rendered compute-role closure still contains no SQLite/database, filesystem
lock, metadata heartbeat, membership service, health daemon, owner tree,
central full-model broker, permanent blacklist, or background checkpoint path.
The fixed-world `train.py` data plane remains hierarchical RCCL with
67,108,864-element buckets. The minimum progress floor is `MIN_NODES`; below
that floor no usable fixed-world child can launch and the supervisor requests
scheduler requeue.

## Implemented policy

After every nonzero child exit, the parent first queries `scontrol` for every
allocated node. DOWN/FAIL/DRAIN/nonresponsive states and failed/missing state
queries are immediately excluded for the current allocation. If Slurm reports
the set healthy, its ordering is stable and the same node set is retried.

A strike is recorded only when Slurm's exact task-exit records uniquely
attribute all direct reporters to one hostname in the just-launched set.
Rank-prefixed traceback text and collective reporter hostnames are ignored;
zero or multiple direct hostnames are ambiguous and record no strike. A first
direct attribution records one strike and retries the host. A second direct
attribution excludes that host from this allocation. Strikes and exclusions
are in-memory allocation-local policy, with append-only operator evidence; they
are neither scheduler mutations nor persistent blacklists.

The no-progress bound counts consecutive failed epochs whose atomic checkpoint
step does not advance. Promotion of a newer checkpoint resets the count to
zero, so the surviving allocation continues while committed work advances.
Requeue is requested only for batch signal/allocation loss, inability to launch
at least `MIN_NODES`, or the configured bounded consecutive no-progress limit.

The launcher now creates `epoch_dir` before opening `launch.env`, the rendered
command, stdout, or stderr. Historical acceptance/clean wrappers no longer
precreate finite epoch directories.

## Deterministic validation

All Python commands sourced the canonical Frontier activation and bound
`PYTHON_BIN=$EMENDER_PYTHON` (Python 3.12.13). No command below invokes `sbatch`,
a real `srun`, or any scheduler mutation; integration resolves temporary fake
`scontrol`, `srun`, activation, and git shims ahead of the system tools.

```bash
source scripts/frontier/activate_emender_frontier.sh
export PYTHON_BIN="$EMENDER_PYTHON"

"$EMENDER_PYTHON" -m pytest -q \
  tests/test_same_allocation_trainpy_restart_launcher.py
# 14 passed

"$EMENDER_PYTHON" -m pytest -q \
  tests/test_same_allocation_trainpy_restart_launcher.py \
  tests/test_checkpoint_finalization.py \
  tests/test_e97_checkpoint_retention_guard.py \
  tests/test_diloco_hierarchical_math.py \
  tests/test_walltime_final_checkpoint.py
# 31 passed

"$EMENDER_PYTHON" -m pytest -q tests/test_frontier_runtime_plumbing.py
# 12 passed

bash -n \
  scripts/frontier/e97_same_allocation_restart.sbatch \
  scripts/frontier/submit_e97_8n_samealloc_acceptance.sh \
  scripts/frontier/submit_e97_32n_clean.sh \
  scripts/frontier/submit_e97_128n_clean.sh
git diff --check
```

The deterministic suite covers immediate Slurm-hard-bad exclusion; identical
healthy-set retry after ambiguous failure with zero strikes; first and second
direct hostname strikes; different/ambiguous reporters; progress reset;
bounded no-progress requeue; allocation-identity reset; exact stable-checkpoint
resume; fresh epoch and port; reduced fixed world; and production-owned epoch
directory creation.

## Preserved accepted evidence

No accepted artifact was rewritten. The production defaults remain K40,
`save_every=200`, `keep_checkpoints=2`, synchronous atomic publication, and
hierarchical 64M-bucket RCCL. The retained ladder verdicts remain:

- 8 nodes: `reports/frontier/e97-same-allocation-8n-5126609-verdict.json`
- 32 nodes: `reports/frontier/e97-same-allocation-32n-5127202-verdict.json`
- 128 nodes: `reports/frontier/e97-same-allocation-128n-5127775-verdict.json`

Their reports and terminal accounting remain under `docs/validation/` and
`reports/frontier/`; no new Frontier execution evidence is claimed.
