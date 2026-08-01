# Final E97 seed / same-set retry diagnosis — Frontier job 5133474

**Date:** 2026-08-01  
**Task:** `wire-final-150b`  
**Physical verdict:** **PARTIAL — not an acceptance pass** (`full_pass=false`)

This report preserves the useful result from payload 5133474 and collector
5133475 without relabeling it. The payload itself completed `0:0`; the durable
collector correctly failed because the original acceptance contract conflated
execution epochs with Slurm allocation attempts, expected an ambiguous failure
although Slurm emitted a unique direct hostname attribution, expected only two
node seed receipts although the job requeued, and did not observe a
post-retry K40 checkpoint before the first allocation's pre-walltime signal.

## Authority and production conformance boundary

The implementation and diagnosis follow **ADR-003, production same-allocation
execution epochs (2026-07-31)** and the production conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, with the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** only synchronous temporary-file/rename plus temporary-symlink/rename
  checkpoints were selected. The failed child left step 2301000 authoritative;
  no partial or newer failed-child checkpoint was promoted.
- **R12:** every fresh child received stable `train/latest.pt`. Epoch 2 on the
  same allocation and epoch 3 after Slurm requeue each loaded step 2301000 on
  all 16 ranks. The final authority advanced to step 2301280.
- **R14 / NDP13:** the failed fixed-world child was bounded by
  `srun --kill-on-bad-exit`, finite wait, and external TERM/KILL deadlines. Its
  process set and communicator were discarded. Fresh epochs used fresh ports.
- **R16:** this was exactly one two-node diagnostic, not a production ladder
  rung. No 4-node or larger job was submitted or authorized.
- **NDP15:** only synchronous checkpoint atomicity applies. No background
  checkpoint, hashing-in-training, mailbox, overlap, or later-apply claim is
  made.
- **NDP17:** native G2-G6 is retired/replaced for ADR-003 production. This run
  claims neither a native service nor communicator shrink.
- **NDP02**, elastic R02-R06/R08-R11, other native requirements, V21S01-V21S17,
  and ISP01-ISP07 remain retired or unclaimed for this fixed-world production
  path.

The compute role remained fixed-world `train.py` with hierarchical RCCL,
67,108,864-element buckets, K40, synchronous checkpoint publication, no
SQLite/database/membership service, no communicator preservation, and no
central full-model broker. `MIN_NODES=2` was the acceptance minimum progress
floor.

## Exact source, seed authority, and scheduler transaction

Executed source:

```text
7930dffcf62ba7cce1d6be45885c830d668521f5
canonical base c625cede2b97ad43af6e1e47a5fd4d58e1dbafcb
payload digest 14b06da1866e65ac5bed99e29787780a50272c268a07b1c926d130922846bea3
```

The submitter sourced `scripts/frontier/activate_emender_frontier.sh` and used
`PYTHON_BIN=$EMENDER_PYTHON`. It materialized and attested this exact immutable
identity before submission:

```text
step                 2300930
accepted tokens      150793748480
bytes                 7719680116
SHA-256               0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
attestation SHA-256   27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb
checkpoint            s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt
manifest              s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json
```

Payload 5133474 was held first. Durable collector 5133475 was registered with
`afterany:5133474` before release. Held and live evidence retained separate
scheduler fields for exactly `Nodes=2`, `Partition=batch`, and `QOS=debug`.
Terminal accounting says:

```text
5133474|e97-final-seed-retry-2n|COMPLETED|0:0|...|2|...|frontier[07774-07775]|batch|debug|...|00:13:54
5133475|collect-e97-final-seed-2n|FAILED|1:0|...|1|...|frontier07515|batch|normal|...|00:00:13
```

The collector failure is the honest `full_pass=false` machine result; payload
completion is not substituted for acceptance.

## Seed staging and compute-role network closure

On each allocation attempt the submit-side content-addressed seed was `sbcast`
to `/tmp/emender-e97-seed-5133474/checkpoint-step-2300930.pt`. Every physical
node independently performed offline size, SHA, seed identity, and attestation
verification before model load. All four retained manifests report
`network_fetches=0`, exact size `7719680116`, exact seed SHA, and the same
attestation digest:

```text
restart 0: frontier00891, frontier01020
restart 1: frontier07774, frontier07775
```

These are two node receipts per allocation attempt, not four nodes in one
attempt. The original collector's `expected ... 2, found 4` error was therefore
an acceptance-model bug rather than a seed-integrity failure.

## Execution epochs and exact observations

The durable epoch ledger is:

```text
execution epoch | rc  | nodes | ranks | port  | promoted | committed step
1               | 137 | 2     | 16    | 33475 | 1        | 2301000
2               | 143 | 2     | 16    | 33476 | 1        | 2301000
3               | 0   | 2     | 16    | 33477 | 1        | 2301280
```

### Epoch 1 — final seed, real K40 work, checkpoint, one direct failure

All 16 ranks loaded the exact job-local step-2300930 seed. Finite training
metrics and five completed hierarchical K40 merges were recorded; representative
throughput was `83051 global_tok/s` with finite loss `2.4483`. At step 2301000,
rank 0 synchronously published `checkpoint_step_2301000_loss_2.4944.pt` and
advanced atomic `latest.pt`.

The one-shot fault then fired exactly once at merge 6 / bucket 1 / rank 1 with
exit code 86. Slurm emitted the trustworthy direct record:

```text
srun: error: frontier00891: task 1: Exited with exit code 86
```

The production policy correctly recorded:

```text
1|frontier00891|strike|direct-task-exit|1
```

One direct strike does **not** exclude the host. Both nodes remained scheduler
healthy, and live plus after-fault `squeue` each recorded:

```text
5133474|RUNNING|2|frontier[00891,01020]|batch|debug|frontier[00891,01020]
```

The failed child was bounded (about 477.19 seconds including real training and
teardown), left no partial checkpoint, and did not publish above step 2301000.
This was not an ambiguous/no-strike physical failure. Ambiguous attribution and
its no-strike policy remain covered by deterministic tests.

### Epoch 2 — genuine same-allocation, same-set reload, interrupted by walltime

The supervisor retained both nodes after the first strike and launched a fresh
16-rank process set on the identical expanded nodelist
`frontier00891,frontier01020`. It changed `MASTER_PORT` from 33475 to 33476,
removed the fault environment, and all 16 ranks reloaded committed step 2301000.
It logged two additional finite training steps.

The original 20-minute submission used `--signal=B:TERM@300`; the batch signal
arrived before epoch 2 completed its first post-retry K40 merge/checkpoint.
Epoch 2 returned 143 and requested Slurm requeue. Therefore this run does prove
`same_node_set_retried=true` and `checkpoint_reloaded=true`, but it does **not**
prove `post_retry_checkpoint_advanced=true` within the same allocation.

### Epoch 3 — requeued allocation continuation

Slurm restart count 1 received `frontier07774,frontier07775`. The batch parent
again staged and independently verified the final seed on both nodes, but
correctly retained newer run checkpoint step 2301000 as restart authority. All
16 ranks loaded step 2301000, completed seven K40 merges with finite metrics,
and synchronously advanced final authority to step 2301280. This demonstrates
requeue-safe run-checkpoint authority; it is not relabeled as the missing
same-allocation post-retry checkpoint.

## Machine verdict and preserved evidence

The collector verdict remains literally `full_pass=false` with these original
errors:

```json
[
  "expected two execution epochs, found 3",
  "ambiguous collective failure recorded a node strike",
  "post-retry K40 progress missing",
  "expected two independent node seed manifests, found 4"
]
```

The first, second, and fourth errors diagnose the original collector contract;
the third identifies the real physical gap. The diagnostic summary committed
alongside this report preserves both the positive facts and the gap without
manufacturing a pass.

Durable roots:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-same-set-retry-2n/runs/e97-final-seed-2n-20260801T070114Z-14b06da1866e
/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-same-set-retry-2n/collectors/5133475/payload-5133474
```

Selected evidence SHA-256 values:

```text
bfc4689d0ca5465264f94329caaa89eada8cc2fdaed96130a439a52a393444da  identity/submission.json
706b91e07576dd0274c48d785680f8e26b95597568fe7492c8d3e3e6d89414a7  identity/squeue-held.txt
2c44015401520c28017c1b52320b8bd38d0f854b14b26f7e994219fe0f8c1aeb  supervisor/execution-epochs.tsv
497a04969d8fe3a372fb22c64076c9b5f09fabc4d84cb79c572101143bf0eec3  supervisor/node-policy.tsv
206b52db6f28545ecc5674a7498972a43528a7da0d3f32ea803907ef9540dff9  identity/slurm-attempts.tsv
e0386ffae7ad788f4a1fb2d6df930509eab9baccf9cc1c7d53e40ddc958610d1  supervisor/final-seed-retry/compute-srun-results.tsv
f1ee01fda91e72aa5e9faa63382c317e4280da4a78eb14ee3caa8a9e9f69dacb  supervisor/final-seed-retry/squeue-live.txt
f1ee01fda91e72aa5e9faa63382c317e4280da4a78eb14ee3caa8a9e9f69dacb  supervisor/final-seed-retry/squeue-allocation-after-fault.txt
7c74298b7e72eb7d7a72fe04b33ccc79de561c56c59c747ed060f90dbfa9596c  verdict.json
70d5111c273e2b7457bafd87bcf5775e4862f8007a6c99060beef6a44ab1a5ee  collector sacct.txt
```

## Subsequent implementation-only correction and stop boundary

Source `278e008189a9b3d38c7c8cc9f14e4b8dec403799` keeps node verification receipts
under allocation-attempt-specific directories, distinguishes execution epochs
from allocation attempts in the collector, encodes the accepted first-direct-
strike/no-exclusion policy, and retains ambiguous/no-strike as deterministic
coverage. It remains based on canonical `c625cede...`.

After that implementation was pushed, a changed replacement payload and its
collector were registered as jobs 5133844/5133845. A user stop arrived before
allocation; both jobs were immediately cancelled while still PENDING. Accounting
records zero elapsed time, no assigned nodes, and no replacement physical
execution. No further submission is authorized without an explicit user
instruction.

## Deterministic validation

All Python validation sourced the canonical Frontier activation and used
`PYTHON_BIN=$EMENDER_PYTHON`:

```bash
source scripts/frontier/activate_emender_frontier.sh
export PYTHON_BIN="$EMENDER_PYTHON"

bash -n \
  scripts/frontier/e97_same_allocation_restart.sbatch \
  scripts/frontier/submit_e97_2n_final_seed_retry.sh \
  scripts/frontier/e97_2n_final_seed_retry_collector.sh \
  scripts/frontier/e97_2n_final_seed_retry_srun_shim.sh

"$EMENDER_PYTHON" -m pytest -q \
  tests/test_same_allocation_trainpy_restart_launcher.py \
  tests/test_e97_s3_seed.py \
  tests/test_checkpoint_finalization.py \
  tests/test_e97_checkpoint_retention_guard.py \
  tests/test_walltime_final_checkpoint.py \
  tests/test_resilient_e97_runtime.py
# 112 passed in 251.60s
```

The deterministic suite covers first-attempt seed binding, pre-first-checkpoint
requeue rebinding away from a prior job's `/tmp`, preservation of a newer run
checkpoint, exact first/second direct-strike policy, ambiguous no-strike,
Slurm-hard-bad exclusion, progress reset, bounded no-progress requeue, fresh
ports/process epochs, and checkpoint/runtime behavior.
