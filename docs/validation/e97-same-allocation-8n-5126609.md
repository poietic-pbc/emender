# Frontier E97 8-node same-allocation acceptance — job 5126609

**Date:** 2026-07-31

**Task:** `run-8-node-e97`

**Verdict:** **PASS** (`full_pass=true`)

## Authority and conformance boundary

This run follows **ADR-003, production same-allocation execution epochs
(2026-07-31)** in `RESILIENT_DILOCO_COMPUTE_POOL.md` and its required
production-path conformance checklist, plus the ADR-003 crosswalk in
`RESILIENT_DILOCO_GAP_MATRIX.md`.

Applicable requirements were handled as follows:

- **R07:** `train.py` synchronously wrote a temporary checkpoint, atomically
  renamed it, and atomically advanced `latest.pt`; the failed epoch selected
  only that committed pointer.
- **R12:** the fresh child received the stable run-level `latest.pt` through
  `--resume` and logged `Resumed at step 1065200` on all 56 ranks.
- **R14 / NDP13:** `srun --kill-on-bad-exit=1 --wait=60` plus external
  TERM/KILL bounds contained the damaged fixed-world step. It returned
  nonzero 31.336 seconds after injection; no communicator was reused.
- **R16:** this exact-source 8-node pass is the first immutable rung of the
  approved production `8 -> 32 -> 128` ladder. It does not submit or authorize
  the next rung by itself.
- **NDP15:** only synchronous atomic checkpoint publication applies. No
  background snapshot, hashing-in-the-training-path, or later-apply claim is
  made. SHA-256 below is evidence collection, not a production protocol.
- **NDP02** and the elastic clause of **NDP17** are explicitly **retired from
  production**. The epoch uses fixed-world RCCL, then destroys the broken child
  and creates a fresh process group. Native G2-G6, communicator shrink,
  leased membership, owner trees, cells, V21S01-V21S17, and ISP01-ISP07 are not
  claimed or tested.

Checklist application: the run used a bounded fixed-world child boundary,
fresh process group and port, deliberate whole-node reduction, and no attempt
to preserve a broken all-rank communicator. The production launcher has no
SQLite/store/lock/heartbeat protocol. The actual failure/deadline path was
exercised with a seven-whole-node minimum progress floor. Training stayed on
GPU/RCCL; Lustre was used only for the synchronous checkpoint boundary, with no
central model broker. Exact commands, identities, scheduler records, checkpoint
artifacts, and machine verdict are retained below.

## Immutable scheduler transaction

The executed training source was the exact pushed production commit:

```text
ac0c90a91c4c8e68265e573cea9cb808e00987ac
```

The submitted launcher was
`scripts/frontier/e97_same_allocation_restart.sbatch` from that checkout.
Acceptance payload digest:

```text
875aa8d47f99f0ef9881f2db6ddfca240ced11d16580e032c2dba382fb7bc996
```

The submit wrapper sourced `scripts/frontier/activate_emender_frontier.sh` and
used `PYTHON_BIN=$EMENDER_PYTHON` for preflight and collection. Payload 5126609
was held; scheduler-owned `afterany:5126609` collector 5126610 was installed;
`Nodes=8`, `Partition=batch`, and `QOS=normal` were checked independently; only
then was the payload released. Held and live `squeue` evidence both say:

```text
5126609|RUNNING/PENDING|8|...|batch|normal|...
```

Terminal accounting is committed in
`reports/frontier/e97-same-allocation-8n-5126609-sacct.txt`:

```text
5126609|e97-samealloc|COMPLETED|0:0|0:9|8||...|batch|normal|...|00:20:43
5126609.0|bash|CANCELLED|0:9||8|64|...||||...|00:09:01
5126609.1|bash|COMPLETED|0:0||7|56|...||||...|00:11:29
5126610|collect-e97-8n-samealloc|COMPLETED|0:0|0:0|1||...|batch|normal|...|00:00:12
```

The exact run and collector roots are:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-acceptance-8n/runs/e97-8n-acceptance-20260731T123756Z-875aa8d47f99
/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-acceptance-8n/collectors/5126610/payload-5126609
```

They retain immutable config/source/payload/seed identities, held/live
scheduler evidence, both child commands and launch environments, rank output,
return codes, all step accounting, checkpoint timing/hash, and the verdict.

## Baseline, checkpoint, and exact fault

The 64-rank child used real E97 1.3B `train.py`, batch 4, K40, singleton GPU
islands, hierarchical RCCL, and 67,108,864-element buckets. It completed exactly
five pre-fault merges:

```text
merge #1 step 1065040
merge #2 step 1065080
merge #3 step 1065120
merge #4 step 1065160
merge #5 step 1065200
```

At step 1065200 it synchronously published
`checkpoint_step_1065200_loss_2.4789.pt`. Observed temporary-file creation to
atomic `latest.pt` publication took **6.964392708 seconds**. Size was
**7,719,679,988 bytes**; evidence SHA-256 was:

```text
37863dc0f81c3224c5dad3c4e9a1c883c21d732a4457e538cbd6db6f34f54ecd
```

The next merge injected exactly one exit:

```json
{"bucket_index":1,"exit_code":86,"label":"sf_x","merge_index":6,"rank":1}
```

Step `5126609.0` returned nonzero. Immediately after it exited, `latest.pt`
still named step 1065200, no checkpoint above step 1065200 existed, and no
checkpoint/latest temporary file remained. The batch allocation was still
RUNNING on all eight scheduler-owned nodes with `batch|normal`.

## Fresh 7-node restart

An acceptance-only node-selection shim deliberately presented the final
allocated node as unavailable to the production launcher's whole-node filter.
This produced the required 8-to-7 relaunch without a physical node crash or a
communicator-shrink test; scheduler evidence honestly retained the live
8-node allocation. The shim and its identity are part of the immutable payload.

The launcher then created epoch 2 / step `5126609.1` with:

- 7 nodes and 56 fresh ranks (previously 8/64),
- new `MASTER_PORT=26611` (previously `26610`),
- the one-shot fault environment removed,
- the stable committed step-1065200 `latest.pt` as `--resume`.

Reload-to-launch downtime was **1.056760426 seconds**. The fresh child logged
`Resumed at step 1065200`, completed **nine** additional K40 merges, and
atomically published final step 1065560. Representative final finite metrics:

```text
step 1065560 | loss 2.5370 | global_tok/s 241890
```

`keep_checkpoints=2` retained steps 1065400 and 1065560, with `latest.pt`
pointing to step 1065560.

## Machine verdict

Committed copy:
`reports/frontier/e97-same-allocation-8n-5126609-verdict.json`.

```json
{
  "allocation_survived": true,
  "failed_step_bounded": true,
  "fresh_srun_launched": true,
  "world_size_changed": true,
  "checkpoint_reloaded": true,
  "post_relaunch_merge_passed": true,
  "no_failed_state_published": true,
  "unchanged_failed_payload_retried": false,
  "full_pass": true
}
```

## Attempt discipline and validation commands

An earlier payload, 5126576, terminated before any child `srun` or model load
because immutable launcher ac0c90a9 writes `epoch-000001/launch.env` before its
in-loop directory creation. Collector 5126577 retained the negative record.
The executed replacement was **not** an unchanged retry: payload bytes changed
to precreate the two finite epoch directories, changing digest
`25b394d...` to `875aa8d...`. Exactly one training/fault payload ran. No
32/128-node or other downstream job was submitted.

Validation used the canonical environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" bash -n \
  scripts/frontier/submit_e97_8n_samealloc_acceptance.sh \
  scripts/frontier/e97_8n_acceptance_srun_shim.sh \
  scripts/frontier/e97_8n_acceptance_scontrol_shim.sh \
  scripts/frontier/e97_8n_samealloc_acceptance_collector.sh
PYTHON_BIN="$EMENDER_PYTHON" "$EMENDER_PYTHON" -m pytest -q \
  tests/test_same_allocation_trainpy_restart_launcher.py \
  tests/test_diloco_merge.py::test_diloco_fault_injection_exits_only_exact_collective \
  tests/test_checkpoint_finalization.py
# 11 passed
sacct -j 5126609,5126610 -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed
```
