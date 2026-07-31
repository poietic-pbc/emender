# Frontier E97 clean 32-node production rung — job 5127202

**Date:** 2026-07-31  
**Task:** `run-clean-32-node`  
**Verdict:** **PASS** (`full_pass=true`)

## Authority and production conformance boundary

This run follows **ADR-003, production same-allocation execution epochs
(2026-07-31)** and the production conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, with the ADR-003 crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** `train.py` synchronously published each checkpoint through a
  temporary file/rename and temporary symlink/rename. Only readable
  `latest.pt` was promoted.
- **R12:** all 256 ranks reloaded the stable step-1065000 seed. The final atomic
  step-1065520 `latest.pt` was independently reopened with the production
  PyTorch mmap serialization boundary, including model and optimizer state.
  The selected `avg`, `outer_beta=0` production mode is stateless and therefore
  intentionally has no outer tensor to serialize.
- **R14 / NDP13:** one bounded fixed-world child ran under the production
  timeout and `srun --kill-on-bad-exit --wait` boundary and shut down cleanly.
  No broken communicator was preserved or shrunk. The clean-rung minimum floor
  was the full 32 nodes.
- **R16:** exact source `ac0c90a9` and the immutable passing 8-node predecessor
  job 5126609 bound this approved `8 -> 32 -> 128` rung. This task submitted no
  128-node or 256-node job.
- **NDP15:** only synchronous checkpoint atomicity applies. The measured
  checkpoint pause is reported honestly below; no background checkpoint,
  overlap, hashing-in-the-training-path, mailbox, or later-apply claim is made.
- **NDP02** and the elastic/native clause of **NDP17** are explicitly
  **retired/incompatible for production**. This child deliberately used
  fixed-world hierarchical RCCL. Native G2-G6, leased READY membership,
  communicator shrink, V21S01-V21S17, and ISP01-ISP07 are not claimed.

Checklist application: the immutable production source has no SQLite,
database, filesystem lock, metadata heartbeat, membership service, cell,
owner tree, or central full-model broker in the rendered compute closure.
Training and the 64M-bucket merge remained on GPU/RCCL; Lustre held only seed,
output, synchronous atomic checkpoints, and evidence. The applicable clean
fixed-world deadline/shutdown path, exact commands, identities, accounting,
metrics, checkpoints, and machine verdict are retained.

## Immutable scheduler transaction and identities

The exact production source and predecessor were:

```text
source_sha=ac0c90a91c4c8e68265e573cea9cb808e00987ac
predecessor_job=5126609
predecessor_payload_digest=875aa8d47f99f0ef9881f2db6ddfca240ced11d16580e032c2dba382fb7bc996
predecessor_verdict_sha256=a5e2d9f181aa1546b109aa7eb1a17fabbbebc55d1cd259f44f4501ee6796bf16
payload_digest=6c2ca2e10120156d260befddbbc41cd2e31d975edaf4361dfa2b385ace4ee4c6
```

The seed identity was:

```text
/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt
resolved checkpoint_step_1065000_loss_2.5386.pt
sha256=c68ea2d95f2721f1f52664f71c1453e4f30a5520b33eb1cf54974185e5a100a4
```

`submit_e97_32n_clean.sh` sourced the canonical Frontier activation and used
`$EMENDER_PYTHON` for preflight and collection. Payload 5127202 was submitted
held. Scheduler-owned `afterany:5127202` collector 5127203 was registered
before release. Held evidence recorded:

```text
5127202|PENDING|32|...|batch|normal|(JobHeldUser)
5127203|PENDING|1|...|batch|normal|(Dependency)
```

The clean envelope failed closed on a live scheduler mismatch before invoking
`samealloc_main`. Its retained live record is literal:

```text
5127202|RUNNING|32|...|batch|normal|frontier[...32 nodes...]
```

Terminal accounting is committed in
`reports/frontier/e97-same-allocation-32n-5127202-sacct.txt`:

```text
5127202   e97-32n-clean        COMPLETED 0:0  NNodes=32             batch normal 00:17:41
5127202.0 bash                 COMPLETED 0:0  NNodes=32 NTasks=256               00:17:28
5127203   collect-e97-32n-clean FAILED   1:0  NNodes=1              batch normal 00:00:30
5127458   collect-e97-32n-fix   COMPLETED 0:0 NNodes=1              batch normal 00:00:24
```

Collector 5127203 retained all durable output but initially returned false for
two validator-only assumptions: it required rank 0's intentionally
rank-filtered NCCL announcement from every rank, and required an outer tensor
from stateless `avg`/beta-zero mode. The training payload itself completed 0:0
and was **not retried**. Corrected validator bytes
`387c1e80a823324edc1c3005a8a21f461a0bde58333c5d0aa0abed071167bbcf`
were retained with the reason and rerun as supplemental scheduler job 5127458
against the same immutable output. It completed 0:0 and produced the literal
`full_pass=true` verdict. No unchanged failed bytes were retried.

Durable roots:

```text
run=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-32n/runs/e97-32n-clean-20260731T134215Z-6c2ca2e10120
payload=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-32n/payloads/6c2ca2e10120156d260befddbbc41cd2e31d975edaf4361dfa2b385ace4ee4c6
initial_collector=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-32n/collectors/5127203/payload-5127202
```

These retain source/payload/config/seed hashes, submission record, held/live
scheduler evidence, exact command and launch environment, all rank output,
step accounting, checkpoint event timestamps/paths, artifacts, and verdict.

## Clean 32-node measurements

The sole child used E97 1.3B (1,286,589,072 parameters), 32 nodes, 256 ranks,
K40, singleton GPU islands, hierarchical RCCL, and 67,108,864-element buckets.
There was no fault environment or fault marker.

Every rank passed the exact launcher's requested-plugin gate before `train.py`:
256/256 post-initialization rank-binding lines, 256/256 step-1065000 resume
lines, and 256/256 final memory/shutdown lines were retained. The rank-0 runtime
manifest resolved the requested plugin to:

```text
/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so
```

The run completed **13 real K40 merges**, steps 1065040 through 1065520.
Measured steady metrics from the machine verdict:

| Metric | Observation |
|---|---:|
| median steady K40 cadence | 67.0 s |
| K40 cadence range | 67-74 s |
| median collective duration | 3,578 ms |
| collective range | 3,367-3,737 ms |
| steady global throughput (median of last 10 logs) | 1,313,863.5 tokens/s |
| final logged global throughput | 1,087,271 tokens/s |
| final logged loss | 2.5112 (finite) |
| clean final step | 1065520 |

The first periodic step-1065200 checkpoint took **6.632327972 s** from first
observed temporary-file creation to atomic `latest.pt` publication. The next
10-step log interval was 22 s versus a steady 16 s interval, an observed
**6.0 s synchronous training pause**. This is intentionally reported as a
foreground pause, not an asynchronous path.

Retention correctly left two checkpoints:

```text
checkpoint_step_1065400_loss_2.4720.pt  7,719,679,988 bytes
checkpoint_step_1065520_loss_2.4857.pt  7,719,680,116 bytes  <- latest.pt
```

No `.tmp` checkpoint/latest file remained. Supplemental collection independently
loaded final step 1065520 with finite loss and model/optimizer state. The child
then emitted `Training complete!`, all 256 ranks reached finalization, the
launcher returned zero, and payload accounting completed 0:0.

## Machine verdict and validation commands

Committed machine verdict:
`reports/frontier/e97-same-allocation-32n-5127202-verdict.json`.
It records `errors=[]`, `checkpoint_reloadable=true`, `clean_shutdown=true`,
and literal `full_pass=true`.

```bash
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" bash -n \
  scripts/frontier/e97_32n_clean_payload.sh \
  scripts/frontier/e97_32n_clean_collector.sh \
  scripts/frontier/submit_e97_32n_clean.sh \
  scripts/frontier/e97_same_allocation_restart.sbatch
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_same_allocation_trainpy_restart_launcher.py \
  tests/test_checkpoint_finalization.py \
  tests/test_diloco_merge.py::test_diloco_fault_injection_exits_only_exact_collective
# 11 passed
sacct -j 5127202,5127203,5127458 -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed
```
