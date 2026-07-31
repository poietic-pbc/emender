# Frontier E97 clean 128-node production rung — job 5127775

**Date:** 2026-07-31
**Task:** `monitor-existing-frontier-128`
**Verdict:** **PASS** (`full_pass=true`)

## Authority and conformance boundary

This run follows **ADR-003, production same-allocation execution epochs
(2026-07-31)** and the production conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, with the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

- **R07:** `train.py` synchronously published checkpoints using its atomic
  temporary-file/rename and temporary-symlink/rename path. Only readable
  `latest.pt` was promoted.
- **R12:** all 1,024 ranks resumed the stable step-1065000 seed. The final
  atomic step-1065480 checkpoint was independently mmap-reloaded with the
  production PyTorch serialization boundary, including model and optimizer
  state. The selected `avg`, `outer_beta=0` mode is stateless and intentionally
  has no outer tensor.
- **R14 / NDP13:** one bounded fixed-world child ran under the production
  timeout and `srun --kill-on-bad-exit --wait` boundary, then shut down cleanly.
  The clean-rung minimum floor was all 128 nodes. No damaged communicator was
  preserved or shrunk.
- **R16:** immutable passing 32-node predecessor job 5127202 and exact source
  `ac0c90a9` bound the approved `8 -> 32 -> 128` production ladder.
- **NDP15:** only synchronous checkpoint atomicity applies. Checkpoint time and
  the observed foreground pause are reported honestly below; no background
  checkpoint, overlap, mailbox, hashing-in-training, or later-apply claim is
  made.
- **NDP02** and the elastic/native clause of **NDP17** are explicitly
  **retired/incompatible for production**. The child deliberately used
  fixed-world hierarchical RCCL. R02-R06/R08-R11 dynamic-pool semantics,
  NDP01/NDP03-NDP12/NDP14/NDP16, V21S01-V21S17, and ISP01-ISP07 are not claimed.

The rendered production compute closure adds no SQLite, database, filesystem
lock, metadata heartbeat, membership service, cell, owner tree, or central
full-model broker. Training and 67,108,864-element-bucket merges stayed on
GPU/RCCL; Lustre held the seed, output, synchronous atomic checkpoints, and
retained evidence. This report records the applicable fixed-world deadline,
shutdown, exact commands, identities, metrics, checkpoint, and terminal
accounting evidence.

## Immutable identities and scheduler transaction

The already-submitted payload was monitored without modification. The retained
runner/source envelope is Git commit `52a60f76`; the staged exact production
source and scheduler identities are:

```text
envelope_commit=52a60f76
source_sha=ac0c90a91c4c8e68265e573cea9cb808e00987ac
payload_job=5127775
collector_job=5127776
payload_digest=e06bdcec1ff8b671135ddc79b11704e9aa526ae44197eb4ded59e65da4e609a5
payload_sha256=8a1d09681daebea2a843c95a217e53d2ee9091db356bd12545d8ce687a3b40a7
collector_sha256=5838a8eebcf72150184405205b3785b9c116ee78c90a5c808aed3021b59a42cf
predecessor_job=5127202
predecessor_payload_digest=6c2ca2e10120156d260befddbbc41cd2e31d975edaf4361dfa2b385ace4ee4c6
predecessor_verdict_sha256=b9d781df522ede0f65cd21c1da67efa8985e8bbaa8822df84fa005ec09ae9306
```

The immutable seed was
`checkpoint_step_1065000_loss_2.5386.pt`, SHA-256
`c68ea2d95f2721f1f52664f71c1453e4f30a5520b33eb1cf54974185e5a100a4`.
The payload configuration fixed 128 nodes, 1,024 ranks, K40, save every 200
steps, retention 2, hierarchical merging, 67,108,864-element buckets, and no
fault injection.

While queued, repeated observations showed payload 5127775 as `PENDING
(Priority)`, never failed, with `Nodes=128`, `Partition=batch`, and `QOS=normal`.
Collector 5127776 remained scheduler-owned `afterany:5127775`, one node on
`batch/normal`. The payload began at 2026-07-31 12:21:12 EDT. Its immutable
envelope retained the live fail-closed observation:

```text
5127775|RUNNING|128|frontier[...128 nodes...]|batch|normal|frontier[...]
```

`scontrol-live.txt` separately records `NumNodes=128`, `NumTasks=1024`,
`Partition=batch`, `QOS=normal`, and `JobState=RUNNING`. Terminal accounting is
committed in `reports/frontier/e97-same-allocation-128n-5127775-sacct.txt`:

```text
5127775   e97-128n-clean         COMPLETED 0:0  NNodes=128              batch normal 00:20:53
5127775.0 bash                   COMPLETED 0:0  NNodes=128 NTasks=1024               00:20:32
5127776   collect-e97-128n-clean COMPLETED 0:0  NNodes=1                batch normal 00:00:30
```

Terminal `scontrol` also preserved payload `NumNodes=128`, `NumTasks=1024`,
`Partition=batch`, `QOS=normal`, `Restarts=0`, and collector job name,
`NumNodes=1`, `Partition=batch`, `QOS=normal`, and its original submit line
with `--dependency=afterany:5127775`. No job was cancelled, requeued, retried,
or submitted by this monitoring task; no 256-node job was created.

Durable evidence roots are:

```text
run=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-128n/runs/e97-128n-clean-20260731T145155Z-e06bdcec1ff8
payload=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-128n/payloads/e06bdcec1ff8b671135ddc79b11704e9aa526ae44197eb4ded59e65da4e609a5
collector=/lustre/orion/bif148/proj-shared/emender/frontier_runs/same-allocation-clean-128n/collectors/5127776/payload-5127775
```

These roots retain submission/config/source/seed/payload hashes, held and live
scheduler evidence, exact output from every rank, execution-epoch evidence,
checkpoint event timestamps, terminal accounting, artifact paths, collector
checksums, and the machine verdict. Existing evidence was not rewritten or
removed.

## Clean 128-node measurements

The sole execution epoch is recorded as
`1|0|128|1024|27776|1`: return code zero, 128 nodes, 1,024 tasks, fresh port,
and successful atomic checkpoint promotion. All **1,024/1,024** ranks emitted
the post-RCCL initialization binding line. Rank 0 announced
`world_size=1024 backend=nccl`, and the runtime manifest resolved the requested
plugin to
`/sw/frontier/rccl-plugins/aws-ofi-nccl/1.19.2/rocm/7.1.1/lib/librccl-net.so`.
There was no fault marker or Python traceback.

The run completed **12 real K40 merges**, steps 1065040 through 1065480.
Every hierarchical merge time was finite and positive.

| Metric | Observation |
|---|---:|
| steady K40 cadence (median) | 70.0 s |
| K40 cadence range | 69-90 s |
| hierarchical merge duration (median) | 5,485 ms |
| hierarchical merge range | 4,946-8,860 ms |
| prior proven 1,024-rank reference | 6,529.6 ms |
| median/reference ratio | 0.8400 |
| steady global throughput (last-10 median) | 5,101,338.5 tokens/s |
| final logged global throughput | 3,990,045 tokens/s |
| final logged loss | 2.5112 (finite) |
| final step | 1065480 |

The first periodic step-1065200 checkpoint took **7.464997459 s** from first
observed temporary-file creation to atomic `latest.pt` publication, or
0.02013 of the first five K40-window duration. The next 10-step log interval
was 35 s versus a steady 17 s, an honestly reported **18.0 s synchronous
foreground pause** rather than an overlap claim.

Retention correctly left two checkpoints:

```text
checkpoint_step_1065400_loss_2.3059.pt  7,719,679,988 bytes
checkpoint_step_1065480_loss_2.4639.pt  7,719,680,116 bytes  <- latest.pt
```

No temporary checkpoint/latest file remained. The collector independently
hashed final `latest.pt` as
`20c17db39a6da547783d216b0e78f545e63be8181e5164697f4ec654ab061b4a`
and mmap-reloaded step 1065480 with finite loss plus model and optimizer state.
The child emitted `Training complete! Final step: 1065480`, launcher return
code was zero, and both payload and collector completed `0:0`.

## Machine verdict and validation

The committed machine verdict is
`reports/frontier/e97-same-allocation-128n-5127775-verdict.json` (SHA-256
`fb41ecacb5c91177498227daef04391e5f9224daa4e14e0f8bf156f5869b3481`).
It records `errors=[]`, `initialized_ranks=1024`, `k40_merges=12`,
`checkpoint_reloadable=true`, `clean_shutdown=true`, and literal
`full_pass=true`.

Collection and independent validation both sourced the canonical Frontier
environment and used `$EMENDER_PYTHON`:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" --version  # Python 3.12.13
"$EMENDER_PYTHON" - reports/frontier/e97-same-allocation-128n-5127775-verdict.json \
  reports/frontier/e97-same-allocation-128n-5127775-sacct.txt
sacct -j 5127775,5127776 -P \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,NNodes,NTasks,NodeList,Partition,QOS,Account,Submit,Start,End,Elapsed
```

The Python validation asserted the exact job/source/payload/predecessor and
queue identities, 128 nodes/1,024 ranks, finite loss/throughput and every merge
time, at least five K40 merges, atomic ~7.7 GB checkpoint retention/reload,
clean completion, empty errors, and literal `full_pass is True`.
