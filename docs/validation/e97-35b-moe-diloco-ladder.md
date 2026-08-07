# E97 35B MoE DiLoCo qualification ledger

## Authority and scope

This ledger follows **Resilient DiLoCo Compute Pool**, Version 1 with the
2026-07-31 **ADR-003 production same-allocation execution-epoch decision**, and
the companion gap-matrix crosswalk. Applicable production safety IDs are
**R07, R12, R14/NDP13, R16, and the checkpoint-atomicity clause of NDP15**.
R02–R06, R08–R11, NDP01–NDP12, NDP14, NDP16–NDP17, V21S01–V21S17, and
ISP01–ISP07 are explicitly retired/unclaimed for this fixed-world production
path. No elastic membership, async-v2.1 overlap, native CXI, communicator
shrink, database authority, or background-checkpoint claim is made.

The requested 2- and 4-node runs are pre-production MoE qualification
observations. ADR-003 defines the production predecessor ladder as
`8 -> 32 -> 128`; therefore neither 2 nor 4 is labeled a production rung.
The planned order is `1 -> 2 -> 4 -> 8 -> 32`, with at least 20 minutes of
actual training at every scale and sequential fail-closed promotion.

## Invariants

- One node is one eight-GCD expert-parallel training island.
- Each GCD owns exactly eight contiguous routed experts.
- Router/shared/backbone state is replicated within the island as required.
- Expert assignments and expert outputs use only the proven eight-rank
  node-local RCCL group. Cross-node expert-token traffic is fatal.
- DiLoCo operates only between complete node islands.
- Every runner records `Partition` and `QOS` independently while live and in
  terminal accounting.
- No rung advances without finite loss/gradients, completed optimizer and
  DiLoCo outer steps, exact token accounting, bounded HBM, checkpoint/restart,
  and an immutable predecessor verdict.
- Compute-role closure must contain no SQLite/database/store/lock/metadata
  heartbeat dependency. ADR-003 uses fixed-world children and atomic
  checkpoints, not a coordination database.

## Status

| Gate | Status | Evidence / next action |
|---|---|---|
| Fused shared+routed forward/backward/aux/optimizer kernels | PASS | job 5181432; eight real GCDs independently passed 25 tests |
| Bounded Triton EP assignment packing/repacking | LOCAL PASS | `tests/test_e97_moe_ep_triton.py`; real eight-rank RCCL round trip next |
| One-node RCCL dispatch/return | PASS | job 5181970: exact 51-row send/receive round trip on ranks 0..7, node `frontier00388`, `batch`/`debug`, exit 0; job 5181922 retained as the corrected device-binding failure |
| Packed local-expert fused compute + backward | MACHINE PASS | job 5181981: differentiable pack/all-to-all/eight-local-expert/return chain and finite gradients on all ranks, `batch`/`debug`, exit 0 |
| End-to-end node-local MoE layer | MACHINE PASS | job 5182030: fused 64-way router, dispatch, local experts, shared expert, return, combine, auxiliary losses, and backward on eight ranks; `batch`/`debug`, exit 0 |
| Shared/backbone node reduction | MACHINE PASS | job 5182096: router/shared gradients averaged only over proven node group and equal on ranks 0..7 |
| Fused ScheduleFree optimizer | MACHINE LAYER PASS | job 5182096: BF16/FP32 same-dtype state, fused step, finite parameters, no master weights |
| Sharded 513B-seed conversion/restart | PASS | jobs 5182549/5182648: atomic checksummed shards, fresh-process restore, and continued optimizer step without 35B single-GCD materialization |
| Full 513B-seed packed-model step | PASS | job 5182363: loss 2.17162, auxiliary 0.03004, 46.28 GB HBM, 751.6 tok/s, exact 513B seed, full-context forward/backward and fused optimizer step, `batch`/`debug`, exit 0 |
| 1 node, >=20 min training | PASS | job 5182403 systems pass; job 5182549 atomically published ~210 GB of checksummed local/replicated model+ScheduleFree shards and a complete manifest (post-publication telemetry typo only); job 5182648 verified every SHA, restored in a fresh process group, and completed a step with loss 1.79291, `batch`/`debug`, exit 0 |
| 2 nodes, >=20 min qualification | PASS | job 5182666: 1,200.738 s, 82 steps, 2,686,976 tokens, two K40 merges (17.206/13.517 s), mean loss 2.66133, median 2,362.63 tok/s, max 48.73 GB HBM, `batch`/`debug`, exit 0; job 5182663 was the pre-load source-staging failure; non-production observation |
| 4 nodes, >=20 min qualification | PASS | job 5182895: 1,207.371 s, 81 steps, 5,308,416 tokens, two K40 merges (14.615/17.029 s), mean loss 2.60014, median 4,639.01 tok/s, max 48.65 GB HBM, `batch`/`debug`, exit 0; non-production observation |
| 8 nodes, >=20 min ADR-003 rung | PASS | job 5183128: 1,211.676 s, 81 steps, 10,616,832 tokens, two K40 merges (28.670/17.848 s), mean loss 2.56924, median 9,611.27 tok/s, max 48.81 GB HBM, `batch`/`debug`, exit 0; immutable source commit `a9e2d542` |
| 32 nodes, >=20 min ADR-003 rung | PASS | job 5183619: 1,201.755 s, 80 steps, 41,943,040 tokens, two K40 merges (24.096/24.103 s), mean loss 2.54960, median 38,953.08 tok/s, max 48.92 GB allocated HBM, `batch`/`debug`, exit 0, immutable source `096303da`; failed attempts 5183284 and 5183306 remain recorded below |

## Retained failed 32-node attempts

- Job `5183284` (`batch`/`debug`) failed after 12 finite steps: rank 209 on
  `frontier04828` reported HIP code 209 (`no kernel image is available`).
- Job `5183306` (`batch`/`debug`) reached the first K40 merge, then rank 54
  failed a 32 MiB RCCL allocation while the variable-routing allocator cache
  held 65.34 GB reserved HBM. It motivated commit `096303da`, which releases
  inactive cache and lane-barriers immediately before each outer merge.
- Neither failed attempt contributes duration or merge acceptance evidence.

## Validation commands

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_moe_ep_triton.py \
  tests/test_e97_moe_triton.py \
  tests/test_e97_moe.py \
  tests/test_e97_facade.py
```

Machine runs additionally retain exact `sbatch`, live
`squeue -o '%i|%P|%q|%T|%N|%j'`, and terminal
`sacct --format=JobIDRaw,JobName,Partition,QOS,State,ExitCode,Elapsed,NodeList`
output, plus checkpoint and run-manifest paths.

Minimum progress floor for each fixed-world observation is all requested node
islands alive for the accepted execution epoch and at least 20 minutes of
recorded training. A rank/node failure terminates the complete child; no broken
communicator is preserved or shrunk. Restart is from the newest readable atomic
checkpoint in a fresh process group, consistent with R07/R12/R14.
