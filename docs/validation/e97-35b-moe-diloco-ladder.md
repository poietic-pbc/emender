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
| End-to-end node-local MoE layer | MACHINE PENDING | fused 64-way router/shared/combine custom autograd passes local oracle; full eight-rank machine run next |
| Shared/backbone node reduction | PENDING | node-local RCCL only |
| Sharded 513B-seed conversion/restart | PENDING | no 35B single-GCD materialization |
| 1 node, >=20 min training | PENDING | prerequisite before scale |
| 2 nodes, >=20 min qualification | PENDING | non-production observation |
| 4 nodes, >=20 min qualification | PENDING | non-production observation |
| 8 nodes, >=20 min ADR-003 rung | PENDING | requires reviewed exact-source acceptance |
| 32 nodes, >=20 min ADR-003 rung | PENDING | requires immutable 8-node predecessor |

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
