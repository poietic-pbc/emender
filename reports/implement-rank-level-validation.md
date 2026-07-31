# Rank-level trainer failure containment validation

## Result and retained gate

The production supervisor now treats each GPU trainer as an independently
fenced contribution identity.  Exhausting one trainer's restart budget records
`rank_retired` and revokes that process incarnation; it does not exit the node
supervisor or terminate the CPU manager, native service, or seven sibling GPU
processes.  A restart is assigned a new process incarnation and exports its
stable global rank separately.  The generation protocol continues to close
only through its explicit eligible-contribution and accepted-token floors.

This is a local/scripted two-node gate, not a claim of a new live Frontier G3
run and not authorization for 4+ nodes.  The retained evidence directory is
`reports/artifacts/implement-rank-level/`.

## Compute Pool v1 conformance

This change was checked against the **Resilient DiLoCo Compute Pool v1
(2026-07-17)** required conformance checklist and its companion matrix.

- R01-R04: the allocation fence remains acquired before launch; topology now
  separates stable rank, process incarnation, manager, and lease identities.
  Revoked incarnations cannot be reused and no launched-rank wait is added.
- R05-R07: existing unequal accepted-token float64 reference tests, explicit
  Q/T closure, and fenced atomic checkpoint tests pass unchanged.  Rank count
  is never used as an aggregation denominator.
- R08-R12: owner selection, bounded retained native buffers, prompt release,
  model-free managers, non-Lustre hot path, new-incarnation catch-up, and outer
  state recovery remain unchanged; retirement only changes membership.
- R13-R16: the backend-neutral membership protocol and bounded deadlines are
  preserved.  This patch passes the scripted two-node rung only and makes no
  4+ readiness claim.  R13's hyperscale-local adapter remains non-applicable to
  this Frontier-specific supervisor correction and remains the documented gap.
- NDP01-NDP17: no dense-path or ABI code changed.  In particular NDP02 and
  NDP13 are strengthened by rank-local exit containment; NDP06 identities,
  NDP09 backpressure, NDP10 idempotence/corruption rejection, NDP11 bounded
  replay, NDP15 fenced publication, and NDP16 telemetry remain exercised by
  their existing tests.  NDP17 remains at its previously retained synthetic
  CXI G2 gate; real-model G3 and later rungs are intentionally non-applicable
  to this local correction and are not claimed.

Checklist: READY membership is leased and bounded; generation identity is
fenced; math is deterministic and token weighted; duplicate replay is
idempotent; stale/corrupt input is rejected; committed evidence is atomic;
dense transport is bounded, non-Lustre, point-to-point, releases buffers, and
has no central full-model broker.  Trainer loss and whole-node loss remain
distinct policies.  The minimum progress floor is the configured `q_min` plus
positive `t_min`, capped against the READY snapshot where a fraction is used.

## Exact validation commands

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_resilient_e97_topology.py
"$EMENDER_PYTHON" -m pytest -q tests/test_resilient_node_quorum.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_reducer.py
"$EMENDER_PYTHON" -m pytest -q tests/test_resilient_e97_true_2n_launcher.py
```

The first regression was committed while failing as `70ea99fd`.  It enumerates
exactly 16 `(node, local GPU, global rank, process, manager)` identities.  The
pool runtime suite covers unequal weights, stale/conflicting/corrupt rejection,
lease/incarnation rejoin, commit above Q/T floors, and bounded rejection below
them.  The launcher suite asserts independent `srun --overlap --no-kill` policy.

## Live-run limitation

No new Slurm allocation was available inside this batch worker session, so a
new live `srun` fault artifact could not honestly be produced.  The retained
artifacts are deterministic local two-node protocol evidence.  A real Frontier
G3 run must still capture observed processes, GPU-reset scope, native CXI byte
counters, checkpoint, topology, and certificate before any 4+ scale claim.
