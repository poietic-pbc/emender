# Shard-owner replay integration evidence — 2026-07-17

This change conforms to `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, architecture
decision version 1 (2026-07-17), and companion requirements R08, R11, and R14.

## Conformance checklist

- R08: `ShardOwnerGeneration` maps every bounded E97 chunk deterministically to
  a live owner, retains sender chunks before point-to-point admission, enforces
  sender and owner byte bounds, replays all retained chunks after reassignment,
  and releases retained/owner payloads immediately after atomic commit. It is an
  in-memory/non-Lustre tensor path and has no full-model broker.
- R11: submission requires the exact leased READY `(worker_id, incarnation)`
  snapshot. Catch-up applies only a complete checksum-valid committed generation;
  stale, duplicate, incomplete, conflicting, and corrupt data fail closed.
- R14: membership leases are bounded by `PeerMembership`; this integration
  requires finite positive transport/replay/apply budgets and a live deadline on
  every submit, replay, aggregate, commit, redistribution, and catch-up operation.
- Failure path: injected loss after one shard finalizes leaves `committed is None`,
  reconstructs every owner accumulator from retained chunks, and permits a clean
  retry. No partial generation becomes visible.
- Minimum progress remains the frozen nonempty accepted identity set supplied by
  the upstream R04/R06 `GenerationAdmission` token/quorum close; this component
  cannot add a partial contribution to that frozen set.

## Measured bounds

The deterministic focused fixture has two contributions of 112 bytes each and a
112-byte committed generation. Its expected peak sender retention is 224 bytes;
one complete replay sends 224 bytes; one catch-up redistribution sends 112 bytes;
commit releases all 224 retained bytes. The test asserts configured retention and
owner limits rather than relying only on these fixture values. The executable
dependency-free orchestration smoke measured 4 retained bytes, 4 replay bytes, 4
redistribution bytes, 4 released bytes, and 12 total point-to-point bytes.

## Exact validation commands

```text
python3.11 -m compileall -q ndm/resilient_shard_owner.py ndm/resilient_e97_reducer.py ndm/resilient_peer_membership.py ndm/resilient_node_quorum.py tests/test_resilient_shard_owner.py
git diff --check
python3.11 -m pytest -q tests/test_resilient_shard_owner.py tests/test_resilient_e97_reducer.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py
pytest -q tests/test_resilient_shard_owner.py tests/test_resilient_e97_reducer.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py
```

`compileall` and `git diff --check` pass. The Python 3.11 pytest command is
environment-blocked because that interpreter has no pytest or torch. The system
pytest is Python 3.6 and is environment-blocked because it has no torch (and is
too old for this Python 3.11 codebase). A direct Python 3.11 orchestration smoke,
with protocol-compatible fake tensor primitives, passes the failure/replay/commit/
catch-up/release path and prints its measured `TrafficMetrics`; its exact heredoc
command is preserved in the WG task log.

## Committed artifacts

- `ndm/resilient_shard_owner.py`
- `tests/test_resilient_shard_owner.py`
- `reports/integrate-shard-owner-20260717.md`
- `reports/integrate-shard-owner-metrics-20260717.json`
