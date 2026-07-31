# Resilient DiLoCo Compute Pool v1 live E97 integration

Date: 2026-07-18
Task: `integrate-resilient-pool-v1`
Authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1
Scope: production split-role runtime integration and deterministic local gates; no Slurm operation

## Outcome

The approved E97 trainer call remains the same model/data/ScheduleFree/K=40 path. Its post-training path now streams one large node-local contribution per trainer, incrementally reduces local token-weighted deltas, freezes fenced contributions from a leased READY snapshot, routes deterministic full-layout shards directly to distributed model-free owners, redistributes committed chunks directly from those owners, and publishes one fenced atomic global commit metadata bundle.

The legacy node-0 `QuorumTransportServer` is no longer imported by the live `resilient_e97_role.py`. Node 0 serves only small membership/freeze control metadata and cannot assemble or broker the full model.

## Version-1 conformance checklist and R01–R16 map

| Requirement | Live integration anchor | Validation |
|---|---|---|
| R01 | `_allocation_admission` acquires/renews the durable fence before any role spawn; lease loser returns zero | allocation loser/telemetry test; focused fence suite |
| R02 | `PoolControlServer._ready` drives DISCOVER→BOOTING→SYNCING→READY and renew/catch-up; manager exit drains; boot UUID is the incarnation | leased-peer and distributed late-join gate |
| R03 | generation open uses `active_snapshot`, cached independently of `node_count`/launched ranks | 3 READY peers, 2-contributor Q/T close, late peer exclusion |
| R04 | live `ContributionIdentity` contains run/fence/generation/attempt/worker/incarnation/sequence; receipts are compact/idempotent | identical replay, conflicting duplicate, corrupt sequence and stale attempt test |
| R05 | one-file local streams reduce incrementally in float64; `TensorLayout.from_flat_stream` binds full E97 shards; vectorized owner reducers are identity-order deterministic | unequal weights and changing order match float64 reference; distributed representative layout matches reference |
| R06 | live control config applies `Q_min`, `T_min`, capped READY fraction and bounded deterministic freeze | missing contributor still reaches 2/10 floor; quorum-collapse unit evidence remains fail closed |
| R07 | fence-specific immutable checkpoint/manifest precede `publish_bundle(commit,checkpoint,latest)` in one current-fence SQLite transaction; authoritative latest advances monotonically | stale bundle records none; retry is idempotent; newer fence rejects old publication and advances latest |
| R08 | each manager hosts only deterministically assigned shards; checksummed direct TCP, global owner byte bound, bounded sender retention, liveness re-probe/reassignment/replay, direct fetch | owner killed after receipts; surviving owner map receives retained replay; no node-0 broker; stream file-count test |
| R09 | manager/control/owner modules contain no model/optimizer; trainers alone call `_run_real_worker` and serialize state | split-role tests and frozen trainer parity test |
| R10 | trainer/aggregate/heartbeat/telemetry paths are mount-checked node local; shared run path contains only durable control, checkpoint/handoff and retained evidence | local mount rejection and subprocess ownership evidence; Frontier attestation remains downstream |
| R11 | late peers are excluded from an open snapshot; new incarnation supersedes old; older-fence node-local recovery is discarded; committed chunks are fetched from owners | lifecycle tests, owner-loss replay, newer-fence restart parity |
| R12 | manifest binds outer-state digest and accepted-token clock; newer current fence may load an older authoritative handoff; old inner work is disposable | atomic global commit/newer-fence restart test and three-generation parity |
| R13 | pool control and owner implementation is scheduler/MPI independent; only adapter code discovers Slurm hosts | local TCP multi-process gate |
| R14 | admission and JSONL telemetry enforce READY≤180s, K40≤420s, exchange+commit≤180s, first atomic generation≤720s; bytes/throughput/high-water/release are emitted | SLO source tests and runtime telemetry assertions |
| R15 | frozen accepted-token weights advance the global clock; full-cohort/equal-weight and changing-membership/unequal-weight results match reference tolerances | deterministic reducer suite and 3/2 distributed gate |
| R16 | local multi-process and representative full-layout gates pass before Frontier; launcher is bounded at 20m with TERM@300 | local tests only; downstream owns the Frontier 2-node run |

Minimum progress in the deterministic gate is `Q_min=2`, `T_min=10`; the active READY snapshot contains three peers. The production mechanism exposes explicit values and does not derive either floor from launched ranks.

## Frozen trainer parity

`configs/frontier/e97_resilient_split_role_flat.json` remains SHA-256 `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c`. The focused parity gate verifies the rendered trainer command still passes the same pinned seed, flat argument JSON, data path, ScheduleFree optimizer migration, real (not synthetic) token stream, optimizer consume/restore path, device mapping and exactly 40 local steps. The only changed call boundary is `delta_consumer=publish_trained_delta`, whose output now enters the integrated pool.

## Transport and storage evidence

- Trainer publication: one `contribution.data` plus one manifest, regardless of microchunk count; the 128-chunk fixture creates exactly two files and writes 2,097,152 data bytes.
- Aggregate publication: one `aggregate.data` plus one manifest.
- The spool byte ledger is guarded by a node-local file lock, so independent trainer/manager processes share one hard byte cap; failed reservations roll back, prompt release reaches zero, and the space is reusable.
- Dense codec: NumPy/Torch contiguous vector conversion; the live path contains no dense `.tolist()` or per-element `struct.pack` loop.
- Sender and owner byte bounds are checked before admission; owner high-water is global across its reducers.
- Sender chunks are retained only through owner receipts/commit and explicitly deleted after local aggregate publication; owner input reducers release on finalize.
- Node-local generation telemetry reports bytes written/read, P2P/replay/redistribution bytes, bytes/s, high-water, released bytes, owner count, accepted tokens and frozen identities.
- The representative owner-loss gate moves 256 contribution bytes initially, replays 256 retained bytes after owner death, and redistributes 128 committed bytes. Full-E97 measurements are intentionally left to the downstream 2-node runner.
- Fence-specific checkpoint/handoff names prevent a late filesystem writer from overwriting a newer allocation. Only the small control-store CAS makes a handoff authoritative; an interrupted identical publication is idempotently recoverable, and `latest` can only advance to a higher generation.

## Stage bounds

The known 212–215s K40 generation cadence remains the expected baseline. The downstream 20m gate supplies stricter stage policy rather than a broad timeout: READY/import 180s, K40 420s, contribution freeze 15s, owner transport 90s, redistribution/apply 60s, exchange-through-atomic-commit 180s, first atomic generation 720s from allocation admission, and TERM handoff 300s before allocation end. The former 900s default is absent from launcher and supervisor enforcement.

## Validation commands

The TDD gate was observed failing first:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_pool_runtime.py
# collection error: ModuleNotFoundError: ndm.resilient_pool_runtime
```

Passing commands recorded during implementation:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_pool_runtime.py tests/test_resilient_e97_true_2n_launcher.py
# 32 passed

/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_e97_runtime.py tests/test_fenced_admission.py tests/test_resilient_e97_split_roles.py tests/test_resilient_e97_reducer.py
# 34 passed

python3.11 -m compileall -q ndm scripts/frontier tests/test_fenced_admission.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py tests/test_resilient_e97_reducer.py tests/test_resilient_shard_owner.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_split_roles.py tests/test_resilient_node_transport.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_true_2n_launcher.py tests/test_async_diloco_real_trainer.py tests/test_train_helpers.py
# PASS

git diff --check
# PASS

/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_fenced_admission.py tests/test_resilient_peer_membership.py tests/test_resilient_node_quorum.py tests/test_resilient_e97_reducer.py tests/test_resilient_shard_owner.py tests/test_resilient_pool_runtime.py tests/test_resilient_e97_split_roles.py tests/test_resilient_node_transport.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_true_2n_launcher.py tests/test_async_diloco_real_trainer.py tests/test_train_helpers.py
# 120 passed in 98.59s (final exact-tree rerun)

/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_resilient_pool_runtime.py tests/test_resilient_e97_true_2n_launcher.py
# 32 passed in 43.69s (rerun after final 420s manager-stage enforcement)
```

## Explicit non-claims

- No Slurm job was submitted, modified, cancelled, queried, or monitored by this task.
- No Frontier runtime success is inferred from local tests.
- The selected Frontier control-store path still requires platform locking/durability qualification (gap-matrix R01/R10).
- No 4+ node scale readiness is claimed until the downstream 2-node rungs pass.
