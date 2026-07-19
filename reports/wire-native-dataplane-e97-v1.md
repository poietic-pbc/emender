# Native data plane wired into real E97 training

Date: 2026-07-19

WG task: `wire-native-dataplane-e97-v1`

Implementation commit: `e3e88abc43d6896bef52ecab8f0b6ab784475a35`

Base commit: `154b7c0681f56f51dcb18d3d4d48dafa18aff180`

## Result

The real split-role E97 launcher now has a separate native production path:
each of the two nodes starts one persistent `ndp_cxi_service`, one model-free
Python manager, and eight GPU trainer processes. A sealed per-node admission
token fd is inherited by the service and its clients. The service starts before
the manager; the manager publishes readiness before trainers start; service
loss terminates the affected role group; normal completion and allocation TERM
drain the manager and service without a peer-wide collective.

Native trainers do not construct `LocalTrainerSpool`. Native managers do not
construct `LocalTrainerSpool` or `DistributedOwnerServer`. Those constructors
and all Python dense-owner calls remain confined to the explicit
`python-tcp-debug` branch. The native launcher requires the exact attested
service/client/transport artifacts, exact production provider `cxi`, a clean
full-layout gate, eight local trainers, and the approved Frontier Python
environment. Python TCP and fixed-world MPI are not production fallbacks.

No Slurm command or job was submitted by this task. Existing jobs `5031461`
and `5031553` are cited only as retained upstream synthetic G2 evidence. The
next authorized operation is the downstream two-node native pool startup gate,
not a real-model or larger-scale submission.

The machine-readable companion is
[`wire-native-dataplane-e97-v1-metrics.json`](wire-native-dataplane-e97-v1-metrics.json).

## Provenance and required gap recovery

The compiled persistent boundary from
`implement-compiled-native-service-rpc-v1` was verified before integration:
the service owns the authoritative local handle registry, authenticates its
metadata-only AF_UNIX RPC with peer credentials and a protected token, accepts
sealed dense memfds by descriptor, survives independent client disconnects,
and returns read-only result descriptors to later clients.

The partial historical branch
`origin/wg/agent-1289/implement-live-split` was inspected only for provenance.
Its commits `5882eca6`, `d5e0437f`, `3715db45`, and `cd7b0682` documented the
transport bridge, bounded replay, checkpoint transition, and an earlier
producer handoff. That branch was not merged. The integration was rebuilt on
current `origin/main`, and its behavior is covered by current tests.

The production guard was changed only after the live branches had all of the
following:

- a producer-direct service allocation and sealed-memfd submit path;
- an eight-client persistent-service result path with independent read-only
  views;
- a compiled native owner transfer path using authenticated CREDIT and
  RESULT_DATA frames rather than Python dense sockets;
- a frozen accepted-set record carrying exact payload roots and token weights;
- bounded initial-send-plus-two-replay accounting per peer/root/chunk;
- deterministic second-stage redistribution, global root validation, and
  fenced checkpoint handoff;
- reload and recovery checks that reject provider, build, config, artifact,
  and source digest drift.

## Runtime architecture

### Trainer-to-service handoff

`NativeTrainerDataPlane` attaches to controller-published generation metadata,
asks the persistent service for the final f32 producer lane, and writes the
established K40 delta directly into that mapping. For every sorted model tensor
it casts the trained value to the base tensor dtype, subtracts the unchanged
base, and projects the bounded piece once to f32 wire storage. It then hashes,
seals, and submits the service buffer through the metadata-only RPC. The only
filesystem publication is a small identity/token/digest marker containing
`trainer_spool_bytes=0` and `dense_files_written=0`.

The service retains the immutable descriptor before acknowledging submission,
so the trainer can release its public buffer handle immediately. The service's
shared-byte ledger and metrics are service-wide across all client processes;
eight trainer clients cannot each consume an independent copy of the configured
limit. Once the manager publishes the result operation handle and root, each
trainer independently asks the service for a read-only view, validates the
fence/generation/layout/root/global-token identity, applies bounded chunks, and
closes its own descriptor.

### Exact hierarchical arithmetic

Local trainer submissions preserve rank order by submission sequence. Each f32
source is converted to f64 and multiplied by its exact uint64 token weight. In
the two-node production mode, attempt 1 exposes the token-weighted f64 node
numerator, not a locally divided mean. That numerator crosses the native owner
path with the node's exact token total.

Attempt 2 registers both immutable f64 node numerators. The service recognizes
them as already weighted, adds them without multiplying by the node weight a
second time, sums the exact node token totals for the denominator, divides once,
and projects once to the final f32 result. This prevents a lossy local
divide/reweight round trip.

The direct eight-trainer parity case uses token weights
`[3,5,7,11,13,17,19,23]`, accepted-token total `98`, mixed f32/f64 base
tensors, and eight independent result views. Its output is byte-identical to
the rank-ordered Python f64 reference. The hierarchical case uses node token
totals `1,000,035` and `172` (`1,000,207` globally) and is byte-identical to an
independent single-division reference.

### Frozen owner transfer and result validation

The pool close record freezes both the accepted payload root and accepted token
weight for every node incarnation. Managers install only current-fence,
same-backend, same-bundle endpoint records from that frozen set.

Before each data direction, the receiver sends one authenticated 320-byte
CREDIT frame per exact chunk. The sender validates the run, fence, generation,
attempt, owner epoch, worker/incarnation, layout, base, permitted result root,
weight, offset, extent, chunk count, deadline, and CRC32C before submitting the
data frame. CREDIT is distinct from completion-queue ownership. RESULT_DATA
frames add payload SHA-256 and the frozen result root; receive proceeds directly
into a bounded memfd. Duplicate authenticated chunks are idempotent, conflicting
identity or bounds fail closed, and a `(peer,result-root,chunk)` can be sent at
most initially plus twice for replay.

After both f64 numerators are registered and the final result is projected,
every frozen manager reports `(result_root, global_weight, result_bytes)` to the
metadata coordinator. Publication is authorized only when all frozen reporters
agree and `global_weight` equals the immutable close decision's exact accepted
tokens.

### Checkpoint, migration, reload, and TERM

The native manager emits a checkpoint proposal bound to run, fence, generation,
attempt, layout, base, result root, global weight, result bytes, and publication
generation. Trainer 0 on node 0 applies the same shared result, writes the one
complete model/inner-optimizer checkpoint, and proposes the Python-owned outer
state, migration record, accepted-token clock, membership, and native runtime
and result digests. The manager finalizes the immutable handoff under the
current allocation lease, verifies authoritative-latest CAS identity, waits for
all eight local trainers to publish independent applied/recovery markers, then
commits and releases the native generation.

Leader and independent trainer recovery checkpoints carry native runtime
digests. Resume from an immutable handoff and reload from node-local recovery
both fail closed if provider, provider digest, build bundle, build manifest,
config, source commit, or artifact digest differs. A recovery record written by
a newer fence is rejected; an older-fence node-local record is disposable;
authoritative committed handoff resume under a verified newer allocation fence
remains supported.

On TERM, trainers stop at bounded progress points and leave independent recovery
state. The manager aborts an uncommitted native generation, drains membership,
and closes its local/fabric session. The supervisor then hands TERM to the
persistent service. No all-rank shutdown rendezvous is introduced.

## Digest-bound launch and checkpoint identity

`runtime_digests` records and validates:

- exact provider name and `provider_sha256`;
- native bundle and build-manifest SHA-256;
- immutable E97 config SHA-256;
- attested source commit;
- every service, local-library, transport-library, and gate artifact SHA-256.

The launcher resolves service/client paths from the attested manifest rather
than from `PATH`. The clean canonical build of implementation commit
`e3e88abc43d6896bef52ecab8f0b6ab784475a35` produced bundle
`162fa79f8e2efe3c4976204f732df5d8d7e4a5ba6009f7eb9d345a395292a2f1`
with:

| Artifact | SHA-256 |
|---|---|
| `ndp_cxi_service` | `9f5f55baf14709b3cc08961f4c5e52769401ccf193028a618e3dfdd9650f89de` |
| `libemender_ndp.so.1` | `9514bc5583daa84c3b915f45f15e8c041e7434ed2d18cba7b8ba3fbc63a120fc` |
| `libemender_ndp_transport.so.1` | `7c0637fdc93f0391a7a8ba969f8ab2ef95cc478eec1d248baf9ffc89fc7985d3` |
| `ndp_frontier_2n_gate` | `69a3c471699e26cc0fd0dda35d4c86e4bac1e47a79b4997beeb4801d0dcab9cd` |

The manifest recorded `source_tree_dirty=false`, `RelWithDebInfo`, C/C++
wrappers from Cray PE 2.7.36, tests enabled, and XPMEM enabled. Producer-direct
memfd is the required live local mechanism; XPMEM remains optional.

## Validation

All Python, CMake, CTest, syntax, and checksum validation ran after sourcing
`scripts/frontier/activate_emender_frontier.sh`. No bare Frontier login Python
was used.

### Task acceptance

- [x] Real trainer parity proves the K40 delta cast/subtract/project semantics,
  exact accepted-token total, rank-ordered weighted result, and eight
  independent service result views.
- [x] The two-node launcher starts one persistent native service, one model-free
  manager, and eight GPU trainers per node, for two managers, two services, and
  sixteen trainers total.
- [x] Native production constructs neither `LocalTrainerSpool` nor
  `DistributedOwnerServer`; the debug branch remains explicit and bounded.
- [x] Producer-direct memfd handoff, frozen accepted payload/weight transfer,
  bounded replay, compiled-provider redistribution, and all-reporter result-root
  validation are exercised. Retained synthetic G2 fault job `5031553` supplies
  the prior CXI owner-loss/reassignment evidence for the same bounded mechanism.
- [x] Provider/build/config/source/artifact digests are in generation metadata,
  recovery checkpoints, native proposals, and authoritative checkpoint
  manifests, and reload validates equality.
- [x] The full native/resilient pre-submit selection passes in the approved
  Python environment. The launcher rejects Python TCP and fixed-world MPI for
  production.
- [x] Seed verification, outer-state migration, checkpoint save/reload,
  newer-fence resume, independent recovery, result-identity CAS, and TERM paths
  are covered by the selected tests and source contracts.
- [x] Implementation commit `e3e88abc` was pushed to the WG task branch for the
  authoritative WG completion/integration gate. No Slurm job was submitted.

### Design authority conformance

The implementation was checked against every compute-pool requirement in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`:

- **R01–R04:** fenced allocation ownership, dynamic READY membership,
  generation identity, and one immutable close decision remain Python-owned;
  native endpoints are opaque leased records.
- **R05–R08:** exact token-weighted f64 arithmetic, explicit quorum/deadlines,
  atomic publication, bounded checked owner chunks, explicit credits, replay,
  and prompt release are wired.
- **R09–R12:** managers remain model-free, hot dense state stays off Lustre,
  incarnation changes and disposable unfinished work are preserved, and outer
  state/accepted-token clocks migrate only through committed handoffs.
- **R13–R16:** control remains backend-neutral, stages are bounded and
  observable, parity covers unequal weights, and the retained two-node
  synthetic G2 gate precedes any real or larger-scale run.

It was also checked against every native specialization requirement in
`docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`:

- **NDP01–NDP04:** hard control/dense separation, no elastic collective,
  persistent exact-provider service topology, and direct sealed-memfd producer
  admission.
- **NDP05–NDP08:** exact v1 arithmetic/layout, complete fenced identities,
  leased native routes, and service-wide pre-admission bounds.
- **NDP09–NDP12:** explicit byte credits separate from CQ completion,
  CRC32C/SHA and idempotence, initial-plus-two replay bound, and one shared
  redistributed result.
- **NDP13–NDP16:** absolute deadlines and route-local containment, stable
  metadata-only v1 ABI, fenced Python checkpoint ownership, independent reload,
  and provider/byte/bound/release telemetry.
- **NDP17:** retained clean/fault synthetic G2 evidence is required before the
  downstream startup/real-model ladder; this task submitted no job and claims no
  G3–G6 result.

The traceability status and remaining operational ladder are updated in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`.

### Commands and results

Canonical native build and CTest:

```text
source scripts/frontier/activate_emender_frontier.sh
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh

100% tests passed, 0 tests failed out of 9
```

Full task pre-submit selection:

```text
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_native_artifact_attestation.py \
  tests/test_native_dataplane_2n_controller.py \
  tests/test_native_dataplane_abi.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_dataplane_reference.py \
  tests/test_native_pool_integration.py \
  tests/test_native_transport_bridge.py \
  tests/test_resilient_e97_rank_lane.py \
  tests/test_resilient_e97_reducer.py \
  tests/test_resilient_e97_runtime.py \
  tests/test_resilient_e97_split_roles.py \
  tests/test_resilient_e97_topology.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_node_quorum.py \
  tests/test_resilient_node_transport.py \
  tests/test_resilient_pool_runtime.py \
  tests/test_validate_native_dataplane_2n_gate.py \
  tests/test_validate_native_dataplane_local.py

148 passed in 204.99s
```

Additional checks passed:

```text
python -m py_compile <all changed Python runtime modules and launchers>
bash -n scripts/frontier/resilient_e97_true_2n.sbatch
git diff --check
tests/test_resilient_pool_runtime.py::test_ready_token_floor_distributed_owner_loss_and_late_join
tests/test_resilient_e97_runtime.py::test_live_native_selection_is_wired_and_python_debug_remains_explicit
```

A repository-wide diagnostic collected 626 tests and reported 73 failures,
552 passes, and one skip. One failure was the checksum-linked gap-matrix
reference changed by this task; it was repaired and its complete file then
passed 7/7, followed by the green 148-test selection above. The remaining 72
cached failures are pre-existing, unrelated accelerator/promotion tests
(`complex_eig_triton`, `e97_async_256_promotion`, `gdn2_nonlin_shell`,
`hetero_overlap`, `pin_autotune_parity`, and `typed_head_mixture`). They are
recorded here rather than misrepresented as a passing repository-wide run.

## Retained evidence and next gate

The pre-existing synthetic G2 pair remains:

- clean job `5031461`;
- fault job `5031553`;
- source `1c179e4ba014b2e54989a552fa5c99df010d7bbe`;
- bundle `411d7d92a5e23ea6838b370c0086265cd46d160878b4c2b1b8efc64e88293df1`.

Those jobs established exact CXI/cxi0/FI_EP_RDM selection, replay/fence/token
accounting, owner close/reopen with a new incarnation, bounded replay, stale
epoch rejection, zero partial commit, and terminal release. They do not prove
this later integrated role commit ran on Frontier. The downstream
`validate-native-pool-v1-2n-startup` task must run the exact integrated commit
and retain startup/topology/service evidence before any real E97 G3 attempt.
