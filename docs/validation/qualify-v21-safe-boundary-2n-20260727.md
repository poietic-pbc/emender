# Async-v2.1 safe-boundary exact-two-node qualification

Date: 2026-07-27

WG task: `qualify-v21-safe-boundary-2n`

Status: **PASSED — G2 job 5099135 and clean job 5099195 are terminal
`COMPLETED 0:0`; the semantic verdict and independent final audit pass.**

## Scope and design authorities

This is the strict clean/performance requalification after the reviewed
safe-boundary implementation and the fail-closed corrections found by the
Frontier runner. It authorizes no fault, convergence, promotion, or scale
execution. The runner read and applied:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, including its conformance
  checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including NDP01–NDP17,
  V21S01–V21S17, and ISP01–ISP07;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`;
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`;
- the ISP01–ISP07 immutable-snapshot amendment;
- `docs/validation/fix-v21-safe-boundary-rendezvous-20260726.md`;
- the predecessor exact-two-node async-v2.1 evidence cited there; and
- the failed job-5081295 report under
  `/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/`
  `qualify-simple-async-v21-2n-clean/`
  `35f33399cb60e40d726fc290b5d9d6f524be9ad0/clean/clean-overlap/`.

The permanently retired job-5081295 payload
`34f4404a856cf7df966f76ef7e9e72ac19dd38518be9c0f88e925198034c5d43`
was not reused. The final payload also differs from the immediately preceding
failed payload
`3c9b127fa3170144dc6492230a1b4be7162ce97c21839f24371f4350d41f01f8`.
No fault, convergence, promotion, or scale job was submitted.

## Pass identity and immutable evidence root

The qualifying source was fetched into a standalone clean clone after the
foreground-OWNED safe-boundary correction was reviewed and pushed:

```text
source commit: 46c8043791e9b14c4cb3376c1fb03ebe7fe6932f
source tree:   660f817c0419943d33d07f91573a6abc933bab7a
branch:        main
HEAD:          46c8043791e9b14c4cb3376c1fb03ebe7fe6932f
origin/main:   46c8043791e9b14c4cb3376c1fb03ebe7fe6932f
ls-remote:     46c8043791e9b14c4cb3376c1fb03ebe7fe6932f
status:        clean
```

The exact clone command and equality record are retained in
`source-identity.txt`. The qualification evidence root is:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/
  qualify-v21-safe-boundary-2n/
  46c8043791e9b14c4cb3376c1fb03ebe7fe6932f/
```

The final identities are:

| Authority | Immutable identity |
|---|---|
| source | `46c8043791e9b14c4cb3376c1fb03ebe7fe6932f` |
| source tree | `660f817c0419943d33d07f91573a6abc933bab7a` |
| source digest | `553bc996723bc1698a37f847c981b1b3864d260d5d2d0cc70cb314f5c46e0184` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| native build manifest | `d0f05e6ea15f38e72950680d710d70229b94eee1dfbc8b3468f33431318b82e3` |
| native build log | `4b57f165e1103c97958eb51e5aed0c72d0b7cc91623665fd0f3f1f2679f84dec` |
| full-layout G2 gate | `bc67ed30791c46892aa1d787f50a99b25b9f97bdead69e95a5678ad7cacfe660` |
| clean controller plan | `a113c215781e20a49611638d80a4a48d5923ddfdadbb7d9a2f0d5c01de6a45f7` |
| policy | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| launcher | `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7` |
| clean payload | `46f0ad69d07dffdf277f25d321051b765befa7f034d0462f4bbdae1082b454cf` |
| seed | `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2` |
| seed authority | `27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| training arguments | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| data object | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| semantic verdict | `8f48b370aa83a8e38b052d7e28871c9b3cb44a919150ded15c943578997dd947` |
| clean terminal accounting | `797c5e8d7aa90ed3f9456677fae089d77f42e9d49bc812cbd38070cf8dc3d9c1` |
| independent final audit | `c7acbda4cf24dce9f257f367bec7a358c3257a59b6d8780eb0bb102c5162e1bd` |

`FINAL-SHA256SUMS.txt` has SHA-256
`9e75284859f42c067eadb0446046a92c7b53ebedbcdf677bd90febe413eb8c07`.
It binds the primary source, build, G2, scheduler, seed, controller, semantic,
checkpoint, and audit artifacts. The 1,322-file retained control/telemetry
manifest is `scheduler-evidence/clean-5099195/retained-control-sha256.txt`
with SHA-256
`eeb4c507d9217fceff64b2c2ebb3b5b4a5bf45e75c8046947dc92fb2f8d031fc`.

## Canonical environment, regression gates, and native rebuild

All Python, pytest, native-build, G2, and submission commands first activated
the canonical Frontier environment and then used `"$EMENDER_PYTHON"`:

```bash
export EMENDER_CONDA_ENV=\
/lustre/orion/bif148/scratch/erikgarrison/emender/\
.envs/olcf-rocm711-torch210-py312
source scripts/frontier/activate_emender_frontier.sh
```

The last fail-closed predecessor, clean job `5098017`, completed all ten
commits/checkpoints and all twenty node-applied authorities but its semantic
validator rejected the run because synchronous native submission checksum and
finite scans were charged to the one-second foreground OWNED clock. The
causal snapshot/admission interruptions were milliseconds; ADR-002 assigns
checksum and validation to the immutable background lane. Terminal accounting
was `FAILED 1:0`, elapsed `01:16:06`, exact two-node `batch/debug`, no
restarts. Its terminal evidence SHA-256 is
`3cdbd26e39968e829ba6a8067d4bcc9da3d73ddf56c6e0084297ead8926757c6`.

The regression
`test_native_owned_bound_uses_foreground_admission_not_background_submit_ack`
failed on `28c029714c76abb41dc50219ed80227693540148` and passed after the
smallest correction. Production now records foreground OWNED at immutable
lane admission; later native checksum/finite validation remains background.
The corrected source gates were:

```text
isolated regression/runtime:             2 passed
broad role/launcher/controller/validator: 178 passed
exact/trainer/v2/snapshot:               53 passed
py_compile:                              passed
git diff --check:                        passed
```

The correction was committed as
`46c8043791e9b14c4cb3376c1fb03ebe7fe6932f` and pushed non-force. The
standalone clean clone then rebuilt the native data plane:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

CTest passed `10/10`; the resulting manifest and installed native bundle
match the identities above.

## Fresh exact-source full-layout G2

The exact G2 submission command was:

```bash
NDP_BUILD_MANIFEST="$PWD/build/native-resilient-dataplane/native-artifacts.json" \
NDP_ARTIFACT_ROOT="../g2" \
NDP_RUN_ID="native-g2-clean-46c8043791e9b14c-20260727T214606Z" \
NDP_PAYLOAD_ID="46c8043791e9b14c-e97-g2-clean-20260727T214606Z" \
NDP_PYTHON_BIN="$EMENDER_PYTHON" \
  scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

G2 job `5099135` passed before the clean job was submitted:

```text
job:         5099135
state/exit:  COMPLETED / 0:0
elapsed:     00:03:47
nodes:       frontier[00443,00467]
Nodes:       2
Partition:   batch
QOS:         debug
Restarts:    0
submit:      2026-07-27 17:46:06 scheduler time
start:       2026-07-27 17:46:07 scheduler time
end:         2026-07-27 17:49:54 scheduler time
```

Literal `PENDING`, `RUNNING`, and terminal accounting artifacts are retained
under `scheduler-evidence/g2-5099135/`; their SHA-256 values are,
respectively,
`2a1c5ddd8242b5d33ba78e6089290ef79e7014554fc8e76a432fb517c160f793`,
`c7839e839e081d0c31a810c4181f98d0ffb0ef7b51c0716b4dc6837e14c0c2e6`,
and
`a4fcb1e89ee6ddf599a55a456d2c1033a12d24d074a443e211be709faa749950`.

The G2 gate records `status: "passed"`, source `46c80437`, provider `cxi`,
median `22.823459178 s`, maximum `23.128127191 s`, and speedup
`4.335953012084959x` over the Python reference. Route and CQ errors are zero.
All-rank barriers, MPI collectives, Python dense-socket bytes, trainer-spool
bytes, disk-replay bytes, and handoff full-copy bytes are zero. The gate
explicitly maps R01–R16 and NDP01–NDP17, and attests no central full-model
broker.

## Canonical clean submission and scheduler proof

Only after G2 passed, the canonical serial controller was invoked:

```bash
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean \
  --nodes 2 \
  --repo "$PWD" \
  --native-build-manifest \
    "$PWD/build/native-resilient-dataplane/native-artifacts.json" \
  --full-layout-gate "../g2/5099135/full-layout-gate.json" \
  --run-root "../clean" \
  --state "../controller-state.json" \
  --output "../clean-qualification-plan.json" \
  --submit
```

This submitted exactly one post-G2 clean model job:

```text
job:         5099195
payload:     46f0ad69d07dffdf277f25d321051b765befa7f034d0462f4bbdae1082b454cf
state/exit:  COMPLETED / 0:0
elapsed:     01:18:07
nodes:       frontier[06411,06435]
Nodes:       2
Partition:   batch
QOS:         debug
Restarts:    0
submit:      2026-07-27 17:53:13 scheduler time
eligible:    2026-07-27 17:53:13 scheduler time
start:       2026-07-27 17:53:14 scheduler time
end:         2026-07-27 19:11:21 scheduler time
```

The one-second queue residence ended before the serial controller returned,
so no false literal-`PENDING` claim is made. The queued/accepted transition
artifact retains the exact `SubmitTime`, `EligibleTime`, `StartTime`,
`NumNodes=2`, `Partition=batch`, `QOS=debug`, controller command, payload,
and exact `sbatch` request. The running artifact independently retains
`5099195|RUNNING|2|batch|debug`; terminal `sacct` and `scontrol` retain
`COMPLETED|0:0|...|2|2|224|batch|debug|frontier[06411,06435]|0`.
The queued-transition, running, and terminal SHA-256 values are:

```text
7664d5b43cb41311038e96ea1cbb3d91e78c17c4069cbcd56f0c7f3ed319619a
994fca9cd8f67b48121f5b96589e49d38f9079c75ec5222ba6f3dec287d900bd
797c5e8d7aa90ed3f9456677fae089d77f42e9d49bc812cbd38070cf8dc3d9c1
```

The terminal `SubmitLine` binds source, native manifest, G2 gate, policy,
launcher, payload, seed, data, tokenizer, training arguments, ten
generations, `B:TERM@60`, zero compute-node network fetches, no injection,
and maximum restarts zero.

## Immutable step-2300930 seed

The clean payload and both compute-node receipts bind exactly:

```text
step:          2300930
tokens:        150793748480
size:          7719680116 bytes
SHA-256:       0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
attestation:   27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb
network fetch: 0
```

`sbcast` staged the immutable seed to
`/tmp/emender-e97-seed-5099195/checkpoint-step-2300930.pt`. Both nodes
reverified its bytes and authority offline before launching model processes:

| Node receipt | SHA-256 |
|---|---|
| `seed-materialization/frontier06411.json` | `0cd58935cbaaa9840d99b654a9e106123a002be8c7ffdcffda6474085cb4388e` |
| `seed-materialization/frontier06435.json` | `05353027835de55c85b7c9897b598fd2f53aa092cfc6791c444acb5dbbcc71af` |

Each receipt records the exact step, token count, size, SHA-256,
attestation, `/tmp` path, Slurm job ID, and `network_fetches: 0`.

## Transaction, restart, and checkpoint proof

The clean run retained:

| Evidence | Count/result |
|---|---:|
| persistent real trainers | 16 |
| warm-up K40 windows per trainer | 2 |
| measured K40 windows per trainer | 10 |
| immutable commit authorities | 10 |
| immutable global checkpoints | 10 |
| checkpoint publication/reload attestations | 160 |
| node-applied authorities | 20 |
| rank receipts inside node-applied authorities | 160 |
| candidate-prepared receipts | 160 |
| boundary-ready receipts | 160 |
| native-applied receipts | 160 |
| manager/native apply releases | 20 |
| global leader releases | 10 |
| role restart values | 11,588 records, all zero |

For every generation on each node, exactly eight
`native-candidate-prepared` receipts precede eight distinct
`native-boundary-ready` receipts; preparation never substitutes for boundary
readiness. The manager release follows the all-eight rendezvous. Exactly
eight `native-applied` receipts then precede one `node-applied` marker.
The twenty immutable node authorities cover both nodes and generations
1–10, each contains ranks 0–7, and every trainer incarnation stays stable.
The ten global leader-release records prove once-only transaction release.

Every trainer has ten `checkpoint_publication` records with
`reload_verified=true`, `latest_cas_verified=true`, `within_slo=true`,
`foreground_blocking=false`, and zero foreground component. All 160
safe-boundary apply records are also reload/latest-CAS verified. Generation
0 telemetry corresponds to immutable checkpoint/version 1 and generation 9
to checkpoint/version 10. The ten independently retained checkpoint hashes
are in `scheduler-evidence/clean-5099195/checkpoint-sha256.txt`; every file
is `2753437091` bytes. The terminal checkpoint SHA-256 is
`05f766ebf0be03d34f4d0ddbe8e070ece65241387ed251765c4d749b06643753`,
and both generation-10 node authorities bind it in all eight rank receipts.

Fifteen terminal per-rank local recovery JSON files legitimately remain at
generation 9 after terminal drain; one is generation 10. They are local
rank-state snapshots, not the common generation-10 recovery authority. The
rank-complete generation-10 apply authorities and the 160
reload-verification records are the authoritative terminal proof. All 16
local recovery files still bind the current fence, run, payload, source,
native manifest, and native bundle.

The semantic validator and independent audit find no partial apply,
unintended restart, stale or superseded identity, generation gap, double x/z
correction, missing rank receipt, timeout, or terminal-signal handoff.

## Performance, hard tails, and bounded resources

The machine semantic verdict is
`clean/clean-overlap/pipelined-performance.json`, SHA-256
`8f48b370aa83a8e38b052d7e28871c9b3cb44a919150ded15c943578997dd947`.
It records `status: "passed"`, `correctness.passed: true`, and
`training_lane.passed: true`.

| Criterion | Observed | Bound/result |
|---|---:|---:|
| raw K40 compute | `83.674129394 s` | retained baseline |
| steady-state cadence | `84.026201556 s` | `1.004207659x <= 1.25x` |
| aggregate foreground idle | `0.0291898862` | `< 0.10` |
| foreground idle maximum / p99 | `32.013019359 / 31.286812645 s` | hard-tail pass |
| unattributed idle maximum / p99 | `1.800887143 / 1.800848040 s` | hard-tail pass |
| foreground result wait | `0 s` | exactly zero |
| native foreground OWNED maximum | `0.020068374 s` | `<= 1 s` |
| snapshot/admission maximum / p99 | `0.020068374 / 0.015315349 s` | `<= 1 s` |
| causal snapshot admission maximum | `0.003382866 s` | `<= 1 s` |
| candidate preparation maximum / p99 | `62.851583992 / 61.707595214 s` | `<= 420 s` |
| boundary rendezvous maximum / p99 | `81.790556243 / 77.305381074 s` | `<= 420 s` |
| manager rendezvous maximum / p99 | `94.825746474 / 94.825746474 s` | `<= 420 s` |
| freeze-to-latest maximum | `6.170150186 s` | `<= 420 s` |
| release-to-apply maximum / p99 | `1.845651751 / 1.842243097 s` | `<= 60 s` |
| total safe-boundary idle maximum / p99 | `83.606101789 / 79.019823153 s` | `<= 480 s` |

The 60-second apply clock begins at the all-eight boundary-ready release,
not at candidate preparation. All 160 rank applies finish within that clock.
The validator evaluates maximum and p99, so a median cannot conceal the hard
tail.

High-water facts are bounded: result mailbox `1`, sealed descriptor `1`,
result staging `0`, mutable interval `0`, and mutable window `0`. Commit and
anchor lags are at most `1`; result-version and speculative lags are `0`.
Runtime and G2 evidence record zero Python dense-socket bytes, Lustre dense
hot-path bytes, trainer-spool bytes, disk-replay bytes/files, MPI
collectives, and all-rank barriers. There is no shared SQLite control plane,
central full-model broker, disk replay, or all-rank collective. Dense bytes
remain in the bounded native CXI/data-plane path.

## Independent final audit and machine verdict

After terminal accounting and the semantic validator passed, an independent
fail-closed audit reloaded the immutable artifacts and passed 544 checks. It
verified source/build/G2/payload/seed identities, exact scheduler state,
every control receipt and incarnation, all checkpoint/reload attestations,
all performance bounds, zero restarts, and forbidden-path counters. Its
machine-readable output is:

```text
scheduler-evidence/clean-5099195/final-validation.txt
SHA-256 c7acbda4cf24dce9f257f367bec7a358c3257a59b6d8780eb0bb102c5162e1bd
passed=true
```

The repository companion
`docs/validation/qualify-v21-safe-boundary-2n-20260727.json` is the compact
machine-readable pass verdict. This report and that verdict are committed
and pushed non-force; the exact publication commit and verified
`HEAD=origin/main=ls-remote main` equality are retained in the WG task log.

## Compute-pool conformance checklist

| Checklist item | Immutable command/artifact and result |
|---|---|
| Authority and scope | The authority list above and accepted v2.1 policy/schema in `clean-qualification-plan.json`; clean-only, no downstream jobs. |
| Clean source | `source-identity.txt` binds clean commit `46c80437`, tree `660f817c`, origin, and `ls-remote`. |
| Native build | Canonical activated environment; `build_native_resilient_dataplane.sh`; CTest `10/10`; manifest `d0f05e...`. |
| Exact two-node G2 | Job `5099135`; literal queued/running/terminal `2/batch/debug`; gate `bc67ed...`; native CXI and exact-reference pass. |
| Exact two-node clean | Job `5099195`; accepted/running/terminal `2/batch/debug`; `COMPLETED 0:0`; payload `46f0ad...`. |
| Immutable seed | Two node receipts bind step `2300930`, tokens, size, SHA, attestation, `/tmp` sbcast, and network fetches zero. |
| Persistent roles and membership | Two managers, 16 trainers, stable incarnations, leased READY current fence, zero restarts. |
| Atomic result/apply | Ten commits/checkpoints; twenty node authorities; 160 complete rank receipts; no partial marker. |
| Safe boundary | Candidate preparation, distinct all-eight boundary readiness, manager release, eight applies, and one marker for every node transaction. |
| Performance | Semantic verdict passes cadence, idle, zero result wait, snapshot, release/apply, max, and p99 bounds. |
| Bounded state/data plane | Capacity/high-water evidence and zero forbidden path counters; native CXI; no shared broker/SQLite/collective/replay. |
| Restart verification | 160 immutable checkpoint publication/reload attestations, 160 apply reload attestations, independent checkpoint hashes. |
| Publication | `FINAL-SHA256SUMS.txt`, 1,322-file retained-control manifest, repository report/verdict, non-force push, remote agreement in WG log. |

## R01–R16

| ID | Qualification evidence |
|---|---|
| R01 | Source identity, payload source digest, build manifest, launcher, policy, data, tokenizer, seed, and every runtime recovery identity are digest-bound. |
| R02 | Exact step-2300930 seed is sbcast to node-local `/tmp`, independently reverified on both nodes, and records `network_fetches=0`. |
| R03 | G2 and clean scheduler artifacts explicitly prove exactly two nodes and separate `Partition=batch`, `QOS=debug` fields through terminal state. |
| R04 | Two persistent node managers and 16 real trainers use stable incarnations; 11,588 restart-bearing supervision records are all zero. |
| R05 | Native G2 exact analytical f64 reference passes; payload weights and exact positive-token floor are immutable. |
| R06 | Each transaction freezes exact current-fence contributions from both READY node workers before commit; no launched-rank quorum shortcut exists. |
| R07 | Ten immutable commit/checkpoint authorities are retained and independently hashed; 160 reload attestations prove restartability. |
| R08 | Candidate preparation is immutable and distinct from all-eight boundary readiness; all 160 preparations and 160 boundary receipts exist. |
| R09 | Manager release follows all-eight readiness; 160 native applies and twenty once-only node markers prove atomic x/z application. |
| R10 | All node authorities contain ranks 0–7 and stable incarnations; no partial apply, stale identity, generation gap, or duplicate correction is present. |
| R11 | Raw K40 windows for all 16 trainers prove `1.00421x` cadence and `0.02919` aggregate idle; foreground result wait is zero. |
| R12 | Maximum and p99 phase/tail metrics are explicit; all snapshot, boundary, manager, result, apply, and total-idle bounds pass. |
| R13 | Immutable mailbox/descriptor/mutable state high-water values and credits are finite; full-model dense bytes never enter Python/Lustre control paths. |
| R14 | Production provider is native CXI; G2 has zero CQ/route errors, barriers, MPI collectives, spool, replay, and full-copy handoff. |
| R15 | Terminal semantic and independent machine verdicts pass; exact artifact paths and SHA-256 manifests are retained and published. |
| R16 | This exact-two-node clean pass is necessary only; it does not itself execute or pass fault, convergence, promotion, or scale gates. |

## NDP01–NDP17

| ID | Qualification evidence |
|---|---|
| NDP01 | Payload and launch attestation bind the compiled native CXI dense transport; Python carries fenced control metadata only. |
| NDP02 | G2 records `all_rank_barriers=0` and `mpi_collectives=0`; production uses point-to-point node service routes. |
| NDP03 | Exactly two persistent model-free native service endpoints, one per node, are retained. |
| NDP04 | Trainers publish sealed immutable candidates; background hashing, transport, aggregation, checkpoint, and validation never read mutable live state. |
| NDP05 | Native CTest `10/10` and fresh full-layout G2 `5099135` bind exact encoding, E97 layout, weights, and deterministic arithmetic. |
| NDP06 | Requests/results/receipts bind run, allocation fence, generation, worker/rank/incarnation, code, policy, layout, payload, and digests. |
| NDP07 | G2 endpoint/route artifacts and live current-fence manager/native telemetry bind production `cxi` membership. |
| NDP08 | Native resident, transport in-flight/retained, slot, mailbox, and descriptor high-water values are finite. |
| NDP09 | Credits/fabric completion and producer-direct publication are explicit; immutable background events have zero foreground component. |
| NDP10 | CRC/SHA/reload/latest-CAS and once-only receipts reject corruption, replay, and conflicting duplicates. |
| NDP11 | Replay/reassignment bounds are finite; the clean pass has no owner loss, restart, stale replay, or expired identity. |
| NDP12 | Trainers consume node-service results; G2 attests no central full-model broker and runtime has no dense fan-out files. |
| NDP13 | Startup, contribution, checkpoint/result, candidate, boundary, apply, and receipt publication use distinct absolute deadlines. |
| NDP14 | Native C ABI, bundle, libraries, service, gate binary, provider, and compact control metadata are separately attested. |
| NDP15 | Immutable preparation/reload precedes all-eight boundary release; eight applies precede one node marker and checkpoint lineage. |
| NDP16 | Per-role/native JSONL, semantic verdict, independent audit, max/p99 metrics, counters, identities, and SHA manifests are retained. |
| NDP17 | Exact-source build/CTest, G2, and clean jobs all pass; clean terminal state is `COMPLETED 0:0`. |

## V21S01–V21S17

| ID | Qualification evidence |
|---|---|
| V21S01 | Policy `async-decoupled-v2.1-simple`, schema `emender-async-policy-v2.1`, code, launcher, bundle, payload, seed, data, and tokenizer are pinned; no v2.0 evidence is relabeled. |
| V21S02 | Commit/anchor lag maximum is one, result/speculative lag zero, and foreground result wait outside bounded phases is zero. |
| V21S03 | Positive exact tokens are the sole contribution floor, weight, denominator, and accepted-token clock. |
| V21S04 | All 16 real trainers run K40 with stateless `eta_outer=1.0`; each has two warm-up and ten measured raw windows. |
| V21S05 | Worker/incarnation/window/base/policy/layout/code/token identity and exact two-node Q/T policy are bound in every accepted transaction. |
| V21S06 | Sixteen persistent trainers exclusively own mutable model/optimizer/iterator state and resume K work after immutable admission. |
| V21S07 | Verified corrections translate ScheduleFree x/z/interval exactly once at the released safe K boundary; all releases/applies pass 60 seconds. |
| V21S08 | Only current-fence, reload/latest-CAS-verified results enter the capacity-one mailbox; invalid/full/late paths cannot block foreground K. |
| V21S09 | Resident, credit, replay, receipt, mailbox, descriptor, mutable-window, and deadline capacities/high-waters are finite. |
| V21S10 | Membership is leased READY; the pass has stable incarnations and no stale identity, expiration, generation gap, one-node authority, or restart. |
| V21S11 | Every accepted node transaction proves eight preparations, eight distinct boundary-ready receipts, manager release, eight applies, and one marker. |
| V21S12 | Production path is compiled native CXI with producer-direct immutable inputs and bounded point-to-point redistribution; all forbidden paths are zero. |
| V21S13 | Raw causal phase, cadence, maximum, p99, idle, lag, and high-water evidence is recomputed fail-closed by the semantic validator and independent audit. |
| V21S14 | Model/outer/token/identity/lag/apply bundles and digest-linked receipts are reload/restart verified; cold start is the exact offline step-2300930 seed. |
| V21S15 | Fresh G2 `5099135` and clean `5099195` are exact-two-node `batch/debug`; downstream fault/replay/convergence is outside this pass. |
| V21S16 | This report creates no promotion authorization or 4+ node rung pass. |
| V21S17 | Scale-only leased-READY finite-close logic is unchanged and unexercised; no launched-rank or `Q_min` early close is claimed. |

## ISP01–ISP07

| ID | Qualification evidence |
|---|---|
| ISP01 | The owning trainer captures coherent immutable candidates; sealed digests and background telemetry prove no mutable live-model read after admission. |
| ISP02 | Snapshot/admission maximum is `0.020068374 s <= 1 s`; raw K telemetry proves mutable training resumes while immutable background work proceeds. |
| ISP03 | Native publication, hashing, aggregation, validation, and checkpoint consume immutable inputs; 160 checkpoint records are non-foreground and reload verified. |
| ISP04 | Mailbox `1`, descriptor `1`, staging `0`, mutable interval/window `0`, credits, replay, receipts, and resident bytes are bounded with no spill. |
| ISP05 | Preparation never substitutes for boundary readiness; eight boundary receipts precede release, and eight applies precede one node marker. |
| ISP06 | Freeze, admission, publish/network, aggregation, checkpoint, result wait, apply/swap, and total idle have distinct causal clocks; maximum and p99 are retained. |
| ISP07 | The validator rejects hidden hard tails; the live pass uses raw events and passes cadence, idle, maximum, p99, snapshot, and released-apply criteria. |

## Final gate result

Every checklist item passes. G2 `5099135`, clean `5099195`, semantic verdict,
and independent audit are immutable and mutually bound to source `46c80437`
and payload `46f0ad69...`. This task is complete. It leaves fault,
convergence, promotion, and scale execution to their separately authorized
downstream gates.
