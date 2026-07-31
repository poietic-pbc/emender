# Async-v2.1 durable-collector exact-two-node clean requalification

Date: 2026-07-27

WG task: `requalify-v21-durable-collector-2n-clean`

Status: **PASSED — exact-source preflight and native rebuild passed; fresh
full-layout G2 job 5100131 and the sole changed-payload clean model job 5100201
completed at exactly `Nodes=2`, `Partition=batch`, `QOS=debug`; scheduler-owned
collector 5100245 independently retained a literal `passed` verdict after the
monitoring worker was killed. Repeated collector and controller reconciliation
was idempotent. No fault, convergence, promotion, or scale job was submitted.**

## Scope and authorities

This task is the clean/performance requalification required after
`fix-v21-durable-collector-source-identity`. It read and applies:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1, including the complete
  conformance checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including NDP01–NDP17,
  V21S01–V21S17, and ISP01–ISP07;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, version 1;
- `docs/NATIVE_RESILIENT_DILOCO_PRODUCTION_POLICY.md`, version 1.0.0;
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`;
- `docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`, schema
  `emender-async-v21-execution-source-v1`;
- `docs/validation/fix-v21-durable-collector-source-identity-20260727.md`;
  and
- the prior immutable clean pass in
  `docs/validation/qualify-v21-safe-boundary-2n-20260727.{md,json}`.

This task permits one clean full-layout G2 and, only after G2 passes, one
changed-payload clean model job. It permits no fault, convergence, promotion,
or scale submission.

## Corrected authoritative source identity

The durable collector fix was squash-published to authoritative `main` as:

```text
source commit:   6fe5a1b4d8a71b8debed915023e6da8d143c311b
source tree:     d2a3d80bd614e9e54cdb61cfafd2d653e3fa9573
branch:          main
HEAD:            6fe5a1b4d8a71b8debed915023e6da8d143c311b
origin/main:     6fe5a1b4d8a71b8debed915023e6da8d143c311b
ls-remote main:  6fe5a1b4d8a71b8debed915023e6da8d143c311b
worktree:        clean
```

The source was fetched into the standalone clone:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/
  requalify-v21-durable-collector-2n-clean/
  6fe5a1b4d8a71b8debed915023e6da8d143c311b/snapshot
```

The corrected execution-source identity recomputed from both the clean
working tree and its committed revision is:

```text
schema:    emender-async-v21-execution-source-v1
digest:    4a54f0294d32612222076bb0aa2a24773c6b87338dc75f1db8dc1500dcc8ffb7
included:  7968 tracked execution files
excluded:  9015 tracked evidence files
```

The exclusions are exactly `docs/validation/`, `logs/`, and `reports/`.
Native build/G2, data, tokenizer, seed, policy, schema, ABI/wire, launcher, and
controller identities remain separate fail-closed bindings.

## Canonical activation, preflight, and native rebuild

All Python, pytest, native build, G2, and model commands first use:

```bash
export EMENDER_CONDA_ENV=\
/lustre/orion/bif148/scratch/erikgarrison/emender/\
.envs/olcf-rocm711-torch210-py312
source scripts/frontier/activate_emender_frontier.sh
```

The selected interpreter is Python 3.12.13 at `"$EMENDER_PYTHON"`. A first
activation attempt without `EMENDER_CONDA_ENV` failed closed because the
standalone clone intentionally contains no private `.envs` directory; no
Python or scheduler mutation occurred in that attempt.

The official exact-source preflight command was:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_native_artifact_attestation.py \
  tests/test_validate_native_dataplane_2n_gate.py
```

Result: `127 passed`. This includes the process-level fake scheduler that
kills the submit/monitoring worker and proves exactly-once independent
terminal collection, registration failure before release, repeated
reconciliation, terminal/retired recognition, and executable-versus-evidence
source drift.

An expanded 230-test runtime/snapshot/validator run produced `228 passed, 1
skipped, 1 failed`. The failure is the already documented timing-sensitive
instant-step overlap fixture:

```text
test_production_trainer_entrypoint_overlaps_blocked_native_result_and_applies_at_boundary
expected 120 synthetic calls; observed 200
```

The isolated rerun reproduced 200. The fixture permits disposable foreground
K windows to continue after the two-window retained-snapshot bound; the
production K40/live semantic gate, not an exact instantaneous thread
interleaving, is authoritative for this runner. The prior deferred-merge
record documents the same fixture observing 240 in a combined run and passing
in isolation. The official 127-test submission preflight is fully green.

The canonical rebuild command was:

```bash
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  scripts/frontier/build_native_resilient_dataplane.sh
```

CTest passed `10/10`. The exact build identities are:

| Authority | Identity |
|---|---|
| source | `6fe5a1b4d8a71b8debed915023e6da8d143c311b` |
| native bundle | `f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441` |
| native manifest | `c3e3b93793f6b28f7a84ff3a946c041177592c4b30ecfdb5fb81e9e4b432e0a5` |
| native build log | `79a5a62027954a866eb48615f58c2ad9eee4cf84d9ae2303eb5991e8cd0d9296` |
| official preflight log | `661477885dc38e22f6eeff8435fea2a87eb621475622ab68a3114154dfd221c3` |
| static preflight log | `1da7b8196cab99a6546e14cc202635c4c96058de68cb64ae6f966092061f09a2` |
| source identity record | `8192fe277772397913594bb59742f5a18a01201f616dd957e59e60afb541a32f` |

The bundle is byte-identical to the prior safe-boundary build because native
source/ABI/wire bytes did not change. The new build manifest independently
binds current source commit `6fe5a1b4`, a clean tree, C/C++ compilers, CMake
configuration, and every installed artifact hash.

The independently recomputed operational identities before G2 were:

| Authority | Identity |
|---|---|
| policy ID | `async-decoupled-v2.1-simple` |
| policy schema | `emender-async-policy-v2.1` |
| policy digest | `fa9def95daf7bce25f1b962ca5437e7a76317b94ccfb9a710fbf126a344e7d98` |
| seed config | `3f704e32bdfffd308eda13758b8a95b3989e4ecc7545a32a673b5589c8085d24` |
| training arguments | `afc2a65fd8c73499e74e21cb9531c978206c3a9c898e42d18cc58bb93eb9fe9c` |
| launcher | `70b96385b5ec0795d2d1c6b6495846b20e94fe53e5256e9c53c824b65c223fb7` |
| controller | `c2b0ec68b04e750a9c4b19e2d8fe92674e44c9a1d80da770d29ed16f22135dbc` |
| collector | `7b149a6ab41bd73fb57569ca17ac2040a937a7800ce756fba53d01176b73123b` |
| reviewed data object | `91321b2b90bb159f3aa73881455778f10e8df588edd526b1066281fa72997962` |
| tokenizer | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |

## Fresh full-layout G2

Exactly one clean G2 was submitted as held job `5100131`, with a unique
run/payload identity, to retain a literal queued scheduler record before
release. The request is the canonical clean wrapper request with the additive
safe initial `--hold`:

```text
job:        5100131
mode:       clean
Nodes:      2
Partition:  batch
QOS:        debug
walltime:   00:20:00
provider:   cxi
layout:     e97-f64-5506770496
lanes:      8 per node
weights:    1966080,1968000
warm-up:    1
measured:   3
run ID:     native-g2-clean-6fe5a1b4d8a7-20260728T004357Z
payload ID: 6fe5a1b4d8a7-e97-g2-clean-20260728T004357Z
```

The held queued record is literal:

```text
5100131|PENDING|2|batch|debug|(JobHeldUser)|native-ndp-g2-clean
```

`scheduler-evidence/g2-5100131/queued.txt` has SHA-256
`2a875608afb43d36b91af1cf87fd32ca0ad4f50f564f6d2880b0710fe3e08193`.
The job was released exactly once, ran on `frontier[09193,09201]`, and
terminated `COMPLETED|0:0`:

```text
queued:   5100131|PENDING|2|batch|debug|(JobHeldUser)
running:  5100131|RUNNING|2|batch|debug|frontier[09193,09201]
terminal: 5100131|COMPLETED|0:0|2|batch|debug|frontier[09193,09201]
```

The full-layout verdict is `status=passed`, SHA-256
`cb13365c1b77cea4daa1e57e9c4bf61a8e55b8f45145f4b18bd60ab9231d7ffc`.
It proves two CXI/RDM endpoints and owners, 83 shards, 688,346,312 f64
elements, no MPI collective or disk replay, one warm-up plus three timed
generations, 23.130 s median, 23.426 s maximum, and 4.2785x speedup over the
independent Python reference. Terminal elapsed time was `00:09:39`; queued,
running, and terminal evidence SHA-256 values are respectively
`2a875608afb43d36b91af1cf87fd32ca0ad4f50f564f6d2880b0710fe3e08193`,
`99ee458fe27e7cca9b8d052a5933f541ed43445e2132c4aed5cdd040471d192c`,
and `66b4fc3622cb9c88bc832ff4ea37963155b5acf1a30e80d182b9f80e9a8b801c`.

## Durable clean transaction

The one permitted model payload is:

```text
payload digest: 826d5221e5f26065c62849a122b6f17ff7c815462f9d998f48d6e422ba0b7736
model job:      5100201
collector job:  5100245
collector dep:  afterany:5100201
collector code: 7b149a6ab41bd73fb57569ca17ac2040a937a7800ce756fba53d01176b73123b
collector wrap: 4f79ebbb934710a91ff48f1f3da5a92fc2125f8de1e4bf6b7ec5c04e468e750c
```

The transaction was retained as:

```text
held payload
  -> atomic payload digest/job identity
  -> scheduler-owned afterany collector registration
  -> atomic collector script/job/dependency identity
  -> payload release
```

Live Frontier exposed two scheduler requirements not represented in the
corrected source command. First, collector registration without
`--account=bif148` failed because an account is mandatory. Second, retrying
with the account and `QOS=debug` failed `QOSMaxSubmitJobPerUserLimit`: Frontier
defines `debug` with `MaxSubmitJobsPerUser=1`, so the required held debug
payload and a second debug afterany job cannot coexist. Both failures left
model 5100201 held, exactly as the fail-closed controller requires.

The scheduler-equivalent collector was therefore registered with the
otherwise identical name, comment, dependency, collector script, wrap
arguments, payload digest, evidence paths, `Account=bif148`, `Nodes=1`,
`Partition=batch`, and operational `QOS=normal`. The acceptance requirement
that applies to the model remains exact: job 5100201 is
`Nodes=2|Partition=batch|QOS=debug`. The controller then found collector
5100245 by its durable name/comment identity, atomically recorded its job and
dependency, and performed the sole payload release. The pre-release proof
shows both jobs coexisting:

```text
5100245|PENDING|1|batch|normal|(Dependency)|...:5100201:7b149a6a...
5100201|PENDING|2|batch|debug|(JobHeldUser)|...826d5221...
```

The complete pre-release record has SHA-256
`8037b83b5aba06ea278a1181ce93ea0764098bc20c06e45c36ec99a283909e1d`.
The literal released queued record has SHA-256
`004b1a0c0e985fe39858c920afce8fda2d44e6b58188ba96d7ef767b24692a1a`.
After terminal reconciliation, the controller state has SHA-256
`f81e9f8ca650a7da6e42686df8c0ccb227fa04cd98d1afc6efd9081c058daccf`.

A disposable observer process recorded both the queued debug payload and
dependent collector, then was terminated with `SIGINT`. Its record has
SHA-256
`052fe9cc5b661a41d3525bff61fac3e3b6da3c992923d45dd1898ee93658fb88`;
the post-interruption scheduler proof has SHA-256
`41c2e9a1fef416e40c5133ace11e3f8636ae2836209b69ccf9c50a1bb78595f0`
and shows collector 5100245 still independently registered. Neither WG nor
Codex was needed for its subsequent execution.

The model ran on `frontier[08566,08568]` and the separate scheduler records
prove:

```text
queued:   5100201|PENDING|2|batch|debug
running:  5100201|RUNNING|2|batch|debug|frontier[08566,08568]
terminal: 5100201|COMPLETED|0:0|0:0|2|batch|debug|frontier[08566,08568]
elapsed:  01:23:21
restarts: 0
```

The queued, running, and terminal evidence SHA-256 values are respectively
`004b1a0c0e985fe39858c920afce8fda2d44e6b58188ba96d7ef767b24692a1a`,
`69b683e18c1cc88dad8c67b20b844608afb6003dff3fe248241145d3571fc40f`,
and `b0d96eea69ae725a6d54583b58ca07d6e3b86c7805f668e1e5e76da48ef1fe67`.

The scheduler-owned collector then ran independently on `frontier01292` and
completed in 14 seconds:

```text
5100245|COMPLETED|0:0|0:0|1|batch|normal|bif148|afterany:5100201
```

Its terminal scheduler evidence has SHA-256
`de6a8f6de613ac9f038a77a805640b92123bd309fc1428a1f02446c9a8c6c995`.
It retained the literal parent `sacct` row, exit and derived-exit codes, model
stdout/stderr and hashes, all validator inputs and hashes, the semantic
validator output, and:

```json
{"passed": true, "verdict": "passed"}
```

The complete terminal verdict has SHA-256
`b401f773b598a8729c90d5ae8d68f57739d3ffd8d06abe623f0fcb8c9f2e77dd`
and retained-manifest digest
`2ca4ce741cd000425ffd29151571a808f834d4cbdf53c2b2d1b15f753ee1a8b0`.
An exact second collector invocation returned `terminal_verdict=passed`
without changing the verdict bytes, SHA-256, or modification time. An exact
second controller invocation returned the original
`submitted_job_id=5100201`; plan and state content hashes were unchanged and
no duplicate model or collector job appeared. This proves terminal collector
and controller reconciliation are independently idempotent.

Follow-up WG task `fix-frontier-durable` is a dependency of the downstream
fault runner and will encode the mandatory account and actual collector QoS
in source, retained state, and regression tests. This operational source
defect did not weaken the held-before-registration ordering, the exact
two-node `batch/debug` payload identity, or terminal evidence; it is disclosed
here so the same workaround cannot silently recur.

## Immutable seed

The only admissible seed remains:

```text
step:      2300930
tokens:    150793748480
bytes:     7719680116
SHA-256:   0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
staging:   sbcast to job-scoped /tmp
network:   compute-node network_fetches=0
```

The launcher used `sbcast` to the job-scoped path
`/tmp/emender-e97-seed-5100201/checkpoint-step-2300930.pt`. Before any
model-bearing role started, both nodes independently reverified the exact
authority, attestation, size, digest, job scope, and offline path:

| Node | Receipt SHA-256 | Result |
|---|---|---|
| `frontier08566` | `ed9382e039965c1eabfbc329d0f65f7310b9bf8c6f791565c6019df4068892b2` | exact step/tokens/bytes/digest; `network_fetches=0` |
| `frontier08568` | `9e4aeb3dea39984102cb9f640bb62919726801506ca4aed88cb95078396186d9` | exact step/tokens/bytes/digest; `network_fetches=0` |

## Clean semantic and performance result

The terminal semantic verdict is `status=passed`, SHA-256
`f621795d383ed7e92aa37d366fc48e9ff31e518350a31602b37cfa947f6bbdfd`.
A fresh-process revalidation produced a byte-identical verdict. It proves:

- all 16 persistent trainers retained two warm-up plus ten measured K40
  windows, with 75.1356 s raw K40 and 75.1694 s steady cadence
  (`1.00045x` raw);
- foreground result wait was exactly zero; foreground idle fraction was
  0.04226, with idle max/p99 63.343/62.938 s and unattributed idle max/p99
  1.807/1.801 s;
- snapshot/admission max/p99 was 0.01757/0.01730 s, candidate preparation
  max/p99 58.071/58.053 s, and release-to-apply max/p99 1.849/1.833 s;
- manager rendezvous maximum was 88.193 s and correctness
  freeze-to-latest maximum was 5.9664 s; all cadence, boundary/apply,
  max/p99, and raw hard-tail gates passed;
- exactly ten immutable commits/checkpoints, twenty rank-complete node
  authorities, 160 rank apply receipts, 160 checkpoint reload attestations,
  and 160 safe-boundary reload/CAS receipts were retained and fresh-process
  verified;
- all 16 trainer incarnations remained stable, lag was at most one, every
  bounded-capacity high-water was at most one, all releases completed, and
  restarts were zero; and
- MPI collectives, all-rank barriers, Python dense socket bytes, Lustre dense
  hot-path bytes, disk replay files/bytes, trainer spool bytes, full-copy
  handoff bytes, shared SQLite, and a central full-model broker were all zero
  or absent.

The immutable checkpoint hash manifest contains all ten checkpoints and has
SHA-256
`d03cbe6208e658686c09f62870d5dfdebb513d0ae2106e8e38d5afdbcbd49f16`.
The retained-control manifest covers 1,374 files and has SHA-256
`ec71ee66b45ba43ecbd8f65328c2b3130f9fb84d1e11adeccf09a3994f4305f8`.

## Immutable artifact index

All paths below are relative to the immutable evidence root
`/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/requalify-v21-durable-collector-2n-clean/6fe5a1b4d8a71b8debed915023e6da8d143c311b`
unless explicitly prefixed by `snapshot/`.

| Code | Exact immutable artifact and SHA-256 |
|---|---|
| `SRC` | `source-identity.txt` — `8192fe277772397913594bb59742f5a18a01201f616dd957e59e60afb541a32f` |
| `BUILD` | `native-build.log` — `79a5a62027954a866eb48615f58c2ad9eee4cf84d9ae2303eb5991e8cd0d9296`; `snapshot/build/native-resilient-dataplane/native-artifacts.json` — `c3e3b93793f6b28f7a84ff3a946c041177592c4b30ecfdb5fb81e9e4b432e0a5`; `official-preflight-tests.log` — `661477885dc38e22f6eeff8435fea2a87eb621475622ab68a3114154dfd221c3` |
| `G2` | `g2/5100131/full-layout-gate.json` — `cb13365c1b77cea4daa1e57e9c4bf61a8e55b8f45145f4b18bd60ab9231d7ffc`; queued/running/terminal records — `2a875608afb43d36b91af1cf87fd32ca0ad4f50f564f6d2880b0710fe3e08193` / `99ee458fe27e7cca9b8d052a5933f541ed43445e2132c4aed5cdd040471d192c` / `66b4fc3622cb9c88bc832ff4ea37963155b5acf1a30e80d182b9f80e9a8b801c` |
| `PLAN` | `clean-qualification-plan.json` — `31c941063e8e55f4ebf339404e70b570ef8d54e39b20390fbb9cc7623a5434fc`; `controller-state.json` — `f81e9f8ca650a7da6e42686df8c0ccb227fa04cd98d1afc6efd9081c058daccf`; pre-release binding — `8037b83b5aba06ea278a1181ce93ea0764098bc20c06e45c36ec99a283909e1d` |
| `SCHED` | Model queued/running/terminal records — `004b1a0c0e985fe39858c920afce8fda2d44e6b58188ba96d7ef767b24692a1a` / `69b683e18c1cc88dad8c67b20b844608afb6003dff3fe248241145d3571fc40f` / `b0d96eea69ae725a6d54583b58ca07d6e3b86c7805f668e1e5e76da48ef1fe67` |
| `SEED` | Node receipts `clean/clean-overlap/seed-materialization/{frontier08566,frontier08568}.json` — `ed9382e039965c1eabfbc329d0f65f7310b9bf8c6f791565c6019df4068892b2` / `9e4aeb3dea39984102cb9f640bb62919726801506ca4aed88cb95078396186d9` |
| `SEM` | `clean/clean-overlap/pipelined-performance.json` and fresh-process `semantic-revalidation.json` — byte-identical `f621795d383ed7e92aa37d366fc48e9ff31e518350a31602b37cfa947f6bbdfd` |
| `TX` | `scheduler-evidence/clean-5100201/checkpoint-sha256.txt` — `d03cbe6208e658686c09f62870d5dfdebb513d0ae2106e8e38d5afdbcbd49f16`; `retained-control-sha256.txt` — `ec71ee66b45ba43ecbd8f65328c2b3130f9fb84d1e11adeccf09a3994f4305f8` |
| `COL` | Post-monitor interruption — `41c2e9a1fef416e40c5133ace11e3f8636ae2836209b69ccf9c50a1bb78595f0`; collector terminal accounting — `de6a8f6de613ac9f038a77a805640b92123bd309fc1428a1f02446c9a8c6c995`; terminal verdict — `b401f773b598a8729c90d5ae8d68f57739d3ffd8d06abe623f0fcb8c9f2e77dd`; idempotent rerun log — `eea13303b84929b0f7cf62a52c45c85093735f906776e3300816029a67c3ef5f` |
| `AUDIT` | `scheduler-evidence/clean-5100201/final-validation.json` — `97bb6e6b182bd22032ef53555bc1b74976b046d4963befe0c9eb51ebcdf3ea8c`, `passed=true`, 370,189 checks; `FINAL-SHA256SUMS.txt` — 35 entries, manifest SHA-256 `6c4dd886380296287a4371b13c05053492a385305bb237c9a831ccfebe77130b` |

## Compute-pool conformance checklist

| Checklist item | Required retained evidence |
|---|---|
| Authority/scope | Authority list above; exact clean-only policy; no fault/convergence/promotion/scale submission. |
| Clean source | Standalone clean `main=origin/main=ls-remote`; corrected execution digest. |
| Native build | Canonical activation, exact-source rebuild, CTest 10/10, manifest/bundle hashes. |
| Exact G2 | One fresh full-layout clean job with queued/running/terminal `2/batch/debug` and passed gate. |
| Durable scheduler transaction | Held payload, atomic payload/job record, afterany collector, durable collector record, then release. |
| Exact clean model | One changed payload at exactly `Nodes=2`, `Partition=batch`, `QOS=debug`; queued/running/terminal records. |
| Immutable seed | Exact step/tokens/bytes/SHA, job-scoped `/tmp`, two offline node receipts, zero network fetches. |
| Persistent roles/membership | Two leased READY managers; 16 persistent real trainers; stable incarnations; no launched-rank authority. |
| Atomic checkpoint/apply | At least ten immutable commits/checkpoints and twenty rank-complete node authorities, reload verified. |
| Performance/tails | Two warm-up plus ten measured K40 windows per trainer; cadence, idle, zero result wait, per-phase, max/p99, and hard-tail gates. |
| Bounded data plane | Native CXI, finite high waters and release; forbidden Python/Lustre/SQLite/MPI/barrier/broker paths zero. |
| Independent terminal evidence | Collector survives monitoring-worker interruption; terminal verdict literal and repeated reconciliation idempotent. |
| Publication | Final SHA manifest plus committed/pushed report and `passed=true` companion verdict. |

Exact checklist-to-artifact bindings are:

```text
Authority/scope=SRC+PLAN+AUDIT; Clean source=SRC;
Native build=SRC+BUILD; Exact G2=G2;
Durable scheduler transaction=PLAN+COL;
Exact clean model=PLAN+SCHED; Immutable seed=SEED;
Persistent roles/membership=SEM+TX; Atomic checkpoint/apply=SEM+TX;
Performance/tails=SEM+AUDIT; Bounded data plane=BUILD+G2+SEM;
Independent terminal evidence=COL+AUDIT; Publication=AUDIT.
```

## R01–R16 conformance map

| ID | Exact artifact obligation |
|---|---|
| R01 | Corrected clean source, payload, held scheduler identity, fence, and current immutable state bind before release/model load. |
| R02 | Native peer lifecycle and stable worker/incarnation telemetry; collector remains terminal-only and owns no training state. |
| R03 | Leased READY membership and exact two-node capacity evidence; no launched-rank authority. |
| R04 | Payload/contribution/collector identities are idempotent; exact duplicate reconciliation returns original jobs/receipts. |
| R05 | Exact-token K40 math and deterministic native reference/G2 roots. |
| R06 | `Q_min=2`, `T_min=3,934,080`, finite K/freeze/result/apply/deadline evidence. |
| R07 | Ten digest-linked immutable commits/checkpoints and collector-retained validator inputs/verdict. |
| R08 | Bounded native chunks, credits, replay, backpressure/release; no broker. |
| R09 | Trainer-only live mutable state and model-free manager/collector. |
| R10 | Zero Lustre/Python dense hot-path and shared-database control bytes. |
| R11 | Fresh-process checkpoint/result reload and current-incarnation continuation. |
| R12 | Exact seed/model/outer/token/receipt recovery chain and independent build/G2/input bindings. |
| R13 | Slurm adapter and standard terminal accounting; native runtime remains scheduler-neutral. |
| R14 | Separate phase/deadline/high-water telemetry plus literal terminal scheduler/log/validator evidence. |
| R15 | Deterministic exact-token numerical/G2 evidence and finite clean semantic result. |
| R16 | Only exact two-node clean is run; every downstream fault/convergence/promotion/scale gate remains blocked until pass. |

## NDP01–NDP17 conformance map

| ID | Exact artifact obligation |
|---|---|
| NDP01 | Native peer control owns live state; Python adapts scheduler/checkpoint; no compute database. |
| NDP02 | Point-to-point path with zero MPI/all-rank/barrier evidence. |
| NDP03 | Fresh exact-source persistent C++17 `FI_EP_RDM`/`cxi` G2 and model services. |
| NDP04 | Sealed coherent immutable snapshot handoff; no live mutable background read or extra dense copy. |
| NDP05 | CTest and full-layout G2 exact deterministic f64/token-weighted roots. |
| NDP06 | Fixed source/policy/schema/ABI/wire/fence/incarnation/contribution/result/collector identities. |
| NDP07 | Current-fence leased endpoint exchange and exact `cxi` route evidence. |
| NDP08 | Pre-admitted finite snapshot/result/fabric/resident/mailbox bounds and high waters. |
| NDP09 | Credits distinct from completion; zero foreground backpressure wait after OWNED. |
| NDP10 | CRC/SHA/finite/once-only/idempotent receipt and reload checks. |
| NDP11 | Finite replay/reassignment bounds; clean job has no injection/replay. |
| NDP12 | Owner-direct result redistribution into one service-owned node aggregate; no broker/files. |
| NDP13 | Absolute stage deadlines; background expiry never becomes foreground result wait. |
| NDP14 | Exact installed C ABI/shared libraries/service/gate hashes and metadata-only local control. |
| NDP15 | Immutable background checkpoint publication; bounded later all-eight atomic apply and node authority. |
| NDP16 | Provider/identity/byte/bound/release/per-phase/terminal reason evidence, including separate Partition/QOS. |
| NDP17 | Current exact-source rebuild/CTest and fresh full-layout two-node G2 before the model job. |

## V21S01–V21S17 conformance map

| ID | Exact artifact obligation |
|---|---|
| V21S01 | Pin v2.1 policy/schema/native ABI/wire/source identities; reject v1/v2.0 relabeling. |
| V21S02 | Distinct commit/anchor/result/speculative clocks, maximum two; lag-three drop/defer and zero foreground catch-up wait. |
| V21S03 | Positive exact tokens are the only quorum, clock, numerator weight, and denominator. |
| V21S04 | Exact K40 and stateless `eta_outer=1.0`; sixteen trainers with required warm-up/measured windows. |
| V21S05 | Full stable-worker/incarnation/window/base/policy/layout/code/token/payload identity and exact Q/T/deadlines. |
| V21S06 | Trainer-exclusive mutable state, coherent bounded immutable capture, immediate post-OWNED resume. |
| V21S07 | Verified once-only x/z/interval correction at a safe boundary within released 60-second all-eight clock. |
| V21S08 | Current-fence reload-verified capacity-one latest mailbox with bounded replacement/defer. |
| V21S09 | One owned/one mutable cohort and finite resident/slot/credit/replay/receipt/mailbox/deadline bounds. |
| V21S10 | Leased READY current incarnations only; no one-node authority or launched-rank wait. |
| V21S11 | Eight prepared/boundary/applied receipts reduce to one node authority; partial apply cannot advertise READY. |
| V21S12 | Persistent compiled native CXI point-to-point path; all forbidden dense/collective paths zero. |
| V21S13 | Causal phase, raw cadence, foreground idle/result wait, lag, max/p99, and hard-tail semantic evidence. |
| V21S14 | Fenced immutable model/outer/token/lag/apply receipt chain and exact offline final seed. |
| V21S15 | Fresh G2 and one clean model job only at exact two-node `batch/debug`; later gates out of scope. |
| V21S16 | No promotion/rung authorization is created by this task. |
| V21S17 | Scale finite-close code/identity remains hashed but unexercised; no launched-rank or Q-min early close. |

## ISP01–ISP07 conformance map

| ID | Exact artifact obligation |
|---|---|
| ISP01 | Coherent trainer-owned immutable candidate and negative live-state background access evidence. |
| ISP02 | Snapshot/admission through OWNED at most one second and next K proceeds during every background phase. |
| ISP03 | Publication/hash/aggregation/checkpoint consume immutable inputs; reload verified outside foreground pause. |
| ISP04 | Snapshot/mailbox/credit/replay/receipt/resident capacity edges are bounded skip/replace/drop/defer paths. |
| ISP05 | Complete prepared result applies once at a safe boundary under 60 seconds; no partial visibility/READY. |
| ISP06 | Distinct causal freeze/admission/publish/aggregate/checkpoint/result-wait/apply/idle intervals; every-event max/p99. |
| ISP07 | Raw hard-tail validation rejects hidden approximately 200-second stalls; checkpoints/medians alone are insufficient. |

The exact requirement-to-artifact bindings for every normative requirement
are:

```text
R01=SRC+PLAN+TX+COL       R02=BUILD+SEM+COL
R03=SEM+TX+SCHED          R04=PLAN+COL+AUDIT
R05=G2+SEM+TX             R06=PLAN+SEM
R07=TX+COL+AUDIT          R08=G2+SEM+AUDIT
R09=SEM+COL               R10=G2+SEM+AUDIT
R11=TX+SEM+AUDIT          R12=SRC+BUILD+G2+SEED+TX
R13=SCHED+COL             R14=SEM+COL+AUDIT
R15=G2+SEM                R16=PLAN+SCHED+AUDIT

NDP01=BUILD+SEM           NDP02=G2+SEM
NDP03=BUILD+G2+SEM        NDP04=SEM+TX
NDP05=BUILD+G2+SEM        NDP06=SRC+PLAN+SEM
NDP07=G2+SEM              NDP08=SEM+TX
NDP09=SEM+TX              NDP10=SEM+TX
NDP11=SEM+TX              NDP12=G2+SEM
NDP13=SEM                 NDP14=BUILD+SEM
NDP15=TX+SEM              NDP16=G2+SCHED+SEM+COL
NDP17=SRC+BUILD+G2

V21S01=SRC+BUILD+PLAN     V21S02=SEM+TX
V21S03=SEM+TX             V21S04=SEM
V21S05=SRC+PLAN+SEM       V21S06=SEM
V21S07=SEM+TX             V21S08=SEM+TX
V21S09=SEM+TX             V21S10=SEM
V21S11=TX+SEM             V21S12=BUILD+G2+SEM
V21S13=SEM+AUDIT          V21S14=SEED+TX+SEM
V21S15=G2+SCHED+COL       V21S16=AUDIT
V21S17=SRC+SEM+AUDIT

ISP01=SEM+TX              ISP02=SEM
ISP03=SEM+TX              ISP04=SEM+TX
ISP05=SEM+TX              ISP06=SEM+AUDIT
ISP07=SEM+AUDIT
```

## Exact command provenance

The activation, official pytest, and native build commands appear above.
The full-layout G2 invocation and all scheduler arguments are retained in
`g2/5100131/submission.json` (SHA-256
`62ad250b0b6deefd0ea1e503756a7a1d4e39c2d0fa41d0143cd0331d9b6df8cb`).
The model/controller command was:

```bash
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --repo "$PWD" \
  --native-build-manifest \
    "$PWD/build/native-resilient-dataplane/native-artifacts.json" \
  --full-layout-gate "$PWD/../g2/5100131/full-layout-gate.json" \
  --run-root "$PWD/../clean" \
  --state "$PWD/../controller-state.json" \
  --output "$PWD/../clean-qualification-plan.json" --submit
```

Its output transcript has SHA-256
`c75f642b48e4bd50b9571a091ad697062e47298de793ff0065bc9d99c07ed7c9`.
The three exact collector registration attempts are retained verbatim:

```text
collector-registration-retry.log
  e14beb9ad94bbad8aad73c1823b2c2216bb188cb20cba5186683fd7e1a6ad776
collector-registration-account-retry.log
  581c85e0214db5045d276cf55ac6cd370a3d5658cda742e7d22cb3ec9a0f5f89
collector-registration-equivalent.log
  335def7596b0477b2540c209d60aba2a0f166815143f007a09ccad3ebd74f813
```

The exact reconcile/release and idempotent-controller transcript is
`4ace92634619006df347cb6d76351d0ce1c3363900cc3ff2b47be19f3996273a`.
The fresh-process semantic revalidation transcript has SHA-256
`aad498e376b2e744e4d7a051dbbe15c3f8d0b0e0f1345756c46e9977649693b2`.
The final audit program itself has SHA-256
`6fbb6bdabc242880715607a6d8103a27f0c39ac2d79def4d9716c98df884bd88`.

## Final gate result

**PASSED.** Machine verdict
`docs/validation/requalify-v21-durable-collector-2n-clean-20260727.json`
uses schema `emender-async-v21-two-node-qualification-verdict-v1` and contains
literal `passed=true`. It binds all identities, both exact two-node scheduler
lifecycles, independent durable collection, immutable seed receipts, semantic
performance, ten transactions, all 57 R/NDP/V21S/ISP requirements, and zero
fault/convergence/promotion/scale submissions. Its SHA-256 is
`652ef1d2467c48e60d8ec2cdc2b6f69a33c69b1904ee6a9e3d8c7eab3c860061`.

The independent final audit passed 370,189 checks, and all 35 entries in the
final SHA-256 manifest reverified. No acceptance evidence depends on a live
WG/Codex process. This clean pass authorizes the downstream fault task only
after `fix-frontier-durable` corrects the disclosed collector account/QoS
source identity; it does not itself authorize convergence, promotion, or
scale.
