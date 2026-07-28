# Durable async-v2.1 collector and execution-source identity

Date: 2026-07-27

WG task: `fix-v21-durable-collector-source-identity`

Status: **PASSED — implementation and fake-scheduler validation complete; no
Slurm job was submitted.**

## Scope and authorities

This implementation read and applied:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1, including its complete
  conformance checklist and R01–R16;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`, including NDP01–NDP17,
  V21S01–V21S17, and the ISP01–ISP07 amendment;
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`, version 1;
- accepted ADR-002 in `docs/ASYNC_DECOUPLED_DILOCO_V2.md`;
- the immutable-snapshot ISP01–ISP07 amendment in the gap matrix; and
- the retained safe-boundary qualification and exact source/evidence
  identities in
  `docs/validation/qualify-v21-safe-boundary-2n-20260727.{md,json}`.

The new normative boundary is
`docs/ASYNC_V21_EXECUTION_SOURCE_IDENTITY.md`, schema
`emender-async-v21-execution-source-v1`. It supplements rather than replaces
any training, native, scheduler, or promotion authority.

Implementation commit:

```text
da3882eb319e0a69f907219e9636231c76bbc694
```

Committed execution-source identity:

```text
schema:   emender-async-v21-execution-source-v1
digest:   4a54f0294d32612222076bb0aa2a24773c6b87338dc75f1db8dc1500dcc8ffb7
included: 7968 tracked files
excluded: 9015 tracked evidence files
```

The exclusion list is exactly `docs/validation/`, `logs/`, and `reports/`.
All other tracked content remains in the digest. Native bundles, G2, data,
tokenizer, training arguments, and the seed remain independently bound.

## Durable transaction and failure semantics

`scripts/frontier/run_async_v21_qualification.py` now performs this
fail-closed transaction while holding the controller state lock:

1. submit the model payload using `sbatch --hold`;
2. atomically record the exact payload digest, deterministic scheduler
   identity, held job ID, and active state;
3. retain the canonical collector payload input;
4. submit exactly one deterministic scheduler-owned collector with
   `--dependency=afterany:<payload-job-id>`;
5. atomically record the collector job ID, script SHA-256, dependency,
   scheduler request, argv digest, and the facts
   `scheduler_owned=true` and `requires_wg_or_codex=false`;
6. only then run `scontrol release <payload-job-id>`.

A failed collector registration records `registration-failed`, leaves the
payload held, and issues no release. Reconciliation first searches `squeue`
and `sacct` for the deterministic full job name/comment identities, so a
worker death after either scheduler side effect does not create a second
payload or collector. Held, queued, running, released, terminal, and retired
payload states retain the original job/collector identities. Terminal and
retired identities return as idempotent existing work and cannot be submitted
again; an unchanged failed/retired payload remains rejected at plan build.

The applicable failure path is collector-registration failure and submit-side
worker death after release. The process-level fake scheduler killed the
monitoring worker while `scontrol release` was outstanding. The already
registered independent collector continued, captured terminal evidence, and
atomically retired the active state without WG or Codex. Running the same
collector argv again left the verdict SHA-256 and mtime unchanged. The fake
scheduler recorded exactly one payload and exactly one collector.

The collector queries the parent allocation with:

```text
sacct -n -X -j <job-id> \
  --format=JobIDRaw,JobName,State,ExitCode,DerivedExitCode,Partition,QOS,NNodes,NodeList,Submit,Eligible,Start,End,ElapsedRaw \
  -P
```

It retries the accounting propagation gap within a finite bound and retains:

- the literal `sacct` pipe-separated output;
- separate `Partition` and `QOS`;
- `ExitCode` and `DerivedExitCode`;
- retained stdout and stderr with sizes and SHA-256s;
- the canonical payload/validator inputs with sizes and SHA-256s;
- the semantic validator verdict when required; and
- exactly one canonical `terminal-verdict.json` with literal `passed` or
  `failed`, a boolean `passed`, and its manifest digest.

The model and collector requests bind `Partition=batch` and `QOS=debug`
separately. The controller retains the exact predecessor map
`4 -> 8 -> 16 -> 32 -> 64 -> 256`, leased-READY V21S17 closure validation,
two-node minimum progress `Q_min=2` and `T_min=3,934,080`, and the final E97
seed:

```text
step:            2300930
accepted tokens: 150793748480
bytes:           7719680116
SHA-256:         0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
```

## Execution-source drift boundary

The controller hashes sorted Git paths and tracked contents with a
domain-separated SHA-256. A historical native-build commit is acceptable only
when its recomputed execution-source digest equals current clean authoritative
`main`; native build and G2 manifests must still bind each other exactly.
Current `HEAD=origin/main` and a clean worktree remain submit-time
requirements.

The regression creates a temporary Git repository and proves:

- commits changing only `docs/validation/`, `reports/`, or `logs/` retain the
  same execution digest;
- changes to controller/launcher code, native protocol code, general runtime
  code, normative policy, schema, public ABI header, data identity,
  tokenizer identity, or seed config all change the digest; and
- the digest recomputed from the working tree equals the digest recomputed
  from its committed Git revision.

The validation-report path is itself explicitly evidence-only. Staging this
report was checked to leave the implementation digest above unchanged; the
post-commit check repeats that assertion. An operational file cannot gain
this exemption by naming itself evidence: the authority forbids execution,
import, policy parsing, configuration parsing, or input selection from the
three excluded prefixes.

## Validation commands and results

All Python commands first used the canonical Frontier activation and
`"$EMENDER_PYTHON"`:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The TDD red gate was retained before implementation:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py
```

Result before implementation: collection failed because
`EVIDENCE_ONLY_PATH_PREFIXES` and the durable collector transaction did not
exist.

Focused final controller/death-window suite:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py
```

Result: `25 passed`.

Broad controller, launcher, serial acceptance, native attestation, and native
gate validation:

```bash
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_v21_qualification_controller.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_exact_2n_acceptance.py \
  tests/test_native_artifact_attestation.py \
  tests/test_validate_native_dataplane_2n_gate.py
```

Result: `126 passed, 1 skipped`. The one skip is explicitly
`canonical native bundle has not been built`; the implementation changes no
native source or ABI, and all ten remaining native-gate tests plus all four
native-artifact attestation tests passed. Rebuilding native code was therefore
not applicable to this controller/evidence-only change.

Compilation and whitespace checks:

```bash
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/run_async_v21_qualification.py \
  scripts/frontier/async_v21_terminal_collector.py \
  tests/support/fake_async_v21_scheduler.py \
  tests/test_async_v21_qualification_controller.py
git diff --check
```

Result: both passed.

Committed working-tree/revision identity equality:

```bash
"$EMENDER_PYTHON" - <<'PY'
from scripts.frontier.run_async_v21_qualification import _source_digest
current = _source_digest()
head = _source_digest(revision="HEAD")
assert current["digest"] == head["digest"]
print(current)
PY
```

Result: passed with digest
`4a54f0294d32612222076bb0aa2a24773c6b87338dc75f1db8dc1500dcc8ffb7`.

No command in this task invoked a real `sbatch`, `scontrol`, `squeue`, or
`sacct`. Those commands ran only through
`tests/support/fake_async_v21_scheduler.py`.

## Compute-pool conformance checklist

| ID | Conformance mapping |
|---|---|
| R01 | Held payload and atomic scheduler-fenced payload/job identity precede collector registration and release; no database is introduced. |
| R02 | The training lifecycle remains native peer-owned; the collector records terminal scheduler state only. |
| R03 | Leased READY membership remains the runtime authority; model `Nodes` is capacity, and scale closure still rejects launched ranks. |
| R04 | Full payload digest plus deterministic scheduler comment is the idempotency identity; identical reconciliation returns the original jobs. |
| R05 | Exact-token math is unchanged and every executable math byte remains in the execution digest. |
| R06 | Existing finite v2.1 deadlines/floors remain pinned; collector accounting propagation also has a finite retry bound. |
| R07 | Parent accounting, logs, validator inputs, and one digest-linked machine verdict are atomically retained. |
| R08 | Native bounded transfer/replay code is unchanged and remains execution-digested. |
| R09 | Collector and controller are model-free; no live model or optimizer is read. |
| R10 | No dense data path changes; collector reads terminal filesystem evidence only after `afterany`. |
| R11 | Runtime catch-up/rejoin semantics are unchanged and hashed; submit-worker death is independently contained. |
| R12 | Native bundle/G2/data/tokenizer/final-seed identities remain separate fail-closed payload bindings. |
| R13 | The collector is a scheduler adapter using standard Slurm identity and accounting surfaces. |
| R14 | Scheduler state, queue fields, exit fields, logs, validator inputs, and literal terminal reason are structured immutable evidence. |
| R15 | Numerical code and data preparation remain in the digest; evidence-only changes cannot impersonate numerical drift. |
| R16 | The exact `4 -> 8 -> 16 -> 32 -> 64 -> 256` predecessor map and two-node gates remain unchanged. |

## Native data-plane conformance

| ID | Conformance mapping |
|---|---|
| NDP01 | No control ownership moves to the collector; native peer control remains live authority and no database is added. |
| NDP02 | No MPI or all-rank operation is introduced; the scheduler dependency is between two jobs, not training peers. |
| NDP03 | Exact `cxi` native bundle/G2 attestation remains required and separately digested. |
| NDP04 | Immutable memfd/XPMEM handoff code is unchanged and execution-digested. |
| NDP05 | Deterministic native arithmetic and source remain exact identity inputs. |
| NDP06 | Policy, payload, launcher, native build, job, collector, dependency, and terminal records carry fixed identities/digests. |
| NDP07 | Leased endpoint exchange is unchanged; the collector uses no endpoint or membership data. |
| NDP08 | Native capacity bounds are unchanged; collector artifacts are bounded to named terminal inputs. |
| NDP09 | Native credits remain background-only; the collector runs only after allocation termination. |
| NDP10 | SHA-256 evidence retention is additive; native CRC/SHA/idempotent receipt behavior is unchanged. |
| NDP11 | Replay/reassignment behavior is unchanged and cannot be excluded as evidence. |
| NDP12 | Owner-direct redistribution is unchanged and remains in executable identity. |
| NDP13 | Model deadlines remain absolute; collector accounting visibility uses a separate bounded retry. |
| NDP14 | ABI and local control sources/headers remain hashed; no ABI symbol or struct changes. |
| NDP15 | Checkpoint/apply behavior is unchanged; the terminal collector retains rather than creates commit authority. |
| NDP16 | Terminal accounting adds separate Partition/QOS, exit/derived-exit, logs/hashes, validator-input hashes, and literal verdict. |
| NDP17 | Exact-source native build/G2 remains required; native gate tests passed apart from the explicitly absent local built bundle. |

## Async-v2.1 conformance

| ID | Conformance mapping |
|---|---|
| V21S01 | Policy/schema/native ABI/wire/controller bytes remain included in execution identity; historical v2.0 stays rejected. |
| V21S02 | Four lag clocks and lag-three behavior are unchanged and digested. |
| V21S03 | Exact tokens remain the sole quantitative floor/clock/weight; `T_min=3,934,080` is retained. |
| V21S04 | K40 and stateless `eta_outer=1.0` remain pinned in payload exports. |
| V21S05 | Full fenced identity, `Q_min=2`, token floor, deadlines, and stable-worker rules remain unchanged. |
| V21S06 | Immutable snapshot ownership behavior is unchanged; collector reads only terminal artifacts. |
| V21S07 | Atomic safe-boundary apply behavior and 60-second bound remain payload/validator inputs. |
| V21S08 | Verified latest mailbox behavior is unchanged and execution-digested. |
| V21S09 | Resident/credit/replay/mailbox bounds remain operational identity, never evidence-only identity. |
| V21S10 | Leased READY membership and returning-incarnation rules remain unchanged; launched-rank closure remains rejected. |
| V21S11 | Atomic all-eight apply/recovery remains semantic validator input; collector cannot fabricate node READY. |
| V21S12 | Persistent compiled point-to-point native path remains separately bundle/G2-bound and execution-digested. |
| V21S13 | The semantic verdict and causal telemetry input are hashed into terminal evidence; missing required clean verdict fails. |
| V21S14 | Final seed step/tokens/bytes/SHA and current-fence immutable publication inputs remain exact. |
| V21S15 | Two-node model request remains exact `Nodes=2`, `Partition=batch`, `QOS=debug`, retained separately through terminal accounting. |
| V21S16 | Promotion still requires review and exact immediate predecessors `2,4,8,16,32,64` for rungs `4,8,16,32,64,256`. |
| V21S17 | Evidence-derived finite close remains over leased READY at group open and still rejects Q-min early close, launched ranks, and all-READY waits. |

## Immutable-snapshot amendment

| ID | Conformance mapping |
|---|---|
| ISP01 | Trainer-only live-state ownership and coherent immutable capture code are unchanged and execution-digested. |
| ISP02 | The 1-second through-OWNED bound and immediate next-K behavior remain semantic validator inputs. |
| ISP03 | Collector runs after terminal state and reads no live trainer object; background pipeline implementation remains hashed. |
| ISP04 | Capacity/credit/mailbox behavior is unchanged; evidence collection has fixed named files and one verdict. |
| ISP05 | Atomic bounded apply semantics remain validator inputs; failed terminal semantics produce literal failed/retired evidence. |
| ISP06 | Collector retains the semantic validator output and hashes it alongside scheduler/log inputs; queue and exit fields are not conflated. |
| ISP07 | Terminal collection cannot substitute checkpoint count or median-only evidence; the required semantic verdict remains mandatory for clean. |

This implementation task claims no new model, native, fault, convergence,
promotion, or scale qualification. It establishes the durable execution and
evidence boundary required before those downstream runs.
