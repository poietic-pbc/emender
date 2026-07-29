# Resilient DiLoCo failure catalog

**Status:** postmortem and rallying reference, 2026-07-29. This note is
descriptive evidence, not a design authority and not a qualification manifest.
The normative authorities remain
[Resilient DiLoCo Compute Pool](RESILIENT_DILOCO_COMPUTE_POOL.md) and its
[traceability and bounded-backlog matrix](RESILIENT_DILOCO_GAP_MATRIX.md),
together with the native data-plane, asynchronous-v2.1, and execution-source
authorities linked from those documents. If this catalog and an authority
disagree, the authority wins.

## Executive summary

The failure history does **not** support the claim that resilient DiLoCo is
mathematically unsound or that the E97 seed/model was corrupted. Most failed
attempts stopped in orchestration: source attestation, launcher argument
propagation, a local socket pathname, seed bootstrap, a collector, SQLite
diagnostics, scheduler policy, or retained-artifact ownership. Several
evaluation and WG failures then made those operational failures more expensive
or temporarily obscured their disposition.

There were nevertheless genuine protocol/runtime defects. The most important
were generation/overlap stalls, generation-close concurrent with recovery,
incarnation and all-eight apply/rejoin handling, and inadequate causal/tail
validation. Job 5105811 is permanent negative evidence for the
closed-generation race; local native stress and Lean proofs are valuable
follow-up evidence, but they do not turn that failed physical job into a pass.
The artifact-root fix likewise does not establish a physical fault/recovery
pass.

The current disposition is deliberately conservative:

- a code or local-test fix is distinguished from a physically requalified
  trainer/native/model path;
- “passed” means the named immutable artifact literally passed, not that a
  nearby attempt reached many generations;
- an observed failure is distinguished from a hypothesized risk discovered by
  review or formal analysis; and
- scale remains blocked until the required real physical gates pass.

### Status vocabulary

| Status | Meaning |
|---|---|
| **Fixed and requalified** | The fix was followed by applicable immutable passing physical evidence on its exact source. The original failed job remains failed. |
| **Fixed but awaiting physical validation** | Code/local/native/formal tests pass, but no qualifying physical job on that exact execution identity has established the behavior. |
| **Process/tooling issue** | The failure was outside model math/data and is addressed operationally or by tooling; it may still have invalidated a run. |
| **Still open** | A required implementation or physical qualification gate remains absent/non-passing. |

## Authority and requirement-family relationship

The compute-pool conformance checklist and all four independent namespaces
remain applicable to implementations and qualification. This catalog uses the
following narrower routing; it does **not** claim that every incident violated
every requirement:

| Failure area | Most directly implicated requirement families |
|---|---|
| Fencing, membership, close/commit, catch-up, recovery and numerical authority | R01–R08, R11–R12, R14–R15; NDP01, NDP05–NDP13, NDP15–NDP16; V21S01–V21S05, V21S07–V21S11, V21S14, V21S17 |
| Native ownership, bounded transport, launcher identity and two-node gating | R08–R10, R13–R16; NDP01–NDP17; V21S05–V21S06, V21S09, V21S12–V21S16 |
| Immutable snapshot overlap, atomic apply and honest timing | R09, R14–R16; NDP04, NDP08–NDP09, NDP13, NDP15–NDP17; V21S02, V21S06–V21S09, V21S11–V21S17; ISP01–ISP07 |
| Durable evidence, checkpoint/seed restart and scheduler identity | R01, R07, R10, R12, R14, R16; NDP10, NDP15–NDP17; V21S01, V21S05, V21S13–V21S16; ISP03, ISP05–ISP07 |

R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07 remain independently
normative as defined in the matrix. A protocol theorem does not discharge a
native byte-path requirement; a checkpoint pass does not discharge overlap;
and a synthetic G2 pass does not discharge real trainer/recovery behavior.

## 1. Actual protocol and runtime correctness

| Failure mode and evidence | Symptom | Actual layer / root cause | Model or data corruption? | Mitigation already implemented | Current validation status |
|---|---|---|---|---|---|
| **Overlap/generation gaps and foreground stalls** — clean job 5078907 | The job produced 12 immutable generations, but corrected postmortem validation measured foreground idle `0.575375...` and alternating approximately 199–212 second gaps across all 16 trainers. The job remained `FAILED`, not a clean pass. | Runtime topology and test design: snapshot publication/result/checkpoint work was not cleanly separated from immediate next-K foreground progress; the original validator also missed standalone control JSON before reaching the real performance failure. | **No observed corruption.** Checkpoints advanced, but their existence could not establish overlap or acceptable tail latency. | Introduced preallocated coherent snapshot slots, immediate mutable-lane resume, background-only publication/checkpoint work, causal phase timing, and ISP-oriented validator tests including the approximately 200-second adversarial tail. | **Fixed and requalified for the later clean exact-source path**: later ten-generation clean jobs, including 5109029/5109030, passed their bound clean semantics. ISP01–ISP07 must still be retained on any new execution identity and at scale. |
| **Snapshot integrity, ownership and capacity risks** — ISP01–ISP05 review findings | Earlier code/tests proved process separation or checkpoint correctness without directly proving coherent capture against optimizer mutation, no background live-state read, nonblocking full-capacity behavior, or atomic visible all-eight apply. | Cross-layer evidence gap, with plausible runtime races: producer ownership, immutable buffer lifetime, mailbox/credit exhaustion, and safe-boundary apply were under-tested. | **Hypothesized risk, not an observed corrupt model.** The gap matrix explicitly marked these partial/gap rather than inventing a corruption event. | Named ISP01–ISP05 regressions, sealed immutable extents, two-slot snapshot behavior, typed skip/defer paths, and all-eight recovery/READY guards were added. | **Fixed but awaiting physical validation** on any changed execution identity. Do not infer a new source pass from old clean artifacts. |
| **Closure lag and generation-boundary accounting** | Early closure/cadence logic could conflate commit, applied anchor, result-at-apply and speculative snapshot lag, or treat two early arrivals/launched ranks as scale closure. | Protocol/policy and validation design. V2.1 requires four distinct clocks, lag at most two, lag-three drop/defer, leased-READY closure, and a finite scale close derived from passing arrival distributions. | **No observed seed corruption.** A bad close could bias accepted work or violate liveness, so it is a correctness risk even without byte corruption. | Versioned v2.1 identities, exact-token-only accounting, distinct lag clocks, deterministic close machinery, and fail-closed scale-controller checks were implemented. | **Still open at physical scale.** V21S17 arrival/closure evidence and physical 8→32→128 rung verdicts are absent. |
| **Closed-generation recovery race** — job **5105811**, task `fix-v2-1-generation` | After an intended node-0 trainer recovery, uninjected node 1 submitted to an already closed generation and raised `RuntimeError: generation is not open`. The extra cohort restart consumed budget; the later planned manager fault ended in `restart_exhausted`. Model 5105811 `FAILED 1:0`; collector 5105812 retained `passed=false` verdict SHA-256 `34961429...`. | **Genuine protocol/runtime race.** Reconstructed native peer control lacked an idempotent, non-mutating catch-up response for a contribution older than authoritative committed state. | **No observed numerical mutation or one-node commit.** The run failed closed, and retained evidence showed no partial/rollback authority; this was still a real liveness/recovery defect. | `fix-v2-1-generation` returns an immutable catch-up/reload receipt for strictly older closed work, validates authoritative receipt/manifest/result/token identity, releases the local result, and avoids charging an unrelated cohort restart. The hardened native kernel, permanent corpus, schedule stress, and Lean model retain the ordering. | **Fixed but awaiting physical validation.** Local Python/native tests passed, and later clean 5109029 passed, but 5105811 remains failed and the subsequent physical fault campaign never completed. |
| **Recovery incarnation and all-eight apply authority** — fixed-source fault job 5108175 | Following intended recovery, replacement management reported `native peer recovery handshake disagrees with manifest`, then `conflicting recovery incarnation rejected`, and exhausted restart budget. Job 5108175 `FAILED 1:0`; its independent collector recorded `passed=false`. | Genuine recovery-state integration defects: stale apply receipts, equal-generation reconstruction, abandoned pre-READY incarnation supersession, and peer-control/manifest agreement. | **No observed corrupted committed state.** Failure was fenced and fail-closed; no partial READY was accepted. | Regression-first fixes recover current apply receipts, clear obsolete authority, allow safe pre-READY incarnation supersession, keep wrong-attempt/fence rejection, and require eight receipts before READY. | **Fixed but awaiting physical validation.** Local pool/launcher and native tests passed; no later full fault/rejoin/fresh-recovery physical pass exists. |
| **Ownership/rejoin/rank containment** | A failure-sensitive launched-rank worldview can abort all work, admit a stale returning process, or allow one node to become authority when `Q_min=2`. Owner loss can expose partial aggregation if replay/reassignment is not bounded. | Architectural/runtime risk, with portions observed in earlier failed harnesses and portions exercised as fault tests. | **Potential numerical corruption if implemented incorrectly**, but retained modern failures failed closed and did not show it. | Native leased READY membership, stable worker plus incarnation, immutable cohorts, strict fence/sequence checks, at most two owner reassignments, retained replay, and all-eight node apply were implemented. Native fault G2 artifacts exercised peer loss/rejoin/replay without partial commit. | **Still open** for the real trainer fault gate; mechanisms are requalified only in synthetic G2/local stress. No rank-count or one-node-authority pass is inferred beyond the exact artifacts. |
| **Integrity rejection and failed-publication invisibility** | Duplicate/conflicting, stale, corrupt, nonfinite, wrong-layout/fence or partially published work could enter an accumulator or become restart authority. | Protocol and transport correctness risk; test/fuzz/formal analysis expanded the catalog beyond incidents seen in model jobs. | **Hypothesized corruption risk; no retained accepted-corrupt model event.** | Fixed-size fenced identities, CRC32C/SHA-256 checks, idempotent identical receipts, typed non-mutating rejects, exact-once commit lineage, and immutable no-replace publication. | **Fixed but awaiting physical validation** of the complete fault campaign; local and synthetic evidence passes. |
| **Checkpoint retention and fresh-allocation state** | Runs could advance a checkpoint but fail terminal validation, lose required historical artifacts, consult mutable `latest`, or restore model without exact outer/token/apply authority. Fresh recovery was repeatedly blocked before execution. | Durable-state and qualification-layer correctness, not aggregation math. | **No demonstrated corrupt checkpoint in the named recent incidents.** Missing/incomplete authority must fail closed. | Immutable checkpoint/manifests and digest-linked receipts, exact base receipt, bounded retention, offline verification, and newer-fence recovery rules are implemented; `latest.json` is non-authoritative. | **Still open** for fresh fault recovery; checkpoint production itself was fixed and requalified in clean evidence. The required real fresh-allocation phase was not reached after 5105811/5108175/5109414. |
| **Lean findings and coverage boundary** — `bootstrap-lean4-resilient-protocol`, `prove-lean4-resilient-protocol-safety`, `conform-native-coordinator-to-lean4` | Formalization exposed the need for total typed race dispositions, explicit bounded-progress assumptions, immutable cohort/commit authority, and complete identities. It also made clear that unconditional progress was unprovable under total participant loss, permanent quorum loss, unbounded faults, or unfair scheduling. | Pure coordination model/proof scope. Lean does not execute libfabric, physical clocks, GPU/model math, snapshot copies, byte buffers or Slurm. | **Neither a corruption event nor proof of physical absence of corruption.** | One executable transition kernel, safety/progress theorems with explicit assumptions, mutation tests, canonical replay, and native/Lean differential corpus including 5105811 were added. | **Fixed and requalified** within formal scope only. Physical coverage limits remain open by design; Lean is scoped evidence, not a replacement runtime or qualification pass. |

## 2. Launcher, controller and artifact ownership

| Failure mode and evidence | Symptom | Actual layer / root cause | Model or data corruption? | Mitigation already implemented | Current validation status |
|---|---|---|---|---|---|
| **Controller interface incompleteness** | The documented controller invocation initially lacked a usable full clean launch; direct file execution could not import repository modules; later the fault command rejected `--prior-gate` or lacked real injection/source bindings. These stopped before `sbatch` or produced superseded source attempts. | CLI/interface and packaging defects between reviewed policy and the executable runner. | **No.** These were pre-submit or pre-role orchestration failures. | Added direct-execution path bootstrap, exact `--prior-gate`, identity-bound phase plans, serialized injections, immutable handoffs, and fail-closed controller tests. | **Fixed and requalified where later clean/fault-baseline artifacts passed.** The entire fault/recovery sequence remains open. |
| **Gate-kind propagation** — job **5104162** and later baseline attempts | Fault-baseline 5104162 failed before runtime roles because the launcher attestation treated a valid `G2-fault-rejoin-replay` artifact as clean `G2`. Subsequent attempts exposed additional missing propagation in the role and manager-internal attestation paths. | Launcher/controller/role argument propagation. `required_gate` defaulted back to clean at successive boundaries. | **No.** Pre-role/pre-READY attestation rejected the artifact; no model mutation was credited. | Explicit immutable gate-kind is now exported, validated and forwarded controller → launcher → CLI/renderer → role → manager attestation, with regressions at each boundary. | **Fixed and requalified through later exact-source clean and fault G2 plus a two-commit baseline.** Full injected fault/recovery remains open. |
| **AF_UNIX pathname length** — job **5102376** | After the 7.7 GB seed was staged, both native services reached CXI `fabric_ready`, then local RPC setup failed before READY. The generated socket path was exactly 108 bytes and had no room for the required terminating NUL. Model 5102376 `FAILED 1:0`; its collector also failed because semantic JSON was absent. | Local launcher/filesystem naming, not native fabric or protocol math. | **No.** Failure occurred before READY/contribution/model progress. | Compact phase-unique `/tmp` control roots, explicit AF_UNIX length checks, short-basetemp tests, and fail-closed missing-semantic collector verdicts. | **Fixed and requalified.** Later source clean job 5103099/collector 5103100 passed ten-generation semantics, and later clean campaigns also passed. |
| **Artifact-root ownership collision** — job **5109414**, task `fix-g2-artifact-root-ownership` | Immediate monitoring created `g2/5109414/scheduler-evidence/` before the batch. The batch correctly refused to overwrite the existing final root and exited `73:0` in eight seconds, before dataplane setup. | **Orchestration/artifact ownership**, not a native fault. Controller, batch and collector shared an implicitly named root. | **No.** No dataplane/model/fault phase ran. | Versioned exclusive namespaces: controller content-addressed records under `controller/`, batch-only no-replace publication under the numeric job root from pre-marked `.batch-storage/`, and collector-only records under `collectors/`. Historical ordering, duplicate monitor, symlink and mkdir/rename races are permanent regressions. | **Fixed but awaiting physical validation.** Canonical build/CTest and Lustre-local regressions passed with no Slurm job, but the post-fix fault G2/model sequence was blocked by execution-scope drift before submission. |
| **Collector source identity and collection ordering** | Early collectors could be constructed from a different source, could be released/registered in the wrong order, or could disappear with the worker. A missing semantic file could crash collection instead of producing an honest false verdict. | Durable evidence orchestration. The collector must be scheduler-owned `afterany`, independently executable and identity-bound. | **No model/data corruption.** It could lose or misattribute evidence and therefore invalidate qualification. | Register-before-release, exact execution-source binding, independent process-level collection, idempotent state, missing-semantic `passed=false`, and immutable terminal records. | **Fixed and requalified for clean collection.** Collectors 5109030 and other later clean/fault-baseline collectors completed independently; no collector may turn a failed model into a pass. |
| **Collector Account/QoS registration** — task `fix-frontier-durable` | A collector omitted Frontier account and requested debug QoS. Since debug allowed one submitted job, an `afterany` collector could not coexist with the held debug payload. | Scheduler-facing controller configuration. | **No.** Evidence durability/launch ordering issue. | Collector identity is explicitly `Account=bif148`, `Partition=batch`, `QOS=normal`; model jobs remain exactly two-node `batch/debug`. Fake scheduler tests reject missing account and a second debug submission. | **Fixed and requalified operationally.** Later scheduler-owned normal-QoS collectors coexisted with held debug models and completed. |
| **Source/log/artifact self-invalidation** | G2 scheduler logs made an otherwise exact snapshot appear dirty; a terminal wildcard checksum treated a live evidence directory as a file error. Small launcher fixes then changed source identity and invalidated an otherwise passing G2/clean artifact. | Execution-source and artifact-manifest hygiene. | **No.** Exact-source evidence became unusable, forcing a new gate. | Narrow ignore rules for generated logs, directory-safe deterministic manifests, content-addressed evidence, and strict execution-source certificates. | **Process/tooling issue**, mitigated; recurrent cost remains. Any protected execution digest change still correctly requires new physical evidence. |

## 3. Environment, scheduler and filesystem

| Failure mode and evidence | Symptom | Actual layer / root cause | Model or data corruption? | Mitigation already implemented | Current validation status |
|---|---|---|---|---|---|
| **Shared SQLite diagnostic/runtime contention** — clean job 5072235 | After eight immutable generations, concurrent supervisor construction of `SQLiteFencedControlStore` raised `sqlite3.OperationalError: database is locked` and terminated the allocation. | Shared-filesystem control/diagnostic implementation, contrary to the later no-database amendment; not DiLoCo aggregation. | **No observed corruption.** The job failed; generations already committed remained immutable. | SQLite was removed from the compute closure. The Slurm fence plus native in-memory peer control and immutable receipt/checkpoint chain are authoritative; rendered-closure tests make SQLite fatal. | **Fixed and requalified.** Later ten-generation clean jobs passed without shared database coordination. |
| **7.7 GB final-seed staging** — task `stage-final-e97-seed-sbcast`; exact size `7,719,680,116` bytes | Compute nodes could not reliably reach S3 authorities; each attempt then spent startup time staging and verifying the large exact seed before reaching the later failure surface. | Environment/network bootstrap. The canonical checkpoint is step 2300930, tokens 150793748480, SHA-256 `0239706e...6a72b2`. | **No corruption when checks pass.** Wrong/partial bytes fail before model load. | Submit-side locked content-addressed fetch and verification, atomic publish, pinned authority attestation, Slurm `sbcast` to `/tmp/emender-e97-seed-$SLURM_JOB_ID`, per-node offline size/SHA verification, and `network_fetches=0`. | **Fixed and repeatedly physically requalified.** Later clean/fault jobs retained two node-local offline receipts, but staging remains a real time cost for each fresh job. |
| **Python/module environment** | Frontier login-shell Python was too old; standalone source snapshots lacked a private `.envs`; direct controller imports failed; several activation attempts stopped before build or submission. | Development/runner environment, not training. | **No.** | Canonical `scripts/frontier/activate_emender_frontier.sh`, explicit approved `EMENDER_CONDA_ENV`, `"$EMENDER_PYTHON"` and `PYTHON_BIN="$EMENDER_PYTHON"`; direct-execution import bootstrap. | **Process/tooling issue, mitigated.** Every new standalone snapshot must still bind the canonical environment explicitly. |
| **Partition/QoS confusion and queue limits** | Default `squeue` output was sometimes treated as if it proved QoS; debug QoS admitted only one submitted job; collectors or repeated attempts could be rejected. Long Priority waits delayed two-node runs by hours. | Scheduler policy/evidence, not protocol liveness. | **No.** | Every runner must request and retain `Partition=batch` and `QOS=debug` separately for model/G2 jobs using `squeue -o '%i|%T|%P|%q'`, `scontrol`, and terminal `sacct`; collectors use normal QoS. One active payload and no unchanged-digest resubmission. | **Process/tooling issue, mitigated.** Queue delay remains external and cannot be “fixed” by weakening evidence. |
| **Filesystem publication semantics** | Frontier Lustre returned `EINVAL` for `renameat2(RENAME_NOREPLACE)`, so a theoretically appropriate no-replace primitive was unavailable. Live scheduler-evidence directories also raced terminal file enumeration. | Filesystem capability/atomic-publication mismatch. | **No.** | Fully written/fsynced immutable files publish by hard link; pre-marked batch directories publish by no-replace symlink; conflicts exit 73 and preserve the winner. Tests run with `TMPDIR` on Frontier Lustre. | **Fixed but awaiting physical validation.** The fallback passed locally on the actual Frontier filesystem; a post-fix job remains open. |
| **Checkpoint retention/storage stalls** | Hashing, checkpoint I/O or retention cleanup could block foreground work or leave collectors observing incomplete state; median cadence/checkpoint counts previously hid long stalls. | Runtime I/O and telemetry boundary. | **No observed accepted corruption**, but incomplete publication must not become authority. | Background immutable checkpoint input, no checkpoint I/O inside snapshot/apply pauses, atomic manifest/receipt publication, bounded retention, causal phase and tail validation. | **Still open** for adversarial fault/fresh recovery and scale; the clean path was requalified. |
| **External outage / service unavailability** | Model/evaluator calls timed out, executors exited, or external source access was unavailable; WG tasks and evaluation stages failed independently of code. Compute-node S3 unavailability is the concrete training-side example. | External service/network availability. | **No.** | Offline seed bootstrap, durable WG checkpoints/messages, retries with retained digests, and honest incomplete/failed states. | **Process/tooling issue.** Outages remain possible; they must not be recorded as protocol passes or trigger duplicate Slurm payloads. |

## 4. WG, evaluator and development workflow

| Failure mode | Symptom | Actual layer / root cause | Model or data corruption? | Mitigation already implemented | Current validation status |
|---|---|---|---|---|---|
| **Evaluator/model timeout and corrupt evaluator cache** | Multiple `.flip-*`/`.evaluate-*` tasks timed out reading prompts; one evaluator reported a corrupt YAML cache. Completed implementation work could remain pending, be retried, or look failed for reasons unrelated to its tests. | Evaluation infrastructure/model service. | **No.** | Durable evaluator verdicts, explicit task logs, remote SHA equality, and separation of implementation status from evaluator status. | **Process/tooling issue.** Historical evaluator failure is not product failure and must not be counted as physical evidence. |
| **Dispatcher/service and worktree ownership churn** | Agents were killed/redispatched; new agents repeatedly failed to spawn because a prior live attempt still owned the isolated worktree. Some resumes inherited stale textual checkpoints. | WG lifecycle/service reconciliation. | **No.** | Worktree ownership checks, task checkpoints, message gates, retry-in-place, explicit reconciliation of current jobs/digests on every resume, and fail-closed no-resubmission rules. | **Process/tooling issue, recurring.** This catalog task itself was redispatched three times before a draft persisted. |
| **Stale WG checkpoints** | Resumed agents were told to inspect already terminal 5105811/5108175 while a newer clean job was active; agents had to issue status corrections before polling. | Context checkpoint staleness, not scheduler state. | **No.** Mishandling could have caused duplicate jobs. | Always reconcile live `squeue`/`sacct`, controller state and immutable payload digest before action; log corrections; never act on a symbolic “current” pointer alone. | **Process/tooling issue, mitigated procedurally.** |
| **Branch/main integration races** | A tested dependency was pushed on a task branch but absent from `origin/main`; automatic merge-back or a concurrent main advance rejected a push. Later broad Lean/native integration changed protected execution-source surfaces beyond the narrow artifact fix. | Git/WG integration and release management. | **No.** It invalidated reuse eligibility and forced requalification. | Non-force integration, fetch/remote equality checks, frozen release-candidate commits, surgical staging, and machine-readable change-scope certificates. | **Still open** as an operational risk. The latest certificate correctly failed: 74 reuse-disallowed paths, 71 outside the allowlist, six protected runtime-surface failures. |
| **Retry and unchanged-digest hazards** | Repeated runner attempts could be tempted to resubmit the same payload after a deterministic failure, or restart from stale task context. Retry exhaustion sometimes reflected workflow churn as well as real failed jobs. | WG/run-control policy. | **No direct corruption**, but duplicate attempts waste allocations and confuse evidence. | One-active-job rule; never retry unchanged payload; stop on first failed phase; `wg wait` for live external state; immutable false-pass reports and retired digests. | **Process/tooling issue, mitigated.** |
| **Separate collectors and false completion** | A model could terminate while its collector remained pending; a worker could exit before collection; collection success (`COMPLETED 0:0`) could be confused with a model pass even when the retained verdict was `passed=false`. | Evidence workflow and human interpretation. | **No.** | Scheduler-owned `afterany` collectors, literal machine verdict, model and collector accounting recorded separately, and downstream release only on the verdict. | **Process/tooling issue**, fixed operationally but still requiring discipline. 5105812 and 5108176 completed collection but preserved failed model verdicts. |

## 5. Feedback-loop and test-design amplification

| Failure mode | Symptom | Actual layer / root cause | Model or data corruption? | Mitigation already implemented | Current validation status |
|---|---|---|---|---|---|
| **Inverted validation feedback loop** | A later-stage launcher/controller bug was reached only after rebuilding exact source, running a full synthetic clean G2, waiting in queue, staging/verifying 7.7 GB, and often completing a ten-generation real clean gate. Fixing that one-line boundary changed source identity, invalidating the clean evidence and restarting the expensive prefix. | Test ordering and execution-source policy interacted badly: cheap interface paths were not exhausted before expensive physical gates, while exact-source invalidation was correctly strict. | **No.** The loop amplified cost, not corruption. | Failing-first boundary tests, fake scheduler/controller integration, local real batch-guard flow, complete argument-propagation tests, allowlisted scope certificate, and short phase targets. | **Still open**, though improved. Only physical runs can validate physical behavior; the goal is to reserve them for paths whose cheap boundaries are already closed. |
| **Median/checkpoint proxies hid tail stalls** | Twelve checkpoints and a healthy median could coexist with alternating approximately 200-second foreground pauses. | Validator design treated aggregate summaries as overlap evidence. | **No observed corruption**, but the claimed performance/liveness conclusion was invalid. | ISP06–ISP07 require causal phases, raw every-event timestamps, maximum/p99, zero foreground result wait, and hard rejection of the adversarial tail trace. | **Fixed but awaiting physical validation** on each new execution identity; validator/local tests pass. |
| **Full clean gates before later fault-launch paths** | Gate-kind, role attestation and manager-internal forwarding bugs were found serially only when the fault artifact reached each deeper layer. Each correction required a new source, G2 and clean campaign. | Insufficient end-to-end launcher coverage coupled to strict source identity. | **No.** All failed before READY or contribution. | End-to-end required-gate regressions now cover controller through manager. The new direction permits clean evidence reuse only under a byte-level protected-surface certificate. | **Still open** for the injected fault path; the boundary fixes were requalified through later fault G2/baseline. |
| **Formal/runtime scope amplification** | A broad Lean/native merge was attractive after protocol findings, but it changed protected execution surfaces and made a narrow artifact-only clean-evidence reuse impossible. | Workstream coupling. Formal evidence and production repair were merged into the same release candidate despite different physical validation needs. | **No.** | Keep Lean as scoped formal/differential evidence; freeze a minimal physical release candidate; reject broad runtime changes in the narrow fault-qualification path. | **Still open.** The current scope certificate failed closed; a fresh minimal release candidate or full exact-source requalification is required. |

## Why one failure consumed hours

The wall-clock cost was rarely the failing line alone. A representative attempt
paid most or all of this prefix:

1. **Queue delay.** Exact two-node `batch/debug` jobs sat pending for Priority,
   sometimes for several hours, while being polled without resubmission.
2. **Seed staging.** Every fresh model job distributed and independently
   verified the `7,719,680,116`-byte final seed before model load.
3. **Exact-source build and G2.** A code change required canonical activation,
   native rebuild/CTest, source/bundle attestation, then a real short clean G2.
4. **Full clean qualification.** The historical workflow often required all
   16 trainers to complete two warm-up plus ten measured K40 windows and ten
   atomic commits before entering a later fault-only launcher branch.
5. **Separate collection.** The scheduler-owned collector ran only after the
   model and had its own queue/accounting lifecycle; its `COMPLETED` state was
   not the verdict.
6. **Source invalidation.** Fixing AF_UNIX naming, gate propagation or an
   attestation call changed the source/execution digest. Correct fail-closed
   policy then made the preceding pass ineligible for reuse.
7. **WG checkpoint/retry churn.** Redispatches, stale resume context and
   worktree-owner conflicts required repeated state reconciliation and could
   not safely be skipped.
8. **Branch/main races.** Concurrent merge-back advanced or diverged main;
   dependencies sometimes had to be integrated non-force and every immutable
   identity regenerated.

This cost explains the simplified direction below. It does not justify
relabeling a failed artifact, simulating a physical gate, or weakening
execution-source identity.

## Current open gates

- The historical ten-generation clean evidence from model **5109029** and
  collector **5109030** passed on source `b756c14f`; it is reusable only if a
  change-scope certificate proves the trainer, native coordination
  kernel/bundle/ABI, model math, policy/schema, checkpoint/apply, seed/data/
  tokenizer and rendered model-execution digests are unchanged.
- The attempted certificate against the broader current main failed
  (`scope_pass=false`, 74 reuse-disallowed paths, 71 outside the allowlist and
  six protected runtime surfaces). Therefore no clean reuse or new Slurm job
  was authorized from that candidate.
- Job 5109414 remains failed negative evidence. Its ownership fix has local
  canonical/Lustre validation but needs a real post-fix short clean G2 and
  fault G2.
- The fixed closed-generation/incarnation paths need a real serialized
  two-node fault/rejoin pass. Native-service loss, manager loss, owner replay
  and new-incarnation recovery must complete without consuming unrelated
  restart budget or creating one-node authority.
- A strictly newer allocation must reload the exact checkpoint/model/outer/
  token/receipt/apply chain and complete the short fresh-recovery window.
- Physical causal overlap/tail evidence and immutable collector verdicts must
  pass on the frozen candidate. Formal, local stress and synthetic G2 evidence
  are complementary, not substitutes.
- No 8-node, 32-node or 128-node systems probe is authorized until the physical
  prerequisites and reviewed direct-scale policy are literally satisfied.

## Simplified current direction

1. Land only the **narrow artifact-ownership correction** on a frozen
   release-candidate source. Do not pull in unrelated runtime/formal history.
2. Generate an **allowlisted change-scope certificate** against the passed
   ten-generation clean source. Enumerate every path and protected execution
   digest; unknown or runtime-affecting drift fails closed.
3. Reuse model 5109029/collector 5109030 clean evidence **only** when trainer,
   native, ABI, model, policy, checkpoint/apply, seed/data/tokenizer and
   rendered execution digests are unchanged. Otherwise perform a full new
   exact-source clean qualification.
4. Run real, short, serialized physical jobs: clean G2, fault G2, a two-commit
   baseline, the fault/rejoin sequence, and a fresh-allocation recovery. Each
   is exactly `Nodes=2`, `Partition=batch`, `QOS=debug`, with its own immutable
   scheduler evidence and scheduler-owned collector.
5. Stop at the first literal failure, never resubmit an unchanged digest, and
   keep the release-candidate source frozen through the campaign.
6. Do **not** use simulated Slurm as qualification evidence and do **not**
   undertake a broad Lean runtime rewrite in this path. Local fake-scheduler
   tests remain appropriate for interface regression only.
7. Retain Lean as scoped formal safety/progress and native-differential
   evidence. It must continue to state its fairness/quorum assumptions and
   physical coverage limits.
8. Advance only through the reviewed direct physical ladder
   **8 → 32 → 128**, with an immutable pass and required formal/native join at
   each applicable boundary. No later rung repairs or excuses a missing
   predecessor.

That direction preserves the useful result of the failure history: the
mathematical and protocol core receives serious race scrutiny, while queue,
launcher, evidence and workflow failures are diagnosed at their actual layer
instead of being mislabeled as model corruption—or, in the other direction,
being allowed to minimize genuine protocol defects.
