# Async v2.1 completion-program quality pass

Date: 2026-07-25

WG task: `.quality-pass-async-v21-completion`

## Outcome

The complete 16-task batch from `reconcile-async-v21-authority` through
`package-async-v21-production` was inspected with `wg show` and edited in
place. The batch is now a pass-gated program with:

- repository reconciliation and an authoritative `origin/main` push before
  design or implementation;
- reviewed design, implementation, atomic node-apply repair, and integration
  before the first Slurm runner;
- exact two-node clean/performance, fault/restart, and
  convergence/reproducibility gates feeding an explicit promotion review;
- a strict `4→8→16→32→64→256` scale ladder, with each task depending on the
  immediately preceding passed rung;
- production packaging only after the 256-node rung;
- concrete runner commands, scheduler evidence fields, monitoring intervals,
  changed-payload-only retry rules, and pass-only `wg done` semantics; and
- a bounded leased-READY-snapshot scale closure that cannot race to commit
  merely because the two-node `Q_min=2` floor arrived.

No runtime code was implemented, no training behavior was changed, and no
Slurm job was submitted, cancelled, or otherwise mutated by this quality pass.

## Authorities reviewed

The review used the following authorities and retained seed evidence:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`
- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`
- `docs/validation/integrate-final-e97-s3-seed-20260722.md`
- `reports/merge-exact-2n-final-seed-debug-qos.md`

The existing authorities define R01–R16, NDP01–NDP17, and the historical
v2.0 V2A01–V2A18 requirements. The design task now has the explicit job of
codifying the simplified v2.1 namespace V21S01–V21S17 and marking v2.0
artifacts incompatible rather than relabelling them.

The exact cold-start seed is fixed throughout all implementation, runner,
promotion, and release gates:

- immutable checkpoint:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- immutable step manifest:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json`
- discovery pointer:
  `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- step: `2300930`
- accepted tokens: `150793748480`
- size: `7719680116`
- SHA-256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

Every task that may run Python, pytest, native builds, CTest, or Frontier
render/preflight now requires:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

and uses `"$EMENDER_PYTHON"` plus `PYTHON_BIN="$EMENDER_PYTHON"` where the
wrapper accepts an interpreter.

## Simplified v2.1 policy IDs

The downstream descriptions use one stable namespace so evaluators and later
rungs do not have to interpret phrases such as “the new requirements”:

| ID | Required contract |
|---|---|
| V21S01 | Explicit v2.1 policy/schema/digest boundary and fail-closed v2.0 incompatibility |
| V21S02 | Distinct commit, applied-anchor, result, and speculative clocks; maximum accepted global/anchor/speculative lag 2; lag 3 drops and catches up |
| V21S03 | Exact tokens are the sole quorum, accepted-token-clock, and deterministic numerical weight |
| V21S04 | K40 stateless exact-token weighted average with `eta_outer=1.0` |
| V21S05 | Fenced contribution identity, exact two-node floors, and at most one contribution per stable worker per transition |
| V21S06 | Persistent continuous training with one immutable owned descriptor and one mutable cumulative adjacent interval |
| V21S07 | K-boundary accepted-delta correction ledger, including ScheduleFree `x/z` translation |
| V21S08 | Capacity-one verified latest mailbox and fenced atomic publication |
| V21S09 | Finite resident/credit/replay/deadline bounds and no third dense cohort |
| V21S10 | Leased READY membership, failure/rejoin with new incarnation, and no invented one-node authority |
| V21S11 | Atomic all-eight-trainer node apply/recovery before the next READY version |
| V21S12 | Model-free compiled memfd/CXI point-to-point dense path; no failure-sensitive all-rank wait, Python/Lustre dense hot path, or central full-model broker |
| V21S13 | Honest stage/lag/cadence/idle/OWNED telemetry with checkpoint-correctness latency reported separately |
| V21S14 | Fenced immutable checkpoint, outer/token restore, exact final-seed bootstrap, and fresh-allocation recovery |
| V21S15 | Exact two-node numerical, clean/performance, fault/restart, deterministic replay, and three-seed convergence gates |
| V21S16 | Explicit pass-only promotion and ordered `4→8→16→32→64→256` ladder |
| V21S17 | Scale-only finite closure over the leased READY snapshot: include all complete admissible pre-close arrivals, never use launched ranks, never early-close merely at `Q_min=2`, and derive close/deadline/cadence from named passing two-node timing evidence |

V21S17 preserves non-blocking elastic membership: it is not an all-READY
barrier. The authorization must define the finite close formula,
minimum-progress/failure behavior, deadline, and cadence with an explicit
calculation from retained two-node arrival and stage distributions. If that
evidence is insufficient, the promotion review remains incomplete rather than
inventing a constant.

## Task edits

All task descriptions now contain a concrete `## Validation` checklist. Code
and runner tasks cite R01–R16, NDP01–NDP17, and V21S01–V21S17.

| Task | Timeout | Dependency gate | Material edit |
|---|---:|---|---|
| `reconcile-async-v21-authority` | 4h | quality pass | Fetch/reconcile/preserve evidence, push `origin/main`, prove remote equality, label prior failed jobs non-promotable |
| `codify-simple-async-v21` | 6h | reconciliation | Define V21S01–V21S17 in normative authorities; preserve R/NDP contracts; no code or Slurm |
| `implement-simple-async-v21` | 12h | design | TDD implementation plus fail-closed canonical serial controller and scale-predecessor/closure checks; no Slurm |
| `fix-atomic-node-apply-v21` | 8h | implementation | Regression-first repair for job 5068873 partial apply/restart ordering; no Slurm |
| `integrate-simple-async-v21` | 8h | atomic-apply fix | Merge/build/test/non-submit render/push gate; verify exact seed, scale lock, and V21S17 rejection paths |
| `qualify-simple-async-v21-2n-clean` | 36h | integration | Concrete current-source G2 clean plus v2.1 clean controller command; retain clean arrival/stage distributions |
| `qualify-simple-async-v21-2n-faults` | 48h | two-node clean | Concrete G2 fault plus serialized v2.1 failure controller command; retain missing/late/rejoin distributions |
| `qualify-simple-async-v21-2n-convergence` | 7d | two-node faults | Precommitted three-seed/two-arm plan, at least 100 commits per arm, deterministic replay, fixed BPB/shock thresholds |
| `authorize-simple-async-v21-scale` | 8h | direct fan-in from clean, faults, and convergence | Independent no-Slurm review; authorizes only four nodes; must derive and pin V21S17 closure |
| `scale-simple-async-v21-4n` | 36h | promotion review | Exact four-node controller command, preceding manifest, pass-only runner |
| `scale-simple-async-v21-8n` | 36h | four-node pass | Exact eight-node controller command, preceding manifest, pass-only runner |
| `scale-simple-async-v21-16n` | 48h | eight-node pass | Exact 16-node controller command, preceding manifest, pass-only runner |
| `scale-simple-async-v21-32n` | 48h | 16-node pass | Exact 32-node controller command, preceding manifest, pass-only runner |
| `scale-simple-async-v21-64n` | 72h | 32-node pass | Exact 64-node controller command, preceding manifest, pass-only runner |
| `scale-simple-async-v21-256n` | 5d | 64-node pass | Exact 256-node controller command, preceding manifest, pass-only runner |
| `package-async-v21-production` | 12h | 256-node pass | No-submit release gate mapping every pass and pinning seed, V21S17, monitoring, rollback, and origin/main |

Tags were added where applicable: `authority-gate`, `pre-slurm`,
`pass-gate`, `tdd`, `changed-payload-only`, `exact-seed`,
`promotion-review`, `no-slurm`, `serial-scale`, and `release-gate`.

## Runner state machine

Every two-node and scale runner explicitly attempts the concrete commands
available to it. The implementation task must provide the stable controller
entry point:

```bash
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean|faults|convergence|scale \
  --nodes N \
  ... \
  --submit
```

The clean runner additionally invokes the current-source G2 clean wrapper;
the fault runner invokes the G2 fault wrapper against the retained clean gate.
Each scale runner supplies the immutable scale authorization and exact
immediately preceding passed manifest.

All runners now require:

- one active job at a time;
- a payload digest over source, policy/schema, native bundle, seed, launcher,
  gate, and parameters;
- no resubmission of an unchanged failed payload;
- queued/running evidence with explicit node count, `Partition`, and `QOS`;
- PENDING observations no more than 30 minutes apart;
- RUNNING progress inspection every 120–300 seconds;
- terminal `sacct` evidence with `NNodes`, `Partition`, `QOS`, state, exit
  code, start/end, and elapsed fields;
- `wg wait` for scheduler/external delay and `wg incomplete` after a concrete
  non-passing attempt; and
- `wg done` only after every checklist item and the machine-readable gate
  verdict pass.

A failure report is useful evidence but is not task completion. This keeps the
failed runner active, waiting, or incomplete and leaves its dependent task
blocked.

## Reviewed graph

The graph was rendered after all edits with:

```bash
wg viz .quality-pass-async-v21-completion --all --no-tui
wg viz .quality-pass-async-v21-completion --all --mermaid --no-tui --columns 240
wg cycles
```

The exact Mermaid render is retained in
`docs/validation/quality-pass-async-v21-completion-20260725.mmd`. The current
quality-pass node itself gates `reconcile-async-v21-authority`; `wg show
reconcile-async-v21-authority` confirms that edge even though the focused
downstream Mermaid renderer begins at reconciliation.

The critical ordering is:

```text
.quality-pass
  → reconcile/push
  → codify V21S authority
  → implement
  → atomic-node-apply fix
  → integrate/build/test/push
  → 2n clean/performance
  → 2n fault/restart
  → 2n convergence/replay
  → explicit promotion review
  → 4 → 8 → 16 → 32 → 64 → 256
  → production package
```

The promotion review also has direct dependencies on all three two-node tasks,
so clean, fault/restart, and convergence are visible as an explicit fan-in.
`wg cycles` reported no cycles.

## Validation performed

Post-edit mechanical validation re-ran `wg show` for every task and checked:

- all 16 tasks have `## Validation`;
- every downstream code/runner task cites R01–R16, NDP01–NDP17, and
  V21S01–V21S17;
- every code/runner task carries the canonical activation/interpreter rule;
- every runner names the canonical controller, `--submit`, exact final seed,
  changed-payload-only rule, PENDING/RUNNING cadence, `wg incomplete`, and
  pass-only `wg done`;
- every scale task names V21S17, the READY snapshot, the `Q_min=2`
  early-close prohibition, launched-rank prohibition, and two-node-derived
  parameters;
- promotion directly depends on clean, faults, and convergence;
- the exact ladder edges are
  `promotion→4→8→16→32→64→256→package`;
- reconciliation precedes design, design precedes implementation,
  integration precedes the first Slurm task; and
- no graph cycle exists.

All checks passed.
