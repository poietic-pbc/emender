# Codify async decoupled DiLoCo v2 validation

Date: 2026-07-24
Task: `codify-async-decoupled-diloco-v2`
Scope: architecture and traceability only; no Slurm job or 4+ node run was
submitted or authorized.

The worktree began clean at
`30efa9fc76bbd433be91d41cb06b640e369d91e4`
(`feat: merge-final-e97-seed-sbcast (agent-1470)`), retaining the requested
final-seed/sbcast lineage.

## Result

ADR-002 defines one exact initial experimental policy:
`async-decoupled-v2.0-exp`, `K=40`, global
`tau_hard/tau_target=6/2`, speculative-local
`sigma_hard/sigma_target=8/2`, cumulative adjacent-window contribution
coalescing, exact `tokens*(7-commit_lag)` aggregation weights, and
`eta=0.5`. These are convergence-gated hypotheses, not claimed optima.

The policy proves the old `tau=0`/full-overlap contract unsatisfiable, retains
job 5062348's failure without using its 5.626083x serial cadence to size an
unstable FIFO, and separates base-at-seal, commit, applied-anchor,
result-at-apply, and speculative local-window lag. Bounded local `OWNED`
transfers descriptor
responsibility to the persistent service; trainers do not wait for fabric send
or receipt completion.

The authoritative policy, compute-pool promotion, native extension boundary,
V2A01–V2A18 traceability, legacy-note status, and this validation record are
the only artifacts changed.

## Compute-pool v1 conformance checklist

The review used the required conformance checklist in
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md).
ADR-002 replaces only the explicitly versioned fresh-only generation semantic;
it preserves the following requirements:

| ID | V2 conformance review |
|---|---|
| R01 | One exclusive expiring allocation lease and strictly newer fence remain prerequisites before model load or mutation. |
| R02 | Stable worker plus new incarnation on return and the DISCOVER/SYNC/READY/DRAIN lifecycle remain unchanged. |
| R03 | Membership is the leased READY snapshot, never launched ranks; `Q_min=2` is a bounded named-contributor floor, not a collective. |
| R04 | The fenced identity is extended with local range, base/digests/tokens/lags; identical replay is idempotent and conflicts reject. |
| R05 | Exact tokens and deterministic sharded binary64 math remain; staleness aggregation weight is carried separately. |
| R06 | `Q_min=2`, `T_min=3,934,080`, a 420 s group deadline, and zero attempt retry are explicit and bounded. |
| R07 | Model/outer/manifest publication and authoritative latest remain one atomic current-fence commit. |
| R08 | Deterministic sharded owners, bounded chunks/credits/replay/release, and no full-model broker remain mandatory. |
| R09 | Managers/services remain model-free; trainers own model and disposable inner state. |
| R10 | Dense descriptors, coalescing/correction buffers, results, membership, and redistribution avoid Lustre and Python dense transport. |
| R11 | Catch-up, late handling, disappear/rejoin, new incarnation, stale drop, and bounded pause/drain rules are explicit. |
| R12 | Required global outer state `{mode,eta,step,accepted_tokens}` is atomically restored; local inner work is disposable on restart. |
| R13 | Policy identities and protocol remain scheduler-neutral; Frontier binds exact CXI separately. |
| R14 | Local ownership, group, named global/local lag/backpressure, commit, checkpoint, and drain waits all have explicit bounds and telemetry. |
| R15 | Scalar/vector/native reference math, exact tokens, changing lag/participation, deterministic replay, and convergence gates are required. |
| R16 | Numerical, failure/restart, performance, and convergence qualification is exactly two-node; no 4+ node authorization is present. |

## Native data-plane conformance

The review used the
[`NATIVE_RESILIENT_DILOCO_DATAPLANE.md`](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
conformance checklist and retained every NDP requirement:

| ID | V2 conformance review |
|---|---|
| NDP01 | Python retains lease/membership/group/commit policy; C++ retains dense handoff/transport/reduction/result lifetime. |
| NDP02 | Only bounded point-to-point work is allowed; no MPI initialization, all-rank collective, barrier, or launched-rank wait is introduced. |
| NDP03 | One persistent model-free C++17 service and exact Frontier `cxi`/`FI_EP_RDM` remain required. |
| NDP04 | Producer-direct service buffers remain; local `OWNED` is descriptor transfer, not a new full-layout copy. |
| NDP05 | Fixed layout/conversion/order/rounding remain; v2 supplies the separately checked exact integer aggregation weight. |
| NDP06 | The wire identity is version-extended, never guessed or encoded into an unchanged v1 ABI. |
| NDP07 | Current-fence leased endpoint exchange and opaque native routes remain unchanged. |
| NDP08 | Two trainer cohorts, one visible/one staging result, registered slots, and the `64,001,671,648` byte conservative admission are bounded before load. |
| NDP09 | Receiver credits remain distinct from send completion and application receipt. |
| NDP10 | CRC32C/SHA-256, finite validation, once-only application, idempotence, and conflict rejection remain mandatory. |
| NDP11 | Sender/service ownership covers bounded replay and at most two owner reassignments; no Lustre spill is added. |
| NDP12 | Owner-direct redistribution still produces one shared node result rather than a central broker or eight result files. |
| NDP13 | Route, local admission, group, apply, publication, and drain deadlines fail locally/cleanly without allocation abort. |
| NDP14 | A new versioned stable ABI carries ranges/lags/weights/correction identity; metadata-only local control remains. |
| NDP15 | Python retains checkpoint choice/publication; native returns a fenced read-only result and drains collectively free. |
| NDP16 | True global/local lag, coalescing, ownership/send/receipt, byte/high-water/release, pause/drop, and terminal reasons are required telemetry. |
| NDP17 | V1 G0–G2 are prerequisites, followed by exact two-node v2 gates; no v2 G6 or 4+ rung is authorized. |

## V2 requirement traceability

The companion
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md)
contains exactly one row for each V2A01–V2A18 requirement. The rows trace mode
and contradiction; continuous cumulative windows; identity; the two lag
clocks/bounds; grouping; weighting; outer math/state; correction-ledger rebase;
mailbox; memory; pause/drop/catch-up; membership/rejoin; fencing/corruption;
atomic restart; native constraints; honest performance; separate correctness
latency; and two-node numerical/failure/convergence/reproducibility gates.

## Exact validation commands

The canonical Frontier environment was activated before every Python command.

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_async_diloco_core.py \
  tests/test_async_diloco_local_simulation.py \
  tests/test_validate_pipelined_e97_performance.py
```

Result: `20 passed in 31.72s`. These are retained v1/scaffolding regression
tests; they do not claim that V2A01–V2A18 is implemented.

The executable ADR code blocks were loaded directly from the Markdown and
checked for cumulative scalar/vector aggregation, hard-lag weighting,
half-step outer apply, accepted-worker correction, nonaccepted-worker anchor,
ScheduleFree parameter-point translation, and the byte formula:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -c 'import re; from pathlib import Path; import numpy as np; text=Path("docs/ASYNC_DECOUPLED_DILOCO_V2.md").read_text(); blocks=re.findall(r"```python\n(.*?)```",text,re.S); ns={}; exec("\n".join(blocks),ns); d0=ns["contribution"]([1.0,2.0],[3.0,6.0],10,0,2); d1=ns["contribution"]([0.0,1.0],[7.0,4.0],5,2,3); mean,tokens=ns["aggregate"](6,[{"base_version":6,"exact_tokens":10,"digest":"00"*32,"delta":d0},{"base_version":0,"exact_tokens":5,"digest":"01"*32,"delta":d1}]); np.testing.assert_allclose(mean,(70*d0+5*d1)/75); assert tokens==15; state,outer=ns["outer_apply"]([10.0,20.0],{"mode":"delta_sgd","eta":0.5,"step":4,"accepted_tokens":100},mean,tokens); assert outer["step"]==5 and outer["accepted_tokens"]==115; rebased,points=ns["safe_boundary_rebase"]([18.0],[10.0],[20.0],[[3.0]],([100.0],)); np.testing.assert_allclose(rebased,[25.0]); np.testing.assert_allclose(points[0],[107.0]); nonaccepted,_=ns["safe_boundary_rebase"]([18.0],[10.0],[20.0],[]); np.testing.assert_allclose(nonaccepted,[28.0]); assert 16*(5506770496//2)+14440737184+5506770496==64001671648; print("ADR-002 executable scalar/vector reference: PASS")'
```

Result: `ADR-002 executable scalar/vector reference: PASS`.

Local Markdown targets and exact requirement-row coverage were checked with:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -c 'import re; from pathlib import Path; files=[Path(p) for p in ("docs/ASYNC_DECOUPLED_DILOCO_V2.md","docs/RESILIENT_DILOCO_COMPUTE_POOL.md","docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md","docs/RESILIENT_DILOCO_GAP_MATRIX.md","docs/ASYNC_QUORUM_DILOCO.md","docs/validation/codify-async-decoupled-diloco-v2-20260724.md")]; missing=[]; [(missing.append((str(f),target)) if not (f.parent/target).resolve().exists() else None) for f in files for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)",re.sub(r"```.*?```","",f.read_text(),flags=re.S)) if not raw.startswith(("http://","https://","#")) for target in [raw.split("#",1)[0].strip("<>")] if target]; assert not missing,missing; matrix=Path("docs/RESILIENT_DILOCO_GAP_MATRIX.md").read_text(); assert all(len(re.findall(rf"^\| {prefix}{i:02d} \|",matrix,re.M))==1 for prefix,start,end in (("R",1,16),("NDP",1,17),("V2A",1,18)) for i in range(start,end+1)); print("local Markdown links: PASS (6 files)"); print("requirement rows: PASS (R01-R16, NDP01-NDP17, V2A01-V2A18)")'
```

Result: all six local-link sets exist; all 51 requirement rows occur exactly
once.

The retained source object and job measurements were checked without a Slurm
query or new submission:

```bash
git cat-file -e \
  20c9d1bec436b6aa6a2eba4e434d2202e9c45762:reports/validate-pipelined-native-2-final-seed.md
git show \
  20c9d1bec436b6aa6a2eba4e434d2202e9c45762:reports/validate-pipelined-native-2-final-seed.md |
  rg '63\.679326|358\.265159|294\.940987|5\.626083|0\.817668|did not overlap'
```

Result: the retained object exists and contains every cited measurement and the
fail-closed overlap error.

Final format and project-source consistency commands:

```bash
git diff --check
cmp -s AGENTS.md CLAUDE.md
```

Result: both commands exit zero. Neither project guide was touched.

## Artifacts

- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`
- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/ASYNC_QUORUM_DILOCO.md`
- `docs/validation/codify-async-decoupled-diloco-v2-20260724.md`

The committed SHA is recorded in the WG task log after the surgical commit; a
SHA cannot be embedded in the same commit whose bytes it identifies.
