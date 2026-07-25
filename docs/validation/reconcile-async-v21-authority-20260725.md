# Async v2 terminal-evidence and main-authority reconciliation

Date: 2026-07-25

WG task: `reconcile-async-v21-authority`

Status: **RECONCILED AUTHORITY; RETAINED RUNS FAILED; NO QUALIFICATION OR
PROMOTION**

## Outcome and scope

The assigned WG worktree reconciles all four relevant histories without
changing training behavior:

- the initially fetched authoritative `origin/main` code-fix tip
  `92e398c2914194847e2aa460f2f7bd0482f2fad4`;
- the retained terminal-report branch
  `origin/wg/agent-1508/re-run-fixed` at
  `d40b3047ba05e3d76b29687219bee1662b4e2b87`;
- the WG auto-squashed local evidence at
  `efd7b8a91d657078d235fdd7e582fff3a3277d5c`; and
- the local completion-program quality artifacts at
  `0cf241e118dfb2c37b2318e94c062440dc39329e`.

The intentional reconciliation merge is
`9ebf06dbf1874d9c6b0daca7ae8211a2158abc46`, with first parent
`0cf241e1...` and second parent `d40b3047...`. The merge is tree-neutral:
`efd7b8a9...` and `d40b3047...` both have tree
`62a9f1d3dad461b3f98bac879d19e6dd0c0786f6`, while the merge retains first
parent tree `7dc08cfe6f9021a81e23faf5a1bbfd3794f7caa7`, which adds only the two
quality-pass documents. Thus the raw retained commit is an ancestor, the
auto-squashed evidence has no divergent payload to choose between, and the
local-only quality evidence is preserved.

No `reset`, `revert`, `checkout`, `restore`, `clean`, `stash`, deletion, or
force push was used. The worktree was clean, including untracked-file
enumeration, before reconciliation. No unrelated user file or data was
removed. `AGENTS.md` and `CLAUDE.md` were not edited; both have Git blob
`e7d929e9dbf67aa3fc788f22d3cce707b641bdb7` and remain byte-identical.

No Python, pytest, native build, render, or preflight command was run. No
`sbatch`, `srun`, `salloc`, `scancel`, or other Slurm-mutating command was run.

## Authorities reviewed

Before the reconciliation merge, the following required authorities and seed
report were read in full:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`
- `docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md`
- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`
- `docs/validation/integrate-final-e97-s3-seed-20260722.md`

The retained terminal report is
`docs/validation/re-run-fixed-async-v2-exact-2n-20260724.md`, Git blob
`f597bc2ad1e2d7daff38ceacd4d549c3d0cdbf05`. The retained final-seed
integration report is Git blob
`bc02facf25c7864c606a56313dc5a287c72fe406`.

R01–R16 and NDP01–NDP17 remain the current compute-pool and native-data-plane
requirements. V2A01–V2A18 describe historical
`async-decoupled-v2.0-exp` behavior and evidence. They are retained honestly
but grant no v2.1 authority. The downstream `codify-simple-async-v21` design
task must establish the v2.1 authority before any v2.1 implementation begins;
historical v2.0 results may not be relabeled as v2.1 evidence.

## Retained terminal evidence is failed evidence

### Job 5066495

The authoritative report retains:

```text
5066495|resilient-e97-true-2n|FAILED|1:0|00:27:37|batch|debug
```

The job used exactly two nodes, `Partition=batch`, and `QOS=debug`. Generation
zero committed with exact accepted tokens `5245440`, separate lag-zero
aggregation/global weight `36718080`, and result root
`7636bfeba03dc38b23a829e850bad29a6e63e31b561002410fae757ab11406e1`.
The first independent post-commit failure was:

```text
ValueError: ScheduleFree z point is missing or malformed
```

That failure led to the narrow lazy-state initialization correction in
`92e398c2`. The job itself remains **FAILED** and grants no clean
qualification, promotion, later phase, or scale authorization.

### Job 5068873

The retained `d40b3047` report adds the terminal changed-payload result:

```text
5068873|resilient-e97-true-2n|FAILED|1:0|00:27:50|2|batch|debug|2026-07-24T19:44:12|2026-07-24T20:12:02|frontier[02299,02304]
```

The source was exactly `92e398c2...`; the prerequisite exact-source G2 job
`5068822` passed first. Job 5068873 reached the generation-zero atomic handoff
with exact accepted tokens `5245440`, separate lag-zero aggregation/global
weight `36718080`, and result root
`04dfbf27ebfe52371e56142e55ddbbae93588440c2906744720d4cf6d6bbab3d`.

The ScheduleFree lazy-state failure did not recur. The first independent
post-commit failure was the node-1 apply-marker boundary: trainers 4–7 reported
missing markers
`native-result-applied-00000000-{03,04,05,06}.json` before the absolute
deadline. Later generation-one native-submit timeouts and stale/invalid
restart metadata were cascades after that failure.

Job 5068873 therefore remains **FAILED**. The committed handoff is useful
diagnostic evidence, but an incomplete all-trainer apply boundary is not an
atomic node apply, clean qualification, promotion, later-phase authorization,
or scale authorization. The serialized controller stayed fail-closed at
`active.phase=clean-overlap`, `active.job_id=5068873`, `next_phase=0`, with an
empty history.

## R01–R16 reconciliation mapping

This task does not claim to execute the runtime requirements. It preserves
their authority and classifies the retained evidence against them.

| ID | Reconciliation check and result |
|---|---|
| R01 | The exclusive lease/newer-fence authority remains unchanged. The failed jobs' fenced source identities are retained; this Git task acquired no training lease and makes no Frontier durable-store qualification claim. |
| R02 | Stable worker/incarnation lifecycle evidence remains in the terminal report. Restart cascades after job 5068873 do not convert expired/invalid metadata into a pass. |
| R03 | Both jobs retain exactly two-node READY/contribution evidence; no launched-rank invariant or scale inference is introduced. |
| R04 | Fenced contribution and generation identities, including stale/invalid restart outcomes, remain labeled honestly. No failed identity is relabeled as accepted v2.1 evidence. |
| R05 | Exact accepted tokens remain distinct from aggregation weight in both job records; the result roots and numerical denominators are unchanged. |
| R06 | The exact two-node floor and bounded deadlines remain visible. Missing apply markers and subsequent submission deadlines fail closed. |
| R07 | Atomic commit/checkpoint/latest authority is preserved. A generation-zero handoff or native result alone does not grant a clean pass; only immutable fenced publication is authoritative. |
| R08 | Native bounded owner/replay/checksum evidence is retained without claiming that a failed end-to-end apply qualified the data plane. |
| R09 | The model-free manager/trainer ownership boundary is unchanged; reconciliation changes documentation/history only. |
| R10 | No dense hot-path behavior or Lustre policy changed. Git reconciliation and immutable evidence retention are not a dense transport path. |
| R11 | Rejoin requires a new valid incarnation and authoritative catch-up. The stale/invalid restart cascades in job 5068873 are failure evidence, not recovery success. |
| R12 | Global outer/token state and fresh-allocation restore requirements remain unchanged. Neither failed run grants a restart qualification. |
| R13 | Backend-neutral protocol boundaries remain unchanged; no adapter claim is added. |
| R14 | Exact terminal states, exit codes, node/partition/QoS fields, first independent failures, absolute-deadline boundary, and downstream cascades are retained as stage evidence. |
| R15 | Exact accepted-token accounting and separate lag-zero numerical weights/result roots are retained. No convergence claim is inferred. |
| R16 | The two-node-before-scale gate remains closed: jobs 5066495 and 5068873 failed, grant no qualification or promotion, and authorize no 4+ node work. |

The especially authority-sensitive rows are R01 (lease/fence, no false durable
store claim), R07 (handoff is not by itself an atomic authoritative commit),
R10 (no hot-path or training-behavior change), R14 (failed stage/deadline
evidence retained), and R16 (no scale authorization).

## NDP01–NDP17 reconciliation mapping

| ID | Reconciliation check and result |
|---|---|
| NDP01 | The Python-control/C++-dense authority boundary is unchanged; only Git history and validation documentation are reconciled. |
| NDP02 | No elastic collective or all-rank behavior was added or exercised. |
| NDP03 | The retained evidence continues to name exact provider `cxi` and the source-pinned native bundle; no new provider qualification is claimed. |
| NDP04 | Direct memfd/XPMEM handoff requirements are unchanged; this task moved no dense payload. |
| NDP05 | Exact token and aggregation-weight values plus result roots are preserved byte-for-byte in the retained report. |
| NDP06 | Fenced frame/contribution/result identities remain source-pinned to `92e398c2`; stale/invalid restart metadata stays rejected evidence. |
| NDP07 | Current-fence endpoint/route requirements remain unchanged and are not inferred from terminal scheduler data. |
| NDP08 | Fixed memory/byte admission bounds remain unchanged; no buffer or scale setting changed. |
| NDP09 | Credit semantics remain unchanged; no send/receipt or credit result is promoted from the failed jobs. |
| NDP10 | Checksums, once-only apply, and corruption/idempotence authority remain unchanged. Missing apply markers fail the end-to-end boundary. |
| NDP11 | Bounded replay/reassignment requirements remain unchanged; restart cascades do not establish successful replay/rejoin. |
| NDP12 | Owner-direct shared-result redistribution remains unchanged; the retained result root does not prove all eight node-local trainers applied it. |
| NDP13 | Absolute stage deadlines and route-local containment remain authoritative; job 5068873 failed at the node-1 apply deadline. |
| NDP14 | The stable versioned ABI/control-channel authority is unchanged. Historical v2.0 ABI evidence cannot be relabeled v2.1. |
| NDP15 | The fenced read-only checkpoint handoff and collective-free drain boundary is explicit: job 5068873 reached a handoff but failed all-eight-trainer apply markers, so it grants no checkpoint/apply qualification. |
| NDP16 | Provider, source, bundle, exact-token/weight/root, scheduler, failure, and deadline telemetry are retained; the failed verdict is not filtered out. |
| NDP17 | The exact-source G2 pass (`5068822`) remains a prerequisite artifact only. The subsequent real two-node job (`5068873`) failed; later G3/G4/G5, 4+, and scale rungs remain unqualified and unauthorized. |

NDP15–NDP17 are the decisive terminal boundary: a fenced native handoff is not
a successful all-trainer apply/checkpoint; telemetry must retain the failure;
and a passing synthetic G2 followed by a failed real two-node job does not open
any later rung.

## Canonical final seed preserved exactly

The reconciliation does not modify either immutable S3 object or its discovery
authority. The exact cold-start identity remains:

- immutable checkpoint:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- immutable manifest:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json`
- discovery pointer:
  `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- step: `2300930`
- accepted tokens: `150793748480`
- size: `7719680116`
- SHA-256:
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`

No checkpoint, manifest, pointer, launcher, materializer, or training file was
edited by this reconciliation.

## Exact reconciliation commands and retained outputs

The assigned worktree was
`/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-1547`
on branch `wg/agent-1547/reconcile-async-v21-authority`.

The required first Git network operation was:

```bash
git fetch origin
```

It completed with exit 0 and no diagnostic output. The initial relevant tips
after that fetch were:

```text
assigned HEAD                         0cf241e118dfb2c37b2318e94c062440dc39329e
local main                            243b94c7e791fad873964ddf622c95dacc6a936f
origin/main                           92e398c2914194847e2aa460f2f7bd0482f2fad4
origin/wg/agent-1508/re-run-fixed     d40b3047ba05e3d76b29687219bee1662b4e2b87
WG auto-squash                        efd7b8a91d657078d235fdd7e582fff3a3277d5c
```

The tree/ancestry audit used:

```bash
git show -s --format='%H %T %P' d40b3047 efd7b8a9 0cf241e1 92e398c2 a4c5fc4d
git diff --name-status efd7b8a9 d40b3047
git merge-base --all origin/main d40b3047
git merge-base --all origin/main HEAD
git merge-base --all d40b3047 HEAD
git merge-base --is-ancestor 92e398c2 d40b3047
```

The empty `git diff` proved the raw terminal branch and its WG auto-squash had
identical trees. The merge bases were, respectively, `92e398c2...`,
`a4c5fc4d...`, and `a4c5fc4d...`; `92e398c2` was already an ancestor of
`d40b3047`.

The exact reconciliation command was:

```bash
git merge --no-ff d40b3047ba05e3d76b29687219bee1662b4e2b87 -m "merge: reconcile async v2 terminal evidence (reconcile-async-v21-authority)"
```

Git reported:

```text
Merge made by the 'ort' strategy.
```

The resulting merge identity is:

```text
commit  9ebf06dbf1874d9c6b0daca7ae8211a2158abc46
tree    7dc08cfe6f9021a81e23faf5a1bbfd3794f7caa7
parent  0cf241e118dfb2c37b2318e94c062440dc39329e
parent  d40b3047ba05e3d76b29687219bee1662b4e2b87
```

Post-merge checks:

```bash
git diff --name-status HEAD^1 HEAD
git merge-base --is-ancestor d40b3047 HEAD
git merge-base --is-ancestor 92e398c2 HEAD
git status --porcelain=v1 --untracked-files=all
cmp -s AGENTS.md CLAUDE.md
```

All returned exit 0; both diff/status outputs were empty. The merge therefore
retained both histories without modifying the first-parent tree or removing
untracked data.

## Authority push receipt

The reconciled merge plus this validation record were first committed as:

```text
73d25e1fc72b6fdd95b867858d20661cd341109a
docs: record async v2 authority reconciliation (reconcile-async-v21-authority)
```

The exact authoritative push command was:

```bash
git push origin HEAD:main
```

The remote accepted the fast-forward and reported:

```text
To github.com:spinozans/emender
   92e398c2..73d25e1f  HEAD -> main
```

The immediate authority checks were:

```bash
git rev-parse HEAD origin/main
git ls-remote origin refs/heads/main
git merge-base --is-ancestor d40b3047 HEAD
git merge-base --is-ancestor 92e398c2 HEAD
git merge-base --is-ancestor 0cf241e1 HEAD
git status --porcelain=v1 --untracked-files=all
cmp -s AGENTS.md CLAUDE.md
```

They returned:

```text
git rev-parse HEAD        73d25e1fc72b6fdd95b867858d20661cd341109a
git rev-parse origin/main 73d25e1fc72b6fdd95b867858d20661cd341109a
git ls-remote main        73d25e1fc72b6fdd95b867858d20661cd341109a
all three ancestry checks exit 0
status output empty
cmp exit 0
```

This follow-up is provenance-only: it records the real push receipt and pushed
SHA after they exist. After committing and pushing it, the same fetch,
`rev-parse`, `ls-remote`, ancestry, clean-status, and byte-identity checks are
repeated, with the final tip recorded in the WG task log.
