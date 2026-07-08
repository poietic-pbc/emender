# Resilient quorum 256n x 1h approval package

Task: `synthesize-resilient-quorum-256n1h-package`  
Date: 2026-07-08

## Recommendation

**No-go for a 256-node x 1-hour debug run.**

Do not submit a 256n x 1h job from the current resilient-quorum evidence set.
The implementation and synthetic failure-injection tests passed, but the live
resilient-quorum ladder stopped at the first 1n rung. Job `4956022` failed
before metrics because the Slurm wrapper passed stale prototype arguments to
`scripts/frontier/async_diloco_e97_multinode.py`. No 8n, 64n, optional bounded
256n debug smoke, 256n x 1h, 12h, production, latest/last-mutating, or
training-continuation job was authorized or submitted by this workstream.

The final intended ladder boundary for this workstream remains:

```text
1n -> 8n -> 64n -> optional single bounded 256n debug smoke
```

The 256n x 1h run remains **approval-package-only**. A later explicit human
approval message or WG task is required before any 1h or 12h submission.

## Evidence set

| Area | Status | Evidence |
| --- | --- | --- |
| Design | Complete | `design-resilient-quorum-diloco-catchup` produced `reports/frontier/resilient-quorum-diloco-catchup-design-20260708.md` in commit `03fc554`. It keeps strict compiled-MPICH `MPI_Reduce` as the fast path/control and defines a separate resilient quorum mode with generation state, timeout, stale rejection, catchup, run-local latest, production guard, and metrics. |
| Implementation | Complete at unit/contract level | `implement-resilient-quorum-diloco` completed in commit `03cd39f`, touching `ndm/async_diloco.py`, MPI/compiled-helper transports, real trainer glue, E97 entrypoint tests, and focused async quorum tests. The task log records `py_compile` plus 43 focused async quorum, MPI, compiled-MPICH, real-trainer, and E97 entrypoint tests passing. |
| Failure injection | Complete at synthetic level | `validate-resilient-quorum-failure-injection` completed in commit `f15557a`; report `reports/resilient-quorum-failure-injection-validation-20260708.md` says 36 focused tests passed and no Slurm jobs were submitted. |
| 1n ladder | Failed before metrics | `run-resilient-quorum-1n8n64n-ladder` submitted only job `4956022` with debug QOS, `00:20:00`, requested `0.333333` node-hours, and seed `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`. The rung failed `FAILED 2:0` after `00:00:35` because `async_diloco_e97_multinode.py` rejected wrapper arguments. |
| 8n ladder | Missing | Not submitted after the 1n hard failure. |
| 64n ladder | Missing | Not submitted after the 1n hard failure. |
| Optional bounded 256n debug smoke | No-go / not submitted | `evaluate-resilient-quorum-256n-debug-gate` produced a no-go report in commit `0f21d3e`; no 256n debug job was submitted because the required 1n/8n/64n ladder evidence was absent. |

## What is robust now

| Claim | Grade | Evidence | Gap before 256n x 1h |
| --- | ---: | --- | --- |
| Nonjoining rank tolerance | `0.55` | Synthetic failure-injection test `test_failure_injection_missing_and_stuck_ranks_advance_without_unanimity` used 4 expected ranks, accepted ranks 0 and 1, timed out ranks 2 and 3, quorum 2, and asserted `quorum_status == "advanced"`, `accepted_updates == 2`, `timed_out_updates == 2`. Design requires non-arrivals to be accounted as timed out/failed without changing the denominator. | No live ladder artifact shows resilient mode advancing with nonjoining ranks on Frontier. |
| Stuck-rank timeout | `0.55` | Same failure-injection scenario explicitly marks stuck/missing ranks timed out; implementation logs report focused async quorum tests passing. Design states stuck ranks must be classified as `timed_out` and must not block a collective. | No 1n/8n/64n Slurm rung emitted `timed_out_ranks`, terminal state, or successful timeout advancement metrics. |
| Stale-generation rejection/acceptance | `0.60` | Failure-injection report says `test_late_base_generation_policy_accepts_current_and_rejects_old_with_metrics` accepts current `base_generation=2` updates and counts an old `base_generation=1` rank as stale. Design defines v1 accepted staleness as zero and forbids stale updates from mutating finalized state. | No live ladder metric reports stale/future/late classification at runtime. |
| Stale/restarted-rank catchup | `0.55` | Failure-injection report says `test_stale_worker_catchup_loads_latest_rebases_and_resets_base_generation` writes a run-local latest generation, has a behind worker observe `global_generation=5`, load current state, preserve local displacement by rebase, and reset next update base generation to 5. Design requires restarted ranks to load only authoritative latest and discard local stale deltas. | No Frontier rung produced catchup events, restart reload metrics, or proof that train.py ranks recover through this path under Slurm. |
| Checkpoint finalization/latest behavior | `0.35` | Design defines run-local `latest.json`/`latest.pt` advancement only after finalized manifests/checkpoints. Failure-injection report proves run-local latest isolation in simulation. The 1n ladder env intended a run-local output root and summary noted missing metrics. | The live 1n rung failed before checkpoint finalization. Summary fields for checkpoint finalization/latest were null or absent; no ladder rung published a resilient-quorum latest/checkpoint. |
| Production latest/last guard | `0.75` | Ladder report recorded production guard path `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`, resolved target before/after, unchanged symlink/file stat, and "No production `latest.pt` or `last` pointer was mutated". Gate report states no prohibited jobs or latest/last mutations were performed. | A successful rerun still needs before/after identity evidence for every passed rung and for any later bounded 256n debug smoke. |
| Metrics completeness | `0.20` | Design defines the required metrics: ranks started/joined, quorum threshold/accepted, stale/failed/timed-out/invalid/late lists, catchup, merge latency, bytes, loss windows, checkpoint ids, latest advancement, terminal status, and production guard. Synthetic tests assert selected counters. | Live ladder metrics are absent. The 1n rung did not create metrics JSON; 8n and 64n were not submitted. There is no 256n debug metrics packet. |

Overall readiness score for 256n x 1h: **0.22 / 1.00**. Confidence: high,
because the explicit go criteria are conjunctive and the first required live
rung failed before emitting the evidence needed for scale safety.

## Ladder and gate detail

### 1n rung attempted

`run-resilient-quorum-1n8n64n-ladder` submitted Slurm job `4956022`:

- Job name: `resilient-quorum-e97-1n`
- Partition/QOS: `batch` / `debug`
- Requested walltime: `00:20:00`
- Requested node-hours: `0.333333`
- Actual elapsed: `00:00:35`
- Approx actual node-hours: `0.009722`
- Nodes: `1`, `frontier05912`
- State/exit: `FAILED`, `2:0`
- Run root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/resilient_quorum_1n8n64n_ladder_20260708/20260708/4956022-20260708T093551Z`

The exact failure class was a wrapper/entrypoint CLI mismatch:

```text
async_diloco_e97_multinode.py: error: unrecognized arguments:
--worker-count-per-node 8 --tokens-per-step 1024 --delta-scale 1.0e-8
--task-id run-resilient-quorum-1n8n64n-ladder --slurm-job-id 4956022
--slurm-job-name resilient-quorum-e97-1n --requested-walltime 00:20:00
--requested-node-hours 0.333333 --command-file ... --stdout-path ...
--stderr-path ... --training-target ... --resume-check --production-latest-path ...
```

Because this failed during argument parsing, the run did not emit live values
for ranks started, quorum accepted, missing/stale/late/timed-out/rejected
counts, catchup events, merge latency, bytes, loss window, or latest/checkpoint
behavior.

### 8n and 64n rungs

No 8n or 64n resilient-quorum ladder job was submitted. This was correct
stop-on-first-failure behavior after the 1n rung failed hard.

### Optional bounded 256n debug smoke

The downstream gate report `evaluate-resilient-quorum-256n-debug-gate` is a
no-go. It cites:

- implementation and failure-injection tasks as complete;
- 1n job `4956022` as failed before metrics;
- 8n and 64n rungs as absent;
- the missing metrics packet for quorum, checkpoint, guard, and terminal state;
- no 256n/1h/12h/production submission and no production latest/last mutation.

No optional bounded 256n debug smoke was submitted, and it should remain blocked
until the live ladder is rerun cleanly.

## Residual risks and invalidated assumptions

- **Invalidated assumption: wrapper compatibility.** The 1n failure shows the
  resilient-quorum Slurm wrapper and current Python entrypoint do not share a
  valid CLI contract. This must be fixed before any scale conclusion is drawn.
- **Unit evidence does not substitute for live scale evidence.** Synthetic
  failure-injection tests cover the protocol semantics, but they do not prove
  Frontier launch, rank startup, transport deadlines, checkpoint publication,
  or metrics completeness at 1n/8n/64n/256n.
- **Checkpoint/finalization remains unproven in live resilient mode.** The
  design and synthetic latest tests are sound, but no resilient-quorum live
  rung finalized a checkpoint or advanced run-local latest.
- **Metrics schema is specified but not observed live.** The absence of a
  metrics JSON on job `4956022` means downstream scale safety fields are
  missing, not merely unfavorable.
- **Strict collective fast path remains fallback/control, not resilience proof.**
  The strict compiled-MPICH `MPI_Reduce` path remains available and tested per
  implementation/failure-injection logs. It is still the performance/control
  path for healthy all-rank worlds, but it does not satisfy nonjoining/stuck
  rank tolerance because strict all-rank collectives are expected to fail or
  hang when required ranks do not enter the reduction.

## Next fix and retest required for approval

1. Fix the wrapper/entrypoint CLI contract for
   `scripts/frontier/async_diloco_e97_2n8n_debug.sbatch` and
   `scripts/frontier/async_diloco_e97_multinode.py`, or explicitly route the
   ladder to the intended compatible entrypoint.
2. Add a focused argument-parsing test that covers the wrapper-only flags that
   broke job `4956022`: `--worker-count-per-node`, `--tokens-per-step`,
   `--delta-scale`, `--task-id`, `--slurm-job-id`, `--requested-walltime`,
   `--requested-node-hours`, `--resume-check`, and
   `--production-latest-path`.
3. Rerun the live ladder from the start: 1n first, 8n only after 1n passes, 64n
   only after 8n passes.
4. For each passed rung, capture ranks started/joined, quorum accepted,
   missing/stale/late/timed-out/rejected counts, catchup events, merge latency,
   bytes, loss window, checkpoint ids, latest/finalization behavior, terminal
   status, and production latest/last before/after identity.
5. Only after a clean 1n/8n/64n ladder should a later gate consider exactly one
   bounded 10-20 minute 256n debug smoke. The 256n x 1h run should not be
   reconsidered until that bounded 256n debug smoke also emits complete metrics
   and leaves production latest/last unchanged.

## Non-authorizing future 256n x 1h shape

This section is included only so a human can see what a future approval package
would need to say. It is **not a recommendation** and **not approval to submit**.

If a future clean rerun changes the recommendation, the proposed 256n x 1h
debug shape should be:

- Nodes: `256` Frontier nodes, one train.py rank per GPU, expected GPU ranks
  `2048`.
- Walltime: `01:00:00`.
- QOS/partition: debug or other human-approved non-production queue, explicitly
  recorded before submission.
- Expected node-hours: `256`.
- Seed policy: use the current verified E97 checkpoint seed only after recording
  symlink target, stat, and checksum/identity; do not silently switch seeds.
- Output policy: run-local output root under a new debug directory; no
  production chain `latest.pt`, `last`, or shared production pointer mutation.
- Checkpoint/finalization monitoring: require per-generation manifest,
  recovery/export checkpoint records, run-local latest identity, walltime
  finalization record, and terminal summary before declaring success.
- Metrics monitoring: require rank starts, quorum denominator/threshold,
  accepted/stale/late/timed-out/rejected/failed counts and rank lists, catchup
  events, merge latency, bytes, loss windows, checkpoint ids, and terminal
  status.
- Stop conditions: stop or fail closed on argument parsing errors, missing
  metrics JSON, below-quorum deferral beyond policy, latest/checkpoint
  finalization failure, production guard change, non-finite loss/update, missing
  rank-start accounting, or unexpected strict-collective-only execution in
  resilient mode.
- Rollback/no-mutation guard: record production latest/last identities before
  and after; if changed without explicit production approval, mark the run
  failed and do not promote any artifact.

Future human approval wording would need to be explicit, for example:

```text
I approve submitting exactly one 256-node x 1-hour resilient-quorum E97 debug
run using the verified seed <seed path and resolved target>, run-local output
root <path>, no production latest/last mutation, no 12h or production
continuation, and stop/fail-closed conditions as listed in
reports/frontier/synthesize-resilient-quorum-256n1h-package-20260708.md.
```

That wording is intentionally not satisfied by the current task. A later human
message or WG task must provide it after the missing evidence is produced.

## Validation statement

This task produced a report only. It did not submit Slurm jobs, did not create
an auto-submitting follow-up task, did not authorize 1h/12h/production work, and
did not mutate production latest/last.
