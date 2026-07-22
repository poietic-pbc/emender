# Pipelined native DiLoCo two-node gate attempt

Date: 2026-07-20  
Task: `validate-pipelined-native-2`  
Result: **manager-freeze-fix exact-source G2 passed; sole K40 replacement 5055899 pending terminal pass**

## Manager-freeze convergence retry (K40 submitted, 2026-07-22)

The prior sole K40 job `5053690` is terminal `FAILED 1:0` after 5:52 on
exactly two nodes.  Its retained evidence shows that all 16 trainers completed
their first K40 and submitted generation-0 contributions, but no atomic
generation completed: managers encountered a deterministic freeze timeout,
native `FREEZE` in an invalid lifecycle state, and a missing peer route during
recovery.  No later serial phase was admitted.

Authoritative `origin/main` has since advanced from that job's source
`53441395245af7fbe767c2e25cc3ad379db07b0e` to
`5f180e25fd0a54852892d5ff59a97b3c1d8737ff`, which contains the reviewed
manager-freeze convergence merge.  A fresh clean `main` clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-freeze-final-20260722T223000Z/source`
matches the fetched remote identity exactly.  The canonical Frontier
GNU/ROCm environment rebuilt its native bundle and CTest passed 10/10.  The
exact build-manifest SHA-256 is
`3b5d2847420fb8ec10038186347664b4faa52d61d68c70b5537c323fdc4e6a48`.

An adjacent per-user scheduler query was empty.  The canonical fail-closed G2
launcher then submitted exactly one job, `5055869`, requesting exactly two
nodes and 20 minutes.  It completed `0:0` in 2:57 on
`frontier[06173,06175]`; exact-source correctness/integrity passed, and the
retained full-layout gate SHA-256 is
`8b44fcc12a0224b49c836afcb32689004d8e973835e844c3b8b6fbfa925c2c33`.
The prior G2 result is retained only as historical telemetry because the
source identity changed.

After G2 was terminal and another scheduler query proved the user queue empty,
the serial controller rebuilt and re-attested the exact source and submitted
exactly one clean K40 replacement, job `5055899`, for exactly two nodes, five
K40 generations, and the two-hour debug bound.  Its first observed state is
`PENDING (Priority)`.  No duplicate, later serial phase, or job larger than
two nodes was submitted.  Live overlap, resilience, rejection, and restart
claims remain withheld pending terminal evidence.

At the 2026-07-22 19:10 EDT resume checkpoint, both `squeue` and `sacct`
still recorded `5055899` as the sole equivalent job: `PENDING (Priority)`,
zero elapsed, two requested nodes, no assigned nodes, and a two-hour limit.
Slurm's current estimate was 20:40 EDT.  This ordinary priority wait does not
authorize a duplicate, so monitoring continues against this same job and no
submission command was issued during the checkpoint.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  This checkpoint
specifically preserves exact pushed identity, serialized two-node admission,
bounded execution, and fail-closed precursor ordering required by
R10/R13/R14/R16 and NDP02/NDP03/NDP13/NDP16/NDP17.  Live overlap,
resilience, rejection, and restart claims remain withheld rather than inferred
from scheduler admission.

## Final K40 replacement terminal harvest (2026-07-22)

The sole authorized K40 replacement, Slurm job `5053690`, reached terminal
`FAILED (1:0)` after 5:52 on exactly two nodes,
`frontier[05345,05350]`.  Its batch allocation requested two nodes and the
two-hour debug bound.  No duplicate, later serial phase, or job larger than
two nodes was submitted.

The immutable execution root is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-20260722T180000Z`.
The exact source is `53441395245af7fbe767c2e25cc3ad379db07b0e`, the native
bundle SHA-256 is
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`,
and the acceptance manifest SHA-256 is
`1f593510d492fcc79c3e3634575fd499eec2b8e57f2a87b9520be8cbe5c42877`.
The preceding exact-source G2 job `5053588` passed correctness/integrity on
two nodes; its measured speedup was 3.866x and remains telemetry under the
approved policy.  It reported 44,322,599,424 useful bytes and 44,323,138,304
wire bytes in each direction, with 25.596-second median and 25.719-second
maximum generation intervals.

The replacement did not complete one atomic generation, so it cannot support
the five-generation overlap, cadence, rejection, loss/rejoin, or checkpoint
claims.  All 16 trainers completed their first K40 and published generation-0
contributions, and both native services retained generation-0 evidence.  The
managers then diverged during the generation-0 freeze: one expired the
deterministic freeze deadline, while the other observed native `FREEZE` in an
invalid lifecycle state; recovery attempts also observed a missing peer route
`(0, 1)`.  The allocation supervisor consequently terminated the step.  This
is a fail-closed lifecycle/coordination failure before an accepted atomic
result, not a strict-overlap performance failure.

The batch stdout/stderr were moved intact from the otherwise-clean source
clone into `phases/clean-overlap/slurm/`.  A resumable-controller harvest was
then attempted.  It correctly did not submit another job, but exposed a
separate terminal-query defect: after Slurm aged the job out of `squeue`, the
controller treated `squeue -j 5053690` exit 1 as a launcher refusal instead of
falling back to `sacct`.  The serial state therefore still records the clean
phase as active even though `sacct` is authoritative and terminal.  Advancing
the fault/rejection/checkpoint phases would violate the clean-gate dependency,
so they remain withheld.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  Exact identity and
two-node admission satisfy the applicable R01/R10/R14/R16 and
NDP02/NDP03/NDP16/NDP17 checks.  Atomic fail-closed behavior was preserved,
but the live run does not validate the remaining R04-R12/NDP06-NDP15 runtime
acceptance claims.

## Production-entrypoint/rank-containment retry (K40 running, 2026-07-22)

Authoritative `origin/main` now resolves to
`53441395245af7fbe767c2e25cc3ad379db07b0e`, which contains the reviewed
production overlap entrypoint and rank-level trainer-failure containment
merges.  This is newer than the source of terminal job `5050642`, so its G2
artifact is historical telemetry and cannot attest the replacement bundle.

A fresh clean `main` clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-20260722T180000Z/source`
matches fetched `origin/main` exactly.  The canonical Frontier GNU/ROCm
environment rebuilt the native bundle from that source; CTest passed 10/10.
The build manifest SHA-256 is
`33693b338d664e380174beac23f59f168f94ea115acd1709e1a43503472490e3`, and
the recorded bundle SHA-256 is
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.

An adjacent per-user queue check was empty.  The canonical fail-closed G2
launcher then submitted exactly one job, `5053588`, requesting exactly two
nodes and 20 minutes.  It completed `0:0` in 2:59 on
`frontier[04040,04042]`.  Exact-source production attestation passed, and the
retained full-layout gate SHA-256 is
`eb7d60395f8842f5a78a516e1176f5108281817ad95e5cf29a7ef1cebfc9c2a8`.

After G2 was terminal, its clone-local scheduler logs were moved intact into
the immutable job artifact directory, another adjacent scheduler query proved
the user queue empty, and the serial controller rebuilt and re-attested the
same exact source and bundle.  It submitted exactly one clean K40 replacement,
job `5053690`, for exactly two nodes, five K40 generations, and the two-hour
debug bound.  The acceptance manifest SHA-256 is
`1f593510d492fcc79c3e3634575fd499eec2b8e57f2a87b9520be8cbe5c42877`.
The job is running on `frontier[05345,05350]`; no duplicate, later serial
phase, or job larger than two nodes has been submitted.  Live claims remain
withheld pending terminal harvest and strict telemetry validation.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  This checkpoint
specifically preserves exact pushed identity, collective-free two-node
admission, bounded execution, and fail-closed predecessor ordering required
by R10/R13/R14/R16 and NDP02/NDP03/NDP13/NDP16/NDP17.  Live overlap,
resilience, rejection, checkpoint-failure, and restart claims remain pending
rather than inferred from scheduler admission.

## Final overlap-scheduler retry (terminal harvest, 2026-07-22)

Slurm records the sole final replacement, job `5050642`, as terminal
`FAILED 1:0` after 31:08 on exactly two nodes,
`frontier[08317,08701]`.  It ran from 2026-07-22 05:51:04 through 06:22:12
EDT.  The allocation payload completed successfully; the batch failed only
when the strict retained-evidence validator rejected the live timing trace
with `generation 0 background did not overlap 1 K40 compute`.

The payload nevertheless completed all five requested atomic K40 generations.
The retained pool-control records close generations 0 through 4 with two READY
node-peer contributions and 5,245,440 accepted tokens each.  Five immutable
handoffs and five restartable checkpoints were published, from
`generation-00000001-fence-00000001` through
`generation-00000005-fence-00000001`; each checkpoint is 7,899,873,331 bytes.
All 16 trainer lanes have fenced native apply receipts through generation 4,
and both node snapshots were harvested.  Dense Python socket and trainer-spool
bytes remain zero.  This establishes exact two-node execution, five atomic
generations, safe fenced application, bounded native handoff, and durable
checkpoint publication, but not the required generation-g/background versus
generation-(g+1)/K40 overlap, sub-10% foreground idle, or at-most-1.25x
steady cadence.

Because the primary clean admission gate failed, the serial controller
correctly withheld the loss/rejoin, invalid-result, checkpoint-publication
failure, and fresh-restart phases.  No duplicate and no job larger than two
nodes was submitted.  The reviewed overlap scheduler change therefore does
not pass the live production flow despite its simulated tests; scale-out must
remain blocked pending a new root-cause fix and another exact-source two-node
replacement.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  Exact identity, G2,
two-node admission, READY membership, five atomic publications, fenced apply,
bounded queues, and fail-closed validation support R04/R06/R07/R10/R11/R13/R14
and NDP02/NDP03/NDP10/NDP13/NDP15/NDP17.  The failed live overlap gate and
unexecuted serial resilience/rejection/restart phases leave R12/R14/R16 and
NDP16/NDP17 acceptance incomplete.

## Final overlap-scheduler retry (submission checkpoint, 2026-07-22)

The prior admission blocker is cleared.  A fresh fetch resolves authoritative
`origin/main` to `32fd9ab15c6908827d094b21ff638f8ec2a24c2b`; reviewed overlap
scheduler commit `84a81e40` is an ancestor.  A new clean `main` clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-overlap-final-20260722T091500Z/source`
matches that pushed identity exactly.  Under the canonical Frontier
environment its clean native build passed CTest 10/10, and the focused exact
two-node renderer/performance suite passed 11/11.  The recorded native bundle
SHA-256 is
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`;
the build-manifest SHA-256 is
`a88fda81badfed62ce82eacf9df82e3b0067b945ca5dd14c31637742b8f36535`.

Because the exact source changed, the old G2 artifact is telemetry only and
cannot attest this bundle.  Immediately after an empty per-user queue check,
the canonical fail-closed G2 launcher submitted job `5050569` for exactly two
nodes.  It reached terminal `FAILED 66:0` after eight seconds, before source
validation or dataplane execution, because the fresh clone has no clone-local
Python environment and the canonical shared interpreter had not been passed
as `NDP_PYTHON_BIN`.  Its stdout/stderr are preserved beneath
`g2-artifacts/5050569/`.

After that job was terminal and the user queue was empty again, the same
exact-source launcher was corrected with the explicit canonical Python path
and submitted job `5050571` for exactly two nodes.  It completed `0:0` in 2:55
on `frontier[08029-08030]`.  The exact-source production attestation passed;
the full-layout gate SHA-256 is
`5c21aeed49872280d255e9164b7f3ee94122b4af2378665248d99d27bb4a6d08`.
It records two leased CXI `FI_EP_RDM` endpoints, exact-reference agreement,
checksum/stale rejection, zero route/CQ errors, 44,322,599,424 useful bytes and
44,323,138,304 wire bytes per direction.  Its timed generations were
23.513984104, 23.924894782, and 22.966800809 seconds; 936,765,198.38 logical
B/s and 4.209x the retained Python comparison are telemetry only.

After G2 was terminal and another adjacent scheduler query proved the user
queue empty, the canonical serial controller rebuilt and re-attested the exact
bundle, then submitted exactly one final clean-overlap K40 replacement: job
`5050642`, exactly two nodes, five K40 generations, and a two-hour bound.  At
2026-07-22 05:21 EDT it is the sole user job and is `PENDING (Priority)`, with
zero elapsed time and no assigned nodes.  The immutable acceptance manifest
SHA-256 is
`97d3a0f5df69c759df570908f8ee4068b629d30e0fd4b8540b66063e6fe19b59`;
the authoritative-stage attestation SHA-256 is
`3183fb7ae31e715b74fd5dd5ca3010e0c5af6d27700ee540e6cc1eb03cc25f05`.
No duplicate, later serial phase, or job larger than two nodes has been
submitted.  Live overlap, cadence, rejection, resilience, and restart claims
remain pending terminal evidence rather than inferred from admission.

This checkpoint conforms to *Resilient DiLoCo Compute Pool* v1
R10/R13/R14/R16 and native data plane NDP02/NDP03/NDP13/NDP16/NDP17:
authoritative identity and bundle provenance are exact, admission is serialized,
and the production gate remains fail-closed.  Live R04/R06/R07/R11/R12/R14/R16
and NDP10/NDP13/NDP15/NDP16/NDP17 evidence remains pending rather than inferred
from scheduler state.

## Overlap-scheduler-fix admission checkpoint (2026-07-22)

The retained sole final job `5047497` remains terminal `FAILED 1:0`; a fresh
`sacct` query confirms that it ran for 30:58 on exactly two nodes from
2026-07-21 19:10:53 EDT through 19:41:51 EDT.  A fresh per-user `squeue`
query is empty, so there is no active equivalent job and no scheduler overlap.

The reviewed live-overlap scheduling fix is commit `84a81e40`, but a fresh
`git fetch origin main` on 2026-07-22 still resolves authoritative
`origin/main` to `5c4950f16bd9ce7cb7d96ab9b67e24efb61ed3a6`.  Consequently the exact
authoritative source has not changed since job `5047497`, and rebuilding or
submitting from it would reproduce the known-bad foreground scheduling rather
than validate the fix.  The exact-source admission gate therefore remains
fail-closed: no replacement, duplicate, later serial phase, or job larger than
two nodes was submitted.

This checkpoint conforms to Compute Pool R10/R13/R14/R16 and native data plane
NDP02/NDP03/NDP13/NDP16/NDP17 by preserving pushed-source identity,
single-allocation admission, and fail-closed bundle/runtime attestation.  The
live criteria in R04/R06/R07/R11/R12/R14/R16 and
NDP10/NDP13/NDP15/NDP16/NDP17 remain incomplete until `84a81e40` (or its
reviewed equivalent) is merged and pushed to authoritative main, after which
exact-source G2 must be refreshed and exactly one two-node replacement may be
submitted from the new identity.

## Telemetry-harvest-fix retry (terminal harvest)

Slurm records the sole final replacement, job `5047497`, as terminal
`FAILED 1:0` after 30:58 on exactly two nodes,
`frontier[07680,07808]`.  The allocation ran from 2026-07-21 19:10:53 EDT
through 19:41:51 EDT.  Its allocation step completed successfully; the batch
failed only when the fail-closed performance validator rejected the retained
two-node trainer telemetry with
`generation 0 background did not overlap 1 K40 compute`.

The payload nevertheless completed all five requested atomic K40 generations.
All 16 trainers reached generation 5 `applied`, both managers reached
generation 5 `published`, and all trainers and managers exited zero.  The
bounded pipeline telemetry reports `handoff_high_water=1`,
`result_high_water=1`, no replacements, and no rejected or stale results for
the clean workload.  Five finalized handoffs and five restartable checkpoints
remain in the immutable phase directory, from
`generation-00000001-fence-00000001` through
`generation-00000005-fence-00000001`.  Native services were evicted only for
`allocation_complete` after role completion.

This is useful evidence for exact two-node execution, five atomic
generations, safe application, bounded/latest-only handoff, and durable
checkpoint publication.  It does not prove the required generation-g
background/generation-(g+1) compute overlap, below-10% foreground idle, or
at-most-1.25x cadence.  Since the primary clean admission gate failed, the
serial controller correctly withheld fault/rejoin, invalid-result,
checkpoint-publication-failure, and fresh-restart phases.  No duplicate and
no four-node-or-larger job was submitted; scale-out remains blocked.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  The exact-source G2
gate and clean run support R04/R06/R07/R10/R11/R13/R14 and
NDP02/NDP03/NDP10/NDP13/NDP15/NDP17.  The failed live overlap gate and
unexecuted resilience/rejection/restart phases leave R12/R14/R16 and
NDP16/NDP17 acceptance incomplete.

## Telemetry-harvest-fix retry (submission and G2 evidence)

Exact-source G2 refresh job `5047138` completed successfully (`0:0`) in 2:55
on exactly two nodes, `frontier[05911-05912]`.  Its correctness/integrity gate
passed with SHA-256
`c41ec3ab8e0b3c54da1d16218a9188e8d7dfb8d6713575a4e089aafdecc3a5a8`;
the gate records exact source `5c4950f16bd9ce7cb7d96ab9b67e24efb61ed3a6`,
bundle `59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`,
two CXI endpoints, exact-reference agreement, checksum/stale rejection, zero
all-rank barriers, zero dense-socket bytes, 44,322,599,424 useful bytes and
44,323,138,304 wire bytes in each direction.  The three timed generations
were 22.984653924, 23.023732634, and 25.124313882 seconds; 956,712,029.89
logical B/s and the 4.298x retained-Python comparison are telemetry only.

After G2 reached terminal success, a fresh scheduler query proved the user
queue empty.  The canonical serial controller rebuilt the exact authoritative
source, passed CTest 10/10, and re-attested the same source and bundle.  The
first submission attempt correctly refused because G2 Slurm logs made the
clone untracked-dirty; those logs were moved intact into the immutable G2
artifact directory and the source was rechecked clean.  At 2026-07-21 18:09
EDT the controller submitted exactly one final clean-overlap K40 replacement,
job `5047497`, for exactly two nodes, five generations, K40, and a two-hour
bound.  Resumed `squeue`/`sacct` checks through 2026-07-21 19:05 EDT still
record it as `PENDING (Priority)`, with zero elapsed time, no assigned nodes,
and the full two-hour limit remaining.  At that check, Slurm estimated a
2026-07-21 20:52 EDT start; this estimate is telemetry, not a second submission
condition.  It is the sole user job; no duplicate, later
serial phase, or four-node-or-larger job has been submitted.  The task remains
active through terminal monitoring and serial artifact harvest.

Authoritative `origin/main` resolves to
`5c4950f16bd9ce7cb7d96ab9b67e24efb61ed3a6`, containing the reviewed
`fix-exact-2n` retained-evidence telemetry correction.  A fresh clean `main`
clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-telemetry-fix-20260721T213400Z/source`
matches that pushed identity exactly.  Its canonical Frontier native bundle
was rebuilt under the adjacent `native-stage/preflight` root and CTest passed
10/10.

The user Slurm queue was empty immediately before submission.  Because this
source identity differs from the retained job-5042988 gate, the fail-closed
path submitted exact-source G2 correctness/integrity refresh job `5047138` at
2026-07-21 17:35 EDT for exactly two nodes.  It is the sole user allocation;
no K40 replacement, duplicate, later serial phase, or four-node-or-larger job
has been submitted.  G2 must reach terminal success before the single final
K40 acceptance replacement is admissible.

Conformance is checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  Exact pushed
source, a clean native build, fixed two-node capacity, empty-queue admission,
and fail-closed source/G2 fencing support R10/R13/R14/R16 and
NDP02/NDP03/NDP13/NDP16/NDP17.  Live overlap, cadence, fault/rejection,
failed-publication retention, and fresh-restart claims remain pending.

## Supervisor-fix retry (terminal harvest)

Slurm records job `5043045` as terminal `FAILED 1:0` after 31:08 on exactly
two nodes, `frontier[02714,03991]`.  The allocation nevertheless completed
the requested clean payload before its post-run gate failed: all 16 trainers
applied generations 0 through 4, both managers published generation 5, and
the retained handoff/checkpoint set contains five restartable payloads,
`generation-00000001` through `generation-00000005`.  Supervisor events show
every trainer and manager exiting zero and both native services being evicted
only for `allocation_complete`; the queue remained bounded/latest-only with
no all-rank wait.

The batch then invoked `validate_pipelined_e97_performance.py` on
`$RESILIENT_E97_BULK_ROOT/telemetry`.  The supervisor had already harvested
the node-local bulk roots into
`retained-evidence/node-{0,1}/telemetry`, so that path contained no records and
the validator failed closed with `ValueError: no real trainer step telemetry`.
A read-only replay over the combined retained trainer telemetry advanced past
that plumbing error but rejected `generation 0 background did not overlap 1
K40 compute`.  Therefore five atomic K40 generations and durable publication
are established, but the required steady-state overlap, below-10% idle, and
at-most-1.25x cadence gates are not claimed.

Because the clean phase did not pass its admission validator, the serial
controller correctly withheld the authorized loss/rejoin, invalid-result,
checkpoint-publication-failure, and fresh-restart phases.  No duplicate and
no four-node-or-larger job was submitted.  Follow-up WG task `fix-exact-2n`
tracks the retained-evidence validator plumbing regression; it must preserve
the live overlap/cadence rejection rather than converting telemetry into a
pass.

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  Exact identity,
two-node admission, five atomic publications, safe application, bounded
queues, and fail-closed post-validation support R04/R06/R07/R10/R11/R13/R14
and NDP02/NDP03/NDP10/NDP13/NDP15/NDP17.  The missing live overlap/cadence
pass and unexecuted resilience/rejection/restart phases leave
R12/R14/R16 and NDP16/NDP17 acceptance incomplete, so scale-out remains
blocked.

Exact-source G2 refresh job `5042988` completed successfully (`0:0`) on
exactly two nodes, `frontier[05335,05337]`, in 3:04.  Its passing
correctness/integrity gate is retained at
`g2-artifacts/5042988/full-layout-gate.json` beneath the immutable controller
root, with SHA-256
`f9acb5af938df5f8b47798d08dc67442a4d71986bb557a6acd61245e24042a6f`.
The throughput fields remain telemetry rather than K40 admission evidence.

After G2 was terminal, a fresh scheduler query proved the user queue empty.
The serial fail-closed launcher rebuilt the exact source, passed native CTest
10/10, and attested bundle digest
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.
The acceptance native manifest SHA-256 is
`c0efedd312b27a19fd0ab911e2e21934359404a9359f0ba691f630726c4b059c`;
the immutable acceptance manifest SHA-256 is
`d1ebb1c2c172cc3e81f06774a94be9303f131071b79db17d070eb8f4342e10b1`.

At 2026-07-21 08:24 EDT the controller submitted exactly one clean-overlap
replacement, job `5043045`, requesting exactly two nodes, five K40
generations, and a two-hour bound.  It is the sole user job and is currently
`PENDING (Priority)`.  At 2026-07-21 08:32 EDT, Slurm's non-binding estimate
was a 13:04 EDT start on `frontier[05238-05239]`; the job still had no
allocation.  No duplicate, later
serial phase, or four-node-or-larger job has been submitted.  The task remains
active through terminal completion; live overlap, cadence, fault/rejection,
and restart claims remain pending rather than inferred from queue state.

Authoritative `origin/main` resolves to
`084f36f05cfa66ed8b2c416941824d80b88cbdf9`, containing the reviewed
supervisor generation-progress recovery fix. A fresh clean `main` checkout at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-supervisor-fix-20260721T114200Z/source`
exactly matches that pushed identity. In the canonical Frontier environment,
its native bundle built successfully and CTest passed 10/10; the immutable
preflight bundle is retained under the adjacent
`native-stage/preflight/install` directory.

The user Slurm queue was empty before submission. Because the source identity
changed from the job-5042670 gate, the fail-closed path submitted exact-source
G2 correctness/integrity refresh job `5042988` at 2026-07-21 07:43 EDT for
exactly two nodes. At this checkpoint it is the sole user allocation and is
`PENDING (Priority)` with no scheduler start estimate. No K40 replacement,
duplicate, fault phase, or four-node-or-larger job has been submitted. The task
remains active through terminal G2 and, only after a passing gate and another
empty-queue check, the single authorized exact two-node K40 replacement.

Conformance is checked against *Resilient DiLoCo Compute Pool* v1 R01-R16 and
*Native resilient DiLoCo data plane* v1 NDP01-NDP17. The exact pushed source,
clean native build, fixed two-node capacity, no-overlap guard, source/bundle
fencing, and fail-closed K40 admission conform to R10/R13/R14/R16 and
NDP02/NDP03/NDP13/NDP16/NDP17. Live five-generation, overlap/cadence,
loss/rejoin, rejection, failed-publication retention, and fresh-restart
evidence remains pending rather than claimed.

## Generation-identity-fix retry

Authoritative `origin/main` resolves to
`dd50c5123f72d91f0618059f6689c2df9ea36233`, containing the reviewed
generation-identity recovery fix. A clean `main` clone at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-generation-identity-20260721T081000Z/source`
exactly matches that pushed commit. Its canonical Frontier native build passed
CTest 10/10; the immutable preflight installation is under the adjacent
`native-stage/preflight/install` directory.

The user Slurm queue was empty immediately before submission. Because this
source identity differs from the retained job-5039234 G2 artifact, the
fail-closed launcher submitted exact-source G2 refresh job `5042670` at
2026-07-21 04:09 EDT for exactly two nodes. It completed `0:0` in 2:54 on
`frontier[04129,04132]` and published the passing correctness/integrity gate at
`g2-artifacts/5042670/full-layout-gate.json` (SHA-256
`c5a95ee303271686bbd2b6051829854e680287bdb3ae1a4f019fcdd7160804cc`).

After G2 left the queue, the serial fail-closed controller rebuilt and
re-attested the exact source. The acceptance native manifest SHA-256 is
`bdc72415338b74112087756762aa3819ee6762dd70a271878bd31b5f3b433ac0`,
and its recorded native bundle digest is
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.
The immutable acceptance manifest SHA-256 is
`10d2167fc9836b9223a6e1dbafbe0b3c807ab20dfaa9ed39c34c368027ad1ee3`.
The adjacent empty-queue guard then submitted exactly one authorized K40
replacement, clean-overlap job `5042682`, for exactly two nodes and five K40
generations. Slurm records that it ran on `frontier[05631,08192]` from
2026-07-21 04:51:41 through 05:05:59 EDT and terminated `FAILED 1:0` after
14:18; its Python step failed after 13:32. No duplicate or
four-node-or-larger job was submitted.

The replacement completed and safely applied generations 0 and 1 on both
nodes. `handoff/latest.json` points to the finalized generation-2 manifest,
which records 10,490,880 accepted tokens and manifest SHA-256
`fc98357c2377a94f785f4776a4eba636e90235780cd91c6fe74c4ffd2fd1bf97`;
both generation-1 and generation-2 restartable checkpoint payloads remain on
disk. All 16 trainers report two handoffs and two applied results, queue
high-water marks of one, no replacements, and zero stale or rejected results.
For node-0 trainer 0, the generation pipeline intervals were 291.345 and
337.965 seconds, foreground waits were 9,619 and 22,243 ns, and both stayed
inside the 420-second bound. Generation-1 manager telemetry separately records
17.560 seconds local reduction, 35.036 seconds owner contribution, 22.066
seconds owner redistribution, and 65.028 seconds total redistribution, with
receive queue high-water one, zero Python dense-socket bytes, zero native
retries/CQ/route errors, and roughly 82.35 MB/s observed native throughput.

Trainer timestamps prove real generation-1 optimizer work through step 79
while generation-0 background collection/reduction/redistribution and apply
completed, and generation-2 safe-boundary redistribution/application began.
This is direct g-background/g+1-compute overlap evidence and keeps observed
foreground control-plane idle far below 10%. It is not the requested
five-generation steady-state sample, so the full cadence gate is not claimed.

The terminal defect is in supervision rather than data-plane integrity. During
the long, still-progressing generation-2 per-trainer redistribution/apply
sequence, both managers were killed for `progress_deadline`, their native
services were rejoined, and the replacement managers restarted at
`runtime_import`. The supervisor then immediately killed them for
`first_atomic_generation_deadline`, even though two atomic generations and a
finalized generation-2 checkpoint already existed, and exhausted their restart
budgets. Thus the clean phase failed after two of five generations and the
serial controller correctly did not submit the fault/rejoin, invalid-result,
checkpoint-publication-failure, or fresh-restart phases.

Conformance is checked against *Resilient DiLoCo Compute Pool* v1 R01-R16 and
*Native resilient DiLoCo data plane* v1 NDP01-NDP17. Exact pushed source,
clean native build, refreshed G2 correctness/integrity, empty-queue guard,
fixed two-node capacity, no-scale guard, R04/R06 safe application, R07/R11
bounded latest-only queues, and NDP10/NDP13/NDP15 integrity/ownership behavior
pass in retained evidence. The premature manager eviction/restart accounting
violates the required R12/R14/R16 and NDP16/NDP17 resilient progress gate;
five-generation cadence plus the named fault, rejection, publication-failure,
and fresh-restart phases remain unproven.

## Lifecycle-fix replacement (terminal harvest)

Authoritative `main` and `origin/main` resolve to
`3cfed722beb086d015cff254f473af2a63eaa492`, which contains the reviewed
generation-start lifecycle repair.  A clean authoritative clone was built in
the canonical Frontier environment.  The native build passed CTest 10/10 and
recorded bundle digest
`59fa632b98999e522be6fee3cda98d095a0fc4c85b0b3a95286b0eb61c19fa6d`.

Because this source differs from the historical job-5037939 source, G2 was
refreshed.  Job `5039166` failed in batch preflight before launching the native
payload because the clean clone did not contain the default clone-local Python
environment; it produced no gate.  After explicitly binding the canonical
approved Python 3.12 interpreter and confirming the user queue was empty, G2
retry `5039234` completed `0:0` in 3:01 on exactly
`frontier[05865,05870]`.  Its exact-source correctness/integrity attestation
passed.  The gate is retained at
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-lifecycle-fix-20260720T202500Z/g2-artifacts/5039234/full-layout-gate.json`
with SHA-256
`d6300536df887cf71f4393214707d9ffab68c89527ce31e981d043d5e304d449`.
G2 telemetry reports 22.792855478, 22.901945645, and 23.460527776 seconds,
961,799,592.29 logical B/s, and 4.321x the retained Python baseline; this ratio
is telemetry rather than K40 admission.

After job 5039234 cleared and a second empty-queue check, the exact serial
launcher rebuilt and re-attested the same source/bundle (CTest 10/10) and
submitted the single authorized replacement K40 clean-overlap job `5039258`.
The immutable controller root is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-lifecycle-fix-20260720T202500Z/`.
Its acceptance manifest requires exactly two nodes, five K40 generations,
background-g/compute-(g+1) overlap, foreground idle below 10%, and cadence no
worse than 1.25x raw K40 compute when background work fits.

Slurm accounting records that the sole replacement started at
`2026-07-20T17:59:06` on exactly `frontier[06619-06620]` and terminated
`FAILED 143:0` at `18:09:09` after `00:10:03`; its Python step failed `1:0`
after `00:08:49`. No duplicate K40 job and no 4-node-or-larger job was
submitted.

The replacement materially advanced beyond job 5037971. Generation 0
completed K40 on all 16 trainers, accepted both nodes and 5,245,440 tokens,
produced identical 5,506,770,496-byte result root
`ede07c3e8d55504bcf189551f5f5cc5fba4f6b6e3bb10fccc4016765bf3a27a3`
on both managers, applied the result at the safe boundary, and atomically
published the finalized generation-1 checkpoint. The retained handoff is
7,899,873,331 bytes with SHA-256
`bf013bed934da7c54c339163bddd204c34457df5e326e851fb01778772242151`;
`latest.json` points only to that finalized manifest. Generation 1 then began
real optimizer/forward/backward work. This proves one atomic generation and
bounded safe handoff/application, but not the required five generations or
generation-g background overlap with generation-(g+1) compute: the observed
checkpoint commit preceded the recorded generation-1 training stages.

Generation-0 per-trainer pipeline elapsed times ranged from 288.477 to
300.584 seconds. All 16 records report one handoff and one applied result,
handoff and result queue high-water marks of one, zero replacements, zero
stale/rejected results, and a maximum foreground wait of 14,058 ns (0.0049%
of the fastest recorded pipeline interval). The two manager contribution
phases took 34.986/35.017 seconds and redistribution took 22.009/22.068
seconds, with receive queue high-water one and no Python dense-socket bytes.
These are useful bounded-queue, background-phase, and foreground-idle
measurements. They are not a five-generation steady-state cadence result, and
the generic performance validator refuses the incomplete artifact layout with
`no real trainer step telemetry`.

The terminal defect is a generation-identity handoff mismatch exposed during
trainer-7 recovery on both nodes: `GenerationMetadata.from_json()` rejected
`native-generation-00000001.json` as `native E97 generation identity is
invalid`. Both trainer-7 roles exhausted their single restart; other trainers
then encountered native buffer/operation release route failures (`-12`) as
the allocation step was torn down. Consequently the clean phase stopped in
generation 1 and the serial controller did not submit fault/rejoin,
invalid-result, checkpoint-publication-failure, or fresh-restart phases.

Conformance for this replacement is checked against *Resilient DiLoCo Compute
Pool* v1 R01-R16 and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.
Exact-source identity, G2 correctness/integrity, two-node admission,
point-to-point CXI selection, and no-overlap/no-scale guards currently pass.
Exact identity, G2 integrity, two-node admission, one atomic safe-boundary
commit, checkpoint publication, and bounded latest-only queue behavior have
live evidence. The replacement does not satisfy R07/R11/R12/R14/R16 and
NDP10/NDP13/NDP15/NDP16/NDP17 acceptance as a whole: five generations,
steady-state overlap/cadence, loss/rejoin, invalid-result rejection,
failed-publication retention, and fresh restart were never reached. No scale
ladder job is admitted from this result.

## Terminal harvest of job 5037971

The existing job was monitored without a duplicate submission.  Slurm accounting
records that `5037971` left the queue, started at `2026-07-20T14:15:45` on
exactly two nodes (`frontier[02939-02940]`), and terminated `FAILED 1:0` at
`14:24:26` after `00:08:41`.  The Python step ran for `00:07:53`; therefore the
job did execute and does not satisfy the narrowly authorized condition for a
replacement submission.

The immutable execution root is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-artifacts-20260720T181000Z/`.
Its retained pool-control record proves that generation 0 froze both READY node
incarnations, met `Q_min=2`, and reached `commit_ready` with 5,245,440 accepted
tokens.  Both node result records agree on result root
`63b3fef285173902e0ee4b54f4e7cab61fac8860c7427d9ef2c3750b9c641477`.
Trainer timestamps include generation-0 K40 work through step 39 on
`node-0-trainer-0` (final timestamp `1784571845.7019308`, loss
`2.5740623474121094`, 8,196 tokens for that step).  This is useful live K40
evidence, but generation 0 never became an atomic committed generation and no
generation 1 training exists.  It consequently cannot establish g/g+1 overlap,
steady-state foreground idle or cadence, five atomic generations, or the later
fault/rejection/restart phases.

The decisive failure sequence is retained in the per-role stderr and
supervision event stream:

- trainer leaders raised `NameError: name 'generation_started' is not defined`
  while calculating the post-generation deadline;
- another trainer failed `ndp_buffer_seal_v1` with route failure `-12`, while
  followers later expired waiting for checkpoint-leader apply release;
- both managers then failed `ndp_control_v1(FREEZE)` with invalid lifecycle
  state `-3`; supervisor restart attempts were exhausted.

No later phase and no 4-node-or-larger job was submitted.  The existing G2 job
`5037939` remains the reused exact-source correctness/integrity and throughput
telemetry gate because submission source identity did not change.

Key immutable SHA-256 identities are:

- acceptance manifest: `adbe2c8108bdfe4a4b538cf0a562e06dfea1dcf97535e5ef5b79f0a380ea14a7`
- runtime identity: `5c0fa27593fa1b173a57512eaee13e8dcb8bfadfb1c1b3783fae4457e02c6805`
- native launch attestation: `a50087225c6b85be4b6d7304582605385ae63de448e3438e9d97eac76d3d4153`
- generation-0 pool-control record: `832467298478d6669d0722268ab6e5cbc454925f029d0b46380e6458236d1440`

Conformance was checked against *Resilient DiLoCo Compute Pool* v1 R01-R16
and *Native resilient DiLoCo data plane* v1 NDP01-NDP17.  The live run confirms
exactly-two-node capacity, leased READY membership, fenced generation identity,
bounded execution, exact-source native launch attestation, contribution freeze,
and fail-closed behavior without an all-rank wait.  It does not satisfy R07,
R11, R12, R14, R16 or the corresponding NDP10/NDP13/NDP15/NDP16/NDP17 live
acceptance evidence because no atomic commit, safe apply, later generation,
failure phase, or fresh restart completed.

## Seventh concrete attempt: refreshed G2 passed and real K40 queued

Authoritative source `87365c5f846a950d7aaa01fec441982c79fc5e50` is merged
and pushed. Its canonical native build passed CTest 10/10. Exact-source G2 job
`5037939` completed `0:0` in 2:51 on exactly `frontier08123` and
`frontier08127`. The retained gate is
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-retry-stage-20260720T180500Z/g2-artifacts/5037939/full-layout-gate.json`.

G2 exact-reference correctness/integrity passed with zero CQ/route errors and
zero retries. It reported 44,322,599,424 useful TX and RX bytes, 44,323,138,304
wire TX and RX bytes, and timed intervals 23.353392370, 22.742834860, and
23.261179257 seconds. Median native logical throughput was 946,946,057.23 B/s
versus retained Python 222,582,457.59 B/s (4.254x). The ratio is telemetry, not
K40 admission; correctness/integrity admitted the real phase.

Initial K40 job `5037915` failed closed before model load because the serial
renderer used a caller-relative batch path and omitted the G2/runtime exports.
The controller now uses an absolute authoritative launcher and `--chdir`, and
binds source, seed, data, tokenizer, bundle, G2, and fence identities. The
focused canonical suite passed 62/62; fix `d3ee23f2` was pushed and merged into
the authoritative commit above.

The corrected controller submitted clean-overlap job `5037971` for exactly two
nodes and five K40 generations. It remains `PENDING (Priority)` at handoff, so
no K40 overlap/cadence claim is made. Controller state and the immutable source
inventory/acceptance manifest are under
`/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-artifacts-20260720T181000Z/`.
No overlapping, later-phase, or 4-node-or-larger job was submitted.

Conformance: *Resilient DiLoCo Compute Pool* v1 R01-R16 and *Native resilient
DiLoCo data plane* v1 NDP01-NDP17. G2 establishes exact fenced identities,
bounded point-to-point CXI, exact weighted math, integrity/rejection, bounded
release, and no MPI/all-rank or Python-dense fallback. The queued K40 phase
retains `Q_min=2` and `T_min=3,934,080`; live timing, loss/rejoin,
invalid-result rejection, and checkpoint failure/restart remain pending.

## Sixth concrete attempt: repaired linkage, controller fix, and live G2 performance blocker

Authoritative `main` first resolved to `692d0292`.  From a clean clone, the
canonical Frontier activation built the exact native bundle and passed CTest
10/10.  Exact-source G2 job `5036978` completed on exactly two nodes in 2:57
and produced a passing full-layout CXI gate.  The serial acceptance controller
then failed closed before `sbatch` because its plan omitted the top-level
`authoritative_stage` consumed by `advance()`.  A regression fix was committed
as `bb5cd54f`, cherry-picked and pushed to authoritative main as `09eac436`;
the focused launcher/runtime suite passed (26 collected, plus 8 passed/1
platform skip in the native gate selector).

Because that source change invalidated the earlier gate identity, the exact
`09eac436` bundle was rebuilt (CTest 10/10) and G2 job `5037046` was concretely
submitted on exactly two nodes.  Both native node payloads passed, but the
authoritative validator rejected publication because observed clean throughput
did not reach 4x the retained Python gate.  Slurm records `FAILED 1:0`, 3:00,
two nodes (`frontier[07388,07408]`).  Therefore no real K40 phase was admitted:
five generations, overlap/idle timing, fault/rejoin, invalid-result rejection,
and checkpoint restart remain unproven.  No 4-node-or-larger job was submitted.

Conformance was checked against Compute Pool v1 R01-R16 and native data plane
v1 NDP01-NDP17.  The source/bundle/G2 fences and exactly-two-node/no-overlap
guards behaved fail closed; NDP17 blocks scale-out until the exact-source G2
performance gate passes.

## Fifth concrete attempt: authoritative rebuild exposes G2 runtime-linkage blocker

On reassignment after the serial launcher fix merged, authoritative `main` and
`origin/main` resolved to
`d25c78b4b414e1a41c5ec2764f6307f93f64b316`.  Because the shared main checkout
contains retained untracked run evidence, the submission was prepared from a
fresh local `main` clone at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-20260720T142140Z/source
```

The clone was clean, on branch `main`, and exactly matched `origin/main`.  The
canonical `scripts/frontier/activate_emender_frontier.sh` activation selected
the approved Python 3.12 environment and Frontier GNU 14.2/Cray MPICH 9.1
module stack.  The reviewed exact acceptance renderer was invoked with
`--submit`, `--state`, and `--native-stage-root`.  Its first canonical native
build initially failed closed because inherited `FI_PROVIDER=cxi` forced five
local socket CTests onto CXI.  The build was repeated without fabric override
variables; all 10 native CTests then passed and the immutable bundle was
installed at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-20260720T142140Z/native-stage/d25c78b4b414e1a41c5ec2764f6307f93f64b316/install/native-artifacts.json
```

Production attestation correctly rejected the retained job-5033120 G2 gate:

```text
source_commit: 85cf5a09... != d25c78b4...
bundle_sha256: f2ac884e... != 7b4c7769...
```

No real-model job was admitted with mismatched evidence.  To concretely obtain
the required exact-source precursor, the canonical G2 launcher was then run
against the rebuilt bundle:

```text
NDP_BUILD_MANIFEST=.../d25c78b4.../install/native-artifacts.json \
NDP_ARTIFACT_ROOT=.../g2-artifacts \
bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

Slurm accepted job `5036108` on exactly two nodes.  It ran on
`frontier[09074-09075]` and terminated `FAILED 1:0` after 80 seconds, before a
G2 generation or full-layout gate was produced.  Both node tasks reported:

```text
ndp_frontier_2n_gate: error while loading shared libraries:
libamdhip64.so.7: cannot open shared object file: No such file or directory
```

The scheduler record and `g2-artifacts/5036108/submission.json` are retained
under the execution root above.  The exact K40 acceptance remains fail-closed:
without an exact-source G2 gate, five live generations and the fault/restart
phases cannot be admitted.  A follow-up must make the canonical native G2
batch runtime expose the approved ROCm 7.1 library path (or install a complete
self-contained runtime), rerun the two-node G2 clean gate, then resume the
serial acceptance controller.  No 4-node-or-larger job was submitted.

Conformance checked against *Resilient DiLoCo Compute Pool* version 1,
requirements R01-R16, and native data plane requirements NDP01-NDP17.  This
attempt demonstrates fail-closed source/bundle/G2 identity enforcement
(R13/R16, NDP13/NDP17), bounded exactly-two-node scheduler mutation, and no
acceptance claim from a partial or unattested run.  Runtime-only overlap,
timing, rejection, loss/rejoin, and atomic restart criteria remain unproven.

## Fourth concrete attempt: reviewed exact acceptance launcher

After the prerequisite launcher merged, authoritative `main` and
`origin/main` both resolved to
`08a26a0f0bc71c7e480010ba1ecb4e024ce37e36`.  The per-user Slurm queue was
empty at `2026-07-20T13:12:07Z`.  From the clean authoritative checkout, after
canonical Frontier activation, the new exact submit path was invoked with:

```text
$EMENDER_PYTHON scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo /lustre/orion/bif148/scratch/erikgarrison/emender \
  --native-build-manifest /lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json \
  --full-layout-gate /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/native-dataplane/5033120/full-layout-gate.json \
  --run-root /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/pipelined-native-2-20260720T131330Z/phases \
  --output /lustre/orion/bif148/scratch/erikgarrison/emender/reports/frontier/pipelined-native-2-20260720T131330Z/acceptance.json \
  --submit
```

All mandatory immutable E97 seed, data, tokenizer, approved training-config,
native-CXI, and fenced-run inputs were exported before this command.  Slurm
accepted the first phase as job `5035685`:

```text
clean-overlap=5035685
5035685|resilient-e97-true-2n|RUNNING|2|02:00:00|frontier[03756-03757]
```

Thus this attempt concretely submitted exactly two nodes.  It submitted no
4-node-or-larger job.  Job `5035685` reached the approved Python/Torch/ROCm
runtime identity check, then failed closed after 2:48 with exit `1:0`, before
model admission or any K40 generation:

```text
ValueError: native build does not match the launched source commit
```

This is the expected enforcement of the source/bundle mismatch noted below;
no runtime-only acceptance claim can be made from this job.

The renderer immediately attempted to enqueue `fault-rejoin` behind the clean
job.  Slurm rejected that second submission before assigning a job ID:

```text
sbatch: error: QOSMaxSubmitJobPerUserLimit
sbatch: error: Batch job submission failed: Job violates accounting/QOS policy
```

The renderer consequently exited 64 and did not attempt the remaining three
phases.  The immutable render and submission transcript are retained under
`reports/frontier/pipelined-native-2-20260720T131330Z/` in the authoritative
checkout.  The clean job must be allowed to finish and harvested; subsequent
phases must then be submitted serially as the QoS slot becomes free.  The
current all-at-once dependency-chain implementation cannot establish the full
acceptance sequence on a QoS configured with a one-job-per-user submit limit.

Two additional lineage issues were exposed by the attempt: the installed
native manifest records source `176ae0bc...` rather than the submitted source
`08a26a0f...`, which caused job `5035685` to fail attestation, and the
renderer's phase-specific
`run_dir`/`restart_from` values are recorded in JSON but are not exported as
`RUN_DIR`/`RESILIENT_E97_RESUME_HANDOFF` to each `sbatch` command.  Runtime
attestation and restart evidence must not be claimed unless the job artifacts
prove those bindings.  A retry requires a canonical native rebuild from the
authoritative commit and a serial phase submit/harvest path compatible with
the one-job debug-QoS limit.

Retry note (agent-1347): at `2026-07-20T12:50:12Z`, a third concrete attempt
fetched `origin/main`, confirmed that the authoritative checkout's `HEAD`,
`main`, and `origin/main` still all resolved to
`176ae0bc11db5bf1cad51008d6891d209867004c`, and activated the canonical
Frontier environment.  The activation selected the current canonical module
stack (including `PrgEnv-gnu/8.7.0`, `cray-mpich/9.1.0`, and
`gcc-native/14.2`).  The exact approved clean two-node command was then invoked
again with the retained native manifest and `$EMENDER_PYTHON`; it exited 69
with `refusing to overlap another user allocation` before `sbatch`.

Immediately after that attempt, job `5035539` remained RUNNING on 32 nodes at
12:01 elapsed of its 30-minute bound.  The authoritative source had not gained
a reviewed real-K40 acceptance submitter: the clean G2 launcher still fixes
three synthetic generations, while `resilient_e97_true_2n.sbatch` still admits
only 20/30-minute gates and therefore cannot safely cover five independently
bounded 420-second K40 windows plus the required control/checkpoint phases.
No job was submitted by this retry, and the live acceptance criteria remain
unclaimed.

Retry note (agent-1346): the retry on the same date re-read both normative
design authorities, verified that the authoritative checkout still resolved
`HEAD`, `main`, and `origin/main` to `176ae0bc11db5bf1cad51008d6891d209867004c`,
and performed a fresh scheduler preflight.  Job `5035539` was still RUNNING on
32 nodes (`validate-native-pool-32n-late-ready-fixed`), so the no-overlap
condition had not changed and no `sbatch` call was permissible.
The exact canonical activation and approved clean-gate command were invoked
again; the launcher exited 69 with `refusing to overlap another user
allocation`, proving that the retry also stopped before `sbatch`.

## Concrete attempt

The canonical Frontier environment was activated with:

```text
source scripts/frontier/activate_emender_frontier.sh
```

The submission source was the authoritative checkout
`/lustre/orion/bif148/scratch/erikgarrison/emender` on branch `main`.
At the attempt, `HEAD`, `main`, and `origin/main` all resolved to:

```text
176ae0bc11db5bf1cad51008d6891d209867004c
```

The exact source native bundle was rebuilt before submission:

```text
PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
```

All 10 native CTests passed. The retained build manifest was:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json
sha256 cad13e0ebb93761f2bd9e8217ded24cb1592d9f9e59367a93dda6ca9c0a51fda
```

At `2026-07-20T12:41:36Z`, the exact approved submission command was
invoked from that authoritative `main` checkout:

```text
NDP_BUILD_MANIFEST=/lustre/orion/bif148/scratch/erikgarrison/emender/build/native-resilient-dataplane/native-artifacts.json \
NDP_PYTHON_BIN="$EMENDER_PYTHON" \
bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

It exited 69 with the expected fail-closed diagnostic:

```text
refusing to overlap another user allocation
```

The conflicting allocation observed immediately after the attempt was:

```text
5035539|validate-native-pool-32n-late-ready-fixed|RUNNING|32|3:24|30:00
```

No `sbatch` invocation was reached and no job of any size was submitted by
this task. In particular, no 4-node-or-larger job was submitted.

## Acceptance status

Because the approved launcher refused submission, no live two-node artifact
was created and none of the runtime-only claims are made. Five atomic K40
generations, overlap/cadence timing, foreground-idle percentage, bounded
latest-only behavior under loss/rejoin, rejection evidence, and atomic
checkpoint restart remain unverified by this attempt.

There is also an acceptance mismatch that must be resolved before a retry:
`submit_native_dataplane_2n_gate.sh clean` fixes `NDP_GENERATIONS=3` and its
batch payload requires one warm-up plus three timed synthetic native data-plane
generations. The present task requires at least five atomic **K40 trainer**
generations plus checkpoint/failure behavior. The real K40 split-role path is
`scripts/frontier/resilient_e97_true_2n.sbatch`; a retry must use a reviewed,
fail-closed two-node submit artifact for that path rather than treating three
synthetic data-plane generations as the requested live training proof.

The merged real-model batch artifact has a second hard mismatch.  Its
supervisor bounds each K40 stage at 420 seconds, while the batch script admits
only 20- or 30-minute non-startup gates.  Five worst-case K40 windows alone are
35 minutes, before foreground handoff/apply, integrity, redistribution, and
checkpoint publication.  Submitting that payload with five generations would
therefore not be a fail-closed way to establish the requested acceptance
artifact.  A reviewed launcher must provide a bounded walltime sufficient for
five K40 generations and must explicitly orchestrate and validate all requested
fault/restart phases; those controls are not present in the named submitter.

## Architecture conformance

Authority checked: *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17),
and *Native resilient DiLoCo data plane v1*. Applicable gap-matrix requirement
IDs are R01-R16 and NDP01-NDP17.

The attempted command conformed to the admission safety portion of R01/R14 and
NDP17: authoritative identity was pinned, the current native bundle was built
and tested under the canonical environment, the launcher was exactly two-node
and CXI-only, and its overlap guard failed closed before scheduler mutation.
The remaining conformance checklist items require a live allocation and are not
claimed here.
