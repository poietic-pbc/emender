# Native G2 throughput diagnosis

Date: 2026-07-20  
Task: `diagnose-and-fix`  
Slurm submissions by this task: **none**

## Result

The 5037046 observation is ordinary transport-rate variation, not a
denominator, repetition, payload, controller-serialization, affinity, module,
or integrity mismatch. It missed the former synthetic comparison by 2.95%,
while exact useful/wire work and all correctness checks matched 5036978. G2 is
now correctly treated as an exact-source correctness/integrity gate whose
throughput comparison remains explicit telemetry. It no longer prevents the
real K40 run. Production performance admission is the live behavior that
matters: background generation-g work must overlap generation-(g+1) K40,
foreground control-plane idle must be below 10%, and steady cadence must be at
most 1.25x raw K40 compute when the background work fits that window.

## Byte and time accounting

One timed generation transfers two 5,506,770,496-byte contributions and
redistributes one 5,506,770,496-byte result to each node.  Thus the useful
numerator is

`2 * (2 * 5,506,770,496) = 22,027,081,984 bytes`.

The physical useful TX observed across both endpoints is 11,013,540,992 bytes
per generation (one physical contribution plus one physical redistribution);
physical wire TX is 11,013,674,752 bytes, exactly 133,760 bytes of bounded
protocol framing above useful TX.  RX is symmetric.  Counting both logical
stages in the numerator is therefore not double-counting a physical stage:
the retained Python baseline uses the identical 22,027,081,984-byte logical
numerator and covers both contribution and redistribution.

The retained Python interval is 98.961446568 s, or 222,582,457.59 logical B/s.
The unchanged 4x floor is 890,329,830.36 B/s and 24.740361642 s.

| Evidence | Timed samples (s) | Median (s) | Logical B/s | Python speedup | Result |
|---|---:|---:|---:|---:|---|
| 5036978 exact source | 21.898900519, 23.591797875, 23.791591642 | 23.591797875 | 933,675,428.24 | 4.1947x | pass |
| 5037046 authoritative 09eac436 | 23.164536252, 26.393715247, 25.491979176 | 25.491979176 | 864,078,954.28 | 3.8821x | fail |
| 5033120 accepted G2 | 19.983974086, 19.476943726, 19.491669693 | 19.491669693 | 1,130,076,711.28 | 5.0771x | pass |
| 5033380 accepted G2 | retained median 22.690315566 | 22.690315566 | 970,770,191.36 | 4.3614x | pass |
| 5034592 accepted G2 | retained median 24.656688684 | 24.656688684 | 893,351,182.16 | 4.0136x | pass |

For both exact jobs, each timed endpoint reports useful TX/RX of
5,506,770,496 bytes and wire TX/RX of 5,506,837,376 bytes (the final shard can
swap the 320-byte direction rounding between peers).  Over the retained warmup
plus three timed generations, 5036978 reports 44,322,599,424 useful TX and
44,323,138,304 wire TX; 5037046 has the same per-generation counters and zero
timed retries.  Both jobs used one warmup plus three measured generations,
83 shards, 64 MiB maximum payload, eight local lanes, `cxi/cxi0`,
`FI_MR_CACHE_MONITOR=kdreg2`, `FI_CXI_ATS=0`, 32 CPUs/task, and
`--cpu-bind=cores`.

## Timed-boundary and overhead audit

The steady-clock interval begins immediately before `ContributionStage` and
ends only after contribution reduction, owner finalization, bounded result
redistribution, generation-goodbye exchange, and transport release to zero.
Per-frame checksum generation/validation and finite-value validation remain in
that interval.  Startup, endpoint exchange, analytical result-root/payload
attestation, JSON output, and the retained one-time local lane reduction are
outside every measured generation for both implementations; changing the
window does not move a boundary.  The observed local reductions were
1.948/1.975 s in 5036978 and 1.965/1.974 s in 5037046, outside the timed path.
The Python controller carries only the two endpoint records before generation
timing and never carries a tensor, so measured controller serialization is
zero dense bytes.  The jobs' identical loader preflights, bindings, provider
facts, byte counters, result digest, zero timed rejection counts, and nearly
identical local-reduction times exclude the proposed environment/affinity,
payload, copy, and controller explanations.

## Deterministic reproducer and policy validation

The deterministic regression test constructs a correct exact G2 result below
the old 4x comparison and proves validation records
`legacy_4x_telemetry_target_met=false` with admission policy
`correctness_only_then_live_k40_performance`, rather than terminating before
K40. Existing corrupt/checksum/nonfinite/replay/source/runtime/protocol tests
remain hard failures.

A new live telemetry validator requires exactly 40 optimizer-step start/end
timestamps per generation and at least two steady generations for every
trainer. It reconstructs raw K40 intervals and generation cadence, translates
manager background stage intervals onto the trainer clock, and fails unless
generation-g native reduction/redistribution/apply/commit overlaps
generation-(g+1) compute. It independently fails at idle >=10% and, whenever
the longest background stage fits raw K40, cadence >1.25x. Deterministic tests
cover the passing path and missing-overlap, idle, cadence, and missing-step
failures. The clean-overlap batch phase runs this validator after the live
supervisor and retains `pipelined-performance.json`; later fault phases remain
correctness/recovery gates.

The exact Frontier performance claim still requires the downstream live
runner; this task intentionally did not submit a job. Local CTest passed 10/10
and the focused Python suites cover launcher, accounting, retained evidence,
live performance policy, and fail-closed behavior.

## Conformance

Checked against *Resilient DiLoCo Compute Pool* version 1 and its required
checklist: R10 (bounded hot-path memory and release), R13 (observability), R14
(sequential gate), R16 (source/evidence identity), and NDP02, NDP03, NDP06,
NDP10, NDP13, NDP16, NDP17.  READY membership, two persistent point-to-point
endpoints, fenced identities, deterministic weighted math, checksums,
idempotence/rejection behavior, no Lustre dense path, no central broker, no
MPI/all-rank invariant, Q_min=2, and T_min=3,934,080 are unchanged.
