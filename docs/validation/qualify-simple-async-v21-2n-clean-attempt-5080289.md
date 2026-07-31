# Async v2.1 exact-two-node clean qualification: non-passing attempt 5080289

Verdict: `passed=false`. Job `5080289` was the sole changed-payload clean
attempt for source `e232338763afbbda048b9cba5f838ecceea728b6`. Its payload
digest `dc84ca31cef40fc20de6744aa2b13f2d56b55231187ab395fcbe3eeba5b81282`
is retired and must never be resubmitted. This attempt authorizes neither a
fault run nor scale work.

## Terminal scheduler and exact launch evidence

Queued/running monitoring used:

```text
squeue -h -j 5080289 -o '%i|%T|%P|%q'
scontrol show job -dd 5080289
```

Both commands retained exactly two allocated nodes,
`frontier[07810-07811]`, `Partition=batch`, and `QOS=debug`. Terminal
accounting used:

```text
sacct -n -X -j 5080289 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

and returned:

```text
5080289|FAILED|1:0|2|batch|debug|2026-07-26T08:39:08|2026-07-26T08:50:36|00:11:28
```

The serial controller argv was:

```text
"$EMENDER_PYTHON" "$SNAPSHOT/scripts/frontier/run_async_v21_qualification.py" \
  --gate clean --nodes 2 --repo "$SNAPSHOT" \
  --seed-config "$SNAPSHOT/configs/frontier/e97_async_256.yaml" \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$RUN_ROOT" --state "$STATE_JSON" \
  --output "$MANIFEST_JSON" --submit
```

The acceptance-manifest SHA-256 is
`06a31b7ac821c57a69e03c6bd2455d47bd624e4d6e74ef29eaac26cd2e6d5387`.
It binds exact source `e2323387...28b6`, source digest
`e69a23da...5b85`, native build manifest `0db1bc6d...d423`, native bundle
`f19e10be...2441`, policy `fa9def95...7d98`, launcher
`70b96385...3fb7`, data `91321b2b...7962`, tokenizer
`94b5ca7d...2069`, and passed G2 job `5080277` with gate SHA-256
`7fe05b1a...303f`.

G2 terminal accounting was
`5080277|COMPLETED|0:0|2|batch|debug|2026-07-26T08:33:20|2026-07-26T08:36:16|00:02:56`.
Its median full-layout sample was `23.234830035` seconds, maximum
`24.879618527` seconds, and speedup `4.259185301548396x`, with zero route
errors, Python dense socket bytes, disk replay bytes, or all-rank barriers.

## Exact seed

The submit-side attestation and both compute nodes agreed on immutable step
`2300930`, accepted tokens `150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Both nodes independently verified
`/tmp/emender-e97-seed-5080289/checkpoint-step-2300930.pt` offline before
model load and reported `network_fetches=0`.

## Exact terminal cause and preliminary metrics

There was no trainer/runtime traceback, OOM, native route/CQ error, or manager
restart. All 16 trainers advanced four K40 windows and the native data plane
completed the first full generation. Both manager cohorts were then terminated
at Unix time `1785070233.36` with `reason=progress_deadline`, followed by
`restart_exhausted`.

Both retained manager progress records were still:

```text
generation=0 stage=owner_transport progress_time=1785070052.38
```

The native telemetry proves the transport itself was healthy and complete well
before that deadline. Node-local reduction took about `19.93` seconds; route
readiness took `0.003`/`0.014` seconds; owner contribution took
`39.45`/`39.57` seconds; owner redistribution took `24.914`/`24.919`
seconds; and complete native redistribution took `73.97`/`74.03` seconds
from its later phase start. The exact result returned and validated about
`129.8` seconds after the manager entered `owner_transport`. All stage records
reported `within_slo=true`, with zero route/CQ errors.

The code then correctly created native result metadata, waited for the trainer
checkpoint proposal, published and verified the immutable checkpoint, and
began collecting trainer apply receipts. It did not, however, advance the
manager progress stage after the completed transport. The supervisor therefore
charged checkpoint publication and peer apply to the already spent
`owner_transport` interval and killed both cohorts after about `181` seconds.
The first immutable checkpoint was nevertheless retained as
`generation-00000001-fence-05080289.pt` with size `2753437091` and SHA-256
`75f9cccd597027f8126b0bf74498fcd2096a5fdae43a4364848ab1e92af5ae78`.
The attempt did not retain a complete all-16 atomic-apply cohort.

The previous pinned-copy repair was proven live. All 16 endpoint snapshot
enqueue/ownership paths completed in at most `0.0437426628` seconds, the
maximum foreground ownership pause was `0.0631920379` seconds, and asynchronous
copy completion took at most `0.1473063356` seconds.

Across the 48 completed trainer windows, the preliminary raw K40 median was
`67.75519612757489` seconds and the preliminary K-boundary cadence median was
`67.75631873868406` seconds (`1.00001657x`). Maximum/p99 raw and cadence were
`70.23644956108183` and `70.23743151500821` seconds respectively. Inter-window
gaps had median `0.0008892575` seconds and maximum `0.0013092961` seconds,
giving preliminary honest foreground idle `0.000013134092743166162`. These
numbers describe early progress only: the run retained one commit and four
windows per trainer, not the required ten commits and 12 windows, so they do
not satisfy the clean gate.

The failing-first regression is
`test_native_manager_advances_progress_after_owner_transport`. The smallest
repair emits `checkpoint_commit` after the native result root has returned and
validated, then emits `peer_apply` after immutable commit verification and
before native trainer apply receipts. Both stages already have the reviewed
`EXCHANGE_COMMIT_HARD_S=180`; the repair does not enlarge any deadline.

## Authority mapping

The attempt was checked against the compute-pool conformance checklist,
R01–R16, NDP01–NDP17, ADR-002 V21S01–V21S17, and ISP01–ISP07. Exact source,
native bundle, queue identity, seed, data/tokenizer, policy/schema, capacity
declarations, native owner transport, bounded snapshot ownership, and
fail-closed supervision were retained. R14, NDP13/NDP15/NDP16,
V21S03/V21S11/V21S13/V21S15, and ISP05 remain non-passing because only one
immutable commit and four windows per trainer were reached and the all-16
atomic apply was interrupted. No failure interval was excluded from the
preliminary metrics.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080289.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080289.json).
