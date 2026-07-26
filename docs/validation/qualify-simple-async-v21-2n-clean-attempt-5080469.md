# Async v2.1 exact-two-node clean qualification: non-passing attempt 5080469

Verdict: `passed=false`. Job `5080469` was the sole changed-payload clean
attempt for source `2ad87c9a8b3d70f9316ad51cb2100c751242f0e1`. Its payload
digest `ba5133b81654fd7215508e29ff63ac276efc2c089dd74c506ff4c0800702f9ed`
is retired and must never be resubmitted. This attempt authorizes neither a
fault run nor scale work.

## Terminal scheduler and exact launch evidence

Queued/running monitoring used:

```text
squeue -h -j 5080469 -o '%i|%T|%P|%q'
scontrol show job -dd 5080469
```

Both commands retained exactly two allocated nodes,
`frontier[03074,03144]`, `Partition=batch`, and `QOS=debug`, with
`Restarts=0`. Terminal accounting used:

```text
sacct -n -X -j 5080469 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

and returned:

```text
5080469|FAILED|1:0|2|batch|debug|2026-07-26T09:16:24|2026-07-26T09:29:17|00:12:53
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
`fa41a2824c044624dddd282d5956abd3eeed8cd6465e34b88f7b13da426ff982`.
It binds exact source `2ad87c9a...f0e1`, source digest
`f271caf8...31b4`, native build manifest `81936e61...9146`, native bundle
`f19e10be...2441`, policy `fa9def95...7d98`, launcher
`70b96385...3fb7`, data `91321b2b...7962`, tokenizer
`94b5ca7d...2069`, and passed G2 job `5080458` with gate SHA-256
`a5a5be8c...624b`.

G2 terminal accounting was
`5080458|COMPLETED|0:0|2|batch|debug|2026-07-26T09:11:20|2026-07-26T09:14:12|00:02:52`.
Its median full-layout sample was `23.573505062` seconds, maximum
`23.888658168` seconds, and speedup `4.197994583697738x`, with zero route
errors, CQ errors, retries, Python dense socket bytes, disk replay bytes, or
all-rank barriers.

## Exact seed

The submit-side attestation and both compute nodes agreed on immutable step
`2300930`, accepted tokens `150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Both nodes independently verified
`/tmp/emender-e97-seed-5080469/checkpoint-step-2300930.pt` offline before
model load and reported `network_fetches=0`.

## Exact terminal cause and preliminary interleave metrics

The manager progress-stage repair from job `5080289` was proven live. Both
managers advanced from `owner_transport` to `checkpoint_commit` about
`128.16`/`131.10` seconds later, before the 180-second owner-transport
deadline, then advanced to `peer_apply`. There was no manager restart, OOM,
native route/CQ error, seed disagreement, or scheduler mismatch.

The first immutable checkpoint was retained as
`generation-00000001-fence-05080469.pt`, size `2753437091`, SHA-256
`933ba061cb53210069f2191b8f0e1013add0dfca4dc8f05e2a091738a1424086`.
Each manager nevertheless received only the rank-zero
`native-applied-00000000-00.json` receipt before its peer-apply deadline and
failed closed waiting for rank one. Neither node emitted a complete node marker.

The immediate cause was the old node-local one-reader chain. Rank zero, one,
and two on the two nodes took respectively about `42.83`/`43.18`,
`49.08`/`48.16`, and `43.06`/`44.62` seconds to materialize the shared
5,506,770,496-byte result while the persistent training lane continued.
Checkpoint writes took `1.33`--`2.18` seconds and hashes took
`2.80`--`2.95` seconds. The manager began its 60-second all-eight transaction
when rank zero crossed the safe boundary, although ranks one and two were
still serially preparing and ranks three through seven could not yet begin.
The deadline was therefore structurally impossible, not a slow or unhealthy
interval selected out of the evidence.

All 16 trainers continued real optimizer work until termination. The retained
telemetry contains 82 complete K40 windows: 14 trainers completed five and
two trainers completed six. Across those inclusive early windows, preliminary
raw K40 median/max were `67.32339283049805`/`69.79031882889103` seconds,
K-boundary cadence median/max were
`67.2150263604708`/`69.86184140492696` seconds
(`0.9983903593465634x`), and honest foreground idle was
`0.00009691070591496506`; median/max inter-window gaps were
`0.0015700520016252995`/`0.0715225760359317` seconds. These are not pass
metrics: no trainer retained the required 12 windows and the run retained only
one atomic commit.

The failing-first regression is
`test_native_all_eight_apply_releases_only_after_background_preparation`.
The smallest repair separates concurrent, background candidate preparation
from the finite foreground transaction. Node zero still gives rank zero
checkpoint priority, then its other seven read-only result views overlap;
node one prepares all eight concurrently. Every trainer checkpoint-hashes and
reload-verifies the same fenced latest result, writes an authenticated ready
receipt, and waits. Only after all eight receipts match does the manager emit
one release and start the reviewed 60-second x/z/safe-boundary apply clock.
The repair retains the 420-second freeze-to-verified-latest bound and removes
the historical 130--150-second rank-serialized generation span that would
make ten commits impossible inside the 45-minute qualification deadline.

## Authority mapping

The attempt was checked against the compute-pool conformance checklist,
R01–R16, NDP01–NDP17, ADR-002 V21S01–V21S17, and ISP01–ISP07. Exact source,
native bundle, queue identity, seed, data/tokenizer, policy/schema, capacity
declarations, native owner transport, bounded snapshot ownership, and
fail-closed supervision were retained. R14, NDP13/NDP15/NDP16,
V21S03/V21S11/V21S13/V21S15, and ISP05 remain non-passing because only one
immutable commit and five to six windows per trainer were reached and the
all-16 atomic apply was incomplete. No failure or backpressured interval was
excluded from the preliminary metrics.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080469.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080469.json).
