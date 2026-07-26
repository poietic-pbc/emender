# Async v2.1 exact-two-node clean qualification: non-passing attempt 5080178

Verdict: `passed=false`. Job `5080178` was the sole changed-payload clean
attempt for source `a3f819e3c4e53c2746e20e77c4e84bbc25a42a26`. Its payload
digest `14b916276f01cbb3198a008a9a9f06219649c1f3b11bb87ae06eeba8d42074d0`
is retired and must never be resubmitted. This attempt authorizes neither a
fault run nor scale work.

## Terminal scheduler and exact launch evidence

Queued/running monitoring used:

```text
squeue -h -j 5080178 -o '%i|%T|%P|%q'
scontrol show job -dd 5080178
```

Both commands retained exactly two allocated nodes,
`frontier[06659,06662]`, `Partition=batch`, and `QOS=debug`. Terminal
accounting used:

```text
sacct -n -X -j 5080178 \
  --format=JobIDRaw,State,ExitCode,NNodes,Partition,QOS,Start,End,Elapsed -P
```

and returned:

```text
5080178|FAILED|1:0|2|batch|debug|2026-07-26T08:00:17|2026-07-26T08:08:20|00:08:03
```

The serial controller argv was:

```text
"$EMENDER_PYTHON" scripts/frontier/run_async_v21_qualification.py \
  --gate clean --nodes 2 --repo "$SNAPSHOT" \
  --seed-config configs/frontier/e97_async_256.yaml \
  --native-build-manifest "$BUILD_MANIFEST" \
  --full-layout-gate "$G2_GATE" \
  --run-root "$RUN_ROOT" --state "$STATE_JSON" \
  --output "$MANIFEST_JSON" --submit
```

The acceptance-manifest SHA-256 is
`af5dbc1bc1d8341b019b284e3228235505fb2887d0877aaf24c8f257cfcd0f2c`.
It binds exact source `a3f819e3...`, native build manifest
`5430cc07...6787`, native bundle `f19e10be...2441`, policy
`fa9def95...7d98`, launcher `70b96385...3fb7`, data
`91321b2b...962`, tokenizer `94b5ca7d...069`, and passed G2 job `5080168`
with gate SHA-256 `542e32e8...976`.

G2 terminal accounting was
`5080168|COMPLETED|0:0|2|batch|debug|2026-07-26T07:55:55|2026-07-26T07:58:45|00:02:50`.
Its median full-layout sample was `23.379851475` seconds, maximum
`25.347806946` seconds, and speedup `4.232766263501133x`, with zero route
errors, Python dense socket bytes, disk replay bytes, or all-rank barriers.

## Exact seed

The submit-side attestation and both compute nodes agreed on immutable step
`2300930`, accepted tokens `150793748480`, size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Both nodes independently verified
`/tmp/emender-e97-seed-5080178/checkpoint-step-2300930.pt` offline before
model load and reported `network_fetches=0`.

## Exact terminal cause and preliminary metrics

Only `node-1-trainer-3` raised:

```text
TimeoutError: persistent async lane admission deadline expired
```

The exception came from the first deadline check in
`PersistentAsyncTrainingLane.start`. Therefore the preceding synchronous
`PersistentRealWorkerSession.snapshot()` had already consumed the absolute
one-second capture-plus-ownership allowance; no lane startup, telemetry I/O,
publication, aggregation, checkpoint, or result wait was involved.

The other 15 trainers remained alive and advanced K windows. Their successful
full-model snapshot captures ranged from `0.40914149675518274` to
`0.8983200970105827` seconds. Node-0 ranks 0 and 6 required
`0.8983200970105827` and `0.8760826420038939` seconds respectively, leaving
no reliable ownership headroom. All eight trainers on a node copied one
full-model GPU state into pageable CPU slots at approximately the same
boundary, so node-local device-to-host bandwidth contention made one capture
exceed the policy bound. The failed trainer has no endpoint telemetry because
that causal record is correctly persisted only after ownership; its absence
plus the first `start()` deadline check makes the capture overrun conclusive.

The attempt produced 522 supervision events, zero restarts, zero checkpoints,
and zero atomic commits. Pass-only interleave, foreground-idle, cadence,
ten-commit, twelve-window, atomic-apply, and fresh-reload requirements were
therefore not reached. In particular, fewer than two warm-up plus ten measured
windows exist per trainer, so no honest steady K40 median or foreground-idle
fraction is defined for this attempt; both preliminary metrics are recorded as
`null`, not inferred from an incomplete first interval.

The failing-first regression is
`test_snapshot_dma_completion_is_deferred_until_after_local_owned`. The
smallest repair retains the two bounded preallocated slots, makes device slots
pinned, enqueues the coherent copy in stream order at the exact K boundary,
transfers local immutable ownership inside the existing one-second bound, and
waits for copy completion on the background publication path before any
snapshot reader. It does not enlarge any reviewed deadline or capacity.

## Authority mapping

The attempt was checked against the compute-pool conformance checklist,
R01–R16, NDP01–NDP17, ADR-002 V21S01–V21S17, and ISP01–ISP07. Exact source,
native bundle, queue identity, seed, data/tokenizer, policy/schema, capacity
declarations, and fail-closed behavior were retained. R14,
NDP13/NDP15/NDP16, V21S03/V21S11/V21S13/V21S15, and ISP03/ISP05 remain
non-passing because one trainer did not transfer snapshot ownership and no
generation committed.

Machine-readable evidence is
[`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080178.json`](../../reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5080178.json).
