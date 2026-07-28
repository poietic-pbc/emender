# Frontier durable collector account and QoS correction

Date: 2026-07-27

WG task: `fix-frontier-durable`

Status: **PASSED — fake-scheduler implementation validation complete; no real
Slurm job was submitted.**

## Result

The model allocation remains exactly `Nodes=2`, `Partition=batch`,
`QOS=debug`. The scheduler-owned `afterany` collector is now submitted as
`Account=bif148`, `Nodes=1`, `Partition=batch`, `QOS=normal`. This is an
intentional separate scheduler identity: Frontier's `debug` QoS permits only
one submitted job per user, so a second debug collector cannot coexist with
the held model payload.

The collector identity retained in controller state is derived from the
actual `sbatch` command options. The fake Frontier rejects a collector with no
account and rejects a second submitted debug-QoS job while the first remains
held. Registration failure leaves the model job held; retry reconciles the
same payload and creates exactly one collector before exactly one release.
The process-level worker-death test proves that the scheduler child survives,
captures terminal accounting and logs, writes literal `passed`, and remains
idempotent on re-execution.

## Validation

All Python commands used the canonical environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Focused controller result:

```text
27 passed
```

Controller, launcher, exact two-node acceptance, native attestation, and
native-gate result:

```text
128 passed, 1 skipped in 94.35s
```

The skip is the pre-existing explicit absence of a locally built canonical
native bundle. Native source and ABI were not changed. `py_compile` passed for
the controller, terminal collector, fake scheduler, and controller tests.
`git diff --check` passed.

No validation command invoked the real `sbatch`, `scontrol`, `squeue`, or
`sacct`; scheduler mutations were confined to the process-level fake.

## Architecture conformance

This correction was checked against the complete conformance checklist in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and the applicable requirement IDs in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. It changes only submit-side collector
scheduler identity and fake-scheduler evidence. The detailed semantic mapping
in `fix-v21-durable-collector-source-identity-20260727.md` remains applicable.

| IDs | Delta conformance mapping |
|---|---|
| R01, R04, R07, R12, R14 | Held payload identity is durable before collector registration; collector account/QoS identity is durable before release; registration failure is fail closed and retry is idempotent. |
| R02, R03, R05, R06, R08, R09, R10, R11, R13, R15, R16 | Training ownership, leased membership, exact math, bounds, native transport, model isolation, recovery, backend boundary, numerical behavior, and ordered promotion are unchanged. |
| NDP01, NDP02, NDP06, NDP10, NDP13, NDP16 | The collector remains outside live native control; the scheduler dependency is point-to-point job metadata; fixed identities, finite retry, and separate scheduler fields are retained. |
| NDP03, NDP04, NDP05, NDP07, NDP08, NDP09, NDP11, NDP12, NDP14, NDP15, NDP17 | Native provider, handoff, arithmetic, routes, bounds, credits, replay, redistribution, ABI, checkpoint authority, and G2 gate are unchanged. |
| V21S01, V21S05, V21S14, V21S15, V21S16, V21S17 | Versioned identities remain fail closed; model jobs remain exact two-node/batch/debug; seed, promotion, and finite closure rules are unchanged; collector Account/QoS is retained separately. |
| V21S02, V21S03, V21S04, V21S06, V21S07, V21S08, V21S09, V21S10, V21S11, V21S12, V21S13 | Lag clocks, exact-token math, K40, snapshot/apply/mailbox bounds, leased READY semantics, atomic node behavior, native path, and causal telemetry are unchanged. |
| ISP01, ISP02, ISP03, ISP04, ISP05, ISP06, ISP07 | Live-state ownership, immediate resume, immutable background inputs, bounded capacity, atomic apply, causal phase telemetry, and tail evidence are unaffected; this task claims no new snapshot-pipeline qualification. |

Thus every required identifier is mapped:
R01–R16, NDP01–NDP17, V21S01–V21S17, and ISP01–ISP07.
