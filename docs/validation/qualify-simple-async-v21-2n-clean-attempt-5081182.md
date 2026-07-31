# Async v2.1 two-node clean attempt 5081182

## Verdict

Job `5081182` is immutable, non-passing evidence. It ran on exactly
`frontier[07663,07877]`, `Partition=batch`, `QOS=debug`, without a scheduler
or process restart, and terminated:

```text
5081182|FAILED|1:0|2|batch|debug|2026-07-26T11:39:50|2026-07-26T11:56:30|00:16:40
```

Payload
`00cfa73079a8f191acc2a3f950e2f07b78a7388b4bb63ee342ae43b5ce3aa339`
is retired and must never be resubmitted.

Queued, nine running, and terminal `squeue`/`scontrol`/`sacct` transcripts are
under
`/lustre/orion/bif148/scratch/erikgarrison/emender-qualification/qualify-simple-async-v21-2n-clean/4b2b991644466ecd9e1e12b108b6bc61aa3e1860/scheduler-evidence/clean-5081182`.
The queued, first-running, and terminal transcript SHA-256 values are,
respectively,
`72ad84e61a6716c1d576ef05256b32d79c0e80f718ee2b3068576d89ce3cc8d7`,
`76a58b7976e3d696240bcf4dfcd7f4f5e0218d51b81a477e6729d622c17bd579`,
and
`16d3c9f1c225fd4f5c63acbdf66c240bb61e76fe27277ed6392e1caf8ed6c967`.

## Exact-source prerequisites

The clean, fetched snapshot was exact source
`4b2b991644466ecd9e1e12b108b6bc61aa3e1860`. Its canonical current-source
native rebuild passed all 10 CTests and produced manifest SHA-256
`063880cd1b64de5f4e863f08df1371217ddef2e18396ba1ea6b1d5839c031f98`
and bundle
`f19e10be9987cfdb551a8dd75c5c88145c3cf35b73c54d3898fe562ce4182441`.

Exact-source G2 job `5081162` passed:

```text
5081162|COMPLETED|0:0|2|batch|debug|2026-07-26T11:32:51|2026-07-26T11:35:53|00:03:02
```

Its gate SHA-256 is
`7becec2c4f5bb0335ff15632f8a5270739c16806eb1abe01a6edfdf5c17719ea`;
median/max full-layout time was `24.203651/25.145998 s`, with 34 retained
bounded retries and zero route errors, CQ errors, Python dense bytes, disk
replay bytes, or all-rank barriers.

The canonical serial controller plan SHA-256 is
`940314f62d55a24a91207858d56346684bbcf8061daa9c604b2c79d58036e34f`.
It binds the source, native bundle, G2, policy, launcher, data, tokenizer,
seed, and the reviewed 2,700-second clean progress deadline.

Both nodes independently verified the exact offline seed staged at
`/tmp/emender-e97-seed-5081182`: step `2300930`, tokens `150793748480`,
bytes `7719680116`, SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`,
and `network_fetches=0`.

## Exact terminal cause

The preceding fix correctly made each manager retain peer-verified immutable
commit authority throughout result preparation and apply. Both terminal
manager states prove:

```text
generation=0
authoritative_generation=1
stage=peer_apply
commit_receipt_digest=7b57a1291cc4e837df39c61411a3d256207c518070661780909d928309131769
```

Generation 1 was therefore durably committed before the 720-second deadline.
The receipt was published at `1785081063279671800 ns`, accepted-token clock
`150798993920`, result root
`10a3be5361d1166ec3757ce8af1323302637af014033ffc800ac46467a70447b`,
and checkpoint SHA-256
`22e6c3513494ad67f1125f037f6af7bda1c25a7823a9d7cbd0d0803bbebdeca8`.

The remaining defect was role scope. The supervisor independently ran the
allocation-wide first-commit predicate against `trainer-0` on each node.
Trainer progress documents intentionally do not duplicate manager-owned
immutable commit authority, so each trainer returned
`first_atomic_generation_deadline` and caused its complete node cohort to
stop. Exact events were:

- node 0 manager/trainer at `1785081384.725629/1785081384.731183`;
- node 1 manager/trainer at `1785081384.798127/1785081384.816025`.

The failing-first regression is
`test_first_commit_deadline_is_owned_by_manager_not_individual_trainer`.
The smallest repair scopes this allocation-wide check to `manager` and
`node-supervisor`. Trainer heartbeat, K40, and stage progress remain governed
by their existing independent fail-closed bounds. No deadline is enlarged.

## Preliminary performance evidence

Serial preparation succeeded for all trainers: 16 materializations
(`42.689960–43.747996 s`, median `42.905637 s`), 16 authenticated
result-materialized receipts, 16 ready receipts, and both node release
receipts. The supervisor interrupted the foreground apply before a complete
trainer apply receipt.

Every trainer retained exactly nine K40 windows, so this is not promotion
evidence. Validator-equivalent preliminary metrics were:

- raw K40 median/max `67.636728/74.256676 s`;
- cadence median/max `67.670376/74.270095 s`, or `1.000497x`;
- aggregate idle `0.000060860`, p99/max gap
  `0.029353/0.044076 s`.

The machine-readable companion is
`reports/frontier/qualify-simple-async-v21-2n-clean-attempt-5081182.json`.

## Validation

- R01–R16 and NDP01–NDP17 retain exact-source, queue, seed, native transport,
  capacity, and fail-closed evidence; no clean promotion is claimed.
- V21S01–V21S17 did not complete because the task still lacks ten commits,
  twelve windows per trainer, all-eight application, and fresh-process
  checkpoint verification.
- ISP01–ISP07 serial preparation passed, but final atomic apply was
  interrupted by the trainer-scoped diagnostic defect.
