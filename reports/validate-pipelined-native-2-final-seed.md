# Final-seed pipelined native DiLoCo exact two-node acceptance

Date: 2026-07-23
Task: `validate-pipelined-native-2-final-seed`
Status: exact-source G2 passed; clean five-generation K40 job 5060027 pending

## Authority and immutable identity

This run follows *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17),
requirements R01-R16, and the native data-plane v1 specialization NDP01-NDP17.
The mandatory conformance checklist is applied to the exact two-node runner:
leased READY membership rather than launched-rank membership; bounded
point-to-point native transport; fenced generation identity and atomic
publication; strict stale/corrupt/non-finite/non-quorum rejection; bounded
latest-only queues; model-free managers; and fresh-fence checkpoint recovery.

The clean authoritative clone is:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T134500Z/source
origin/main = f6003e32e14b89e0fde1f6b7f47b6402285d7b39
```

The immutable seed binding is step 2300930, 150793748480 accepted tokens,
7719680116 bytes, and SHA256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
Every submission in this acceptance is constrained to exactly two nodes,
`Partition=batch`, and `QOS=debug`; no 4-node-or-larger submission is
authorized.

## Batch-runtime seed scoping proof

The focused production regression ran under the canonical Frontier
environment and passed 5/5 cases:

```bash
"$EMENDER_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_resilient_e97_exact_2n_acceptance.py::test_submit_shell_preserves_job_id_until_batch_runtime \
  tests/test_e97_s3_seed.py::test_destination_job_scope_fails_closed
```

The exact `rendered.sbatch` retains the single-quoted literal
`'/tmp/emender-e97-seed-${SLURM_JOB_ID}'` at submit time. The batch script
performs the only expansion after Slurm supplies the live job ID, and the
materializer independently rejects unset, empty, or mismatched job/destination
identities before download or model load.

## Exact native bundle and scheduler record

The canonical native build from exact source passed CTest 10/10. Its manifest
is:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T134500Z/native-stage/f6003e32e14b89e0fde1f6b7f47b6402285d7b39/install/native-artifacts.json
SHA256 546766dc559a9b4fc8231155bce215bc1ed2d48fd69c4ed7824a0894f65ff5cc
```

The real acceptance preflight repeated the clean build from a fresh `main`
checkout at the same exact commit:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-20260723T134500Z/source-clean
origin/main = HEAD = f6003e32e14b89e0fde1f6b7f47b6402285d7b39
authoritative native bundle = 9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a
native manifest SHA256 = 546766dc559a9b4fc8231155bce215bc1ed2d48fd69c4ed7824a0894f65ff5cc
acceptance manifest SHA256 = dde613b097cc9ee8549cf3ac757e897565fa8028b335bebadec5f42a15bc93e1
```

The user queue was empty immediately before submission. Historical job
5059548 was terminal `FAILED 1:0`, exactly two nodes, `Partition=batch`,
`QOS=debug`, and therefore was not an active equivalent. Because its G2
attestation used source `7ab92ada...`, source identity changed and required a
fresh G2.

Exactly one exact-source G2 prerequisite was submitted:

```text
5059795|native-ndp-g2-clean|COMPLETED|0:0|2|batch|debug|00:02:58
```

The terminal accounting query explicitly returned `Partition=batch` and
`QOS=debug`. The retained full-layout gate is
`g2-artifacts/5059795/full-layout-gate.json` under the execution root.

After G2 completed, the first serial acceptance attempt did not reach
`sbatch`: unrelated untracked files in the first exact clone made the mandatory
all-untracked source check remain in a Lustre metadata wait. A new local clone
was checked out on `main` at the same immutable `origin/main`, and its complete
`git status --porcelain --untracked-files=all` returned clean. The unmodified
launcher then repeated the exact-source inventory, native build (CTest 10/10),
production G2 attestation, and adjacent empty-user-queue check.

Exactly one real clean replacement was submitted:

```text
5060027|resilient-e97-true-2n|PENDING|2|batch|debug|0:00|2:00:00
```

The retained serial controller records phase `clean-overlap`, five K40
generations, job 5060027, and the requested `Partition=batch`/`QOS=debug`.
No fault/rejection/checkpoint/restart phase, duplicate, or job larger than two
nodes has been submitted. This task is parked while job 5060027 is pending.

The 2026-07-23 10:55 EDT resume check again found 5060027 as the sole job in
the user queue. `squeue -o '%i|%T|%D|%P|%q|%R|%j'` reported
`5060027|PENDING|2|batch|debug|(Priority)|resilient-e97-true-2n`, and the
independent `sacct` record agreed on `PENDING`, two nodes, `Partition=batch`,
and `QOS=debug`. It had not started or consumed runtime. The retained
controller state still names this job as the five-generation `clean-overlap`
phase, so no equivalent or later-phase submission was made.

The 2026-07-23 11:15 EDT resume check still found 5060027 as the sole user
queue job: `PENDING`, reason `Priority`, exactly two nodes,
`Partition=batch`, and `QOS=debug`. Independent `sacct` and `scontrol -dd`
records agree, show zero runtime, and retain the exact clean-overlap command,
five-generation request, final-seed identity, native/G2 manifests, and
two-hour walltime. Slurm currently estimates a 13:10 EDT start. No duplicate,
later serial phase, or job larger than two nodes was submitted.

The 2026-07-23 11:40 EDT resume check again found 5060027 as the sole user
queue job. `squeue` reported `PENDING`, reason `Priority`, exactly two nodes,
`Partition=batch`, and `QOS=debug`; `sacct` and `scontrol -dd` independently
agree and show zero elapsed runtime. The retained command still binds the
clean-overlap phase to five generations, the exact final-seed identity, source
`f6003e32...`, and the attested native/G2 manifests. Slurm now estimates a
13:06 EDT start. No duplicate, later serial phase, or job larger than two
nodes was submitted.

## Validation checkpoint

- R01-R16 and NDP01-NDP17: reviewed against the version-1 authority; runtime
  claims remain pending.
- Exact source, bundle, final seed, and deferred/live job-ID scoping: recorded.
- Exact-source G2: passed, job 5059795, `COMPLETED 0:0`, 2 nodes,
  `Partition=batch`, `QOS=debug`, with retained full-layout gate.
- Scheduler binding: exact two nodes, `Partition=batch`, `QOS=debug`; no
  duplicate and no scale-out. Exactly one clean five-generation replacement,
  job 5060027, is pending.
- Five K40 generations, overlap/SLO metrics, fault/rejoin, rejection,
  checkpoint failure, and fresh restart: not yet claimed while the clean job
  is pending; later serial phases remain unsubmitted.
