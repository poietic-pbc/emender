# Authoritative bundle and serial exact-2n acceptance fix

The canonical renderer now treats submission as a resumable controller rather
than a dependency-chain generator. Its first submission invocation requires a
clean, pushed `main`, inventories every tracked source file, removes any prior
commit-named staging directory, runs the canonical native build plus CTest into
that stage, records the binary manifest, and verifies the exact production G2
gate against the rebuilt source and bundle. It rechecks the complete source
inventory immediately before `sbatch`.

Only one phase can be active. Each invocation either submits exactly one phase,
returns the resumable wait exit code 75 for a pending/running/accounting-pending
job, or harvests terminal scheduler evidence before advancing. No Slurm
dependency is queued. Clean overlap, fault/rejoin, invalid-result rejection,
publication failure, and fresh restart therefore execute serially, with the
previous terminal evidence exported as the next phase handoff. The intentional
publication failure must fail; every other phase must complete.

## Validation

- `tests/test_resilient_e97_exact_2n_acceptance.py`: 5 tests pass, including
  stale installed-source rejection, a simulated one-job QOS submission, and a
  PENDING phase that remains in resumable wait without another `sbatch`.
- Focused canonical launcher and attestation suites pass:
  `tests/test_resilient_e97_true_2n_launcher.py` and
  `tests/test_native_artifact_attestation.py`.
- Python compilation and shell syntax checks pass.
- The canonical Frontier native build succeeds and CTest passes 10/10.
- Dry-run invariants remain exact two nodes, real native CXI, K40, five initial
  clean generations, ordered fences/handoffs, and explicit prohibition of
  4/8/32/64/256-node acceptance submission.
- No `sbatch`, `srun`, or other Slurm submission was executed during this task.

Conformance checklist: Compute Pool v1 requirements R01-R16 remain bound in
the rendered manifest. The fix particularly reinforces R01/R02 (exclusive
fenced allocation and no overlapping writer), R08/R09 (immutable atomic
handoff/publication), R10/R11 (bounded failure and fresh-allocation recovery),
R13 (source/config/artifact identity), and R16 (retained evidence). Native data
plane requirements NDP01-NDP17 remain required; exact rebuild/gate verification
specifically reinforces NDP01-NDP04, NDP10, NDP13, NDP15, and NDP17.
