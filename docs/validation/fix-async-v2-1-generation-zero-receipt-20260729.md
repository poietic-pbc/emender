# Async v2.1 generation-zero receipt normalization

Task: `fix-async-v2-1`

Date: 2026-07-29

Slurm submissions: **none**

## Defect and repair

Frontier model job 5111908 exposed a Python/native representation mismatch
during cold-start peer recovery.  The immutable manifest correctly represents
the absence of a prior generation-zero commit receipt as `""`; the native
fixed-width 32-byte digest ABI serializes the same zero-initialized sentinel as
64 zero hexadecimal characters.  The recovery validator previously compared
those representations literally and stopped before READY.

`_native_commit_receipts_agree` now treats only `""` and exactly 64 zero
hexadecimal characters as the equivalent no-prior-commit identity, and only
when `generation == 0`.  Generation greater than zero still uses exact string
identity.  Manifest digest, result root, accepted-token clock, node-apply
receipts, generation, status, and apply requirement validation are unchanged.

## Test evidence

The canonical Frontier environment was sourced with:

```text
source scripts/frontier/activate_emender_frontier.sh
```

The regression was run before the implementation and failed at
`_validate_native_recovery_handshake` when the native zero sentinel was
compared with the empty manifest receipt.  After the implementation:

- `test_generation_zero_native_recovery_normalizes_only_no_commit_receipt_and_readies`
  accepts the empty and fixed-width-zero forms, calls the recovered-peer READY
  transition, and rejects short zero, 31-byte zero, nonzero 32-byte, and
  non-digest spellings.
- `test_nonzero_native_recovery_authority_mismatches_remain_fail_closed`
  independently corrupts receipt, manifest, result, accepted tokens, and
  node-apply authority at generation 3; every case rejects.
- The focused recovery set passed: 7 tests.
- The applicable controller, launcher, and runtime suite completed 190/191
  tests on its first run.  The unrelated fake-scheduler timing test did not
  release its test payload within its polling window and passed immediately
  when rerun alone (1/1).  A clean consolidated rerun then passed 191/191.
- `git diff --check` and Python byte compilation passed.

## Compute-pool conformance checklist

- Allocation/fencing and source identity: unchanged.  The repair does not
  change claim selection, scheduler fence, run/source/code/payload identity,
  incarnation sequencing, or stale-fence behavior.
- Membership and READY: generation-zero recovery may reach READY only after a
  valid native recovery response; all existing status, generation, boolean
  apply requirement, and READY incarnation checks remain active.
- Generation/commit atomicity: no commit is synthesized.  The two accepted
  spellings both mean that no prior commit exists at generation zero.
- Recovery authority: generations above zero retain exact receipt plus
  manifest/result/token/apply comparisons.  Noncanonical generation-zero
  values fail closed.
- Bounds and transport: no deadline, retry, buffer, RPC, CXI, memfd, payload,
  aggregation, redistribution, or checkpoint behavior changes.
- Evidence/operations: regression coverage is durable; no Slurm command was
  submitted.

### Requirement coverage

- **R01–R16:** conformant.  The change is directly relevant to R02/R03
  membership admission, R07 atomic authority, R12 exact recovery, and R14
  bounded recovery; R01, R04–R06, R08–R11, R13, and R15–R16 are unchanged and
  remain guarded by the applicable suite.
- **NDP01–NDP17:** conformant.  The Python/native ABI boundary now gives the
  native fixed-width zero digest its one canonical generation-zero meaning.
  Native fencing, incarnation, generation, manifest, result, token, apply,
  transport, ownership, memory, and checkpoint rules are not weakened.
- **V21S01–V21S17:** conformant.  V21S10/V21S11 recovery and atomic node READY
  gain the cold-start regression; lag, exact-token math, snapshot, transport,
  checkpoint, restart, performance, and convergence semantics are unchanged.
- **ISP01–ISP07:** conformant.  This evidence-only boundary repair changes no
  direct systems-scale policy, qualification ordering, scheduler transaction,
  immutable execution-source identity, or promotion gate.
