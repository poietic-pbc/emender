# Async v2.1 generation-closed recovery fix

Date: 2026-07-28

WG task: `fix-v2-1-generation`

This change fixes the live recovery race retained by fault/rejoin job 5105811
and terminal-verdict SHA-256
`3496142919fd3aea359f4e14051a57bfda6d54bd57d4599964b01cdce7049721`.
The original evidence is retained on branch
`wg/agent-1602/qualify-simple-async-v21-2n-faults` in
`docs/validation/qualify-simple-async-v21-2n-faults-20260728.{md,json}`.

## Behavior

When reconstructed native peer control is already authoritative for generation
`g+1`, a late `g` contribution can no longer escape the RPC boundary as
`generation is not open`. If the exact `(g, attempt)` volatile admission is
absent and `g` is strictly older than committed authority, peer control returns
an idempotent `catch_up` receipt containing:

- the requested generation and attempt;
- the authoritative committed generation;
- the immutable commit-receipt and manifest digests;
- the authoritative result root and accepted-token clock; and
- `requires_reload=true`.

The path does not recreate an admission or accumulator. It does not touch
membership, node-apply receipts, accepted tokens, result roots, or any
incarnation. A current or future generation with no admission still fails
closed. Run and allocation-fence checks still precede dispatch.

The native manager validates that every returned digest is complete, the
authoritative generation strictly advances the submitted generation, the token
clock is nonnegative, and reload is explicitly required. It then writes a
fenced generation-catch-up handoff, releases the local native result, and exits
successfully. The supervisor therefore does not charge this fenced recovery
outcome against the unrelated atomic-cohort restart budget.

## Validation

The regression
`test_reconstructed_control_returns_idempotent_closed_generation_receipt`
was written and run before the implementation. It failed with the retained
`RuntimeError: generation is not open`. It now drives the manager-facing
`contribute_and_freeze` call twice against reconstructed generation-4 authority
using a late generation-3, superseded-incarnation contribution. Both calls
return the identical catch-up receipt.

The regression pins `Q_min=2` and snapshots all mutable and numerical authority
before the late call. It proves no admission, membership, node-apply,
commit-generation, receipt, manifest, result-root, or token-clock mutation.
The configured recovered apply authority contains two nodes and therefore
cannot manufacture one-node commit authority; no node-apply transaction is
created, so no partial all-eight apply can be published.

Canonical Frontier activation:

```text
source scripts/frontier/activate_emender_frontier.sh
EMENDER_PYTHON=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
```

Passing checks:

```text
2 passed in 25.39s
  test_reconstructed_control_returns_idempotent_closed_generation_receipt
  test_stale_duplicate_and_corrupt_contribution_receipts

79 passed in 36.93s
  tests/test_resilient_pool_runtime.py
  tests/test_resilient_e97_true_2n_launcher.py

native build: passed
CTest: 10/10 passed
```

## Conformance checklist

- Compute-pool R01–R04: the allocation fence remains mandatory; a stale
  generation receives immutable catch-up authority and cannot mutate a stale
  incarnation.
- R06 and R11: `Q_min=2` is unchanged, no admission is opened, and ordinary
  peer-control reconstruction produces a bounded recovery receipt rather than
  a fatal manager RPC.
- R07, R12, and R15: receipt/manifest/result/token authority is returned
  verbatim and remains unchanged.
- NDP06, NDP10, and NDP13: late delivery is fenced and idempotent, has no
  accumulator mutation, and is contained without an unrelated cohort restart.
- NDP12 and V21S07: the path cannot create a node-apply receipt; existing
  all-eight apply authority remains unchanged.
- V21S02, V21S05, V21S10, and V21S11: the exact live lag/recovery race now
  resolves to authoritative reload instructions while preserving two-node
  commit authority and atomic cohort semantics.

This implementation does not authorize reusing any prior payload digest.
Qualification must package a new payload identity and rerun the clean and fault
G2 gates plus the clean exact-two-node pass before retrying the fault/rejoin
phase. It does not authorize a four-or-more-node submission.
