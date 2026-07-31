# E97 256-node production promotion blocked (2026-07-12)

Task: `promote-pinned-1525000-e97`

## Outcome

No production job was submitted. The first fail-closed pre-submit invariant
failed, so no approval JSON was created and the submission command was not
invoked.

After `git fetch origin`, the tracked worktree was clean and the authoritative
commits were:

- `HEAD`: `3834ba7f4e241e2334dbb7dd27eef918d5475df0`
- `origin/main`: `3834ba7f4e241e2334dbb7dd27eef918d5475df0`
- `build/e97-256/smoke/promotion.json` `origin_commit`:
  `57884f17138843359fcdf164e178e329f6cb6f71`

The retained smoke attestation was introduced by commit `3834ba7`; its recorded
origin is the earlier launcher commit `57884f1`. Consequently, it is impossible
for the current repository commit, `origin/main`, and the immutable retained
attestation's origin commit to all be equal. Rewriting `promotion.json` would
violate the instruction to reuse the retained smoke attestation.

## Gate evidence

The exact required non-submitting command was run:

```text
python3 scripts/frontier/check_e97_async_promotion.py --smoke build/e97-256/smoke --production build/e97-256/production --policy configs/frontier/e97_async_256_parity_policy.json --require-promotion
```

Before the gate was hardened, it exited 0 because it checked only that
`origin_commit` was syntactically a SHA-1. The gate now enforces the required
three-way equality and exits 1 with:

```json
{"detail":{"head":"3834ba7f4e241e2334dbb7dd27eef918d5475df0","origin_main":"3834ba7f4e241e2334dbb7dd27eef918d5475df0","promotion":"57884f17138843359fcdf164e178e329f6cb6f71"},"kind":"origin_commit","ok":false}
```

The parity portion otherwise reported fingerprint
`623f8cf34e88b614dabfcf5b130f7a6cb1ca702efe48b44149d5765cc3bf850a`
and only the allowlisted profile differences: walltime `00:20:00` to
`12:00:00`, QoS `debug` to `normal`, and unchanged partition `batch`.

## Safety and validation

- No reviewed approval JSON exists.
- The `--submit` command was not run; therefore there is no production job ID.
- No stable seed pointer or retained smoke artifact was modified.
- All eight superseded tasks named by the assignment remain paused and tagged
  `superseded-by-1525000`.
- `python3 -m pytest -q tests/test_e97_async_256_promotion.py`: 52 passed.

The promotion requires an explicit authority decision that reconciles the
contradictory commit invariant without falsifying the successful smoke's
recorded launch origin.
