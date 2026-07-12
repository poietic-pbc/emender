# E97 step 1,525,000 production graph quality pass

This review establishes one submission-authority chain:

`quality-pass-e97-1525000` → `integrate-proven-e97` →
`smoke-exact-256-node` → `promote-pinned-1525000-e97`.

The integration task is also gated by the completed exact-launcher rerun
`rerun-exact-proven-256`. There are no back-edges in this chain. Integration
must merge and push the canonical launcher and refreshed-seed pin to
`origin/main`; smoke must execute and attest the exact pinned rendering; only
then may production submit the attested production rendering.

## Immutable seed contract

- Object: `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_1525000/checkpoint_step_1525000_loss_2.4378.pt`
- Step: `1525000`
- Loss: `2.4378`
- Tokens: `99.9424B`
- Size: `7719679924` bytes
- SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

`latest_emender_E97_1.3B.json` is intake evidence only. No renderer, wrapper,
smoke command, or production command may resolve it at launch.

## Canonical authority

- `configs/frontier/e97_async_256.yaml`
- `configs/frontier/e97_async_256_job4962400_golden.json`
- `configs/frontier/e97_async_256_parity_policy.json`
- `scripts/frontier/render_e97_async_256.py`
- `scripts/frontier/check_e97_async_promotion.py`
- `scripts/frontier/submit_e97_async_256_smoke.sh`
- `reports/frontier/e97-async-256-job4962400-canonical-20260712.md`
- `reports/frontier/e97-async-256-rerun-job4974616-20260712.md`

The task descriptions require both profiles to be rendered to
`build/e97-256/{smoke,production}` and checked with:

```bash
python3 scripts/frontier/check_e97_async_promotion.py \
  --smoke build/e97-256/smoke \
  --production build/e97-256/production \
  --policy configs/frontier/e97_async_256_parity_policy.json
```

Production additionally requires `--require-promotion`, and submission must
flow through the same checker with `--submit --approval <reviewed-approval.json>`.
The smoke `promotion.json` must bind the successful Slurm job, `COMPLETED 0:0`
state, 256 nodes/2048 ranks, immutable seed fields, normalized fingerprint,
and exact `origin/main` commit.

## Difference allowlist

The only permitted smoke/production differences are:

1. walltime: `00:20:00` → `12:00:00`;
2. queue/QoS: `debug` → `normal`.

The partition remains `batch`. Launcher bytes, all other argv and scheduler
fields, environment, config, referenced hashes, code commit, topology/ranks,
seed, model/data, optimizer/DiLoCo settings, signals, checkpoint cadence, and
training stop budget must be identical. Any other difference fails closed.

## Superseded paths

The following paused tasks are tagged `superseded-by-1525000` and retain no
submission authority: `retry-refreshed-e97`,
`monitor-fixed-olcf-e97-256n12h-production`, `validate-production-async`,
`run-exact-256-node`, `register-refreshed-e97`, `submit-and-monitor`,
`promote-exact-successful-256`, and
`submit-fixed-olcf-e97-256n12h-production`.

## Terminal gates

Integration ends only after the exact seed verification, parity tests, clean
surgical commit, push, and equality of local `HEAD` and `origin/main`. Smoke
ends only after terminal `COMPLETED 0:0`, exact ranks 0–2047, finite loss, an
expected DiLoCo merge, and finalized/reloaded checkpoint evidence. Production
cannot dispatch until that exact smoke attestation and fingerprint validate;
it must retain the submission command/job ID and initial healthy runtime
evidence on `origin/main`.
