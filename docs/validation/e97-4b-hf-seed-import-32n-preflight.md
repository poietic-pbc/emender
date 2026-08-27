# E97 4B Hugging Face seed import: 32-node canary preflight

**Status:** submitted as payload `5354034` with collector `5354035`. No
continuation is authorized by this preflight. Immutable execution source:
`51032a4cf4b1f3e78a3da4c5e084134f556b5dd0`.

## Immutable seed

- repository: `spinozans/e97-4b-training-checkpoints`;
- revision: `8bf6f0e9241a3eb869676fdf6b92578ced8a6f00`;
- path: `checkpoints/step_012800_tokens_6710886400/checkpoint_step_012800_loss_2.8143.pt`;
- SHA-256: `81fcc932e93df59a478e43b31afc5f0b310f58b8a5deab91a73e5be1a4925ed9`;
- bytes: 24,276,093,119;
- step/token/loss: 12,800 / 6,710,886,400 / 2.8142955899;
- model tensors: 237;
- ScheduleFree state entries: 236 in one parameter group;
- source world: eight ranks, B32/K32, Pile corpus;
- no stateful DiLoCo outer state is required for stateless averaging.

The file was downloaded with `hf download` at the immutable revision, then
`sha256sum -c SHA256SUMS` passed on Frontier. It is raw trusted `torch.save`
pickle and must never be loaded before digest verification.

## Explicit phase transition

The source metadata correctly declares that it is not an exact world-256
resume. The canary uses an explicit, fail-closed counter transition:

- previous identity: counter-v1, Pile digest
  `5eb92c0f16157710c90e33b02fd5b7852b30713d4c754f4220ad7120155db464`,
  world eight;
- new identity: counter-v2, CommaPile digest
  `44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9`,
  world 256;
- tokenizer digest remains
  `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069`;
- immutable new-stream origin equals the source accepted-token boundary,
  6,710,886,400;
- source step 12,800 is K32-aligned;
- new checkpoint cursors and step clocks are phase-relative and preserve the
  transition record durably.

Silent sampler relabeling remains prohibited. The explicit transition flag
fails on a non-K source, a wrong origin, invalid previous metadata, or an
already matching identity, before model mutation.

## Canary schedule

- `Partition=batch`, `QOS=debug`, `Requeue=0`;
- 32 nodes / 256 ranks / one GCD per task;
- one-hour limit;
- B1, context 2,048, K32;
- source step 12,800 to target step 13,312;
- 512 new optimizer steps / 268,435,456 new tokens / 16 merges;
- saves at steps 13,056 and 13,312, both K-aligned;
- unchanged ScheduleFree LR `0.00047431158698290157`;
- exact imported model and optimizer state, changing optimizer state storage
  from source CPU offload to the qualified Frontier GPU-resident path;
- checkpoints retain the source/new sampler transition provenance and must pass
  hash plus mmap reload validation.

## Validation

```text
source scripts/frontier/activate_emender_frontier.sh
bash -n scripts/frontier/submit_e97_4b_from_scratch.sh \
  scripts/frontier/e97_4b_from_scratch.sbatch \
  scripts/frontier/e97_4b_from_scratch_collector.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_train_helpers.py tests/test_frontier_e97_4b_from_scratch.py
```

Result: 23 passed. Direct trusted-checkpoint preflight resolved the exact
6,710,886,400 boundary and emitted the expected old/new transition identities.

## Architecture scope

This is ADR-003 fixed-world qualification. Applicable safety intent is R07,
R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic R02--R06/R08--R11
and NDP02, async-v2.1 V21S01--V21S17, ISP01--ISP07, and native NDP17 are
explicitly unclaimed. This canary is a reviewed seed-import experiment, not a
production scale-ladder pass.
