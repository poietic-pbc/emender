# E97-MoE masked-SFT preserved-optimizer canaries

Date: 2026-08-16  
Verdict: **the natural continuation control was implemented exactly and
rejected. Preserving the mature 282B ScheduleFree state worsened held-out
assistant loss at `5e-6`, the parent's stored `1e-4`, and the original
`1.007e-3`. Do not scale these branches.**

## Purpose

The first masked-SFT canaries restored the 282B model but constructed fresh
ScheduleFree state. This experiment tested the more literal continuation:
restore the exact 282B model, `z`, second moment, ScheduleFree clocks/weights,
and hyperparameter group, change only the sampler/objective to assistant-masked
Tülu, and optionally override only the learning rate.

The parent optimizer authority was mature:

- `k=16016`;
- `lr=1e-4`, `lr_max=0.001007`;
- `weight_sum=0.016241008784005282`;
- betas `(0.9, 0.95)`;
- weight decay `0.01`;
- checkpointed in ScheduleFree eval mode.

Implementation commit: `e4ece1462162b67fc67a2cfccc0d04bb5c7a23dc`.
The transition fails closed on exact parent generation, step, accepted-token
clock, mature optimizer group, and complete eight-shard authority. Published
SFT checkpoints bind the old counter sampler and the new record-pack sampler
with transition status `counter-to-sft-preserve-optimizer`.

## Results

All jobs used one complete eight-GCD node-local EP island, identical packs and
validation examples, 4K contexts, assistant-only loss, `Partition=batch`,
`QOS=debug`, and `Requeue=0`.

| Job | Restored LR | Updates | Packed / target tokens | Initial validation NLL | Final validation NLL | Delta |
|---|---:|---:|---:|---:|---:|---:|
| 5280803 | stored `1e-4` | 8 | 229,705 / 156,520 | 1.67005 | 1.84175 | **+0.17170** |
| 5280876 | `1.007e-3` override | 8 | 229,705 / 156,520 | 1.66979 | 1.69731 | **+0.02752** |
| 5280927 | `1.007e-3` continuation | 64 cumulative | 1,787,701 / 1,206,497 | 1.66979 | 1.79765 | **+0.12785** |
| 5280986 | `5e-6` override | 64 | 1,787,701 / 1,206,497 | 1.66996 | 1.68024 | **+0.01028** |

All runs completed `0:0`, remained finite, and published complete checkpoints.
The `1.007e-3` hypothesis was materially better than stored `1e-4` over the
first eight updates, but it still regressed and worsened further by 64. The
low-LR preserved-state control also regressed. No branch passes even the
teacher-forced gate, so generation evaluation or multinode scaling is not
justified.

Canonical diagnostic manifests:

- stored `1e-4`, 8 updates:
  `997a6473189d4e34fb299dd36436c63182a389ddb37a7cf18ac2443c8f99edb4`;
- `1.007e-3`, 8 updates:
  `afad1bc16cb675e44e426bf4943c11fcb2fe6aef127b2e4cddca5c27183f7601`;
- `5e-6`, 64 updates:
  `a87514eeb929515466e76a83b4a2492ab24e4c8b3e86db4dc3c062f1b0bab79c`.

Job 5280798 failed in launcher preflight before model load because the submitted
source hash was mistyped. The exact source identity was diagnosed and locally
verified before replacement job 5280803; it consumed five seconds and produced
no training authority.

## Interpretation

Preserving optimizer history is a reasonable continuation hypothesis, but the
stored ScheduleFree state is not neutral history. Its `z`, second moment,
weight sum, and clocks encode the long-context pretraining trajectory and
unmasked token objective. The abrupt transition changes context length,
sampling unit, loss support, normalization, and gradient distribution. The
physical controls show that this history is harmful rather than stabilizing on
the masked objective.

This result also narrows the prior diagnosis: fresh state was a confound, but
it was not the reason masked SFT failed to generate. Fresh `5e-6` improved
held-out masked NLL, whereas preserved `5e-6` regressed on the same validation
gate. The behavioral failure must be investigated separately.

## Decision

- Preserve these outputs only as diagnostic evidence.
- Do not continue or scale any preserved-optimizer branch.
- Retain fresh-state masked SFT as the better optimization basis on measured
  held-out likelihood.
- Before more 35B exposure, prove cached/full-prefix decoding parity and add
  exact-format held-out Tülu generation. Then decide whether a fresh-state LR
  sweep or precision-correct full-model update is warranted.

This is ADR-003 fixed-world evidence. Applicable safety intent: R07/R12 atomic
complete restart authority, R14/NDP13 bounded fail-stop execution, R16 evidence
discipline, and NDP15 atomic checkpoint publication. Elastic R02-R06/R08-R11,
NDP02 dynamic semantics, V21S01-V21S17, and ISP01-ISP07 are explicitly retired
and unclaimed for these runs.
