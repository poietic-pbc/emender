# E97-MoE router-preserved K64 masked-SFT through 512 updates

## Verdict

The fixed eight-node K64 recipe is healthy through 512 cumulative local
updates. Exact held-out assistant NLL improves monotonically, native RS
termination improves materially, WikiText retention improves, recurrent state
remains finite, and every expert remains used with zero dropped tokens.
Behavior is increasingly assistant-shaped but is not yet reliably correct or
instruction-compliant. Continued stock-Tülu exposure is justified; targeted
long-context supervision remains deferred.

## Authorities

- 64 updates: `step-02338600-tokens-0000282084265308`, manifest
  `6d9b58398c3e65d02b243f7cb7e0ae01a981dd6bb264026519f4e99f67dd8604`.
- 256 updates: `step-02338792-tokens-0000282126856548`, manifest
  `92bd172d34f181b90d984dad70a7f2b53ff75d1918db777808e78fdc06f1fed4`.
- 512 updates: `step-02339048-tokens-0000282183671698`, manifest
  `31e5673e2e67245cfb5ecb1efc8932519a6943752fd2cd904ddeb2edf12742b8`.

The 512-update authority contains eight local shards, two optimizer groups,
`sampler.absolute_rank_sample_index=512`, 113,581,970 packed tokens,
80,653,570 assistant targets, and the original router-preserved transition.

## Training evidence

- 64-update canary job `5286211`: `COMPLETED 0:0`, `Partition=batch`,
  `QOS=debug`, one target-weighted K64 merge.
- 64→256 continuation job `5286911`: `COMPLETED 0:0`, 192 updates and three
  K64 merges.
- 256→512 continuation job `5287559`: `COMPLETED 0:0`, 256 updates and four
  K64 merges.
- All successful jobs used eight nodes / 64 ranks, `Requeue=0`, LR `1e-4`,
  weight decay `0.01`, complete 4K packs, and exact optimizer/sampler resume.

Two fail-closed pre-training attempts published no checkpoint or sampler
advance: job `5286678` rejected a one-node scheduler binding (`exit 65`), and
job `5286827` rejected one reconstructed optimizer group against the
checkpoint's two groups. Commit `8ca1bcb6` added explicit split-group
reconstruction on resume; 26 focused tests passed before job `5286911`.

## Exact held-out validation

All runs enumerate exactly 1,777 packs and 4,352,510 targets once.

| Updates | Job | target-weighted NLL | bootstrap 95% interval | unused experts |
|---:|---:|---:|---:|---:|
| 64 | 5286452 | 1.257225 | [1.223413, 1.290011] | 0/layer |
| 256 | 5287343 | 1.216287 | [1.182476, 1.248943] | 0/layer |
| 512 | 5287807 | 1.187856 | [1.153995, 1.220652] | 0/layer |

## Behavioral/general-LM dose response

All panels use the immutable paired panel
`fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f`
and native FP32 recurrent caches.

| Updates | Assistant NLL | WikiText 2K/8K/32K NLL | greedy/sample RS stops |
|---:|---:|---|---|
| 64 | 1.703961 | 3.112549 / 3.026001 / 3.006592 | 5/16, 7/16 |
| 256 | 1.681451 | 3.095459 / 3.009155 / 2.991333 | 7/16, 9/16 |
| 512 | 1.670430 | 3.083496 / 2.996216 / 2.979004 | 10/16, 10/16 |

At 512 updates, candidate-minus-parent WikiText NLL is about `-0.27` at all
three contexts and assistant-response NLL is `-0.4631` paired (bootstrap
`[-0.5542,-0.3538]`). MMLU is `-0.0156` and normalized HellaSwag `-0.0039`;
both paired intervals include zero. Retrieval remains near chance with
negative margins and is not claimed. All 296 recurrent-health observations
are finite. Panel routing has zero dropped tokens and no unused expert.

Generations now terminate much more often and show recognizable explanations,
lists, code, and concise responses. They remain frequently repetitive,
factually wrong, and noncompliant with exact formats or constraints. This is
not sufficient to begin targeted long-context specialization, but the
monotonic likelihood/termination response and preserved LM/routing health
support additional stock-Tülu exposure without changing LR or optimizer.

## ADR-003 binding

This is fixed-world ADR-003 execution. Applicable safety intent is
R07/NDP15 atomic complete checkpoint publication, R12 stable exact restart,
R14/NDP13 fail-stop teardown, and R16 attended immutable eight-node admission.
Dynamic R02–R06/R08–R11, NDP01–NDP12/NDP14/NDP16/NDP17,
V21S01–V21S17, and ISP01–ISP07 remain retired/unclaimed. There is no SQLite,
shared-filesystem database/lock/heartbeat, communicator shrink, requeue, or
automatic relaunch.
