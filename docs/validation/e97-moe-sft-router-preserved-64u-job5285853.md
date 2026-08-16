# E97 MoE router-preserved 64-update evaluation

Date: 2026-08-16

Verdict: **promote the selected recipe to the independent eight-node K64 canary.**

## Bound artifacts

- Training job: `5285448`, `COMPLETED 0:0`, `Partition=batch`, `QOS=debug`.
- Evaluation job: `5285853`, `COMPLETED 0:0`, `Partition=batch`, `QOS=debug`.
- Candidate generation:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-router-preserved-1e4-64u-canonical-v1/checkpoints/step-02338600-tokens-0000282071877429`
- Candidate manifest SHA-256:
  `f190f0bcf735c07dc7f59ec27ac98fc8c0257718d8a331fc62ab6c529314814d`.
- Evaluation panel SHA-256:
  `fcb1fbd09ca38c27fec03be945c7cbfcb45cff4dbcaf8a4bf4afcb2ef013018f`.

The checkpoint contains eight complete shards, two optimizer groups, and the
recorded router-preserved/nonrouter-fresh transition. It consumed 1,787,701
packed tokens and 1,206,497 assistant targets in 64 updates.

## Exact masked-SFT validation

All 1,777 validation packs and 4,352,510 targets were enumerated once.

- Parent NLL: `1.67079623`.
- Candidate NLL: `1.28223657` (`-0.38855966`).
- Candidate bootstrap interval: `[1.24861556, 1.31485678]`.
- Every layer retained zero unused experts.
- Layer-10 max/mean: `4.97136`; normalized entropy: `0.93222`.

## External paired evaluation

Candidate minus parent:

- Assistant-response NLL: `-0.47601`, bootstrap interval
  `[-0.55940, -0.38600]`.
- WikiText NLL 2K: `-0.25049`.
- WikiText NLL 8K: `-0.25586`.
- WikiText NLL 32K: `-0.25525`.
- MMLU accuracy: `-0.01563`, interval `[-0.05859, 0.02734]`.
- HellaSwag normalized accuracy: `-0.01172`, interval
  `[-0.03516, 0.01172]`.

The multiple-choice changes are inconclusive and do not show a robust collapse.
Retrieval remains near chance and is not a promotion claim.

All 296 recurrent-health observations were finite. Candidate layer-10 routing
on the broad panel retained normalized entropy `0.90108`, minimum expert count
1,114, and no dropped tokens.

## Native generation

The candidate now produces recognizable assistant responses, refusals, lists,
code, and formatted answers. It remains substantially undertrained: repetition,
incorrect reasoning, excessive refusal, constraint failures, and nontermination
are common. RS termination nevertheless increased:

- greedy: parent 2/16, candidate 5/16;
- sampled: parent 2/16, candidate 8/16.

This is meaningful emergence after only 1.8M packed tokens, not finished
assistant quality. Requiring polished coherence at this screening dose would be
overly conservative. The eight-node K64 arm supplies approximately eight times
the supervised exposure while directly qualifying the target cross-node merge.

## Decision

Run an independent fixed eight-node world from the 282B parent with the same
full-model recipe: router state preserved, nonrouter state fresh, LR `1e-4`,
weight decay `0.01`, 4K complete packs, K64, and one target-weighted DiLoCo
merge. Publish the canonical node-zero eight-shard checkpoint. Then perform
one-node exact validation and the matched behavioral/general-LM panel before
resuming that same fixed world toward 512 updates.

## ADR-003 scale binding

The next runner is fixed-world ADR-003 production, not resilient/elastic
compute-pool execution. Applicable safety intent is R07/NDP15 atomic complete
checkpoint publication, R14/NDP13 fail-stop bounded rank teardown, R16's
attended immutable eight-node admission, and scheduler `Requeue=0`. R12 restart
is not exercised by the initial canary. Dynamic R02–R06/R08–R11,
NDP01–NDP12/NDP14/NDP16/NDP17, V21S01–V21S17, and ISP01–ISP07 are explicitly
retired/unclaimed. The runner has one fixed 64-rank child, no SQLite, shared
filesystem database/lock/heartbeat, communicator shrink, automatic relaunch,
or scheduler requeue. Any rank failure terminates the job and cannot publish a
canonical checkpoint.
