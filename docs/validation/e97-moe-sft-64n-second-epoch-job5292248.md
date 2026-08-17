# E97 MoE masked-SFT 64-node second-epoch verdict

## Verdict

The operator-directed 64-node K64 fork successfully consumed approximately one
additional eligible Tülu-pack epoch in one debug-QoS job, but the recipe has
reached a quality plateau. Preserve the result as evidence; do **not** add more
unchanged Tülu exposure at LR `1e-4`.

Job `5292248` completed `384/384` updates on 64 nodes / 512 ranks in `01:11:41`
with `Partition=batch`, `QOS=debug`, and `Requeue=0`. It retained exact model and
two-group ScheduleFree state from the 4,096-update checkpoint while explicitly
transitioning the deterministic SFT data world from 64 to 512 ranks. The run
processed 681,627,831 packed tokens and 483,968,394 assistant targets, equal to
1.098 eligible packed-token and target-token epochs. Aggregate steady throughput
was about 267K packed tokens/s. All six K64 DiLoCo merges completed, no error,
OOM, timeout, or nonfinite state was logged, and all routing layers retained zero
unused experts.

This is an attended, operator-reviewed bespoke fixed-world topology. It claims
ADR-003 safety intent R07/NDP15 (atomic complete checkpoint publication),
R12 (explicit verified restart), R14/NDP13 (whole-child fail-stop teardown), and
R16 evidence discipline. It does not claim elastic/native/v2.1 requirements
R02-R06/R08-R11, NDP01-NDP12/NDP14/NDP16/NDP17, V21S01-V21S17, or
ISP01-ISP07, and it is not represented as a rung in the default `8 -> 32 -> 128`
systems ladder.

## Authorities

4,096-update fork parent:

- checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-router-preserved-1e4-4k-k64-8n-v1/checkpoints/step-02342632-tokens-0000282978786792`
- manifest SHA-256: `c2eebf519580e29a2111e3925a1d1bcbac3058c32f1dad5522b92906ccc36a9c`
- cumulative packed tokens / assistant targets: `908,697,064 / 644,996,633`

64-node result:

- checkpoint: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-router-preserved-1e4-4k-k64-64n-epoch2-v1/checkpoints/step-02343016-tokens-0000283660414623`
- manifest SHA-256: `c8a73445e23484758a4988f197a88284ffd882aa2d3fe95a8c1ac77a32535ba8`
- step / sampler cursor: `2343016 / 4480`
- cumulative packed tokens / assistant targets: `1,590,324,895 / 1,128,965,027`
- two optimizer groups; nonrouter group `k=4480`
- source: `1c5c6425bf00581fd382388a93a188242144752e`

## Exact validation

Job `5293060` completed on one node with `Partition=batch`, `QOS=debug`, and
`Requeue=0`, enumerating all 1,777 packs and 4,352,510 targets exactly once:

- target-weighted NLL: `1.137555841483053`
- bootstrap 95% interval: `[1.105090753140539, 1.1690852130967506]`
- pack-mean NLL: `1.1824971400124842`
- unused experts: zero in every layer

This regresses from the best measured 2,816-update exact NLL `1.098255722355298`.

## Behavioral panel

Job `5293290` completed on two nodes with `Partition=batch`, `QOS=debug`, and
`Requeue=0`.

- assistant-response NLL: `1.623386` (2,816-update result was `1.615048`)
- WikiText NLL: `3.023682 / 2.943237 / 2.926025` at 2K/8K/32K
- MMLU: `0.207031`
- HellaSwag normalized: `0.433594`
- RS stops: `9/16` greedy and `10/16` sampled
- recurrent observations: 100% finite
- dropped/unused experts: zero

Termination and response shape improved: several responses are shorter, use
requested list/answer forms, and terminate with RS. Ordinary assistant quality is
still below acceptance: arithmetic, debugging, code, JSON, editing, and factual
answers are frequently incorrect; repetition and constraint violations remain.
The extra epoch did not improve assistant-response NLL and worsened exact held-out
SFT NLL.

## Decision

More unchanged stock-Tülu training is not justified by the dose response. Retain
the 2,816-update checkpoint as the best exact-NLL authority, retain 4,096 and
4,480 as immutable exposure evidence, and next diagnose the supervision/interface
mismatch before any further scarce training. In particular, compare canonical
`Assistant:` generation separately from the unsupervised compatibility
`Assistant reasoning:` entry point and test whether multi-record recurrent
packing without explicit record-boundary state reset is impairing inference.
