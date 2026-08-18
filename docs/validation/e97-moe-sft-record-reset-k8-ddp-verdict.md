# E97-MoE recurrent-reset SFT and DDP verdict

Record-reset K8 training job `5299516` completed 64 updates and job `5300323`
continued through cumulative reset update 192. Every unrelated packed record
ran from clean recurrent state. K8 produced canonical checkpoints at reset
updates 16, 32, 48, 64, 128, and 192. The retained late authorities are reset
128 (`step-02342760-tokens-0000283007226491`) and reset 192
(`step-02342824-tokens-0000283021425838`).

Behavioral jobs `5299908` and `5302104` show that reset training improves some
likelihood and termination properties but does not produce reliable semantic
behavior. From reset 64 to reset 128, WikiText NLL improved by 0.0044--0.0069
across evaluated contexts. From reset 64 to reset 192, paired mean
assistant-response NLL improved by `-0.02629`, bootstrap
`[-0.04298, -0.01309]`, while WikiText improved by 0.0022--0.0032. Greedy RS
stops remained 7/16 at reset 64 and 192; sampled RS stops increased from 3/16
to 11/16. HellaSwag and the answer-prior-sensitive MMLU panel did not move
reliably.

Qualitative generations remain factually wrong, instruction-violating, and
frequently repetitive. Better stopping is not equivalent to better modeling.
Reset 128 is the conservative general-LM authority; reset 192 has the strongest
paired assistant-response likelihood and sampled stopping evidence, but neither
is a reliable assistant.

A corresponding-shard DDP control was implemented in `dcee190e`. Job `5302098`
completed 16 synchronized optimizer updates with exact global target weighting,
node-local EP, no model averaging, and checkpoints at updates 8 and 16. DDP was
systems-correct but slow (roughly 54--86 seconds/update). Behavioral job
`5302541` found no improvement: DDP-16 assistant NLL was 1.6342 versus 1.6251
at its reset-64 parent, greedy RS stops were 4/16 versus 7/16, and WikiText NLL
regressed slightly but consistently from DDP-8 to DDP-16. Further DDP SFT is
not supported.

This production path conforms to ADR-003 fixed-world safety intent R07/R12,
R14/NDP13, R16, and checkpoint atomicity from NDP15. Elastic R02--R06/R08--R11,
native NDP02/NDP17, and V21S/ISP requirements are intentionally unclaimed and
retired for this path.

## Decision

Stop both unchanged Tulu record-reset K8 continuation and DDP continuation.
The reset fixed a genuine state-distribution mismatch, but the remaining
assistant failure is not explained by formatting or model-averaging cadence.
The next bounded diagnostic should test whether the pipeline can deliberately
overfit and reproduce a tiny isolated conversation set before changing corpus,
LR, or launching another large training run.
