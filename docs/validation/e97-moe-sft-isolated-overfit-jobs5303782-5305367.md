# E97-MoE isolated-conversation overfit probe

Job `5303782` trained from reset-128 on one immutable pack containing eight
Alpaca-eval conversations (898 packed tokens, 676 assistant targets), with a
fresh optimizer and recurrent reset at every record. The diagnostic authority
manifest is `692d3ea0ec879621509f170154896f71965aaa987a5e2351e83a62dba62676e8`;
the pack manifest is
`7ec9f8e8d672e5d61117b1548e38d43edadde86782a8e10157888bc84ddbf411`.
This is diagnostic data and not a production authority.

The run completed 64 one-node updates under `Partition=batch`, `QOS=debug`,
with terminal `ExitCode=0:0`. Loss fell from 1.7471 on the first update to
0.0292 by update 9, 0.00822 at update 33, and 0.00636 at update 64. Canonical
diagnostic checkpoints were retained at updates 32 and 64.

Job `5305367` evaluated the reset-128 parent and both probe checkpoints using
the exact eight training prompts. The parent reproduced 0/8, stopped on 0/8,
and had mean normalized string similarity 0.043. Both update 32 and update 64
reproduced all eight references exactly, including terminal RS behavior: 8/8
exact, 8/8 stopped, similarity 1.0.

This is strong evidence that masked targets, recurrent record resets,
optimization, sharded checkpoint publication/loading, cached decoding, and RS
stopping can jointly learn and reproduce isolated conversations. It rejects a
fundamental train/decode pipeline failure as the explanation for the broad SFT
lineage's unreliable assistant behavior.

It does not establish broad generalization. On the 120 paired-panel assistant
responses excluded from training, update 32 worsened token-weighted NLL by
+0.0159 and update 64 by +0.0227. Paired unweighted deltas were +0.0675 and
+0.0885 respectively. Exact memorization therefore coexisted with negative
held-out transfer. The next experiment must optimize diversity and data
mixture, retain truly excluded behavioral validation, and stop before tiny-set
overfit.

The existing Tulu mixture is dominated by WildChat and long synthetic math:
WildChat contributes 200.8M assistant targets and PersonaHub math contributes
160.5M of 598.9M total. A bounded focused-mixture arm should retain cleaner
general-instruction sources (No Robots, OASST1, FLAN, and manual IFData), use
unchanged LR/objective/reset/K8 mechanics, and compare against reset-128 on
held-out assistant likelihood and generations before any larger continuation.
