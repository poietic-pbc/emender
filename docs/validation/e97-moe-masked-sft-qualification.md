# E97-MoE assistant-masked SFT qualification

Date: 2026-08-15  
Verdict: **qualified through the eight-node K64 rung; authorize one matched
32-node Debug-QoS LR canary split into independent 16-node arms.**

## Architecture disposition

This is the fixed-world ADR-003 production specialization of
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, not elastic/native/async-v2.1
research. Applicable safety requirements are R07/R12 checkpoint and exact
restart, translated R14/NDP13 whole-child fail-stop termination, R16 sequential
systems evidence, and NDP15 synchronous atomic checkpoint publication. Dynamic
R02–R06/R08–R11, NDP01–NDP12/NDP14/NDP16/NDP17, V21S01–V21S17, and
ISP01–ISP07 are explicitly retired/unclaimed for this path. Each job has one
fixed world, `srun --kill-on-bad-exit`, `Requeue=0`, no shrink/relaunch, and no
SQLite, shared-filesystem database, live lock, or metadata heartbeat.

## Immutable data authorities

Tülu token-plus-mask build job `5276678` completed `0:0` in 7m14s with
`Partition=batch`, `QOS=debug`.

- authority manifest: `e2461a28e7699637e282564589aeea35a150dc9e6e3cd62ffd6a2b5c0a2f47d8`
- 939,343 observed input rows;
- 938,586 accepted records and 757 fail-closed empty-content rejects;
- 941,780,868 tokens;
- 598,860,582 assistant-target tokens;
- 933,002,693 train tokens and 593,342,062 train targets;
- 8,778,175 validation tokens and 5,518,520 validation targets;
- independent sampled authority validation: pass.

The deterministic complete-record 4K pack authority is:

`/lustre/orion/bif148/proj-shared/emender/sft/tulu3-v1/packs-4096-v1`

- manifest: `010dc5f326f351463d8af675f2d6a906c8f056e48ee4c93b1ee34dddd1b202ea`;
- independent validation receipt SHA-256:
  `4e12c4c5e6b4f2f4130ea90a40cf3337a8641a2746dd48c13b0a8d7fb1d18a53`;
- 179,110 train packs containing 904,771 complete records;
- 621,085,925 train source tokens and 440,814,770 train targets;
- 1,777 validation packs containing 8,985 complete records;
- every eligible record appears exactly once in the catalog;
- no record is split;
- pack IDs are sampled with replacement by
  `emender-record-pack-counter-v1`.

At 4K, 24,623 long train records are intentionally excluded rather than
truncated or split. They contain 311,916,768 tokens and 152,527,292 targets.
They remain in the immutable token authority for a later long-context SFT
stage; this exclusion is explicit and prevents silently importing truncated
conversations into the assistant-foundation canary.

## Objective and clocks

Source `18c8eae0b820f3cce8d2153b44f8838bf383fb65` implements:

- mmap token, uint8 assistant mask, record index, and pack index loading;
- deterministic global-rank/absolute-cursor counter samples with replacement;
- complete-record padding with exact actual-length accounting;
- next-token assistant masks (`mask[:, 1:]`), including targeted terminal RS;
- chunked CE sum parity against dense PyTorch CE;
- exact node-wide target-token normalization;
- SUM reduction for replicated gradients, matching expert gradients already
  accumulated through differentiable all-to-all;
- target-token-weighted cross-node ScheduleFree x/z averaging at K boundaries;
- separate model accepted-token, SFT total-token, assistant-target-token, and
  per-rank pack cursors in every checkpoint;
- fresh optimizer state from the model-only 282B parent;
- fixed held-out masked-NLL evaluation.

The parent is bound by manifest
`95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1` at
step 2,338,536 and accepted tokens 282,070,089,728.

Local validation command:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_masked_sft_launcher.py \
  tests/test_e97_masked_sft_canary_launcher.py \
  tests/test_masked_sft_dataset.py \
  tests/test_e97_moe.py \
  tests/test_e97_moe_checkpoint.py \
  tests/test_e97_moe_ep_triton.py
```

Result: 48 passed before canary-source publication. Earlier focused sets also
passed 52/52 and 40/40.

## One-node and fresh-process restore

Job `5276974` completed `0:0`, `Partition=batch`, `QOS=debug`, elapsed 12m43s.
It loaded the model-only 282B authority, initialized fresh ScheduleFree state,
ran one 4K masked update, validated, and atomically published:

`step-02338537-tokens-0000282070117446`

- SFT tokens: 27,718;
- assistant targets: 17,797;
- cursor: 1;
- training loss: 1.54912;
- held-out masked NLL: 1.66918;
- peak rank-0 HBM: 46.29GB.

A new allocation/job/process group, job `5277148`, loaded that complete
checkpoint, restored optimizer and all clocks, consumed exactly cursor 1,
advanced to cursor 2, and published:

`step-02338538-tokens-0000282070144744`

It completed `0:0` in 7m32s on `Partition=batch`, `QOS=debug`. Final clocks were
55,016 SFT tokens and 35,298 targets. Held-out masked NLL was 1.66930. This is
R07/R12 exact fresh-process restart evidence; no damaged communicator or
automatic restart was used.

## Eight-node K64 qualification

Job `5277224` completed `0:0` in 21m32s with explicit
`Partition=batch`, `QOS=debug`.

- eight nodes / 64 ranks / eight complete EP islands;
- 64 local updates and one target-weighted corresponding-lane DiLoCo merge;
- merge duration: 50.03s;
- SFT total tokens: 14,175,580;
- assistant-target tokens: 10,096,735;
- exact per-rank cursor: 64;
- mean training masked NLL: 1.69309;
- final held-out masked NLL: 1.72870 on 322,264 targets;
- maximum observed rank-0 HBM: 46.29GB;
- finite loss/router auxiliary throughout;
- final checkpoint publication: 69.07s.

Canonical authority:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/sft-qualification/8n-k64-v1/checkpoints/step-02338600-tokens-0000282084265308`

The final manifest binds the exact parent, data and pack manifests, world 64,
context 4096, sampler key 42, cursor 64, and both token clocks.

## Canary authorization

The passing 1-node -> fresh restore -> 8-node K64 evidence authorizes one
32-node fixed-world Debug allocation split into matched independent 16-node
arms. Both consume identical counter coordinates at world 128 from the 282B
parent. The only scientific difference is LR `2e-6` versus `5e-6`; a configured
checkpoint delay prevents simultaneous canonical publication. Each arm runs
512 updates and eight K64 merges, then publishes one final authority. No
production extension is authorized by training loss alone; final held-out NLL,
generation coherence, stopping, routing, and general-LM evaluation decide the
next action.
