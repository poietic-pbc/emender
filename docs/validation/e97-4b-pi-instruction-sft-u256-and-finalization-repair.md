# E97 4B Pi SFT update-256 comparison and repair decision

**Decision:** stop the broad mixed-curriculum branch at update 256; retain update
64 as the behavioral parent; run a bounded final-only repair with a fresh
Schedule-Free optimizer.

## Training completion

The mixed Pi/Tulu branch completed 256 finite updates:

- input tokens: 7,841,473;
- assistant targets: 3,955,698;
- update-256 loss window: 0.624470;
- final K8 merge: 16.09 seconds;
- update-256 SHA-256:
  `b3dfb2954510cc543599994057add0c57212d160a29e3fd9700c2a9a9793db44`;
- mmap reload and 4,045,972,080-parameter validation: passed.

Retained checkpoint hashes:

| Update | Loss window | SHA-256 |
|---:|---:|---|
| 128 | 0.5882 | `54f5831a7c73db6d3cfbab0a77d2a81c9a5c65825cd8e363a3f8ae4a35067368` |
| 192 | 0.5778 | `08aefb3ad1fd9d2cd27af6ab058e10743a09c0e163b22d8329a7aef1b3308cce` |
| 256 | 0.6245 | `b3dfb2954510cc543599994057add0c57212d160a29e3fd9700c2a9a9793db44` |

## Behavioral selection

Every retained checkpoint ran through the same 120-task real-Pi panel.

| Update | Full pass | Protocol/grounded final | Exact arguments | No cycle | Sandbox |
|---:|---:|---:|---:|---:|---:|
| 64 | **87/120** | **88/120** | **107/120** | **108/120** | **120/120** |
| 128 | 79/120 | 80/120 | 99/120 | 100/120 | 116/120 |
| 192 | 78/120 | 81/120 | 98/120 | 101/120 | 120/120 |
| 256 | 79/120 | 82/120 | 99/120 | 102/120 | 120/120 |

Updates 128--256 did not improve the promotion bottleneck. Recovery-test stayed
0/20, edit fell from 9/20 at update 64 to 3--5/20, and direct bash also
regressed. This confirms the plan's requirement to select behaviorally rather
than by training loss. More undifferentiated mixed-curriculum updates are not
authorized.

## Root-cause-specific repair

The real Pi OpenAI-completions adapter serializes a successful tool with empty
text as `(no tool output)`. The original Pi-native SFT authority instead trained
that context as an empty `Tool:` body. The dominant observed failure occurs
exactly after successful empty-output verification: the model repeats the
verification call instead of transitioning to `Final:`. Longer recover-test
histories expose the mismatch on 20/20 held-out tasks.

A derivative immutable repair authority therefore:

1. reconstructs every source trajectory and checks its original serialization;
2. changes only empty successful tool context to `(no tool output)`;
3. makes all prior assistant actions context-only;
4. targets only the evidence-grounded terminal `Final:`;
5. also targets one terminal newline, matching canonical one-line serving.

Repair authority:

- source: local Pi-native authority
  `48f6b7ecb0083f09402e2f0715b95d7ca71ba45a2711375b811472ccdeb804e1`;
- records: 20,000 (19,810 train / 190 validation);
- input tokens: 5,571,273;
- terminal targets: 712,760;
- authority SHA-256:
  `887120163f3531e98f9606bdbf2d02ff171d4811c60df77c9d3881c247056129`;
- 4K packs: 1,405 train / 14 validation;
- pack SHA-256:
  `7307fe07fcbd957c7ee27e23037f4c7d3a57de63e5f2f4b3a0ea1b0ec2b015a0`;
- complete pack validation: passed for 1,419 packs / 20,000 records.

The repair starts from the update-64 saved `x` weights, not the behaviorally
worse update-256 endpoint. It deliberately starts a fresh Schedule-Free
optimizer at LR `5e-6`; old moments encode the superseded mixed curriculum.
The bounded first repair was 64 updates with K8 checkpoints at repair updates
32 and 64. Real-Pi evaluation rejected both: terminal completion improved to
119/120, but action-only masking caused premature finalization and degraded tool
execution. Update 32 reached 62/120 full passes and update 64 reached 52/120.
An eight-update micro-repair had the same failure mode (75/120).

## Balanced live-aligned repair and promotion

The accepted repair retains the exact live `(no tool output)` context but targets
**all** assistant actions as well as the terminal final/newline. Its immutable
authority has 20,000 records, 5,571,273 input tokens, and 2,400,225 assistant
target tokens:

- authority SHA-256:
  `4a1cf86f9089cc3f2f79884f845d26d487d88b20fb32b589a8095d4824bc6a20`;
- 1,419 completely validated 4K packs;
- pack-manifest SHA-256:
  `e4c24641cbbe1f70c84c48fcb5fdd815bb6aa6a67c285a2c08c3a4cd2cf2eaa0`.

Eight updates from the update-64 saved `x` weights used a fresh Schedule-Free
optimizer, LR `5e-6`, warmup 8, and K8 synchronization. The checkpoint contains
4,045,972,080 unique parameters, passed mmap reload validation, and has SHA-256
`b799802741737058c4de74e233a8af8e6a9a18977cbf753ad44f9037a27c3da8`.

The first balanced evaluation exposed a serving false positive: the cycle guard
rejected a focused test rerun after an intervening edit. A failed check followed
by repair and the same verification is progress, not a cycle. Commit `95f50c29`
therefore rejects only immediately consecutive identical actions; the server
and protocol suite passes 24/24 tests. This does not relax the no-progress guard:
an unchanged missing-path read repeated immediately is still rejected.

The corrected, fixed 120-task real-Pi panel accepted the checkpoint:

| Metric | Result |
|---|---:|
| Full task pass | **119/120** |
| Protocol-valid / grounded final | **119/120** |
| Schema-valid calls | **120/120** |
| Exact tool sequence | **120/120** |
| Exact tool arguments | **119/120** |
| Sandbox postcondition | **120/120** |
| No identical-call cycle | **119/120** |

Bash, edit, read, write, and recover-test each passed 20/20; recover-read passed
19/20. The sole failure immediately repeated the same missing-path read and was
correctly stopped by the cycle guard. This checkpoint is behaviorally promoted;
none of the lower-loss broad or final-only endpoints is promoted.
