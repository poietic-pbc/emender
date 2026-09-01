# EMENDER E97 4B Pi instruction-tuning plan

**Status:** first-phase historical plan; broad continuation is governed by
[`EMENDER_E97_4B_BROAD_POSTTRAINING_PLAN.md`](EMENDER_E97_4B_BROAD_POSTTRAINING_PLAN.md)

**Created:** 2026-08-31

**Parent authority:** E97 4B step 24,448 / 99,723,771,904 accepted tokens

**Parent SHA-256:** `3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568`
**Operational Hugging Face repository:** <https://huggingface.co/spinozans/emender-e97-4b-pi-instruction-checkpoints>

## Objective

Produce a full-parameter instruction-tuned E97 4B model that operates as a
bounded coding and tool agent inside Pi. The first promotion target is reliable
repository inspection, exact file mutation, command and test execution, ordinary
error recovery, and evidence-grounded completion through Pi's core `read`,
`bash`, `edit`, and `write` tools.

Pi remains responsible for the conversation loop, tool execution, sandbox,
grounding, cycle detection, and termination. The model learns the exact
model-side action protocol and how to select and sequence those tools.

## Frozen model and execution identities

The lineage begins from:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/
  e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-256n-recovery-r3/train/
  checkpoint_step_024448_loss_2.3981.pt
```

The checkpoint is an E97 recurrent foundation with:

- 4,045,972,080 trainable parameters;
- residual width 3,840;
- 18 layers and 60 heads;
- recurrent state side 64;
- MLP ratio 2.5;
- tokenizer `p50k_base`;
- saved ScheduleFree `x` weights and a complete optimizer state.

Initial SFT reconstructs the parent's ScheduleFree train/`y` weights from the
trusted optimizer checkpoint, then starts a fresh SFT optimizer. Published agent
checkpoints store and evaluate the new optimizer's saved/`x` weights. Every
checkpoint binds the parent digest, source commit, data manifests, sampler
identity, target-token clock, topology, and optimizer configuration.

## Pi contract

The first authority pins a compact Pi system contract and the exact argument
shapes used by the E97 server bridge:

```text
Action: read
Arguments: {"path":"...","offset":1,"limit":200}

Action: bash
Arguments: {"command":"..."}

Action: edit
Arguments: {"path":"...","oldText":"...","newText":"..."}

Action: write
Arguments: {"path":"...","content":"..."}
```

A tool turn contains exactly one complete action. A completed task ends with a
natural-language `Final:` response grounded in observed tool output. Tool
observations retain the text produced by Pi. There is no record separator inside
a conversation, and unrelated records begin from clean recurrent state.

The initial task curriculum covers:

1. bounded file reads and exact-value extraction;
2. repository search, Git inspection, and command execution;
3. configuration and manifest interpretation;
4. exact edits followed by focused tests;
5. new-file creation followed by validation;
6. recovery from a missing path, nonzero command, failed test, or stale edit;
7. concise final summaries naming the changed files and commands run.

## Data authority

The first immutable mixture is constructed by assistant target tokens:

| Component | Target share | Purpose |
|---|---:|---|
| Pi-native verified trajectories | 65% | Exact tool selection, arguments, observations, recovery, and completion |
| Tulu-3 masked instruction replay | 35% | General instruction following, explanations, and code-language retention |

Pi-native trajectories are mechanically generated or execution-verified. Clean
successes and successful recovery traces are retained. For a failed rollout, a
correct next action may be attached to the exact prefix at the first divergence.
System, user, and tool-result tokens are context only. Assistant reasoning,
action objects, and final responses are targets.

Authorities are immutable binary token/mask/index objects with SHA-256 manifests.
Complete records are packed without splitting. Each pack preserves record
boundaries so the recurrent state is reset between unrelated examples.

## Training recipe

### Qualification

The first physical run is a one-node, eight-rank B1/4K masked-SFT memory and
kernel qualification. It verifies:

- exact parent train/`y` recovery;
- fused E97 execution;
- B1 at 4,096-token context with activation checkpointing;
- node-local eight-rank DDP;
- assistant-target normalization;
- finite loss and gradients;
- atomic checkpoint write, SHA-256, and mmap reload.

### Eight-node canary

After qualification, the first behavior canary uses:

```text
nodes                         8
ranks                         64
node-local DDP island         8 ranks
batch                         1 pack/rank
context                       4,096 tokens
cross-island outer average    K8
optimizer                     fresh ScheduleFree AdamW
learning rate                 1.0e-5
betas                         (0.9, 0.95)
weight decay                  0.01
gradient clip                 1.0
warmup                        8 updates
canary                         32 or 64 updates
```

Within each node, DDP produces one B8-equivalent optimizer trajectory. Across
the eight node-local islands, bounded ScheduleFree `x` and `z` averaging occurs
only at K-aligned boundaries. Each configured merge bucket genuinely bounds HBM.

### Bounded first tune

An accepted canary continues to 256 updates with checkpoints at updates 32, 64,
128, and 256. At 64 ranks and 4,096-token packs, 256 updates expose approximately
67.1 million input tokens. Checkpoint selection is behavioral rather than
loss-minimizing.

## Evaluation and promotion

All promotion evidence is produced through real Pi conversations against frozen
disposable repositories. The fixed panel contains:

- direct one-tool protocol tasks;
- multi-turn read and repository-inspection tasks;
- ordinary error-recovery tasks;
- edit/write/test tasks in disposable worktrees;
- no-tool instruction tasks;
- cycle, path, and sandbox invariants.

The first promotion gates are:

| Measure | Gate |
|---|---:|
| Protocol-valid conversations | 100% |
| Schema-valid tool calls | at least 99.5% |
| Correct direct tool and arguments | at least 95% |
| Grounded final responses | at least 95% |
| Read-only repository task success | at least 80% |
| Recovery task success | at least 75% |
| Repeated identical-call loops | 0 |
| Sandbox and path invariant passes | 100% |

Patch/test success is reported separately until a sufficiently broad fixed panel
exists. Updates 32, 64, 128, and 256 are all eligible for selection; the last
checkpoint is not promoted automatically.

## Durable publication

The private operational repository
`spinozans/emender-e97-4b-pi-instruction-checkpoints` retains trusted raw
`torch.save` checkpoints and their optimizer state. Each checkpoint directory
contains:

- the raw checkpoint;
- `SHA256SUMS`;
- parent, source, data, sampler, and topology metadata;
- scheduler `Partition` and `QOS` evidence;
- terminal accounting;
- mmap reload validation;
- fixed-panel evaluation receipts;
- a promotion verdict.

Raw checkpoints are pickle-bearing operational artifacts. Consumers verify the
immutable Hugging Face revision and SHA-256 before loading. A future public
release will be a separately curated weights-only BF16 safetensors repository
with tokenizer, loader, model card, provenance, checksums, and behavioral scope.

## Execution sequence

1. Pin the compact Pi contract and build the deterministic Pi-native authority.
2. Construct the target-token-weighted Pi/Tulu mixture and 4K complete-record packs.
3. Validate serialization, target masks, RS exclusion, payload hashes, and sampler determinism.
4. Implement and test the E97 4B masked-SFT trainer and fail-closed Frontier launcher.
5. Commit and push an immutable source snapshot.
6. Run the one-node B1/4K qualification on `Partition=batch`, `QOS=debug`.
7. Inspect terminal scheduler, checkpoint, hash, reload, loss, gradient, and HBM evidence.
8. Submit the eight-node canary only after the qualification verdict.
9. Evaluate every retained checkpoint through the frozen real-Pi panel.
10. Upload complete checkpoint receipts to the private Hugging Face repository and record immutable revisions.

## Resilient-training traceability

This SFT path uses fixed-world fail-stop execution and atomic K-aligned
checkpoints. It claims the ADR-003 safety intent of R07/R12, R14/NDP13, R16, and
NDP15 checkpoint atomicity from `RESILIENT_DILOCO_GAP_MATRIX.md`. It does not
claim elastic, native data-plane, or async-v2.1 conformance. A rank, collective,
node, or child failure terminates the job; diagnosis and operator review precede
any replacement submission.
