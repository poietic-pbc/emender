# E97-MoE long-context study plan

Status: execution path qualified through 32 Frontier nodes; production study not
launched.

Qualification receipt:
`docs/validation/e97-moe-long-context-qualification.md`.

## Scientific question

Compare continued adaptation of the frozen trained E97-MoE asset at:

1. 32,768 tokens with full 32K BPTT;
2. 131,072 tokens with literal full 128K BPTT;
3. a 32K-to-128K curriculum.

A separately frozen 128K-forward/32K-TBPTT arm remains an operational fallback.

The architecture, tensor schema, expert topology, recurrence, tokenizer, corpus,
and RS handling remain unchanged.

The primary 128K arm is literal full 128K BPTT. The fallback must always be
described as **128K forward context / 32K gradient horizon**, never as full
128K BPTT.

## Parent authority

Use only the complete trained checkpoint:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/
  scale-ladder-mainfc52/checkpoints/
  step-02338080-tokens-0000250797359104
```

Parent step: `2,338,080`  
Parent accepted tokens: `250,797,359,104`

Every arm starts a new counter-v2 phase at this exact accepted-token boundary.
The arm-specific context and launch world are part of its new identity.

## Frozen execution controls

| Control | 32K arm | 128K arm |
|---|---:|---:|
| context | 32,768 | 131,072 |
| batch/GCD | 1 | 1 |
| loss chunk | 2,048 | 2,048 |
| projection chunk | 2,048 | 2,048 |
| recurrence checkpoint interval | 16 | 16 |
| layer activation checkpointing | enabled | enabled, one 11-block group |
| bounded materialization segment | disabled | 32,768, states attached |
| forward state horizon | 32K | 128K |
| gradient horizon | 32K | 128K |
| checkpointed loss chunks | optional | enabled |
| MoE execution bound | full 32K | full 32K segment |
| ScheduleFree state CPU offload | disabled | enabled |
| cache trim | every step | every step and pre-backward |
| expert backend | rocBLAS | rocBLAS |
| RS behavior | visible token 218, no reset | visible token 218, no reset |

Recommended first study world: 32 nodes / 256 GCDs.

- 32K global tokens/step: 8,388,608
- 128K global tokens/step: 33,554,432
- 32K DiLoCo: K2
- 128K DiLoCo: K1

These K choices keep optimizer drift bounded while avoiding a merge more often
than every 32K–128K tokens per rank.

## Learning-rate study

Do not inherit the pretraining LR accidentally. Checkpoint restore also restores
optimizer param groups. Use `--resume-lr-override` explicitly and verify the
`restart_loaded.learning_rates` receipt.

Run a short 32-node comparison at:

- `1e-4`
- `2e-4`

Use 8 optimizer steps per context. Stop an arm for non-finite values, abrupt
loss increase, routing collapse, or sustained gradient/memory failure. Select
one LR before allocating adaptation tokens. A lower 128K LR may be selected if
its four-times-larger token batch responds differently.

## Matched comparisons

A context-length comparison is confounded if it matches only steps or only
wall time. Report both:

1. **Matched accepted tokens** — primary scientific comparison.
2. **Matched optimizer steps** — short diagnostic comparison.
3. **Matched node-hours** — operational efficiency comparison.

Suggested first matched-token tranche: 1B accepted tokens per direct arm.
At 32 nodes this is approximately:

- 32K: 120 steps;
- 128K: 30 steps.

Do not commit to a larger tranche until evaluation shows a useful long-context
signal.

## Arms

### A. Direct 32K

Start from the parent trained checkpoint and transition to a new 32K/world-256
counter identity. Train full 32K BPTT.

### B. Direct 128K

Start independently from the same parent and transition to a new
128K/world-256 counter identity. Build one full-BPTT graph from four bounded
32K materialization segments without detaching recurrent states.

### C. Curriculum

Start from the selected direct-32K checkpoint, then create another explicit
counter phase for 128K at that checkpoint's exact accepted-token boundary.
Never reuse or relabel the direct-128K stream.

## Evaluation

Evaluate each parent/intermediate checkpoint at identical held-out windows:

- 2K and 8K short-context regression;
- 32K validation loss;
- 128K validation loss;
- RS-separated multi-record continuation;
- passkey/retrieval distances distributed across 8K, 16K, 32K, 64K, and 128K;
- router load, entropy, maximum probability, and expert imbalance by position;
- recurrent-state finite/max-absolute diagnostics by layer and position.

Report loss by position bucket, not only one mean. The key question is whether
128K forward state improves distant-token prediction without damaging short
context.

## Instruction tuning after adaptation

Long-context SFT should begin only after selecting a base arm.

- Keep RS as the visible example/conversation separator.
- Mask prompt/user tokens and train assistant tokens only.
- Use 32K as the common SFT window.
- Include a smaller set of genuine 128K examples using the qualified bounded
  materialization / full-128K-BPTT execution path.
- Use a new explicit sampler/data identity and a lower LR than adaptation.
- Preserve the selected long-context checkpoint as an immutable parent.

## Operator runbook

1. Confirm no other Debug-QoS job is active.
2. Verify `HEAD` and freeze an immutable source commit.
3. Verify the parent `latest` generation and manifest hashes.
4. Choose the exact node count before constructing sampler identity.
5. Set counter-v2 origin to the parent checkpoint's accepted tokens.
6. Set `SAMPLER_TRANSITION_FROM_COUNTER=1` only for the first epoch of a new
   phase.
7. Set `RESUME_LR_OVERRIDE` explicitly.
8. Launch with `scripts/frontier/e97_35b_moe_long_context_debug.sbatch` for
   qualification only. A production launcher must additionally save K-aligned
   canonical checkpoints.
9. Require `restart_loaded`, `data_stream_ready`, `phase_profile`, `step`, and
   scheduler receipts.
10. After completion, verify accepted-token increments:
    `context × batch × world` per optimizer step.
11. Never continue a phase with a different world/context identity without a
    new explicit counter transition.

## Remaining production work

Before a multi-billion-token run:

- add these controls to the fixed-world production launcher;
- save a canonical K-aligned checkpoint from the new phase;
- perform a fresh-process restore at the same context/world;
- verify explicit LR persistence across that restore;
- freeze evaluation inputs and token budget;
- recalculate Frontier allocation after all MoE/post-training charges.
