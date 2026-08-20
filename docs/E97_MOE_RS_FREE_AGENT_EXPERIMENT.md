# E97 MoE RS-Free Agent Experiment

Status: active attended experiment  
Decision authority: real Pi behavioral rollouts, not loss  
Dense control: direct-CLI champion `fca9b06c521fb9407cd7bed2d7049f36e9a93948d6cd98899b6ad9b9e4fc6b01`

## Hypothesis

The earlier MoE assistant studies never received the complete protocol change that unlocked the dense agent. They used RS as an assistant terminator and did not execute coherent RS-free CLI conversations through real Pi. Record-reset SFT fixed state contamination between unrelated examples but was not equivalent to removing RS inside one continuing tool task.

Test whether the 282.070B MoE authority can learn and execute the current RS-free direct-CLI policy. The MoE may benefit from 35B parameters of conditional storage and 3.14B active parameters/token, while its unchanged shared recurrent state may remain a limitation for unfamiliar help-driven discovery.

## Immutable inputs

- MoE parent: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/long-context-32k-from-64k-step2338090/milestones/step-02338536-tokens-0000282070089728`
- Parent manifest: `95828109b7082fde427712cad2e81574571058f1411dd08dcd6cf3016e37b0f1`
- Dense graph seed: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/milestones/step-2322520-tokens-513013841920/checkpoint_step_2322520_loss_2.2798.pt`
- Direct curriculum: `/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-cli-direct-v3`
- Direct packs: `/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-cli-direct-v3-packs-4096`
- Real-Pi task authority: `/lustre/orion/bif148/proj-shared/emender/sft/dense-agent-cli-v1`
- CLI image SHA-256: `bb85bf69530c9b30515acc4c1946a0402309d1c629222f03fdcc2d2df5f133ee`

Authority and pack hashes must be read from and pinned to their manifests before submission.

## Protocol requirements

1. RS is never serialized between coherent user, assistant, tool, and follow-up turns.
2. Generation stops after one complete structured action.
3. An emitted legacy RS is not a successful completion and is never committed as an accepted assistant turn.
4. Recurrent state resets only between unrelated tasks or records.
5. Pi owns tool execution, grounding, cycle detection, and termination.
6. All MoE model calls use the fused fail-closed eight-way node-local EP path.
7. All ranks execute the same collective schedule; completed generation is padded or collectively stopped.
8. Every training gradient is normalized by exact assistant target count.

## Ladder

### G0: serving implementation and CPU checks

Implement a rank-zero HTTP coordinator and seven EP workers. Qualify command broadcast, exact cache lineage, shadow generation, structured-action stopping, and fail-closed worker behavior.

### G1: one-node parent baseline

Run the unmodified 282.070B parent through real Pi using the RS-free direct system. Begin with four balanced direct tasks, then expand only if the server is stable. Record exact argv, grounding, malformed actions, repetition, rank health, and scheduler `Partition=batch` plus `QOS=debug` evidence.

### G2: one-node tiny overfit

Build or select eight isolated RS-free CLI conversations. Train only as a mechanics qualification. Require exact real-Pi reproduction. K1/model averaging, if used, remains qualification-only and cannot become a promotion authority.

### G3: synchronized direct-policy training

Use 8 nodes initially and 16 only when measured throughput or curriculum coverage justifies it. Use corresponding-lane synchronized gradients, not K1 model averaging. Save immutable checkpoints at bounded target exposures corresponding approximately to 8, 32, and 64 updates. Do not continue automatically after a failed behavioral checkpoint.

### G4: behavioral comparison

Evaluate through real Pi on:

- frozen direct 40;
- a powered direct regression panel;
- randomized held-out CLI dialects;
- command discovery;
- option discovery;
- recovery after nonzero exit;
- grounded finalization;
- cycle and repetition protection;
- sandbox invariants.

Compare by exact assistant-target exposure and real outcomes. Report loss only as a diagnostic.

## Stop rules

Stop and diagnose before resubmission if:

- any EP rank misses or reorders a collective;
- recurrent replay and cached execution disagree unexpectedly;
- the tiny overfit cannot reproduce its examples;
- direct behavior does not emerge at bounded checkpoints;
- training improves likelihood while real-Pi behavior remains flat or worsens;
- grounding or sandbox invariants regress;
- scheduler partition/QoS evidence is absent or incorrect.

## Promotion rule

The dense direct checkpoint remains champion unless the MoE passes a powered real-Pi panel with no credible direct, grounding, safety, or general-language regression. Unfamiliar CLI discovery is an independent research metric and is not inferred from direct-policy success.
