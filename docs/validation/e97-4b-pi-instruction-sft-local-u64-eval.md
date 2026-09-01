# E97 4B Pi SFT local update-64 evaluation

**Verdict:** training/checkpoint accepted; behavioral promotion rejected; continue the bounded tune to update 256.

## Checkpoint

- update: 64
- input tokens: 1,958,186
- assistant targets: 994,695
- loss window: 0.685563
- bytes: 24,276,128,699
- SHA-256: `73cf78b9bea0111ef1ac2eeab9ee732873793fc4a58921ca3ee3ec4f7f9e0c17`
- unique parameters: 4,045,972,080
- mmap reload: passed
- optimizer storage: pinned CPU

All 56 continuation updates were finite. The update-64 K8 merge completed in
15.71 seconds. Peak HBM remained below the 48 GB device bound.

## Frozen real-Pi panel

Evaluation source: `3594f4354986c7beb37975d264c4c772873341a9`

Evaluation root:
`/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/evals/e97-4b-pi-core-u64-3594f435-r5`

The run used eight independent CUDA servers, the real installed Pi agent loop,
the hash-pinned Apptainer core-tool image, and 120 held-out disposable repository
tasks. It completed with return code zero.

| Measure | Result | Gate | Verdict |
|---|---:|---:|---|
| Fully passing conversations | 87/120 (72.5%) | reported | — |
| Protocol-valid conversations | 88/120 (73.3%) | 100% | fail |
| Schema-valid tool calls | 120/120 (100%) | >=99.5% | pass |
| Exact tool sequence | 108/120 (90.0%) | diagnostic | — |
| Exact tool arguments | 107/120 (89.2%) | >=95% | fail |
| Grounded clean final | 88/120 (73.3%) | >=95% | fail |
| Sandbox postcondition | 120/120 (100%) | 100% | pass |
| No identical-call loop | 108/120 (90.0%) | 100% | fail |

By kind:

- read: 20/20;
- bash: 19/20;
- write: 20/20;
- edit: 9/20;
- recover-read: 19/20;
- recover-test: 0/20.

The dominant failure is specific rather than general tool incompetence. The
model usually performs the requested mutation and verification correctly—every
sandbox reached its required postcondition—but after a successful empty-output
verification it may repeat that same command instead of issuing the final. The
server rejects the second identical call as a cycle. Longer recover-test
transcripts showed this on all 20 examples; shorter write trajectories did not.
One recovery-read example repeated the known-missing path. One direct bash task
chose `read` instead of the required direct `bash` call while still grounding
its answer.

The evaluator also uncovered two harness defects before this authoritative run:
its direct-bash expected command was not reconstructed, and unconstrained final
text could continue into transcript-like material. Both were corrected and
covered by tests. Canonical serving now stops one-line finals at their first
newline; action calls still stop only after a complete JSON object.

Update 64 is therefore useful evidence but is not promotable. Continuing the
same frozen sampler and optimizer to updates 128 and 256 is authorized. Selection
remains behavioral: the next checkpoint must reduce post-verification loops and
produce clean finals without regressing exact tools or sandbox invariants.
