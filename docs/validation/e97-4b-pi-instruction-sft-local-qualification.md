# E97 4B Pi instruction SFT local qualification

**Verdict:** accepted for a bounded local update-64 canary.

The Frontier account was paused pending renewal, so the one-node qualification
was adapted to the resident eight RTX 6000 Ada GPUs. This is a local fixed-world
DDP authority and makes no Frontier scheduler or cross-island claim.

## Bound identities

- Source: `44b2fd0218568561e8c3e446aa965ef9e8b10dbc`
- Parent: step 24,448 / 99,723,771,904 tokens
- Parent SHA-256:
  `3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568`
- Local mixture manifest:
  `40d1650459c01756e0772a6058c100029c3b6e3a87aa68947515972ffaa6b1c5`
- Local pack manifest:
  `a8e839fe9a25d3ee8c98c02d504f8b04a8c44d238df20a80f7e762b848121a33`

Local manifest hashes differ from Frontier because the manifests retain build
paths and timestamps. The deterministic payloads match the Frontier authority:
Pi tokens/mask/index, mixed tokens/mask/index, and all three pack files have the
published reference SHA-256 values. Counts are identical: 46,004,079 assistant
targets before 4K filtering and 40,953,254 retained targets in 21,252 packs.

## Local adaptation

- world 8, one eight-rank DDP island;
- B1/rank, context 4,096;
- exact assistant-target-normalized DDP gradient;
- fresh Schedule-Free AdamW, LR `1e-5`, warmup 8;
- pinned-CPU BF16 `z` and second moments, bounded 67,108,864-element GPU staging;
- per-rank NUMA memory binding and Triton cache;
- K8 Schedule-Free `x/z` consensus.

Pinned state initialization allocated 16,183,888,320 host bytes per rank and
completed in 20.57 seconds. Peak reported HBM was 27,238,981,120 bytes allocated
and 27,976,007,680 bytes reserved, safely below 48 GB.

## Result

Run:
`/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_pi_instruction_local/runs/e97-4b-pi-sft-local-q2-44b2fd02`

Eight updates processed 240,315 input tokens and 115,005 assistant targets.
Losses were finite and moved from 1.9503 at update 1 to 1.4010 at update 8;
the eight-update mean was 1.7195. Pre-clip gradient norms were finite. The K8
merge completed in 14.96 seconds.

Checkpoint:

- `checkpoint_agent_sft_u000008_loss_1.7195.pt`
- bytes: 24,276,128,699
- SHA-256:
  `12950070d3b066c0d3e045233d964f6aec2223c10f27ec23e4a00b65f0aafcb6`
- unique parameters: 4,045,972,080
- mmap reload: passed
- optimizer storage: pinned CPU

Training completed successfully (`return-code.txt=0`). The initial post-run
verifier counted tied `embedding.weight` and `lm_head.weight` twice; the loader
and trainer had preserved the tie and correctly counted 4,045,972,080 unique
parameters. The verifier was corrected to deduplicate exact shared-storage
views, and the immutable checkpoint then passed checksum and mmap validation.

This path retains ADR-003 fixed-world safety intent for R07/R12, R14/NDP13,
R16, and NDP15 atomic checkpoint publication. It does not claim elastic,
native-data-plane, async, Frontier scheduler, or eight-island conformance.
