---
license: other
tags:
- emender
- e97
- schedule-free
- diloco
- training-checkpoint
---

# Emender E97 4B exact training checkpoints

Private operational repository for durable, checksummed training checkpoints of
the 4,045,972,080-parameter E97 base campaign.

These are raw `torch.save` training checkpoints. They contain model and
Schedule-Free optimizer state and use Python pickle. Load only from this trusted
repository and verify the adjacent `SHA256SUMS` first. They are not portable
model releases.

`LATEST.json` identifies the newest published durable. Each checkpoint directory
contains its immutable metadata, original training arguments, and checksum.

Exact sampler continuation is fixed to the checkpoint's recorded data world.
In particular, a world-8 checkpoint cannot exactly resume the planned Frontier
world-256 campaign. It may only be used as an explicitly reviewed model-only
bootstrap under a new sampler lineage, or for an exact world-8 continuation.
