# E97 4B durable checkpoint transport

The local rolling run retains only three periodic checkpoints. Milestone
checkpoints must first be hard-linked or atomically copied outside that rolling
directory, checksummed, and then published to the private operational Hub
repository:

`spinozans/e97-4b-training-checkpoints`

These are raw pickle-bearing `torch.save` training checkpoints, not portable
model releases. Treat the Hub repository as trusted private storage, pin every
download to a full Hub commit, and verify `SHA256SUMS` before loading.

## Local archive and publish

Use the source commit that actually launched the checkpoint lineage, not the
current checkout by assumption:

```bash
CHECKPOINT=/path/to/run/latest.pt \
SOURCE_COMMIT=<40-character-training-source-commit> \
PUBLISH_CONFIRM=1 \
scripts/publish_e97_4b_durable_checkpoint.sh
```

The publisher:

1. mmap-inspects checkpoint and sampler authority;
2. requires the qualified 4,045,972,080-parameter E97 graph identity;
3. hard-links on the same filesystem, otherwise atomically copies;
4. computes SHA-256;
5. writes immutable metadata and updates a durable `latest` symlink;
6. stages the checkpoint, arguments, checksum, and `LATEST.json`;
7. uploads to the private `spinozans` repository.

Default local locations are:

* `/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_durable_checkpoints`
* `/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_hf_staging`

## Frontier download

On Frontier, use a full Hub commit from the publish receipt:

```bash
REVISION=<40-character-hub-commit> \
REMOTE_DIR=checkpoints/step_011776_tokens_6174015488 \
TRANSFER_MODE=exact \
EXPECTED_WORLD_SIZE=8 \
scripts/frontier/download_e97_4b_durable_checkpoint.sh
```

The downloader sources `activate_emender_frontier.sh`, downloads only the
selected durable, verifies SHA-256, mmap-loads the trusted checkpoint, and
cross-checks step, token clock, model size, sampler schema, and world size.
Private Hub access requires a readable token in the Frontier environment.

## Fixed-world limitation

The local checkpoint counter sampler is fixed to data world size 8. It cannot
be described or launched as an exact resume of the planned Frontier world-256
from-scratch campaign. Exact continuation requires world 8.

A download intended only to inspect or extract weights must say so explicitly:

```bash
TRANSFER_MODE=model-only CONFIRM_MODEL_ONLY=1 ... \
  scripts/frontier/download_e97_4b_durable_checkpoint.sh
```

That command does not launch training. Model-only initialization on world 256
requires a separately reviewed bootstrap implementation and a new sampler
lineage; it must not restore or claim continuity of the world-8 optimizer and
counter cursor.
