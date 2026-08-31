# E97 4B Pi instruction SFT preflight

**Status:** one-node qualification submitted

**Date:** 2026-08-31
**Source:** `0c1ae44d` for physical qualification; `dadb7346` adds the subsequent real-Pi evaluation panel

## Bound authorities

- Parent checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-from-scratch/runs/e97-4b-hybrid-ddp-256n-recovery-r3/train/checkpoint_step_024448_loss_2.3981.pt`
- Parent SHA-256:
  `3ace004251643acf2e7c7f720e8f29968ad0a483441553c0c885b87b3df84568`
- Parent step/tokens: 24,448 / 99,723,771,904
- Pi-native authority:
  `/lustre/orion/bif148/proj-shared/emender/sft/pi-native-core-v1`
- Pi-native manifest:
  `c0d8a8934908da87a5624fb976f7872cc6b2ca3eb667892bb3a98126e08f9bae`
- Mixed authority:
  `/lustre/orion/bif148/proj-shared/emender/sft/e97-4b-pi-instruction-mix-v2`
- Mixed manifest:
  `b0caebab3c25d4f700a7e723e8cca24f1213184c85af9286645ca8b731750ec4`
- 4K packs:
  `/lustre/orion/bif148/proj-shared/emender/sft/e97-4b-pi-instruction-mix-v2/packs-4096-v1`
- Pack manifest:
  `eef729241438058f4866030fb397c98447ef7246e22ffa36bd10fba02ddebda7`
- Independent pack validation:
  `19c8a97dfb40c8109f4a7b2ed83e18434491d20a428c0920bd290d8b548ad11b`

The mixed authority contains 46,004,079 assistant targets before 4K complete-record
filtering. The validated packs retain 40,953,254 assistant targets across 21,252
packs. The retained target mixture is approximately 63.5% exact Pi-native and
36.5% Tulu-3 replay, closely matching the planned 65/35 curriculum.

## Objective and topology

The trainer reconstructs the foundation checkpoint's ScheduleFree train/`y`
weights, starts a fresh ScheduleFree AdamW optimizer, and computes assistant-only
loss. Every complete record is forwarded with a clean recurrent state. Tool
observations remain context and never become prediction targets.

One-node qualification configuration:

```text
world size            8
node-local DDP         8 ranks
batch                  B1/rank, B8 island
context                4096
activation checkpoint  enabled, group 1
learning rate          1e-5
warmup                 8 updates
DiLoCo                  K8 bounded x/z merge
target/save update     8
merge bucket           67,108,864 elements
```

Within the DDP island, the trainer compensates for DDP's gradient averaging so
the gradient is normalized by the exact island assistant-target count. Multiple
records inside one pack accumulate under `no_sync`; the final record performs
one node-local DDP synchronization.

## Static validation

The canonical Frontier environment ran:

```text
45 passed
  tests/test_e97_4b_pi_instruction_sft.py
  tests/test_masked_sft_dataset.py
  tests/test_e97_tulu3_sft.py
  tests/test_e97_facade.py
  tests/test_train_helpers.py

27 passed
  tests/test_e97_4b_pi_instruction_sft.py
  tests/test_e97_agent_protocol.py
  tests/test_e97_agent_server.py
```

Python compilation, shell parsing, `git diff --check`, deterministic authority
construction, payload hashing, complete-record packing, and independent pack
validation passed.

## Frontier submission

- Payload job: `5387623`
- Run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-4b-pi-instruction/runs/e97-4b-pi-sft-1n-b1c4k-q1`
- Requested scheduler identity: `Partition=batch`, `QOS=debug`
- Nodes/ranks: 1 / 8
- Walltime: 02:00:00
- Requeue: 0

Queued evidence explicitly reported:

```text
5387623|PENDING|batch|debug|2:00:00|(Priority)
QOS=debug
JobState=PENDING
Requeue=0
TimeLimit=02:00:00
Partition=batch
NumTasks=8
```

## First submission result and correction

Job `5387623` entered `RUNNING` on `frontier08567` with the correct live
`Partition=batch` and `QOS=debug`, then failed in 32 seconds with exit `65:0`
before `srun`, model load, or GPU training. The launcher required the mutable
working checkout's current `HEAD` to equal the already pinned submitted commit.
The repository legitimately advanced while the job was queued, even though the
launcher was prepared to execute `git archive "$SOURCE_COMMIT"`.

The correction removes the mutable-HEAD equality check while retaining
`git cat-file -e "${SOURCE_COMMIT}^{commit}"`, exact `git archive` execution,
and the recorded source identity. A regression test requires both training and
evaluation launchers to accept an older pinned commit from an advanced checkout.
No model checkpoint or `latest.pt` was published by job `5387623`.

Per the attended fail-stop policy, no replacement submission is made until this
diagnosis and corrected immutable source receive operator review.

## Qualification acceptance

The qualification passes when the terminal evidence establishes:

1. `Partition=batch`, `QOS=debug`, and `Requeue=0` in live and accounting evidence;
2. exact immutable source, parent, authority, and pack identities;
3. fused B1/4K forward/backward execution on all eight ranks;
4. finite target-normalized loss and gradients;
5. bounded HBM without OOM;
6. a K8 merge of ScheduleFree `x` and `z`;
7. atomic update-8 checkpoint and `latest.pt`;
8. retained SHA-256 and mmap reload receipt;
9. terminal `COMPLETED 0:0`.

This path claims ADR-003 safety intent for R07/R12, R14/NDP13, R16, and NDP15
checkpoint atomicity. It is fixed-world fail-stop execution and makes no
elastic, native-data-plane, or async-v2.1 claim.
