# E97 7.94B Schedule-Free CPU-offload qualification

Date: 2026-08-21

## Scope and verdict

This receipt qualifies the local one-GPU optimizer-storage mechanism for the
square-readout E97 candidate `d=4608`, `L=25`, `H=96`, `n=48`, SwiGLU ratio
2.5. The instantiated graph has **7,940,197,056 trainable parameters**.

The mechanism retains Schedule-Free AdamW. BF16 model parameters and gradients
remain on the GPU; BF16 `z` and `exp_avg_sq` remain in pinned CPU memory and
are streamed through bounded GPU buckets for GPU arithmetic. There are no FP32
master weights and no CPU Adam update.

**Verdict:** the implementation, exact-size one-GPU 2,048-context capacity
probe, checkpoint continuation, and two-GPU fixed-world DiLoCo average path
pass. This is a local
systems qualification, not authorization for a long training run or a
Frontier scale rung.

## Architecture boundary

The governing authority is
[`RESILIENT_DILOCO_COMPUTE_POOL.md`](../RESILIENT_DILOCO_COMPUTE_POOL.md),
ADR-003, with the production crosswalk in
[`RESILIENT_DILOCO_GAP_MATRIX.md`](../RESILIENT_DILOCO_GAP_MATRIX.md).
Applicable safety intent is R07/NDP15 checkpoint atomicity, R12 optimizer-state
restart, and R16 evidence discipline. This local fixed-world work does not
claim R02–R06/R08–R11 dynamic-pool semantics, NDP02 native no-all-rank
semantics, NDP17 scale promotion, V21S01–V21S17, or ISP01–ISP07. No scheduler,
allocation, queue, membership, or production recovery behavior changed.

## Implementation

- `ndm/schedulefree_offload.py`
  - initializes state directly in pinned CPU memory, including a fresh run;
  - preallocates state before the first forward/backward capacity boundary;
  - streams fixed-size `z`/second-moment buckets through the GPU;
  - releases completed gradients;
  - preserves Schedule-Free train/eval basis transforms;
  - restores checkpoint state without PyTorch's normal whole-state GPU cast.
- `train.py`
  - exposes `--offload_schedulefree_state` and bucket/pinning controls;
  - reports logical transfer volume and optimizer-update duration;
  - uses direct CPU-FP32 to GPU-BF16 model conversion, avoiding a transient
    whole-model FP32 GPU copy;
  - streams offloaded Schedule-Free `z` through bounded GPU DiLoCo buckets;
  - supports the fixed-world `avg` outer optimizer and fails closed on the
    not-yet-implemented offloaded-z translation required by `momentum`/`sfsgd`.
- `scripts/probe_e97_8b_schedulefree_offload.sh` reproduces the exact-size
  capacity probe.

## Exact-size one-GPU evidence

Command:

```bash
DATA=/tmp/emender-sf-offload-e2e/data.txt \
GPU=0 CHUNK_SIZE=64 \
OUTPUT=/tmp/emender-e97-8b-offload-steady64 \
scripts/probe_e97_8b_schedulefree_offload.sh \
  --probe_optimizer_steps 2
```

Observed on one RTX 6000 Ada 48 GB:

| Measurement | Result |
|---|---:|
| Parameters | 7,940,197,056 |
| Model baseline HBM | 15,144.7 MiB |
| After forward | 15,215.2 MiB |
| After backward | 30,305.7 MiB |
| Peak allocated HBM | 31,189.6 MiB |
| Pinned CPU optimizer state | 29.580 GiB |
| Logical H2D per update | 29.580 GiB |
| Logical D2H per update | 29.580 GiB |
| One-time state initialization | 28.610 s |
| First measured streamed update | 3.977 s |
| Second measured streamed update | 4.001 s |

This short-context optimizer-storage microprobe predated the SiLU fused-path
guard correction described below; none of its recurrence timing is used as
throughput evidence. The two measured updates show that the former
tensor-at-a-time host fencing is not in the steady path. The fixed
67,108,864-element staging bound adds less
than 1 GiB above the post-backward allocation. Approximately 17 GiB of the
48 GiB device remains outside the observed allocated peak at context 64.

### Full 2,048-context fused-path capacity and throughput

The reproduction command below now pins the full optimized-path contract,
including `--gate_activation silu`; `train.py` fails closed if a BF16 E97
configuration would instead select eager recurrence.

```bash
DATA=/tmp/emender-sf-offload-e2e/data.txt \
GPU=0 BATCH_SIZE=4 CHUNK_SIZE=2048 \
CHECKPOINT_INTERVAL=64 \
PROJECTION_CHUNK_SIZE=512 \
LOSS_CHUNK_SIZE=256 \
OUTPUT=/tmp/emender-e97-8b-offload-fused-b4-2048 \
scripts/probe_e97_8b_schedulefree_offload.sh
```

The runtime identified the executed implementation as
`e97-sequential-split-edit-triton`, `state=tanh`, and `eager_fallback=False`.
Batch 4, 8, and 12 each completed a forward, backward, and GPU-executed
streamed Schedule-Free update:

| Measurement | Batch 4 | Batch 8 | Batch 12 |
|---|---:|---:|---:|
| Tokens/update | 8,192 | 16,384 | 24,576 |
| Forward | 2.883 s | 4.708 s | 5.734 s |
| Backward | 7.249 s | 12.763 s | 18.453 s |
| Optimizer | 3.960 s | 3.986 s | 4.005 s |
| Total | 14.092 s | 21.457 s | 28.192 s |
| Throughput/GPU | 581.3 tok/s | 763.6 tok/s | 871.7 tok/s |
| Compute-only throughput | 808.5 tok/s | 937.8 tok/s | 1,016.1 tok/s |
| After forward | 23,195.6 MiB | 31,169.4 MiB | 39,149.4 MiB |
| After backward | 30,377.0 MiB | 30,374.8 MiB | 30,374.8 MiB |
| Peak allocated HBM | 31,626.6 MiB | 33,357.5 MiB | 39,787.6 MiB |

The model baseline is 15,213.6 MiB and pinned optimizer state is 29.580 GiB.
Thus batch 12 retains about 8.9 GiB of physical HBM headroom. Batch 16 with the
same projection/loss chunks reached the loss computation but failed closed on
a 786 MiB allocation with only 198.69 MiB free; the recurrence itself did not
fail. A second batch-16 attempt with projection chunks reduced from 512 to 256
and loss chunks reduced from 256 to 64 also reached cross-entropy, then failed
a 198 MiB allocation with 16.69 MiB free. The additional projection checkpoint
boundaries increased retained pressure, so this is negative rather than passing
capacity evidence. Batch 14 with the proven 512 projection chunk and a smaller
128-token loss chunk likewise reached cross-entropy, then failed a 344 MiB
allocation with 244.69 MiB free. Batch 12 is therefore the qualified capacity
boundary for these 48 GB devices. Eight independent
learners would retain about 236.64 GiB of optimizer state in a host with
approximately 1 TiB RAM; capacity is sufficient, while NUMA-local placement
remains required for sustained throughput.

Batch 12 clears the provisional 600 tokens/s/GPU gate and projects to about
6,974 tokens/s across eight independent learners. An idealized 160B-token pass
at that rate is approximately 266 days, before checkpoint, data, startup,
failure, and evaluation overhead. The preferred 1,200 tokens/s/GPU target is
not yet met. This single-update capacity result therefore warrants sustained
and factorial measurements; it does not by itself authorize a long seed.

### Rejected eager-path diagnostic and guard correction

Initial 2,048-context probes inherited `train.py`'s default
`--gate_activation sigmoid`. Although they set `--use_triton 1`, the model's
actual optimized-path predicate requires a SiLU output gate. Sigmoid therefore
silently selected the eager time-step scan while the former preflight guard
incorrectly printed `NO eager fallback`. The resulting batch-1, batch-4, and
batch-8 rates (13.34, 46.05, and 57.74 tokens/s/GPU) are rejected
configuration diagnostics and must not be used as E97 throughput evidence.

The probe now passes `--use_gate 1 --gate_activation silu --linear_state 0`
explicitly. `e97_fused_runtime_contract_failures` checks Triton, gate presence,
exact SiLU activation, write-gate absence, and value-residual absence; BF16 E97
training raises before model execution if any clause fails. Focused regression
coverage exercises every rejected clause. A direct isolated-recurrence check
also measured the square `n=48` recurrence at roughly 2--2.5x the corresponding
`n=32` recurrence, not the 40--100x end-to-end anomaly. The primary anomaly was
the accidental eager path, not Schedule-Free offload or ordinary `n=48`
scaling.

## Fresh checkpoint and continuation

A two-step BF16 E97 fresh run with full state offload completed, atomically
published `latest.pt`, and a new process resumed it at step 2 / 128 accepted
tokens and completed step 3 / 192 accepted tokens. The restore path retained
optimizer state on CPU rather than transiently casting it to the parameter
GPU.

This exercise exposed and fixed an independent retention bug: periodic and
final checkpoints at the same step but different loss strings could cause
`keep_checkpoints=1` to delete the newly published `latest.pt` target. The
retention path now always retains the checkpoint published by the current
call, and a regression test resolves `latest.pt` strictly.

## Two-GPU fixed-world DiLoCo evidence

A real two-rank NCCL run completed two independent optimizer steps and two K=1
merges with:

```text
--offload_schedulefree_state
--diloco --diloco_k 1
--diloco_outer_optimizer avg
--diloco_merge_bucket_numel 65536
```

Both Schedule-Free `x` and CPU-resident `z` reached consensus through bounded
GPU staging. Rank 0 atomically wrote a resolvable final `latest.pt`. Unit
coverage separately forces tiny 31-element optimizer/merge buckets and checks
model and `z` consensus across two Gloo ranks.

## Tests

```bash
python -m py_compile train.py ndm/schedulefree_offload.py
python -m pytest -q \
  tests/test_schedulefree_cpu_offload.py \
  tests/test_checkpoint_finalization.py \
  tests/test_diloco_merge.py \
  tests/test_train_helpers.py
```

Result: **50 passed**. The tests cover reference Schedule-Free trajectory parity,
BF16 CUDA pinned state, state preinitialization, checkpoint reload without a
GPU state cast, bounded-bucket DiLoCo consensus, and same-step checkpoint
retention.

## Remaining gates

- Benchmark an ordinary multi-step training loop so optimizer cost is reported
  together with forward/backward throughput.
- Add NUMA-local worker binding before an eight-GPU run; GPUs 0–3 are local to
  NUMA node 0 and GPUs 4–7 to node 1.
- Only the fixed-world `avg` DiLoCo outer path is currently admitted with CPU
  offload. Momentum and outer Schedule-Free remain fail closed.
- Behavioral scaling, data/token budget, and long-run stability are separate
  model-quality decisions.
