# Resilient E97 bootstrap boundary after job 5027064

## Preserved result and changed payload rationale

Job `5027064` is terminal with zero finalized generations. Its immutable report
is `reports/frontier/run-resilient-e97-2-smoke-5027064-20260718.md`; no retry is
active. The pinned Python 3.12.13 / torch 2.10.0+rocm7.1 / HIP 7.1.25424 /
Triton 3.6.0 identity passed and all two managers plus sixteen GPU trainers
launched. Every trainer completed seed verification/load and entered the E97
model construction path, but the old telemetry began only inside the first
optimizer step. Consequently it could not distinguish model construction,
device transfer, model-state restore, bf16 conversion, ScheduleFree optimizer
construction/state restore, or data-iterator construction.

The changed payload emits node-local timestamps and liveness heartbeats at
each of those bootstrap boundaries. These are diagnostic liveness records, not
committed-generation progress and cannot satisfy the generation gate. The
900-second generation/progress deadline remains unchanged. A subsequent smoke
must use the shortest request that safely contains startup, the 900-second
fail-fast, and TERM finalization; the rejected 20-minute admission is not
restored.

## Exact known-good comparison

The retained job-5000436 command uses the same real E97 dimensions, batch size
4, chunk size 2048, bf16, ScheduleFree optimizer, K=40, pinned step-1525000
checkpoint, and CommaPile input. The split-role launcher uses one trainer per
GPU with `ROCR_VISIBLE_DEVICES=<local rank>` and `cuda:0` inside the isolated
process, which is equivalent to the known-good `--gpus-per-task=1
--gpu-bind=closest ... --device cuda:0` mapping. It intentionally differs in
the resilient topology and transport: sixteen trainers plus two model-free
managers, dynamic 6/8 node quorum, bounded point-to-point/node-local bulk
transport, and no launched-rank collective. Failure injection remains disabled
for the startup smoke. There is no synthetic model, synthetic data, control
mode, or alternate optimizer.

## Validation and conformance

Focused command (approved runtime):

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python -m pytest -q tests/test_train_helpers.py tests/test_resilient_e97_true_2n_launcher.py tests/test_resilient_e97_runtime.py tests/test_resilient_node_transport.py
```

Result: `47 passed in 111.41s`.

Checked against version 1 of `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and R03,
R06, R09, R10, R14, and R16. The node-local telemetry improves bounded boot
diagnosis without changing fenced math, membership, transport, or checkpoint
authority. R07, R11, R12, and R16 remain unpassed until live immutable
generation, failure/rejoin, TERM checkpoint, and fresh-allocation continuation
evidence exists.
