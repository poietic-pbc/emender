# E97 MoE HIP-209 ragged Triton specialization diagnosis

Date: 2026-08-07

## Finding

The repeated 256-node failures were not evidence for three independently bad
Frontier nodes.  All three direct failures ended in Triton's first-load path
(`CompiledKernel._init_handles -> driver.active.utils.load_binary`) with HIP
error 209 (`hipErrorNoBinaryForGpu`):

| Job | completed local steps | rank / node | first failing kernel |
|---|---:|---|---|
| 5191722 | 117 | 1972 / `frontier09172` | `_local_expert_counts_kernel` |
| 5193991 | 174 | 2004 / `frontier09427` | `_unpack_rows_kernel` |
| 5194968 | 63 | 1612 / `frontier07237` | `_unpack_rows_kernel` |

Job 5192502's node-local RCCL timeout is consistent with one lane becoming
stuck while loading/compiling a specialization, although it did not retain a
direct HIP-209 traceback and is not claimed as proof of that mechanism.

The MoE EP implementation declared the received ragged assignment count
`ROWS` as `tl.constexpr` in four kernels.  `ROWS` changes by rank, layer, and
step.  Each new value therefore compiled and loaded another specialization
into that process's HIP module table.  One rank can accumulate thousands of
modules during a sustained run.  A fixed world of 2,048 ranks fails when any
one process reaches the ROCm/Triton module-loading failure, explaining why the
same code appeared stable in shorter/smaller runs.  Historical dense E97 runs
did not execute these new ragged MoE EP kernels.

This matches the public upstream failure mode documented by PyTorch PR 184285:
long Triton specialization/autotuning sequences can retain HIP modules until
ROCm reports error 209.  The installed Frontier environment is PyTorch
2.10.0+rocm7.1 with Triton 3.6.0 and does not contain the later Triton 3.7.1 /
PyTorch 2.12 module-release work described by that change.

References:

- AMD HIP error codes: `hipErrorNoBinaryForGpu` is code 209.
  <https://rocm.docs.amd.com/projects/HIP/en/develop/reference/error_codes.html>
- PyTorch module-lifetime fix/backport discussion:
  <https://github.com/pytorch/pytorch/pull/184285>
- Upstream intermittent ROCm Triton HIP-209 report:
  <https://github.com/pytorch/pytorch/issues/147838>

## Fix

Source commit `5fd2e2f8` removes the ragged row count from the four Triton
kernel signatures.  Their launch grids already contain exactly one program per
valid row, so the bounds predicates and compile-time row constant were
unnecessary.  Dimension/block constants remain specialized.  This reduces the
four unbounded row-dependent module families to one module each per fixed
dimension.

Upgrading the complete PyTorch/Triton/ROCm production environment was rejected
as the first solution: it changes substantially more qualified code and the
upstream module-release work targets newer PyTorch/Triton branches.  Replacing
Triton packing with eager PyTorch was also rejected because the production MoE
path is explicitly fail-closed against eager GPU fallback.

## Validation

Local structural/oracle suite:

```text
18 passed in 15.43s
```

ROCm parity job **5195279**, `Partition=batch`, `QOS=debug`, exit `0:0`:

```text
tests/test_e97_moe_ep_triton.py: 5 passed in 30.86s
```

ROCm ragged-cardinality job **5195327**, `Partition=batch`, `QOS=debug`, exit
`0:0`, ran six different received row counts (23,003 through 28,573), verified
exact pack/unpack, and observed exactly one in-process compiled cache entry for
each affected kernel:

```text
_local_expert_counts_kernel       caches device 0: 1
_assign_local_packed_rows_kernel  caches device 0: 1
_repack_rows_kernel               caches device 0: 1
_unpack_rows_kernel               caches device 0: 1
```

Bounded machine validation job **5195332** used source `864d8f3c`, exact 320
steps at 256 nodes, K40, `SAVE_EVERY=80`, `Partition=batch`, `QOS=debug`, and
`Requeue=0`.  It passed beyond all prior direct failure points and terminated
`COMPLETED 0:0` in 52:51.  It retained all eight K40 merges (22.82--28.10 s),
four atomic checkpoints (67.40--98.13 s), 5,368,709,120 newly accepted tokens,
and final authority:

```text
step=2323760
accepted_tokens=10547625984
checkpoint=step-02323760-tokens-0000010547625984
```

No HIP-209, Triton-load, nonfinite, HBM, or collective error occurred.  Peak
allocated HBM was 53,582,800,896 bytes/GCD.  This is a direct machine pass of
the proposed root-cause fix, not merely a shorter-than-failure smoke.

The measured 320-step timing predicted 960 steps would slightly exceed two
hours after staging/model restore.  The bounded sustained successor therefore
used exact 880 steps, 22 K40 merges, `SAVE_EVERY=80`, and a two-hour debug-QoS
safety envelope rather than guessing 960.

Sustained job **5195870** used the same tested code source `864d8f3c`,
`Partition=batch`, `QOS=debug`, 256 nodes/2,048 ranks, and `Requeue=0`.  It
terminated `COMPLETED 0:0` after 1:45:03:

```text
steps=880
new_accepted_tokens=14763950080
merges=22
checkpoints=11
final_step=2324640
final_accepted_tokens=25311576064
final_checkpoint=step-02324640-tokens-0000025311576064
```

All 22 merges completed in 20.60--31.66 s (median 23.33 s); checkpoints took
64.09--94.11 s (median 71.26 s).  Plain-step median was 3.319 s and recent
median throughput was 5.07M token/s.  Peak allocated HBM was 54,722,904,576
bytes/GCD.  Whole-run mean language-model loss was 2.18483 (perplexity 8.889),
last-100 mean was 2.17900 (perplexity 8.837), and last-100 mean auxiliary loss
was 0.01157.  The finite batch-loss range was 1.8393--2.5841.

Across both post-fix machine jobs, one fresh 2,048-rank world completed 320
steps and another completed 880: 1,200 steps, 30 merges, 15 checkpoints, and
20,132,659,200 newly accepted tokens without HIP-209 or a Triton load failure.
The reduction in JIT specialization also removed ongoing compile overhead:
ordinary throughput rose from roughly 3.6--3.8M token/s in the failing jobs to
about 5.0M token/s sustained.  This independent performance effect strongly
corroborates the module-proliferation diagnosis.

## Architecture conformance

Authority is `RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs (2026-07-31 decision), with the
`RESILIENT_DILOCO_GAP_MATRIX.md` production crosswalk.

Applicable safety intent: **R07**, **R12**, **R14/NDP13**, **R16**, and
**NDP15** checkpoint atomicity.  The fixed-world child remains fail-stop; this
change does not preserve, shrink, or automatically relaunch a broken
communicator.  Checkpoints remain atomic canonical eight-shard authority and
a human-approved fresh job is the only recovery path.  The explicit operator
review in this session authorizes the bounded 256-node diagnosis/validation.

Explicitly retired and unclaimed for this ADR-003 work: **R02-R06,
R08-R11; NDP01, NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and
ISP01-ISP07**.  No elastic membership, native data plane, asynchronous overlap,
background checkpoint, or communicator-shrink claim is made.  The rendered
production compute role adds no SQLite/database/lock/metadata-heartbeat path.
