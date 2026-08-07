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

The next bounded machine validation is one exact 320-step 256-node epoch with
K40, `SAVE_EVERY=80`, `Partition=batch`, `QOS=debug`, and `Requeue=0`.  It must
pass beyond all three prior direct failure points, retain eight K40 merges and
four atomic checkpoints, and exit zero before a longer epoch is authorized.

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
