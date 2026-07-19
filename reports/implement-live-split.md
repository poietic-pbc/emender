# Live split-role native dense wiring: implementation record

Date: 2026-07-19

Authority: Resilient DiLoCo Compute Pool v1. This pass checked R01-R16 and
NDP01-NDP17. No Slurm job was submitted.

## Implemented and validated in this pass

- `NativeManagerSession` now transfers frozen owner frames only through
  `NativeTransport.send` and receives them only through
  `NativeTransport.receive`. A current installed route and a nonzero 32-byte
  result root are mandatory.
- The retained-frame identity is `(worker_id, result_root)`. The initial send
  and at most two owner-loss replays are permitted; a fourth send fails closed.
  Route removal therefore prevents replay to stale membership.
- Commit continues to validate run, fence, generation, attempt, layout digest,
  base digest, result root, global weight, result bytes, publication-manifest
  digest, and authoritative-latest identity before native state is released.
- The regression `test_frozen_owner_transfer_uses_native_fabric_and_bounds_replay`
  directly proves that frozen bytes invoke the native fabric wrapper, that
  receive authenticates the installed route, and that late/unfrozen and excess
  replay traffic is rejected.

These changes advance R03-R08, R10-R11, R14-R15 and NDP01, NDP06-NDP13,
NDP15-NDP16. They do not weaken R01-R02, R09, R12-R13, or R16/NDP02-NDP05,
NDP14/NDP17.

## Remaining nonconformance

The objective is not complete. The live launcher still constructs
`LocalTrainerSpool` and `DistributedOwnerServer`, and the compiled local ABI
still implements a process-local singleton rather than its advertised
manager/trainer seqpacket boundary. Consequently NDP01, NDP04, NDP12, and
NDP14 remain gaps in the live path, and R08/R10 plus the downstream NDP17 gate
cannot be claimed. Native selection must remain fail-closed until producer
memfd transfer, accepted-set execution, redistribution, and fenced checkpoint
handoff are connected across the actual split processes.

## Validation commands

```text
python3.11 -m py_compile ndm/native_pool_runtime.py tests/test_native_pool_integration.py
ctest --test-dir build/native-resilient-dataplane-build --output-on-failure
ctest --test-dir build/native-resilient-dataplane-asan --output-on-failure
```

Python 3.11 has no installed pytest module in this environment; the system
pytest is tied to unsupported Python 3.6. The focused regression is therefore
committed but could not be executed here.
