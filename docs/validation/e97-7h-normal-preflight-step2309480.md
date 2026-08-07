# E97 seven-hour normal-QOS continuation preflight

Date: 2026-08-03

The final two-hour embedded-clock chaining job `5146135` completed `0:0` in one 256-node execution epoch. The stable production authority now resolves as follows:

- `RUN_ID=e97-final-seed-production-256n`
- `RUN_DIR=/lustre/orion/bif148/proj-shared/emender/frontier_runs/final-seed-production-256n/runs/e97-final-seed-production-256n`
- checkpoint: `checkpoint_step_2309480_loss_2.2845.pt`
- step: `2309480`
- top-level and metadata `total_tokens`: `294238945280`
- checkpoint bytes: `7719680180`
- authority: embedded checkpoint

This publication adds a fail-closed seven-hour `Partition=batch`, `QOS=normal` submitter while retaining the accepted fixed-world K40, save-every-200, keep-two, 64M-bucket hierarchical RCCL data plane and same-allocation recovery. The existing two-hour debug submitter remains unchanged. The shared payload envelope now validates explicit expected partition, QOS, and time limit, defaulting to the previously accepted batch/debug/two-hour binding.

The seven-hour allocation has an eight-hour child-epoch timeout, 480-minute training ceiling, and the existing 15-minute walltime-finalization margin. At measured production throughput it is expected to finish near 500B total tokens.

## Validation

This is the same ADR-003 fixed-world production path and conforms to R07/R12/R14/R16 and NDP13/NDP15: synchronous atomic checkpoint/token authority, stable restart identity, bounded termination/requeue, explicit scale evidence, and no background checkpoint or independently advancing token service.
