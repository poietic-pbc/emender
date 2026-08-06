# E97 35B MoE one-node fused-kernel smoke — job 5181432

- Date: 2026-08-06 UTC
- Source commit: `eb536d33ea3aa43d76a3edb434840029207523a4`
- Branch: `feature/e97-35b-moe-triton`
- Submission: `scripts/frontier/e97_35b_moe_1n_kernel_smoke.sbatch`
- Node: `frontier02813`
- Allocation: 1 node, 8 tasks, 1 Slurm-bound GCD per task
- Fused ABI: `emender-e97-moe-triton-v2`

## Scheduler binding

Live scheduler evidence while the job was running:

```text
5181432|batch|debug|RUNNING|frontier02813|e97-35b-moe-kernel
JobId=5181432
QOS=debug
JobState=RUNNING
Reason=None
Partition=batch
NodeList=frontier02813
```

Terminal accounting evidence:

```text
JobIDRaw|JobName|Partition|QOS|State|ExitCode|Elapsed|NodeList
5181432|e97-35b-moe-kernel|batch|debug|COMPLETED|0:0|00:01:05|frontier02813
5181432.batch|batch|||COMPLETED|0:0|00:01:05|frontier02813
5181432.extern|extern|||COMPLETED|0:0|00:01:05|frontier02813
5181432.0|bash|||COMPLETED|0:0|00:00:59|frontier02813
```

This explicitly verifies `Partition=batch` and `QOS=debug` independently.

## Result

Every rank ran the real ROCm/Triton kernel suite on its assigned GCD:

```text
rank 0: 25 passed in 51.84s
rank 1: 25 passed in 51.84s
rank 2: 25 passed in 52.29s
rank 3: 25 passed in 51.84s
rank 4: 25 passed in 52.29s
rank 5: 25 passed in 52.30s
rank 6: 25 passed in 52.29s
rank 7: 25 passed in 51.84s
```

The suite exercised fused shared+routed forward, fused expert/input/router and
auxiliary-loss backward, custom autograd, and the fused ScheduleFree AdamW
tensor update. It also exercised CPU structural conversion and the explicit
E97 checkpoint/generation facade. Each rank used an isolated Triton cache.

## Scope and remaining gate

This was a kernel smoke, not production-checkpoint parity, training, or a
performance measurement. The eight ranks independently exercised their bound
GCDs; this job did not claim node-local expert all-to-all. Production training
remains fail-closed until eight-GCD RCCL expert dispatch/return, shared-gradient
reduction, packed local-expert storage, and sharded conversion/checkpointing are
complete.

Durable logs:

- `logs/frontier/e97_moe/e97-35b-moe-kernel-5181432.out`
- `logs/frontier/e97_moe/e97-35b-moe-kernel-5181432.err`
