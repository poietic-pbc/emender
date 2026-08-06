# E97 35B MoE one-node 20-minute training — job 5182403

- Source commit: `d97b74c1` (immutable `git archive` execution)
- Seed: final E97 step 2,322,520 / 513,013,841,920 tokens
- Node: `frontier02806`
- Allocation: one node, eight GCD ranks
- Scheduler: `Partition=batch`, `QOS=debug`
- Terminal state: `COMPLETED`, exit `0:0`, elapsed `00:22:52`
- Measured training: `1,203.146 s`
- Optimizer steps: 82
- Accepted tokens: 1,343,488
- Full context: 2,048 tokens, batch 1 per rank
- First loss: 2.171618
- Final loss: 2.860573
- Mean loss: 2.803359
- Median throughput: 1,110.88 tokens/s per node
- Maximum allocated HBM: 48,100,272,128 bytes

Every step used one shared plus 64 routed experts, top-3 routing, eight local
experts per GCD, node-local RCCL assignment/return, fused backward including
router auxiliary gradients, node-only replicated-gradient averaging, and the
fused ScheduleFree update without master weights. No expert assignment crossed
the node boundary. All losses, auxiliary losses, gradients, parameters, and
optimizer states remained finite.

Live scheduler evidence named both fields independently:

```text
JobID=5182403 Partition=batch QOS=debug State=RUNNING NodeList=frontier02806
```

Terminal accounting:

```text
5182403|e97-35b-moe-1n-20m|batch|debug|COMPLETED|0:0|00:22:52|frontier02806
```

Training JSONL:

`/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/one-node-20m-5182403/training.jsonl`

This passes the duration/systems gate. Atomic sharded checkpoint/restart remains
required before a two-node qualification run.
