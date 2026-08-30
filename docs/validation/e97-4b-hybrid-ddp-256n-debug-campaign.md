# E97 4B 256-node hybrid-DDP debug campaign

**Status:** operator-authorized after successful 96-node phases through step
20,224 / 28,856,811,520 tokens. Pending 96-node phase 3 payload `5373274`
and collector `5373275` were cancelled without running. The operator accepted
the debug-QoS policy tradeoff to obtain a scientific answer before allocation
expiry. After subsequent 96- and 256-node debug submissions remained pending,
the operator made a final `256 nodes or bust` decision with a 48-hour deadline:
one 256-node, eight-hour, normal-QoS payload will run all remaining 4,224
updates from the accepted parent to step 24,448 / 99,723,771,904 tokens. This
removes the extra canary queue cycle. The expected runtime is approximately
seven hours; the eight-hour request retains final-checkpoint margin. The
pending alternatives were cancelled before running and consumed no compute.

The first final payload `5374260` entered its 256-node allocation but failed
closed after eight seconds, before `srun`, because the new campaign config
omitted legacy top-level keys read unconditionally by the shared payload:
`bootstrap_smoke_steps` and `debug_continuation_train_minutes` (the
`production_train_minutes` key was already present). It performed zero updates
and left the accepted parent untouched; collector `5374261` correctly rejected
the unchanged 768-rank checkpoint. The corrected config supplies all 18 loader
fields, and both pytest plus an exact extraction smoke validate the schema
before the explicitly authorized replacement submission.

Corrected payload `5381420` then trained healthily for 3h15m through stream
step 22,012 and merge 55 before Slurm declared `NODE_FAIL` due explicitly to
`frontier02966`. No model, optimizer, OOM, nonfinite, or distributed-training
failure preceded the infrastructure loss. Collector `5381421` completed `0:0`
and validated the latest durable 2,048-rank checkpoint at step 21,760 /
54,626,615,296 tokens, loss 2.4504, SHA-256
`245b8dfaa55a40ea67ad457809c3cbb0345f3cf779df277597d6e991bfbb20e2`.
The operator explicitly authorized one same-world recovery from that authority.
It must not repeat the counter transition. Remaining work is 2,688 updates /
45,097,156,608 tokens; measured prior throughput projects approximately 4.4h
training plus startup/finalization, so the recovery requests six hours on 256
nodes at normal QoS.

## Parent authority

- checkpoint: `e97-4b-hybrid-ddp-96n-debug-s02/train/checkpoint_step_020224_loss_2.7239.pt`;
- SHA-256: `74bae676f8a73b509f52851c8735fe07572fbe574de286e817200fb9a855e584`;
- step/tokens/world: 20,224 / 28,856,811,520 / 768 ranks;
- final last-100 loss: 2.5823;
- payload/collector `5369350`/`5369351`: `COMPLETED 0:0`;
- 32/32 K32 merges, atomic checkpoint, SHA receipt, and mmap reload passed.

## Topology and transition

The scale step uses 256 nodes / 2,048 ranks with one eight-rank DDP island per
node, B4/rank = effective B32/island, K32, context 2,048, and the validated LR
`0.00047431158698290157`. It preserves the competent local ScheduleFree
trajectory while increasing outer-average island count from 96 to 256.
Counter-v2 makes one explicit world transition at the accepted-token boundary
28,856,811,520. Subsequent epochs are exact same-world continuation.

Each optimizer update accepts 16,777,216 aggregate tokens. The remaining
70,866,960,384 tokens equal exactly 4,224 updates.

| Phase | Updates | Target step | Target tokens |
|---|---:|---:|---:|
| c1 | 768 | 20,992 | 41,741,713,408 |
| c2 | 768 | 21,760 | 54,626,615,296 |
| c3 | 768 | 22,528 | 67,511,517,184 |
| c4 | 768 | 23,296 | 80,396,419,072 |
| c5 | 768 | 24,064 | 93,281,320,960 |
| c6 | 384 | 24,448 | 99,723,771,904 |

All targets are K32-aligned; full phases are also save256-aligned. The phase table remains the exact token arithmetic and checkpoint schedule,
but the final operator override executes c1--c6 continuously in one eight-hour
normal-QoS allocation. Periodic save256 checkpoints retain recovery points at
every listed full-phase boundary; the final target is K32-aligned.

## Gate

Require live and terminal `Partition=batch` and `QOS=debug`, `Requeue=0`,
payload and collector `COMPLETED/0:0`, all expected K32 merges and 2,048-rank
finalization evidence, finite losses/gradients, no OOM/timeout/distributed/data
failure, HBM evidence, K-aligned atomic `latest.pt`, SHA-256 receipt, and mmap
reload. A hard failure stops fresh submission. Rank-zero loss is supporting
evidence only; fixed-panel evaluation remains required for scientific topology
promotion.

This is ADR-003 fixed-world execution. Applicable safety intent is R07, R12,
R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic, async-v2.1, ISP, and
native-dataplane conformance remain explicitly unclaimed.
