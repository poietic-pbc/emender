# E97 35B MoE: 4-, 8-, and 32-node qualification

## Verdict

PASS. The sequential `1 -> 2 -> 4 -> 8 -> 32` ladder completed. Each accepted
rung contains at least 20 minutes of measured training, finite losses, fused
ScheduleFree optimizer steps, and two K40 cross-node DiLoCo merges. Each node
was one isolated eight-GCD expert-parallel island; only corresponding parameter
and ScheduleFree `x/z` shards used the cross-node lane groups.

| Nodes | Job | Immutable source | Measured s | Steps | Tokens | Mean loss | Median tok/s | K40 merges (s) | Max allocated HBM/GCD |
|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| 4 | 5182895 | `0dfa4215` | 1,207.371 | 81 | 5,308,416 | 2.60014 | 4,639.01 | 14.615, 17.029 | 48.65 GB |
| 8 | 5183128 | `a9e2d542` | 1,211.676 | 81 | 10,616,832 | 2.56924 | 9,611.27 | 28.670, 17.848 | 48.81 GB |
| 32 | 5183619 | `096303da` | 1,201.755 | 80 | 41,943,040 | 2.54960 | 38,953.08 | 24.096, 24.103 | 48.92 GB |

All three accepted jobs ended `COMPLETED`, exit `0:0`. Live `squeue` and
terminal `sacct` independently reported `Partition=batch` and `QOS=debug`.
The 32-node allocation was:

```
frontier[00054,00185,00258,00434,00570,00760,00862,01001,01059,01204,
01343,01468,01545,01711,01893,01998,02148,02338,02523,02735,02876,
02966,03107,03216,03351,03477,03606,03740,03921,03981,04113,04243]
```

## 32-node correction history

Job `5183284` failed after 12 finite steps on a single-node HIP code-209 kernel
load error (`frontier04828`, rank 209). Job `5183306` reached step 39 but RCCL
could not allocate a 32 MiB merge workspace because inactive variable-routing
blocks left 65.34 GB reserved. Commit `096303da` added a CUDA cache release and
cross-node lane barrier immediately before K40 merges. Job `5183619` then
completed both merges and the duration gate. Failed jobs are not counted.

## Durable artifacts

- Ledger: `docs/validation/e97-35b-moe-diloco-ladder.md`
- Runner: `scripts/frontier/e97_35b_moe_train.py`
- Slurm wrapper: `scripts/frontier/e97_35b_moe_train_multinode_20m.sbatch`
- 32-node telemetry:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/32-node-20m-5183619/training.jsonl`
- Raw stdout/stderr: `logs/frontier/e97_moe/e97-35b-moe-32n-20m-5183619.{out,err}`

## Authority conformance

This is fixed-world ADR-003 qualification under
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`. It satisfies the applicable checklist
for R07, R12, R14/NDP13, R16, and atomic checkpointing under NDP15. It does not
claim elastic membership, communicator shrink, async-v2.1 overlap, or expert
traffic across nodes. Atomic sharded publication and fresh-process restoration
were separately passed by jobs `5182549` and `5182648` before scale promotion.
