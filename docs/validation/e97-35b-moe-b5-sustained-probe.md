# E97 35B MoE sustained B5 probe

## Verdict

FAIL: microbatch five is not sustained-safe with the current packed
ScheduleFree state and variable dropless routing allocator pattern. B4 remains
the production setting.

## Attempts

### Job 5186988 — native expandable allocator

- `Partition=batch`, `QOS=debug`, node `frontier08508`
- immutable source `c1e768aa`
- 18 completed finite steps, 1,474,560 tokens
- median throughput after warmup: 13,865.66 tok/s
- maximum recorded allocated HBM: 60,573,778,944 bytes
- failed on rank 6 requesting another 1.22 GiB with zero physical HBM free
- allocator report: 55.29 GiB allocated and 6.52 GiB reserved/unallocated

### Job 5186998 — cache trim every four steps

- `Partition=batch`, `QOS=debug`, node `frontier09269`
- immutable source `f6fea831`
- 44 completed finite steps, 3,604,480 tokens
- median throughput after warmup: 13,454.05 tok/s
- maximum recorded allocated HBM: 60,629,838,336 bytes
- failed on rank 3 requesting another 1.32 GiB with zero physical HBM free
- allocator report: 52.98 GiB allocated and 8.82 GiB reserved/unallocated

Explicit step-boundary `torch.cuda.empty_cache()` delayed but did not eliminate
the fragmentation cliff because live split blocks inside variable ragged GEMM
segments cannot be released.

### Job 5187147 — asynchronous allocator probe

`backend:cudaMallocAsync` mapped to the ROCm async allocator but produced a GPU
memory-access fault on rank 5 before training. It is not admissible on this
Frontier MI250X/PyTorch stack.

## Conclusion

B5 is about 15% faster than sustained B4 while it runs, but neither native
allocator attempt approached the required 20 minutes. B6 had already failed in
job 5186086. Production remains B4. A larger microbatch requires recovering
persistent HBM (for example node-local sharding of replicated ScheduleFree
state), not relying on allocator tuning.
