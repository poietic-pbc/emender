# Resilient E97 2-node foundation gate

Task: `complete-resilient-e97`

## Retained job 5000436 diagnosis

The exact last authoritative boundary was generation 9, step 1525400, loss
2.4184.  The immutable model checkpoint is 15,439,252,298 bytes with SHA-256
`ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`.
All 2048 ranks subsequently entered generation 10 and wrote
`collective_reduce_complete`; none wrote `collective_reduce_reduced` or
`return_written`.  The strongest supported root cause is therefore a stall
immediately after the all-rank compiled Cray MPICH reduction returned and
before its result was consumed or returned.  Retained evidence cannot further
separate MPI progress/finalization from helper post-collective logic.  The
blocking all-rank boundary prevented generation-10 publication.

## Implementation and control validation

- Audited commits `5786eb1`, `6367b01`, and `e8bd90a`.
- Payload commits: `11d8c66`, `43853ca`.
- The transport is framed point-to-point TCP and imports no MPI/RCCL/TCPStore.
- Added atomic generation-manifest publication, finite-value and conflicting
  duplicate rejection, monotonic generation deadlines, heartbeat eviction,
  fenced payload-identity apply acknowledgements, and generation-bounded disk
  replay pruning.
- Focused command (known Frontier Python 3.12 / torch 2.10 ROCm environment):
  `python -m pytest -q tests/test_resilient_node_quorum.py tests/test_resilient_node_transport.py`
- Result: 17 passed.
- The retained generation-9 checkpoint passed `torch.load(..., mmap=True)` and
  exposed checkpoint metadata, model, optimizer, loss, and step state.

## Live attempt 5000818

Submitted at 2026-07-15T08:21:35Z using only debug QoS:

```text
sbatch -N 2 -q debug -t 02:00:00 -J resilient-e97-2n-20260715T082135Z \
  --export=...,ASYNC_QUORUM_TRANSPORT=resilient-node-quorum-sharded-p2p,\
ASYNC_TRAINPY_RANKS=16,ASYNC_EXPECTED_RANKS=16,ASYNC_GLOBAL_QUORUM=15,\
DILOCO_K=40,ASYNC_LOCAL_STEPS=40,ASYNC_GENERATIONS=8,... \
  scripts/frontier/trainpy_async_quorum_2n_smoke.sbatch
```

Scheduler identity: job `5000818`, partition `batch`, QoS `debug`, two nodes,
16 GPU ranks, time limit exactly `02:00:00`.  Its input is the verified
generation-9 handoff above and code/payload commit `43853ca`.

No normal-QoS or production job was submitted, modified, or cancelled.
