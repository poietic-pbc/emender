# True resilient E97 implementation progress, 2026-07-16

Task: `implement-true-resilient`

Commit `1ceeaf9` replaces sentinel accounting at the new live-gate boundary
with an explicit topology contract and launcher: exactly two CPU-only managers
and sixteen independently supervised, GPU-bound real trainer steps (eight per
physical node).  Every trainer command receives `ASYNC_LOCAL_STEPS=40`.
Managers and trainers use separate `srun --overlap --no-kill --exact` steps,
so there is no allocation-wide process group or all-rank collective in the
supervision boundary.  The legacy sentinel lane remains tracked solely for
historical runners and is neither invoked nor counted by the new launcher.

The launcher is pinned to two nodes, debug QoS, `02:00:00`, and `TERM@300`.
It fails closed unless distinct manager and approved E97 trainer commands are
provided.  No live job was submitted from this commit because the manager and
trainer role executables are not yet connected to the recovered transport;
submitting placeholder commands would violate the implementation gate and
would repeat an unproven payload.

Focused validation in the established ROCm environment:

```text
python -m pytest -q tests/test_resilient_e97_topology.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_node_transport.py tests/test_resilient_node_quorum.py \
  tests/test_resilient_e97_rank_lane.py
26 passed in 19.18s
```

Python compileall, `bash -n` for the new and shared launchers, and
`git diff --check` also passed.

The generation-9 seed was independently revalidated after implementation:

- path: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z/async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`
- size: `15439252298`
- SHA256: `ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`

This report is deliberately a progress record, not a claim that the mandatory
survivable-failure and fresh-allocation restart gates passed.
