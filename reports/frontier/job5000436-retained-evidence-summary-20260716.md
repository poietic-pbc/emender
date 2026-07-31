# Retained-evidence summary: Frontier job 5000436

This is a read-only summary requested on 2026-07-16. No Slurm job was
submitted, modified, or cancelled while producing it. Scheduler timestamps
below are America/New_York; UTC equivalents are included where operationally
relevant.

- Allocation `5000436` started `2026-07-15 02:45:14 EDT`, ended
  `03:27:57 EDT`, and ran `00:42:43`. It was manually cancelled once with
  `scancel 5000436` at `2026-07-15T07:27:57Z`; final state was
  `CANCELLED by 19032`.
- Ten generations (0--9), equivalently ten merge/finalization events,
  finalized. Their publication epochs were `1784098399`, `1784098611`,
  `1784098826`, `1784099040`, `1784099255`, `1784099470`, `1784099684`,
  `1784099898`, `1784100112`, and `1784100327`. Intervals were 212--215 s,
  averaging 214.2 s/generation, with no cadence degradation.
- The final valid state was generation 9, step 1,525,400, loss 2.4184, at
  `async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`
  beneath run root
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z`.
  It is 15,439,252,298 bytes and SHA256
  `ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`.
- Cancellation did **not** create a new shutdown checkpoint. The termination
  wrapper only preserved `continuation/last-valid.json` (via immutable
  `last-valid-20260715T072807Z.json`) pointing to the already atomically
  published generation-9 checkpoint; `latest.pt` and `latest.json` remained
  generation 9.
- Expected and observed participation was 256 nodes and 2,048 ranks. All ten
  finalized generations accepted 2,048/2,048 updates. All 2,048 heartbeat
  files subsequently entered generation 10. Retained evidence shows no rank
  or node dropout, `NODE_FAIL`, OOM, nonfinite/NaN signal, or pre-cancellation
  nonzero exit. Signal/nonzero exits coincide only with manual cancellation:
  allocation `0:0` (derived `0:15`), compute step `CANCELLED 0:15`, batch
  `FAILED 143:0`, extern `COMPLETED 0:0`.
- Exact stall boundary: all 2,048 helper traces contain
  `collective_reduce_complete`, but none contains
  `collective_reduce_reduced` or `return_written`. Thus every rank crossed the
  compiled helper's collective call, while no rank consumed/published the
  result. The strongest supported root cause is a hang between helper
  collective completion and reduced-result return/publication. Retained
  evidence cannot distinguish helper post-collective logic from MPI progress
  or finalization, but it proves the all-rank boundary prevented generation-10
  publication.
- Relevant evidence/code commits are `5786eb1`, `6367b01`, `e8bd90a`, the
  cancellation/evidence commit `1bfddce`, and this task's resilient foundation
  sequence through `4a2ff18` (including `11d8c66`, `43853ca`, `df46fc6`,
  `954b68d`, `0133bff`, and `0bbf237`). At capture start, local HEAD
  `4a2ff18` matched its upstream. The only untracked paths were the retained
  stdout/stderr of already-terminal debug job 5009365; no tracked source
  change was dirty.

Primary retained sources: `reports/cancel-stalled-debug-5000436.md` and
`reports/report-live-progress-5000436-20260715T073655Z.md`.
