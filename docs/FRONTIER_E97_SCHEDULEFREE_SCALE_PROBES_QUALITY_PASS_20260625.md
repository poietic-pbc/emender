# E97 schedule-free scale probes quality pass (2026-06-25)

Task: `quality-pass-e97`

## Verdict

Pass. The schedule-free mini-track is operationally bounded and correctly
staged as audit -> parallel cheap probes -> synthesis -> conditional 16-node
probe -> final synthesis. The track keeps E97-MLP as the only research arm and
does not authorize a 32-node or 64-node schedule-free launch.

## Graph reviewed

- `audit-e97-schedule`: audits exact schedule-free outer args, checkpoint
  save/load/eval semantics, source checkpoint choice, and exact 4-node/8-node
  command recipes. It explicitly forbids Slurm submissions, GDN2/CMAES, and
  any 32/64-node schedule-free launch.
- `run-e97-schedule`: bounded 4-node schedule-free E97-MLP smoke after the
  audit. It permits at most one 4-node training job, records node-hours and
  run/checkpoint/eval evidence, and forbids 16/32/64-node jobs.
- `run-e97-schedule-2`: bounded 8-node schedule-free E97-MLP smoke after the
  audit. It permits at most one 8-node training job, records the same evidence,
  and forbids 16/32/64-node jobs.
- `synthesize-e97-schedule`: joins the 4-node and 8-node probes. It submits no
  Slurm jobs and decides whether to proceed to 16 nodes, pause for a fix, or
  deprioritize schedule-free.
- `run-e97-schedule-3`: bounded 16-node schedule-free E97-MLP smoke after the
  4/8 synthesis only. It must submit no job if 4/8 evidence is not clean and
  promising, and it forbids 32/64-node jobs.
- `synthesize-e97-schedule-2`: final no-Slurm synthesis comparing 4/8/16
  schedule-free evidence to existing avg evidence and recommending whether a
  separate 32-node task should be created or accelerated.

## Validation checklist

- [x] Downstream tasks have concrete validation criteria. Each task reviewed
  has a `## Validation` section with explicit completion evidence, operational
  records, and scope boundaries.
- [x] The 4-node and 8-node schedule-free probes can run in parallel after the
  config audit. Both depend on `audit-e97-schedule` and neither depends on the
  other.
- [x] The 16-node probe is conditional on 4/8 evidence. `run-e97-schedule-3`
  depends on `synthesize-e97-schedule` and requires no submission unless the
  4/8 synthesis recommends proceeding.
- [x] No 32/64-node schedule-free job is authorized by this quality pass. The
  mini-track can only recommend whether a separate 32-node task should be
  created or accelerated after 4/8/16 evidence.
- [x] `run-64-node-e97` remains paused and E97-MLP-only scope is preserved.
  The reviewed schedule-free run tasks explicitly forbid GDN2/CMAES and 64-node
  launches.

## Notes for downstream agents

- There is a separate existing task, `run-e97-32-node-3`, titled "Run E97
  32-node schedule-free diagnostic". That task is part of the separate 32-node
  recipe ladder after `run-e97-32-node-2`, not a downstream task of this
  schedule-free 4/8/16 mini-track. This quality pass does not grant additional
  32-node authority to the mini-track.
- If schedule-free 4/8/16 evidence supports a 32-node probe, the final
  synthesis should either recommend creating/accelerating a separate explicitly
  scoped task or defer schedule-free. It should not submit jobs itself.
- `run-64-node-e97` is currently open but paused, with logs recording that it
  was paused after the 32-node avg retry repeated the loss regression. Leave it
  paused throughout this mini-track.
