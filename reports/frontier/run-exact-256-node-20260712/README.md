# Exact E97 256-node smoke retry 4974389

The reviewed `fix-silent-pre` change was incorporated before a fresh render.
All three mandatory pre-submit commands exited zero on 2026-07-12. The smoke
and production bundles share fingerprint
`8ccb57708744e19c68bf8ad68f48b849047abb8d015d627e1eb547ca29e5bdb3`.
Canonical parity permits only time/derived stop budget and QoS differences;
both profiles use account `bif148` and partition `batch`, and neither has a
reservation.

The exact submission command was:

```text
sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch
```

Slurm job `4974389` requests exactly 256 nodes, eight GPU tasks per node, and
2,048 ranks, with `batch`/`debug` and walltime `00:20:00`. No scheduler or
environment override was supplied. At this checkpoint the job is pending for
priority. The production external `latest.pt` pointer was recorded before
submission.

No `promotion.json` exists and no production job has been submitted. Promotion
remains forbidden until on-node preflight, all 2,048 rank starts and accepted
updates, finite loss, a DiLoCo merge, checkpoint finalize/reload, unchanged
external pointer, and terminal `COMPLETED 0:0` are all proven.
