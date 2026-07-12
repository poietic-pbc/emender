# Exact 256-node E97 async smoke retry failure: job 4972494

Date inspected: 2026-07-12

## Immutable submission identity

- Job: `4972494`
- Bundle fingerprint: `ecf537fdd8f69add499cf4330abaa8670cd079aa09d746e0f2ce2b898bd5037d`
- Rendered sbatch SHA-256: `259593acbf9c819afb4c10ff21b5438acfffe52acae045b08b13fb205f85daea`
- Bundle manifest SHA-256: `53dcdecc877b3fa4609ddcbba859989ce1c8d18763660f32b1e32dbc73941f5d`
- Exact submission: `sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch`
- Queue: account `bif148`, partition `batch`, QoS `debug`, walltime `00:20:00`
- Requested topology: 256 nodes, 8 GPU tasks/node, 2,048 ranks

The mandatory smoke render, production render, and parity check had exited zero
before submission; their canonical parity result is retained in
`retry-4972494-pending.md`. No command-line scheduler or environment override
was used.

## Terminal scheduler evidence

On resumption, the job was no longer present in `squeue` or `scontrol`. Slurm
accounting retained the following canonical records:

```text
JobIDRaw|JobName|Partition|QOS|Account|State|ExitCode|Elapsed|Timelimit|NNodes|NTasks|Start|End
4972494|e97-async-256-smoke|batch|debug|bif148|FAILED|1:0|00:00:06|00:20:00|256||2026-07-11T15:51:21|2026-07-11T15:51:27
4972494.batch|batch|||bif148|FAILED|1:0|00:00:06||1|1|2026-07-11T15:51:21|2026-07-11T15:51:27
4972494.extern|extern|||bif148|COMPLETED|0:0|00:00:07||256|256|2026-07-11T15:51:21|2026-07-11T15:51:28
```

There is no `srun` job step. Both exact output files exist but are zero bytes:

```text
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  e97-async-256-smoke-4972494.out
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  e97-async-256-smoke-4972494.err
```

No run directory, preflight evidence, rank-start evidence, metrics, accepted
updates, DiLoCo merge, finalized/reloaded checkpoint, or successful terminal
state exists. The external production `latest.pt` still resolves to
`checkpoint_step_1065000_loss_2.5386.pt`.

## Promotion disposition and repair

`build/e97-256/smoke/promotion.json` is absent. No production submission was
made. This run is rejected and must never be used as promotion evidence.

The silent exit occurred in the generated batch script before `srun`; because
both streams are empty and the controller record has expired, the exact failing
shell command cannot be recovered from this run. WG prerequisite
`fix-silent-pre` tracks reproduction and correction of the export-NONE/module/
preflight/helper path with phase diagnostics. A new exact smoke must be freshly
rendered and pass parity after that reviewed fix.
