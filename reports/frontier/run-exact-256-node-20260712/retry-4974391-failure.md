# Exact smoke retry 4974391: pre-srun module bootstrap failure

## Disposition

The exact immutable smoke retry was submitted and allocated at the required
topology, but failed closed before `srun`. It is rejected for promotion. No
production job was submitted and no `promotion.json` was written.

## Immutable submission

- Job ID: `4974391`
- Fingerprint: `0ab9f67cec2b7d24016ab02d04e5029038b560140ce222725dd3e36d12de1b68`
- Exact command: `sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch`
- Queue: partition `batch`, QoS `debug`, account `bif148`, no reservation
- Walltime: `00:20:00`
- Topology: 256 nodes, 2,048 tasks, 8 tasks/node, 1 GPU/task, 7 CPUs/task
- Submit: `2026-07-12T03:46:14-04:00`
- Start: `2026-07-12T03:46:51-04:00`
- End: `2026-07-12T03:46:53-04:00`

The fresh smoke render, production render, and strict parity gate exited zero
before submission. The only allowed profile differences remained duration and
stop budget plus concrete partition/QoS; the account was identical and the
reservation absent.

## Terminal evidence and root cause

Slurm reports job state `FAILED`, exit code `72:0`, runtime `00:00:02`, and an
allocation of exactly 256 nodes. `scontrol show job 4974391` retains the exact
`SubmitLine`, immutable script path, scheduler geometry, and output paths.

Standard output is empty. Standard error contains the new phase diagnostics:

```text
e97-presrun phase=bootstrap status=begin job_id=4974391 host=frontier00122 export=NONE
e97-presrun phase=bundle-binding status=begin fingerprint=0ab9f67cec2b7d24016ab02d04e5029038b560140ce222725dd3e36d12de1b68
e97-presrun phase=module-bootstrap status=begin fingerprint=0ab9f67cec2b7d24016ab02d04e5029038b560140ce222725dd3e36d12de1b68
module initialization is unavailable: /etc/profile.d/modules.sh
```

Thus the silent-failure instrumentation worked and isolated the next defect:
the immutable launcher assumes `/etc/profile.d/modules.sh`, which does not
exist in the Frontier batch environment under `--export=NONE`. The failure
occurred before helper verification and before `srun`.

## Acceptance consequences

- On-node preflight did not complete.
- No `srun` step exists.
- Zero of 2,048 ranks started or contributed updates.
- No finite-loss, accepted-update, DiLoCo-merge, metrics, rank-start,
  checkpoint-finalization, or checkpoint-reload evidence exists.
- The bundle is not promotable; no production submission is authorized.

A prerequisite repair must discover or initialize Frontier's supported module
bootstrap deterministically under `--export=NONE`, retain fail-closed behavior,
and pass focused tests plus fresh render/parity validation before another exact
256-node retry.
