# Exact diagnostic retry 4974391

After job 4974389 failed before `srun`, the missing reviewed prerequisite
commit `08558a5` was incorporated. Fresh smoke and production renders and the
mandatory parity gate exited zero with fingerprint
`0ab9f67cec2b7d24016ab02d04e5029038b560140ce222725dd3e36d12de1b68`.
The rendered script contains the phase/error trap. The exact command was
`sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch`, producing job `4974391`.

No promotion or production submission has occurred. Full terminal acceptance
evidence remains required.
