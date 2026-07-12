# Slurm state resolution: exact E97 smoke job 4972494

Observed from `login04` on 2026-07-12 at 03:27 EDT (07:27 UTC).

## Conclusion

Job `4972494`, immutable-bundle fingerprint
`ecf537fdd8f69add499cf4330abaa8670cd079aa09d746e0f2ce2b898bd5037d`,
received the requested 256-node allocation but failed in the batch step after
six seconds. It did not reach an `srun` step. This is failure evidence, not
promotion evidence. No production job was submitted and no promotion was
performed by this monitoring task.

The parent `run-exact-256-node` file-only wait was stale: both output files had
already been created, but were empty. A queued user message woke the parent at
07:26 UTC; this task then sent the concrete terminal state to the parent and
attempted `wg resume run-exact-256-node`. The resume command correctly reported
that the task was no longer paused. The parent acknowledged the guidance and is
active for full failure handling.

## Commands attempted

```text
squeue -j 4972494 -o '%i|%T|%r|%S|%V|%M|%l|%D|%C|%P|%N'
slurm_load_jobs error: Invalid job id specified
exit: 1

sacct -X -j 4972494 --starttime 2026-07-01 --format=JobIDRaw,JobName,Partition,State,Reason,Submit,Eligible,Start,End,Elapsed,Timelimit,NNodes,NCPUS,NodeList,ExitCode,DerivedExitCode -P
exit: 0

sacct -j 4972494 --starttime 2026-07-01 --format=JobIDRaw,JobName,Partition,State,Reason,Submit,Eligible,Start,End,Elapsed,Timelimit,NNodes,NCPUS,NodeList,ExitCode,DerivedExitCode -P
exit: 0

scontrol show job -dd 4972494
slurm_load_jobs error: Invalid job id specified
exit: 1

scontrol show step 4972494
scontrol: error: scontrol_print_step: slurm_get_job_steps(4972494) failed: Invalid job id specified
exit: 1

seff 4972494
/usr/bin/bash: seff: command not found
exit: 127
```

The live-controller queries establish that the completed job has been purged
from `slurmctld`; historical accounting remains available through `sacct`.

## Historical accounting state

```text
JobIDRaw|JobName|Partition|State|Reason|Submit|Eligible|Start|End|Elapsed|Timelimit|NNodes|NCPUS|ExitCode|DerivedExitCode
4972494|e97-async-256-smoke|batch|FAILED|None|2026-07-11T15:22:53|2026-07-11T15:22:53|2026-07-11T15:51:21|2026-07-11T15:51:27|00:00:06|00:20:00|256|28672|1:0|0:0
4972494.batch|batch||FAILED||2026-07-11T15:51:21|2026-07-11T15:51:21|2026-07-11T15:51:21|2026-07-11T15:51:27|00:00:06||1|56|1:0|
4972494.extern|extern||COMPLETED||2026-07-11T15:51:21|2026-07-11T15:51:21|2026-07-11T15:51:21|2026-07-11T15:51:28|00:00:07||256|28672|0:0|
```

All timestamps above are Slurm's local time (`America/New_York`, EDT). The job
waited 28 minutes 28 seconds from submission to allocation. The terminal reason
field is `None`; Slurm records the batch process exit code `1:0` and derived
exit code `0:0`. The absence of any job step besides `batch` and `extern`
confirms that the launcher never created the expected 2048-rank `srun` step.

## Allocation and topology

- Cluster/account: `frontier` / `bif148`
- Partition/QoS: `batch` / `debug`
- Requested topology: 256 nodes, 8 tasks/node, 2048 tasks, 7 CPUs/task,
  1 GPU/task, 20-minute walltime
- Accounting allocation: 256 nodes, 28,672 CPUs, 125 TiB memory,
  556,262 energy units
- Requested TRES: `billing=14336,cpu=14336,mem=125T,node=256`
- Allocated TRES: `billing=28672,cpu=28672,energy=556262,mem=125T,node=256`
- Batch host: `frontier00002`
- Full compressed node allocation:
  `frontier[00002-00005,00129-00132,00257,00263,00266,00268,00385,00394,00398,00402,00517,00520-00521,00526,00646,00650,00652,00654,00769-00772,00897-00899,00901,01025-01028,01166-01169,01292-01295,01419-01422,01548-01551,01675-01678,01800,01803,01806-01807,01922-01923,01927,01936,02063-02066,02179,02192-02194,02320-02323,02439,02447-02449,02566,02576-02577,02581,02690,02705-02707,02826,02829-02831,02945-02948,03073-03076,03201-03204,03329-03332,03457-03460,03585-03588,03713-03715,03718,03841,03844-03846,03970-03972,04097-04099,04225-04227,04353-04355,04481-04483,04609-04611,04737-04739,04865-04867,04993-04995,05121-05122,05125,05251-05253,05377-05379,05505-05507,05633-05635,05761-05763,05889-05891,06017-06019,06145-06147,06275-06277,06403,06406-06407,06529-06530,06533,06658-06660,06785-06787,06916-06917,06924,07041-07043,07188-07189,07195,07297,07299-07300,07426-07428,07553-07554,07557,07681-07683,07809-07810,07813,07939-07941,08065-08067,08193-08195,08321-08323,08449-08451,08577-08579,08705,08707-08708,08833-08835,08961-08963,09089-09091,09217-09218,09220,09345-09347,10113-10115]`

## Exact paths and filesystem evidence

Accounting records the work directory as:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949
```

The submitted script directives were:

```text
#SBATCH --output=e97-async-256-smoke-%j.out
#SBATCH --error=e97-async-256-smoke-%j.err
```

Therefore the exact expanded paths are:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/e97-async-256-smoke-4972494.out
/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/e97-async-256-smoke-4972494.err
```

Both files exist, are zero bytes, and have mtime
`2026-07-11 15:51:25 -0400`. Reading their first 240 lines produced no output.
Thus the concrete failure mode available from retained evidence is: batch exit
`1:0` before any logged preflight output or `srun` step. The empty files do not
support a more specific application-level diagnosis.

## Wakeup and notification record

At 07:27 UTC this task sent `run-exact-256-node` a WG message containing the
terminal state, timestamps, exit codes, exact output/error paths, empty-file
finding, and missing `srun` step. It also logged the finding on the parent. The
parent had already transitioned out of `waiting` after a queued user message,
so the explicit `wg resume` returned `Task 'run-exact-256-node' is not paused`.
This replaces the stale file-only wait with an active parent worker that can
preserve evidence, create any required fix dependency, and rerun exact smoke.

