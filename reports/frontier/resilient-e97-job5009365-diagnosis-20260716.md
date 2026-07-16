# Terminal diagnosis: resilient E97 job 5009365

Job `5009365` used changed payload `0133bff` after the recorded job-5000869
OOM diagnosis. Slurm records debug QoS, two nodes, exact `02:00:00` limit,
start `2026-07-16 04:28:47 EDT`, end `04:32:50`, and terminal `FAILED 90:0`.
Compute step `.0` ran `00:03:46` and ended `OUT_OF_MEMORY 0:125`; Slurm
reported `MaxRSS=64,590,084K`. Retained stderr identifies task/rank 4 on
`frontier06070` as OOM-killed at `04:32:47`, after which Slurm terminated the
step. The batch step failed `90:0`; extern completed `0:0`.

All 16 trainer processes were launched, eight on each of `frontier06070` and
`frontier06074`, from the verified job-5000436 generation-9/step-1525400
handoff. The retained output contains runtime initialization but no accepted
resilient update, aggregate, apply, generation publication, metrics manifest,
or new checkpoint. Therefore zero generations finalized and the immutable
job-5000436 checkpoint remains the last valid restart state; no partial
job-5009365 state is eligible.

The mmap checkpoint-load change did not reduce the observed per-task peak or
prevent node OOM. The strongest supported cause remains eight full E97
trainer/model/optimizer processes per node exceeding bounded host memory;
the launch still maps `--node-rank` to each global rank and declares
`--node-count 16`, and it launches no separately supervised one-per-physical-
node managers. This is a repeat of the failure class, not evidence that the
resilient protocol boundary ran. Per task requirements, another retry is
prohibited until a different topology/memory fix is tested and committed.

Retained evidence:

- `logs/frontier/trainpy_async_quorum/resilient-e97-2n-20260716T081224Z-5009365.out`
- `logs/frontier/trainpy_async_quorum/resilient-e97-2n-20260716T081224Z-5009365.err`
- `sacct -j 5009365 --format=JobID,JobName,QOS,State,ExitCode,Elapsed,Timelimit,Start,End,NodeList,MaxRSS`
