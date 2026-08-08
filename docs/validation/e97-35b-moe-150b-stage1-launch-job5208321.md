# E97 35B MoE 150B stage-1 launch — job 5208321

Date: 2026-08-08

The operator-authorized stage-1 continuation was submitted as Frontier job
**5208321** from immutable source
`54bf2f2b03b2ef8e1ae65d4176df2d8453a96bec`.

```text
Partition=batch
QOS=normal
State=RUNNING
Nodes=256
Tasks=2048
TimeLimit=06:00:00
Requeue=0
MAX_STEPS=2960
TRAIN_MINUTES=0
RUN_ID=e97-moe-scale-ladder-mainfc52
```

The job started at `2026-08-08T14:38:24`. Submit-time and live job records both
separately verified `Partition=batch` and `QOS=normal`; the stable run identity
retains `squeue-5208321.txt`, `scontrol-5208321.txt`, and
`source-5208321.txt`. The launcher resolved the complete canonical 100.474B
`checkpoints/latest` authority whose manifest SHA-256 is
`933d35abec874d5c88dcb31fbd05815ef47c9b0e508563158e4c26afc46b5550`.

An initial submission, job **5208227**, failed in one second with exit `128:0`
before the batch script or model load because its submit working directory was
login-node-local `/tmp` and therefore unavailable on compute nodes. It consumed
about 0.071 node-hours and produced no run mutation. The immutable source was
moved to a clean Lustre worktree and resubmitted without changing training
code or scientific parameters.

The target remains step `2332080`, accepted tokens `150134063104`, after exactly
2,960 new steps and 74 K40 merges. Stage 2 remains unauthorized.

## Validation and conformance

This is the fixed-world ADR-003 production path. Applicable safety intent is
R07, R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. R02-R06, R08-R11;
NDP01, NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07 are
retired and unclaimed for this execution. No elastic, async-overlap,
communicator-shrink, automatic restart, database, SQLite, or filesystem-lock
claim is made.

Launch evidence commands:

```bash
squeue -j 5208321 -o '%i|%P|%q|%T|%D|%M|%l'
scontrol show job 5208321 -o
sacct -j 5208227 -X --format=JobIDRaw,State,ExitCode,Elapsed,Partition,QOS,AllocNodes
```
