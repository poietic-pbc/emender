# Repaired E97 256-node production: job 4980157

Task: `deploy-repaired-e97-256`

Date: 2026-07-13

## Submission

The exact successful repaired smoke bundle from job `4979704` was promoted
once to Slurm job `4980157`. The pre-submit checker reported only the four
authorized changes: nodes 2 to 256, mechanically derived ranks 16 to 2048,
QoS debug to normal, and walltime 00:20:00 to 12:00:00. The partition remains
`batch`. Both renders retain generations 1,000,000, steps 40,000,000, and
local steps 40.

The production execution tree is materialized from the exact smoke launch
commit `9fff689c9f9252b6a264773c207f8f8ca8509666`. The reviewed parity
fingerprint is
`a4a493eb60c6425f3df2dea71436ded0b43fee282de6f2bac87a0a49e4f0ad5b`.
The approved gate and evidence-only commit `9ac7fe2` was pushed to
`origin/main` before submission, with a clean tracked worktree and
`HEAD == origin/main`.

The pinned step-1,525,000 seed independently hashed to
`1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`
immediately before submission. The retained before/after stable seed pointer
manifests compare equal.

The submission command exited zero and returned job ID `4980157`. An atomic
exclusive attempt marker prevents a second invocation. Immediate Slurm state
was `PENDING (Priority)`, with exactly 256 nodes requested, 2,048 tasks at
eight per node, normal QoS, and a 12-hour limit.

## Monitoring

Terminal monitoring is in progress. A short `COMPLETED 0:0` exit will be
rejected; success requires scheduler-controlled runtime near 12 hours,
multiple healthy merges beyond generation 0 and step 1,525,040, a reloadable
final checkpoint, unchanged stable seed pointer, and healthy publication.

At 2026-07-13 12:57 UTC, job `4980157` remained `PENDING (Priority)` with
the original request intact: partition `batch`, QoS `normal`, 256 nodes,
2,048 tasks at eight per node, and a 12-hour limit. Slurm projected a start
at 2026-07-13 14:24 EDT and an end at 2026-07-14 02:24 EDT. Monitoring did
not invoke the submit path; the exclusive attempt count remains one.

At 2026-07-14 08:53 UTC, the same exclusive job remained `PENDING
(Priority)`. Its request was unchanged, and Slurm's revised projected start
was 2026-07-14 16:00 EDT. Accounting still reported zero elapsed allocation
time. No submission path was invoked; monitoring continues against job
`4980157` only.
