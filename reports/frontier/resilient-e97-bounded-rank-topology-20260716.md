# Bounded resilient E97 rank topology after job 5009365

Job 5009365 proved that mmap checkpoint loading alone does not bound the old
eight-model-per-node launch: rank 4 was OOM-killed at 64,590,084 KiB before a
manager exchange or finalized generation.  The immutable job-5000436
generation-9, step-1525400 checkpoint remains the only restart handoff.

The changed payload makes quorum membership physical-node based.  A 2-node
allocation still launches and records all 16 Slurm ranks, but only local rank
zero on each node owns an E97 trainer/manager and checkpoint state.  The other
seven ranks per node are independently supervised sentinel lanes: they publish
atomic heartbeats, never execute or load the trainer payload, follow the
trainer's atomic terminal record, propagate its exit status, and fail at a
bounded deadline.  Trainer manager IDs use `SLURM_NODEID` (0..1), eliminating
the prior false 16-node identity and eight full CPU workspaces per node.

This is a debug-only memory/topology correction.  It does not constitute the
mandatory live injection/restart proof or production authorization.  A retry
is permitted only after the focused tests, commit, push, and payload hash are
recorded, and it must remain 2 nodes, 16 Slurm ranks, debug QoS, and exactly two
hours.

Validation before commit:

- runner `bash -n`: pass;
- lane module compilation with the project Python: pass;
- rank-lane tests: 3 pass;
- resilient transport tests: 9 pass;
- combined focused run reached 27 passing tests in 120 seconds; the remaining
  broader DiLoCo suite was interrupted because it exceeded this bounded local
  validation window, not because of a test failure.
