# Resilient E97 changed-payload startup smoke — job 5023212

Real (not `--test-only`) submission at `2026-07-17T19:29:05-04:00` from
fetched authoritative commit `0b0e95279e7b550beab5a136c5c624ba7259cd03`.
The clean compute-visible checkout had `HEAD == origin/main`.

- Run: `run-resilient-e97-2-smoke-20260717T232905Z-0b0e952`
- Payload: `0b0e952-20260717T232905Z-startup-smoke-flat-e97-config`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`, 16 allocated GPUs
- Injection: none; requested finalized generations: one
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`
- Training arguments: `configs/frontier/e97_resilient_split_role_flat.json`

The immutable run directory retains the exact command and rendered parity
result. Parity reports `ok=true`, no forbidden or missing fields, identical
model/data/optimizer/launcher/checkpoint/network paths, and only the approved
failure-injection, node-count, QoS, and walltime fields differ from production.
The smoke disables all injection. No full 02:00:00 gate was submitted.

This payload follows job 5022000's pre-generation model-shape failure. It is
changed, committed, pushed, fetched, and guarded so a nested/non-flat training
configuration fails before roles launch. Validation passed 20 launcher tests,
two focused runtime/transport reruns, shell syntax, compile, and diff checks.

Conformance: *Resilient DiLoCo Compute Pool* version 1; applicable R02, R03,
R04, R06, R08, R09, R10, R14, R16. Acceptance still requires both model-free
managers, all sixteen real trainers, bounded heartbeats and network/node-local
transport, and one immutable finalized generation.

## Terminal result and diagnosis

Slurm accounting recorded `FAILED (ExitCode 1:0)`: submitted
`2026-07-17T19:30:22-04:00`, started `19:30:44`, and ended `19:36:12`.
Queue time was 22 seconds and runtime was 5 minutes 28 seconds. Both managers
and all sixteen trainers launched with the intended identities, but no
generation finalized.

The live event stream proves a deterministic pre-generation failure: each
trainer was evicted at `heartbeat_deadline`, restarted twice, and trainer 0
then exhausted its bounded restart budget. The trainer stopped its import
heartbeat immediately before `_load_real`; cloning the real E97 checkpoint
under eight-way node startup exceeded 60 seconds, and unlike managers the
trainer had never started the independent liveness heartbeat. This was not a
model, checkpoint-integrity, network, or generation-quorum rejection.

Job 5023212 is retained and will not be retried unchanged. The next changed
payload keeps trainer liveness active independently of generation progress
during checkpoint load and training, while the separate 900-second progress
deadline remains authoritative. Another short 2-node smoke is mandatory.
Conformance checked against architecture version 1 and R02, R06, R09, R14,
and R16; this failed run supplies no R16 acceptance proof.
