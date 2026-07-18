# Resilient E97 changed-payload startup smoke — job 5022000

Real (not `--test-only`) submission at `2026-07-17T19:05:11-04:00` from
fetched authoritative commit `696baab80a01475c75ec657e5ca7cbb78f5e399d`.

- Run: `run-resilient-e97-2-smoke-20260717T230418Z-696baab`
- Payload: `696baab-20260717T230418Z-startup-smoke-slurm-nodeid`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`, 16 allocated GPUs
- Injection: none; requested finalized generations: one
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`

The exact command was retained under the immutable run directory before
submission. This payload follows job 5021992's pre-heartbeat failure and adds
the tested `SLURM_NODEID` live node-rank binding. Launcher tests passed 19/19;
compile, shell syntax, and diff checks passed. No full gate was submitted.

Conformance: *Resilient DiLoCo Compute Pool* version 1; applicable R02, R03,
R04, R06, R08, R09, R10, R14, R16. Required smoke evidence remains both
model-free managers, all sixteen real trainers, bounded heartbeats, network/
node-local transport, and one immutable finalized generation.

## Terminal result and diagnosis

Slurm accounting recorded `FAILED (ExitCode 1:0)`: submitted
`2026-07-17T19:05:11-04:00`, started `19:06:01`, and ended `19:09:23`.
Queue time was 50 seconds; runtime was 3 minutes 22 seconds. The supervisor
event stream proves that two managers and all sixteen trainers started with
the intended node/local-rank identities. No generation finalized.

Every real trainer then failed checkpoint loading because the immutable
command incorrectly selected the older
`configs/frontier/e97_async_256_job4962400_golden.json`, whose nested keys are
ignored by the split-role flat override loader. Consequently the loader built
its tiny defaults (`vocab_size=256`, `dim=8`) while the pinned checkpoint has
`embedding.weight=[50281,1792]`. The bounded supervisor restarted trainers
twice and failed after restart exhaustion. This is a pre-generation payload
failure; job 5022000 is retained and will never be retried unchanged.

The changed payload makes the batch launcher fail closed unless
`RESILIENT_E97_TRAIN_ARGS_JSON` resolves to the reviewed flat E97 split-role
configuration. A new short 2-node smoke remains mandatory before any full
02:00:00 gate. Conformance checked against architecture version 1 and R02,
R03, R06, R09, R14, and R16; this failed run supplies no R16 acceptance proof.
