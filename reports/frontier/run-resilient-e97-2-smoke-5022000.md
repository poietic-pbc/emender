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
