# Resilient E97 changed-payload startup smoke — job 5023816

Real (not `--test-only`) submission from fetched authoritative commit
`71ccb3d7b43d72591bf5f3ee88c806c250841e62` with clean compute-visible
`HEAD == origin/main`.

- Run: `run-resilient-e97-2-smoke-20260717T234133Z-71ccb3d`
- Payload: `71ccb3d-20260717T234133Z-startup-smoke-trainer-liveness`
- Immediate state: `PENDING (Priority)`
- Resources: exactly 2 nodes, debug QoS, `00:20:00`, 16 GPUs
- Injection: none; requested finalized generations: one
- Seed SHA256: `1da27d2e09bc6c6f5ffc30e3e4476df1cebd807267431c8524de1a5b0dc5bca9`
- Training arguments: approved flat E97 split-role configuration

The immutable run directory retains the exact command. This changed payload
follows 5023212's pre-generation heartbeat failure and keeps trainer liveness
independent of the generation-progress deadline during checkpoint load and
real training. Launcher validation passed 21/21 and focused runtime/transport
validation passed 2/2. No full gate was submitted.

Conformance: *Resilient DiLoCo Compute Pool* version 1; applicable R02, R03,
R04, R06, R08, R09, R10, R14, R16. Acceptance still requires two model-free
managers, sixteen real trainers, network/node-local traffic, bounded progress,
and one immutable finalized generation.
