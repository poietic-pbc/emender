## Objective
After live 2, 4, 8, 16, 32, 64, 128, and 256-node gates pass, create an immutable auditable production-authorization package. This task is packaging and review only: it must not submit, modify, cancel, or otherwise touch any Slurm job. Production execution requires a separate WG task created only after explicit user authorization.

## Requirements
- Refuse packaging if any rung is absent, failed, unresolved, not live Frontier evidence, not debug QoS with a 2-hour walltime, not chained from the prior immutable checkpoint, or lacks the required pre/post-injection generations.
- Synthesize all live rung evidence, repairs, regression reruns, injected failures, quorum/fencing/aggregate behavior, cadence, bounded resource measurements, checkpoint/reload results, whole-allocation restart evidence, and final 256-node handoff.
- Freeze the exact passing code commit, launcher/helpers, environment/modules, model/data/optimizer, node-manager protocol, deadlines, membership/quorum, generation/coordinator fencing, aggregate redistribution/apply, replay/catch-up, checkpoint/finalization/restart semantics, and payload hashes.
- Explicitly distinguish survivable in-allocation failure (healthy quorum continued) from Slurm whole-allocation termination (durable restart from a verified immutable handoff in a new allocation).
- Render, but do not execute, a prospective 256-node/2048-rank, 12-hour normal-QoS command from the final ladder checkpoint. Store it as inert text with an authorization-required banner; do not call `sbatch`, `srun`, `scontrol`, `scancel`, or an equivalent submission/control path.
- Produce a normalized diff against the successful 256-node 2-hour debug rung. Only QoS, walltime, and mechanically unique prospective identity may differ; the seed must be the exact final 256-node handoff.
- Document residual failure modes, durable restart procedure, fail-fast cancellation policy for a future authorized operator, and monitoring cadence. Preserve immutable evidence without overwriting earlier manifests.

## Validation
- [ ] Passing live evidence exists for every sequential 2, 4, 8, 16, 32, 64, 128, and 256-node debug-QoS 2-hour gate, including required controlled failures and post-failure generations.
- [ ] Every rung is chained from the preceding immutable verified checkpoint and has no unresolved failure or unchanged retry.
- [ ] The 256-node whole-allocation termination/restart proof and final handoff checksum/reload proof pass.
- [ ] Code, payload, environment, launch, and resilient-protocol identities are immutable, committed, pushed, and tied to clean HEAD matching its remote.
- [ ] The normalized debug-to-prospective-production diff contains only QoS, walltime, and prospective identity differences; the seed is unchanged.
- [ ] In-allocation survival, Slurm termination, durable restart, residual failure modes, and future fail-fast/operator policy are explicit.
- [ ] A concrete candidate command is stored only as inert text and clearly requires separate explicit user authorization.
- [ ] Scheduler accounting and shell history show this packaging task submitted, modified, and cancelled no Slurm job of any QoS.
