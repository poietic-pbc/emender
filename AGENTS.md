# Emender project guide

Work directly in the current attended coding session. Do not start or use the
WorksGood (`wg`) service, dispatcher, workers, or completion/review machinery.
Keep implementation and Frontier debugging single-threaded unless the operator
explicitly requests otherwise.

## Frontier development environment

Before running Python, pytest, native builds, or Slurm submission preflight on
Frontier, source the canonical project environment:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

Do not use bare `python`, `python3`, or a guessed version such as `python3.11`
before activation: Frontier's login-shell default is Python 3.6 and cannot parse
this repository. After activation, invoke `"$EMENDER_PYTHON" -m pytest ...` and
pass `PYTHON_BIN="$EMENDER_PYTHON"` to wrappers that accept an interpreter. The
activation script is authoritative for the module stack and approved Python
3.12 environment; task-specific scripts may add settings but must not duplicate
or replace that setup.

## Frontier scheduler queue verification

Treat Slurm partition and QoS as separate required evidence. Every Frontier
submission or runner that claims a queue binding must request both fields and
retain scheduler output that names both `Partition` and `QOS` explicitly (for
example, `squeue -o '%i|%P|%q'` and/or
`sacct --format=JobIDRaw,Partition,QOS,...`). Never infer QoS from the default
`squeue` `PARTITION` column. Iterative acceptance submissions must use
`Partition=batch` plus `QOS=debug`, verify both fields while the job is
queued/running and again in terminal accounting evidence, and fail closed on a
missing or different value.

## Resilient DiLoCo design authority

Before changing, testing, running, or scaling resilient training behavior, read
[`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`](docs/RESILIENT_DILOCO_COMPUTE_POOL.md).
It is the normative architecture. Cite the applicable requirement IDs from
[`docs/RESILIENT_DILOCO_GAP_MATRIX.md`](docs/RESILIENT_DILOCO_GAP_MATRIX.md) in
validation reports. Older async design notes and the current harness are
evidence and scaffolding, not competing design authorities.

Keep `AGENTS.md` and `CLAUDE.md` synchronized.
