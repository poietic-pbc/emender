# Pipelined runtime authoritative-line integration

Date: 2026-07-20
Task: `integrate-pipelined-runtime`

## Integrated change

The retained prerequisite history was inspected through `dbdace57`. Its
substantive parent, `b66b4462`, wires `NativeGenerationPipeline` into the
Frontier native trainer and adds a production-role selector regression test.
Those two file changes were applied byte-for-byte to the current authoritative
line. The named commits `4b9960a0`, `58bc6a30`, and `dbdace57` contain only
repeated launch-attempt reporting, so that unrelated report churn was not
integrated.

The exact two-node launcher remains
`scripts/frontier/submit_native_dataplane_2n_gate.sh clean`. Its production
selection is the native CXI backend, two nodes, eight trainers per node, and the
full E97 layout. The submitter intentionally requires authoritative `main` and
a clean source tree. This worker validated its selector and shell syntax but
did not execute it; no Slurm job was submitted.

## Validation

All Python and native commands ran after sourcing
`scripts/frontier/activate_emender_frontier.sh` and used
`"$EMENDER_PYTHON"` / `PYTHON_BIN="$EMENDER_PYTHON"`.

```text
$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py \
  tests/test_resilient_e97_true_2n_launcher.py \
  tests/test_resilient_e97_runtime.py
88 passed in 105.18s

$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py \
  tests/test_native_dataplane_failure.py \
  tests/test_native_pool_integration.py
33 passed in 25.96s

PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 \
  bash scripts/frontier/build_native_resilient_dataplane.sh
10/10 CTest tests passed

bash -n scripts/frontier/submit_native_dataplane_2n_gate.sh \
  scripts/frontier/resilient_e97_true_2n.sbatch
passed
```

## Architecture conformance

Authority: *Resilient DiLoCo Compute Pool*, version 1 (2026-07-17), and
*Native resilient DiLoCo data plane v1*. Applicable gap-matrix requirements
are R01-R16 and NDP01-NDP17.

- R02-R06, R11, R14-R16: the pipeline is fenced by run, epoch, generation,
  attempt, trainer incarnation, layout, and base digest. Admission remains the
  leased READY snapshot with bounded deadlines and the configured Q/T floor;
  launched node count is capacity, not membership.
- R04-R08 and NDP05-NDP12: the existing deterministic token-weighted native
  reduction, strict stale/corrupt rejection, idempotent receipts, atomic
  committed evidence, bounded replay/backpressure, and direct memfd/CXI path
  remain unchanged. The pipeline is an ownership/boundary policy and neither a
  dense Python transport nor a central full-model broker.
- R01, R07, R09-R10, R12-R13 and NDP01-NDP04, NDP13-NDP16: the persistent
  model-free native service retains sealed producer storage before release;
  trainers continue to own model/optimizer state; the result is admitted only
  at the safe generation boundary after commit verification. No Lustre dense
  hot-path, Python dense socket, MPI collective, or second committer is added.
- NDP17: local exact-source native and selector gates pass. Live G2 execution
  remains a downstream authoritative-main task and is not claimed here.

The approved two-node minimum progress floor remains `Q_min=2` and
`T_min=3,934,080` accepted tokens. This integration submitted no Slurm job.
