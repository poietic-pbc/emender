# Integrate OLCF runtime/comm branches into main

Task: `integrate-olcf-runtime-main`
Date: 2026-07-04

## Summary

Integrated the completed OLCF runtime/RCCL WG branches into local `main` in
the requested dependency order and prepared `main` for push to `origin/main`.
All branch merges completed with the default `ort` strategy and no manual
conflict resolutions.

## Integrated Inputs

- `origin/wg/agent-532/study-olcf-pytorch` at `8d681ca`
  - Added `reports/frontier/olcf-stack-alignment-20260704.md`.
- `origin/wg/agent-540/frontier-runtime-comm-plumbing` at `40201d8`
  - Added shared Frontier runtime/RCCL helper plumbing, runtime manifest
    capture, and `tests/test_frontier_runtime_plumbing.py`.
- `origin/wg/agent-543/debug-current-runtime-rccl-plugin` at `a493566`
  - Added current-runtime RCCL plugin validation report and debug wrapper.
- Local `main` already contained `ca13c7c` and `112ce4f`
  from prepare-OLCF-runtime-candidate before this integration.
- `origin/wg/agent-549/debug-updated-olcf-runtime` at `9cae91e`
  - Added updated OLCF runtime debug wrappers and report.
- `origin/wg/agent-552/decide-runtime-comm-scaleout` at `b38d006`
  - Added runtime communication scaleout decision report.

The post-merge integration HEAD before adding this report was:

```text
1fa843cf163b4f9582b2dca5df0c1f657adf8b87
```

The exact final pushed commit is the commit that contains this report; it was
recorded in the WG task log after push.

## Validation

Passed:

```text
python3 -m py_compile train.py
bash -n scripts/frontier/debug_smoke_one_node.slurm scripts/frontier/rccl_allreduce_diag.sbatch scripts/frontier/frontier_runtime_env.sh scripts/frontier/e97_current_runtime_rccl_plugin_debug.sbatch scripts/frontier/e97_updated_olcf_runtime_debug.sbatch scripts/frontier/gdn2_updated_olcf_runtime_debug.sbatch scripts/frontier/gdn2_updated_olcf_runtime_preflight.sbatch
git diff --check
```

The static assertions from the lightweight plumbing test were also run without
pytest and passed:

```text
static frontier runtime plumbing assertions passed
```

Not fully runnable in this local environment:

```text
python3 -m pytest -q tests/test_frontier_runtime_plumbing.py
/usr/bin/python3: No module named pytest
```

An ad hoc standard-library replay of the full test body then reached the next
missing dependency:

```text
Traceback (most recent call last):
  File "<stdin>", line 44, in <module>
  File "/lustre/orion/bif148/scratch/erikgarrison/emender/train.py", line 24, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Risk: the lightweight plumbing test's static shell/report assertions passed,
and `train.py` compiles, but the dynamic Python import path could not be
executed here because this environment lacks both `pytest` and `torch`.

## Safety Notes

- No production Slurm job was submitted.
- No production chain symlink was modified.
- Unrelated untracked WG, log, data, core, and scratch files were left
  untouched.
