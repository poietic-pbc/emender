# Exact pipelined real-E97 two-node acceptance launcher

The authoritative renderer is
`scripts/frontier/render_resilient_e97_exact_2n_acceptance.py`.  Its default
operation is a no-submit JSON render.  `--submit` is separately guarded by a
clean `main`, exact equality with `origin/main`, native artifact digest checks,
the retained full-layout G2 gate, and an empty per-user Slurm queue.

The manifest fixes two nodes, eight real trainers per node, K40, native CXI,
and five independently named/fenced phases: five-generation clean overlap,
bounded native-service fault/rejoin, invalid-result rejection, expected
checkpoint-publication failure, and a fresh-allocation restart from the last
authoritative generation.  Every phase retains identical source commit and
native bundle identity.  No 4+ node submission is renderable.

Each debug phase requests `02:00:00`, which accommodates real E97 model load,
five K40 generations, checkpoint I/O, and controlled shutdown without
weakening performance-sensitive stage bounds. Handoff/apply/publication are
180 seconds, integrity is 120 seconds, and quorum/progress are 420 seconds.

## Validation

Conformance was checked against **Resilient DiLoCo Compute Pool version 1**,
requirements **R01–R16**, and **Native resilient DiLoCo data plane version 1**,
requirements **NDP01–NDP17**. The manifest records those requirement IDs and
binds READY membership policy, exact two-node capacity, Q/T bounded generation
progress, immutable source/native identities, separate allocation fences,
strict invalid-result and publication-failure phases, and authoritative
fresh-restart lineage. The real launcher remains native CXI, point-to-point,
bounded, model-free-manager split-role code; it does not add an MPI/all-rank,
Python-dense, Lustre-hot-path, or central-broker alternative. Minimum progress
remains `Q_min=2`, `T_min=3934080` accepted tokens.

Commands were run after canonical Frontier activation; no `sbatch` or other
Slurm submission command was executed:

```text
$EMENDER_PYTHON -m pytest -q tests/test_resilient_e97_exact_2n_acceptance.py tests/test_resilient_e97_true_2n_launcher.py
54 passed

$EMENDER_PYTHON -m pytest -q tests/test_native_pipeline.py tests/test_resilient_e97_true_2n_launcher.py tests/test_resilient_e97_runtime.py tests/test_resilient_e97_exact_2n_acceptance.py
90 tests collected; pipeline/launcher/runtime selector completed successfully

bash -n scripts/frontier/resilient_e97_true_2n.sbatch
$EMENDER_PYTHON -m py_compile scripts/frontier/render_resilient_e97_exact_2n_acceptance.py

PYTHON_BIN="$EMENDER_PYTHON" BUILD_JOBS=8 bash scripts/frontier/build_native_resilient_dataplane.sh
10/10 CTest tests passed; native install and artifact manifest recording passed
```

The dry-run test verifies node count, generation count, K40, fence ordinals,
fault target, invalid/publication injection selection, restart source, absolute
artifact paths, common source/bundle identity, bounded deadlines, and absence
of any 4+ node submission command.
