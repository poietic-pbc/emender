# Exact-source G2 ROCm runtime linkage

Frontier job 5036108 reached the authoritative native bundle but the clean
batch payload could not load `libamdhip64.so.7`.  The installed binary only
carried its origin-relative bundle RUNPATH, and the batch shell neither loaded
the canonical ROCm module nor rejected inherited loader paths.

The canonical build now resolves the ROCm 7.1.1 and Cray libfabric runtime
directories from the reviewed module stack and embeds both alongside the
origin-relative bundle directory.  Configuration fails if either absolute
directory lacks its required SONAME; caller-supplied relative cache paths are
rejected before canonicalization so they cannot resolve against an unreviewed
host working directory.  The G2 batch payload clears inherited
`LD_LIBRARY_PATH`, loads the canonical modules, reconstructs the loader path
from the bundle and reviewed module directories, and fails before `srun` if
`ldd` reports any unresolved library or if `libamdhip64.so.7` resolves outside
the reviewed ROCm directory.  `loader-preflight.txt` retains the RUNPATH, full
resolution, canonical paths, and SHA-256 identity of every resolved library.

Source and binary attestation is unchanged: the batch still calls
`validate_build_manifest(..., require_clean=True)` before resolving the hashed
installed gate, and the build manifest records the exact installed bytes and
CMake cache.

## Validation and conformance

This change conforms to *Resilient DiLoCo Compute Pool*, version 1, and applies
R10, R14, R16 and NDP02, NDP03, NDP13, NDP16, NDP17.  It changes no membership,
generation, transport, aggregation, recovery, minimum-progress, or commit
semantics.  The gate remains exactly two nodes, point-to-point production CXI,
bounded by its existing deadlines, and produces no Slurm submission here.

- `tests/test_frontier_runtime_plumbing.py` exercises ROCm resolution under
  `env -i` and its missing-root failure.
- `tests/test_validate_native_dataplane_2n_gate.py` runs `ldd` on the installed
  gate after constructing the canonical clean module environment, proves
  `libamdhip64.so.7` resolves from `/opt/rocm-7.1.1/lib`, and rejects any
  unresolved SONAME.
- Focused Python result: 25 passed across runtime plumbing, native G2 launcher,
  and exact two-node acceptance.
- Canonical native build succeeded; CTest passed 10/10; installed RUNPATH is
  `$ORIGIN/../lib64:/opt/rocm-7.1.1/lib:/opt/cray/libfabric/2.3.1/lib64`.
- Shell syntax and `git diff --check` pass.

No Slurm command was invoked.
