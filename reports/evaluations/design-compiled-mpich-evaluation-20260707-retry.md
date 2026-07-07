# Evaluation: design-compiled-mpich

Date: 2026-07-07
Evaluator: agent-812
Task: `design-compiled-mpich`

## Verdict

Overall grade: 0.00 / 1.00

Confidence: high

Rubric underspecified: no. The task has four explicit validation criteria and
clear domain constraints: design a compiled Cray MPICH helper or extension to
replace the failed mpi4py dense transport for train.py-native async DiLoCo on
Frontier.

## Evidence reviewed

- `wg show design-compiled-mpich`
- `wg log design-compiled-mpich --list`
- `reports/frontier/frontier-mpi-dense-async-diloco-design-20260707.md`
- `git log --oneline -- reports/frontier/frontier-mpi-dense-async-diloco-design-20260707.md`

The task's prior actor attempt completed immediately after assignment and left
no task-specific design artifact, WG artifact, progress log, or commit for
`design-compiled-mpich`. A nearby report exists at
`reports/frontier/frontier-mpi-dense-async-diloco-design-20260707.md`, but that
file is from commit `fae3a26` for `implement-frontier-mpi` and explicitly
describes a Python mpi4py dense transport over Cray MPICH. It is not a compiled
C/C++ MPICH helper or extension design, and it does not satisfy this task's
"replace failed mpi4py" premise.

## Dimension scores

1. Compares subprocess helper vs Python extension vs other IPC: 0.00

   No task-specific report was produced. The adjacent mpi4py report does not
   compare subprocess, extension, or explicit IPC options for a compiled helper.

2. Selects one implementation route with concrete APIs, data format, lifecycle,
   and failure handling: 0.00

   No selected compiled-helper route exists for this task. The adjacent report
   documents an mpi4py implementation and its envelope format, which is outside
   the requested compiled replacement path.

3. States how dense delta buckets move without mpi4py and without live Lustre
   quorum: 0.00

   The available report moves dense buckets through mpi4py `MPI.BYTE`
   point-to-point calls. That directly conflicts with this task's requirement to
   avoid mpi4py. It does exclude live Lustre for updates, but the required
   non-mpi4py dense movement design is absent.

4. Gives a minimal implementation/test plan through 1n/2n/8n/64n and 256n
   debug gate: 0.00

   No compiled-helper implementation/test ladder was provided. The task
   specifically asks for validation through 1n, 2n, 8n, 64n, plus a fail-closed
   256n debug gate.

## Rationale

The actor did not deliver the requested design. The only relevant-looking
document in the tree is for a different task and chooses the opposite transport
direction: mpi4py remains the active dense data plane with a future compiled
extension left as possible work. Because the requested deliverable is a design
report for the compiled MPICH helper itself, there is no partial credit to award
against the explicit validation criteria.

Recommended WG disposition: keep `design-compiled-mpich` incomplete/retryable
so `implement-compiled-mpich` remains blocked until a real compiled-helper
design is available.
