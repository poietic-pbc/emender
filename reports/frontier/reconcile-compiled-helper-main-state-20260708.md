# Compiled helper main-state reconciliation

Date: 2026-07-08
Task: `reconcile-compiled-helper-main-state`

## Decision

`main` and `origin/main` are already aligned at the same commit after fetching
from `origin`. No reset, merge, force push, or Slurm submission was performed.
Future scale tasks should start from `origin/main` commit
`1ef9d6234c967848de89b798816076eac66266b9`.

## Commit identities

- Implementation branch commit accepted by `origin/main`:
  `5ef33b18583fac4695197e2a78d1a2049b58ed40`
  (`feat: add compiled MPICH collective reducer (implement-compiled-mpich-2)`).
- Local automatic squash merge commit:
  `ac7b5b8335d7faa3d2cef554d9fd288a57295bd4`
  (`feat: implement-compiled-mpich-2 (agent-840)`).
- Local `main` after `git fetch origin --prune`:
  `1ef9d6234c967848de89b798816076eac66266b9`.
- `origin/main` after `git fetch origin --prune`:
  `1ef9d6234c967848de89b798816076eac66266b9`.
- 8n/64n run task report commit:
  `1ef9d6234c967848de89b798816076eac66266b9`
  (`report: compiled helper 8n64n ladder (run-compiled-helper)`).

## Equivalence evidence

`main...origin/main` has no remaining divergence:

```text
git rev-list --left-right --count main...origin/main
0	0
```

The two implementation commits are different commit objects with different
subjects and timestamps, but they carry the same patch content:

```text
git show ac7b5b8 --pretty=format: --patch | git patch-id --stable
6f9002df17ddaeffbae1c52a47e04c45bf53b2db 0000000000000000000000000000000000000000

git show 5ef33b1 --pretty=format: --patch | git patch-id --stable
6f9002df17ddaeffbae1c52a47e04c45bf53b2db 0000000000000000000000000000000000000000
```

Branch containment confirms the accepted implementation path is the one on
`main`, while the local automatic squash commit remains isolated on the
run-task branch:

```text
5ef33b1 contains:
* main
+ wg/agent-844/reconcile-compiled-helper-main-state
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
  remotes/origin/wg/agent-832/implement-compiled-mpich-2

ac7b5b8 contains:
+ wg/agent-841/run-compiled-helper

1ef9d62 contains:
* main
+ wg/agent-844/reconcile-compiled-helper-main-state
  remotes/origin/HEAD -> origin/main
  remotes/origin/main
```

## Run evidence tied to commit

The 8n/64n run report is committed on `origin/main` at
`1ef9d6234c967848de89b798816076eac66266b9` and records:

- 8n Slurm job `4954290`: pass, 64 expected ranks, 64 accepted updates.
- 64n Slurm job `4954317`: pass, 512 expected ranks, 512 accepted updates.
- No 128n, 256n, or production job submitted by `run-compiled-helper`.
- Only run-local `async_run/latest.json` and run-local checkpoint records were
  written.

The run-local manifests checked during this reconciliation were:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_8n/4954290-20260708T012807Z/artifacts/manifest.json`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/trainpy_async_quorum_smoke/20260708/E97_1.3B_step1065000_trainpy_compiled_helper_64n/4954317-20260708T013528Z/artifacts/manifest.json`

## Safety notes

- No force push was used.
- No local alignment command was required because local `main` and
  `origin/main` were already the same object.
- Untracked WG/runtime/log/data files visible in the shared working tree were
  not staged, modified, or removed.
- No Slurm command was run and no Slurm job was submitted.
