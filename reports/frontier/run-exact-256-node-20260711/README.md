# Exact 256-node E97 async smoke attempt: job 4972201

Date: 2026-07-11

## Result

The mandatory machine parity gate passed and the exact content-addressed smoke
script was submitted without command-line options or environment overrides:

```text
sbatch build/e97-256/smoke/rendered.sbatch
```

Slurm accepted job `4972201` with fingerprint
`24e0fdbff99b4fcfb3b4ff729ff26a372d66db4a4b166730e92ff62526d5fb8e`.
The allocation was exactly 256 nodes and 2048 tasks (8 tasks/node, 7 CPUs/task,
1 GPU/task), account `bif148`, partition `batch`, QoS `debug`, and walltime
`00:20:00`.

The job ended `FAILED 127:0` after six seconds, before `srun`. It is therefore
not promotable and no `promotion.json` was written.

## Mandatory gate

The initial literal invocation failed because the login environment had no
`python` command. After loading the site Python module, the three required
commands ran literally and exited zero. Both profiles rendered fingerprint
`24e0fdbff99b4fcfb3b4ff729ff26a372d66db4a4b166730e92ff62526d5fb8e`.
Canonical parity evidence reported only:

- walltime: `00:20:00` versus `12:00:00`
- partition: `batch` versus `batch`
- QoS: `debug` versus `normal`

The account remained `bif148` and the canonical configuration has no
reservation.

## Failure evidence

The batch stderr is:

```text
sha256sum: /var/spool/slurmd/job4972201/fingerprint.sha256: No such file or directory
cat: /var/spool/slurmd/job4972201/fingerprint-file.sha256: No such file or directory
scripts/frontier/build_compiled_mpich_dense_helper.sh: line 38: CC: command not found
```

This exposes two pre-`srun` defects in the immutable launcher:

1. `BUNDLE_DIR` is derived from `BASH_SOURCE[0]`. Slurm executes a spooled copy
   of the batch script, so it resolves to `/var/spool/slurmd/job4972201`, not
   the submitted content-addressed bundle. The fingerprint sidecars are not
   present there.
2. The script requests `#SBATCH --export=NONE` but does not restore the declared
   module/runtime environment before invoking the helper build. `CC` is absent.

The fingerprint check is followed by `|| exit 72` without grouping the
pipeline's two missing-file diagnostics into a successful preflight. The job
continued to helper compilation and then failed with exit 127. It never reached
the exact-rank launcher, so there are no rank starts, accepted updates, loss,
merge, metrics, or finalized/reloaded checkpoint acceptance artifacts.

The external seed pointer was hashed before and after the attempt and remained
unchanged. No smoke `promotion.json` exists, and no production job was
submitted.

## Artifacts

- `pre-submit-gate.log`: both the missing-interpreter attempt and successful
  mandatory gate
- `rendered.sbatch`, `fingerprint.sha256`, and copied immutable manifests
- `sbatch-command.txt` and `submission.txt`
- `scontrol-submit.txt`, `sacct-submit.txt`, `sacct-final.txt`
- `job-4972201.out`, `job-4972201.err`, and `live-monitor.log`
- `external-pointer-before.txt` and `external-pointer-after.txt`
