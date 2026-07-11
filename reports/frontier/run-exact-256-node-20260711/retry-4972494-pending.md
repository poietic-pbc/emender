# Exact 256-node E97 async smoke retry: job 4972494

Date: 2026-07-11

## Post-fix immutable bundle

The reviewed on-node preflight fix from `fix-immutable-e97` commit `b71831f`
was incorporated before rendering. The three mandatory commands were then run
literally after loading the site `python` module, and all exited zero:

```text
python scripts/frontier/render_e97_async_256.py --profile smoke --out build/e97-256/smoke
python scripts/frontier/render_e97_async_256.py --profile production --out build/e97-256/production
python scripts/frontier/check_e97_async_promotion.py --smoke build/e97-256/smoke --production build/e97-256/production --policy configs/frontier/e97_async_256_parity_policy.json
```

Both renders emitted fingerprint
`ecf537fdd8f69add499cf4330abaa8670cd079aa09d746e0f2ce2b898bd5037d`.
The parity checker returned:

```json
{"ok": true, "fingerprint": "ecf537fdd8f69add499cf4330abaa8670cd079aa09d746e0f2ce2b898bd5037d", "allowed_differences": {"time": ["00:20:00", "12:00:00"], "partition": ["batch", "batch"], "qos": ["debug", "normal"]}}
```

Thus account `bif148` is identical, the partition is identical, QoS and the
deterministically derived duration/stop budget are the only differences, and
reservation remains absent.

## Exact submission

No `sbatch --export` or other command-line overrides were used. The exact
command was:

```text
sbatch /lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-949/build/e97-256/smoke/rendered.sbatch
```

Slurm accepted job `4972494`. `scontrol show job -dd` confirmed:

- command is the content-addressed smoke `rendered.sbatch` above;
- account `bif148`, partition `batch`, QoS `debug`;
- walltime `00:20:00`;
- exactly 256 nodes and 2048 tasks, 8 tasks/node, 7 CPUs/task, and 1 GPU/task;
- `SubmitLine` is exactly the command recorded above;
- no dependency or reservation.

At `2026-07-11T19:26:23Z`, the job remained `PENDING (Priority)`. A subsequent
`squeue --start` estimated `2026-07-11T16:54:00-04:00`. The job has not yet
received an allocation, so on-node preflight, rank, update, loss, merge,
checkpoint, and final Slurm acceptance evidence do not yet exist. No
`promotion.json` was written and no production job was submitted.

This is an in-progress execution record, not promotion evidence. The runner
must resume monitoring job `4972494`, retain its output and manifests, and
write promotion only if every exact-topology acceptance criterion passes.
