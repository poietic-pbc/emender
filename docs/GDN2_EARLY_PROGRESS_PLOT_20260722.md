# GDN2 Early-Progress Smoothed-Loss Plot Smoke

Date: 2026-07-22 UTC
Task: `test-early-progress`

## Scope

Performed a plotting-only smoke for the live GDN2-MLP 8-GPU DiLoCo control.
No training process was stopped, restarted, signaled, or reconfigured. No S3
write/upload command was run. No checkpoint was modified.

The authoritative handoff named in the task,
`docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md`, is not tracked in this worktree;
it was read from the base checkout as:

```text
/home/erikg/ndm/docs/EMENDER_DILOCO_OPS_HANDOFF_20260722.md
```

The GDN2 launcher is likewise untracked in the base checkout and was read
read-only from:

```text
/home/erikg/ndm/scripts/launch_gdn2_mlp_8gpu_diloco.sh
```

## Live Run Verified

Run identity:

```text
run_id: gdn2_mlp_full_20260722T083424Z
run_root: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp
run_log: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log
run_dir: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/runs/gdn2_gdn2-mlp_1.3B_20260722_083444
```

The launch manifest and `args.json` confirmed:

```text
level: gdn2-mlp
world_size: 8
per_gpu_batch_size: 4
chunk_size: 2048
grad_accum: 1
log_every: 25
save_every: 500
tokens_per_step: 65,536
```

Token conversion followed the ops handoff exactly:

```text
tokens = optimizer_step * (world_size * batch_size * chunk_size * grad_accum)
tokens = optimizer_step * (8 * 4 * 2048 * 1)
tokens = optimizer_step * 65,536
```

## Process Health

Before plotting, read-only `ps` and `nvidia-smi pmon -c 1` showed:

```text
torchrun PID: 3754241
wrapper bash PID: 3754406
worker rank PIDs: 3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616
GPU clients: one python3 worker on each GPU 0-7, 99% SM utilization on all GPUs
```

After plotting, the same torchrun and worker-rank PIDs were still present:

```text
torchrun PID: 3754241
wrapper bash PID: 3754406
worker rank PIDs: 3754608, 3754609, 3754611, 3754612, 3754613, 3754614, 3754615, 3754616
GPU clients: one python3 worker on each GPU 0-7, 98-99% SM utilization
```

Checkpoint stat values were unchanged before and after plotting for the visible
stable checkpoints:

```text
checkpoint_step_000500_loss_5.8263.pt 7720576663 bytes mtime 2026-07-22 08:43:45.243086915 +0000
checkpoint_step_001000_loss_5.3572.pt 7720576663 bytes mtime 2026-07-22 08:52:45.467074401 +0000
checkpoint_step_001500_loss_4.9854.pt 7720576663 bytes mtime 2026-07-22 09:01:44.580061914 +0000
checkpoint_step_002000_loss_4.6663.pt 7720576663 bytes mtime 2026-07-22 09:10:41.282049482 +0000
```

Only `run.log` grew naturally while training continued.

## Plot Protocol

The canonical E97 plotter is:

```text
scripts/plot_e97_diloco_loss.py
```

Compatibility finding: its regex parser is compatible with the GDN2 log lines,
but its built-in plot is E97/step-axis specific: it labels the figure as E97
and plots training step on the x-axis. Because the task required a GDN2
token-versus-smoothed-loss plot and prohibited unreviewed repo edits, I used a
one-off bounded plotter under the run `ops` directory. It uses the same parser
shape, lineage deduplication, and smoothing formula as the documented E97
protocol.

Bounded input:

```text
snapshot: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/run.log.snapshot.tail_2097152_20260722T0916Z.log
snapshot_size_bytes: 21181
bounded_tail_bytes: 2097152
source: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/run.log
```

Smoothing matched the handoff:

```text
method: trailing moving average over effective plotted loss records
window: min(80, max(5, effective_point_count // 40))
effective_point_count: 95
moving_average_window: 5
```

## Reproducible Command

The exact plotting command was:

```bash
RUN_ROOT=/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp
RUN_DIR="$RUN_ROOT/runs/gdn2_gdn2-mlp_1.3B_20260722_083444"
OPS="$RUN_ROOT/ops/test-early-progress_20260722T0916Z"
LOG="$RUN_ROOT/run.log"
SNAP="$OPS/run.log.snapshot.tail_2097152_20260722T0916Z.log"
PLOT="$OPS/gdn2_mlp_early_progress_tokens_smoothed_loss_20260722T0916Z.png"
METRICS="$OPS/gdn2_mlp_early_progress_metrics_20260722T0916Z.json"
SCRIPT="$OPS/plot_gdn2_early_progress.py"

mkdir -p "$OPS"
stat -c '%n %s %Y %y' "$LOG" > "$OPS/pre_snapshot_log_stat.txt"
tail -c 2097152 "$LOG" > "$SNAP"
python "$SCRIPT" \
  --snapshot "$SNAP" \
  --run-root "$RUN_ROOT" \
  --run-dir "$RUN_DIR" \
  --tokens-per-step 65536 \
  --output "$PLOT" \
  --metrics "$METRICS"
```

The `SCRIPT` file used above is stored outside git at:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/plot_gdn2_early_progress.py
```

## Result

Generated plot:

```text
path: /mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/gdn2_mlp_early_progress_tokens_smoothed_loss_20260722T0916Z.png
size_bytes: 147865
mtime: 2026-07-22T09:17:45.179040+00:00
sha256: b62ccdd1373f6005687a96ad4a6aed448a38a740fc17ee0e3722a08dc8c3a333
readability_check: matplotlib read OK, shape=(1200, 2240, 4), dtype=float32, min=0.000000, max=1.000000, mean=0.989579
```

Parser metrics:

```text
raw_points: 95
effective_points: 95
superseded_points: 0
raw_step_range: 25..2375
token_range: 1,638,400..155,648,000
latest_step: 2375
latest_tokens: 155,648,000
latest_time: 2026-07-22T09:17:18+00:00
latest_raw_loss: 4.5753
latest_smoothed_loss: 4.538960000000012
checkpoint_steps_visible: 500, 1000, 1500, 2000
```

Sanity checks passed:

```text
nonfinite_effective_points: 0
nonmonotonic_steps: 0
nonmonotonic_tokens: 0
duplicate_steps_removed: 0
duplicate_induced_corruption: false
malformed_step_like_lines: 0
dropped_final_partial_line: false
final_record_rejected: false
```

## Repository State

Operational outputs were written only under:

```text
/mnt/nvme1n1/erikg/diloco_8gpu/gdn2_mlp/ops/test-early-progress_20260722T0916Z/
```

No plot or parser temporary artifact was written inside the git repository.
The only intentional repository change is this report. The pre-existing
untracked `.wg` entry in this worktree was preserved and not staged.
