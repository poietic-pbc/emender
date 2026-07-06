# Validate Refreshed E97 Seed 4n20m Async DiLoCo Debug

Task: `validate-refreshed-seed-4n20m`

Decision: **PASS for the `submit-refreshed-e97-async-256n12h` gate**. Do not treat this report as a 256-node submission record; this task submitted no 256-node job.

## Launch Gate Evidence

- Checkout used for the passing run: `/lustre/orion/bif148/scratch/erikgarrison/emender`, branch `main`, commit `80af9cdfc7b1105f63dc2fa81ed46dd1345c1305`, matching `origin/main` after the wrapper fix commit.
- Wrapper: `scripts/frontier/async_diloco_e97_2n8n_debug.sbatch`.
- Queue/partition: `-p batch -q debug`. Frontier exposes `batch` as the partition and `debug` as the QoS.
- Refreshed seed manifest: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/seed_manifest.json`.
- Refreshed seed latest: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt`.
- Seed resolved path: `/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/checkpoint_step_1065000_loss_2.5386.pt`.
- Seed size/SHA from manifest: `7719679924` bytes, `c68ea2d95f2721f1f52664f71c1453e4f30a5520b33eb1cf54974185e5a100a4`.
- Non-production output root: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m`.
- Production latest guard: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt`.

## Attempts

### 4946475

Result: failed before run directory creation.

- Slurm state: `FAILED`, exit `127:0`, elapsed `00:00:05`, nodes `4`.
- Cause: wrapper computed `REQUESTED_NODE_HOURS` with `python` before conda activation; login/compute shell only exposed `python3` before activation.
- Fix committed and pushed: `80af9cd` (`fix: avoid preactivation python in async debug wrapper (validate-refreshed-seed-4n20m)`).

### 4946476

Result: failed as unsafe induced-drop probe.

- Slurm state: `FAILED`, exit `1:0`, elapsed `00:00:41`, nodes `4`.
- Evidence: async entrypoint ran from `main` commit `80af9cd` and used the refreshed `latest.pt`.
- Cause: `ASYNC_INDUCE_LOCAL_LAG_DROP=1` with prescribed `ASYNC_LOCAL_QUORUM=8` left `accepted=7 quorum=8` and failed local quorum. This demonstrates that local induced drop is not safe under the required `local_quorum=8` setting. The passing validation was therefore run without induced drop.

### 4946479

Result: pass.

Exact launch command:

```bash
sbatch -N 4 -p batch -q debug -t 00:20:00 --export=ALL,WG_TASK_ID=validate-refreshed-seed-4n20m,TASK_ID=validate-refreshed-seed-4n20m,REFRESHED_E97_SEED_LATEST=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,SEED_LATEST_PATH=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,E97_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/checkpoints/emender_E97_1.3B_20260702_111457_step_1065000/latest.pt,TRAINING_TARGET=E97_1.3B_step1065000_async_diloco_debug_4n20m_20260706,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m,ASYNC_NODE_COUNT=4,ASYNC_GLOBAL_QUORUM=4,ASYNC_LOCAL_QUORUM=8,ASYNC_RECOVERY_EVERY_GENERATIONS=1,ASYNC_RESUME_CHECK=1,PRODUCTION_LATEST_GUARD=/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt scripts/frontier/async_diloco_e97_2n8n_debug.sbatch
```

Slurm accounting:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|AllocTRES
4946479|async-diloco-e97-2n8n|COMPLETED|0:0|00:00:51|4|billing=448,cpu=448,energy=95465,mem=2000G,node=4
4946479.batch|batch|COMPLETED|0:0|00:00:51|1|cpu=56,mem=500G,node=1
4946479.extern|extern|COMPLETED|0:0|00:00:51|4|billing=448,cpu=448,mem=2000G,node=4
```

The wrapper-recorded requested node-hours were `1.333333` for 4 nodes and a 20-minute walltime. Actual Slurm elapsed was 51 seconds.

## Passing Artifacts

- Metrics JSON: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/artifacts/async_diloco_e97_4n_metrics.json`
- Command record: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/artifacts/command.txt`
- Environment record: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/artifacts/env.txt`
- Run log: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/logs/async_diloco_e97_2n8n.log`
- Slurm stdout: `logs/frontier/async_diloco_e97/async-diloco-e97-2n8n-4946479.out`
- Slurm stderr: `logs/frontier/async_diloco_e97/async-diloco-e97-2n8n-4946479.err`
- Summary: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/summaries/summary.md`
- Run directory: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run`

## Metrics Summary

Metrics conclusion: `pass`.

- Training target: `E97_1.3B_step1065000_async_diloco_debug_4n20m_20260706`.
- Seed checkpoint before/after in metrics: unchanged, resolved to refreshed step `1065000`; `modified_by_run=false`.
- Configured quorum: nodes `4`, local quorum `8`, global quorum `4`, workers per node `8`.
- Effective quorum: global `4`; local by node `8,8,8,8`; local distribution min/max/average `8/8/8.0`.
- Update counts: global accepted `4`, timed out `0`, stale `0`, failed `0`, invalid `0`; local total accepted `32`, timed out `0`, stale `0`, failed `0`, invalid `0`.
- Timing: elapsed `17.965699925087392s`, global merge `1.370503478916362s`, global rebase `0.0s`, checkpoint `0.0001664359588176012s`.
- Local merge timings: node-000 `1.7457728979643434s`, node-001 `1.6225364869460464s`, node-002 `1.7181060181465s`, node-003 `1.6810850698966533s`.
- Tokens/sec: aggregate `1823.9200329870034`, global `26909.923285877092`; local node range `6727.480821469273` to `7122.534810590973`.
- Loss windows: global `loss_100=0.9900000000000001`; each local node also recorded `loss_100=0.9900000000000001`.
- Checkpoint finalization: duration `0.0001664359588176012s`, overhead `0.01366814844907546%`, total recorded checkpoint size `16685` bytes, debug latest advanced to run-local `async_run/latest.json`.

## Manifests And Recovery

Generation manifests:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run/generations/gen_000000/manifest.json`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run/generations/gen_000001/manifest.json`

Recovery checkpoint records:

- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run/recovery_checkpoints/gen_000000/initial.json`
- `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run/recovery_checkpoints/gen_000001/initial.json`

Resume-from-latest result:

- `resume_check.tested=true`.
- Selected generation `0`, published generation `1`.
- Resume latest path: `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_diloco_e97_refreshed_seed_4n20m/20260706/4946479-20260706T094716Z/async_run/latest.json`.
- Resume metrics: accepted updates `4`, timed out `0`, stale `0`, failed `0`, merge `1.266438404796645s`, rebase `0.0s`, checkpoint `0.00016333907842636108s`, resume source generation `0`, tokens/sec `26909.923285877092`.

## Production Latest Guard

External before and after evidence matched.

Before:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|7719679569|1782849877|594487543672603835|regular file
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|231|1783092774|594487587176043644|symbolic link
```

After:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|7719679569|1782849877|594487543672603835|regular file
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260630/E97_1.3B_step483000_b4_k80_64n_hier_g4_bucket64m_avg_24h/4911454-20260630T164035Z/train/emender_E97_1.3B_20260630_124257/checkpoint_step_489920_loss_2.4894.pt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/chains/E97_1.3B_step489920_b4_k80_64n_hier_g4_bucket64m_avg/latest.pt|231|1783092774|594487587176043644|symbolic link
```

Metrics also recorded `production_latest_guard.changed=false`.

## 256n12h Gate

Gate decision: **go** for the downstream 256-node 12-hour submission task, subject to that task preserving its own explicit hard gates and human approval requirements.

Rationale:

- The refreshed E97 seed path from `register-refreshed-e97-seed-runner` was used and verified.
- The async entrypoint ran from integrated `origin/main`.
- The passing run used only the non-production debug output root.
- Production `latest.pt` was unchanged before and after.
- Metrics recorded quorum, update counts, timings, checkpoint size/overhead, throughput, and loss windows.
- Generation manifests and recovery checkpoint records exist for the initial generation and the resume generation.
- Resume-from-latest in the non-production run directory was tested and passed.
- No 256-node job was submitted by this task.
