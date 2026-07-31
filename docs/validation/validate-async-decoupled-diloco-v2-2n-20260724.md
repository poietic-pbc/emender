# Async decoupled DiLoCo v2 exact-two-node validation

Date: 2026-07-24

Task: `validate-async-decoupled-diloco-v2-2n`

Status: **FAILED — do not authorize scale-out.**  The refreshed exact-source
G2 prerequisite passed, but the required-success `clean-overlap` phase failed
in Slurm job `5065388` before generation 0 could publish.  The reviewed serial
gate therefore stopped: no fault, invalid-result, failed-publication, restart,
or 4+ node job was submitted.  The serial state deliberately continues to name
`5065388` as its sole active job so that a broken terminal-harvest path cannot
cause a duplicate submission.

## Authority and checklist

The run is governed by:

- `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, version 1;
- `docs/RESILIENT_DILOCO_GAP_MATRIX.md`;
- `docs/ASYNC_DECOUPLED_DILOCO_V2.md`, ADR-002.

The acceptance manifest cites every authoritative v1 requirement,
**R01–R16**, every native requirement, **NDP01–NDP17**, and every async-v2
requirement, **V2A01–V2A18**.  In expanded form, the v1 compute-pool set is
R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R11, R12, R13, R14,
R15, and R16; the native set is NDP01, NDP02, NDP03, NDP04, NDP05, NDP06,
NDP07, NDP08, NDP09, NDP10, NDP11, NDP12, NDP13, NDP14, NDP15, NDP16, and
NDP17; and the async-v2 set is V2A01, V2A02, V2A03, V2A04, V2A05, V2A06,
V2A07, V2A08, V2A09, V2A10, V2A11, V2A12, V2A13, V2A14, V2A15, V2A16,
V2A17, and V2A18.

The run plan is exactly:

1. `clean-overlap`: generations 0–12, including at least five K40 windows and
   the strict cadence/idle admission;
2. `fault-rejoin`: generations 12–15, native-service loss at generation 13;
3. `invalid-result-rejection`: generations 15–17, injected invalid result at
   generation 16;
4. `checkpoint-publication-failure`: generation 17, expected allocation
   failure without partial publication;
5. `fresh-restart`: generations 17–19 under a fresh fence, resuming from the
   last authoritative handoff.

The controller submits at most one phase per invocation.  Subsequent phases
were not submitted after `5065388` failed the clean admission.

## Current merged source

The authoritative source was fetched and verified before any submission:

```text
git fetch origin main
git rev-parse HEAD origin/main HEAD^{tree} origin/main^{tree}
git ls-remote origin refs/heads/main

HEAD        243b94c7e791fad873964ddf622c95dacc6a936f
origin/main 243b94c7e791fad873964ddf622c95dacc6a936f
remote main 243b94c7e791fad873964ddf622c95dacc6a936f
tree        fef57ca571c2c3740e6a629a643c5b0755d01858
```

The submit checkout is
`/lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-243b94c7e791-main`.
It is on branch `main`, clean including untracked files, and byte-identical to
current `origin/main`.  The acceptance source inventory contains every tracked
file and has canonical inventory digest
`c5e22feaeccfa58be97fdf6e0e9d3b5db330a51a537aa33f22163a06bd521995`.

All Python, native-build, and launch preparation commands ran after:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The activation selected:

```text
EMENDER_PYTHON=/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python
Python 3.12.13
```

## Refreshed source-pinned native G2

The previously retained G2 artifact was source- and bundle-pinned to older
code, so it was not reused as current-source evidence.  The exact current
source was rebuilt with:

```bash
source scripts/frontier/activate_emender_frontier.sh
REPO=$PWD \
SOURCE_DIR=$PWD/native \
BUILD_DIR=/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-build \
INSTALL_DIR=/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-install \
PYTHON_BIN="$EMENDER_PYTHON" \
scripts/frontier/build_native_resilient_dataplane.sh
```

Configuration, build, install, and all 10 CTests passed.  The resulting clean
manifest binds:

```text
source_commit  243b94c7e791fad873964ddf622c95dacc6a936f
bundle_sha256  9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a
manifest SHA   21c4a0b6bd3354449dadf10b7f6f247c05b4c89d0fa08889719fe8f1a440d200
```

The prerequisite was then concretely submitted, not merely inspected:

```bash
source scripts/frontier/activate_emender_frontier.sh
REPO=$PWD \
NDP_BUILD_MANIFEST=/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-install/native-artifacts.json \
NDP_ARTIFACT_ROOT=/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-evidence \
NDP_PYTHON_BIN="$EMENDER_PYTHON" \
scripts/frontier/submit_native_dataplane_2n_gate.sh clean
```

Job `5065331` completed in `00:02:59`, `ExitCode=0:0`, with exact accounting
evidence `AllocNodes=2`, `Partition=batch`, and `QOS=debug`.  The refreshed
gate passed on provider `cxi`, with two leased endpoints, eight trainer lanes
per node, zero MPI/all-rank barriers, zero Python dense-socket bytes, zero
trainer-spool bytes, bounded high-water memory, and post-release transport
bytes equal to zero.  Its median full-layout transfer plus redistribution was
23.586323650 seconds.  The immutable gate is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-evidence/5065331/full-layout-gate.json
SHA256 08213db1aeb19a95da7f125c37b38d908068d897d834b36838825465eb959afa
```

## Submit-side seed and authoritative render

The real acceptance controller was invoked exactly as follows:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" scripts/frontier/render_resilient_e97_exact_2n_acceptance.py \
  --repo "$PWD" \
  --native-build-manifest /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-install/native-artifacts.json \
  --full-layout-gate /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/native-g2-evidence/5065331/full-layout-gate.json \
  --run-root /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-runs \
  --output /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-manifest.json \
  --state /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-state.json \
  --native-stage-root /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-native-stage \
  --submit
```

The controller performed a second clean current-source build.  It reproduced
the G2 bundle digest exactly and passed exact-gate attestation before
submission.  The authoritative stage manifest and inventory are:

```text
acceptance manifest SHA256       fb419a373d8f5e0ff8f349d92275b167e836c0672bbd0ed55ccf2b3964dee699
authoritative stage SHA256       31aed0271070cae68576e8dcfb79dd539c8e280139799211323eb4ed763d39ba
native build manifest SHA256     21c4a0b6bd3354449dadf10b7f6f247c05b4c89d0fa08889719fe8f1a440d200
native bundle SHA256             9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a
rendered batch script SHA256     2a5a19144be83b7e207a80b1528a3b3d34d91c76c2208355cc504f5d59c1681e
```

Before `sbatch`, the submit host resolved and cross-checked the immutable S3
step manifest and latest pointer, re-hashed the entire content-addressed cold
cache, and retained:

```text
step              2300930
seed tokens       150793748480
seed bytes        7719680116
seed SHA256       0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2
cache reuse       true, after complete size/digest verification
attestation SHA   27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb
```

The terminal run did execute node-local `sbcast` and offline verification.
Both node records are retained and independently bind the exact submit-side
attestation, checkpoint size, and checkpoint digest with
`network_fetches=0`; their exact identities and digests are reported below.

## Slurm submission and pending history

At the 2026-07-24 06:16 EDT observation:

```text
JobID   State    Reason    Nodes  Partition  QOS    Start estimate
5065388 PENDING  Priority  2      batch      debug  2026-07-24 08:28 EDT
```

`scontrol show job -dd 5065388` independently reported:

```text
JobState=PENDING Reason=Priority
Account=bif148 QOS=debug
Partition=batch
NumNodes=2-2
ReqTRES=cpu=2,mem=1000G,node=2,billing=2
```

The scheduler submit line is exactly:

```text
sbatch --parsable -N 2 -t 02:00:00 -p batch --qos=debug
       --network=job_vni
       --chdir /lustre/orion/bif148/proj-shared/emender/source-snapshots/emender-243b94c7e791-main
       [immutable exports]
       /lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-runs/clean-overlap/rendered.sbatch
```

The immutable exports bind `RESILIENT_E97_NODE_COUNT=2`,
`RESILIENT_E97_GENERATIONS=12`, K40, `Q_min=2`, `T_min=3934080`,
`tau_hard=6`, `tau_target=2`, `sigma_hard=8`, `sigma_target=2`,
`eta_outer=0.5`, provider `cxi`, the exact source/native/G2 identities above,
and the step-2300930 seed identity.  The serial state currently contains:

```text
next_phase=0
active.phase=clean-overlap
active.job_id=5065388
active.scheduler_request.partition=batch
active.scheduler_request.qos=debug
history=[]
```

### Pending-job resume observation

At 2026-07-24 07:22:45 EDT (11:22:45 UTC), the resumed worker fetched
`origin/main` again and verified both the remote-tracking ref and
`ls-remote origin refs/heads/main` still resolve to
`243b94c7e791fad873964ddf622c95dacc6a936f`. It did not submit another job.
Slurm and the serial controller state independently continued to identify
`5065388` as the sole active phase:

```text
squeue: JobID=5065388 State=PENDING Nodes=2 Partition=batch QOS=debug
        Reason=Priority StartTime=2026-07-24T08:48:00
sacct:  JobID=5065388 State=PENDING NNodes=2 Partition=batch QOS=debug
scontrol: NumNodes=2-2 Partition=batch QOS=debug
state: next_phase=0 active.phase=clean-overlap active.job_id=5065388 history=[]
```

The serial state SHA-256 at this observation is
`1b71e28c0216395b6c835056ea6b03e55b2b149e2d81177290c68ec216df3e37`.
The acceptance manifest, rendered batch script, seed attestation, G2 gate, and
native build manifest retained the exact digests reported above. At that
pending observation, per-node offline seed verification and all runtime
criteria remained pending; no result was inferred from queued state.

## Terminal clean-phase evidence

Slurm started the only real acceptance allocation at 07:56:14 EDT and ended it
at 08:07:16 EDT.  Exact terminal accounting was collected with:

```bash
sacct -j 5065388 -X -n -P \
  --format=JobIDRaw,JobName,State,ExitCode,NNodes,Partition,QOS,Submit,Eligible,Start,End,Elapsed,Timelimit,Reason,NodeList
```

and returned:

```text
5065388|resilient-e97-true-2n|FAILED|1:0|2|batch|debug|2026-07-24T06:15:13|2026-07-24T06:15:13|2026-07-24T07:56:14|2026-07-24T08:07:16|00:11:02|02:00:00|None|frontier[06764-06765]
```

This is explicit `Nodes=2`, `Partition=batch`, and `QOS=debug` evidence; QoS
is not inferred from the partition column.  `squeue -u "$USER"` was empty
afterward.  No duplicate or later phase was submitted.

### Per-node offline seed verification

The allocation copied the already submit-verified step-2300930 seed to
job-scoped `/tmp/emender-e97-seed-5065388/` storage and verified it without a
network fetch on both nodes:

| Node | Step | Size (bytes) | Checkpoint SHA-256 | `network_fetches` | Record SHA-256 |
|---|---:|---:|---|---:|---|
| `frontier06764` | 2300930 | 7,719,680,116 | `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2` | 0 | `85fcb03ffb4efc7304c512ce30825b3f25b0f087e852a3a8cfe11744cf2efd97` |
| `frontier06765` | 2300930 | 7,719,680,116 | `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2` | 0 | `9ef0e260b12a284b3dc8e1e9d57123909af5bddc80877aa445418f7a28c5da9a` |

Both records bind the submit-side attestation SHA-256
`27e234891df02b64b9db77fc784c341e5a3ae6e87418b8f1af167776d1d710bb`.
The retained paths are
`acceptance-runs/clean-overlap/seed-materialization/frontier06764.json` and
`frontier06765.json` under the validation root.

### Honest partial K-window and lag telemetry

The scheduler output attests the intended topology:

```text
topology managers=2 real_trainers=16 trainers_per_node=8 local_steps=40 collective=none
```

The control plane froze exactly the two leased READY peers
`node-0/b38004841442492ab69fce33887a6c73` and
`node-1/5f5fe1520bca47a4a06e27bd11b0b44c`.  Generation 0 reached the
in-memory `commit_ready` freeze condition with `Q=2`,
`accepted_tokens=5,245,440`, and no active-membership fraction.  This status is
not a durable model commit.

Each of the 16 trainers transferred one immutable `[0,1)` descriptor with:

```text
exact_tokens=327,840
aggregation_weight=2,294,880
base_global_version=0
base_lag_at_seal=0
window_count=1
python_dense_socket_bytes=0
trainer_spool_bytes=0
```

The frozen totals were therefore exact tokens `5,245,440` and v2 aggregation
weight `36,718,080`.  Both node managers reported
`commit_global_version=0`, `commit_lag=0`, exact node tokens `2,622,720`, and
node aggregation weight `18,359,040`.  There is no apply version,
applied-anchor lag, result-version lag, accepted durable commit, or checkpoint
identity to report because the owner-result metadata failed before
redistribution/publication.  No lagged work is relabeled fresh.

Only two sequential K40 windows completed per trainer (local windows 0 and 1);
the third had 30–34 of 40 local steps when the allocation stopped.  Thus the
32 lane-window completions are not five sequential windows per lane and cannot
satisfy the reviewed admission.  The partial observations were:

| Metric | Retained partial value |
|---|---:|
| complete raw K observations | 32 |
| raw K duration, min / median / max | 65.576216 / 69.441028 / 73.908250 s |
| consecutive-boundary observations | 16 |
| boundary cadence, min / median / max | 65.578323 / 67.181019 / 67.801226 s |
| cadence/raw ratio, min / median / max | 1.000027x / 1.000030x / 1.000035x |
| derived foreground idle, min / median / max | 0.001796 / 0.002029 / 0.002334 s |
| derived foreground-idle fraction, min / median / max | 0.0000268 / 0.0000302 / 0.0000346 |
| local `OWNED` acknowledgment, min / median / max | 3.674575 / 3.677498 / 3.776824 s |
| node-local f64 reduction, min / median / max | 19.757644 / 19.846174 / 19.934704 s |
| owner-contribution exchange, min / median / max | 39.886482 / 39.891159 / 39.895837 s |

The cadence and idle samples show genuine overlap for one interval, but the
background path was not healthy, only two full windows existed, and local
`OWNED` exceeded ADR-002/V2A16's one-second bound on every trainer.  These
numbers therefore do **not** constitute a performance pass even though the
partial cadence and idle ratios are numerically inside their limits.

The native path resolved exact provider `cxi`.  Per-manager retained maxima
were `useful_tx_bytes=5,510,614,144`,
`useful_rx_bytes=5,510,614,144`, `wire_tx_bytes=5,510,667,264`,
`wire_rx_bytes=5,510,667,264`, `in_flight_high_water=67,109,184`,
`retained_high_water=67,109,184`, and
`released_bytes=11,013,646,912`.  The owner receive queue high-water was one
frame.  Python dense-socket, trainer-spool, handoff-full-copy, replay, retry,
route-error, checksum-reject, nonfinite-reject, stale-reject, duplicate, and
conflict counters remained zero up to the failure.  Two CQ cancellation errors
per service appeared during failure drain.  Zero rejection counters are not a
substitute for the unsubmitted invalid-input phase.

### First failing invariant

Both managers completed the node reduction and full CXI owner-contribution
exchange, then failed with the identical traceback:

```text
RuntimeError: owner result metadata is invalid
```

The retained values isolate the production bug.  The v2 native attempt-2
result correctly reports global aggregation weight `36,718,080`, the sum of
the two frozen node aggregation weights.  However,
`PoolControlServer._owner_result` still applies the v1-only predicate:

```text
weight != close.accepted_tokens
```

where `close.accepted_tokens=5,245,440`.  In v1 those quantities are equal; in
ADR-002 v2 they are intentionally separate (`exact_tokens * (7-commit_lag)`).
The valid v2 owner metadata is consequently rejected before publication.  The
supervisor restarted both managers and their native services twice with new
incarnations; the restarted services then encountered `FREEZE: invalid
lifecycle state`, and both managers emitted `restart_exhausted`.  This does
show dynamic reincarnation rather than a launched-rank wait, but it is a clean
gate failure, not the reviewed injected loss/rejoin pass.

No immutable global checkpoint, checkpoint manifest, authoritative `latest`,
or publication artifact exists in this phase.  The frozen contribution record
is retained at
`acceptance-runs/clean-overlap/retained-evidence/pool-control/generation-00000000.jsonl`,
SHA-256
`20be749950a773bc9434d1f89b1a741c1b4459f0c4b4167569bc99eab37b388c`.
This absence correctly prevents fresh-allocation continuation from being
claimed.

### Terminal-controller defects and no-duplicate state

The exact serial controller was concretely rerun with the same manifest,
state, native build, G2 gate, and `--submit` arguments.  It refused before any
`sbatch` in two fail-closed steps:

1. Slurm's default scheduler stdout/stderr paths were inside the authoritative
   checkout, making it unclean.  The two untracked logs were preserved, not
   deleted, as `clean-overlap/scheduler-5065388.out` and `.err` with SHA-256
   `b13c7d909a32836d5d28d18922ef2bf531234c92d7695979ee844cf7c55c9bac`
   and
   `257c83f8754b07b704818fa1f848c04395d88ecd4d0b8d2edf08dc39792c15f1`;
   the authoritative checkout is clean again.
2. On the clean retry, terminal job `5065388` was absent from `squeue`, whose
   nonzero `Invalid job id` exit escapes `_scheduler_state` before its `sacct`
   fallback.  The controller returned 64:

   ```text
   acceptance launcher refused: Command '['squeue', '-h', '-j', '5065388',
   '-o', '%T|%P|%q']' returned non-zero exit status 1.
   ```

The acceptance state remains unchanged at SHA-256
`1b71e28c0216395b6c835056ea6b03e55b2b149e2d81177290c68ec216df3e37`,
with `next_phase=0`, `active.job_id=5065388`, and empty history.  This stale
active record is intentionally retained because clearing it manually could
permit a duplicate or out-of-order submission.  The terminal state is instead
preserved by the exact `sacct` record, this report, and the phase evidence.

Two bounded fixes and a dependency-joined rerun were filed:
`fix-async-v2`, `fix-exact-2n-2`, and `re-run-async`.  No retry is authorized
until both fixes are reviewed and merged, current-source native/G2 evidence is
rebuilt, and the serial run starts from a new clean validation root.

### Retained evidence identity

The immutable phase root is:

```text
/lustre/orion/bif148/proj-shared/emender/validation/validate-async-decoupled-diloco-v2-2n-243b94c7e791/acceptance-runs/clean-overlap
```

After preservation of the scheduler logs it contains 172 files and 7,633,229
bytes.  A sorted inventory hashed records in the exact form
`relative-path NUL decimal-size NUL file-sha256 LF`; its SHA-256 is
`9aa4f17b330675c1b16a951d2326b408551c2f296dc124a1dabe17c79fe9925b`.
Additional anchors are:

| Artifact | SHA-256 |
|---|---|
| `runtime-identity.json` | `5c0fa27593fa1b173a57512eaee13e8dcb8bfadfb1c1b3783fae4457e02c6805` |
| `native-dataplane-launch-attestation.json` | `aa212a510e8c32778931a482feb4edf4df6c2b832ed9f88014ba9ba78329266d` |
| `supervision/allocation-lease.json` | `f6245a0a46bd33e5831ea45edb328d97e51898932b56b5192ebbb3f26fe45c14` |
| `supervision/events.jsonl` | `00fdc742eb85ab1293f72e076945dbe471f0419ff5172ac6b9192d00fcecab4f` |
| `logs/node-0-manager.err` | `316bc893b5d16d1593b3a597f1e4c5fd4035e3a381c192ed1d53882cbf2daa6d` |
| `logs/node-1-manager.err` | `316bc893b5d16d1593b3a597f1e4c5fd4035e3a381c192ed1d53882cbf2daa6d` |
| `control/pool-v1.sqlite3` | `b9216e1b9c66966a1b780359f3283e03457219585c54df9372c05dac5fc4da4d` |

## Validation

- [x] Cited V2A01–V2A18 and authoritative v1 R01–R16/NDP01–NDP17.
- [x] Verified current merged `origin/main`, exact source/tree/inventory
  digests, a clean authoritative `main`, and an exact reproducible native
  bundle.
- [x] Refreshed and passed the current-source G2 prerequisite as job `5065331`
  on exactly two `batch/debug` nodes.
- [x] Submit-side step-2300930 authority, size, and complete digest
  verification passed.
- [x] Real phase `5065388` ran on exactly 2 nodes, `Partition=batch`,
  `QOS=debug`; no 4+ or duplicate job was submitted.
- [x] Submit-side and both per-node step-2300930 digest/size checks passed
  after `sbcast`, with `network_fetches=0`.
- [ ] Five sequential local K40 windows did not complete: only two per trainer
  completed before generation-0 owner metadata failed.  Honest partial
  base/commit, token/weight, queue/memory/release, provider/byte, membership,
  cadence, and idle telemetry is reported above; apply/checkpoint telemetry
  does not exist.
- [ ] The clean performance admission failed.  One partial cadence interval
  was numerically within 1.25x and 10%, but the background path failed and
  every local `OWNED` acknowledgment exceeded one second.
- [ ] Leased READY membership, dynamic reincarnation, exact CXI, zero Python
  dense/spool bytes, and collective-free topology were observed, but the
  generation did not commit and restart exhausted.
- [ ] Loss/rejoin, duplicate/stale/corrupt/nonfinite/wrong-fence rejection,
  owner recovery, failed publication, and fresh-allocation continuation were
  not submitted after the prerequisite clean phase failed.
- [x] Exact commands, Slurm IDs/states/timings, report paths/digests, the
  absence of a checkpoint manifest, and the fail-closed verdict are retained.

Verdict: **FAILED.  Scale-out is not authorized.**  Job `5065388` found a
v2 exact-token/aggregation-weight control-plane mismatch before the first
durable commit; the strict gate stopped in the reviewed order.  Fix and
current-source G2 qualification must precede the dependency-joined two-node
rerun.  This artifact does not authorize manually clearing the serial state or
submitting any later phase.
