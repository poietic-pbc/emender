# E97 35B MoE stage-1 continuation to 150.134B accepted tokens

Date: 2026-08-08

## Decision

The operator explicitly authorizes only stage 1 of the proposed continuation:
resume the immutable 100.474B canonical authority and train to exactly
150,134,063,104 accepted tokens. A continuation toward 200.466B is not
authorized by this decision and requires a separate review of the stage-1
systems, loss, routing, and held-out evidence.

```text
nodes=256
ranks=2048
partition=batch
qos=normal
time_limit=06:00:00
max_steps=2960
train_minutes=0
diloco_k=40
save_every=160
keep_checkpoints=2
requeue=0
```

The restored complete eight-shard checkpoint is:

```text
step=2329120
accepted_tokens=100473503744
checkpoint=step-02329120-tokens-0000100473503744
manifest_sha256=933d35abec874d5c88dcb31fbd05815ef47c9b0e508563158e4c26afc46b5550
```

It is independently protected by a local hard-linked milestone and a verified
complete S3 publication. The exact arithmetic is:

```text
2960 * 16777216 = 49660559360 new accepted tokens
100473503744 + 49660559360 = 150134063104 final accepted tokens
2329120 + 2960 = 2332080 final step
2960 / 40 = 74 K40 merges
```

With `save_every=160`, the epoch has 18 periodic publications and one exact
final publication, for 19 canonical checkpoint publications. The six-hour
request is only a safety envelope; scientific stopping is the exact positive
step boundary. Based on job 5201882, expected elapsed time is about five hours
and expected allocation use is about 1,280 node-hours; the maximum request is
1,536 node-hours.

This epoch retains the historical sampler identity already present in the
100.474B chain (`42 + restored_starting_step + global_rank`). It does not claim
the future counter-based sampler schema. Introducing that schema inside this
stage would confound continuation and is therefore deferred to an explicitly
recorded boundary after this authority.

## Stop and evidence policy

Any rank, node, collective, timeout, nonfinite value, HIP error, or checkpoint
publication failure terminates the fixed world nonzero. There is no automatic
restart, communicator shrink, scheduler requeue, emergency checkpoint, or
promotion to stage 2. Only complete K-aligned canonical checkpoints are restart
authority.

After successful completion, preserve the complete 150.134B checkpoint as an
immutable hard-linked milestone before retention can remove it. Evaluate at
minimum:

- terminal scheduler state with `Partition=batch`, `QOS=normal`, and exit `0:0`;
- exactly 2,960 new steps, 74 merges, and 19 checkpoint publications;
- finite language-model and routing auxiliary losses;
- ordinary and effective throughput, HBM high-water, merge tails, and
  checkpoint tails;
- the final complete eight-shard manifest at step 2,332,080 and
  150,134,063,104 accepted tokens;
- fixed held-out quality and expanded routing telemetry when available.

Stage 2 toward 200,465,711,104 tokens remains a separate operator decision.

## Validation and architecture conformance

Authority is `docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, ADR-003 production
same-allocation execution epochs (2026-07-31), and the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. Applicable safety intent is **R07**,
**R12**, **R14/NDP13**, **R16**, and **NDP15** checkpoint atomicity. The
operator explicitly reviewed this 256-node continuation. The rendered role is
one bounded fixed-world child, has no SQLite/database/lock/metadata-heartbeat
control path, and does not attempt to preserve, shrink, or automatically
relaunch a broken communicator.

Explicitly retired and unclaimed are **R02-R06, R08-R11; NDP01,
NDP03-NDP12, NDP14, NDP16-NDP17; V21S01-V21S17; and ISP01-ISP07**. This run
makes no elastic membership, native data-plane, asynchronous overlap,
background-checkpoint, or communicator-shrink claim.

Pre-submission validation must include:

```bash
source scripts/frontier/activate_emender_frontier.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_e97_moe_production_launcher.py
bash -n scripts/frontier/e97_35b_moe_production.sbatch
bash -n scripts/frontier/submit_e97_35b_moe_scale.sh
sha256sum \
  /lustre/orion/bif148/proj-shared/emender/frontier_runs/e97-35b-moe/scale-ladder-mainfc52/checkpoints/latest/manifest.json
```

The submitter must retain live scheduler output that separately names
`Partition=batch`, `QOS=normal`, 256 nodes, and `06:00:00`; the job must also
verify `Requeue=0` before model load.
