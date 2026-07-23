# Final-seed pipelined native DiLoCo two-node acceptance

Date: 2026-07-23  
Task: `validate-pipelined-native-2-final-seed`  
Status: sbcast retry in progress; exact-source G2 refresh job 5062165 pending

## Retained terminal attempt

Job 5060027 is terminal `FAILED 1:0` after 00:08:38 on exactly two nodes with
`Partition=batch` and `QOS=debug`. Its exact `f6003e32` native bundle and G2
attestation passed before both compute nodes timed out fetching the S3
authority document. No trainer or model load occurred, zero K40 generations
ran, and no later phase or larger allocation was submitted.

The reviewed submit-side sbcast integration supersedes that invalid
compute-node network dependency. Because the integration changed authoritative
source identity, the older bundle and G2 gate cannot be reused.

## Exact-source sbcast retry

A fresh clean authoritative checkout was created at:

```text
/lustre/orion/bif148/scratch/erikgarrison/emender-exact2n-final-seed-sbcast-20260723T200500Z/source
```

Its `HEAD` and `origin/main` both resolve to
`2e485fa70588f6b5b764416e8efac3dcfa6aaee4`. The canonical Frontier
environment was activated with the approved Python 3.12 environment. A clean
native rebuild passed all 10 CTests. The installed manifest records clean
source `2e485fa7`, bundle SHA-256
`9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a`,
and manifest SHA-256
`4292f332ecdef76f5dff57788408418f5714e61ad2202f7d2e8b70aa4f572552`.

After an immediately adjacent empty-user-queue check, exactly one prerequisite
G2 refresh was submitted:

```text
5062165|PENDING|batch|debug|2|0:00|(Priority)|native-ndp-g2-clean
```

`sacct` independently records job 5062165 as `PENDING`, zero elapsed, exactly
two nodes, `Partition=batch`, and `QOS=debug`. No replacement K40 job, later
serial phase, duplicate, or allocation larger than two nodes has been
submitted. This task is parked until the exact-source G2 job is terminal.

## Seed bootstrap proof

The merged batch bootstrap contains no S3 URI, HTTP URL, or network-fetch
client. Submit-side `prefetch` locks a digest-addressed cache, revalidates both
live authority documents, and accepts a regular single-link cache entry only
after checking exact size `7719680116` and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The immutable identity remains step 2300930 and 150793748480 accepted tokens.

The rendered batch script retains the literal
`'/tmp/emender-e97-seed-${SLURM_JOB_ID}'`, expands it only under the live
allocation ID, uses compiled `sbcast` for both the checkpoint and pinned
authority attestation, and runs one offline verifier per node. Each verifier
checks the live job ID, node-local filesystem/path, checkpoint size/SHA,
attestation digest and authority bytes, and records `network_fetches: 0`
before `RESILIENT_E97_SEED` is exported or the allocation supervisor can load
a model.

## Conformance and validation

Conformance was checked against *Resilient DiLoCo Compute Pool*, version 1,
and the companion gap matrix: R01-R16 and NDP01-NDP17. The exact-source,
two-node, explicit queue, bounded-wait, native-bundle, immutable-seed,
node-local handoff, and ordered-rung constraints are satisfied for this
pending prerequisite. Runtime-only requirements remain unclaimed until G2
passes and the clean five-generation K40 phase completes.

Commands and observations:

```text
source scripts/frontier/activate_emender_frontier.sh
scripts/frontier/build_native_resilient_dataplane.sh
10/10 CTests passed
squeue -u $USER -h
empty immediately before submission
NDP_BUILD_MANIFEST=... NDP_ARTIFACT_ROOT=... \
  bash scripts/frontier/submit_native_dataplane_2n_gate.sh clean
5062165
squeue/sacct: PENDING, 2 nodes, Partition=batch, QOS=debug
```

Five K40 generations, overlap and idle/cadence metrics, useful/wire bytes,
loss/rejoin recovery, invalid-result rejection, failed checkpoint publication,
and fresh restart are not yet claimed. The serial phases remain gated behind
the clean five-generation result.
