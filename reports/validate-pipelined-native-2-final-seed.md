# Final-seed pipelined native DiLoCo two-node acceptance

Date: 2026-07-23  
Task: `validate-pipelined-native-2-final-seed`  
Status: clean five-generation job 5062348 pending; scheduler wait checkpoint

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
G2 refresh was submitted. It is now terminal:

```text
5062165|COMPLETED|0:0|00:02:52|batch|debug|2|frontier[08634,08641]
```

The retained full-layout artifact passes its `SHA256SUMS` verification and
records source `2e485fa7`, bundle
`9884a02d84bd9560a15314c26e386868350b865e688cc4c701c802f4f686227a`,
manifest `4292f332ecdef76f5dff57788408418f5714e61ad2202f7d2e8b70aa4f572552`,
provider `cxi`, two READY endpoints, zero MPI/all-rank collectives, and the
exact `batch/debug` scheduler binding.

With the user queue empty, the serial fail-closed controller rebuilt the same
bundle (10/10 CTests), re-attested it against G2, fully revalidated the
submit-side checkpoint cache, and submitted exactly one clean phase:

```text
5062348|resilient-e97-true-2n|PENDING|batch|debug|2|0:00|(Priority)
```

`sacct`, `squeue`, and `scontrol` agree on two requested nodes,
`Partition=batch`, `QOS=debug`, zero runtime, and five K40 generations.
Controller state names `clean-overlap` as its only active phase and has empty
history. No duplicate, fault/rejection/checkpoint/restart phase, or allocation
larger than two nodes has been submitted. This task is parked until 5062348 is
terminal.

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

For job 5062348, the submit cache is a regular, single-link file of exactly
`7719680116` bytes. Its digest-addressed basename and the pinned attestation
bind SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`,
step 2300930, 150793748480 tokens, and byte-identical live authority
documents. The attestation SHA-256 is
`274465e2676b99cc0154388454643f49207c04874aed80a41f474c07c99d96cb`.
The exact rendered batch script SHA-256 is
`8992a017b26ddfc090ff69170171a593a8047935e0edd84be531e16db007c730`;
a source audit finds zero S3/HTTP URL or network-client reference. It retains
the literal deferred template, expands it using the live `SLURM_JOB_ID`,
sbcasts both pinned files, and requires every node-local size/hash/authority
verifier to succeed before exporting the model seed.

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
sacct: COMPLETED 0:0, 2 nodes, Partition=batch, QOS=debug
sha256sum -c g2-artifacts/5062165/SHA256SUMS
PASS (6/6)
render_resilient_e97_exact_2n_acceptance.py ... --submit
SUBMITTED phase=clean-overlap job_id=5062348
squeue/sacct/scontrol: PENDING, 2 nodes, Partition=batch, QOS=debug
```

Five K40 generations, overlap and idle/cadence metrics, useful/wire bytes,
loss/rejoin recovery, invalid-result rejection, failed checkpoint publication,
and fresh restart are not yet claimed. The serial phases remain gated behind
the clean five-generation result.
