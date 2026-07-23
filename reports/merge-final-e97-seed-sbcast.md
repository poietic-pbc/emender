# Final E97 sbcast bootstrap integration review

This review integrates the submit-side E97 seed bootstrap from staged commit
`03e75b1501008cac0632ed62b548ee5fdd4e8c5b` with authoritative
`origin/main` at `f6003e32e14b89e0fde1f6b7f47b6402285d7b39`. The conflict-free
ancestry-preserving integration commit is
`f913289e0751cfb2aeb94d6036827500a26f06ef`. No Slurm job was submitted and
the production checkpoint was not downloaded during this review.

## Reviewed safety properties

Submit-side acquisition in
`scripts/frontier/materialize_e97_s3_seed.py::prefetch` takes an exclusive
digest-specific `flock`, removes interrupted temporaries while holding that
lock, and revalidates both live authority documents before accepting either a
cache hit or a new download. A cache hit is accepted only after checking that
it is a regular, single-link file with the exact byte count and SHA-256. A new
download is written to a same-directory temporary, hashed while streaming,
flushed and fsynced, renamed to the digest-addressed final name, followed by a
directory fsync and a second full identity verification. Incorrect existing
entries are removed under the lock and reacquired. The attestation embeds the
exact authority bytes, their digests and parsed values, and is itself published
with temporary-file fsync, atomic replacement, and directory fsync. Authority
drift, malformed authority bytes, partial downloads, cache corruption, aliases,
and concurrent acquisition therefore fail closed or converge on one completely
verified content-addressed object.

The exact submit driver calls `prefetch` immediately before `sbatch` and exports
only the verified cold-cache path, pinned attestation path, and attestation
digest. The batch bootstrap contains no S3 URI, HTTP URL, or network-fetch
client. It requires executable Slurm `sbcast`, rejects an existing job-scoped
directory, creates `/tmp/emender-e97-seed-${SLURM_JOB_ID}` on every allocated
node, broadcasts the checkpoint and attestation, and launches exactly one
offline verifier per node. Each verifier checks the live job ID, rejects a
shared checkpoint filesystem, requires regular single-link bytes with exact
size/SHA, checks the attestation-file digest, replays both authority-byte and
parsed-field validations, and emits `network_fetches: 0`. Only after all
verifiers succeed is `RESILIENT_E97_SEED` exported and the allocation
supervisor reached. Trainers consequently consume only the node-local `/tmp`
copy; the Lustre cold cache is never a trainer input or resilient dense
hot-path location.

The immutable seed remains step `2300930`, accepted tokens `150793748480`,
size `7719680116`, and SHA-256
`0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.
The exact request remains separately pinned to `Partition=batch` and
`QOS=debug`.

## Conformance

This integration was checked against *Resilient DiLoCo Compute Pool*, version
1, and its companion gap matrix:

- **R01**: seed verification completes before any model load; allocation
  admission fencing remains downstream and unchanged.
- **R07**: checkpoint and attestation publication are fsynced, atomic,
  immutable/content-addressed, and fail closed.
- **R09**: the manager remains model-free; only trainers later load the
  node-local verified checkpoint.
- **R10**: Lustre is cold bootstrap/evidence storage only. Dense update,
  aggregation, heartbeat, membership, and redistribution paths are unchanged
  and remain non-Lustre.
- **R13**: Slurm-specific `sbcast` is confined to the Frontier batch adapter;
  the resilient protocol remains scheduler-neutral.
- **R14**: every bootstrap operation is ordered before role startup, fails
  closed, and leaves digest/job/node evidence.
- **R16**: the request remains the exact two-node debug rung and grants no
  permission to scale beyond it.
- **NDP13**: bootstrap errors are contained before local role groups start;
  existing absolute native-stage deadlines are unchanged.
- **NDP15**: the read-only seed handoff is digest pinned while Python retains
  checkpoint admission/publication policy and collective-free drain is
  unchanged.
- **NDP16**: immutable seed identity, authority digests, attestation digest,
  job identity, host identity, local byte identity, and zero network fetches
  are recorded.
- **NDP17**: the full-layout G2 prerequisite and ordered two-node-before-scale
  ladder remain intact; this integration does not claim a new execution rung.

## Validation

The canonical environment was activated with
`source scripts/frontier/activate_emender_frontier.sh`; every Python command
used `"$EMENDER_PYTHON"` (Python 3.12.13).

```text
"$EMENDER_PYTHON" -m py_compile \
  scripts/frontier/materialize_e97_s3_seed.py \
  scripts/frontier/render_resilient_e97_exact_2n_acceptance.py
PASS

bash -n scripts/frontier/resilient_e97_true_2n.sbatch
PASS

"$EMENDER_PYTHON" -m pytest -q \
  tests/test_e97_s3_seed.py \
  tests/test_resilient_e97_exact_2n_acceptance.py
38 passed

rg source audit of resilient_e97_true_2n.sbatch for urllib, requests, curl,
wget, aws s3, HTTP URLs, and S3 URIs
PASS: no matches

rg exact identity/queue audit
PASS: step/tokens/size/SHA exact; PARTITION=batch; QOS=debug
```

The bounded tests cover successful atomic publication and cache reuse,
concurrent acquisition, corrupt-cache reacquisition, live-authority drift,
authority-attestation corruption, wrong job identity, missing/corrupt
node-local copies, scheduler queue evidence, and batch ordering. They use mock
bytes and subprocess fixtures only.
