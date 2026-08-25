# E97 4B matched-clock 32-node qualification preflight

**Decision:** authorized by the attended operator as a bounded debug-QoS test.
This is a new random-init qualification lineage and cannot resume or enter the
stopped 2,048-rank B5/K25 lineage.

## Scientific question

Does restoring the Lambda eight-GPU seed workflow's aggregate optimizer clock
repair the Frontier from-scratch loss curve?

The operator reports that the Lambda 4B B32/K32 eight-GPU run reaches about
3 nats by 2B tokens. The Frontier qualification factorizes the same aggregate
clock across 32 nodes:

```text
Lambda:   8 ranks * B32 * 2048 = 524,288 tokens/update
Frontier: 256 ranks * B1 * 2048 = 524,288 tokens/update

Lambda:   524,288 * K32 = 16,777,216 tokens/merge
Frontier: 524,288 * K32 = 16,777,216 tokens/merge
```

The rank-local batch factorization differs and remains part of the experiment.
The corpus also remains the pinned Frontier CommaPile mainmix, so absolute loss
is not asserted to be directly interchangeable with an unverified remote log.

## Bound configuration

- 32 nodes / 256 ranks / eight tasks per node / one GCD per task;
- `Partition=batch`, `QOS=debug`, `Requeue=0`;
- walltime 02:00:00;
- B1, context 2,048, K32;
- 2,048 fixed steps / 1,073,741,824 aggregate tokens;
- checkpoint every 256 steps, hence every checkpoint and terminal target is
  K-aligned;
- ScheduleFree LR `0.00047431158698290157` unchanged;
- random initialization, sampler key 42;
- private node-local Triton cache per job and global rank;
- immutable source/config/payload identity and terminal collector.

Expected use is below 64 node-hours because the fixed target should finish
before the two-hour allocation limit. No automatic continuation is authorized.

## Architecture scope

This is ADR-003 fixed-world qualification. Applicable safety intent is R07,
R12, R14/NDP13, R16, and NDP15 checkpoint atomicity. Elastic research
R02--R06/R08--R11 and NDP02, async-v2.1 V21S01--V21S17, ISP01--ISP07, native
NDP17, and the production scale ladder are explicitly unclaimed. The job is a
new reviewed topology for a bounded scientific probe, not a scale-ladder pass.

## Validation

```text
source scripts/frontier/activate_emender_frontier.sh
bash -n scripts/frontier/submit_e97_4b_from_scratch.sh \
  scripts/frontier/e97_4b_from_scratch.sbatch \
  scripts/frontier/e97_4b_from_scratch_collector.sh
"$EMENDER_PYTHON" -m pytest -q tests/test_frontier_e97_4b_from_scratch.py
```

Result: 5 passed. Submission still must retain queued/running and terminal
records naming both `Partition` and `QOS` explicitly.
