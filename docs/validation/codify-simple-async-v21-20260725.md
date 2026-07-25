# Simple asynchronous DiLoCo v2.1 authority handoff

Date: 2026-07-25 UTC  
Task: `codify-simple-async-v21`  
Reviewed design commit:
`c3d687dab3b1eb56e6b3ceb1258a16e6d4dfe04d`

## Outcome

The normative asynchronous policy is now
`async-decoupled-v2.1-simple`. ADR-002 was reduced from 716 to 385 lines and
now serves as the implementation authority rather than a record of the
over-specified v2.0 experiment.

The synchronized authority set is:

- [Resilient DiLoCo Compute Pool](../RESILIENT_DILOCO_COMPUTE_POOL.md)
- [ADR-002: simple asynchronous DiLoCo v2.1](../ASYNC_DECOUPLED_DILOCO_V2.md)
- [Native resilient DiLoCo data plane v1](../NATIVE_RESILIENT_DILOCO_DATAPLANE.md)
- [Resilient DiLoCo traceability and bounded backlog](../RESILIENT_DILOCO_GAP_MATRIX.md)
- [historical async-quorum handoff](../ASYNC_QUORUM_DILOCO.md)

The gap matrix is the only normative definition matrix for V21S01–V21S17.
Every row has a preserved R/NDP mapping, required test/evidence, and bounded-gap
column. The union of its base mappings is exactly R01–R16 and NDP01–NDP17.
The compute-pool checklist is separately cross-walked and remains mandatory;
the V21 rows do not substitute for a base requirement.

Historical `async-decoupled-v2.0-exp`, V2A01–V2A18, hard lag 6/8,
lag-adjusted/separate aggregation weight, and half-step outer update remain
honest only in dated reports. They are incompatible with v2.1 and cannot be
renamed, migrated, used as a fallback, or treated as qualification/promotion
evidence. No retained historical report was rewritten.

## Fixed policy and architecture

The reviewed authority fixes K40; independent commit, applied-anchor,
result-version, and speculative-window clocks with maximum two; lag-3
drop/catch-up; exact tokens as the only quantitative quorum, token clock, and
numerical weight; and stateless `eta_outer=1.0` exact-token averaging. The exact
two-node profile retains `Q_min=2`, `T_min=3,934,080`, no active fraction,
zero generation retry, and explicit READY/K/group/OWNED/correctness/catch-up
deadlines.

Persistent training remains bounded to one immutable native-owned
eight-trainer descriptor plus one mutable adjacent interval. The K-boundary
accepted-delta ledger translates ScheduleFree `x`, `z`, and the mutable
interval start exactly once. A node advertises READY for an applied version
only after all eight trainer recovery markers agree; a partial apply restarts
all eight from verified latest. The verified latest mailbox has capacity one,
atomic replacement, bounded staging, and no result FIFO.

The conservative E97 native-service admission remains
`64,001,671,648` bytes under the 64-GiB cap, with no third dense cohort.
R01–R16 and NDP01–NDP17 continue to require a fenced leased READY pool,
model-free compiled memfd/XPMEM plus exact-CXI point-to-point dense path,
bounded credits/replay/release, no launched-rank/all-rank wait, no
Python/Lustre dense hot path, no central full-model broker, and atomic
checkpoint/latest publication. The v2.1 boundary is
`NDP_ABI_V21=0x00020001`, wire protocol 2.1, with distinct policy,
contribution, and committed-manifest schema identities.

## Seed, gates, and promotion

Cold start remains the exact retained authority:

- checkpoint:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/checkpoint_step_2300930_loss_2.4365.pt`
- step manifest:
  `s3://spinozans/emender/e97-diloco/emender_E97_1.3B_20260709_084606/step_2300930/manifest.json`
- discovery pointer:
  `s3://spinozans/emender/e97-diloco/latest_emender_E97_1.3B.json`
- step `2300930`; accepted tokens `150793748480`; size `7719680116`;
  SHA-256
  `0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2`.

The submit side verifies authority bytes and the full checkpoint SHA, publishes
a digest-pinned attestation, and uses `sbcast` to stage checkpoint and
attestation at `/tmp/emender-e97-seed-${SLURM_JOB_ID}`. Every node verifies the
local bytes and emits `network_fetches=0` before a model-bearing role starts.

Numerical, clean/performance, fault/restart, deterministic-replay, and
predeclared three-seed convergence gates must all pass at exactly two nodes
with `Partition=batch` and `QOS=debug`. Only a separate pass review may
authorize four nodes. The serial ladder is
`4 -> 8 -> 16 -> 32 -> 64 -> 256`, with an immutable pass from each immediate
predecessor.

Scale closure is not the two-node `Q_min=2` condition. Promotion must pin a
finite formula over the leased READY snapshot, include every complete
admissible pre-close arrival, ignore launched ranks, and show explicit
close/deadline/cadence arithmetic from digested passing two-node arrival/stage
distributions. Until such evidence and review exist, 4+ rendering, preflight,
and submission fail closed.

## Validation

No runtime code or training behavior changed. No native build, renderer,
preflight, `sbatch`, `srun`, `salloc`, `scancel`, or other scheduler-mutating
command was run.

The canonical environment was activated before the Python authority check:

```bash
source scripts/frontier/activate_emender_frontier.sh
```

The exact activated authority command was:

```bash
"$EMENDER_PYTHON" - <<'PY'
from pathlib import Path
import re

files = [Path(p) for p in (
    'docs/RESILIENT_DILOCO_COMPUTE_POOL.md',
    'docs/ASYNC_DECOUPLED_DILOCO_V2.md',
    'docs/NATIVE_RESILIENT_DILOCO_DATAPLANE.md',
    'docs/RESILIENT_DILOCO_GAP_MATRIX.md',
    'docs/ASYNC_QUORUM_DILOCO.md',
    'docs/validation/codify-simple-async-v21-20260725.md',
)]
for path in files:
    text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
    for raw in re.findall(r'\[[^\]]+\]\(([^)]+)\)', text):
        if raw.startswith(('http://', 'https://', '#', 'mailto:')):
            continue
        target = raw.split('#', 1)[0].strip('<>')
        assert not target or (path.parent / target).resolve().exists(), (
            path, target
        )

matrix = Path('docs/RESILIENT_DILOCO_GAP_MATRIX.md').read_text()
for prefix, end in (('R', 16), ('NDP', 17), ('V21S', 17)):
    for i in range(1, end + 1):
        assert len(re.findall(
            rf'^\| {prefix}{i:02d} \|', matrix, re.M
        )) == 1
rows = [
    line.split('|')[1:-1]
    for line in matrix.splitlines()
    if line.startswith('| V21S')
]
assert len(rows) == 17
assert all(
    len(row) == 5 and all(cell.strip() for cell in row)
    for row in rows
)
base = '\n'.join(row[2] for row in rows)
assert set(re.findall(r'(?<!NDP)R\d\d', base)) == {
    f'R{i:02d}' for i in range(1, 17)
}
assert set(re.findall(r'NDP\d\d', base)) == {
    f'NDP{i:02d}' for i in range(1, 18)
}

adr = Path('docs/ASYNC_DECOUPLED_DILOCO_V2.md').read_text()
active, historical = adr.split('## Historical v2.0 disposition', 1)
assert all(term not in active for term in (
    'exact_tokens * (7 - commit_lag)', 'eta=0.5',
    'tau_hard', 'sigma_hard',
))
assert all(term in active for term in (
    'async-decoupled-v2.1-simple', 'NDP_ABI_V21 = 0x00020001',
    'K = 40', 'Q_min = 2', 'T_min = 3,934,080',
    'eta_outer = 1.0', '64,001,671,648',
    '4 -> 8 -> 16 -> 32 -> 64 -> 256', 'step `2300930`',
    'accepted tokens `150793748480`', 'size `7719680116`',
    '0239706e1f67e4823008a3a2754894b5b94dc1663580d2e40c1c74f7dd6a72b2',
    'network_fetches=0', 'Partition=batch', 'QOS=debug',
))
assert all(term in historical for term in (
    'async-decoupled-v2.0-exp', 'V2A01–V2A18',
    'exact_tokens * (7 - commit_lag)', 'eta=0.5',
))
assert Path('AGENTS.md').read_bytes() == Path('CLAUDE.md').read_bytes()
print('authority and handoff links: PASS')
print('requirement rows and complete base crosswalk: PASS')
print('v2.1 active contract and v2.0 incompatibility: PASS')
print('AGENTS.md/CLAUDE.md lock-step: PASS')
PY
```

The activated check performed these assertions:

```text
- every local Markdown link in the five synchronized documents resolves;
- R01–R16, NDP01–NDP17, and V21S01–V21S17 each have exactly one matrix row;
- no V2A row remains in the current implementation matrix;
- every V21 row has nonempty requirement, base-map, test/evidence, and
  bounded-gap cells;
- the V21 base-map union is exactly R01–R16 plus NDP01–NDP17;
- active ADR text contains no v2.0 lag-weight/half-step contract;
- the policy, ABI, K/floors/eta, resident bound, exact seed, queue fields,
  offline bootstrap, and ordered scale ladder are pinned;
- the historical section explicitly retains the incompatible v2.0 identity
  and math; and
- AGENTS.md and CLAUDE.md are byte-identical.
```

Result:

```text
authority links: PASS
requirement rows: PASS (R01-R16, NDP01-NDP17, V21S01-V21S17 exactly once)
base crosswalk coverage: PASS
v2.1 constants/incompatibility/seed/scale assertions: PASS
AGENTS.md/CLAUDE.md lock-step: PASS
```

The exact final shell checks are:

```bash
git diff --check
cmp -s AGENTS.md CLAUDE.md

for id in $(seq -w 1 17); do
  test "$(rg -c "^\\| V21S${id} \\|" \
    docs/RESILIENT_DILOCO_GAP_MATRIX.md)" = 1
done

test "$(sed -n '/^| V21S/p' docs/RESILIENT_DILOCO_GAP_MATRIX.md \
  | rg -o 'R[0-9]{2}|NDP[0-9]{2}' | sort -u | wc -l)" = 33
```

The final pushed `origin/main` SHA and remote equality check are retained in
the WG task log after publication because a commit cannot embed its own
content-derived SHA.
