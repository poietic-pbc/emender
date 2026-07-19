# Persistent native live service v1: recovery audit and bounded implementation

**Task:** `complete-native-live-service-v1`  
**Date:** 2026-07-19  
**Authority:** Resilient DiLoCo Compute Pool v1 and Native resilient DiLoCo
data plane v1. No Slurm command or job was submitted.

## Retry hardening: descriptor control-message cardinality

The recovered producer-direct `SOCK_SEQPACKET` boundary now parses every file
descriptor in an `SCM_RIGHTS` control message and rejects a packet unless it
contains exactly one descriptor. Previously, a control message containing two
packed descriptors was interpreted as only its first descriptor, leaking the
second into the service process and violating bounded admission. Received
descriptors are also marked close-on-exec atomically where Linux provides
`MSG_CMSG_CLOEXEC`, and JSON boolean/string coercions are no longer accepted for
fence, generation, attempt, extent, weight, or sequence fields.

The regression launches the listener and sender in separate processes, sends
two sealed memfds in one metadata-only packet, and proves fail-closed rejection.
This strengthens R03/R05/R08/R10/R14 and NDP01/NDP04/NDP06/NDP08/NDP13/NDP14.
It does not close NDP03/NDP14: the accepting service is still Python-owned and
the compiled local ABI remains a process-local singleton. Production therefore
remains fail-closed. No Slurm job was submitted.

## Outcome

This pass selectively recovered two sound partial commits and hardened the
producer-direct descriptor boundary, but it did **not** complete the requested
persistent native service. Production therefore remains correctly fail-closed.
The remaining gap must not be relabeled as complete: `libemender_ndp.so.1`
still stores its `Service` singleton inside each loading process, and
`ndp_cxi_service` owns only the fabric endpoint. The Python split-role launcher
does not yet join those components into one persistent per-node native owner.

Recovered commits:

* `d5e0437` was recovered as `b7260d7`: frozen owner-frame transfer stays on
  the compiled transport ABI and permits only the initial send plus two
  identity-stable replays.
* `3715db4` was recovered as `750f27d`: fenced checkpoint approval performs one
  native COMMIT transition and introduced metadata-only `SOCK_SEQPACKET` /
  `SCM_RIGHTS` producer memfd transfer.
* `cd7b068` was intentionally not cherry-picked. It changed only a report and
  claimed validations that its preceding implementation did not perform.

Commit `0f3e762` adds receiver-side exact-seal, run, fence, generation,
attempt, accepted trainer incarnation, sequence, positive weight, exact extent,
layout digest, payload SHA-256, and finite-f32 validation. A node-local atomic
metadata ledger rejects an admitted identity after a service-side object
restart, including identical replay and conflicting reuse. Dense bytes are
read only through the received descriptor; the seqpacket contains JSON
metadata and one `SCM_RIGHTS` fd. The positive regression forks the manager
receiver into a separate process.

## Validation performed

Approved Python:
`/lustre/orion/bif148/scratch/erikgarrison/emender/.envs/olcf-rocm711-torch210-py312/bin/python`
(Python 3.12.13).

Retry validation on 2026-07-19 sourced the canonical activation script from
`origin/main@92b37d4`. The first clean discovery run correctly failed one of
143 tests because the installed native manifest identified the preceding
source commit. Rebuilding with
`scripts/frontier/build_native_resilient_dataplane.sh` at `413fff0` passed all
8 CTests and refreshed that attestation. The subsequent clean discovery run
over every current `tests/test_native*.py` and `tests/test_resilient*.py` file
passed **143/143 in 204.57 seconds**. This retry did not submit Slurm work and
does not change the architectural gaps or fail-closed conclusion below.

* Unified normal build and CTest: `scripts/frontier/build_native_resilient_dataplane.sh build/native-resilient-dataplane` — 8/8 passed.
* ASan/UBSan configure/build/CTest with `NDP_ENABLE_SANITIZERS=ON` — 8/8
  passed in the native Cray environment. Repeating under the full activated
  ROCm environment passed 5/8; the three libfabric-using processes were failed
  only by LeakSanitizer reporting the same 61,512 bytes retained by external
  `libhsa-runtime64.so.1` initialization (no project allocation stack). Address
  and undefined-behavior checks emitted no project finding. This environment
  difference remains explicit rather than disabling leak detection.
* Focused handoff and stuck-peer rerun — 7/7 passed.
* Broad native/resilient suite — 156 passed and two failures on the first run.
  The native failure was a deliberately stale pre-commit build manifest and
  passed after rebuilding at the implementation commit. The stuck-process
  timing test was affected by two earlier orphaned pytest invocations; after
  terminating those exact test PIDs it passed alone. A correctly activated
  clean rerun reached 157/158 and reproduced only that loaded-node SIGTERM
  delay. The regression now uses the production-shaped bounded escalation
  `terminate -> join -> kill -> join`, preventing orphaned synthetic peers;
  the final correctly activated broad rerun passed 158/158 in 179.18 seconds.

### Final retry validation

The final retry repeated validation after sourcing the canonical Frontier
activation script for every Python/build command. The first broad invocation
was intentionally allowed to exercise source attestation and rejected the
stale pre-commit native build manifest (142 passed, one attestation failure).
Rebuilding at `413fff0` with
`PYTHON_BIN="$EMENDER_PYTHON" scripts/frontier/build_native_resilient_dataplane.sh
build/native-resilient-dataplane` passed normal CTest 8/8 and emitted a current
manifest. A clean rerun of every current `tests/test_native*` and
`tests/test_resilient*` module then passed **143/143 in 211.59 seconds** under
Python 3.12.13. This retry submitted no Slurm command or job. The passing rerun
validates the landed component and fail-closed behavior; it does not change the
NDP03/NDP14 persistent compiled-service gap described below.

## Required conformance checklist

Compute Pool v1 R01–R16 was checked. This pass advances R03–R05, R08–R10,
R14–R15. R01–R02, R06–R07, R11–R13, and R16 are unchanged control-plane or
retained-G2 mechanisms. R08 and R10 remain nonconforming in the live path
because the owner transfer and redistribution are not driven by one external
native service. The minimum progress floor remains explicit `Q_min` and
`T_min`; no launched-rank invariant or collective was added.

Native NDP01–NDP17 was checked. NDP04, NDP06, NDP08, NDP10–NDP11,
NDP13–NDP15 gain component evidence. NDP01, NDP03, NDP07, NDP09, NDP12,
NDP16, and NDP17 remain incomplete for the live split-role path. In particular:

* READY/freeze policy remains in Python and every descriptor binds the current
  fenced generation identity.
* The mapped descriptor boundary is bounded and does not write trainer-sized
  files or carry dense bytes through Python TCP.
* Frozen fabric replay is bounded to two reassignments and native COMMIT is
  exactly once after full result-root/fence publication validation.
* One shared native result view exists in the component lifecycle, but live
  owner-direct redistribution and trainer apply are not wired across the
  external service boundary.
* Production `native-cxi` and `native-test` selection still stops in
  `_require_wired_dense_runtime` before `manager()` or `trainer()` can
  construct `LocalTrainerSpool` or `DistributedOwnerServer`. This is the
  required safe state, not completion evidence.

## Remaining blocking implementation

Move the local `Service` state and C ABI command execution into the persistent
`ndp_cxi_service` process (or a single joined service binary), implement the
complete metadata RPC with fd passing for controller and trainer clients, and
make handles service-owned across client process lifetimes. Then wire frozen
accepted-set owner send/receipt/reassignment, owner-direct redistribution,
shared result mapping, release, and fenced checkpoint proposal into the live
manager/trainer roles. Only after the separate trainer/manager/service
regression and a clean full focused suite pass may the fail-closed guard be
opened. No real E97 or additional Slurm gate is authorized by this pass.
