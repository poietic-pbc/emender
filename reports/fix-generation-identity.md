# Generation-identity recovery handoff fix

Date: 2026-07-21

## Production diagnosis

The retained `native-generation-00000001.json` records from Frontier job
5039258 were valid when their managers published them.  On node 0 the record
was written at unix second `1784585346` with generation deadline
`1784585715282347119`, leaving about 369 seconds.  Its fence, generation,
attempt, owner epoch, element count, three digests, and exact runtime digest
attestation were all valid.

The initiating failure happened earlier, at generation-0 result cleanup.
Trainer 7 had already consumed and applied its read-only native result mapping,
but `ndp_buffer_release_v1` returned `EROUTE (-12)` after the service route
disappeared.  `Buffer.close()` raised that cleanup result as a new primary
failure.  Supervision then rebuilt the 1.3B-parameter trainer and restored its
checkpoint repeatedly.  By the time the restarted trainer reached
`GenerationMetadata.from_json`, the otherwise-correct generation-1 record had
expired, so the structural validator reported the misleading invalid-identity
error.  The producer and consumer did not disagree about a field; teardown had
incorrectly converted a completed result into a slow reconnect to an expiring
identity.

## Fix

Native buffer, operation, and client release now treat `EROUTE` like the
already-supported fence/shutdown terminal outcomes.  This applies only after
the process-local descriptor/reference has been released.  Admission,
submission, refresh, result lookup, and all other data-path route failures
still propagate.  Thus cleanup is bounded and cannot replace the primary
generation outcome or drive supervisor restart exhaustion.

Generation metadata parsing also rejects coercible or ambiguous integer
values (including booleans and non-finite floats), non-mapping runtime
attestation, and non-hex digests.  Deadline freshness, positive fence/attempt/
owner/size, schema, and expected run/fence/generation checks remain fail closed.
The regression fixture contains the exact node-0 generation-1 payload from job
5039258 and evaluates it at its retained publication time; separate mutations
cover stale, partial, corrupt, non-finite, and obsolete-fence identities.

## Architecture conformance

The change was checked against the conformance checklist in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md` and the normative Compute Pool design:

- **R07, R12, NDP15:** atomic checkpoint/latest publication and fenced recovery
  identity are unchanged; stale or obsolete recovery metadata remains rejected.
- **R11:** a valid same-fence next-generation reconnect remains admissible,
  while disappeared routes terminate only idempotent local cleanup.
- **R14, NDP13:** metadata and all service operations retain absolute bounded
  deadlines; route-local cleanup can no longer create restart exhaustion.
- **R16, NDP17:** no scale rung or live acceptance claim is advanced by this
  implementation, and no Slurm job was submitted.
- **NDP10:** checksum, digest, finite-value, replay, and once-only admission
  checks are unchanged; metadata digest syntax is stricter.
- **NDP16:** exact provider/build/config/source runtime attestation in the
  generation identity is preserved and covered by the production fixture.

## Validation

- Exact job-5039258 generation-1 metadata and fail-closed mutation tests: pass.
- Lost-route native release/client cleanup regression: pass against the native
  service fixture.
- Canonical Frontier runtime/launcher plus native failure suites: 98 passed.
- Canonical native build: pass; CTest: 10/10 passed.
- No Slurm job was submitted.
