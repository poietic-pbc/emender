# Resumed terminal audit: E97 exact 256-node debug

Read-only audit time: `2026-07-15T07:41:52Z`.

No job was submitted, cancelled, requeued, updated, or otherwise mutated by
this resumed audit.

Terminal `sacct` accounting was:

```text
4980157|CANCELLED by 19032|0:0|2026-07-13T08:06:54|None|2026-07-15T03:28:54|00:00:00|12:00:00|0|0|None
5000436|CANCELLED by 19032|0:0|2026-07-15T02:31:50|2026-07-15T02:45:14|2026-07-15T03:27:57|00:42:43|02:00:00|256|28672|None
```

The matching `squeue` query returned no rows. Debug job `5000436` therefore
remains terminal after 42 minutes 43 seconds rather than scheduler-controlled
finalization near two hours. Production job `4980157` never started or
received an allocation; its external cancellation is covered by the existing
terminal audit.

The persistent run root remains:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/async_quorum_b4k40_ladder_20260709/20260715/E97_1.3B_step1065000_async_quorum_b4k40_ladder_256n/5000436-20260715T064518Z
```

`continuation/last-valid.json` still resolves to immutable manifest
`last-valid-20260715T072807Z.json`. The manifest is readable with SHA256
`ef6eaeb0822e4028e20357e4ed3461c3587b9ea9371e2fa432824ff6f4940a2c` and
selects generation 9 checkpoint
`async_run/checkpoints/emender_E97_100m_20260715/checkpoint_step_1525400_loss_2.4184.pt`.
That checkpoint still exists on persistent Lustre at `15439252298` bytes. The
manifest records the full-file checksum computed during terminal recovery as
`ee9d69d9c3efd5696042b30ad1ad57236d5035876bae5ce2e9cc2010e5017fd3`.
Its `step` field is JSON null, so consumers must preserve that caveat and
derive step 1,525,400 from the filename plus finalized generation evidence.

The task remains incomplete: the scale gates passed (256 nodes, 2,048 ranks,
generations 0 through 9, ten all-rank merges, and 400 steps), but the two-hour
scheduler boundary and independent model/optimizer/step/async-chain reload
were not completed. No retry or replacement is authorized from this task.
