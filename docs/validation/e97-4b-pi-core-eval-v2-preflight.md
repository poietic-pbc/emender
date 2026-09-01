# E97 4B Pi core evaluation v2 preflight

**Status:** frozen evaluation-only authority; no candidate has trained on these
records or task families.

The original 120-task panel is a core-tools smoke test whose validation records
share templates with the Pi-native training authority. V2 adds compositional,
template-family-held-out tasks before any subsequent instruction-tuning stage.

## Frozen authority

- Root: `/mnt/nvme1n1/erikg/sft/pi-core-eval-v2-template-heldout`
- Records: 240, all evaluation-only
- Seed: 2,601,041
- Manifest SHA-256:
  `b7d308bbcaaa6526234fadd8f59a77be5f2f4cfac3cd67d38f584863c9444c29`
- Metadata SHA-256:
  `b61864f653e8f2b069aaf1102fd6b6852fa6b216072107ed577bf5070a9b0c55`

There are 40 tasks in each family:

1. `inspect-chain`: read a pointer, follow it, and ground the final in the
   referenced file;
2. `search-edit`: locate an unknown target using bash, avoid a distractor, edit,
   and verify;
3. `multi-edit`: inspect and update two files while preserving unrelated state;
4. `recover-edit`: recover from a failed exact edit using a read and corrected
   edit;
5. `diagnose-test`: run a failing cross-file assertion, inspect two authorities,
   repair, and rerun;
6. `write-from-spec`: read a text specification, create typed JSON, and validate
   it.

The panel contains 920 expected tool calls. A clean-room mechanical preflight
materialized every sandbox, executed all declared tool calls (including the 80
intentional failure transitions), and verified every terminal postcondition:
240/240 tasks and 920/920 tool contracts validated.

## Scoring and safety

V2 uses the real Pi CLI, immutable Apptainer sandbox, and the same four registered
tools (`read`, `bash`, `edit`, and `write`). Full pass requires exact tool order
and arguments, schema validity, sandbox postconditions, one grounded final,
zero Pi errors, and no immediately repeated identical call. The evaluator remains
backward compatible with the original panel but accepts authority-declared call,
postcondition, and final-evidence contracts for new task families.

The promoted live-aligned u8 checkpoint and its update-64 behavioral parent must
both run against this exact frozen authority and corrected serving runtime before
new v3 training data is designed. Results diagnose the next curriculum; they do
not retroactively alter the original smoke-test receipt.
