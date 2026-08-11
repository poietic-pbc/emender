# E97 instruction continued-training corpus (50B)

Status: **complete and validated**. This document is the normative build
contract and final resource description.

Final authoritative receipt:

- main file: 153,710,521,868 bytes; 3,049,886 records;
- accounted construction tokens: 50,000,556,252 (556,252 complete-record overshoot);
- main SHA-256: `224acfad07fb5778b89b3630ac0851c7ee5743c45250173836b77fc298123da8`;
- long-only file: 4,665,941,053 bytes; 22,535 records;
- long-only SHA-256: `9b68ad96c8df5114caf29b9ef049835fda0c9748d2b0e5e89593db48e9e9481f`;
- minimum long-only record length: exactly 32,768 p50k tokens;
- validation job `5229854`: `COMPLETED 0:0`.

The training run must still stop at exactly 50,000,000,000 accepted tokens; the
small corpus overshoot exists solely because complete records are never split.

## Deliverables

The build produces two plain UTF-8 text files under
`/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/`:

1. `e97_instruction_50b_v1.txt` — the authoritative globally shuffled 50B
   mixture used for training.
2. `e97_instruction_50b_v1_long32k.txt` — the selected 3% long-context tranche
   only. Every record in this file has at least 32,768 `p50k_base` tokens and
   also occurs in the authoritative mixture.

Both files use exactly the existing CommaPile data interface: complete records
encoded as plain UTF-8 with one literal ASCII Record Separator byte (`0x1e`)
between records. There is no canonical pretokenized artifact. Training retains
online tokenization and random contiguous byte-window sampling; only the data
path, corpus digest, and explicit sampler phase identity change.

## Frozen mixture

Token quotas are measured after canonical serialization with `p50k_base`.
They define construction of the tokenizer-independent text authority.

| Bucket | Share | Target tokens |
|---|---:|---:|
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v3` | 25% | 12,500,000,000 |
| `nvidia/Nemotron-SFT-SWE-v3` | 15% | 7,500,000,000 |
| `nvidia/Open-SWE-Traces` | 12% | 6,000,000,000 |
| `nvidia/Nemotron-Terminal-Corpus` | 12% | 6,000,000,000 |
| `SALT-NLP/SWE-chat` | 10% | 5,000,000,000 |
| `open-thoughts/AgentTrove` | 10% | 5,000,000,000 |
| `GAIR/OpenSWE` | 5% | 2,500,000,000 |
| `TIGER-Lab/SWE-Next-SFT-Trajectories` | 5% | 2,500,000,000 |
| `open-thoughts/OpenThoughts3-1.2M` | 3% | 1,500,000,000 |
| long-context records drawn from the sources above | 3% | 1,500,000,000 |

The long-context tranche is deliberate oversampling. Its selected occurrences
are written both to the long-only file and into the main mixture. A record may
therefore occur once through its named-source quota and again through the long
tranche. Sampling with replacement is allowed when a source has fewer unique
serialized tokens than its quota; all repeat factors are reported.

Complete records are never truncated to hit a quota. Each bucket stops after
reaching or crossing its target, so actual totals may exceed targets by at most
one selected record per bucket. The manifest reports every target, actual
count, and overshoot. The continued-training run, independently, accepts
exactly 50B training tokens.

## Serialization

Each Hugging Face row, or each complete trajectory reconstructed from related
rows, becomes one record. Source-specific adapters preserve ordered system,
user, assistant, reasoning, tool-call, tool-result, terminal, code, and patch
content while excluding administrative metadata. Labels are ordinary text
(e.g. `System:`, `User:`, `Assistant:`, `Tool:`); no model-specific special
chat tokens are introduced. Structured values use deterministic compact JSON.
Line endings are normalized to LF. Invalid UTF-8 is replaced. Any embedded raw
`0x1e` inside a record is replaced with one ASCII space and counted.

Message objects are serialized in source order. This intentionally differs
from the Nemotron chat card's recommendation to train only its final assistant
turn: this corpus preserves the complete restored trajectory and applies the
same ordinary causal loss to every token, as required by the study design.
Empty message fields are not invented. The Nemotron chat source deliberately
withholds some externally
sourced seed prompts; those rows are admitted only after NVIDIA's pinned
prompt reconstruction succeeds against the authorized pinned upstream
datasets. Some LMSYS prompts remain redacted even after gated access; rows whose
prompt hash cannot be recovered are excluded rather than emitted as incomplete
trajectories. Every unavailable hash and excluded row is counted in the prompt
restoration receipt. Records that remain incomplete or contain no usable
conversational text are rejected and counted.

## Determinism and provenance

The build pins every Hugging Face repository revision, config, and split. A
fixed seed controls record selection and global record-level shuffle. The
adjacent receipts record:

- repository revisions and source files;
- serializer version and per-source schema;
- unique and selected records/tokens/bytes;
- repeat factors and quota overshoot;
- 32K/64K/128K record-length counts;
- embedded-RS replacements and rejected records;
- shuffle seed;
- output sizes, RS counts, and SHA-256 digests;
- `p50k_base` tokenizer digest used to define this mixture.

The final files remain tokenizer-independent. A future tokenizer change keeps
the bytes unchanged, records a new tokenizer identity, remeasures the corpus,
and begins a new explicit sampler phase.

## Training contract

No model, loss, masking, recurrence, separator handling, or data-loader behavior
changes. Training uses ordinary autoregressive next-token prediction over every
token. RS remains visible p50k token 218 and does not reset recurrent state.
The authoritative main file is passed to the existing runner as `DATA`; the
long-only file is an auxiliary resource and is not a second live input to the
frozen 50B mixture.

## Access gate

Construction fails closed unless every pinned source is readable. At build
start, the current Hugging Face credential could inspect metadata but did not
have file access to `SALT-NLP/SWE-chat`; `GAIR/OpenSWE` also requires gated-file
verification. Restoring withheld Nemotron chat prompts may additionally require
accepted access to `lmsys/lmsys-chat-1m`. No source may be silently omitted,
renamed, or reweighted.
