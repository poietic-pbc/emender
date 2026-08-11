# E97 50B instruction corpus construction receipt

Status: **complete and validated.** Both final corpus files and adjacent
machine-readable receipts are published.

Normative build contract: `docs/E97_INSTRUCTION_CORPUS_50B.md`.

## Staging

Pinned public snapshots were downloaded on the Frontier login/data-transfer
path because compute nodes cannot reach Hugging Face. The initial compute-node
probe job `5224097` was cancelled after the fail-closed receipt reported
`Address family not supported by protocol`; it wrote no accepted source.

The public snapshot pass staged 101 GB under:

```text
/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/sources/
```

Complete public snapshots:

- Nemotron Instruction/Chat v3 (prompt restoration still pending);
- Nemotron SWE v3;
- Open-SWE-Traces;
- Nemotron Terminal Corpus;
- AgentTrove;
- SWE-Next trajectories;
- OpenThoughts3-1.2M;
- WildChat-1M restoration source.

The operator subsequently accepted `SALT-NLP/SWE-chat`, `GAIR/OpenSWE`, and
`lmsys/lmsys-chat-1m`. The pinned snapshots were downloaded with Xet disabled
and the final `download-receipt.json` has zero failures. No source was omitted
or reweighted.

## Public-source serialization inventory

The first inventory attempt, job `5224589`, failed before reading data because
the compute node could not download p50k. The launcher was corrected to bind
the already qualified project-shared p50k cache. Jobs `5224700`, `5225478`,
`5225934`, and `5226185` then created resumable shard inventories. The final
public pass `5226185` completed `0:0`.

| Source | Accepted records | Serialized p50k tokens | Serialized bytes | >=32K records | >=128K records | Rejected rows |
|---|---:|---:|---:|---:|---:|---:|
| AgentTrove | 1,568,412 | 19.451B | 60.7 GB | 61,379 | 154 | 128,435 |
| Nemotron SWE v3 | 237,970 | 12.106B | 38.4 GB | 178,432 | 4,729 | 0 |
| Nemotron Terminal | 366,154 | 6.863B | 21.9 GB | 41,658 | 1 | 0 |
| Open-SWE-Traces | 207,489 | 17.042B | 53.4 GB | 203,691 | 13,353 | 0 |
| OpenThoughts3 | 1,200,000 | 19.326B | 59.7 GB | 38 | 0 | 0 |
| SWE-Next | 3,693 | 0.065B | 0.2 GB | 172 | 0 | 0 |

The inventories occupy 214 GB and include random-access record bytes, fixed
binary offset/token indexes, per-source JSON receipts, record-stream SHA-256,
length buckets, rejection counts, and embedded-RS replacement counts.
The three gated/restored inventories subsequently added:

| Source | Accepted records | Serialized p50k tokens | Serialized bytes | >=32K records | >=128K records |
|---|---:|---:|---:|---:|---:|
| Nemotron Instruction/Chat v3 | 812,124 | 5.813B | 17.0 GB | 15,438 | 224 |
| SWE-chat | 5,804 | 0.386B | 1.15 GB | 2,996 | 772 |
| GAIR/OpenSWE | 12,431 | 1.115B | 3.45 GB | 12,297 | 612 |

SWE-Next's 2.5B target required 39 shuffled epochs; SWE-chat required 13,
GAIR/OpenSWE 3, and Nemotron Instruction/Chat 3. The final manifest reports
exact selected counts and overshoot for every bucket.

Adjacent machine-readable receipt:

```text
/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/
  public-inventory-summary.json
```

## Protected prompt restoration

The pinned LMSYS and WildChat snapshots restored 280,495 Nemotron chat rows.
The upstream protected datasets still redact 5,110 unique LMSYS and 14,286
unique WildChat prompt hashes; 75,287 affected rows were excluded rather than
emitted as incomplete trajectories. The exact hashes/counts and source/output
digests are recorded in `nemotron-prompt-restoration.json`.

## Final construction

The final inventories contain all nine named sources. Job `5227925` completed
all quota selection and the full 256-bucket external shuffle spool, then timed
out while emitting the main file. Job `5229672` deterministically replayed the
selection clock, verified the complete spool, and emitted both files in 16:10.

Authoritative all-data file:

```text
/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/
  e97_instruction_50b_v1.txt
```

- 153,710,521,868 bytes;
- 3,049,886 complete records / 3,049,885 RS delimiters;
- 50,000,556,252 accounted p50k construction tokens;
- complete-record overshoot: 556,252 tokens;
- SHA-256: `224acfad07fb5778b89b3630ac0851c7ee5743c45250173836b77fc298123da8`.

Derived long-only file:

```text
e97_instruction_50b_v1_long32k.txt
```

- 4,665,941,053 bytes;
- 22,535 complete records / 22,534 RS delimiters;
- 1,500,095,041 accounted p50k tokens;
- SHA-256: `9b68ad96c8df5114caf29b9ef049835fda0c9748d2b0e5e89593db48e9e9481f`.

## Final validation

Job `5229854` completed `0:0` in 08:43. It streamed both files in full and
verified strict UTF-8, byte counts, RS counts, record counts, and SHA-256. It
retokenized every long-only record with the frozen p50k authority: minimum
32,768 tokens, maximum 1,556,124. The unchanged online loader also returned
correct samples at 2K, 32K, and 128K contexts. Machine receipt:
`e97_instruction_50b_v1.validation.json`.

The main file is the sole training input. The long-only file is a derived audit
and optional future sampling resource; its records already occur in the main
mixture.
