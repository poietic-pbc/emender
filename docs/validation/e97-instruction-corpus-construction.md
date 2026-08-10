# E97 50B instruction corpus construction receipt

Status: **in progress; blocked only on operator acceptance of three Hugging
Face gated datasets.** No final corpus has been published yet.

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

Current gated failures are explicit in `download-receipt.partial.json`:

- `SALT-NLP/SWE-chat`;
- `GAIR/OpenSWE`;
- `lmsys/lmsys-chat-1m`, required to restore withheld Nemotron chat prompts.

No unavailable source has been omitted or reweighted.

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
SWE-Next's 2.5B target will require approximately 38.3 complete passes; the
final quota receipt will report the exact selected count and overshoot.

Adjacent machine-readable receipt:

```text
/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/
  public-inventory-summary.json
```

## Remaining sequence

1. Operator accepts the three gated datasets.
2. Rerun the pinned downloader; require `download-receipt.json` with zero
   failures.
3. Restore Nemotron protected prompts from the pinned local LMSYS/WildChat
   snapshots and publish its receipt.
4. Inventory Nemotron Instruction/Chat, SWE-chat, and GAIR/OpenSWE.
5. Run the deterministic 50B quota selector and external record shuffle.
6. Validate both UTF-8/RS files, publish SHA-256 and final manifest, and replace
   adjacent `.partial` documentation with final `README.md` and source spec.
