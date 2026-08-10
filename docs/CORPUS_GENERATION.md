# Training corpus generation index

The canonical training inputs are plain UTF-8 byte streams with literal ASCII
Record Separator (`0x1e`) bytes between complete records. They are tokenized
online by the training loader. The text files, not derived token IDs, are the
data authorities.

## CommaPile base-training corpus

Authority:

```text
/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
```

Builder: `scripts/build_commapile_mainmix.py`  
Staging job: `scripts/frontier/stage_commapile_mainmix.sbatch`  
Historical construction/staging receipt:
`docs/FRONTIER_COMMAPILE_MAINMIX_STAGE_20260621.md`

The builder reads the 31 Comma v0.1 sources, applies the published main-stage
effective-token weights as byte quotas for the existing random-byte sampler,
samples and interleaves complete documents deterministically, replaces embedded
RS bytes, and writes one 1,000,000,725,401-byte mmap-friendly text stream. Its
adjacent source manifest and staging receipt are the provenance authorities.

## E97 50B instruction continued-training corpus

Normative specification: `docs/E97_INSTRUCTION_CORPUS_50B.md`  
Pinned sources: `configs/frontier/e97_instruction_50b_sources.json`  
Downloader: `scripts/download_e97_instruction_sources.py`  
Prompt restoration: `scripts/restore_nemotron_chat_prompts.py`  
Inventory/serialization: `scripts/inventory_e97_instruction_source.py`  
Final quota selection/shuffle: `scripts/build_e97_instruction_corpus.py`

Final directory:

```text
/lustre/orion/bif148/proj-shared/commapile/e97_instruction_50b_v1/
```

It contains one authoritative all-source 50B mixture and one auxiliary file
containing the selected >=32K-token long-context tranche. Both have the same
plain UTF-8/RS interface as CommaPile. The authoritative mixture includes the
long tranche; training points only at the authoritative mixture. Adjacent
README, source snapshot, inventory, quota, digest, and build receipts document
the exact construction.
