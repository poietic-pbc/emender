# E97 paper corpus/tokenizer/sampler receipt

Status: **PASS**

WG task: `e97-paper-corpus-receipt`

Frontier job: `5207741`

Verification interval: `2026-08-08T14:01:17-04:00` to
`2026-08-08T14:59:03-04:00`

This is the immutable, Git-retained identity receipt required before the E97
paper training arms may consume a new scientific stream. It records the real
scheduler-produced artifact; it is not a login-node hash or a prospective pass.

## Frozen identities

| Identity | Value |
|---|---|
| Corpus path | `/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt` |
| Corpus byte size | `1,000,000,725,401` |
| Corpus SHA-256 | `44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9` |
| Tokenizer | `p50k_base` |
| Tokenizer cache path | `/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15` |
| Tokenizer cache byte size | `836,186` |
| Tokenizer cache SHA-256 | `94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069` |
| Sampler schema | `emender-byte-window-counter-v1` |
| Fixed sampler key | `42` |

The sampler schema names the versioned counter-based byte-window protocol in
`docs/E97_GDN2_NONLINEARITY_PAPER_PLAN.md`: deterministic byte position, fixed
read size, UTF-8 replacement, drop-first-token behavior, bounded retries tied
to sample identity, and the pinned tokenizer above. This freezes the protocol
identity; it does not relabel any legacy stream or by itself claim sampler-code
qualification. Implementations and checkpoints must use this exact schema/key
or fail closed.

The historical construction manifest retained at
`docs/s3-audit/delete-transferred-spinozans/commapile_mainmix_v0.1_1tb.txt.manifest.json`
records the same corpus digest and construction seed. Job `5207741` independently
recomputed the decompressed canonical artifact's digest.

## Source and submission identity

The job reported this source identity:

```text
source_repo=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-4
submitted_source_commit=42990607036e7a59c9aed13795f49aa875ffd443
actual_source_commit=42990607036e7a59c9aed13795f49aa875ffd443
job_source=scripts/frontier/e97_paper_corpus_receipt.sbatch
job_source_at_reported_commit_sha256=a3fe65508412ac51f2cfb6ef5a5ff4d312d3034d0aa8c39f05c28bac3df550a0
```

The job compared its submitted commit environment variable with `git rev-parse
HEAD` and failed closed on disagreement. Frontier accounting stored the exact
spooled batch script. After completion, the accounting copy was extracted and
compared byte-for-byte with the named file at the reported commit. The landed
`./scripts/frontier/e97_paper_corpus_receipt.sbatch` and Git-retained spooled
copy are also byte-identical, so the submitted script, reported source commit,
and delivered script are bound by bytes rather than commit labels alone.

The following comparison transcript and its inputs are included under
`docs/validation/e97-paper-corpus-sampler-receipt-artifacts/`:

```console
$ sacct -j 5207741 --batch-script
Batch Script for 5207741
--------------------------------------------------------------------------------
#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-corpus-receipt
#SBATCH -p batch
#SBATCH --qos=debug
[the complete 92-line accounting output is retained in terminal-batch-script-sacct.txt]
$ tail -n +4 terminal-batch-script-sacct.txt > slurm-5207741-spooled-batch-script.sbatch
$ sha256sum -- slurm-5207741-spooled-batch-script.sbatch
a3fe65508412ac51f2cfb6ef5a5ff4d312d3034d0aa8c39f05c28bac3df550a0  slurm-5207741-spooled-batch-script.sbatch
$ git show 42990607036e7a59c9aed13795f49aa875ffd443:scripts/frontier/e97_paper_corpus_receipt.sbatch | sha256sum
a3fe65508412ac51f2cfb6ef5a5ff4d312d3034d0aa8c39f05c28bac3df550a0  -
$ cmp slurm-5207741-spooled-batch-script.sbatch <(git show 42990607036e7a59c9aed13795f49aa875ffd443:scripts/frontier/e97_paper_corpus_receipt.sbatch)
spooled_script_matches_source_commit=true
```

Thus the executed scheduler script is bound to source commit
`42990607036e7a59c9aed13795f49aa875ffd443`, independent of the relative path
used at submission. This does not assert that unrelated worktree files were
clean; the hash job did not execute them.

Exact submission command and output:

```console
$ E97_CORPUS_RECEIPT_REPO=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-4 E97_CORPUS_RECEIPT_SOURCE_COMMIT=42990607036e7a59c9aed13795f49aa875ffd443 sbatch --parsable scripts/frontier/e97_paper_corpus_receipt.sbatch
5207741
```

The source requests account `bif148`, `Partition=batch`, `QOS=debug`, one node,
one task, 16 CPUs/task, an explicit `01:00:00` limit, and `--no-requeue`. Slurm
allocated one node and the job completed in `00:57:46`. A monitored attempt to
extend the limit was denied by Slurm; the recorded terminal limit remained
`01:00:00`, so the successful artifact is from the original bounded request.

## Exact scheduler hash command and output

The following is copied verbatim from the scheduler stdout artifact. The job
fails closed if either byte size or digest differs from its expected value.

```console
$ stat -c path=%n\ size_bytes=%s\ mtime=%y\ mode=%A\ owner=%U\ group=%G -- /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
path=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt size_bytes=1000000725401 mtime=2026-06-21 02:21:57.000000000 -0400 mode=-rw-r--r-- owner=erikgarrison group=bif148
corpus_hash_start=2026-08-08T14:01:17-04:00
$ sha256sum -- /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9  /lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
corpus_hash_end=2026-08-08T14:59:03-04:00
$ stat -c path=%n\ size_bytes=%s\ mtime=%y\ mode=%A\ owner=%U\ group=%G -- /lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
path=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15 size_bytes=836186 mtime=2026-07-17 19:38:45.000000000 -0400 mode=-r--r--r-- owner=erikgarrison group=bif148
$ sha256sum -- /lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069  /lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
corpus_path=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
corpus_size_bytes=1000000725401
corpus_sha256=44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
tokenizer_path=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
tokenizer_size_bytes=836186
tokenizer_sha256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
receipt_verdict=PASS
receipt_end=2026-08-08T14:59:03-04:00
```

## Live scheduler evidence

Separately named live queue fields:

```text
live_partition=batch
live_qos=debug
live_requeue=0
```

Exact live queue output captured while job `5207741` was running:

```console
$ squeue -h -j 5207741 -o %i\|%P\|%q\|%T\|%R
5207741|batch|debug|RUNNING|frontier01802
```

The independent live `scontrol show job -o 5207741` record contains:

```text
JobId=5207741 JobName=e97-corpus-receipt Account=bif148 QOS=debug JobState=RUNNING Reason=None Requeue=0 Restarts=0 RunTime=00:00:12 TimeLimit=01:00:00 Partition=batch NodeList=frontier01802 Command=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-4/scripts/frontier/e97_paper_corpus_receipt.sbatch
```

The complete exact live `scontrol` line is retained in the durable artifact
listed below; the line above preserves the relevant fields without presenting
unrelated scheduler metadata as a second authority.

## Terminal scheduler evidence

Separately named terminal accounting fields:

```text
terminal_partition=batch
terminal_qos=debug
terminal_requeue=0
```

Exact terminal accounting command and output:

```console
$ sacct -X -j 5207741 --format=JobIDRaw,JobName%24,Partition,QOS,State,ExitCode,Elapsed,Timelimit,Submit,Start,End,AllocNodes,AllocCPUS,ReqTRES%80 -P
JobIDRaw|JobName|Partition|QOS|State|ExitCode|Elapsed|Timelimit|Submit|Start|End|AllocNodes|AllocCPUS|ReqTRES
5207741|e97-corpus-receipt|batch|debug|COMPLETED|0:0|00:57:46|01:00:00|2026-08-08T14:01:15|2026-08-08T14:01:17|2026-08-08T14:59:03|1|112|billing=16,cpu=16,mem=500G,node=1
```

Because `sacct` has no `Requeue` output field on this Frontier Slurm version,
the separately captured terminal `scontrol show job -o 5207741` record supplies
that field and contains:

```text
JobId=5207741 JobName=e97-corpus-receipt Account=bif148 QOS=debug JobState=COMPLETED Reason=None Requeue=0 Restarts=0 ExitCode=0:0 RunTime=00:57:46 TimeLimit=01:00:00 StartTime=2026-08-08T14:01:17 EndTime=2026-08-08T14:59:03 Partition=batch NodeList=frontier01802
```

## Durable raw artifacts

The scheduler stdout, exact Slurm-spooled script, and live/terminal scheduler
evidence are included in this Git output at:

```text
docs/validation/e97-paper-corpus-sampler-receipt-artifacts/
```

A read-only project-filesystem mirror remains at
`/lustre/orion/bif148/proj-shared/emender/validation/e97-paper-corpus-receipt/`.
The Git copies make every presented validation byte independently reviewable
without access to that external filesystem.

Key artifact digests from the included `SHA256SUMS`:

```text
3f1d3c5c75af0c50f43e0412f76ea35bac3b6adf7cafeda99b2fc37d8fc62b81  slurm-5207741.out
1dd9ab4e49514446bfca96ba88b4394787287eb019ca36a541abde74ef747cee  submission-command.txt
465ddb11815d6a18fa3e101242ab227c026b05f3f3d3b30c3c47ce99c0fba890  live-scontrol.txt
e4e765e3fdf330351cd548d3b97c045d2873acb021029f2281ddeb4589dc3ce9  live-squeue.txt
d3ca1a0ec18aabe668e0f79960978f2d2e3d8001e70794258e01d2902780de5e  terminal-sacct.txt
3434e57705c9dcd8f199f72657e220e569b14bca34c58ea18958d58252dea8f1  terminal-scontrol.txt
a3fe65508412ac51f2cfb6ef5a5ff4d312d3034d0aa8c39f05c28bac3df550a0  slurm-5207741-spooled-batch-script.sbatch
cc5640ec7fa34bb50546b1da84dc303f1f94ad5186614eb7d39eb68e4aae141a  source-provenance.txt
58aa14c94a75f9c9141bcd71d8da84e8d914c5dd59bf1ec281fd1d375ee6261e  terminal-batch-script-sacct.txt
```

`sha256sum -c docs/validation/e97-paper-corpus-sampler-receipt-artifacts/SHA256SUMS`
verifies all included raw files, and `cmp` verifies the included spooled script
against `scripts/frontier/e97_paper_corpus_receipt.sbatch`.

This Git receipt is the content-addressed publication of those facts. The job
read only the two named canonical artifacts, queried only its own scheduler ID,
and wrote only its dedicated validation directory. It did not query, cancel,
modify, continue, or otherwise touch job `5201882`, and it makes no assertion
about or mutation to any historical MoE authority.
