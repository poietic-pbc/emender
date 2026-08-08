#!/bin/bash
#SBATCH -A bif148
#SBATCH -J e97-corpus-receipt
#SBATCH -p batch
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:00:00
#SBATCH --no-requeue
#SBATCH -o /lustre/orion/bif148/proj-shared/emender/validation/e97-paper-corpus-receipt/slurm-%j.out
#SBATCH -e /lustre/orion/bif148/proj-shared/emender/validation/e97-paper-corpus-receipt/slurm-%j.out

set -euo pipefail
umask 027

readonly CORPUS_PATH=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
readonly EXPECTED_CORPUS_BYTES=1000000725401
readonly EXPECTED_CORPUS_SHA256=44f4c33471e0d49686453d81850380532bdc4a09e15c71b78eb8ec2d71bbcaa9
readonly TOKENIZER_PATH=/lustre/orion/bif148/proj-shared/emender/tokenizers/tiktoken/p50k_base/ec7223a39ce59f226a68acc30dc1af2788490e15
readonly EXPECTED_TOKENIZER_BYTES=836186
readonly EXPECTED_TOKENIZER_SHA256=94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069
readonly SAMPLER_SCHEMA=emender-byte-window-counter-v1
readonly FIXED_SAMPLER_KEY=42
readonly REPO=${E97_CORPUS_RECEIPT_REPO:?set E97_CORPUS_RECEIPT_REPO to the submitted source tree}
readonly SUBMITTED_SOURCE_COMMIT=${E97_CORPUS_RECEIPT_SOURCE_COMMIT:?set E97_CORPUS_RECEIPT_SOURCE_COMMIT to the submitted commit}

fail() {
  printf 'RECEIPT_FAIL: %s\n' "$*" >&2
  exit 1
}

printf 'receipt_start=%s\n' "$(date -Is)"
printf 'host=%s\n' "$(hostname -f)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
printf 'slurm_partition_env=%s\n' "${SLURM_JOB_PARTITION:-unset}"
printf 'slurm_qos_env=%s\n' "${SLURM_JOB_QOS:-unset}"
printf 'slurm_requeue_request=disabled_by_sbatch_no_requeue\n'
printf 'source_repo=%s\n' "$REPO"
printf 'submitted_source_commit=%s\n' "$SUBMITTED_SOURCE_COMMIT"
actual_source_commit=$(git -C "$REPO" rev-parse HEAD)
printf 'actual_source_commit=%s\n' "$actual_source_commit"
[[ "$actual_source_commit" == "$SUBMITTED_SOURCE_COMMIT" ]] || \
  fail "source commit mismatch: submitted=$SUBMITTED_SOURCE_COMMIT actual=$actual_source_commit"
printf 'sampler_schema=%s\n' "$SAMPLER_SCHEMA"
printf 'fixed_sampler_key=%s\n' "$FIXED_SAMPLER_KEY"

printf '%s\n' '== live scheduler evidence =='
printf '$ scontrol show job -o %q\n' "$SLURM_JOB_ID"
live_scheduler_output=$(scontrol show job -o "$SLURM_JOB_ID")
printf '%s\n' "$live_scheduler_output"
[[ "$live_scheduler_output" == *"Partition=batch"* ]] || fail 'live Partition is not batch'
[[ "$live_scheduler_output" == *"QOS=debug"* ]] || fail 'live QOS is not debug'
[[ "$live_scheduler_output" == *"Requeue=0"* ]] || fail 'live Requeue is not 0'

printf '%s\n' '== canonical artifact identities =='
printf '$ stat -c %q -- %q\n' 'path=%n size_bytes=%s mtime=%y mode=%A owner=%U group=%G' "$CORPUS_PATH"
stat -c 'path=%n size_bytes=%s mtime=%y mode=%A owner=%U group=%G' -- "$CORPUS_PATH"
corpus_bytes=$(stat -c '%s' -- "$CORPUS_PATH")
[[ "$corpus_bytes" == "$EXPECTED_CORPUS_BYTES" ]] || \
  fail "corpus byte-size mismatch: expected=$EXPECTED_CORPUS_BYTES actual=$corpus_bytes"

corpus_hash_start=$(date -Is)
printf 'corpus_hash_start=%s\n' "$corpus_hash_start"
printf '$ sha256sum -- %q\n' "$CORPUS_PATH"
corpus_hash_output=$(sha256sum -- "$CORPUS_PATH")
printf '%s\n' "$corpus_hash_output"
corpus_sha256=${corpus_hash_output%% *}
printf 'corpus_hash_end=%s\n' "$(date -Is)"
[[ "$corpus_sha256" == "$EXPECTED_CORPUS_SHA256" ]] || \
  fail "corpus SHA-256 mismatch: expected=$EXPECTED_CORPUS_SHA256 actual=$corpus_sha256"

printf '$ stat -c %q -- %q\n' 'path=%n size_bytes=%s mtime=%y mode=%A owner=%U group=%G' "$TOKENIZER_PATH"
stat -c 'path=%n size_bytes=%s mtime=%y mode=%A owner=%U group=%G' -- "$TOKENIZER_PATH"
tokenizer_bytes=$(stat -c '%s' -- "$TOKENIZER_PATH")
[[ "$tokenizer_bytes" == "$EXPECTED_TOKENIZER_BYTES" ]] || \
  fail "tokenizer byte-size mismatch: expected=$EXPECTED_TOKENIZER_BYTES actual=$tokenizer_bytes"
printf '$ sha256sum -- %q\n' "$TOKENIZER_PATH"
tokenizer_hash_output=$(sha256sum -- "$TOKENIZER_PATH")
printf '%s\n' "$tokenizer_hash_output"
tokenizer_sha256=${tokenizer_hash_output%% *}
[[ "$tokenizer_sha256" == "$EXPECTED_TOKENIZER_SHA256" ]] || \
  fail "tokenizer SHA-256 mismatch: expected=$EXPECTED_TOKENIZER_SHA256 actual=$tokenizer_sha256"

printf 'corpus_path=%s\n' "$CORPUS_PATH"
printf 'corpus_size_bytes=%s\n' "$corpus_bytes"
printf 'corpus_sha256=%s\n' "$corpus_sha256"
printf 'tokenizer_path=%s\n' "$TOKENIZER_PATH"
printf 'tokenizer_size_bytes=%s\n' "$tokenizer_bytes"
printf 'tokenizer_sha256=%s\n' "$tokenizer_sha256"
printf 'receipt_verdict=PASS\n'
printf 'receipt_end=%s\n' "$(date -Is)"
