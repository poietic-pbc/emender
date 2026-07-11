#!/bin/bash
# Sole production authority. Never call sbatch directly from a legacy wrapper.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
exec python3 scripts/frontier/check_e97_async_promotion.py \
  --smoke build/e97-256/smoke --production build/e97-256/production \
  --policy configs/frontier/e97_async_256_parity_policy.json \
  --require-promotion --submit --approval "${1:?exact-fingerprint approval JSON required}"
