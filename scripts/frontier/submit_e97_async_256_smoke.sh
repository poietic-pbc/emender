#!/bin/bash
# Exact job-4962400 smoke authority. This intentionally performs no overrides.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo"
python3 scripts/frontier/render_e97_async_256.py --profile smoke --out build/e97-256/smoke
python3 scripts/frontier/render_e97_async_256.py --profile production --out build/e97-256/production
python3 scripts/frontier/check_e97_async_promotion.py \
  --smoke build/e97-256/smoke --production build/e97-256/production \
  --policy configs/frontier/e97_async_256_parity_policy.json
exec python3 -c 'import json,os,sys; a=json.load(open(sys.argv[1]))["sbatch_argv"]; os.execvp(a[0],a)' \
  build/e97-256/smoke/launch-inputs.json
