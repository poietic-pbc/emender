#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
toolchain=$("$package_root/scripts/bootstrap.sh")

cd "$package_root"
exec env \
  PATH="$toolchain/bin:$PATH" \
  LEAN_SYSROOT="$toolchain" \
  "$toolchain/bin/lake" "$@"
