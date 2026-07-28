#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cache_root=${EMENDER_RESILIENT_LEAN_CACHE:-"$package_root/.cache/resilient-lean"}
release=4.26.0
archive_name="lean-${release}-linux.tar.zst"
archive_sha256=873c252b1c6b1392e5720ad8d5a137aabbe72c9f96a930fdb5a1dd1ddc5da454
release_url="https://github.com/leanprover/lean4/releases/download/v${release}/${archive_name}"
archive="$cache_root/downloads/$archive_name"
toolchain="$cache_root/toolchains/lean-${release}-linux"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "the pinned bootstrap artifact supports Linux x86_64 only" >&2
  exit 65
fi

mkdir -p "$cache_root/downloads" "$cache_root/toolchains"

if [[ ! -x "$toolchain/bin/lean" || ! -x "$toolchain/bin/lake" ]]; then
  if [[ ! -f "$archive" ]]; then
    if [[ "${EMENDER_RESILIENT_LEAN_OFFLINE:-0}" == 1 ]]; then
      echo "offline mode: pinned Lean archive is not cached at $archive" >&2
      exit 69
    fi
    curl --fail --location --retry 3 --continue-at - \
      --output "$archive" "$release_url"
  fi

  printf '%s  %s\n' "$archive_sha256" "$archive" |
    sha256sum --check --status -
  extract_root=$(mktemp -d "$cache_root/toolchains/.extract.XXXXXX")
  cleanup_extract() {
    rm -rf -- "$extract_root"
  }
  trap cleanup_extract EXIT
  tar --use-compress-program=unzstd -xf "$archive" -C "$extract_root"
  if [[ ! -x "$extract_root/lean-${release}-linux/bin/lean" ]]; then
    echo "pinned Lean archive did not contain the expected toolchain" >&2
    exit 70
  fi
  if [[ -e "$toolchain" ]]; then
    echo "refusing to overwrite incomplete toolchain cache: $toolchain" >&2
    exit 71
  fi
  mv "$extract_root/lean-${release}-linux" "$toolchain"
fi

actual_lean=$("$toolchain/bin/lean" --version)
actual_lake=$("$toolchain/bin/lake" --version)
case "$actual_lean" in
  "Lean (version 4.26.0, "*"commit d8204c9fd894f91bbb2cdfec5912ec8196fd8562, Release)") ;;
  *)
    echo "cached Lean toolchain identity mismatch: $actual_lean" >&2
    exit 72
    ;;
esac
case "$actual_lake" in
  "Lake version 5.0.0-src+d8204c9 (Lean version 4.26.0)") ;;
  *)
    echo "cached Lake identity mismatch: $actual_lake" >&2
    exit 72
    ;;
esac

printf '%s\n' "$toolchain"
