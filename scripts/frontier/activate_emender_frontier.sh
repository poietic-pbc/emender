#!/bin/bash
# Canonical interactive/worker environment for Emender development on Frontier.
# Source this file; it intentionally updates the caller's modules and Python.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/frontier/activate_emender_frontier.sh; do not execute it" >&2
  exit 64
fi

_emender_frontier_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
_emender_frontier_repo=$(cd "${_emender_frontier_script_dir}/../.." && pwd -P)

if ! type module >/dev/null 2>&1; then
  echo "Frontier's module command is unavailable; start a Frontier login shell first" >&2
  return 69
fi

# WG workers use linked worktrees. Resolve the shared checkout so every worktree
# uses the one approved, durable environment rather than trying to create its own.
_emender_frontier_common_git=$(
  git -C "$_emender_frontier_repo" rev-parse --path-format=absolute --git-common-dir 2>/dev/null
) || {
  echo "cannot resolve the Emender git common directory" >&2
  return 70
}
_emender_frontier_shared_repo=$(dirname "$_emender_frontier_common_git")

export REPO=${REPO:-$_emender_frontier_repo}
export EMENDER_CONDA_ENV=${EMENDER_CONDA_ENV:-${_emender_frontier_shared_repo}/.envs/olcf-rocm711-torch210-py312}

# shellcheck source=frontier_runtime_env.sh
source "${_emender_frontier_script_dir}/frontier_runtime_env.sh"
frontier_load_default_modules || return
frontier_activate_emender_conda_env || return
frontier_assert_emender_conda_env || return

export EMENDER_PYTHON="${EMENDER_CONDA_ENV}/bin/python"
if [[ ! -x "$EMENDER_PYTHON" ]]; then
  echo "approved Emender Python is unavailable: $EMENDER_PYTHON" >&2
  return 66
fi
if [[ $(readlink -f "$(command -v python)") != $(readlink -f "$EMENDER_PYTHON") ]]; then
  echo "environment activation did not select $EMENDER_PYTHON" >&2
  return 66
fi

"$EMENDER_PYTHON" - <<'PY' || return
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Emender requires Python 3.12+, found {sys.version}")
PY

export PYTHON_BIN="$EMENDER_PYTHON"
unset _emender_frontier_common_git _emender_frontier_repo
unset _emender_frontier_script_dir _emender_frontier_shared_repo
