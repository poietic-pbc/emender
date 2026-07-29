#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
manifest="$package_root/proof-manifest-v1.json"

if [[ ! -f "$manifest" ]]; then
  echo "missing proof manifest: $manifest" >&2
  exit 1
fi

jq -e '
  .schema == "emender-resilient-lean-proof-manifest-v1" and
  .toolchain.lean == "leanprover/lean4:v4.26.0" and
  .toolchain.lake == "5.0.0-src+d8204c9" and
  .model.authoritative_transition ==
    "ResilientProtocol.transition" and
  .model.executable_trace_fold ==
    "ResilientProtocol.executeEvents" and
  .proof_policy.runtime_boolean_is_not_a_theorem == true and
  .proof_policy.theorems_are_propositions_over_transition == true and
  .proof_policy.boolean_substitutes == [] and
  .proof_policy.safety_fairness_hypotheses == false and
  (.proof_policy.forbidden_lean_tokens | sort) ==
    (["admit", "axiom", "native_decide", "opaque", "sorry", "unsafe"] | sort) and
  .progress.unconditional_claim == false and
  .progress.required_assumptions == [
    "finite_close_and_stage_deadlines",
    "surviving_eligible_stable_worker_quorum",
    "surviving_exact_token_floor",
    "bounded_permitted_failures",
    "bounded_owner_reassignments",
    "eventual_delivery_and_processing",
    "fair_scheduling_of_enabled_transitions"
  ] and
  (.artifacts | type == "array" and length > 0) and
  ([.artifacts[].path] | length == (unique | length)) and
  all(.artifacts[];
    (.path | type == "string") and
    (.sha256 | type == "string") and
    (.sha256 | test("^[0-9a-f]{64}$")))
' "$manifest" >/dev/null

mapfile -t entries < <(
  jq -r '.artifacts[] | [.path, .sha256] | @tsv' "$manifest"
)

for entry in "${entries[@]}"; do
  IFS=$'\t' read -r relative expected <<<"$entry"
  if [[ -z "$relative" || -z "$expected" ||
        ! "$expected" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid proof manifest artifact entry: $entry" >&2
    exit 1
  fi
  case "$relative" in
    /*|../*|*/../*|*/..)
      echo "escaping proof manifest path: $relative" >&2
      exit 1
      ;;
  esac
  artifact="$package_root/$relative"
  if [[ ! -f "$artifact" ]]; then
    echo "missing proof artifact: $relative" >&2
    exit 1
  fi
  actual=$(sha256sum "$artifact" | awk '{print $1}')
  if [[ "$actual" != "$expected" ]]; then
    echo "proof digest mismatch: $relative" >&2
    echo "expected $expected" >&2
    echo "actual   $actual" >&2
    exit 1
  fi
done

toolchain_file=$(<"$package_root/lean-toolchain")
toolchain_manifest=$(jq -r '.toolchain.lean' "$manifest")
if [[ "$toolchain_file" != "$toolchain_manifest" ]]; then
  echo "lean-toolchain does not match proof manifest" >&2
  exit 1
fi

if ! rg -Fq 'def transition (state : RunState) (event : Event)' \
    "$package_root/ResilientProtocol/Kernel.lean"; then
  echo "authoritative transition symbol is missing" >&2
  exit 1
fi

echo "PASS proof manifest: ${#entries[@]} bound artifacts"
