#!/usr/bin/env python3
"""Run canonical traces through Lean and the compiled production service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ndm.native_lean_conformance import (  # noqa: E402
    ConformanceDivergence,
    TraceFormatError,
    canonical_json,
    run_differential_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Differentially replay one canonical resilient-coordination trace "
            "through Lean 4 and the production native mutation path."
        )
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=ROOT / "build/native-resilient-dataplane/native-artifacts.json",
    )
    parser.add_argument(
        "--lean-runner",
        type=Path,
        default=(
            ROOT
            / "formal/resilient/.lake/build/bin/resilient-conformance"
        ),
    )
    parser.add_argument(
        "--divergence-directory",
        type=Path,
        default=ROOT / "reports/conformance/divergences",
    )
    parser.add_argument(
        "--fault-event-index",
        type=int,
        help=(
            "Deliberately alter exact_tokens at this native contribution "
            "translation; used only to prove first-divergence detection."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the canonical agreement report as well as printing it.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        report = run_differential_trace(
            trace_path=arguments.trace,
            build_manifest=arguments.build_manifest,
            lean_runner=arguments.lean_runner,
            repository=ROOT,
            divergence_directory=arguments.divergence_directory,
            fault_event_index=arguments.fault_event_index,
        )
    except ConformanceDivergence as error:
        print(canonical_json(error.report), file=sys.stderr)
        return 1
    except (TraceFormatError, OSError, RuntimeError, ValueError) as error:
        print(
            canonical_json(
                {
                    "schema": "emender-native-lean-conformance-run-v1",
                    "verdict": "fail_closed",
                    "error": str(error),
                }
            ),
            file=sys.stderr,
        )
        return 2
    encoded = canonical_json(report)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
