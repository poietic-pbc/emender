from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ndm.native_lean_conformance import (
    ConformanceDivergence,
    TRACE_SCHEMA_DIGEST,
    TraceFormatError,
    audit_production_call_path,
    canonical_json,
    load_canonical_trace,
    run_differential_trace,
    run_lean_oracle,
    state_digest,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "formal/resilient/corpus/native-v1"
CORPUS_MANIFEST = CORPUS / "manifest.json"
BUILD_MANIFEST = Path(
    os.environ.get(
        "EMENDER_NDP_BUILD_MANIFEST",
        ROOT / "build/native-resilient-dataplane/native-artifacts.json",
    )
)
LEAN_RUNNER = Path(
    os.environ.get(
        "EMENDER_LEAN_CONFORMANCE_RUNNER",
        ROOT / "formal/resilient/.lake/build/bin/resilient-conformance",
    )
)


def _require_runtime() -> None:
    if not BUILD_MANIFEST.is_file():
        pytest.skip("canonical native bundle has not been built")
    if not LEAN_RUNNER.is_file():
        pytest.skip("pinned Lean conformance runner has not been built")


def _manifest() -> dict[str, object]:
    return json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))


def test_corpus_is_canonical_digest_bound_and_keeps_job_5105811() -> None:
    manifest = _manifest()
    assert manifest["schema"] == "emender-native-lean-conformance-corpus-v1"
    assert manifest["traceSchemaDigest"] == TRACE_SCHEMA_DIGEST
    entries = manifest["entries"]
    assert len(entries) == 15
    identities = {entry["id"] for entry in entries}
    assert "native-job-5105811-generation-3-close-restart-rejoin" in identities
    assert {
        "native-duplicate-conflict-stale-late-corrupt",
        "native-leased-ready-delay-expiry-insufficient",
        "native-owner-replay-reassignment-abort",
        "native-participant-restart-open",
        "native-service-restart-open",
        "native-service-restart-closed",
        "native-service-restart-committed-apply",
        "native-manager-restart-open",
        "native-manager-restart-closed",
        "native-manager-restart-committed-apply",
        "native-trainer-restart-open",
        "native-trainer-restart-closed",
        "native-trainer-restart-committed-apply",
        "native-fresh-fence-recovery",
    }.issubset(identities)
    for entry in entries:
        path = ROOT / entry["path"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        trace = load_canonical_trace(path)
        assert len(trace["steps"]) == entry["events"]
        assert entry["id"] in entry["replayCommand"]


def test_native_loader_rejects_duplicate_reordered_and_incomplete_traces(
    tmp_path: Path,
) -> None:
    retained = next(
        entry
        for entry in _manifest()["entries"]
        if entry["id"]
        == "native-job-5105811-generation-3-close-restart-rejoin"
    )
    source = ROOT / retained["path"]
    decoded = load_canonical_trace(source)

    reordered = tmp_path / "reordered.json"
    reordered.write_text(
        json.dumps(decoded, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceFormatError, match="deterministic"):
        load_canonical_trace(reordered)

    incomplete = dict(decoded)
    incomplete.pop("sourceDigest")
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(
        canonical_json(incomplete) + "\n", encoding="utf-8"
    )
    with pytest.raises(TraceFormatError, match="identity set"):
        load_canonical_trace(incomplete_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schemaVersion":"x","schemaVersion":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(TraceFormatError, match="duplicate"):
        load_canonical_trace(duplicate)


def test_lean_runner_rejects_canonical_unknown_event_before_native_mutation(
    tmp_path: Path,
) -> None:
    _require_runtime()
    retained = _manifest()["entries"][0]
    decoded = load_canonical_trace(ROOT / retained["path"])
    event = decoded["steps"][0]["event"]
    event["unknownProductionEvent"] = event.pop("peerTransition")
    malformed = tmp_path / "unknown-event.json"
    malformed.write_text(canonical_json(decoded) + "\n", encoding="utf-8")
    with pytest.raises(TraceFormatError, match="Lean rejected"):
        run_lean_oracle(malformed, LEAN_RUNNER)


def test_source_audit_reaches_the_production_transition_entry_point() -> None:
    audit_production_call_path(ROOT)


def test_every_corpus_trace_matches_live_production_service(
    tmp_path: Path,
) -> None:
    _require_runtime()
    reports = []
    for entry in _manifest()["entries"]:
        report = run_differential_trace(
            trace_path=ROOT / entry["path"],
            build_manifest=BUILD_MANIFEST,
            lean_runner=LEAN_RUNNER,
            repository=ROOT,
            divergence_directory=tmp_path / "divergences",
        )
        assert report["verdict"] == "agreement"
        assert report["events"] == entry["events"]
        assert len(report["finalStateDigest"]) == 64
        assert report["identityManifest"]["native"]["localAbi"] == 0x00010000
        assert report["identityManifest"]["callPath"][-1] == (
            "coordination::step"
        )
        reports.append(report)
    assert sum(report["events"] for report in reports) == 486


def test_fault_injected_actual_mutation_emits_first_divergence_replay(
    tmp_path: Path,
) -> None:
    _require_runtime()
    retained = next(
        entry
        for entry in _manifest()["entries"]
        if entry["id"]
        == "native-job-5105811-generation-3-close-restart-rejoin"
    )
    with pytest.raises(ConformanceDivergence) as caught:
        run_differential_trace(
            trace_path=ROOT / retained["path"],
            build_manifest=BUILD_MANIFEST,
            lean_runner=LEAN_RUNNER,
            repository=ROOT,
            divergence_directory=tmp_path,
            fault_event_index=25,
        )
    report = caught.value.report
    assert report["verdict"] == "first_divergence"
    assert report["eventIndex"] == 25
    assert report["eventKind"] == "contribution"
    native_call = report["native"]["calls"][-1]
    assert native_call["projection"] == "contribution-receipt"
    assert native_call["preStateDigest"] != native_call["postStateDigest"]
    assert any("exactTokens" in item or "receiptDigest" in item
               for item in report["differences"])
    replay = Path(report["replayTrace"])
    replay_trace = load_canonical_trace(replay)
    assert len(replay_trace["steps"]) == 26
    assert str(replay.resolve()) in report["replayCommand"]
    assert "--fault-event-index 25" in report["replayCommand"]
    report_path = Path(report["reportPath"])
    assert report_path.is_file()
    assert (
        json.loads(report_path.read_text(encoding="utf-8"))["reportPath"]
        == str(report_path.resolve())
    )


def test_python_state_digest_matches_lean_oracle() -> None:
    _require_runtime()
    retained = _manifest()["entries"][0]
    oracle = run_lean_oracle(ROOT / retained["path"], LEAN_RUNNER)
    for step in oracle["steps"]:
        assert state_digest(step["state"]) == step["stateDigest"]
