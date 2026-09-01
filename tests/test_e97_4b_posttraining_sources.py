import json
from pathlib import Path

from scripts.audit_e97_4b_posttraining_sources import SCHEMA, SOURCES


def test_posttraining_source_pins_are_immutable_and_fail_closed():
    assert len(SOURCES) >= 6
    for source in SOURCES.values():
        assert len(source["revision"]) == 40
        int(source["revision"], 16)
        assert len(source["card_sha256"]) == 64
        int(source["card_sha256"], 16)
        assert source["repository_bytes"] > 0
        assert source["decision"] != "admitted"


def test_committed_source_audit_matches_pins():
    receipt = json.loads(Path(
        "docs/validation/e97-4b-broad-posttraining-source-audit.json"
    ).read_text())
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "pass"
    assert set(receipt["sources"]) == set(SOURCES)
    for name, source in SOURCES.items():
        observed = receipt["sources"][name]
        for key, value in source.items():
            assert observed[key] == value
        assert observed["verified"] is True
