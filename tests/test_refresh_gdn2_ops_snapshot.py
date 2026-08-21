from __future__ import annotations

import datetime as dt

from scripts.refresh_gdn2_ops_snapshot import Point, target_eta


def _rates():
    rate = {"steps_per_sec": 1.0, "tokens_per_sec": 65536.0}
    return {"recent_1h": rate, "recent_6h": rate, "since_launch": rate}


def test_completed_target_has_zero_remaining_not_negative_eta() -> None:
    point = Point(
        step=2_323_600,
        loss=2.4,
        tok_s=7800.0,
        global_tok_s=62_000.0,
        timestamp=dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc),
        source="test",
        order=1,
        source_line=1,
    )
    result = target_eta(point, 150_000_000_000, _rates())
    assert result["target_reached"] is True
    assert result["remaining_tokens"] == 0
    assert result["remaining_steps_exact"] == 0
    assert result["tokens_over_target"] > 0
    assert result["primary_eta"]["duration_seconds"] == 0
