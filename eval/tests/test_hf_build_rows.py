"""Tests for build.py pure helpers (filtering/joining/stats), no network."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.build import build_stats, filter_rows, join_speakers


def _exp(uid, city, mtg, date, inc="include", start=0.0, end=2.0):
    return {"utterance_id": uid, "city_id": city, "meeting_id": mtg,
            "meeting_date": date, "include_status": inc,
            "start": start, "end": end, "audio_url": "http://x/a.mp3",
            "initial_before_text": "b", "final_after_text": "a",
            "error_categories": [], "reviewer_notes": None}


def test_filter_keeps_only_includes_and_logs():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00", inc="exclude"),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00", inc="uncertain")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u1"]
    assert drops["not_include"] == 2


def test_filter_drops_denylist_and_temporal():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "rhodes", "jul17_2025", "2025-07-17 10:00:00"),
            _exp("u3", "athens", "m2", "2026-06-05 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys={("rhodes", "jul17_2025")})
    assert [r["utterance_id"] for r in kept] == ["u1"]
    assert drops["denylist"] == 1 and drops["temporal_test"] == 1


def test_filter_drops_missing_or_malformed_dates():
    rows = [_exp("u1", "athens", "m1", None),
            _exp("u2", "athens", "m1", "unknown"),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u3"]
    assert drops["bad_date"] == 2


def test_filter_rejects_multiple_audio_urls_per_meeting():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00")]
    rows[1]["audio_url"] = "http://x/OTHER.mp3"
    with pytest.raises(ValueError):
        filter_rows(rows, excluded_keys=set())


def test_filter_drops_bad_spans():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00", start=5.0, end=5.0),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00", start=9.0, end=5.0),
            _exp("u3", "athens", "m1", "2025-01-01 10:00:00")]
    kept, drops = filter_rows(rows, excluded_keys=set())
    assert [r["utterance_id"] for r in kept] == ["u3"]
    assert drops["bad_span"] == 2


def test_filter_raises_on_duplicate_composite_key():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u1", "athens", "m1", "2025-01-01 10:00:00")]
    with pytest.raises(ValueError):
        filter_rows(rows, excluded_keys=set())


def test_join_speakers_matches_and_counts_missing():
    rows = [_exp("u1", "athens", "m1", "2025-01-01 10:00:00"),
            _exp("u2", "athens", "m1", "2025-01-01 10:00:00")]
    spk = {"u1": "person-9"}
    joined, n_missing = join_speakers(rows, spk)
    assert joined[0]["speaker_id"] == "person-9"
    assert joined[1]["speaker_id"] is None
    assert n_missing == 1


def test_build_stats_hours_and_percentages():
    rows = [
        {"split": "train", "duration_s": 3600.0, "city_id": "athens",
         "error_categories": ["homophone"], "has_overlap": False,
         "boundary_status": "ok", "speaker_id": "s1"},
        {"split": "validation", "duration_s": 1800.0, "city_id": "argos",
         "error_categories": [], "has_overlap": True,
         "boundary_status": "suspect_bleed_in", "speaker_id": "s2"},
    ]
    st = build_stats(rows)
    assert st["total_hours"] == 1.5
    assert st["by_split"]["train"]["hours"] == 1.0
    assert st["by_split"]["validation"]["pct_hours"] == 33.3
    assert st["overlap_rows"] == 1
    assert st["boundary_status"]["suspect_bleed_in"] == 1
    assert st["by_split"]["train"]["speakers"] == 1
