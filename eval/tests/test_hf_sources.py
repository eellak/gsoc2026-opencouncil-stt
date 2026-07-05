"""Tests for combined-source helpers (slug date parsing + span dedupe)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.hf_export.sources import dedupe_by_span, slug_date, span_key


def test_slug_date_standard():
    assert slug_date("aug14_2025") == "2025-08-14 00:00:00"
    assert slug_date("jan13_2025") == "2025-01-13 00:00:00"
    assert slug_date("apr24_2026") == "2026-04-24 00:00:00"


def test_slug_date_numbered_variants():
    # two conventions for a 2nd-meeting-that-day: N before OR after the year
    assert slug_date("oct30_2_2025") == "2025-10-30 00:00:00"
    assert slug_date("apr28_2025_2") == "2025-04-28 00:00:00"


def test_slug_date_unparseable():
    assert slug_date("weird-slug") is None
    assert slug_date("2025_special") is None
    assert slug_date("") is None


def test_span_key_rounds():
    a = {"audio_url": "http://x/a.mp3", "start": 10.111, "end": 12.999}
    b = {"audio_url": "http://x/a.mp3", "start": 10.113, "end": 13.001}
    assert span_key(a) == span_key(b)  # within 0.02s -> same clip


def test_dedupe_prefers_correction_over_no_edit():
    rows = [
        {"utterance_id": "u1", "audio_url": "m/a.mp3", "start": 1.0, "end": 2.0,
         "source": "no_edit", "_rank": 2},
        {"utterance_id": "u2", "audio_url": "m/a.mp3", "start": 1.0, "end": 2.0,
         "source": "correction", "_rank": 0},  # same span, correction wins
        {"utterance_id": "u3", "audio_url": "m/b.mp3", "start": 5.0, "end": 6.0,
         "source": "no_edit", "_rank": 2},
    ]
    kept, dropped = dedupe_by_span(rows)
    ids = {r["utterance_id"] for r in kept}
    assert ids == {"u2", "u3"}
    assert dropped == 1


def test_dedupe_prefers_lower_rank_within_same_span():
    rows = [
        {"utterance_id": "leftover", "audio_url": "m/a.mp3", "start": 1.0, "end": 2.0,
         "source": "correction", "_rank": 1},
        {"utterance_id": "include", "audio_url": "m/a.mp3", "start": 1.0, "end": 2.0,
         "source": "correction", "_rank": 0},  # include (rank 0) beats leftover (rank 1)
    ]
    kept, dropped = dedupe_by_span(rows)
    assert [r["utterance_id"] for r in kept] == ["include"]
    assert dropped == 1
