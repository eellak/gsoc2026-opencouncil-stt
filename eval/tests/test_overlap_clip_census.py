import pytest

from eval.controlled_eval.overlap_clip_census import (
    GUARD_SEC,
    active_seconds,
    judge,
    overlap_seconds,
    waterfall,
)


def test_active_seconds_merges_same_label_and_ignores_order():
    turns = [("B", 5.0, 6.0), ("A", 0.0, 2.0), ("A", 1.0, 3.0)]
    assert active_seconds(turns, 0.0, 10.0) == {"A": 3.0, "B": 1.0}
    assert active_seconds(list(reversed(turns)), 0.0, 10.0) == {"A": 3.0, "B": 1.0}


def test_active_seconds_clips_to_the_window():
    assert active_seconds([("A", -5.0, 1.0), ("A", 9.0, 40.0)], 0.0, 10.0) == {"A": 2.0}


def test_overlap_counts_distinct_labels_only():
    assert overlap_seconds([("A", 0.0, 2.0), ("A", 1.0, 3.0)], 0.0, 10.0) == 0.0
    assert overlap_seconds([("A", 0.0, 2.0), ("B", 1.0, 3.0)], 0.0, 10.0) == 1.0
    assert overlap_seconds([("A", 0.0, 3.0), ("B", 1.0, 2.0), ("C", 1.5, 2.5)],
                           0.0, 10.0) == pytest.approx(1.5)


def test_clean_single_speaker_utterance_is_accepted():
    turns = [("A", 0.0, 20.0)]
    row = judge(turns, 5.0, 9.0, 20.0)
    assert row["accepted"] is True
    assert row["flags"] == []
    assert row["primary"] == "A"
    assert row["coverage"] == 1.0


def test_a_second_speaker_inside_rejects():
    turns = [("A", 0.0, 20.0), ("B", 6.0, 7.0)]
    row = judge(turns, 5.0, 9.0, 20.0)
    assert row["accepted"] is False
    assert "competing_inside" in row["flags"] and "overlap_guarded" in row["flags"]


def test_silence_gap_below_coverage_rejects():
    turns = [("A", 5.0, 6.0)]
    row = judge(turns, 5.0, 9.0, 20.0)
    assert row["accepted"] is False
    assert row["flags"] == ["low_coverage"]
    assert row["coverage"] == pytest.approx(0.25)


def test_another_speaker_alone_just_outside_the_cut_rejects():
    turns = [("A", 5.0, 9.0), ("B", 9.05, 9.6)]
    row = judge(turns, 5.0, 9.0, 20.0)
    assert row["accepted"] is False
    assert row["flags"] == ["competing_guard"]


def test_tie_between_labels_rejects():
    turns = [("A", 5.0, 7.0), ("B", 7.0, 9.0)]
    row = judge(turns, 5.0, 9.0, 20.0)
    assert row["accepted"] is False
    assert "no_primary" in row["flags"]


def test_no_speech_at_all_rejects():
    row = judge([("A", 15.0, 18.0)], 5.0, 9.0, 20.0)
    assert row["accepted"] is False
    assert "no_primary" in row["flags"]


def test_guard_band_is_clipped_at_the_recording_edges():
    turns = [("A", 0.0, 4.0)]
    row = judge(turns, 0.0, 4.0, 4.0)
    assert row["accepted"] is True
    assert row["guard"] == [0.0, 4.0]
    assert GUARD_SEC == 0.3


def test_tolerances_admit_a_sliver_and_reject_just_past_it():
    ok = judge([("A", 5.0, 9.0), ("B", 6.0, 6.04)], 5.0, 9.0, 20.0)
    bad = judge([("A", 5.0, 9.0), ("B", 6.0, 6.06)], 5.0, 9.0, 20.0)
    assert ok["accepted"] is True
    assert bad["accepted"] is False


def test_waterfall_assigns_each_rejection_once():
    rows = [
        {"accepted": True, "flags": []},
        {"accepted": False, "flags": ["competing_inside", "overlap_guarded"]},
        {"accepted": False, "flags": ["low_coverage"]},
    ]
    counts = waterfall(rows)
    assert sum(counts.values()) == 2
    assert counts["competing_inside"] == 1 and counts["low_coverage"] == 1


def test_padding_grows_into_silence_but_stops_at_another_speaker():
    from eval.controlled_eval.overlap_clip_census import pad_bounds
    turns = [("A", 5.0, 9.0), ("B", 9.1, 12.0)]
    assert pad_bounds(turns, 5.0, 9.0, 20.0) == (4.75, 9.1)
    assert pad_bounds([("A", 5.0, 9.0)], 5.0, 9.0, 9.05) == (4.75, 9.05)


def test_contiguous_pack_covers_one_span_of_one_speaker():
    from eval.controlled_eval.overlap_clip_census import contiguous_packs
    turns = [("A", 0.0, 40.0)]
    rows = [{"audio_url": "u", "city_id": "c", "meeting_id": "m", "person_id": "p1",
             "utterance_id": f"u{i}", "start": 1.0 + 3.0 * i, "end": 3.5 + 3.0 * i,
             "duration_sec": 2.5} for i in range(9)]
    packs = contiguous_packs(rows, {"u": turns}, {"u": 40.0})
    assert len(packs) == 1
    pack = packs[0]
    assert pack["n_utterances"] == 9
    assert pack["speech_sec"] == pytest.approx(22.5)
    assert pack["span_sec"] <= 29.0
    assert pack["start"] < 1.0 and pack["end"] > 25.5     # padded into silence


def test_contiguous_pack_breaks_on_speaker_change():
    from eval.controlled_eval.overlap_clip_census import contiguous_packs
    rows = [{"audio_url": "u", "city_id": "c", "meeting_id": "m",
             "person_id": "p1" if i < 4 else "p2",
             "utterance_id": f"u{i}", "start": 1.0 + 3.0 * i, "end": 3.5 + 3.0 * i,
             "duration_sec": 2.5} for i in range(9)]
    assert contiguous_packs(rows, {"u": [("A", 0.0, 40.0)]}, {"u": 40.0}) == []
