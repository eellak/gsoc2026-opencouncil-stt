from eval.tsfusion.timing import MAX_BRACKET, place

EX = [{"speaker": "A", "start": 0.0, "end": 20.0},
      {"speaker": "B", "start": 20.0, "end": 40.0}]
BOUNDS = (0.0, 40.0)


def src(start=None, end=None, prov="observed_word"):
    if start is None:
        return None
    return {"start": start, "end": end, "provenance": prov}


def test_an_observed_column_stores_the_union_and_the_median_midpoint():
    cols = [{"soniox": src(1.0, 1.4), "whisper": src(1.1, 1.6)}]
    (c,) = place(cols, EX, BOUNDS)
    assert c.time_method == "observed"
    assert (c.time_start, c.time_end) == (1.0, 1.6)      # union, never narrower
    assert c.time_uncertainty == 0.6
    assert abs(c.representative - 1.275) < 1e-9          # median of 1.2 and 1.35
    assert not c.time_conflict


def test_sources_that_do_not_overlap_within_tolerance_are_flagged():
    # the classic MSA collision: the same word twice, seconds apart
    cols = [{"soniox": src(1.0, 1.4), "whisper": src(8.0, 8.4)}]
    (c,) = place(cols, EX, BOUNDS)
    assert c.time_conflict
    assert c.conflict_gap == 6.6
    # and the interval is honest about how wide it therefore is
    assert c.time_uncertainty == 7.4


def test_a_small_disagreement_is_not_a_conflict():
    cols = [{"soniox": src(1.0, 1.4), "whisper": src(1.5, 1.9)}]
    (c,) = place(cols, EX, BOUNDS)
    assert not c.time_conflict


def test_scribe_only_columns_are_bracketed_and_divided_monotonically():
    cols = [{"soniox": src(1.0, 1.2)},
            {"scribe": src()}, {"scribe": src()}, {"scribe": src()},
            {"soniox": src(2.2, 2.4)}]
    got = place(cols, EX, BOUNDS)
    mid = got[1:4]
    assert [c.time_method for c in mid] == ["bracketed"] * 3
    # every bracketed column carries the FULL gap as its interval ...
    assert all((c.time_start, c.time_end) == (1.2, 2.2) for c in mid)
    assert all(c.time_uncertainty == 1.0 for c in mid)
    # ... while the layout point advances monotonically inside it
    reps = [c.representative for c in mid]
    assert reps == sorted(reps)
    assert all(1.2 <= r <= 2.2 for r in reps)
    assert not any(c.unresolved for c in mid)


def test_a_bracket_wider_than_the_cap_is_unresolved():
    cols = [{"soniox": src(1.0, 1.2)}, {"scribe": src()},
            {"soniox": src(9.0, 9.2)}]
    got = place(cols, EX, BOUNDS)
    c = got[1]
    assert c.time_method == "bracketed"
    assert c.unresolved
    assert f"> {MAX_BRACKET}s" in c.unresolved_reason


def test_a_bracket_crossing_a_handover_is_unresolved_even_when_narrow():
    cols = [{"soniox": src(19.8, 19.9)}, {"scribe": src()},
            {"soniox": src(20.2, 20.3)}]
    got = place(cols, EX, BOUNDS)
    assert got[1].unresolved
    assert "handover" in got[1].unresolved_reason


def test_a_bracket_crossing_a_silence_is_unresolved():
    ex = [{"speaker": "A", "start": 0.0, "end": 5.0},
          {"speaker": "A", "start": 8.0, "end": 12.0}]
    cols = [{"soniox": src(4.9, 5.0)}, {"scribe": src()},
            {"soniox": src(5.2, 5.3)}]
    got = place(cols, ex, (0.0, 12.0))
    assert got[1].unresolved
    assert "silence" in got[1].unresolved_reason


def test_the_leading_edge_extrapolates_a_little_then_gives_up():
    cols = [{"scribe": src()}, {"soniox": src(5.0, 5.3)}]
    got = place(cols, EX, BOUNDS)
    assert got[0].time_method == "extrapolated"
    assert got[0].unresolved                       # extrapolation is never trusted
    assert got[0].time_end == 5.0
    assert got[0].time_start >= 4.0                # capped reach


def test_too_many_columns_past_the_last_anchor_go_to_the_unplaced_gutter():
    cols = [{"soniox": src(5.0, 5.3)}] + [{"scribe": src()} for _ in range(10)]
    got = place(cols, EX, BOUNDS)
    assert [c.time_method for c in got[1:]] == ["unplaced"] * 10
    assert all(c.time_start is None for c in got[1:])
    assert all(c.unresolved for c in got[1:])


def test_a_column_set_with_no_anchor_at_all_is_entirely_unplaced():
    cols = [{"scribe": src()} for _ in range(3)]
    got = place(cols, EX, BOUNDS)
    assert [c.time_method for c in got] == ["unplaced"] * 3
    assert all(c.representative is None for c in got)


def test_overlapping_anchors_do_not_produce_a_negative_bracket():
    cols = [{"soniox": src(3.0, 4.0)}, {"scribe": src()},
            {"whisper": src(3.5, 3.8)}]
    got = place(cols, EX, BOUNDS)
    c = got[1]
    assert c.time_start <= c.time_end
    assert c.time_uncertainty == 0.0
