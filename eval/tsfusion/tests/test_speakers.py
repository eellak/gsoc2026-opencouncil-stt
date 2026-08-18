from eval.tsfusion.speakers import (assign, crosses_handover, crosses_silence,
                                    handovers, silence_gaps)


def turn(spk, a, b):
    return {"speaker": spk, "start": a, "end": b}


EX = [turn("A", 0.0, 10.0), turn("B", 10.0, 20.0), turn("A", 25.0, 30.0)]


def test_a_word_well_inside_one_turn_is_named():
    c = assign(3.0, 3.5, EX, EX)
    assert c.state == "named" and c.speaker == "A"
    assert c.overlap_fraction == 1.0


def test_max_overlap_beats_the_midpoint_rule():
    # A stops at 5.0, B starts at 5.4. The midpoint of [4.5, 6.0) is 5.25, which
    # falls in the gap and would answer "nobody"; by overlap B holds more of the
    # interval than A does, and both are visible.
    ex = [turn("A", 0.0, 5.0), turn("B", 5.4, 6.0)]
    c = assign(4.5, 6.0, ex, ex)
    assert c.speaker == "B"
    assert c.state == "ambiguous"          # A still holds a real share
    assert c.runner_up == "A"
    assert c.shares == {"B": 0.4, "A": 0.3333}


def test_a_straddling_word_is_ambiguous_not_forced():
    c = assign(9.5, 10.5, EX, EX)
    assert c.state == "ambiguous"
    assert round(c.overlap_fraction, 3) == 0.5
    assert round(c.runner_up_fraction, 3) == 0.5


def test_a_word_in_a_gap_is_non_speech():
    c = assign(21.0, 22.0, EX, EX)
    assert c.state == "non_speech"
    assert c.speaker is None


def test_regular_diarization_multiplicity_marks_overlap():
    regular = EX + [turn("C", 3.0, 4.0)]
    c = assign(3.2, 3.4, EX, regular)
    assert c.state == "overlap"
    assert c.multiplicity == 2
    # the exclusive lane still supplies one name, and it is labelled as such
    assert c.speaker == "A"


def test_no_diarization_and_unresolved_are_distinct_from_non_speech():
    assert assign(3.0, 3.5, [], []).state == "no_diarization"
    assert assign(3.0, 3.5, EX, EX, unresolved=True).state == "unresolved"


def test_zero_width_interval_does_not_divide_by_zero():
    c = assign(5.0, 5.0, EX, EX)
    assert c.state == "named" and c.speaker == "A"


def test_handovers_and_silence():
    assert handovers(EX) == [10.0, 22.5]
    assert crosses_handover(9.0, 11.0, EX)
    assert not crosses_handover(3.0, 4.0, EX)
    assert silence_gaps(EX) == [(20.0, 25.0)]
    assert crosses_silence(19.0, 21.0, EX)
    assert not crosses_silence(3.0, 4.0, EX)
