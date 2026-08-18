from eval.controlled_eval.scoring import edist
from eval.tsfusion.refalign import align_to_reference, place_deletions

EX = [{"speaker": "A", "start": 0.0, "end": 10.0},
      {"speaker": "A", "start": 20.0, "end": 30.0}]
BOUNDS = (0.0, 30.0)


def test_distance_matches_the_frozen_scorer():
    # invented Greek, not council speech: no transcript text goes in git
    ref = "καλημερα σε ολους τους φιλους της γειτονιας".split()
    hyp = "καλημερα σε ολους φιλους της γειτονιας μας".split()
    a = align_to_reference("scribe", ref, hyp)
    assert a.distance == edist(ref, hyp)
    assert a.counts["delete"] == 1 and a.counts["insert"] == 1


def test_operations_reconstruct_both_sequences():
    ref = "ενα δυο τρια τεσσερα".split()
    hyp = "ενα τρια πεντε τεσσερα".split()
    a = align_to_reference("soniox", ref, hyp)
    assert [o.ref_word for o in a.ops if o.ref_word] == ref
    assert [o.hyp_word for o in a.ops if o.hyp_word] == hyp


def test_a_repeated_word_makes_the_mapping_ambiguous():
    ref = ["ναι", "ναι"]
    hyp = ["ναι"]
    a = align_to_reference("whisper", ref, hyp)
    assert a.counts["delete"] == 1
    # which of the two "ναι" was deleted is not decidable, and is marked so
    assert a.counts["ambiguous"] >= 1


def test_an_unambiguous_deletion_is_not_flagged():
    ref = ["ενα", "δυο", "τρια"]
    hyp = ["ενα", "τρια"]
    a = align_to_reference("whisper", ref, hyp)
    (d,) = a.deletions()
    assert d.ref_word == "δυο"
    assert not d.ambiguous


def test_a_deletion_another_system_emitted_takes_that_interval():
    a = align_to_reference("whisper", ["ενα", "δυο", "τρια"], ["ενα", "τρια"])
    evs = place_deletions("whisper", a,
                          ref_anchor_time={0: (1.0, 1.2), 2: (3.0, 3.2)},
                          proposals={1: {"soniox": (2.0, 2.4)}},
                          exclusive=EX, bounds=BOUNDS)
    (e,) = evs
    assert e.method == "proposal"
    assert (e.time_start, e.time_end) == (2.0, 2.4)
    assert e.label == "asr_deletion"
    assert "soniox" in e.note


def test_a_deletion_nobody_emitted_is_bracketed_between_timed_neighbours():
    a = align_to_reference("whisper", ["ενα", "δυο", "τρια"], ["ενα", "τρια"])
    (e,) = place_deletions("whisper", a, {0: (1.0, 1.2), 2: (3.0, 3.2)}, {},
                           EX, BOUNDS)
    assert e.method == "bracketed"
    assert (e.time_start, e.time_end) == (1.2, 3.0)


def test_a_one_sided_anchor_gives_an_open_ended_interval():
    a = align_to_reference("whisper", ["ενα", "δυο"], ["ενα"])
    (e,) = place_deletions("whisper", a, {0: (1.0, 1.2)}, {}, EX, BOUNDS)
    assert e.method == "open_right" and e.open_ended
    assert e.time_end == BOUNDS[1]

    a2 = align_to_reference("whisper", ["ενα", "δυο"], ["δυο"])
    (e2,) = place_deletions("whisper", a2, {1: (5.0, 5.2)}, {}, EX, BOUNDS)
    assert e2.method == "open_left" and e2.open_ended
    assert e2.time_start == BOUNDS[0]


def test_a_deletion_in_detected_silence_is_not_called_an_asr_deletion():
    # the bracket 12.0..18.0 sits in the diarizer's gap between the two turns
    a = align_to_reference("whisper", ["ενα", "δυο", "τρια"], ["ενα", "τρια"])
    (e,) = place_deletions("whisper", a, {0: (11.0, 12.0), 2: (18.0, 19.0)}, {},
                           EX, BOUNDS)
    assert e.label == "reference_only_no_speech_detected"
    # and the wording never claims the word was inaudible
    assert "audible" not in e.note


def test_a_deletion_inside_speech_stays_an_asr_deletion():
    a = align_to_reference("whisper", ["ενα", "δυο", "τρια"], ["ενα", "τρια"])
    (e,) = place_deletions("whisper", a, {0: (1.0, 1.2), 2: (2.0, 2.2)}, {},
                           EX, BOUNDS)
    assert e.label == "asr_deletion"


def test_no_anchors_at_all_leaves_the_event_unplaced():
    a = align_to_reference("whisper", ["ενα"], [])
    (e,) = place_deletions("whisper", a, {}, {}, EX, BOUNDS)
    assert e.method == "unplaced"
    assert e.time_start is None
