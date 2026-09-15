#!/usr/bin/env python3
"""Tests for the scorer, from Astra's pre-run list.

The point of each: a metric that passes these cannot be gamed by the two
degenerate behaviours that would otherwise look like success — merging
everything into one voice, or splitting everything into singletons.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from score_arms import assign, bcubed, reference_turns, score  # noqa: E402
from score_naming import naming_report  # noqa: E402


def seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def turn(start, end, identity, resolved=True):
    return {"start": start, "end": end, "identity": identity, "resolved": resolved}


# --- the two degenerate systems -------------------------------------------

def test_total_merge_destroys_precision_but_not_recall():
    """One voice for three people: merging is visible, and recall stays perfect."""
    turns = [turn(0, 1, "A"), turn(2, 3, "B"), turn(4, 5, "C")]
    segs = [seg(0, 10, "V0")]
    r = score(turns, segs)
    assert r["bcubed"]["precision"] < 0.4
    assert r["bcubed"]["recall"] == 1.0


def test_total_fragmentation_destroys_recall_but_not_precision():
    """One voice per utterance: perfect purity, and the metric still refuses it."""
    turns = [turn(0, 1, "A"), turn(2, 3, "A"), turn(4, 5, "A")]
    segs = [seg(0, 1, "V0"), seg(2, 3, "V1"), seg(4, 5, "V2")]
    r = score(turns, segs)
    assert r["bcubed"]["precision"] == 1.0
    assert r["bcubed"]["recall"] < 0.4
    assert r["bcubed"]["f1"] < 0.6  # cannot pass a gate on F


def test_perfect_reconstruction_scores_one():
    turns = [turn(0, 1, "A"), turn(2, 3, "B"), turn(4, 5, "A")]
    segs = [seg(0, 1, "V0"), seg(2, 3, "V1"), seg(4, 5, "V0")]
    r = score(turns, segs)
    assert r["bcubed"]["f1"] == 1.0


def test_label_permutation_leaves_the_score_unchanged():
    turns = [turn(0, 1, "A"), turn(2, 3, "B")]
    a = score(turns, [seg(0, 1, "V0"), seg(2, 3, "V1")])
    b = score(turns, [seg(0, 1, "zzz"), seg(2, 3, "aaa")])
    assert a["bcubed"] == b["bcubed"]


# --- the failure modes that would flatter an arm --------------------------

def test_refusing_to_answer_cannot_improve_the_score():
    """Unassigned turns become singletons, so silence is not free."""
    turns = [turn(0, 1, "A"), turn(2, 3, "A"), turn(4, 5, "A")]
    answered = score(turns, [seg(0, 5, "V0")])
    silent = score(turns, [seg(0, 1, "V0")])  # covers only the first turn
    assert silent["unassigned"] == 2
    assert silent["bcubed"]["f1"] < answered["bcubed"]["f1"]


def test_rare_speaker_absorption_is_penalised():
    """A one-turn speaker swallowed by the dominant voice must cost precision."""
    turns = [turn(0, 10, "CHAIR"), turn(10, 20, "CHAIR"), turn(20, 21, "RARE")]
    absorbed = score(turns, [seg(0, 21, "V0")])
    correct = score(turns, [seg(0, 20, "V0"), seg(20, 21, "V1")])
    assert absorbed["bcubed"]["precision"] < correct["bcubed"]["precision"]


def test_duration_weighting_does_not_let_one_long_turn_hide_many_short_ones():
    """Fifteen short speakers merged into the chair's voice: unweighted sees it."""
    turns = [turn(0, 600, "CHAIR")] + [turn(600 + i, 601 + i, f"P{i}") for i in range(15)]
    segs = [seg(0, 700, "V0")]
    r = score(turns, segs)
    assert r["bcubed"]["precision"] < 0.4                      # one vote per turn
    assert r["bcubed_duration_weighted"]["precision"] > 0.9    # seconds say otherwise
    # Both are reported precisely because they disagree here.


def test_unresolved_tags_are_counted_not_silently_dropped():
    turns = [turn(0, 1, "A"), turn(2, 3, "tag:x", resolved=False)]
    r = score(turns, [seg(0, 3, "V0")])
    assert r["turns"] == 2
    assert r["reference_identities"] == 2
    assert r["reference_resolved_people"] == 1
    assert r["bcubed_resolved_only"]["items"] == 1


# --- assignment rules ------------------------------------------------------

def test_naming_only_counts_enrolled_labels_as_a_claim():
    """An arm that emits SPEAKER_xx has not misnamed anyone — it named no one.

    Scoring those as wrong names made a0/a3/c3 look like they misnamed 40% of
    speech, which is false and makes the voiceprint arms look better than they are.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from score_naming import naming_report

    turns = [turn(0, 1, "person_A"), turn(2, 3, "person_B")]
    enrolled = {"person_A", "person_B"}

    plain = [seg(0, 1, "SPEAKER_00"), seg(2, 3, "SPEAKER_01")]
    r = naming_report(turns, plain, enrolled)
    assert r["no_name_claimed"] == 1.0
    assert r["wrong_name_on_unenrolled"] == 0.0
    assert r["wrong_name_on_enrolled"] == 0.0

    named = [seg(0, 1, "person_A"), seg(2, 3, "person_A")]  # second one wrong
    r2 = naming_report(turns, named, enrolled)
    assert r2["correct_over_all_turns"] == 0.5
    assert r2["wrong_name_on_enrolled"] == 0.5
    assert r2["no_name_claimed"] == 0.0


def test_naming_flags_a_name_given_to_someone_never_enrolled():
    turns = [turn(0, 1, "tag:unknown_person")]
    enrolled = {"person_A"}
    r = naming_report(turns, [seg(0, 1, "person_A")], enrolled)
    assert r["wrong_name_on_unenrolled"] == 1.0


def test_assignment_sums_overlap_per_speaker():
    """A speaker split across several segments must not lose to one long segment.

    Diarizers emit one segment per continuous stretch of voice, so this is the
    normal case, not a corner case.
    """
    t = turn(0, 10, "A")
    segs = [seg(0, 3, "V0"), seg(3, 6, "V1"), seg(6, 9, "V0")]  # V0 = 6s, V1 = 3s
    assert assign(t, segs) == "V0"
    # and the loser's single segment being the widest does not save it
    segs2 = [seg(0, 2, "V0"), seg(2, 5, "V1"), seg(5, 7, "V0"), seg(7, 9, "V0")]
    assert assign(t, segs2) == "V0"  # 6s across three segments vs V1's 3s


def test_assignment_uses_max_overlap_not_midpoint():
    """The midpoint must land on the loser, or the test proves nothing."""
    t = turn(0, 10, "A")
    segs = [seg(0, 4, "V0"), seg(4, 6, "V1"), seg(6, 10, "V0")]
    # midpoint 5 sits inside V1; total overlap is V0 8s against V1 2s
    assert assign(t, segs) == "V0"


def test_no_overlap_returns_none():
    assert assign(turn(0, 1, "A"), [seg(5, 6, "V0")]) is None


def test_empty_input_does_not_divide_by_zero():
    assert bcubed([])["f1"] == 0.0


# --- reference construction ------------------------------------------------

def test_person_id_wins_over_tag_so_split_tags_collapse_to_one_person():
    """One person carrying several tags must be ONE reference identity."""
    snap = {
        "speakerTags": [
            {"id": "t1", "personId": "p1"},
            {"id": "t2", "personId": "p1"},   # same person, second tag
            {"id": "t3", "personId": None},
        ],
        "transcript": [
            {"speakerTagId": "t1", "utterances": [
                {"startTimestamp": 0, "endTimestamp": 1, "text": "a"}]},
            {"speakerTagId": "t2", "utterances": [
                {"startTimestamp": 2, "endTimestamp": 3, "text": "b"}]},
            {"speakerTagId": "t3", "utterances": [
                {"startTimestamp": 4, "endTimestamp": 5, "text": "c"}]},
        ],
    }
    turns = reference_turns(snap)
    assert [t["identity"] for t in turns] == ["p1", "p1", "tag:t3"]
    assert [t["resolved"] for t in turns] == [True, True, False]
