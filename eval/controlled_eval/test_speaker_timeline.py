"""Frozen tests for the speaker-timeline sweep behind wayfinder #17 round 2.

These exist because the first version of `active_intervals` tracked a SET of active
labels instead of per-speaker counts, and a set gets several shapes of real diarizer
output wrong. Every case below is one of those shapes. If one of these breaks, the
partition in `exp_speaker_fusion.py` is silently wrong and A1/A2 are not reportable.

    python3 -m pytest eval/controlled_eval/test_speaker_timeline.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/harold/opencouncil-fine-tuning")))
from eval.controlled_eval.exp_speaker_fusion import (  # noqa: E402
    active_intervals, cells, handover_cuts, other_boundary_times, overlap_intervals,
    split_tokens)


def S(*triples):
    return [{"speaker": s, "start": a, "end": b} for s, a, b in triples]


def sets(segs):
    return [(round(a, 3), round(b, 3), sorted(sp)) for a, b, sp in active_intervals(segs)]


# --------------------------------------------------------------- active_intervals
def test_adjacent_same_speaker_segments_merge():
    # the set-based version dropped [1,2] entirely: A was added and removed at t=1
    assert sets(S(("A", 0, 1), ("A", 1, 2))) == [(0.0, 2.0, ["A"])]


def test_overlapping_same_speaker_segments_stay_active():
    # the first `end` must not clear a speaker who is still talking in another segment
    assert sets(S(("A", 0, 2), ("A", 1, 3))) == [(0.0, 3.0, ["A"])]


def test_zero_length_segment_cannot_evict_a_live_speaker():
    assert sets(S(("A", 0, 3), ("A", 1, 1))) == [(0.0, 3.0, ["A"])]


def test_inverted_segment_is_ignored():
    assert sets(S(("A", 0, 2), ("B", 5, 4))) == [(0.0, 2.0, ["A"])]


def test_a_ends_exactly_when_b_starts():
    assert sets(S(("A", 0, 1), ("B", 1, 2))) == [(0.0, 1.0, ["A"]), (1.0, 2.0, ["B"])]


def test_overlap_is_exposed():
    assert sets(S(("A", 0, 2), ("B", 1, 3))) == [
        (0.0, 1.0, ["A"]), (1.0, 2.0, ["A", "B"]), (2.0, 3.0, ["B"])]
    assert overlap_intervals(S(("A", 0, 2), ("B", 1, 3))) == [(1.0, 2.0)]


def test_duplicate_segments_do_not_double_end():
    assert sets(S(("A", 0, 2), ("A", 0, 2))) == [(0.0, 2.0, ["A"])]


# ------------------------------------------------------------------ handover_cuts
def test_direct_handover_across_a_gap():
    cuts, spans = handover_cuts(S(("A", 0, 1), ("B", 2, 3)))
    assert cuts == [1.5] and spans == [(1.0, 2.0)]


def test_overlap_mediated_handover_is_detected():
    # {A} -> {A,B} -> {B}: neither step is disjoint, but the floor does pass to B.
    # The first version of the rule missed exactly this, i.e. every handover that
    # happens THROUGH overlap - the ones this experiment is about.
    cuts, _ = handover_cuts(S(("A", 0, 2), ("B", 1, 3)))
    assert cuts == [2.0]


def test_same_speaker_pause_is_not_a_handover():
    cuts, _ = handover_cuts(S(("A", 0, 1), ("A", 2, 3)))
    assert cuts == []


def test_exclusive_style_timeline_reduces_to_label_change():
    segs = S(("A", 0, 1), ("A", 1, 2), ("B", 2, 3), ("A", 3, 4))
    cuts, _ = handover_cuts(segs)
    assert cuts == [2.0, 3.0]


# ------------------------------------------------------------- placebo pool hygiene
def test_placebo_pool_excludes_handover_transitions():
    segs = S(("A", 0, 2), ("B", 1, 3))          # overlap-mediated handover at t=2
    cuts, spans = handover_cuts(segs)
    pool = other_boundary_times(segs, spans)
    assert cuts == [2.0]
    assert all(t not in cuts for t in pool)


def test_placebo_pool_keeps_same_speaker_joins():
    segs = S(("A", 0, 1), ("A", 2, 3))
    _, spans = handover_cuts(segs)
    assert other_boundary_times(segs, spans) == [1.5]


# ------------------------------------------------------- partition exactly-once rule
def test_partition_assigns_every_token_exactly_once():
    toks = list("abcdefg")
    times = [0.1, 0.5, 1.2, 1.4, 2.9, 3.1, 9.0]
    cs = cells([1.0, 3.0])
    parts = split_tokens(toks, times, cs)
    assert [t for p in parts for t in p] == toks
    assert sum(len(p) for p in parts) == len(toks)


def test_partition_with_no_cuts_is_the_whole_sequence():
    toks = list("abc")
    parts = split_tokens(toks, [0.0, 1.0, 2.0], cells([]))
    assert parts == [toks]
