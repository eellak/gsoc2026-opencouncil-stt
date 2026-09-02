"""Tests for the tempo probe (spec docs/specs/2026-08-28-tempo-probe.md).

Two of these encode mistakes that were already made once. `test_a_stretched_timestamp
_maps_back_by_multiplication` fixes a draft that divided instead, which at 0.7 put every
word at twice its real time. `test_an_unmatched_hypothesis_word_is_not_charged_as_an
_error` fixes a guard that would have rejected exactly the phenomenon under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import tempo_probe as TP  # noqa: E402


# ---------------------------------------------------------------- geometry

def test_a_stretched_timestamp_maps_back_by_multiplication():
    """atempo=0.7 makes the clip LONGER, so a returned time is LATER than the real one.

    Original 10 s -> returned near 14.29 s -> back to 10 s by multiplying by 0.7.
    """
    assert TP.to_original_ms(14286, 0.7) == pytest.approx(10000, abs=2)
    assert TP.to_original_ms(11765, 0.85) == pytest.approx(10000, abs=2)
    assert TP.to_original_ms(10000, 1.0) == 10000


def test_pacing_does_not_touch_timestamps():
    """`pace070` changes only how fast bytes are pushed; the audio is untouched, so its
    timestamps are already in original time."""
    arm = TP.ARMS["pace070"]
    assert arm.tempo == 1.0 and arm.feed == 0.7
    assert TP.to_original_ms(4321, arm.tempo) == 4321


def test_the_expected_duration_of_a_stretched_clip_is_checked():
    """The harness refuses a stretch whose output length is not what the rate implies;
    a silently failed filter would look like a null result."""
    assert TP.expected_dur(35.0, 0.7) == pytest.approx(50.0, abs=0.01)
    assert TP.expected_dur(35.0, 1.0) == 35.0


# ---------------------------------------------------------------- episodes

def _blocks():
    return [
        {"id": "p0", "s": 0.0, "e": 5.0, "text": "καλησπέρα σας", "ov_with": []},
        {"id": "p1", "s": 6.0, "e": 9.0, "text": "να πω κάτι", "ov_with": ["p2"]},
        {"id": "p2", "s": 7.5, "e": 10.0, "text": "όχι τώρα", "ov_with": ["p1"]},
        {"id": "p3", "s": 9.5, "e": 12.0, "text": "μισό λεπτό", "ov_with": ["p2"]},
        {"id": "p4", "s": 20.0, "e": 22.0, "text": "τελειώσαμε", "ov_with": []},
    ]


def test_linked_blocks_form_one_episode_not_three():
    """p1-p2 and p2-p3 are separate links but one conversation; scoring them apart
    would let a word matched in one be counted again in the other."""
    eps = TP.episodes(_blocks())
    assert len(eps) == 1
    ep = eps[0]
    assert ep.block_ids == ("p1", "p2", "p3")
    assert ep.start == 6.0 and ep.end == 12.0


def test_a_block_that_overlaps_nothing_is_not_an_episode():
    eps = TP.episodes(_blocks())
    assert all("p0" not in e.block_ids and "p4" not in e.block_ids for e in eps)


def test_an_asymmetric_ov_with_still_links():
    """The human page wrote ov_with on one side only in some cells."""
    blocks = [
        {"id": "a", "s": 1.0, "e": 3.0, "text": "ναι", "ov_with": ["b"]},
        {"id": "b", "s": 2.0, "e": 4.0, "text": "όχι", "ov_with": []},
    ]
    eps = TP.episodes(blocks)
    assert len(eps) == 1 and eps[0].block_ids == ("a", "b")


def test_uncertain_blocks_are_kept_apart_from_certain_ones():
    blocks = [
        {"id": "a", "s": 1.0, "e": 3.0, "text": "ναι βεβαίως", "ov_with": ["b"]},
        {"id": "b", "s": 2.0, "e": 4.0, "text": "όχι", "ov_with": ["a"],
         "text_unc": True},
    ]
    ep = TP.episodes(blocks)[0]
    assert ep.certain_words == ["ναι", "βεβαιωσ"]
    assert ep.uncertain_words == ["οχι"]


# ---------------------------------------------------------------- matching

def test_matching_is_one_to_one_so_a_repeat_is_not_double_counted():
    ep = TP.Episode(("a",), 0.0, 10.0, ["ναι", "ναι"], [])
    hyp = [TP.Tok("ναι", 1000, 1200)]
    m = TP.score_episode(ep, hyp)
    assert m.matched == 1 and m.n_ref == 2


def test_a_hypothesis_word_outside_the_episode_span_does_not_count():
    ep = TP.Episode(("a",), 5.0, 8.0, ["ναι"], [])
    hyp = [TP.Tok("ναι", 100, 300)]          # at 0.1-0.3 s, far outside
    assert TP.score_episode(ep, hyp).matched == 0


def test_matching_ignores_accents_and_case_like_the_project_normalizer():
    ep = TP.Episode(("a",), 0.0, 10.0, ["δημου"], [])
    hyp = [TP.Tok("Δήμου,", 1000, 1200)]
    assert TP.score_episode(ep, hyp).matched == 1


def test_an_unmatched_hypothesis_word_is_not_charged_as_an_error():
    """The human wrote the reference at 1.0x. A word the slowdown genuinely recovers
    may be absent from the reference; charging it as an insertion would reject exactly
    what the experiment is looking for. It is counted and sent to adjudication."""
    ep = TP.Episode(("a",), 0.0, 10.0, ["ναι"], [])
    hyp = [TP.Tok("ναι", 1000, 1200), TP.Tok("παρακαλώ", 2000, 2400)]
    m = TP.score_episode(ep, hyp)
    assert m.matched == 1
    assert m.unmatched_hyp == ["παρακαλω"]
    assert not hasattr(m, "insertions"), "unmatched words must not be scored as errors"


# ---------------------------------------------------------------- run order

def test_every_arm_runs_in_every_cycle_in_a_shuffled_order():
    plan = TP.run_plan(["c1", "c2"], cycles=3)
    assert len(plan) == 2 * len(TP.ARMS) * 3
    for cell in ("c1", "c2"):
        for cyc in range(3):
            arms = [j.arm for j in plan if j.cell == cell and j.cycle == cyc]
            assert sorted(arms) == sorted(TP.ARMS)


def test_the_order_is_shuffled_but_reproducible():
    a = [(j.cell, j.arm, j.cycle) for j in TP.run_plan(["c1", "c2"], cycles=3)]
    b = [(j.cell, j.arm, j.cycle) for j in TP.run_plan(["c1", "c2"], cycles=3)]
    assert a == b
    ordered = [j.arm for j in TP.run_plan(["c1"], cycles=3)][:len(TP.ARMS)]
    assert ordered != sorted(TP.ARMS) or len(TP.ARMS) < 3


def test_the_arms_are_interleaved_not_blocked_by_arm():
    """All of one arm first would confound the arm with server-side drift."""
    plan = TP.run_plan(["c1", "c2", "c3"], cycles=3)
    first = [j.arm for j in plan[:len(TP.ARMS)]]
    assert len(set(first)) > 1


# ---------------------------------------------------------------- envelope

def test_the_noise_envelope_uses_every_arm_not_only_the_control():
    """One control-vs-control repeat is a single draw. The envelope is the spread of
    each arm across its own cycles (Codex db4d8ac8)."""
    per_cycle = {"proc10": [0.50, 0.52, 0.51], "t085": [0.55, 0.49, 0.58]}
    env = TP.noise_envelope(per_cycle)
    assert env == pytest.approx(0.09, abs=1e-9)      # widest within-arm spread


def test_a_gain_inside_the_envelope_does_not_advance():
    v = TP.verdict(gain=0.02, envelope=0.09, meetings_agreeing=6, n_meetings=6,
                   non_overlap_loss=0.0)
    assert v["advance"] is False and "envelope" in v["why"]


def test_a_gain_that_costs_the_rest_of_the_cell_does_not_advance():
    v = TP.verdict(gain=0.15, envelope=0.03, meetings_agreeing=5, n_meetings=6,
                   non_overlap_loss=0.02)
    assert v["advance"] is False and "non-overlap" in v["why"]


def test_a_gain_carried_by_two_meetings_does_not_advance():
    v = TP.verdict(gain=0.15, envelope=0.03, meetings_agreeing=2, n_meetings=6,
                   non_overlap_loss=0.0)
    assert v["advance"] is False and "meetings" in v["why"]


def test_all_three_conditions_together_advance():
    v = TP.verdict(gain=0.15, envelope=0.03, meetings_agreeing=5, n_meetings=6,
                   non_overlap_loss=0.005)
    assert v["advance"] is True


# ---------------------------------------------------------------- hygiene

def test_no_transcript_text_reaches_the_repository_summary():
    """Same rule as every other artifact here: council speech stays in the cache."""
    summary = TP.repo_summary({
        "t085": {"recall": 0.61, "n_ref": 287, "matched": 175,
                 "unmatched_hyp": ["κάτι", "άλλο"]},
    }, envelope=0.03, cells=15, meetings=6)
    blob = str(summary)
    assert "κάτι" not in blob and "άλλο" not in blob
    assert summary["arms"]["t085"]["n_unmatched_hyp"] == 2


# ---------------------------------------------------- words, not sub-word pieces

def test_sub_word_pieces_are_glued_back_into_words():
    """Soniox streams PIECES: "Κ", "υ", "ρ", "ία", " Αγ". The first version of this
    module scored those pieces as words, so gold WORDS were matched against ASR PIECES
    and every number was meaningless. Harold caught it from the adjudication page,
    which was showing things that are not words."""
    toks = [
        {"text": "Κ", "start_ms": 840, "end_ms": 900, "is_final": True, "confidence": 0.99},
        {"text": "υ", "start_ms": 960, "end_ms": 1020, "is_final": True, "confidence": 0.89},
        {"text": "ρ", "start_ms": 1020, "end_ms": 1080, "is_final": True, "confidence": 0.99},
        {"text": "ία", "start_ms": 1080, "end_ms": 1140, "is_final": True, "confidence": 0.99},
        {"text": " Αγ", "start_ms": 1200, "end_ms": 1260, "is_final": True, "confidence": 0.49},
        {"text": "α", "start_ms": 1260, "end_ms": 1320, "is_final": True, "confidence": 0.99},
        {"text": "πη", "start_ms": 1380, "end_ms": 1440, "is_final": True, "confidence": 0.99},
        {"text": "τού", "start_ms": 1500, "end_ms": 1560, "is_final": True, "confidence": 0.98},
    ]
    got = TP.words_from_tokens(toks, 1.0)
    assert [w.text for w in got] == ["Κυρία", "Αγαπητού"]
    assert (got[0].start_ms, got[0].end_ms) == (840, 1140)
    assert (got[1].start_ms, got[1].end_ms) == (1200, 1560)


def test_a_words_span_is_mapped_back_from_stretched_time():
    toks = [{"text": "να", "start_ms": 14286, "end_ms": 15714, "is_final": True,
             "confidence": 0.9}]
    w = TP.words_from_tokens(toks, 0.7)[0]
    assert w.start_ms == pytest.approx(10000, abs=2)
    assert w.end_ms == pytest.approx(11000, abs=2)


def test_non_final_pieces_are_dropped():
    toks = [
        {"text": "να", "start_ms": 100, "end_ms": 200, "is_final": True,
         "confidence": 0.9},
        {"text": " πω", "start_ms": 300, "end_ms": 400, "is_final": False,
         "confidence": 0.9},
    ]
    assert [w.text for w in TP.words_from_tokens(toks, 1.0)] == ["να"]


def test_a_word_group_words_would_drop_is_refused_not_scored_around():
    """`group_words` skips a word with no confidence or no start time, which is right
    for a confidence study and wrong here: a silently missing word becomes a deletion
    charged to whichever arm produced it."""
    toks = [{"text": "να", "end_ms": 200, "is_final": True, "confidence": 0.9}]
    with pytest.raises(ValueError, match="dropped"):
        TP.words_from_tokens(toks, 1.0)
