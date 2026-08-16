"""Tests 3 and 4 of `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md` §5.

Test 3 is the one that matters: the first draft of the rule mathematically could not
fire on the case it exists for, and only an oracle test on a constructed interval
catches that. Test 4 pins the event scoring, where fragmentation could otherwise turn
one omission into three true positives.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import density_omission as D    # noqa: E402


def seg(spk, a, b):
    return {"speaker": spk, "start": a, "end": b}


RHO = 2.0          # tokens per second per speaker, fixed for these tests


# --------------------------------------------------------- test 3: rule semantics
def test_single_speaker_regime_is_exactly_the_old_zero_word_rule():
    reg = [seg("A", 0.0, 4.0)]
    assert D.raw_flags(reg, [], RHO) == [(0.0, 4.0)]
    assert D.raw_flags(reg, [1.0], RHO) == []          # one word blocks it, as before
    assert D.old_rule_flags(reg, []) == [(0.0, 4.0)]
    assert D.old_rule_flags(reg, [1.0]) == []


def test_single_speaker_eligibility_is_wall_clock_and_half_open():
    assert D.eligible([seg("A", 0.0, 1.5)], RHO) == [(0.0, 1.5, 1)]
    assert D.eligible([seg("A", 0.0, 1.49)], RHO) == []


def test_two_speakers_one_lost_flags_exactly_at_the_boundary():
    """The case the whole rule exists for: {A,B} for 4 s, A transcribed at rho, B gone.

    obs = rho * dur = 8, missing = 2 - 8/(2*4) = 1.0, and the threshold is inclusive.
    A strict '< 0.5 * rho' density rule scores exactly 0.5 * rho here and does NOT
    fire, which is why that design was rejected before it ran.
    """
    reg = [seg("A", 0.0, 4.0), seg("B", 0.0, 4.0)]
    times = [0.1 * i for i in range(1, 9)]             # 8 tokens inside [0,4)
    assert D.missing_speakers(2, 8, 4.0, RHO) == 1.0
    assert D.raw_flags(reg, times, RHO, threshold=1.0) == [(0.0, 4.0)]
    assert D.raw_flags(reg, times, RHO, threshold=1.25) == []
    # one extra recognised word and it no longer looks like a whole speaker is gone
    assert D.raw_flags(reg, times + [0.95], RHO, threshold=1.0) == []


def test_three_speakers_with_two_transcribed_does_not_flag():
    reg = [seg("A", 0.0, 3.0), seg("B", 0.0, 3.0), seg("C", 0.0, 3.0)]
    times = [0.1 * i for i in range(1, 13)]            # 12 tokens = 2 speakers' worth
    assert D.missing_speakers(3, 12, 3.0, RHO) == 1.0
    assert D.raw_flags(reg, times, RHO, threshold=1.25) == []
    # and with only one speaker's worth it is two speakers short
    assert D.raw_flags(reg, times[:6], RHO, threshold=1.25) == [(0.0, 3.0)]


def test_overlap_eligibility_is_tied_to_min_del_run():
    """rho_single * dur >= 3 - a lost speaker too short to produce a 3-word deletion
    run cannot produce a scorable truth event either."""
    short = [seg("A", 0.0, 1.4), seg("B", 0.0, 1.4)]   # rho*dur = 2.8 < 3
    assert D.eligible(short, RHO) == []
    ok = [seg("A", 0.0, 1.5), seg("B", 0.0, 1.5)]      # rho*dur = 3.0
    assert D.eligible(ok, RHO) == [(0.0, 1.5, 2)]


def test_duration_only_comparator_cannot_see_the_speaker_count():
    reg = [seg("A", 0.0, 4.0), seg("B", 0.0, 4.0)]
    times = [0.1 * i for i in range(1, 9)]
    # speaker-aware: one whole speaker missing. duration-only: output is exactly the
    # single-speaker rate, so nothing is missing at all.
    assert D.raw_flags(reg, times, RHO, 1.0) == [(0.0, 4.0)]
    assert D.raw_flags(reg, times, RHO, 1.0, force_single=True) == []


# ------------------------------------------------------ test 4: event scoring
def test_adjacent_flags_merge_before_matching():
    spans = [(0.0, 2.0), (2.0, 4.0), (6.0, 7.0)]
    assert D.merge(spans) == [(0.0, 4.0), (6.0, 7.0)]


def test_matching_is_one_to_one_and_counts_are_exact():
    """Two adjacent flags cover ONE deletion run; a third flag matches nothing."""
    flags = D.merge([(0.0, 2.0), (2.0, 4.0), (10.0, 11.0)])
    truths = [(1.0, 3.0), (20.0, 21.0)]
    tp, mf, mt = D.match(flags, truths)
    assert (tp, mf, mt) == (1, [0], [0])
    assert len(flags) == 2                       # merging happened first
    assert tp / len(flags) == 0.5                # precision
    assert tp / len(truths) == 0.5               # recall


def test_one_truth_event_cannot_be_claimed_twice():
    flags = [(0.0, 1.0), (0.5, 1.5)]             # deliberately unmerged
    truths = [(0.4, 0.6)]
    tp, mf, _ = D.match(flags, truths)
    assert tp == 1 and mf == [0]


def test_truth_events_need_three_consecutive_deletions():
    ops = ["M", "D", "D", "M", "D", "D", "D", "M"]
    times = [float(i) for i in range(len(ops))]
    assert D.truth_events(ops, times) == [(4.0, 6.0)]


def test_calibrate_budget_prefers_fewer_flags_on_a_tie():
    reg = [seg("A", 0.0, 4.0), seg("B", 0.0, 4.0)]
    times = [0.1 * i for i in range(1, 5)]
    per_window = [(D.observed(reg, times, RHO), RHO)]
    thr = D.calibrate_budget(per_window, target=1)
    got = D.merge(D.raw_flags(reg, [0.1 * i for i in range(1, 5)], RHO, thr,
                              force_single=True))
    assert len(got) == 1
