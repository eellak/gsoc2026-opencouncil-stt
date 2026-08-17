"""Locks the frozen definitions of `exp-2026-08-17-insertion-fidelity`.

These are freeze tests, not behaviour tests. The three things a later change could
break silently, and each of which the Codex design review (job 847c449f) called
fatal if got wrong:

  * support is INJECTIVE and blind to alignment A, so one gold occurrence cannot
    support two system tokens and a token's support cannot depend on whether the
    metric happened to call it an insertion;
  * undecidability is ASYMMETRIC — proximity to a block never licenses
    "not supported", only wholly-inside-certain-coverage does;
  * "overlap" means the actual intersection of two gold blocks, not adjacency to
    a block that participates in one somewhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.insertion_fidelity import (  # noqa: E402
    SUPPORTED, UNDEC, UNSUPPORTED, classify, contains, corroborate, intersect,
    local_support_matching, meets, merge, pairwise_overlap_regions,
)


def blk(bid, s, e, spk="A", text="", unc=False, ov=None):
    return {"id": bid, "s": s, "e": e, "spk": spk, "text": text,
            "text_unc": unc, "spk_unc": False, "ov_with": ov or []}


def gold(tok, s, e, bid="b0", ov=False):
    return {"tok": tok, "block": bid, "s": s, "e": e, "ov": ov}


def unit(tok, s, e=None):
    return {"tok": tok, "start": s, "end": s if e is None else e}


# ------------------------------------------------------------------ interval math
def test_merge_and_contains_and_meets():
    assert merge([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert contains([(0, 2), (3, 4)], 0.5, 1.5)
    assert not contains([(0, 2), (3, 4)], 1.5, 3.5)   # the gap is not covered
    assert not contains([(0, 2)], 1.5, 2.5)
    assert meets([(0, 2)], 1.9, 5.0)
    assert not meets([(0, 2)], 2.0, 3.0)
    assert intersect([(0, 5)], [(1, 2), (4, 9)]) == [(1, 2), (4, 5)]


def test_overlap_is_the_intersection_not_the_whole_block():
    # p0 runs 0-10 and overlaps p1 only during 9-10. A word at 2 s is inside a
    # block that participates in an overlap, and is NOT simultaneous speech.
    bs = [blk("p0", 0, 10, ov=["p1"]), blk("p1", 9, 12, ov=["p0"])]
    assert pairwise_overlap_regions(bs) == [(9, 10)]
    assert not meets(pairwise_overlap_regions(bs), 2.0, 2.4)
    assert meets(pairwise_overlap_regions(bs), 9.5, 9.7)


def test_overlap_partner_outside_the_region_is_ignored():
    # `ov_with` can point at a block the region dropped; no interval, no overlap.
    assert pairwise_overlap_regions([blk("p0", 0, 10, ov=["gone"])]) == []


# ------------------------------------------------------------------ the matching
def test_matching_is_injective_over_gold_occurrences():
    # the system says "ναι" twice, the human said it once -> exactly one support
    g = [gold("ναι", 1.0, 2.0)]
    u = [unit("ναι", 1.2), unit("ναι", 1.4)]
    m = local_support_matching(u, g, tau=0.5)
    assert len(m) == 1 and set(m.values()) == {0}


def test_matching_is_injective_over_system_tokens():
    g = [gold("ναι", 1.0, 2.0, "b0"), gold("ναι", 1.0, 2.0, "b0")]
    m = local_support_matching([unit("ναι", 1.5)], g, tau=0.5)
    assert len(m) == 1


def test_matching_needs_temporal_admissibility():
    g = [gold("ναι", 10.0, 11.0)]
    assert local_support_matching([unit("ναι", 1.0)], g, tau=0.5) == {}
    # ... and tau is what buys the boundary case
    assert local_support_matching([unit("ναι", 9.6)], g, tau=0.5) == {0: 0}
    assert local_support_matching([unit("ναι", 9.6)], g, tau=0.25) == {}


def test_matching_prefers_the_temporally_closer_gold_occurrence():
    g = [gold("ναι", 0.0, 1.0, "b0"), gold("ναι", 5.0, 6.0, "b1")]
    assert local_support_matching([unit("ναι", 5.4)], g, tau=1.0) == {0: 1}


def test_all_system_tokens_compete_not_only_insertions():
    """The matching never sees alignment A. A token that PUB also has still
    consumes its gold occurrence, so the duplicate cannot claim it."""
    g = [gold("ναι", 1.0, 2.0)]
    u = [unit("ναι", 1.1), unit("ναι", 1.9)]
    m = local_support_matching(u, g, tau=0.5)
    assert list(m) == [0]           # the closer token wins, whatever A said


# ---------------------------------------------------------------- classification
def _cell(gold_list, blocks, pseq, units, cov, ov=()):
    return {"cell": "c", "meeting": "m", "city": "x", "stream": "R",
            "gold": gold_list, "pseq": pseq, "units": {"S": units}, "blocks": blocks,
            "certain_cov": list(cov), "ov_regions": list(ov), "snx_stats": {}}


def test_unsupported_needs_the_whole_window_inside_certain_coverage():
    # A word at 5.0 s sits 3 s outside the only annotated block. Under the
    # rejected symmetric rule it would have been "not said"; here it is
    # UNDECIDABLE, because the human never claimed anything about that time.
    g = [gold("ναι", 1.0, 2.0)]
    cell = _cell(g, [blk("b0", 1.0, 2.0)], ["ναι"],
                 [unit("ναι", 1.2), unit("οχι", 5.0)], [(1.0, 2.0)])
    rows = classify(cell, "S", tau=0.5)
    assert [r["cls"] for r in rows] == [UNDEC]


def test_unsupported_when_the_window_is_wholly_annotated():
    g = [gold("ναι", 0.0, 10.0)]
    cell = _cell(g, [blk("b0", 0.0, 10.0)], ["ναι"],
                 [unit("ναι", 1.0), unit("οχι", 5.0)], [(0.0, 10.0)])
    rows = classify(cell, "S", tau=0.5)
    assert [r["cls"] for r in rows] == [UNSUPPORTED]


def test_repeated_word_pub_has_once_is_possible_but_not_forced():
    """The human said "ναι" in two separate blocks and PUB has it once.

    PUB could be reading either occurrence, so NEITHER is forced-unmatched. This
    is the whole reason the verdict is a band: a single backtrace would have
    picked one and called it a reference omission.
    """
    g = [gold("ναι", 0.0, 2.0, "b0"), gold("ναι", 5.0, 7.0, "b1")]
    cell = _cell(g, [blk("b0", 0.0, 2.0), blk("b1", 5.0, 7.0)], ["ναι"],
                 [unit("ναι", 1.0), unit("ναι", 6.0)], [(0.0, 2.0), (5.0, 7.0)])
    rows = classify(cell, "S", tau=0.5)
    assert len(rows) == 1
    assert rows[0]["cls"] == SUPPORTED
    assert rows[0]["pub_unmatched_forced"] is False
    assert rows[0]["pub_unmatched_possible"] is True


def test_word_pub_does_not_have_at_all_is_forced_unmatched():
    g = [gold("ναι", 0.0, 2.0, "b0"), gold("οχι", 5.0, 7.0, "b1")]
    cell = _cell(g, [blk("b0", 0.0, 2.0), blk("b1", 5.0, 7.0)], ["ναι"],
                 [unit("ναι", 1.0), unit("οχι", 6.0)], [(0.0, 2.0), (5.0, 7.0)])
    rows = classify(cell, "S", tau=0.5)
    assert len(rows) == 1
    assert rows[0]["cls"] == SUPPORTED and rows[0]["pub_unmatched_forced"] is True


def test_duplicate_of_a_word_pub_already_has_is_not_an_omission():
    """gold said it once, PUB has it, the system says it twice.

    Which of the two copies alignment A calls the insertion, and which of them
    the matcher gives the single gold occurrence to, are both arbitrary here. The
    classification is arbitrary-proof: whichever way it falls, the extra copy is
    never counted as PUB-unmatched, because the one gold occurrence behind it IS
    matched by PUB in every optimal alignment.
    """
    g = [gold("ναι", 0.0, 4.0)]
    cell = _cell(g, [blk("b0", 0.0, 4.0)], ["ναι"],
                 [unit("ναι", 1.0), unit("ναι", 2.0)], [(0.0, 4.0)])
    rows = classify(cell, "S", tau=0.5)
    assert len(rows) == 1
    assert rows[0]["pub_unmatched_forced"] is not True
    assert rows[0]["pub_unmatched_possible"] is not True
    assert rows[0]["cls"] in (SUPPORTED, UNSUPPORTED)


def test_pub_match_band_separates_forced_from_possible():
    """A repeated word PUB has only once: neither occurrence is FORCED unmatched,
    because an optimal alignment exists that matches either one."""
    from eval.insertion_fidelity import pub_match_band
    every, some = pub_match_band(["ναι", "ναι"], ["ναι"])
    assert some == {0, 1} and every == set()
    # ... and a word PUB does not have at all is forced-unmatched
    every, some = pub_match_band(["ναι", "οχι"], ["ναι"])
    assert 1 not in some and 0 in every


def test_matching_recovers_the_pair_a_greedy_pass_would_lose():
    """token 1 can take gold A or B, token 2 can take only A. A greedy pass gives
    A to token 1 and strands token 2; maximum cardinality does not."""
    g = [gold("ναι", 0.0, 2.0, "b0"), gold("ναι", 4.0, 6.0, "b1")]
    u = [unit("ναι", 1.0), unit("ναι", 1.1)]
    assert len(local_support_matching(u, g, tau=5.0)) == 2


def test_uncertain_block_time_is_never_scored_as_unsupported():
    # the human heard speech at 5-6 s and could not transcribe it: that time is
    # not certain coverage, so a word there is undecidable, not an error.
    g = [gold("ναι", 0.0, 4.0)]
    cell = _cell(g, [blk("b0", 0.0, 4.0), blk("b1", 5.0, 6.0, unc=True)],
                 ["ναι"], [unit("ναι", 1.0), unit("τι", 5.5)], [(0.0, 4.0)])
    assert [r["cls"] for r in classify(cell, "S", tau=0.5)] == [UNDEC]


# -------------------------------------------------------------- corroboration
def test_corroboration_is_injective_and_nearest_in_time():
    rows = [{"tok": "ναι", "start": 1.0}, {"tok": "ναι", "start": 1.2}]
    corroborate(rows, [unit("ναι", 1.15)], 1.5, "w")
    assert [r["w"] for r in rows] == [False, True]      # one witness, one citation
