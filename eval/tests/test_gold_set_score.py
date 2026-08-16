"""Frozen behaviour of the gold-set scorer, written before any number was read.

The list follows the Codex review of the plan (job 75bfc337): normalisation,
core-boundary fixtures, uncertainty masking, cpWER speaker invariance, the
permutation-locked-once rule, overlap fixtures, and multiset conservation in the
candidate audit.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.gold_set_score import (  # noqa: E402
    align_ops, cp_wer, gold_view, in_intervals, masked_intervals, multiset_flags,
    multiset_support, region_blocks, region_span, sdi_counts, speaker_accuracy,
    wilson,
)
from eval.controlled_eval.scoring import wtoks  # noqa: E402


# ------------------------------------------------------------- normalisation
def test_norm_strips_tonos_and_lowercases():
    assert wtoks("Ωραία, ΚΑΤΑ πλειοψηφία.") == ["ωραια", "κατα", "πλειοψηφια"]


def test_norm_does_not_fold_final_sigma():
    # documented property of the frozen project normalizer
    assert wtoks("λόγος") != wtoks("λόγοσ")


def test_norm_drops_punctuation_only_tokens():
    assert wtoks("...  —  ") == []


def test_norm_nfc_and_nfd_agree():
    import unicodedata
    s = "Καρατζά"
    assert wtoks(unicodedata.normalize("NFC", s)) == wtoks(unicodedata.normalize("NFD", s))


# ------------------------------------------------------------ core boundaries
def B(i, s, e, spk="A", text="a b", **kw):
    d = {"id": i, "s": s, "e": e, "spk": spk, "text": text, "t_src": "human",
         "text_unc": False, "spk_unc": False, "ov_with": []}
    d.update(kw)
    return d


BLOCKS = [
    B("in", 11.0, 12.0),                 # wholly inside the core
    B("left", 9.0, 11.0),                # straddles core start
    B("right", 24.0, 26.0),              # straddles core end
    B("out", 1.0, 2.0),                  # wholly outside
    B("edge_start", 10.0, 11.0),         # starts exactly at core start
    B("edge_end", 24.0, 25.0),           # ends exactly at core end
]


def test_core_strict_takes_only_wholly_contained_blocks():
    got = {b["id"] for b in region_blocks(BLOCKS, "core_strict")}
    assert got == {"in", "edge_start", "edge_end"}


def test_core_envelope_takes_every_intersecting_block_whole():
    got = {b["id"] for b in region_blocks(BLOCKS, "core_envelope")}
    assert got == {"in", "left", "right", "edge_start", "edge_end"}


def test_clip_region_takes_everything():
    assert len(region_blocks(BLOCKS, "clip")) == len(BLOCKS)


def test_region_span_is_the_hull_of_the_included_blocks():
    assert region_span(region_blocks(BLOCKS, "core_envelope"), "core_envelope") == (9.0, 26.0)
    # the clip region is the hull of the annotated blocks, NOT the nominal 0-35 s:
    # the human's blocks do not tile the clip
    assert region_span(BLOCKS, "clip") == (1.0, 26.0)


def test_zero_length_block_at_core_start_is_not_in_the_envelope():
    # e > CORE[0] is strict, so a zero-length block at 10.0 does not intersect
    assert region_blocks([B("z", 10.0, 10.0)], "core_envelope") == []


# ---------------------------------------------------------- uncertainty masking
def test_uncertain_block_leaves_both_the_reference_and_the_span_intact():
    """Its text must not come back as a system insertion: the span still covers it,
    so hypothesis words spoken there are aligned against nothing only if they truly
    have no counterpart."""
    ans = {"b": {"blocks": [B("a", 11.0, 12.0, text="ενα δυο"),
                            B("u", 12.0, 13.0, text="τρια τεσσερα", text_unc=True)]}}
    seq, by_spk, ov, blocks, kept = gold_view(ans, "core_envelope")
    assert seq == ["ενα", "δυο"]
    assert {b["id"] for b in blocks} == {"a", "u"}          # span keeps the uncertain block
    assert region_span(blocks, "core_envelope") == (11.0, 13.0)


def test_uncertain_tokens_are_reported_not_silently_dropped():
    ans = {"b": {"blocks": [B("u", 11.0, 12.0, text="ενα δυο", text_unc=True)]}}
    seq, *_ = gold_view(ans, "core_envelope")
    assert seq == []
    seq2, *_ = gold_view(ans, "core_envelope", drop_uncertain=False)
    assert seq2 == ["ενα", "δυο"]


# --------------------------------------------------------------------- cpWER
def test_cpwer_is_invariant_to_renaming_hypothesis_speakers():
    ref = {"A": ["ενα", "δυο"], "B": ["τρια"]}
    h1 = {"S0": ["ενα", "δυο"], "S1": ["τρια"]}
    h2 = {"zzz": ["τρια"], "aaa": ["ενα", "δυο"]}
    assert cp_wer(ref, h1)["err"] == cp_wer(ref, h2)["err"] == 0


def test_cpwer_pads_missing_and_extra_speakers():
    ref = {"A": ["ενα"], "B": ["δυο"]}
    assert cp_wer(ref, {"S0": ["ενα"]})["err"] == 1                 # missing speaker
    assert cp_wer({"A": ["ενα"]}, {"S0": ["ενα"], "S1": ["δυο"]})["err"] == 1  # extra


def test_cpwer_handles_empty_sides():
    assert cp_wer({"A": []}, {"S0": []})["err"] == 0
    assert cp_wer({"A": ["ενα"]}, {"S0": []})["err"] == 1


def test_cpwer_punishes_a_speaker_swap():
    ref = {"A": ["ενα"], "B": ["δυο"]}
    swapped = {"S0": ["δυο"], "S1": ["ενα"]}
    assert cp_wer(ref, swapped)["err"] == 0     # cpWER is permutation-free by design
    # the swap is caught by speaker accuracy, not by cpWER — see the test below


def test_cpwer_refuses_absurd_speaker_counts():
    with pytest.raises(ValueError):
        cp_wer({str(i): ["x"] for i in range(9)}, {"h": ["x"]})


# ------------------------------------------------------- speaker attribution
def _cell(blocks):
    return {"b": {"blocks": blocks}}


def test_speaker_accuracy_uses_the_locked_mapping_not_a_fresh_one():
    """A mapping chosen on the whole core must be reused inside overlap even when a
    different mapping would score the overlap subset better."""
    blocks = [
        B("q1", 11.0, 12.0, spk="A", text="ενα δυο τρια"),
        B("q2", 12.0, 13.0, spk="B", text="τεσσερα πεντε εξι"),
        B("o1", 13.0, 14.0, spk="A", text="επτα", ov_with=["o2"]),
        B("o2", 13.0, 14.0, spk="B", text="οκτω", ov_with=["o1"]),
    ]
    pub = {"P0": ["ενα", "δυο", "τρια", "οκτω"], "P1": ["τεσσερα", "πεντε", "εξι", "επτα"]}
    good = {"A": "P0", "B": "P1"}
    r = speaker_accuracy(_cell(blocks), "core_envelope", pub, good)
    # inside overlap the locked mapping gets both words wrong
    assert r["n_certain_ov"] == 2
    assert r["correct_attr_ov"] == 0
    # a mapping re-optimised on the overlap subset would have scored 2/2 — we must
    # not be able to reach that number through this function
    flipped = {"A": "P1", "B": "P0"}
    r2 = speaker_accuracy(_cell(blocks), "core_envelope", pub, flipped)
    assert r2["correct_attr_ov"] == 2
    assert r2["correct_attr"] < r["correct_attr"]   # and it is worse on the whole core


def test_speaker_accuracy_excludes_spk_uncertain_blocks_from_its_denominator():
    blocks = [B("a", 11.0, 12.0, spk="A", text="ενα δυο"),
              B("b", 12.0, 13.0, spk="B", text="τρια", spk_unc=True)]
    r = speaker_accuracy(_cell(blocks), "core_envelope", {"P0": ["ενα", "δυο", "τρια"]},
                         {"A": "P0"})
    assert r["n_certain"] == 2


def test_speaker_recall_cannot_exceed_recognition():
    blocks = [B("a", 11.0, 12.0, spk="A", text="ενα δυο τρια")]
    r = speaker_accuracy(_cell(blocks), "core_envelope", {"P0": ["ενα"]},
                         {"A": "P0"})
    assert r["correct_attr"] <= r["correct_recog"] <= r["n_certain"]


# ------------------------------------------------------------------ overlap
def test_overlap_tokens_come_from_ov_with_not_from_time_intersection():
    """Two blocks may share time without being declared simultaneous; the human's
    ov_with is the authority, exactly as the schema decision says."""
    blocks = [B("a", 11.0, 13.0, spk="A", text="ενα"),
              B("b", 12.0, 14.0, spk="B", text="δυο")]      # touching, but no ov_with
    _, _, ov, _, _ = gold_view(_cell(blocks), "core_envelope")
    assert ov == []
    blocks[0]["ov_with"] = ["b"]
    blocks[1]["ov_with"] = ["a"]
    _, _, ov2, _, _ = gold_view(_cell(blocks), "core_envelope")
    assert sorted(ov2) == ["δυο", "ενα"]


# ------------------------------------------------------- candidate multisets
def test_multiset_support_never_reuses_a_gold_token():
    assert multiset_support(["ναι", "ναι", "ναι"], ["ναι"]) == 1
    assert multiset_support(["ναι", "ναι"], ["ναι", "ναι", "ναι"]) == 2


def test_multiset_flags_are_capacity_limited_and_positional():
    flags = multiset_flags(["ναι", "ναι"], ["ναι"])
    assert flags == [True, False]


def test_multiset_flags_with_no_witness_are_all_false():
    assert multiset_flags(["α", "β"], None) == [False, False]


def test_candidate_set_is_built_without_gold():
    """Insertions of PUB relative to ADP depend only on those two token streams."""
    adp = ["ενα", "τρια"]
    pub = ["ενα", "δυο", "τρια"]
    cand = [pub[j] for op, _, j in align_ops(adp, pub) if op == "I"]
    assert cand == ["δυο"]


# ------------------------------------------------------------------ alignment
def test_align_ops_conserves_every_token_occurrence_exactly_once():
    ref, hyp = ["α", "β", "γ", "δ"], ["α", "χ", "δ", "ε"]
    ops = align_ops(ref, hyp)
    used_r = [i for o, i, _ in ops if o in ("M", "S", "D")]
    used_h = [j for o, _, j in ops if o in ("M", "S", "I")]
    assert sorted(used_r) == list(range(len(ref)))
    assert sorted(used_h) == list(range(len(hyp)))


def test_sdi_counts_sum_to_the_edit_distance():
    c = sdi_counts(["α", "β", "γ"], ["α", "χ"])
    assert c["err"] == c["S"] + c["D"] + c["I"] == 2
    assert c["N"] == 3


def test_empty_reference_and_empty_hypothesis():
    assert sdi_counts([], [])["err"] == 0
    assert sdi_counts([], ["α"])["I"] == 1
    assert sdi_counts(["α"], [])["D"] == 1


def test_align_ops_is_deterministic_under_equal_cost_alternatives():
    a = align_ops(["α", "β"], ["β", "α"])
    b = align_ops(["α", "β"], ["β", "α"])
    assert a == b


# --------------------------------------------------------------------- misc
def test_wilson_is_undefined_on_an_empty_denominator():
    assert wilson(0, 0) == [None, None]


def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(7, 10)
    assert lo < 0.7 < hi


# --------------------------------------------------- uncertainty time masking
def test_masked_intervals_removes_the_time_of_an_uncertain_block():
    blocks = [B("a", 11.0, 12.0), B("u", 12.0, 13.0, text_unc=True), B("c", 13.0, 14.0)]
    assert masked_intervals(blocks, (11.0, 14.0)) == [(11.0, 12.0), (13.0, 14.0)]


def test_masked_intervals_merges_adjacent_and_nested_uncertain_blocks():
    blocks = [B("u1", 11.0, 13.0, text_unc=True), B("u2", 12.0, 12.5, text_unc=True)]
    assert masked_intervals(blocks, (11.0, 14.0)) == [(13.0, 14.0)]


def test_masked_intervals_is_the_whole_span_when_nothing_is_uncertain():
    assert masked_intervals([B("a", 11.0, 12.0)], (10.0, 14.0)) == [(10.0, 14.0)]


def test_masked_intervals_can_be_empty():
    assert masked_intervals([B("u", 9.0, 30.0, text_unc=True)], (10.0, 25.0)) == []


def test_in_intervals_is_half_open():
    ivs = [(10.0, 12.0)]
    assert in_intervals(10.0, ivs) and not in_intervals(12.0, ivs)


def test_blocks_outside_the_region_are_cut_from_the_scored_time():
    """core-strict keeps only the middle block, so the straddling neighbours' audio
    must leave the scored intervals with it."""
    inside = [B("in", 12.0, 13.0)]
    outside = [B("left", 9.0, 12.0), B("right", 13.0, 26.0)]
    ivs = masked_intervals(inside, (12.0, 13.0), outside)
    assert ivs == [(12.0, 13.0)]
    assert not in_intervals(11.5, ivs) and not in_intervals(13.5, ivs)


def test_hypothesis_words_over_an_uncertain_block_do_not_become_insertions():
    """The bug this masking exists to kill: a cell whose reference is mostly
    uncertain otherwise scores a WER above 1 purely from system words the human
    refused to vouch for."""
    blocks = [B("a", 11.0, 12.0, text="ενα δυο"), B("u", 12.0, 20.0, text_unc=True,
                                                    text="χ ψ ω κ λ μ ν ξ")]
    ivs = masked_intervals(blocks, region_span(blocks, "core_envelope"), )
    hyp = [(11.2, "ενα"), (11.6, "δυο")] + [(12.0 + i * 0.5, "θορυβος") for i in range(16)]
    kept = [w for t, w in hyp if in_intervals(t, ivs)]
    assert kept == ["ενα", "δυο"]
