"""Implementation tests required by the prereg before any number is believed.

`docs/specs/2026-08-16-adapter-confidence-prereg.md`, section "Implementation tests".
Codex asked for exactly these three (job 06a6ad95a3d14af4bd53ba76308a6b66): they catch
confidence attached to the wrong token, an inverted score direction, deletions
accidentally counted, and mishandled duplicated or tied scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.adapter_confidence_probe import (  # noqa: E402
    _units_from_words, auroc, bottom_decile_by_rank, midranks, per_meeting_auroc,
    spearman,
)


def _w(raw, start, p, seg=0, i=0):
    return {"raw": raw, "start": start, "end": start + 0.1, "p_word": p,
            "seg_conf": 0.7, "seg_idx": seg, "word_id": ("c", seg, i)}


# ------------------------------------------------- 1. alignment / accounting fixture
def _fixture():
    """ref has 6 tokens, the hypothesis 5, so at least one reference token is DELETED.
    hyp words: 'alpha', 'beta-two' (-> two tokens), ',' (punctuation-only, dropped),
               'delta', 'extra'. What the tests pin is the accounting, not which
               particular op the aligner assigns."""
    ref = ["alpha", "beta", "gamma", "omega", "psi", "delta"]
    words = [_w("alpha", 0.0, 0.90, 0, 0), _w("beta-two", 1.0, 0.20, 0, 1),
             _w(",", 2.0, 0.05, 0, 2), _w("delta", 3.0, 0.95, 1, 3),
             _w("extra", 4.0, 0.10, 1, 4)]
    return ref, words


def test_every_token_gets_exactly_one_label_and_deletions_make_no_row():
    ref, words = _fixture()
    units, counts, dropped, split = _units_from_words(
        words, ref, ("S", "D", "I"), False, "c", "m", [])
    assert dropped == 1, "the punctuation-only word must be dropped, not scored"
    assert split == 1, "'beta-two' must be recorded as splitting into >1 token"
    # one row per normalized hypothesis token that received an M/S/I label
    assert len(units) == sum(counts[k] for k in ("M", "S", "I"))
    assert all(u["op"] in ("M", "S", "I") for u in units)
    # deletions exist in the alignment but never become rows
    assert counts["D"] >= 1
    assert all(u["err"] == int(u["op"] != "M") for u in units)
    # M is the negative class; S and I are positive
    assert {u["op"] for u in units if u["err"] == 0} == {"M"}


def test_multi_token_word_duplicates_its_probability():
    ref, words = _fixture()
    units, _c, _d, _s = _units_from_words(words, ref, ("S", "D", "I"), False,
                                          "c", "m", [])
    beta = [u for u in units if u["word_id"] == ("c", 0, 1)]
    assert len(beta) == 2, "'beta-two' must contribute two rows"
    assert {u["p_word"] for u in beta} == {0.20}, "both rows inherit the word's p"
    assert {u["tok"] for u in beta} == {"beta", "two"}


def test_reference_index_is_unique_per_system():
    ref, words = _fixture()
    units, _c, _d, _s = _units_from_words(words, ref, ("S", "D", "I"), False,
                                          "c", "m", [])
    idx = [u["ref_i"] for u in units if u["ref_i"] is not None]
    assert len(idx) == len(set(idx)), "a reference index may be claimed at most once"
    assert all(u["ref_i"] is None for u in units if u["op"] == "I"), \
        "insertions have no reference index"


def test_reconciliation_token_sequence_matches_alignment_input():
    """The sequence rebuilt from the emitted words must equal what align_ops saw."""
    from eval.controlled_eval.scoring import wtoks
    ref, words = _fixture()
    units, _c, _d, _s = _units_from_words(words, ref, ("S", "D", "I"), False,
                                          "c", "m", [])
    rebuilt = [t for w in words for t in wtoks(w["raw"])]
    assert [u["tok"] for u in units] == rebuilt
    assert all(0.0 <= u["p_word"] <= 1.0 for u in units)


def test_segment_proxy_is_constant_within_a_segment():
    ref = ["a", "b"]
    words = [_w("a", 0.0, 0.9, 0, 0), _w("b", 1.0, 0.1, 0, 1)]
    words[1]["seg_conf"] = words[0]["seg_conf"]
    units, _c, _d, _s = _units_from_words(words, ref, ("S", "D", "I"), False,
                                          "c", "m", [])
    per_seg = {}
    for u in units:
        per_seg.setdefault(u["seg_idx"], set()).add(u["seg_conf"])
    assert all(len(v) == 1 for v in per_seg.values())


# --------------------------------------------------------- 2. known-AUROC fixtures
def test_auroc_direction_and_ties():
    # score is 1 - p, so a LOW probability on every error is a perfect detector
    perfect_p = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    assert auroc([1 - p for p in perfect_p], labels) == 1.0
    assert auroc([p for p in perfect_p], labels) == 0.0        # inverted
    assert auroc([0.5] * 4, labels) == 0.5                     # all tied


def test_macro_within_meeting_path():
    units = []
    for m, ps, ys in (("m1", [0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]),
                      ("m2", [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])):
        for p, y in zip(ps, ys):
            units.append({"p_word": p, "err": y, "meeting": m})
    r = per_meeting_auroc(units, "p_word")
    assert r["per_meeting"]["m1"]["auroc"] == 1.0
    assert r["per_meeting"]["m2"]["auroc"] == 0.0
    assert r["mean_within_meeting_auroc"] == 0.5


def test_macro_skips_meetings_with_one_class():
    units = [{"p_word": 0.1, "err": 1, "meeting": "m1"},
             {"p_word": 0.9, "err": 0, "meeting": "m1"},
             {"p_word": 0.5, "err": 0, "meeting": "m2"}]
    r = per_meeting_auroc(units, "p_word")
    assert r["per_meeting"]["m2"]["auroc"] is None
    assert r["n_meetings_informative"] == 1
    assert r["mean_within_meeting_auroc"] == 1.0


# ---------------------------------------------------------- ranks and combination
def test_midranks_scale_ties_and_degenerate_groups():
    assert midranks([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert midranks([5.0, 5.0, 5.0]) == [0.5, 0.5, 0.5]   # all tied -> 0.5
    assert midranks([7.0]) == [0.5]                       # singleton -> 0.5
    r = midranks([1.0, 2.0, 2.0, 3.0])
    assert r[1] == r[2], "tied values must share a midrank"
    assert r[0] == 0.0 and r[3] == 1.0


def test_mean_rank_combination_is_symmetric():
    a, b = midranks([0.1, 0.5, 0.9]), midranks([0.9, 0.5, 0.1])
    comb = [(x + y) / 2 for x, y in zip(a, b)]
    assert comb == [0.5, 0.5, 0.5], "opposed rankings must cancel"


def test_spearman_extremes():
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == -1.0
    assert spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None   # degenerate


def test_bottom_decile_keeps_ties_and_reports_realized_rate():
    rows = [{"r": 0.0}] * 5 + [{"r": float(i) / 10} for i in range(1, 16)]
    out = bottom_decile_by_rank(rows, "r")
    assert out["n_flagged"] >= 5, "every observation tied at the cutoff is included"
    assert out["realized_flag_rate"] == out["n_flagged"] / len(rows)
    assert out["realized_flag_rate"] >= 0.10


# ------------------------------ tokenizer symmetry on the 247-window descriptive arm
def test_247_arm_tokenizes_hypothesis_with_the_same_normalizer_as_its_reference():
    """The benchmark reference uses the filler-stripped tokenizer. If the hypothesis
    were tokenized with plain `wtoks`, every emitted hesitation filler would become a
    spurious insertion."""
    from eval.adapter_confidence_probe import word_units_ftoks
    from eval.controlled_eval.eval_freeze import ftoks
    from eval.controlled_eval.scoring import wtoks

    filler = next((t for t in ("εεε", "εε", "μμμ", "εεεε")
                   if wtoks(t) and not ftoks(t)), None)
    assert filler, "no hesitation filler survives wtoks but not ftoks"
    words = [_w("καλά", 0.0, 0.9, 0, 0), _w(filler, 1.0, 0.3, 0, 1)]
    units, dropped, _split = word_units_ftoks(words)
    assert dropped == 1, "the filler word must drop out, as it does from the reference"
    assert [u["tok"] for u in units] == ftoks("καλά")


# ----------------------- the joined-string fusion hazard the gate contrast walked into
def test_joining_segment_texts_without_a_separator_fuses_words():
    """`"".join(segment.text)` is what the decode script stores, and faster-whisper does
    not always put a leading space on a segment. Scoring the joined string charges a
    `word_timestamps` contrast for fusions created by the join, at exactly the segment
    boundaries the setting moves. Tokens must be built per segment."""
    from eval.controlled_eval.scoring import wtoks

    segs = [{"text": " το συμβούλιο"}, {"text": "δεν εισηγήθηκε"}]
    joined = wtoks("".join(s["text"] for s in segs))
    per_segment = [t for s in segs for t in wtoks(s["text"])]
    assert per_segment == ["το", "συμβουλιο", "δεν", "εισηγηθηκε"]
    assert joined != per_segment, "the join must be shown to lose a word boundary"
    assert len(joined) == len(per_segment) - 1
