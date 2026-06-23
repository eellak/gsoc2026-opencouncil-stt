"""TDD acceptance test for the Design-A matrix harness (matrix.py).

Proves the SCORING MECHANICS on a tiny frozen fixture before any multi-hour CPU
run: micro-averaged WER, meeting-clustered bootstrap, per-category n_clips /
n_meetings / directional flags, strict (fail-loud) LLM-output extraction, and a
report that surfaces the low-n flags. No model, no clips, no LLM calls — all four
stages are supplied as in-memory hypotheses.

Run: ../../.venv-asr/bin/python -m pytest eval/autoresearch/test_matrix.py -q
 (or) ../../.venv-asr/bin/python eval/autoresearch/test_matrix.py
"""
from __future__ import annotations

import matrix as M


# ---- tiny frozen fixture: 2 meetings, 2 categories, 4 corrections ----
# latin tokens so normalize_el (lowercase, strip-diacritics, strip-punct) is a no-op
ROWS = [
    {"utterance_id": "u1", "city": "argos", "meeting": "m1",
     "text": "a b c", "error_categories": ["substitution_phonetic"]},
    {"utterance_id": "u2", "city": "argos", "meeting": "m1",
     "text": "d e", "error_categories": ["punctuation_capitalization"]},
    {"utterance_id": "u3", "city": "orestiada", "meeting": "m2",
     "text": "f g h i", "error_categories": ["substitution_phonetic"]},
    {"utterance_id": "u4", "city": "orestiada", "meeting": "m2",
     "text": "k", "error_categories": ["substitution_phonetic"]},
]
HYP_A = {"u1": "a x c", "u2": "d e", "u3": "f g h", "u4": "k"}      # 2 word-edits / 10 ref-words
HYP_B = {"u1": "a b c", "u2": "d e", "u3": "f g h", "u4": "k"}      # LLM fixes u1
HYP_C = {"u1": "a b c", "u2": "d e", "u3": "f g h i", "u4": "k"}    # FT fixes u1 & u3 -> 0 edits
HYP_D = dict(HYP_C)


def approx(x, y, tol=1e-9):
    return abs(x - y) < tol


def test_edit_distance():
    assert M.edit_distance(list("abc"), list("abc")) == 0
    assert M.edit_distance("a b c".split(), "a x c".split()) == 1   # 1 sub
    assert M.edit_distance("f g h i".split(), "f g h".split()) == 1  # 1 del
    assert M.edit_distance([], "a b".split()) == 2                   # 2 ins


def test_micro_average_aggregate():
    recs, dropped = M.build_records(ROWS, HYP_A, HYP_B, HYP_C, HYP_D)
    assert dropped == []
    # stage A word_norm = (1+0+1+0)/(3+2+4+1) = 2/10
    assert approx(M.micro(recs, "A", "word_norm"), 0.2)
    # stage C fixed u1 & u3 -> 0 edits
    assert approx(M.micro(recs, "C", "word_norm"), 0.0)
    # micro-average is NOT the mean of per-clip WERs (that would be (1/3+0+1/4+0)/4)
    mean_of_wers = (1/3 + 0 + 1/4 + 0) / 4
    assert not approx(M.micro(recs, "A", "word_norm"), mean_of_wers)


def test_per_category_n_flags():
    recs, _ = M.build_records(ROWS, HYP_A, HYP_B, HYP_C, HYP_D)
    pc = M.per_category(recs)
    phon = pc["substitution_phonetic"]
    assert phon["n_clips"] == 3 and phon["n_meetings"] == 2
    assert phon["directional"] is True            # n_clips < 30
    # phonetic stage A: edits (1+1+0)/(3+4+1) = 2/8 = 0.25
    assert approx(phon["A"]["word_norm"], 0.25)
    assert approx(phon["C"]["word_norm"], 0.0)
    punct = pc["punctuation_capitalization"]
    assert punct["n_clips"] == 1 and punct["n_meetings"] == 1
    assert punct["directional"] is True


def test_bootstrap_clusters_by_meeting_and_is_deterministic():
    recs, _ = M.build_records(ROWS, HYP_A, HYP_B, HYP_C, HYP_D)
    lo1, hi1 = M.bootstrap_ci(recs, "A", "word_norm", n_boot=200, seed=0)
    lo2, hi2 = M.bootstrap_ci(recs, "A", "word_norm", n_boot=200, seed=0)
    assert (lo1, hi1) == (lo2, hi2)               # deterministic with fixed seed
    assert lo1 <= 0.2 <= hi1
    # a single-meeting subset (punct = only m1) has no between-meeting variance
    punct_recs = [r for r in recs if "punctuation_capitalization" in r["cats"]]
    lo, hi = M.bootstrap_ci(punct_recs, "A", "word_norm", n_boot=200, seed=0)
    assert approx(lo, 0.0) and approx(hi, 0.0)


def test_paired_delta_resamples_same_meetings():
    recs, _ = M.build_records(ROWS, HYP_A, HYP_B, HYP_C, HYP_D)
    mean, lo, hi = M.paired_delta_ci(recs, "A", "C", "word_norm", n_boot=200, seed=0)
    # C is better than A -> A - C >= 0 on the point estimate
    assert approx(mean, 0.2)                       # 0.2 - 0.0
    assert lo <= mean <= hi


def test_extract_fix_is_strict_fail_loud():
    assert M.extract_fix("1. hello world") == "hello world"
    assert M.extract_fix("hello world") is None    # no numbering -> unparseable
    assert M.extract_fix("1. a\n2. b") is None      # two lines -> ambiguous
    assert M.extract_fix("") is None


def test_llm_stage_records_failures_and_keeps_alignment():
    # fix_fn that returns a clean line for u1, garbage for u3
    def fix_fn(city, hyp):
        return "1. fixed" if hyp == HYP_A["u3"] is False else (
            "1. " + hyp if hyp != HYP_A["u3"] else "garbage no number")
    hyps, failures = M.run_llm_stage(ROWS, HYP_A, fix_fn)
    assert "u3" in failures                         # unparseable -> recorded, not silent
    assert "u3" not in hyps
    assert hyps["u1"] == HYP_A["u1"]                # parsed back the echoed line


def test_dropped_uids_excluded_consistently():
    # if a uid is missing from any stage it must be dropped from ALL stages
    bad_B = {k: v for k, v in HYP_B.items() if k != "u3"}
    recs, dropped = M.build_records(ROWS, HYP_A, bad_B, HYP_C, HYP_D)
    assert dropped == ["u3"]
    assert all(r["uid"] != "u3" for r in recs)


def test_report_contains_n_and_flags():
    recs, _ = M.build_records(ROWS, HYP_A, HYP_B, HYP_C, HYP_D)
    md = M.render_report(recs, n_boot=200, seed=0, meta={"seeds": [0]})
    assert "n_clips" in md and "n_meetings" in md
    assert "directional" in md
    assert "substitution_phonetic" in md
    # paired-delta reads the experiment cares about
    for tag in ("C-A", "B-A", "D-C", "D-B"):
        assert tag in md


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
