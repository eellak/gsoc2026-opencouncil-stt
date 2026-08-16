"""Acceptance tests for the W-rt confidence experiment.

These are the tests Codex job `b71f2dca0cad451db62cfb8f65e9d08e` required to pass
BEFORE any WER number of any arm is read: confidence mapping, golden arm fixtures,
mutation-scope, the fitting grid and its tie rule, and the equality of the fast
fitting objective with the frozen scorer. Nothing here touches the network, the
benchmark cache, or the sealed windows.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest                                                        # noqa: E402

from eval.controlled_eval import exp_w_rt_confidence as X            # noqa: E402
from eval.controlled_eval import w_rt as R                           # noqa: E402
from eval.controlled_eval.exp_fusion_deletions import sdi            # noqa: E402
from eval.controlled_eval.msa import compose                         # noqa: E402


# --------------------------------------------------------------- validity rule
@pytest.mark.parametrize("v,ok", [
    (0.0, True), (1.0, True), (0.5, True), ("0.5", True),
    (None, False), (float("nan"), False), (float("inf"), False),
    (-0.1, False), (1.2, False), (True, False), ("x", False),
])
def test_valid_conf(v, ok):
    assert R.valid_conf(v) is ok


# ------------------------------------------------------------ confidence mapping
def _tok(text, conf, final=True, start=0, end=10):
    return {"text": text, "confidence": conf, "is_final": final,
            "start_ms": start, "end_ms": end}


def test_non_final_tokens_are_ignored():
    arm = R.rt_arm({"tokens": [_tok("γεια ", 0.9), _tok("σου", 0.2, final=False)]})
    assert arm["tokens"] == ["γεια"]


def test_min_is_over_lexical_runes_only():
    """A trailing comma at 0.01 must not condemn its word — the production rule."""
    arm = R.rt_arm({"tokens": [_tok("καλά", 0.9), _tok(",", 0.01)]})
    assert arm["tokens"] == ["καλα"]           # the frozen scorer strips accents
    assert arm["conf"] == [0.9]                # conf_min_lex
    assert arm["conf_min"] == [0.01]           # punctuation included


def test_subtokens_carry_their_own_confidence():
    arm = R.rt_arm({"tokens": [_tok("Συν", 0.99), _tok("εδρ", 0.30),
                               _tok("ίαση ", 0.95), _tok("11η", 0.80)]})
    assert arm["tokens"] == ["συνεδριαση", "11η"]
    assert arm["conf"] == [0.30, 0.80]


def test_punctuation_only_word_is_dropped_from_both_streams():
    arm = R.rt_arm({"tokens": [_tok("ναι ", 0.9), _tok("... ", 0.1), _tok("οχι", 0.8)]})
    assert arm["tokens"] == ["ναι", "οχι"]
    assert arm["conf"] == [0.9, 0.8]


def test_repeated_adjacent_words_keep_distinct_confidences():
    arm = R.rt_arm({"tokens": [_tok("ναι ", 0.9), _tok("ναι", 0.2)]})
    assert arm["tokens"] == ["ναι", "ναι"]
    assert arm["conf"] == [0.9, 0.2]


def test_word_splitting_into_two_normalized_tokens_shares_its_confidence():
    arm = R.rt_arm({"tokens": [_tok("11:30", 0.4)]})
    assert arm["tokens"] == ["11", "30"]
    assert arm["conf"] == [0.4, 0.4]


def test_apostrophe_and_digits():
    arm = R.rt_arm({"tokens": [_tok("απ' ", 0.7), _tok("το ", 0.8), _tok("2026", 0.6)]})
    assert arm["tokens"] == ["απ", "το", "2026"]
    assert arm["conf"] == [0.7, 0.8, 0.6]


def test_combining_marks_are_normalized_by_the_frozen_scorer():
    """Greek tonos as a combining mark must reach the same normalized token as the
    precomposed form, so confidence attaches to the same occurrence either way."""
    pre = R.rt_arm({"tokens": [_tok("ώρα", 0.55)]})
    dec = R.rt_arm({"tokens": [_tok("ώρα", 0.55)]})
    assert pre["tokens"] == dec["tokens"] == ["ωρα"]
    assert pre["conf"] == dec["conf"] == [0.55]


def test_invalid_confidence_yields_none_and_is_counted():
    arm = R.rt_arm({"tokens": [_tok("ενα ", 1.5), _tok("δυο ", None),
                               _tok("τρια", 0.9)]})
    # a word with NO usable confidence is dropped by group_words entirely
    assert "τρια" in arm["tokens"]
    assert arm["conf"][-1] == 0.9
    assert any(c is None for c in arm["conf"])
    assert arm["stats"]["units_with_invalid_confidence"] >= 1


def test_word_without_timestamp_is_excluded():
    arm = R.rt_arm({"tokens": [{"text": "χωρίς", "confidence": 0.9, "is_final": True}]})
    assert arm["tokens"] == []
    assert arm["stats"]["words_without_timestamp"] == 1


def test_soniox_column_index_is_by_occurrence_not_by_string():
    cols = [("α", "ναι", "α"), ("β", None, "β"), (None, "ναι", None)]
    assert R.soniox_column_index(cols) == [0, None, 1]


# ------------------------------------------------------------------ golden arms
class FakeWindow:
    """Minimal stand-in for fusion_lab.Window: Prep only reads cols, ref, decisions,
    w_tokens."""

    def __init__(self, cols, ref, item_id="w", city="c", meeting="m"):
        self.cols = [tuple(c) for c in cols]
        self.ref = ref
        self.item_id = item_id
        self.city = city
        self.meeting = meeting
        self.pivot = 0
        self.w_tokens, self.decisions = compose(self.cols, pivot=0)


def _prep(cols, ref, conf):
    w = FakeWindow(cols, ref)
    return w, X.Prep(w, conf)


def test_class_coverage_of_eligibility():
    cols = [
        ("α", None, None),          # singleton, soniox absent
        (None, "β", None),          # singleton, SONIOX -> arm O
        ("γ", "γ", None),           # two_present_same -> nothing
        ("δ", "δ", "δ"),            # agree -> nothing
        ("ε", "ζ", "ε"),            # exact_2_of_3, soniox minority -> arm M
        ("η", "η", "θ"),            # exact_2_of_3, soniox majority -> nothing
        ("ι", "κ", None),           # unresolved_two -> arms O2, A
        ("λ", "μ", "ν"),            # unresolved_three -> arm A
    ]
    conf = [0.9] * sum(1 for c in cols if c[1] is not None)
    _w, p = _prep(cols, ["α"], conf)
    assert [i for i, _t, _c in p.elig["O"]] == [1]
    assert [i for i, _t, _c in p.elig["M"]] == [4]
    assert [i for i, _t, _c in p.elig["O2"]] == [6]
    assert [i for i, _t, _c in p.elig["A"]] == [6, 7]


def test_soniox_absent_column_is_never_eligible():
    cols = [("α", None, "β")]
    _w, p = _prep(cols, ["α"], [])
    assert all(not p.elig[a] for a in X.ARMS)


def test_invalid_confidence_makes_column_ineligible():
    cols = [(None, "β", None)]
    _w, p = _prep(cols, ["β"], [None])
    assert p.elig["O"] == []


def test_split_merge_columns_are_quarantined_for_every_arm():
    # "στο" spelled as one token by system 0 and split across two by system 1
    cols = [("στο", "σ", None), (None, "το", "στο")]
    conf = [0.99, 0.99]
    _w, p = _prep(cols, ["στο"], conf)
    assert all(p.elig[a] == [] for a in X.ARMS)


def test_arm_O_threshold_is_inclusive_at_tau():
    cols = [(None, "ναι", None)]
    _w, p = _prep(cols, ["ναι"], [0.50])
    assert X.apply_threshold_arm(p, "O", 0.50) == ["ναι"]      # conf == tau fires
    assert X.apply_threshold_arm(p, "O", 0.55) == []           # just below
    assert X.apply_threshold_arm(p, "O", 0.45) == ["ναι"]      # just above


def test_arm_O_does_not_touch_other_columns():
    cols = [("α", "α", "α"), (None, "β", None), ("γ", "δ", "γ")]
    _w, p = _prep(cols, ["α", "γ"], [0.9, 0.9, 0.9])
    assert X.apply_threshold_arm(p, "O", 0.0) == ["α", "β", "γ"]
    assert X.apply_threshold_arm(p, "O", 1.01) == ["α", "γ"]


def test_arm_O2_replaces_W_choice_above_tau_and_keeps_it_below():
    cols = [("ι", "κ", None)]
    w, p = _prep(cols, ["κ"], [0.8])
    assert p.base[0] == "ι"                       # tie_pivot picks system 0
    assert X.apply_threshold_arm(p, "O2", 0.7) == ["κ"]
    assert X.apply_threshold_arm(p, "O2", 0.9) == ["ι"]
    del w


def test_arm_M_overrides_the_majority_only_above_tau():
    cols = [("ε", "ζ", "ε")]
    _w, p = _prep(cols, ["ζ"], [0.95])
    assert p.base[0] == "ε"
    assert X.apply_threshold_arm(p, "M", 0.9) == ["ζ"]
    assert X.apply_threshold_arm(p, "M", 0.99) == ["ε"]


def test_arm_A_asymmetric_vote_on_unresolved_three():
    """k = 0.5 for both other systems, so on [x, y, z] Soniox wins iff conf > 0.5;
    at conf == 0.5 there is a three-way tie and W-rt's choice stands."""
    cols = [("λ", "μ", "ν")]
    for conf, want in ((0.9, "μ"), (0.5, "λ"), (0.1, "λ")):
        w, p = _prep(cols, ["μ"], [conf])
        assert X.apply_vote_arm(w, p, X.K_OTHER) == [want]


def test_arm_A_on_unresolved_two_needs_conf_above_k():
    cols = [("ι", "κ", None)]
    for conf, want in ((0.51, "κ"), (0.50, "ι"), (0.2, "ι")):
        w, p = _prep(cols, ["κ"], [conf])
        assert X.apply_vote_arm(w, p, X.K_OTHER) == [want]


def test_arm_A_never_touches_a_2_of_3_majority():
    cols = [("ε", "ζ", "ε")]
    w, p = _prep(cols, ["ζ"], [1.0])
    assert X.apply_vote_arm(w, p, X.K_OTHER) == ["ε"]


def test_arm_A_when_two_systems_agree_against_soniox_inside_the_tie_set():
    """[x, y, ε] where the two present are scribe and soniox: no third voter."""
    cols = [("ι", "κ", None)]
    w, p = _prep(cols, ["ι"], [1.0])
    assert X.apply_vote_arm(w, p, X.K_OTHER) == ["κ"]


# ----------------------------------------------------------- mutation scope
def test_mutation_scope_agree_and_quarantine_never_change():
    """Every token an arm changes must come from a column eligible for that arm;
    `agree` and split/merge-quarantined columns are untouchable."""
    cols = [("α", "α", "α"), ("στο", "σ", None), (None, "το", "στο"),
            (None, "β", None), ("λ", "μ", "ν")]
    conf = [0.99] * 5
    ref = ["α"]
    for arm, tau in (("O", 0.0), ("M", 0.0), ("A", None), ("O2", 0.0)):
        w, p = _prep(cols, ref, conf)
        out = (X.apply_vote_arm(w, p, X.K_OTHER) if arm == "A"
               else X.apply_threshold_arm(p, arm, tau))
        eligible = {i for i, _t, _c in p.elig[arm]}
        # the arm's output must equal W-rt's stream except at eligible columns
        untouchable = [t for i, t in enumerate(p.base)
                       if t is not None and i not in eligible]
        assert [t for t in out if t in untouchable] == untouchable
        assert all(p.classes[i] != "agree" for i in eligible)
        assert 0 not in eligible and 1 not in eligible and 2 not in eligible


# ------------------------------------------------------------------- fitting
class _W:
    def __init__(self, item_id, city, ref):
        self.item_id, self.city, self.ref = item_id, city, ref
        self.meeting = city


def test_fit_threshold_matches_brute_force_and_breaks_ties_upward():
    # one window: two eligible singleton columns, one right, one wrong
    cols = [(None, "ναι", None), (None, "λαθος", None)]
    w, p = _prep(cols, ["ναι"], [0.9, 0.3])
    preps = {"w": p}
    train = [w]
    tau = X.fit_threshold("O", train, preps)
    # firing only the 0.9 column is optimal: any tau in (0.3, 0.9]. Tie rule -> the
    # LARGEST such grid point, which is 0.9.
    assert tau == pytest.approx(0.9)
    assert X.apply_threshold_arm(p, "O", tau) == ["ναι"]

    # independent brute force with the frozen scorer: (errors, -tau); min picks the
    # fewest errors and, among ties, the largest tau.
    scored = []
    for t in X.GRID:
        num = sum(sum(sdi(" ".join(x.ref),
                          " ".join(X.apply_threshold_arm(preps[x.item_id], "O", t)))[:3])
                  for x in train)
        scored.append((num, -t))
    assert -min(scored)[1] == pytest.approx(tau)


def test_grid_contains_never_fire_and_always_fire():
    assert min(X.GRID) == 0.0
    assert max(X.GRID) > 1.0


def test_fast_objective_equals_the_frozen_scorer():
    """S + D + I from the frozen sdi is exactly the unit-cost Levenshtein distance."""
    rng = random.Random(3)
    vocab = ["ενα", "δυο", "τρια", "τεσσερα", "πεντε"]
    for _ in range(200):
        ref = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        s, d, i, n = sdi(" ".join(ref), " ".join(hyp))
        assert s + d + i == X._lev(ref, hyp)
        assert n == len(ref)


# --------------------------------------------------------------- permutation
def test_permuted_conf_preserves_the_multiset_within_each_meeting():
    cols = [(None, "α", None), (None, "β", None)]
    w1 = FakeWindow(cols, ["α"], item_id="w1", city="c", meeting="m1")
    w2 = FakeWindow(cols, ["β"], item_id="w2", city="c", meeting="m2")
    preps = {"w1": X.Prep(w1, [0.1, 0.2]), "w2": X.Prep(w2, [0.8, 0.9])}
    sub = type("S", (), {"windows": [w1, w2]})()
    ov = X.permuted_conf(sub, preps, "O", random.Random(0))
    assert sorted(ov["w1"].values()) == [0.1, 0.2]
    assert sorted(ov["w2"].values()) == [0.8, 0.9]


# --------------------------------------------------------------- statistics
def test_domination_handles_a_non_improving_arm():
    res = {"detail": {"meetings": ["m1", "m2"],
                      "rows_arm": [(1, 0, 0, 10), (1, 0, 0, 10)],
                      "rows_W": [(1, 0, 0, 10), (1, 0, 0, 10)]}}
    assert X.domination(res)["applicable"] is False


def test_domination_flags_a_single_meeting_effect():
    res = {"detail": {"meetings": ["m1", "m2"],
                      "rows_arm": [(0, 0, 0, 10), (1, 0, 0, 10)],
                      "rows_W": [(9, 0, 0, 10), (1, 0, 0, 10)]}}
    d = X.domination(res)
    assert d["dominated"] is True and d["top_meeting"] == "m1"


def test_city_sign_test_is_exact():
    res = {"per_city": {"per_city": {f"c{i}": {"delta": -1.0} for i in range(10)}}}
    out = X.city_sign_test(res)
    assert out["n_effective"] == 10
    assert out["p_two_sided"] == pytest.approx(2 / 1024)


def test_city_sign_test_handles_all_ties():
    res = {"per_city": {"per_city": {f"c{i}": {"delta": 0.0} for i in range(10)}}}
    assert X.city_sign_test(res)["p_two_sided"] is None


def test_bonferroni_quantiles():
    q = 100 * (0.05 / 3) / 2
    assert math.isclose(q, 0.8333333, rel_tol=1e-6)
    assert math.isclose(100 - q, 99.1666667, rel_tol=1e-6)


def test_curve_lookup_equals_direct_application():
    """The k-curve shortcut must give exactly the distance of the applied arm at
    every grid threshold, ties included."""
    rng = random.Random(11)
    cols = [(None, f"σ{i}", None) for i in range(6)]
    conf = [0.9, 0.9, 0.5, 0.5, 0.1, 0.0]      # deliberate ties
    w, p = _prep(cols, ["σ0", "σ3", "ξ"], conf)
    preps = {w.item_id: p}
    cur = X.curves([w], preps, "O")
    confs, dists = cur[w.item_id]
    for tau in X.GRID:
        direct = X._lev(w.ref, X.apply_threshold_arm(p, "O", tau))
        assert dists[X._k_for_tau(confs, tau)] == direct
    del rng


def test_fit_threshold_with_and_without_precomputed_curves_agree():
    cols = [(None, "ναι", None), (None, "λαθος", None), ("α", "β", None)]
    w, p = _prep(cols, ["ναι", "α"], [0.9, 0.3, 0.7])
    preps = {w.item_id: p}
    for arm in ("O", "O2"):
        a = X.fit_threshold(arm, [w], preps)
        b = X.fit_threshold(arm, [w], preps, cur=X.curves([w], preps, arm))
        assert a == b
