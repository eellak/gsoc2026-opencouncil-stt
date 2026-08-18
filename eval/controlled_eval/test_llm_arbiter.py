#!/usr/bin/env python3
"""Tests for F1, the LLM majority arbiter. EXPLORATORY_CONTAMINATED_NOT_CONFIRMATORY.

The load-bearing one is `test_eligibility_is_reference_blind`: the single easiest way
to ruin this experiment is to narrow the eligible set with something only the reference
knows ("the 1,245 wrong majorities"). Everything else here guards the outcome
partition, the cache key and the no-silent-fallback rule.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import exp_llm_arbiter as A          # noqa: E402
from eval.controlled_eval.fusion_lab import Window             # noqa: E402
from eval.controlled_eval.msa import compose                   # noqa: E402


def mkwin(cols, ref, item_id="w1", city="athens", meeting="m1") -> Window:
    w_tokens, decisions = compose(cols, pivot=0)
    return Window(item_id=item_id, city=city, meeting=meeting, ref=ref,
                  hyps=[[], [], []], pivot=0, cols=[tuple(c) for c in cols],
                  decisions=decisions, w_tokens=w_tokens, v_tokens=list(w_tokens),
                  in_training=False)


COLS = [
    ("το", "το", "το"),              # agree
    ("δήμο", "δήμο", "δήμος"),       # exact_2_of_3  -> eligible
    ("και", "κι", None),             # unresolved_two
    (None, "εγώ", None),             # singleton
    ("θέμα", "θέματα", "θέμα"),      # exact_2_of_3  -> eligible
]


# ------------------------------------------------------------------ eligibility
def test_eligibility_picks_exactly_the_2_of_3_columns():
    assert A.eligible_columns(COLS) == [1, 4]


def test_eligibility_is_reference_blind():
    """Replacing the reference with garbage may not move a single eligible column,
    a single question, or a single rendered prompt."""
    import inspect
    assert "ref" not in inspect.signature(A.eligible_columns).parameters

    class Sub:
        def __init__(self, ws):
            self.windows = ws

    good = mkwin(COLS, ["το", "δήμος", "και", "εγώ", "θέμα"])
    junk = mkwin(COLS, ["ΧΧΧ"] * 40)
    a = A.build_questions(Sub([good]), {})
    b = A.build_questions(Sub([junk]), {})
    assert a == b and len(a) == 2
    for q in a:
        assert A.render(q, 1) == A.render(
            [x for x in b if x["id"] == q["id"]][0], 1)


def test_majority_minority_and_w_agreement():
    w = mkwin(COLS, [])
    for i in A.eligible_columns(COLS):
        maj, mino = A.majority_minority(COLS[i])
        assert w.decisions[i]["token"] == maj
        assert maj != mino


def test_majority_minority_rejects_non_2_of_3():
    with pytest.raises(AssertionError):
        A.majority_minority(("α", "β", "γ"))


# ------------------------------------------------------------------ the question
def test_masked_slot_hides_the_current_token():
    w = mkwin(COLS, [])
    pos = A.w_positions(w.decisions)
    ctx = A.masked_context(w.w_tokens, pos[1])
    assert "_____" in ctx
    assert "δήμο" not in ctx.split()          # W's own token is not shown
    assert "το" in ctx.split()


def test_both_orders_are_exact_swaps_and_pass3_repeats_pass1():
    q = {"id": "x#1", "majority": "δήμο", "minority": "δήμος",
         "context": "α _____ β", "terms": "", "flip": True}
    a1, b1 = A.render(q, 1)[A.LABELS[0]], A.render(q, 1)[A.LABELS[1]]
    a2, b2 = A.render(q, 2)[A.LABELS[0]], A.render(q, 2)[A.LABELS[1]]
    assert (a1, b1) == (b2, a2)
    assert A.render(q, 3) == A.render(q, 1)   # A/A replicate, same mapping
    assert A.token_for(q, 1, A.LABELS[0]) == a1
    assert A.token_for(q, 2, A.LABELS[0]) == a2


def test_prompt_never_names_a_provider_or_the_majority():
    q = {"id": "x#1", "majority": "δήμο", "minority": "δήμος",
         "context": "α _____ β", "terms": "", "flip": False}
    blob = A.batch_wire([q], 1)
    for bad in ("scribe", "soniox", "runpod", "whisper", "πλειοψηφ", "majority"):
        assert bad.lower() not in blob.lower()


# ------------------------------------------------------------------ batching/cache
def test_batches_are_stable_and_identical_across_passes():
    qs = [{"id": f"w#{i}"} for i in range(50)]
    b1 = A.plan_batches(qs, 12)
    b2 = A.plan_batches(list(reversed(qs)), 12)
    assert b1 == b2                                    # order of input is irrelevant
    assert sum(len(b) for b in b1) == 50
    assert [q["id"] for b in b1 for q in b] == [q["id"] for b in b2 for q in b]


def test_cache_key_changes_with_batch_size_prompt_and_pass():
    qs = [{"id": f"w#{i}", "majority": "α", "minority": "β", "context": "x _____ y",
           "terms": "", "flip": False} for i in range(30)]
    b12 = A.plan_batches(qs, 12)
    b24 = A.plan_batches(qs, 24)
    q = qs[0]
    k12 = A.cache_key(q, 1, [b for b in b12 if q in b][0])
    k24 = A.cache_key(q, 1, [b for b in b24 if q in b][0])
    kp2 = A.cache_key(q, 2, [b for b in b12 if q in b][0])
    assert len({k12, k24, kp2}) == 3


# ------------------------------------------------------------------ parsing
def test_answers_of_rejects_duplicates_unknowns_and_bad_labels():
    batch = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    got = [{"id": "a", "pick": "Α", "conf": 80},
           {"id": "b", "pick": "Β", "conf": 70},
           {"id": "b", "pick": "Α", "conf": 60},       # duplicate -> invalid
           {"id": "c", "pick": "ΤΡΙΤΟ", "conf": 50},   # illegal label
           {"id": "zz", "pick": "Α", "conf": 90}]      # never asked
    out = A.answers_of(got, batch)
    assert out == {"a": {"pick": "Α", "conf": 80}, "b": None,
                   "c": None, "d": None}


def test_latin_lookalikes_are_the_same_choice():
    assert A.norm_label("A") == A.LABELS[0]
    assert A.norm_label(" b ") == A.LABELS[1]
    assert A.norm_label("αποχη") == A.ABSTAIN
    assert A.norm_label("Γ") is None
    assert A.norm_label(3) is None


# ------------------------------------------------------------------ resolution
def _q(qid="w1#1", flip=False):
    return {"id": qid, "majority": "MAJ", "minority": "MIN", "context": "x _____ y",
            "terms": "", "flip": flip}


def _answer(pick, conf=80):
    return {"pick": pick} if pick == A.ABSTAIN else {"pick": pick, "conf": conf}


def _caches(qs, batch_size, p1, p2):
    b = {q["id"]: bb for bb in A.plan_batches(qs, batch_size) for q in bb}
    return {1: {A.cache_key(q, 1, b[q["id"]]): p1(q) for q in qs},
            2: {A.cache_key(q, 2, b[q["id"]]): p2(q) for q in qs}}


def test_outcome_partition_is_exhaustive_and_in_precedence_order():
    qs = [_q("w#0"), _q("w#1"), _q("w#2"), _q("w#3"), _q("w#4")]
    lab = {}
    # w#0 override, w#1 confirm, w#2 explicit abstain, w#3 order disagree, w#4 invalid
    def p(pass_no):
        def f(q):
            i = int(q["id"].split("#")[1])
            if i == 0:
                return _answer(next(label for label in A.LABELS
                                     if A.token_for(q, pass_no, label) == "MIN"))
            if i == 1:
                return _answer(next(label for label in A.LABELS
                                     if A.token_for(q, pass_no, label) == "MAJ"))
            if i == 2:
                return _answer(A.ABSTAIN if pass_no == 2 else A.LABELS[0])
            if i == 3:
                return _answer(A.LABELS[0])  # same LABEL both passes = opposite tokens
            return None                       # missing answer -> invalid
        return f
    caches = _caches(qs, 12, p(1), p(2))
    r = A.resolve(qs, caches, 12)
    assert [r[f"w#{i}"]["outcome"] for i in range(5)] == [
        "override", "confirm", "abstain_explicit", "order_disagree", "invalid"]
    assert set(o for o in A.OUTCOMES) == {v["outcome"] for v in r.values()}


def test_conf_threshold_zero_reproduces_the_fixture_partition():
    qs = [_q("w#0"), _q("w#1")]

    def minority(q, pass_no):
        return next(label for label in A.LABELS
                    if A.token_for(q, pass_no, label) == q["minority"])

    def majority(q, pass_no):
        return next(label for label in A.LABELS
                    if A.token_for(q, pass_no, label) == q["majority"])

    caches = _caches(qs, 12,
                     lambda q: _answer(minority(q, 1), 0),
                     lambda q: _answer(minority(q, 2), 100))
    r = A.resolve(qs, caches, 12, conf_threshold=0)
    assert [r[q["id"]]["outcome"] for q in qs] == ["override", "override"]


def test_low_confidence_override_becomes_confirm_only():
    qs = [_q("w#0")]

    def minority(q, pass_no):
        return next(label for label in A.LABELS
                    if A.token_for(q, pass_no, label) == q["minority"])

    caches = _caches(qs, 12,
                     lambda q: _answer(minority(q, 1), 20),
                     lambda q: _answer(minority(q, 2), 20))
    assert A.resolve(qs, caches, 12)["w#0"]["outcome"] == "override"
    raised = A.resolve(qs, caches, 12, conf_threshold=50)
    assert raised["w#0"]["outcome"] == "confirm"
    assert raised["w#0"]["outcome"] in {"override", "confirm"}


@pytest.mark.parametrize("answer", [
    {"pick": "Α"},
    {"pick": "Α", "conf": "50"},
    {"pick": "Α", "conf": 101},
])
def test_bad_non_abstain_confidence_is_invalid(answer):
    qs = [_q("w#0")]
    caches = _caches(qs, 12, lambda q: answer, lambda q: answer)
    assert A.resolve(qs, caches, 12)["w#0"]["outcome"] == "invalid"


def test_invalid_never_becomes_a_silent_confirm():
    """A missing answer must be counted `invalid`, not folded into "kept W"."""
    qs = [_q("w#0")]
    caches = _caches(qs, 12, lambda q: None, lambda q: A.LABELS[0])
    r = A.resolve(qs, caches, 12)
    assert r["w#0"]["outcome"] == "invalid"
    assert "token" not in r["w#0"]


def test_abstain_beats_order_disagree_in_precedence():
    qs = [_q("w#0")]
    caches = _caches(qs, 12, lambda q: _answer(A.ABSTAIN),
                     lambda q: _answer(A.LABELS[0]))
    assert A.resolve(qs, caches, 12)["w#0"]["outcome"] == "abstain_explicit"


def test_excluded_buckets_never_enter_build_questions():
    class Sub:
        def __init__(self, ws):
            self.windows = ws

    cols = [
        ("2024", "2024", "2025"),
        ("και", "και", "κι"),
        ("δήμο", "δήμο", "δήμος"),
    ]
    qs = A.build_questions(Sub([mkwin(cols, [])]), {})
    assert [q["col"] for q in qs] == [2]


def test_term_ranking_is_context_first_and_deterministic():
    ctxs = {("athens", "m1"): SimpleNamespace(present={
        "early": {"term": {"canonical": "Αβγδ"}},
        "seen": {"term": {"canonical": "Ωμέγα"}},
        "near": {"term": {"canonical": "δήμος"}},
    })}
    context = "η ΩΜΈΓΑ εμφανίζεται εδώ"
    a = A.terms_for(ctxs, "athens", "m1", context, ("δήμο", "άλλο"))
    b = A.terms_for(ctxs, "athens", "m1", context, ("δήμο", "άλλο"))
    assert a == b
    assert a.split(", ")[0] == "Ωμέγα"


def test_msa_alignment_cache_key_is_frozen():
    from eval.controlled_eval.fusion_lab import _cache_path
    assert A.ALIGN_CACHE_EXPECTED == "align_65b1c4d64618a429.json"
    assert _cache_path().name == "align_65b1c4d64618a429.json"


# ------------------------------------------------------------------ application
def test_override_alters_exactly_one_index_and_only_that_one():
    w = mkwin(COLS, [])
    idea = A.F1Arbiter({"w1": {1: "δήμος"}})
    out = idea.apply(w, None)
    assert idea.applied == 1 and idea.collisions == 0 and idea.mapping_failures == 0
    assert sum(1 for a, b in zip(out, w.w_tokens) if a != b) == 1
    assert out[A.w_positions(w.decisions)[1]] == "δήμος"


def test_no_override_is_the_identity():
    w = mkwin(COLS, [])
    assert A.F1Arbiter({}).apply(w, None) == w.w_tokens


def test_override_on_an_epsilon_column_hard_fails():
    w = mkwin(COLS, [])
    with pytest.raises(AssertionError):
        A.F1Arbiter({"w1": {3: "x"}}).apply(w, None)   # col 3 is a dropped singleton?


# ------------------------------------------------------------------ pilot
def test_pilot_sample_is_deterministic_and_reference_blind():
    qs = [{"id": f"w#{i}"} for i in range(500)]
    a = [q["id"] for q in A.pilot_sample(qs, 120)]
    b = [q["id"] for q in A.pilot_sample(list(reversed(qs)), 120)]
    assert a == b and len(a) == 120 and len(set(a)) == 120
