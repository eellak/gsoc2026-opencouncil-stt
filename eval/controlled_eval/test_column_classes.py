"""Frozen tests for the column partition and arm eligibility of wayfinder #24.

The cases are Codex job 55293f6b's list, which is the evaluation that the
classification is implemented as designed. The two that matter most:

  * a real token majority [x, x, y] is SETTLED and no arm may touch it;
  * a token-boundary disagreement is quarantined from both arms, because an arm
    editing one column of it cannot see the neighbouring column it must agree with.

    .venv-eval/bin/python -m pytest eval/controlled_eval/test_column_classes.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.controlled_eval.column_classes import (  # noqa: E402
    column_class, column_flags, eligibility, split_merge_columns,
)

E = None


def test_the_partition():
    assert column_class((E, E, E)) == "invalid"
    assert column_class(("λογος", E, E)) == "singleton"
    assert column_class(("λογος", "λογος", E)) == "two_present_same"
    assert column_class(("λογος", "λογος", "λογος")) == "agree"
    assert column_class(("τυχη", "τυχη", "τειχη")) == "exact_2_of_3"
    assert column_class(("τυχη", "τειχη", E)) == "unresolved_two"
    assert column_class(("τυχη", "τειχη", "θηκη")) == "unresolved_three"


def test_agree_is_reachable():
    # the first draft of this partition made `agree` unreachable
    assert column_class(("α", "α", "α")) == "agree"
    assert column_class(("α", "α", E)) != "agree"


def test_a_token_majority_is_never_eligible():
    cols = [("τυχη", "τυχη", "τειχη")]
    assert eligibility(cols) == {}
    for c in (("λ", "λ", "λ"), ("λ", "λ", E), ("λ", E, E)):
        assert eligibility([c]) == {}


def test_homophone_columns_go_to_H_and_never_see_epsilon():
    cols = [("τυχη", "τειχη", E)]
    assert eligibility(cols) == {0: "H"}
    assert column_flags(cols[0])["strict_homophone"]
    # occupancy was settled 2:1 before H runs, so H substitutes and cannot delete
    assert column_class(cols[0]) == "unresolved_two"


def test_two_string_columns_are_not_eligible_for_C():
    # two strings have no character majority; only H may act there
    cols = [("δημος", "δρομος", E)]
    assert eligibility(cols) == {}


def test_C_takes_three_distinct_near_character_columns():
    cols = [("δημου", "δημος", "δημο")]
    assert eligibility(cols) == {0: "C"}


def test_C_abstains_when_the_candidates_are_far_apart():
    cols = [("δημος", "συμβουλιο", "προεδρος")]
    assert eligibility(cols) == {}


def test_H_wins_the_overlap():
    cols = [("τυχη", "τειχη", "τηχη")]
    assert column_flags(cols[0])["strict_homophone"]
    assert eligibility(cols) == {0: "H"}


def test_loose_only_homophones_are_a_separate_variant():
    cols = [("αυτο", "αφτο", E)]
    assert eligibility(cols, loose=False) == {}
    assert eligibility(cols, loose=True) == {0: "H"}


def test_split_merge_is_flagged_and_quarantined():
    cols = [("στο", "σ", E), (E, "το", "στο")]
    assert split_merge_columns(cols) == {0, 1}
    assert eligibility(cols) == {}


def test_split_merge_does_not_fire_on_ordinary_columns():
    cols = [("δημος", "δημος", "δημος"), ("ειναι", "ινε", E)]
    assert split_merge_columns(cols) == set()
    assert eligibility(cols) == {1: "H"}


def test_partial_homophone_is_a_flag_not_an_eligibility():
    col = ("τυχη", "τειχη", "δρομος")
    f = column_flags(col)
    assert f["partial_homophone"] and not f["strict_homophone"]
    assert eligibility([col]) == {}


def test_classification_is_invariant_to_provider_order():
    import itertools
    col = ("δημου", "δημος", "δημο")
    ks = {column_class(p) for p in itertools.permutations(col)}
    assert ks == {"unresolved_three"}
    assert all(eligibility([p]) == {0: "C"} for p in itertools.permutations(col))
