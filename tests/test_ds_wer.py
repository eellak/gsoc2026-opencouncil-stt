"""Tests for DS-WER, written before the implementation.

DS-WER is proposal Milestone 2's metric: Levenshtein restricted to domain-critical
words. Everything that can go quietly wrong with it is a counting rule, not an
algorithm - a term substituted for another term counted twice, an insertion of an
ordinary word counted as a domain error, a division by an empty denominator
reported as a perfect score. Each of those is a test here.

    .venv-eval/bin/python -m pytest tests/test_ds_wer.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds_wer import TermList, ds_wer  # noqa: E402


def tl(*terms) -> TermList:
    """Terms as (id, canonical, *aliases)."""
    return TermList([{"id": i, "canonical": c, "aliases": list(a)}
                     for i, c, *a in terms])


KAB = ("t_kabosos", "Καμπόσος", "Καμπόσος", "Καμπόσου")
SEL = ("t_selis", "Σελής", "Σελής", "Σελή")
DIMOS = ("t_dimos", "Δήμος Άργους", "Δήμος Άργους")


# ------------------------------------------------------------------ the basics
def test_exact_match_is_zero_error():
    r = ds_wer("ο κύριος Καμπόσος μίλησε", "ο κύριος Καμπόσος μίλησε", tl(KAB))
    assert (r["S"], r["D"], r["I"], r["N"]) == (0, 0, 0, 1)
    assert r["ds_wer"] == 0.0


def test_term_substituted_by_ordinary_word_is_one_substitution():
    r = ds_wer("ο κύριος Καμπόσος μίλησε", "ο κύριος καμπόστος μίλησε", tl(KAB))
    assert (r["S"], r["D"], r["I"], r["N"]) == (1, 0, 0, 1)
    assert r["ds_wer"] == 1.0


def test_missing_term_is_a_deletion():
    r = ds_wer("ο κύριος Καμπόσος μίλησε", "ο κύριος μίλησε", tl(KAB))
    assert (r["S"], r["D"], r["I"], r["N"]) == (0, 1, 0, 1)


def test_term_appearing_only_in_the_hypothesis_is_a_domain_insertion():
    r = ds_wer("ο κύριος μίλησε καλά", "ο κύριος Καμπόσος μίλησε καλά", tl(KAB))
    assert (r["S"], r["D"], r["I"]) == (0, 0, 1)


def test_insertion_of_an_ordinary_word_is_not_a_domain_error():
    r = ds_wer("ο Καμπόσος μίλησε", "ο Καμπόσος μίλησε πολύ", tl(KAB))
    assert (r["S"], r["D"], r["I"]) == (0, 0, 0)


def test_term_for_term_substitution_is_counted_once():
    """The failure mode this rule exists for: S on the reference side and I on the
    hypothesis side for a single alignment operation would double-charge it."""
    r = ds_wer("ο Καμπόσος μίλησε", "ο Σελής μίλησε", tl(KAB, SEL))
    assert (r["S"], r["D"], r["I"]) == (1, 0, 0)
    assert r["S"] + r["D"] + r["I"] == 1


# ------------------------------------------------------- repeats and reordering
def test_repeated_terms_count_separately():
    r = ds_wer("Καμπόσος και Καμπόσος", "Καμπόσος και καμπόστος", tl(KAB))
    assert r["N"] == 2
    assert (r["S"], r["D"], r["I"]) == (1, 0, 0)


def test_reordered_terms_cost_something():
    """Levenshtein has no move operation: a swap is not free, and pretending it is
    would let a system that scrambles the roll call score as if it read it."""
    r = ds_wer("Καμπόσος Σελής", "Σελής Καμπόσος", tl(KAB, SEL))
    assert r["N"] == 2
    assert r["S"] + r["D"] + r["I"] > 0


# ----------------------------------------------------- multiword and overlapping
def test_multiword_term_matches_as_one_unit():
    r = ds_wer("στον Δήμος Άργους σήμερα", "στον Δήμος Άργους σήμερα", tl(DIMOS))
    assert (r["N"], r["S"], r["D"], r["I"]) == (1, 0, 0, 0)


def test_multiword_term_half_wrong_is_one_substitution_not_two():
    r = ds_wer("στον Δήμος Άργους σήμερα", "στον Δήμος Άργος σήμερα", tl(DIMOS))
    assert r["N"] == 1
    assert r["S"] + r["D"] + r["I"] >= 1


def test_overlapping_terms_resolve_longest_match_leftmost():
    """A short term that is a *prefix* of a long one must not win.

    "Άργος" and "Άργος Μυκήνες" are both terms; on the text "Άργος Μυκήνες" a
    shortest-match matcher takes the prefix, leaves "Μυκήνες" loose, and reports
    two domain terms where there is one.
    """
    t = tl(("t_argos", "Άργος", "Άργος"),
           ("t_am", "Άργος Μυκήνες", "Άργος Μυκήνες"))
    r = ds_wer("στο Άργος Μυκήνες σήμερα", "στο Άργος Μυκήνες σήμερα", t)
    assert r["N"] == 1, "the long term should consume the short one"
    assert r["per_term"]["t_am"]["N"] == 1
    assert r["per_term"].get("t_argos", {"N": 0})["N"] == 0


def test_a_term_nested_inside_a_longer_one_still_matches_on_its_own():
    t = tl(("t_argos", "Άργος", "Άργος"),
           ("t_am", "Άργος Μυκήνες", "Άργος Μυκήνες"))
    r = ds_wer("στο Άργος σήμερα", "στο Άργος σήμερα", t)
    assert r["per_term"]["t_argos"]["N"] == 1


# -------------------------------------------------------------- Greek normalizing
@pytest.mark.parametrize("hyp", [
    "ο κυριος καμποσος μιλησε",     # no tonos
    "Ο ΚΥΡΙΟΣ ΚΑΜΠΟΣΟΣ ΜΙΛΗΣΕ",     # uppercase
])
def test_matching_is_case_and_tonos_insensitive(hyp):
    r = ds_wer("ο κύριος Καμπόσος μίλησε", hyp, tl(KAB))
    assert (r["N"], r["S"], r["D"], r["I"]) == (1, 0, 0, 0)


def test_final_sigma_is_not_folded():
    """The frozen scorer keeps ς and σ distinct. That is a real error under this
    project's normalizer and the metric must not quietly disagree with it."""
    r = ds_wer("ο Καμπόσος μίλησε", "ο Καμπόσοσ μίλησε", tl(KAB))
    assert r["S"] == 1


def test_hyphens_split_into_tokens_on_both_sides():
    t = tl(("t_am", "Άργος Μυκήνες", "Άργος - Μυκήνες"))
    r = ds_wer("το Άργος - Μυκήνες", "το Άργος Μυκήνες", t)
    assert (r["N"], r["S"], r["D"], r["I"]) == (1, 0, 0, 0)


# ------------------------------------------------------------- edges and failures
def test_no_terms_in_the_reference_returns_na_not_zero():
    r = ds_wer("καλημέρα σε όλους", "καλημέρα σε όλους", tl(KAB))
    assert r["N"] == 0
    assert r["ds_wer"] is None


def test_ds_wer_can_exceed_one_via_insertions():
    r = ds_wer("ο Καμπόσος", "ο Καμπόσος Σελής Σελής", tl(KAB, SEL))
    assert r["N"] == 1
    assert r["ds_wer"] > 1.0


def test_wrong_city_term_list_finds_nothing():
    r = ds_wer("ο Καμπόσος μίλησε", "ο Καμπόσος μίλησε",
               tl(("t_x", "Τσομπανίδης", "Τσομπανίδης")))
    assert r["N"] == 0 and r["ds_wer"] is None


def test_duplicate_alias_is_rejected_at_load():
    with pytest.raises(ValueError, match="alias"):
        tl(("t_a", "Καμπόσος", "Καμπόσος"), ("t_b", "Άλλος", "Καμπόσος"))


def test_alignment_is_deterministic():
    args = ("Καμπόσος Σελής Καμπόσος", "Σελής Καμπόσος Σελής", tl(KAB, SEL))
    first = ds_wer(*args)
    for _ in range(5):
        assert ds_wer(*args) == first


def test_frozen_city_files_load_and_carry_terms():
    for city in ("argos", "orestiada"):
        p = ROOT / f"research/ds_wer/terms/{city}.json"
        terms = TermList(json.loads(p.read_text())["terms"])
        assert len(terms) > 20
