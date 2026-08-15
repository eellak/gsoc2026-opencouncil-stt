"""Tests for arm B (hotwords biasing): roster -> hotwords construction and the
preregistered runtime assertions. No model is loaded; the tokenizer is stubbed.

Spec: docs/specs/2026-08-12-serving-stack-plan.md, arm B, including both frozen
amendments: surnames only (one per person, no full names in the primary arm),
ranked by SHA-256(salt || meeting || surname_key), greedy inclusion of whole
surnames up to the budget, never a truncated surname, never silent drops.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.serving_stack.roster_hotwords import (  # noqa: E402
    FROZEN_SALT, HOTWORD_TOKEN_BUDGET, build_hotwords, build_hotwords_detail,
    count_tokens, hash_rank, norm_key, ordered_candidates)


class FakeTok:
    """Stub tokenizer: every whitespace-separated chunk costs `per_word` ids.
    Mirrors the real interface: .encode(text, add_special_tokens=False).ids."""

    def __init__(self, per_word: int = 1):
        self.per_word = per_word

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False, \
            "budget must be counted the way faster-whisper encodes (no specials)"
        return SimpleNamespace(ids=[0] * (self.per_word * len(text.split())))


class CharTok:
    """Stub tokenizer where cost tracks surname length, to exercise greedy
    whole-surname inclusion (a long surname is skipped, a later short one fits)."""

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[0] * len(text.replace(" ", "")))


# "nowhere" has no research/ds_wer/terms/nowhere.json -> roster-fallback path.
CITY = "nowhere"


def rosters(entries, key=f"{CITY}/m1"):
    return {key: entries}


def build(entries, tok=None, meeting="m1", **kw):
    return build_hotwords_detail(CITY, meeting, tok or FakeTok(),
                                 rosters=rosters(entries, f"{CITY}/{meeting}"),
                                 **kw)


# --------------------------------------------------------------------- coverage
def test_uncovered_meeting_is_none():
    tok = FakeTok()
    assert build_hotwords(CITY, "nope", tok, rosters={}) is None
    assert build_hotwords(CITY, "m1", tok, rosters=rosters([])) is None
    d = build_hotwords_detail(CITY, "m1", tok, rosters={})
    assert d["hotwords"] is None and d["reason"] == "uncovered"


# ----------------------------------------------------------------------- budget
def test_budget_never_exceeded():
    # 100 persons: surname costs 10 ids; way over 160
    entries = [f"Ονομα{i:03d} Επωνυμο{i:03d}" for i in range(100)]
    tok = FakeTok(per_word=10)
    hw = build_hotwords(CITY, "m1", tok, rosters=rosters(entries))
    assert hw is not None
    assert count_tokens(tok, hw) <= HOTWORD_TOKEN_BUDGET


def test_greedy_keeps_hash_order_and_reports_every_drop():
    entries = [f"Ονομα{i:03d} Επωνυμο{i:03d}" for i in range(100)]
    tok = FakeTok(per_word=10)
    d = build(entries, tok)
    ordered, meta = ordered_candidates(CITY, "m1", entries)
    kept, dropped = d["kept"], d["dropped"]
    # kept and dropped partition the candidate list, both preserving hash order:
    # nothing is silently discarded.
    assert sorted(kept + dropped, key=ordered.index) == ordered
    assert set(kept) | set(dropped) == set(ordered)
    assert kept == [e for e in ordered if e in set(kept)]
    assert dropped == [e for e in ordered if e in set(dropped)]
    assert dropped, "test must exercise the overflow path"
    assert d["hotwords"] == ", ".join(kept)
    # single-word surnames at 10 ids -> exactly 16 fit in 160
    assert len(kept) == 16 and d["tokens"] == 160
    assert meta["n_surnames"] == 100 and meta["order"] == "sha256-salted"


def test_no_truncated_surname_greedy_skips_whole_names_only():
    # one 100-char surname (cost ~100) among 1-char surnames, budget tiny:
    # the long one is skipped WHOLE and later short ones still enter.
    entries = ["Α" * 100] + [chr(0x391 + i) for i in range(1, 15)]  # Β, Γ, ...
    tok = CharTok()
    d = build(entries, tok, budget=30)
    assert "Α" * 100 in d["dropped"]
    assert all(s in entries for s in d["kept"]), "only whole surnames appear"
    assert len(d["kept"]) >= 10, "short surnames after the long one still fit"
    assert d["tokens"] <= 30
    # no substring of the long surname leaked into the string
    assert "ΑΑΑ" not in d["hotwords"]


def test_first_entry_over_budget_yields_none_not_empty_string():
    tok = FakeTok(per_word=200)   # a single surname costs 200 > 160
    d = build(["Ονομα Επωνυμο"], tok)
    assert d["hotwords"] is None
    assert d["reason"] == "nothing_fits_budget"
    assert d["dropped"], "the overflow must be reported, not silent"


def test_budget_counts_the_joined_string_not_per_entry_sums():
    entries = ["Αλφα Βητα", "Γαμμα Δελτα", "Εψιλον"]
    tok = FakeTok(per_word=3)
    d = build(entries, tok)
    assert d["tokens"] == count_tokens(tok, d["hotwords"])


def test_secondary_budget_same_policy_more_room():
    entries = [f"Ονομα{i:03d} Επωνυμο{i:03d}" for i in range(100)]
    tok = FakeTok(per_word=10)
    d160 = build(entries, tok, budget=160)
    d200 = build(entries, tok, budget=200)
    assert len(d200["kept"]) == 20 and d200["tokens"] <= 200
    assert len(d160["kept"]) == 16
    # identical policy: both follow the same hash order
    ordered, _ = ordered_candidates(CITY, "m1", entries)
    assert d200["kept"] == [e for e in ordered if e in set(d200["kept"])]


def test_budget_must_stay_under_upstream_silent_cut():
    with pytest.raises(AssertionError, match="223"):
        build(["Ονομα Επωνυμο"], FakeTok(), budget=223)


# --------------------------------------------------------- hash-order semantics
def test_order_is_hash_ranked_not_alphabetical():
    entries = [f"Επωνυμο{i:02d}" for i in range(30)]
    ordered, _ = ordered_candidates(CITY, "m1", entries)
    assert sorted(ordered, key=norm_key) != ordered, \
        "30 surnames coming out alphabetical means the hash rank is not applied"
    expect = sorted(entries, key=lambda s: (hash_rank(f"{CITY}/m1", s),
                                            norm_key(s)))
    assert ordered == expect


def test_determinism_given_salt_and_meeting():
    entries = ["Γάμμα Δέλτα", "Άλφα Βήτα", "Ζήτα", "Κ. Λάμδα"]
    tok = FakeTok(per_word=2)
    a = build(entries, tok)["hotwords"]
    b = build(entries, tok)["hotwords"]
    c = build(list(reversed(entries)), tok)["hotwords"]   # input order irrelevant
    assert a == b == c
    assert FROZEN_SALT == "oc-hotwords-2026-08-12", "the salt is frozen"


def test_different_meetings_produce_different_orderings():
    entries = [f"Επωνυμο{i:02d}" for i in range(20)]
    o1, _ = ordered_candidates(CITY, "m1", entries)
    o2, _ = ordered_candidates(CITY, "m2", entries)
    assert set(o1) == set(o2)
    assert o1 != o2, "the meeting id must reshuffle the hash rank"


def test_excluded_tail_differs_across_meetings():
    entries = [f"Ονομα{i:02d} Επωνυμο{i:02d}" for i in range(30)]
    tok = FakeTok(per_word=10)
    d1 = build(entries, tok, meeting="m1")
    d2 = build(entries, tok, meeting="m2")
    assert set(d1["dropped"]) != set(d2["dropped"]), \
        "hash ranking exists to decorrelate the excluded subset across meetings"


# ------------------------------------------------------------- candidate pools
def test_surnames_only_no_full_names_in_primary_arm():
    entries = ["Βασίλης Μαυρίδης", "Άννα Ζαχαρίου", "Γιώργος Καδόγλου"]
    d = build(entries, FakeTok())
    assert set(d["kept"]) == {"Μαυρίδης", "Ζαχαρίου", "Καδόγλου"}
    assert all(" " not in e for e in d["kept"]), "full names are dropped entirely"


def test_initial_dotted_entry_gives_surname_only():
    d = build(["Β. Μαυρίδης", "Αδρακτάς"], FakeTok())
    assert set(d["kept"]) == {"Αδρακτάς", "Μαυρίδης"}
    assert "Β." not in d["hotwords"]


def test_party_and_comma_entries_are_skipped_in_fallback():
    entries = ["Αλλαγή Πορείας - Δημήτριος Καμπόσος",   # >2 words: party-like
               "Ορεστιάδα Νέα Ξανά",                     # >2 words
               "Α, Β",                                   # comma entry
               "Βασίλης Μαυρίδης"]
    d = build(entries, FakeTok())
    assert d["kept"] == ["Μαυρίδης"]


def test_single_word_first_name_is_not_a_surname():
    d = build(["Βασίλης", "Αδρακτάς", "Βασίλης Μαυρίδης"], FakeTok())
    assert set(d["kept"]) == {"Αδρακτάς", "Μαυρίδης"}


def test_term_file_source_preferred_over_roster_parsing():
    terms = [{"id": "person:μαυριδης", "canonical": "Μαυρίδης",
              "klass": "person_surname", "covers": ["Βασίλης Μαυρίδης"],
              "aliases": ["μαυριδη", "μαυριδης"]},
             {"id": "person:αδρακτας", "canonical": "Αδρακτάς",
              "klass": "person_surname", "covers": ["Κωνσταντίνος Αδρακτάς"],
              "aliases": ["αδρακτα", "αδρακτας"]}]
    # dotted roster entry matches via the surname word; canonical surface is used
    d = build(["Β. Μαυρίδης"], FakeTok(), terms=terms)
    assert d["source"] == "term_file"
    assert d["kept"] == ["Μαυρίδης"]
    # person matched via a verbatim cover -> still surname only
    d2 = build(["Κωνσταντίνος Αδρακτάς"], FakeTok(), terms=terms)
    assert d2["kept"] == ["Αδρακτάς"]
    # roster person not in the term list -> no candidates from term path
    d3 = build(["Άγνωστος Ανθρωπος"], FakeTok(), terms=[])
    assert d3["hotwords"] is None and d3["reason"] == "no_persons_matched"


# ------------------------------------------------------------------------ dedup
def test_dedup_after_nfc_casefold():
    composed = "Παπαδόπουλος"                            # NFC
    decomposed = unicodedata.normalize("NFD", composed)  # same name, NFD
    upper = composed.upper()
    d = build([composed, decomposed, upper, "Νικολάου"], FakeTok())
    assert d["n_duplicates"] == 2
    assert len(d["kept"]) == 2
    assert sorted(norm_key(e) for e in d["kept"]) == \
        sorted({norm_key(composed), norm_key("Νικολάου")})


def test_surname_dedup_across_dotted_and_full_entries():
    d = build(["Β. Μαυρίδης", "Βασίλης Μαυρίδης", "Μαυρίδης"], FakeTok())
    assert d["kept"] == ["Μαυρίδης"]


# ----------------------------------------------- runtime assertions (hotwords_arm)
@pytest.fixture(scope="module")
def arm():
    from scripts.serving_stack import hotwords_arm
    return hotwords_arm


def test_kwargs_are_control_plus_hotwords_only(arm):
    control = {"language": "el", "beam_size": 5}
    kw = arm.transcribe_kwargs("Άλφα, Βήτα", control=control)
    assert kw == {"language": "el", "beam_size": 5, "hotwords": "Άλφα, Βήτα"}
    # no hotwords -> exact control, kwarg absent (exact no-op for the decoder)
    assert arm.transcribe_kwargs(None, control=control) == control


def test_prefix_set_fails_the_assertion(arm):
    with pytest.raises(AssertionError, match="prefix"):
        arm.transcribe_kwargs("Άλφα", control={"prefix": "x"})


def test_initial_prompt_set_fails_the_assertion(arm):
    with pytest.raises(AssertionError, match="initial_prompt"):
        arm.transcribe_kwargs("Άλφα", control={"initial_prompt": "y"})


def test_real_control_carries_neither_prefix_nor_initial_prompt(arm):
    kw = arm.transcribe_kwargs(None)          # control = DA.CONTROL
    assert kw == arm.DA.CONTROL
    assert "hotwords" not in kw


def test_over_budget_hotwords_fail_before_decode(arm):
    tok = FakeTok(per_word=1)
    too_long = " ".join(f"w{i}" for i in range(HOTWORD_TOKEN_BUDGET + 5))
    with pytest.raises(AssertionError, match="budget"):
        arm.transcribe_kwargs(too_long, control={}, tokenizer=tok)
    ok = " ".join(f"w{i}" for i in range(10))
    assert arm.transcribe_kwargs(ok, control={}, tokenizer=tok)["hotwords"] == ok
    # the secondary mode checks against ITS budget
    mid = " ".join(f"w{i}" for i in range(180))
    with pytest.raises(AssertionError, match="budget"):
        arm.transcribe_kwargs(mid, control={}, tokenizer=tok)
    assert arm.transcribe_kwargs(mid, control={}, tokenizer=tok,
                                 budget=200)["hotwords"] == mid


def test_empty_hotwords_string_is_rejected(arm):
    with pytest.raises(AssertionError, match="empty"):
        arm.transcribe_kwargs("   ", control={})


def test_secondary_mode_writes_its_own_file(arm):
    assert arm.dest_path("eval").name == "eval-B-hotwords.json"
    assert arm.dest_path("eval", 200).name == "eval-B-hotwords-200.json"
