"""Safety tests for arm E post-hoc roster name repair (scripts/serving_stack).

Run: .venv-eval/bin/python -m pytest eval/tests/test_name_repair.py -q

Uses the real frozen term files (research/ds_wer/terms) and rosters
(data/pii/rosters_full.json); no network, no cache dependency.

Note on the plan's example case: the spec suggested 'σελης -> Σελλής' as a
known-good firing, but the frozen conservative rule forbids any correction of
tokens shorter than 6 characters and the Step-0 audit recorded that case as
'no_candidate'. It is tested here as a MUST-NOT-fire budget case; the MUST-fire
case is a real audit firing (αδραχτας -> αδρακτας, argos/apr24_2026).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from serving_stack.name_repair import (  # noqa: E402
    RosterContext, build_context, repair, rnorm, select, suffix_family_abstain,
)

TERMS_DIR = ROOT / "research/ds_wer/terms"
ROSTERS = ROOT / "data/pii/rosters_full.json"


def city_terms(city):
    data = json.loads((TERMS_DIR / f"{city}.json").read_text())
    return [t for t in data["terms"] if t["klass"] == "person_surname"]


def ctx_for(city, meeting):
    rosters = json.loads(ROSTERS.read_text())
    return build_context(city_terms(city), rosters[f"{city}/{meeting}"])


# ---------------------------------------------------------------- no-op cases

def test_noop_without_roster():
    text = "Ο Αδραχτάς είναι παρών, και η Μιχαηλίδου επίσης."
    for ctx in (None, build_context(city_terms("argos"), []), RosterContext()):
        res = repair(text, ctx)
        assert res.text == text          # byte-identical
        assert res.changes == []


def test_never_touches_exact_roster_form():
    ctx = ctx_for("argos", "apr24_2026")
    # valid alias surfaces, with accents/case as they appear in output text
    text = "Ο Αδρακτάς παρών, ο Λιόλιος απών, η Ξηνταροπούλου παρούσα."
    res = repair(text, ctx)
    assert res.text == text
    assert res.changes == []
    assert select(rnorm("Αδρακτάς"), ctx, text)["decision"] == "abstain_already_valid"


# ---------------------------------------------------------------- must fire

def test_known_good_case_fires():
    # real audit firing: hyp 'αδραχτας' -> roster 'αδρακτας' (d=1), argos/apr24_2026
    ctx = ctx_for("argos", "apr24_2026")
    text = "Παρακαλώ, Αδραχτάς παρών, συνεχίζουμε."
    res = repair(text, ctx)
    assert len(res.changes) == 1
    ch = res.changes[0]
    assert ch["original"] == "Αδραχτάς"
    assert ch["replacement"] == "Αδρακτας"      # case-matched surface form
    assert ch["term"] == "person:αδρακτας"
    assert res.text == "Παρακαλώ, Αδρακτας παρών, συνεχίζουμε."


def test_idempotence():
    ctx = ctx_for("argos", "apr24_2026")
    text = "Αδραχτάς παρών, Λιωλιος παρών, ο Σελής απών."
    once = repair(text, ctx)
    twice = repair(once.text, ctx)
    assert twice.text == once.text
    assert twice.changes == []


# ------------------------------------------------- the amendment guard cases

def test_michailidou_must_not_fire():
    # the audit's only correct->wrong failure: out-of-roster feminine variant
    ctx = ctx_for("orestiada", "jan21_2026")
    text = "Η κυρία Μιχαηλίδου είπε ότι συμφωνεί."
    res = repair(text, ctx)
    assert res.changes == []
    assert res.text == text
    # and it is the guard doing the blocking, not margin/tie/signal
    sel = select(rnorm("Μιχαηλίδου"), ctx, text)
    assert sel["decision"] == "abstain_suffix_family"


def test_suffix_guard_shape():
    assert suffix_family_abstain("μιχαηλιδου", "μιχαηλιδη")
    assert suffix_family_abstain("τσομπανιδου", "τσομπανιδη")
    # d=2 case the guard must NOT block (difference not a productive suffix)
    assert not suffix_family_abstain("ξινταροπουλο", "ξηνταροπουλου")
    assert not suffix_family_abstain("καραβιντασ", "καραβιδασ")


# -------------------------------------------------- budget / normalization

def test_short_token_never_corrected():
    # σελης (5 chars) is below the length budget: frozen rule forbids it
    ctx = ctx_for("argos", "apr24_2026")
    text = "Σελής Χαράλαμπος παρών."
    res = repair(text, ctx)
    assert res.text == text and res.changes == []
    assert select(rnorm("Σελής"), ctx, text)["decision"] == "no_candidate"


def test_final_sigma_and_tonos():
    ctx = ctx_for("argos", "apr24_2026")
    # tonos on the wrong token must not stop the match; replacement keeps final ς
    res = repair("Ο Λιώλιος έχει τον λόγο.", ctx)
    assert len(res.changes) == 1
    assert res.changes[0]["replacement"] == "Λιολιος"
    assert res.changes[0]["replacement"].endswith("ς")
    assert res.text == "Ο Λιολιος έχει τον λόγο."


def test_uppercase_preserved():
    ctx = ctx_for("argos", "apr24_2026")
    res = repair("ΑΔΡΑΧΤΑΣ ΠΑΡΩΝ", ctx)
    assert len(res.changes) == 1
    assert res.changes[0]["replacement"] == "ΑΔΡΑΚΤΑΣ"
    assert res.text == "ΑΔΡΑΚΤΑΣ ΠΑΡΩΝ"


def test_punctuation_preserved_around_change():
    ctx = ctx_for("argos", "apr24_2026")
    res = repair("(Αδραχτάς;) ναι — «Αδραχτάς».", ctx)
    assert res.text == "(Αδρακτας;) ναι — «Αδρακτας»."


def test_common_word_abstains():
    from collections import Counter
    rosters = json.loads(ROSTERS.read_text())
    terms = city_terms("argos")
    freq = Counter({rnorm("αδραχτας"): 10})
    ctx = build_context(terms, rosters["argos/apr24_2026"], seen_freq=freq)
    res = repair("Αδραχτάς παρών.", ctx)
    assert res.changes == []
    assert select(rnorm("Αδραχτάς"), ctx, "x")["decision"] == "abstain_common"
