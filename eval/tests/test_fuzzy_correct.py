"""Self-test — deterministic fuzzy name/toponym corrector.

Tests the MECHANICS (Codex-reviewed design): length-dependent edit limits,
leave-already-correct-alone, name-like filtering, phrase replacement, and the
separate match-normalisation (case/accent-insensitive) that still emits the
glossary's canonical spelling.
"""
from eval.fuzzy_correct import _max_edit, _name_like, fuzzy_correct


def test_max_edit_is_length_dependent():
    assert _max_edit(4) == 1 and _max_edit(5) == 1
    assert _max_edit(6) == 2 and _max_edit(9) == 2
    assert _max_edit(10) == 3 and _max_edit(14) == 3


def test_name_like_filters_lowercase_common_words():
    assert _name_like("Τσουπάκης") is True
    assert _name_like("ΔΕΥΑΟ") is True
    assert _name_like("βάση") is False        # common word -> not a fuzzy candidate


def test_applies_phonetic_name_fix():
    # «Ρημάκης» (ρ for δ) -> roster spelling «Δημάκης» (len 7, edit 1)
    out = fuzzy_correct("μίλησε ο Ρημάκης", ["Δημάκης"])
    assert "Δημάκης" in out
    assert "Ρημάκης" not in out


def test_leaves_already_correct_token_unchanged():
    # token differs from the term only by accent -> match-equivalent -> NOT a fix
    out = fuzzy_correct("μίλησε ο Δημακης", ["Δημάκης"])
    assert "Δημακης" in out                    # untouched (no phonetic error)


def test_respects_edit_limit_far_token_not_replaced():
    # «θάλασσα» is far from «Δημάκης» -> beyond any edit limit -> untouched
    out = fuzzy_correct("η θάλασσα είναι", ["Δημάκης"])
    assert out == "η θάλασσα είναι"


def test_name_like_only_excludes_lowercase_glossary_entry():
    # lowercase common-word glossary entry must NOT drive a fuzzy replacement
    out = fuzzy_correct("η βάσι εδώ", ["βάση"], name_like_only=True)
    assert "βάση" not in out                   # not replaced (entry not name-like)


def test_ner_gate_blocks_replacement_outside_spans():
    text = "μίλησε ο Ρημάκης"
    # gate covers only «μίλησε» -> the name is outside the gate -> untouched
    out = fuzzy_correct(text, ["Δημάκης"], allowed_spans=[(0, 6)])
    assert "Ρημάκης" in out and "Δημάκης" not in out
    # gate covers the name span -> replacement allowed
    s = text.index("Ρημάκης")
    out2 = fuzzy_correct(text, ["Δημάκης"], allowed_spans=[(s, s + len("Ρημάκης"))])
    assert "Δημάκης" in out2


def test_replaces_multitoken_phrase_canonically():
    # phonetic phrase error «γεωγιο» (missing ρ) -> canonical «Άγιο Γεώργιο»
    out = fuzzy_correct("στον αγιο γεωγιο πήγαμε", ["Άγιο Γεώργιο"], phrases=True)
    assert "Άγιο Γεώργιο" in out
    assert "γεωγιο" not in out


def test_does_not_touch_accent_only_phrase():
    # match-equivalent (accent/case only) -> left as-is per Codex (not a "fix")
    out = fuzzy_correct("στον αγιο γεωργιο πήγαμε", ["Άγιο Γεώργιο"], phrases=True)
    assert out == "στον αγιο γεωργιο πήγαμε"
