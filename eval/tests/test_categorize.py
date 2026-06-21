"""Sanity tests for the heuristic (text-only) error categorizer.

Examples drawn from docs/reference/ui-error-categories.md. The categorizer is
approximate by design (no audio, no NER); these pin the clear cases.
"""
from eval.categorize import categorize


def test_accent():
    assert categorize("δημος", "δήμος") == "accent_tonos"


def test_final_sigma():
    assert categorize("οπωσ", "όπως") in ("final_sigma", "accent_tonos")


def test_word_boundary():
    assert categorize("τονδήμο", "τον δήμο") == "word_boundary"


def test_number_date():
    assert categorize("άρθρο εβδομήντα πέντε", "άρθρο 75") == "number_date"


def test_acronym():
    assert categorize("δ ε υ α", "ΔΕΥΑ") == "acronym_abbreviation"


def test_named_entity():
    assert categorize("στο πετράλωνα", "στα Πετράλωνα") == "named_entity"


def test_homophone():
    assert categorize("να γίνη", "να γίνει") == "homophone"


def test_punctuation_only():
    assert categorize("ναι κυριε προεδρε", "ναι κυριε προεδρε.") == "punctuation_capitalization"
