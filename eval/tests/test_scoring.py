"""Self-test #3 — scoring normalisation.

Greek pairs differing only by accent/punctuation/final-sigma/spacing score
equivalent; a real lexical correction is still caught; overcorrection stays 0
on normalised-equivalent changes.
"""
import pytest

from eval.scoring import greek_normalize, score_pair


@pytest.mark.parametrize(
    "a,b",
    [
        ("όπως,", "οπωσ"),            # accent + punctuation + final sigma
        ("ο δήμος", "ο δημος "),      # accent + trailing space + final sigma
        ("Μαρία;", "μαρια"),          # accent + casing + punctuation
        ("ΖΕΜΕΝΟ", "ζεμενο"),         # casing
        ("γίνει.", " γινει "),        # accent + punctuation + spacing
    ],
)
def test_normalisation_equivalence(a, b):
    assert greek_normalize(a) == greek_normalize(b)


def test_distinct_words_not_equivalent():
    assert greek_normalize("έδωσε") != greek_normalize("έδειξε")


def test_real_correction_is_caught():
    # input has the error, model applies the gold fix
    s = score_pair(input_raw="να γίνη", model_output="να γίνει", gold="να γίνει")
    assert s["normalized_exact"] is True
    assert s["edit_application"] == pytest.approx(1.0)
    assert s["overcorrection"] == pytest.approx(0.0)


def test_unfixed_error_scores_zero_edit_application():
    s = score_pair(input_raw="να γίνη", model_output="να γίνη", gold="να γίνει")
    assert s["normalized_exact"] is False
    assert s["edit_application"] == pytest.approx(0.0)


def test_overcorrection_zero_on_normalised_equivalent():
    # model only changed accents/sigma — gold left it unchanged → no harm
    s = score_pair(input_raw="ο δήμος", model_output="ο δημος", gold="ο δήμος")
    assert s["normalized_exact"] is True
    assert s["overcorrection"] == pytest.approx(0.0)


def test_overcorrection_detected_on_real_harm():
    # gold needed no change, but model rewrote an unchanged token
    s = score_pair(input_raw="ο δήμος είναι", model_output="ο δήμος ήταν", gold="ο δήμος είναι")
    assert s["overcorrection"] > 0.0


def test_pure_deletion_requires_actual_removal():
    # gold deletes the spurious «να»; echoing the input must NOT get credit
    unfixed = score_pair(input_raw="θα να πάμε", model_output="θα να πάμε", gold="θα πάμε")
    assert unfixed["edit_application"] == pytest.approx(0.0)
    fixed = score_pair(input_raw="θα να πάμε", model_output="θα πάμε", gold="θα πάμε")
    assert fixed["edit_application"] == pytest.approx(1.0)


def test_hallucinated_insertion_counts_as_harm():
    # gold left it unchanged; model invented extra words
    s = score_pair(input_raw="ναι", model_output="ναι ευχαριστώ πολύ", gold="ναι")
    assert s["overcorrection"] > 0.0


def test_retaining_error_while_adding_fix_is_not_full_credit():
    # model keeps the wrong token AND appends the right one — partial, not 1.0
    s = score_pair(input_raw="να γίνη", model_output="να γίνη γίνει", gold="να γίνει")
    assert s["edit_application"] < 1.0
