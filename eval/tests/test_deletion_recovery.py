# -*- coding: utf-8 -*-
"""The one thing the deletion-recovery screen has to get right: which words the
verbatim arm emitted and the clean arm did not.

A substitution is not a recoverable word -- both arms said something in that slot
and one of them is wrong, which is a different problem with a different fix. Only
an insertion relative to clean is a candidate for putting back. Getting that
distinction wrong would inflate every count in the screen.

No network, no audio, no cache.
"""
from __future__ import annotations

from eval.deletion_recovery import screen as S


def toks(s: str) -> list[str]:
    from eval.controlled_eval.scoring import wtoks
    return wtoks(s)


def test_identical_streams_have_nothing_to_recover():
    a = toks("ο δήμαρχος είπε ότι")
    assert S.only_in_donor(a, list(a)) == []


def test_a_word_the_donor_adds_is_a_candidate():
    base = toks("ο δήμαρχος είπε ότι")
    donor = toks("ο δήμαρχος μας είπε ότι")
    assert S.only_in_donor(base, donor) == ["μας"]


def test_a_filler_the_donor_adds_is_a_candidate_too_and_is_labelled_later():
    base = toks("ο δήμαρχος είπε ότι")
    donor = toks("ο δήμαρχος εεε είπε ότι")
    assert S.only_in_donor(base, donor) == ["εεε"]


def test_a_substitution_is_not_a_recoverable_word():
    base = toks("ο δήμαρχος είπε ότι")
    donor = toks("ο δήμαρχος ήπε ότι")
    assert S.only_in_donor(base, donor) == []


def test_several_additions_come_back_in_order():
    base = toks("συνεδρίαση του δημοτικού συμβουλίου")
    donor = toks("η συνεδρίαση του δημοτικού μας συμβουλίου σήμερα")
    assert S.only_in_donor(base, donor) == ["η", "μας", "σημερα"]


def test_an_empty_base_makes_every_donor_word_a_candidate():
    donor = toks("δύο λέξεις")
    assert S.only_in_donor([], donor) == donor


def test_the_oracle_does_not_credit_a_token_that_matches_somewhere_else():
    """The oracle's job is the ceiling of a selection rule, so it may only keep an
    addition that actually helps. Counting any unconsumed occurrence of the token
    anywhere in the reference credits restorations that make the transcript worse:
    here the reference says «α α β», the base said «α β», and the donor adds a
    trailing «α». The token is in the reference and unconsumed -- but putting it
    back at the END raises the distance from 1 to 2.
    """
    base = ["α", "β"]
    donor = ["α", "β", "α"]
    ref = ["α", "α", "β"]
    out, kept = S.restore(base, donor, oracle_ref=ref)
    assert kept == 0, "an addition that does not reduce the distance is not oracle-kept"
    assert out == base


def test_the_oracle_keeps_an_addition_that_does_reduce_the_distance():
    base = ["ο", "δήμαρχος", "είπε"]
    donor = ["ο", "δήμαρχος", "μας", "είπε"]
    ref = ["ο", "δήμαρχος", "μας", "είπε"]
    out, kept = S.restore(base, donor, oracle_ref=ref)
    assert kept == 1 and out == ref


def test_the_frozen_filler_rule_is_the_one_already_in_use():
    # imported, never retyped: a second filler lexicon would silently make the
    # recovered-word count disagree with every nofillers number in the project
    assert S.is_filler("εεε") and S.is_filler("μμμ") and S.is_filler("χμμ")
    assert not S.is_filler("μας") and not S.is_filler("ο") and not S.is_filler("ναι")
