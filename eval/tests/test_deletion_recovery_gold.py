# -*- coding: utf-8 -*-
"""The restore step, as it runs on the gold set: which donor words go back in,
and where.

Both arms are the same vendor on the same audio, so a restored word carries its
own timestamp and belongs at that instant. Everything here is about the three ways
that can go wrong: putting back a word the base already said (a substitution is not
a missing word), putting back a hesitation, and putting back a word the donor
itself was unsure of.

No network, no audio, no cache.
"""
from __future__ import annotations

from eval.deletion_recovery import gold_score as G


def w(raw: str, start: float, logprob: float = -0.001) -> dict:
    return {"raw": raw, "start": start, "end": start + 0.3, "logprob": logprob}


def test_identical_streams_restore_to_themselves():
    clean = [w("ο", 1.0), w("δήμαρχος", 1.4)]
    donor = [w("ο", 1.0), w("δήμαρχος", 1.4)]
    out = G.restore(clean, donor, min_logprob=None)
    assert [x["raw"] for x in out] == ["ο", "δήμαρχος"]


def test_a_donor_only_word_goes_back_at_its_own_time():
    clean = [w("ο", 1.0), w("δήμαρχος", 1.4), w("είπε", 2.2)]
    donor = [w("ο", 1.0), w("δήμαρχος", 1.4), w("μας", 1.9), w("είπε", 2.2)]
    out = G.restore(clean, donor, min_logprob=None)
    assert [x["raw"] for x in out] == ["ο", "δήμαρχος", "μας", "είπε"]


def test_a_hesitation_is_never_restored():
    clean = [w("ο", 1.0), w("είπε", 2.2)]
    donor = [w("ο", 1.0), w("εεε", 1.6), w("είπε", 2.2)]
    out = G.restore(clean, donor, min_logprob=None)
    assert [x["raw"] for x in out] == ["ο", "είπε"]


def test_a_word_the_donor_was_unsure_of_is_gated_out():
    clean = [w("ο", 1.0), w("είπε", 2.2)]
    donor = [w("ο", 1.0), w("μας", 1.6, logprob=-2.5), w("είπε", 2.2)]
    assert [x["raw"] for x in G.restore(clean, donor, min_logprob=None)] == ["ο", "μας", "είπε"]
    assert [x["raw"] for x in G.restore(clean, donor, min_logprob=-1.0)] == ["ο", "είπε"]


def test_a_substitution_is_not_a_missing_word():
    clean = [w("ο", 1.0), w("ήπε", 2.2)]
    donor = [w("ο", 1.0), w("είπε", 2.2)]
    out = G.restore(clean, donor, min_logprob=None)
    assert [x["raw"] for x in out] == ["ο", "ήπε"], "the base keeps its own slot"


def test_restoring_never_drops_a_base_word():
    """And a donor word that lands in a base word's slot stays out.

    The first version of this test expected «και» to come back. It does not, and
    should not: the aligner charges it against «δύο» as a substitution, which the
    frozen rule excludes. What the restore owes is that every base word survives.
    """
    clean = [w("ένα", 1.0), w("δύο", 2.0), w("τρία", 3.0)]
    donor = [w("ένα", 1.0), w("και", 1.5), w("τρία", 3.0)]
    out = G.restore(clean, donor, min_logprob=None)
    assert [x["raw"] for x in out] == ["ένα", "δύο", "τρία"]
    assert all(x["raw"] in [y["raw"] for y in out] for x in clean)


def test_the_grid_of_thresholds_is_frozen_and_ordered():
    # named here so a later edit that adds a "winning" threshold shows up in the diff
    assert G.THRESHOLD_GRID[0][0] == "none"
    assert [n for n, _ in G.THRESHOLD_GRID] == ["none", "p25", "median", "p75"]
