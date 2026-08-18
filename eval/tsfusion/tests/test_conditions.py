import hashlib
from pathlib import Path

from eval.controlled_eval.msa import compose, vote_column
from eval.tsfusion.conditions import CONF_MIN, c3_eligible, sdi
from eval.tsfusion.speakers import SpeakerCall
from eval.tsfusion.timing import ColumnTime

ROOT = Path(__file__).resolve().parents[3]

# The frozen alignment module keys an 18 MB cache. If this changes, every number on
# the page belongs to a different alignment than the rest of the project's.
MSA_SHA16 = "3751fe5a13320e2b"


def test_the_frozen_msa_module_is_untouched():
    got = hashlib.sha256(
        (ROOT / "eval/controlled_eval/msa.py").read_bytes()).hexdigest()[:16]
    assert got == MSA_SHA16, f"msa.py changed: {got}"


def test_sdi_totals_agree_with_the_scorer():
    from eval.controlled_eval.scoring import edist
    ref = "ενα δυο τρια τεσσερα πεντε".split()
    hyp = "ενα τρια τεσσερα εξι πεντε".split()
    r = sdi(ref, hyp)
    assert r["errors"] == edist(ref, hyp)
    assert r["S"] + r["D"] + r["I"] == r["errors"]
    assert r["wer"] == r["errors"] / len(ref)


def ct(method="observed", prov="observed_word", system="soniox"):
    return ColumnTime(time_start=1.0, time_end=1.3, time_method=method,
                      sources={system: {"start": 1.0, "end": 1.3,
                                        "provenance": prov}})


def sc(state="named", frac=1.0, mult=1):
    return SpeakerCall(state=state, speaker="A", overlap_fraction=frac,
                       multiplicity=mult)


def test_the_target_case_qualifies():
    col = (None, "θαλασσα", None)
    ok, tok, why = c3_eligible(col, ct(), sc(), {"soniox": 0.99})
    assert ok and tok == "θαλασσα" and why["fail"] is None


def test_a_two_of_three_column_is_never_eligible():
    # the occupancy vote already keeps these, so acting on them is a no-op
    ok, _, why = c3_eligible(("θαλασσα", "θαλασσα", None), ct(), sc(), {"soniox": 0.99})
    assert not ok and "occupancy" in why["fail"]


def test_a_scribe_only_column_is_never_eligible():
    # Scribe has no timestamps at all, so there is no anchor to reason from.
    # Refused before the timing checks are even consulted.
    ok, _, why = c3_eligible(("θαλασσα", None, None), ct(), sc(), {})
    assert not ok and "Scribe" in why["fail"]


def test_inferred_timing_can_never_activate_the_rule():
    for method in ("bracketed", "extrapolated", "unplaced"):
        ok, _, why = c3_eligible((None, "θαλασσα", None), ct(method), sc(),
                                 {"soniox": 0.99})
        assert not ok and method in why["fail"]


def test_a_proportionally_split_interval_can_never_activate_the_rule():
    ok, _, why = c3_eligible((None, "θαλασσα", None),
                             ct(prov="derived_within_raw_word"), sc(),
                             {"soniox": 0.99})
    assert not ok and "derived_within_raw_word" in why["fail"]


def test_overlap_and_partial_containment_are_refused():
    for call in (sc("overlap", 1.0, 2), sc("ambiguous", 0.6, 1),
                 sc("named", 0.7, 1), sc("non_speech", 0.0, 0)):
        ok, _, why = c3_eligible((None, "θαλασσα", None), ct(), call,
                                 {"soniox": 0.99})
        assert not ok, why


def test_low_confidence_is_refused_at_the_stated_threshold():
    ok, _, _ = c3_eligible((None, "θαλασσα", None), ct(), sc(),
                           {"soniox": CONF_MIN - 0.001})
    assert not ok
    ok2, _, _ = c3_eligible((None, "θαλασσα", None), ct(), sc(),
                            {"soniox": CONF_MIN})
    assert ok2
    # no confidence at all is a refusal, not a pass
    assert not c3_eligible((None, "θαλασσα", None), ct(), sc(), {})[0]


def test_the_rule_only_ever_targets_columns_the_vote_drops():
    # every eligible column has exactly one token, and `vote_column` returns
    # epsilon for those, so C3 can only ever ADD words, never remove or swap one
    col = (None, "θαλασσα", None)
    assert vote_column(col, pivot=1)[0] is None
    toks, _ = compose([col], pivot=1)
    assert toks == []
