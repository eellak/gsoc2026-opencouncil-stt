"""Contract tests for the descriptive term-utilisation diagnostic."""

from collections import Counter

from eval.controlled_eval.exp_roster_selection import restricted_repair
from eval.controlled_eval.exp_term_utilisation import (
    CAUSE_MISSING_ROSTER,
    CAUSE_PROTECTED_AGREEMENT,
    CAUSE_SURFACE_ABSENT,
    classify_near_miss,
)
from serving_stack.name_repair import RosterContext, rnorm


def context_for(*, aliases=("testterm",)):
    term_id = "term:test"
    term = {
        "id": term_id,
        "canonical": "TESTTERM",
        "aliases": list(aliases),
        "covers": ["Alex Test"],
    }
    ctx = RosterContext(seen_freq=Counter())
    ctx.present = {
        term_id: {"term": term, "persons_in_meeting": ["Alex Test"]}
    }
    for alias in term["aliases"]:
        normalized = rnorm(alias)
        ctx.valid_aliases.add(normalized)
        ctx.alias_surface[(term_id, normalized)] = alias
    return term_id, ctx


def test_term_in_list_one_character_near_miss_fires_and_corrects():
    term_id, ctx = context_for()

    result = classify_near_miss(
        "TESTTERM", "TESTTERX", ctx, protected=set(), term_id=term_id
    )

    assert result["outcome"] == "fired-and-corrected"
    assert result["cause"] is None
    assert result["replacement"] == "TESTTERM"


def test_agreed_wrong_token_is_protected_from_repair():
    term_id, ctx = context_for()

    result = classify_near_miss(
        "TESTTERM",
        "TESTTERX",
        ctx,
        protected={rnorm("TESTTERX")},
        term_id=term_id,
    )

    assert result["outcome"] == "did-not-fire"
    assert result["cause"] == CAUSE_PROTECTED_AGREEMENT
    assert restricted_repair("TESTTERX", ctx, {rnorm("TESTTERX")})[0] == "TESTTERX"


def test_missing_roster_is_a_recorded_no_op():
    term_id = "term:test"

    result = classify_near_miss(
        "TESTTERM", "TESTTERX", RosterContext(), protected=set(), term_id=term_id
    )

    assert result["outcome"] == "did-not-fire"
    assert result["cause"] == CAUSE_MISSING_ROSTER
    assert result["replacement"] is None


def test_missing_inflected_surface_is_distinct_from_distance_miss():
    term_id, ctx = context_for()

    result = classify_near_miss(
        "TESTTERMS", "TESTTERXS", ctx, protected=set(), term_id=term_id
    )

    assert result["outcome"] == "did-not-fire"
    assert result["cause"] == CAUSE_SURFACE_ABSENT
    assert result["cause"] != "phonetic-distance"

