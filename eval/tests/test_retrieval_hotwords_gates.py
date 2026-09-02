"""Tests for the retrieval-hotwords arm's frozen advance rule (spec section 3.4).

The gates are a pure function of already-measured quantities, so they are tested
here without decoding anything. Every boundary in the spec table is exercised on
both sides, because a gate that is off by one comparison operator silently changes
the verdict of an experiment that costs 76 CPU-minutes and cannot be rerun with a
different rule afterwards.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.serving_stack.retrieval_hotwords_arm import (  # noqa: E402
    CFG, DELETION_PARTIAL_MARGIN, NAME_GAIN_MIN, NONINFERIORITY_MARGIN,
    block_key, evaluate_gates, name_domination)


def passing(**over):
    """A measurement set where every gate passes; override one to test a boundary."""
    m = dict(
        wer_delta=-0.0020, wer_hi=-0.0005,
        del_point=-0.0030, del_hi=-0.0010,
        ins_delta=0.0001,
        name_gain=20, precision_delta=0.01, oor_delta=0,
        loo_reversal=False, name_dom_share=0.20,
    )
    m.update(over)
    return m


# ------------------------------------------------------------------ frozen values

def test_the_config_matches_the_values_frozen_in_the_spec():
    """These were written down on 2026-08-27 before any decode. If this test is
    edited, the spec table and the ledger caveat must be edited in the same change."""
    # STRICT: LOOSE merges beta/upsilon and would skip a misspelled inflected name as
    # already-correct against its own recorded alias. Corrected 2026-08-27 before any
    # decode; see the spec table and the ledger caveat, changed in the same commit.
    assert CFG.loose is False
    assert CFG.max_distance == 1
    assert CFG.min_skeleton_len == 4
    assert CFG.k_max == 3
    assert CFG.token_budget == 24
    assert CFG.include_exact is False
    assert NAME_GAIN_MIN == 18
    assert NONINFERIORITY_MARGIN == 0.0010
    assert DELETION_PARTIAL_MARGIN == 0.0028


def test_the_cluster_block_is_city_and_meeting():
    """Meeting slugs repeat across cities; a meeting-only block merges two cities
    into one cluster and understates the interval."""
    assert block_key({"city": "argos", "meeting_id": "apr30_2_2026"}) == "argos/apr30_2_2026"
    assert block_key({"city": "orestiada", "meeting_id": "apr30_2_2026"}) != \
        block_key({"city": "argos", "meeting_id": "apr30_2_2026"})


# ------------------------------------------------------------------------ verdicts

def test_all_eight_gates_pass_gives_advance():
    out = evaluate_gates(passing())
    assert out["verdict"] == "ADVANCE"
    assert all(out["gates"][k] is True for k in out["gates"])


def test_a_halved_deletion_penalty_is_partial_not_advance():
    """The case the PARTIAL tier exists for: the penalty is smaller but not gone."""
    out = evaluate_gates(passing(del_point=-0.0002, del_hi=+0.0020))
    assert out["gates"]["3_deletion_upper"] is False
    assert out["verdict"] == "PARTIAL"


def test_partial_requires_the_deletion_point_estimate_at_or_below_zero():
    out = evaluate_gates(passing(del_point=+0.0010, del_hi=+0.0020))
    assert out["verdict"] == "STOP"


def test_partial_does_not_extend_past_its_margin():
    out = evaluate_gates(passing(del_point=-0.0001, del_hi=DELETION_PARTIAL_MARGIN))
    assert out["verdict"] == "PARTIAL"
    out = evaluate_gates(passing(del_point=-0.0001,
                                 del_hi=DELETION_PARTIAL_MARGIN + 1e-9))
    assert out["verdict"] == "STOP"


def test_a_failed_name_gate_blocks_partial_too():
    """PARTIAL is a place for a real but incomplete deletion result, not a consolation
    prize for an arm that did nothing for names."""
    out = evaluate_gates(passing(name_gain=0, del_point=-0.0002, del_hi=+0.0020))
    assert out["verdict"] == "STOP"


# ------------------------------------------------------------------- gate boundaries

def test_gate_boundaries_are_on_the_side_the_spec_says():
    assert evaluate_gates(passing(wer_delta=0.0))["gates"]["1_wer_point"] is False
    assert evaluate_gates(passing(wer_delta=-1e-9))["gates"]["1_wer_point"] is True

    assert evaluate_gates(passing(wer_hi=NONINFERIORITY_MARGIN))["gates"]["2_wer_upper"] is False
    assert evaluate_gates(passing(wer_hi=NONINFERIORITY_MARGIN - 1e-9))["gates"]["2_wer_upper"] is True

    assert evaluate_gates(passing(del_hi=0.0))["gates"]["3_deletion_upper"] is True
    assert evaluate_gates(passing(del_hi=1e-9))["gates"]["3_deletion_upper"] is False

    assert evaluate_gates(passing(ins_delta=0.0005))["gates"]["4_insertion"] is False
    assert evaluate_gates(passing(ins_delta=0.0004999))["gates"]["4_insertion"] is True

    assert evaluate_gates(passing(name_gain=NAME_GAIN_MIN - 1))["gates"]["5_name_gain"] is False
    assert evaluate_gates(passing(name_gain=NAME_GAIN_MIN))["gates"]["5_name_gain"] is True

    assert evaluate_gates(passing(name_dom_share=0.25))["gates"]["8_name_domination"] is True
    assert evaluate_gates(passing(name_dom_share=0.2500001))["gates"]["8_name_domination"] is False


def test_precision_and_out_of_roster_share_one_gate():
    assert evaluate_gates(passing(precision_delta=-1e-9))["gates"]["6_name_precision"] is False
    assert evaluate_gates(passing(oor_delta=1))["gates"]["6_name_precision"] is False
    assert evaluate_gates(passing(oor_delta=-1))["gates"]["6_name_precision"] is True


# ------------------------------------- gates 7 and 8 are n/a without a name gain

def test_stability_gates_are_not_applicable_when_there_is_no_name_gain():
    """'25 % of a quantity indistinguishable from zero' is undefined. Reported n/a,
    never passed -- the trap section 5 of the spec already names for its own gate 6."""
    out = evaluate_gates(passing(name_gain=0))
    assert out["gates"]["7_loo_reversal"] is None
    assert out["gates"]["8_name_domination"] is None
    assert out["verdict"] == "STOP"


def test_name_domination_is_computed_on_the_name_gain_not_on_wer():
    """One meeting supplying more than a quarter of the extra correct mentions fails,
    however the WER moved."""
    a = {"w1": {"matched": 10}, "w2": {"matched": 10}, "w3": {"matched": 10}}
    b = {"w1": {"matched": 30}, "w2": {"matched": 12}, "w3": {"matched": 12}}
    wids = ["w1", "w2", "w3"]
    blocks = {"w1": "argos/m1", "w2": "argos/m2", "w3": "orestiada/m3"}

    dom = name_domination(b, a, wids, blocks)
    assert dom["name_gain"] == 24
    assert dom["max_share"] == 20 / 24
    assert dom["max_unit"] in ("w1", "argos/m1")
    assert dom["dominated"] is True


def test_name_domination_reports_none_when_the_gain_is_not_positive():
    a = {"w1": {"matched": 10}}
    b = {"w1": {"matched": 10}}
    dom = name_domination(b, a, ["w1"], {"w1": "argos/m1"})
    assert dom["name_gain"] == 0
    assert dom["max_share"] is None
    assert dom["dominated"] is None


def test_name_domination_checks_meetings_as_well_as_windows():
    """Two windows of the same meeting can each stay under 25 % while the meeting
    they share does not."""
    # gains 3, 3, 3, 3, 4 over a total of 16: no single window reaches 25 %
    # (the largest is 4/16 = 0.25 exactly, which passes), but w1 and w2 share a
    # meeting and together supply 6/16 = 37.5 %.
    a = {f"w{i}": {"matched": 10} for i in range(1, 6)}
    b = {"w1": {"matched": 13}, "w2": {"matched": 13}, "w3": {"matched": 13},
         "w4": {"matched": 13}, "w5": {"matched": 14}}
    wids = list(a)
    blocks = {"w1": "argos/m1", "w2": "argos/m1", "w3": "argos/m2",
              "w4": "argos/m3", "w5": "orestiada/m4"}
    dom = name_domination(b, a, wids, blocks)
    assert dom["name_gain"] == 16
    assert max(b[w]["matched"] - a[w]["matched"] for w in wids) / 16 <= 0.25
    assert dom["max_share"] == 6 / 16      # the shared meeting, not either window
    assert dom["max_unit"] == "argos/m1"
    assert dom["dominated"] is True
