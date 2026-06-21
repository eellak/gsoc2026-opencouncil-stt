"""Self-test — stratified propose/select split for the anti-overfit loop.

The split must be: stratified by category (~frac per category), disjoint,
exhaustive, and deterministic (stable across calls and row order). This is the
core safeguard against the proposer being scored on rows it was shown.
"""
from collections import Counter

from eval.improve_loop import strat_split


def _rows(n):
    return [{"utterance_id": f"u{i}", "category": "a" if i % 2 == 0 else "b"}
            for i in range(n)]


def test_split_disjoint_and_exhaustive():
    rows = _rows(100)
    propose, select = strat_split(rows, frac_propose=0.6)
    pid = {r["utterance_id"] for r in propose}
    sid = {r["utterance_id"] for r in select}
    assert pid & sid == set()                  # no row in both halves
    assert pid | sid == {r["utterance_id"] for r in rows}  # nothing dropped


def test_split_is_stratified_per_category():
    rows = _rows(100)  # 50 of cat a, 50 of cat b
    propose, _ = strat_split(rows, frac_propose=0.6)
    pc = Counter(r["category"] for r in propose)
    assert pc["a"] == 30 and pc["b"] == 30     # 60% of each category


def test_split_is_deterministic_and_order_independent():
    rows = _rows(100)
    p1, s1 = strat_split(rows, frac_propose=0.6)
    p2, s2 = strat_split(list(reversed(rows)), frac_propose=0.6)
    assert {r["utterance_id"] for r in p1} == {r["utterance_id"] for r in p2}
    assert {r["utterance_id"] for r in s1} == {r["utterance_id"] for r in s2}
