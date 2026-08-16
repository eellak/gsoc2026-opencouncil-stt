"""Frozen tests for the three-way alignment and per-column vote of wayfinder #22.

The five cases below are the ones Codex job 8112dc72 named as the evaluation that the
MSA is implemented correctly, plus the exhaustive small-case check against brute force.
If any of these breaks, arm W is not measuring what the report says it measures.

    python3 -m pytest eval/controlled_eval/test_msa.py -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/harold/opencouncil-fine-tuning")))
from eval.controlled_eval.msa import (  # noqa: E402
    align3, columns_cost, compose, oracle_columns, progressive_align, vote_column,
)


def rows(cols):
    """Strip epsilon from each aligned row; must recover the inputs exactly."""
    return [[e for e in (c[i] for c in cols) if e is not None] for i in range(3)]


# ------------------------------------------------------------------ named cases
def test_identity():
    a = ["a", "b", "c"]
    cols = align3(a, list(a), list(a))
    assert cols == [("a", "a", "a"), ("b", "b", "b"), ("c", "c", "c")]
    assert columns_cost(cols) == 0


def test_shared_insertion():
    cols = align3(["a", "c"], ["a", "b", "c"], ["a", "b", "c"])
    assert (None, "b", "b") in cols
    toks, _ = compose(cols, pivot=0)
    assert toks == ["a", "b", "c"]


def test_shared_deletion():
    cols = align3(["a", "b", "c"], ["a", "c"], ["a", "c"])
    assert ("b", None, None) in cols
    toks, _ = compose(cols, pivot=0)
    assert toks == ["a", "c"]


def test_occupancy_tie_emits_a_token_not_epsilon():
    """(epsilon, x, y): two systems heard a word, one heard silence.

    A flat vote makes this epsilon because x != y. That is the deletion failure the
    hierarchical vote exists to prevent, and it is the single correction that
    Codex's review turned on.
    """
    cols = align3(["a", "c"], ["a", "x", "c"], ["a", "y", "c"])
    mid = [c for c in cols if c[0] is None]
    assert len(mid) == 1 and set(mid[0][1:]) == {"x", "y"}
    tok, why = vote_column(mid[0], pivot=0)
    assert tok is not None, "occupancy majority must win before identity"
    assert why.startswith("tie_")
    toks, _ = compose(cols, pivot=0)
    assert len(toks) == 3


def test_tie_break_prefers_the_pivot_then_frozen_priority():
    col = ("x", "y", "z")
    assert vote_column(col, pivot=1) == ("y", "tie_pivot")
    assert vote_column((None, "y", "z"), pivot=0) == ("y", "tie_priority")
    assert vote_column(("x", "x", "z"), pivot=2) == ("x", "majority")
    assert vote_column((None, None, "z"), pivot=2) == (None, "epsilon")


# ------------------------------------------------------------ exhaustive vs brute
def brute_cost(a, b, c):
    """Minimum sum-of-pairs cost by enumerating every alignment, for tiny inputs."""
    from functools import lru_cache

    def pair(x, y):
        if x is None and y is None:
            return 0
        if x is None or y is None:
            return 1
        return 0 if x == y else 1

    @lru_cache(maxsize=None)
    def go(i, j, k):
        if i == len(a) and j == len(b) and k == len(c):
            return 0
        best = None
        for m in range(1, 8):
            ni = i + (1 if m & 1 else 0)
            nj = j + (1 if m & 2 else 0)
            nk = k + (1 if m & 4 else 0)
            if ni > len(a) or nj > len(b) or nk > len(c):
                continue
            ea = a[i] if m & 1 else None
            eb = b[j] if m & 2 else None
            ec = c[k] if m & 4 else None
            v = pair(ea, eb) + pair(ea, ec) + pair(eb, ec) + go(ni, nj, nk)
            if best is None or v < best:
                best = v
        return best

    return go(0, 0, 0)


def test_exhaustive_small_cases():
    alpha = ["x", "y"]
    seqs = [list(s) for n in range(4) for s in itertools.product(alpha, repeat=n)]
    checked = 0
    for a in seqs:
        for b in seqs:
            for c in seqs:
                cols = align3(a, b, c, band=8)
                assert columns_cost(cols) == brute_cost(tuple(a), tuple(b), tuple(c)), \
                    (a, b, c, cols)
                assert rows(cols) == [a, b, c]
                assert all(any(e is not None for e in col) for col in cols)
                checked += 1
    assert checked == len(seqs) ** 3


def test_progressive_recovers_inputs_in_every_ordering():
    a, b, c = ["a", "b", "d"], ["a", "x", "b", "c"], ["b", "c", "d"]
    for order in itertools.permutations((0, 1, 2)):
        cols = progressive_align([a, b, c], order)
        assert rows(cols) == [a, b, c], order
        assert all(any(e is not None for e in col) for col in cols)


def test_progressive_is_never_better_than_exact():
    a, b, c = ["a", "b", "d"], ["a", "x", "b", "c"], ["b", "c", "d"]
    exact = columns_cost(align3(a, b, c))
    for order in itertools.permutations((0, 1, 2)):
        assert columns_cost(progressive_align([a, b, c], order)) >= exact


# ----------------------------------------------------------------------- oracle
def test_oracle_picks_the_reference_where_any_system_has_it():
    cols = [("a", "a", "a"), ("q", "b", "z"), ("c", "c", "c")]
    assert oracle_columns(cols, ["a", "b", "c"]) == ["a", "b", "c"]


def test_oracle_may_drop_a_column_that_only_hurts():
    cols = [("a", "a", "a"), ("q", None, "z")]
    assert oracle_columns(cols, ["a"]) == ["a"]


def test_oracle_cannot_invent_a_token():
    cols = [("a", "a", "a")]
    out = oracle_columns(cols, ["a", "b"])
    assert out == ["a"]


def test_oracle_beats_or_matches_the_vote_everywhere():
    from eval.controlled_eval.scoring import edist
    cases = [
        (["a", "b", "c"], ["a", "x", "c"], ["a", "b", "z"], ["a", "b", "c"]),
        (["a"], ["a", "b"], ["a", "b"], ["a", "b"]),
        (["q", "w"], ["e", "r"], ["t", "y"], ["q", "r"]),
    ]
    for a, b, c, ref in cases:
        cols = align3(a, b, c)
        voted, _ = compose(cols, pivot=0)
        orc = oracle_columns(cols, ref)
        assert edist(orc, ref) <= edist(voted, ref), (a, b, c, ref, voted, orc)
