#!/usr/bin/env python3
"""The taxonomy's local DP copy must agree with the frozen `msa.oracle_select`.

`exp_majority_taxonomy.oracle_tables` re-implements the oracle recurrence so that
`msa.py` stays byte-identical (fusion_lab hashes it into the alignment cache key).
That duplication is only safe if it is pinned by a test.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.exp_majority_taxonomy import (   # noqa: E402
    class_of, oracle_tables, relation_of, support, surface_suffix_neighbor,
    w_cost_if, w_tables,
)
from eval.controlled_eval.msa import oracle_columns        # noqa: E402
from eval.controlled_eval.scoring import edist             # noqa: E402

VOCAB = ["α", "β", "γ", "δ", "ε"]


def _rand_case(rng):
    n = rng.randint(1, 7)
    cols = []
    for _ in range(n):
        col = tuple(rng.choice(VOCAB + [None]) for _ in range(3))
        if all(e is None for e in col):
            col = (rng.choice(VOCAB), None, None)
        cols.append(col)
    ref = [rng.choice(VOCAB) for _ in range(rng.randint(0, 7))]
    return cols, ref


def test_local_dp_cost_matches_frozen_oracle():
    rng = random.Random(11)
    for _ in range(300):
        cols, ref = _rand_case(rng)
        _, _, star = oracle_tables(cols, ref)
        assert star == edist(oracle_columns(cols, ref), ref)


def test_every_column_has_a_supported_candidate():
    """Some candidate of every column must lie on an optimal path."""
    rng = random.Random(13)
    for _ in range(200):
        cols, ref = _rand_case(rng)
        F, B, star = oracle_tables(cols, ref)
        for i, col in enumerate(cols):
            toks = {e for e in col if e is not None}
            eps_ok = any(e is None for e in col) and any(
                F[i][j] + B[i + 1][j] == star for j in range(len(ref) + 1))
            assert eps_ok or any(support(F, B, ref, i, e, star)["any"] for e in toks)


def test_w_cost_if_reproduces_edit_distance():
    rng = random.Random(17)
    for _ in range(200):
        hyp = [rng.choice(VOCAB) for _ in range(rng.randint(1, 8))]
        ref = [rng.choice(VOCAB) for _ in range(rng.randint(0, 8))]
        F, B, ed = w_tables(hyp, ref)
        assert ed == edist(hyp, ref)
        p = rng.randrange(len(hyp))
        assert w_cost_if(F, B, ref, p, hyp[p]) == ed
        for e in VOCAB:
            alt = hyp[:p] + [e] + hyp[p + 1:]
            assert w_cost_if(F, B, ref, p, e) == edist(alt, ref)
        assert w_cost_if(F, B, ref, p, None) == edist(hyp[:p] + hyp[p + 1:], ref)


def test_surface_suffix_neighbor_shape():
    assert surface_suffix_neighbor("δημαρχου", "δημαρχος")
    assert not surface_suffix_neighbor("δημος", "νομος")
    assert not surface_suffix_neighbor("και", "κι")


def test_class_partition_is_total():
    flags = {"any_numeric": False, "any_protocol": False, "any_entity": False,
             "pair_role": "content_content"}
    for rel in ("alignment_artifact", "no_target", "phonemic_key_equivalent",
                "surface_suffix_neighbor", "unrelated_substitution"):
        assert class_of(rel, flags)
    assert relation_of("α", None, False) == "no_target"
    assert relation_of("α", "β", True) == "alignment_artifact"
