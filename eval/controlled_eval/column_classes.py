#!/usr/bin/env python3
"""What kind of decision each MSA column is, and which arm is allowed to touch it.

The partition is Codex job 55293f6b's, adopted before any WER number was computed. It
replaces a first draft that folded [x, x, y] into the homophone / near-character
classes; that draft would have let an arm overturn a real 2-of-3 token majority, which
is the one thing the per-column vote of `exp-2026-08-16-composition-over-selection`
got right.

CLASSES (a partition, decided by occupancy and identity only):

  invalid            no system has a token (cannot occur)
  singleton          [x, eps, eps] - one system heard a word, two heard nothing
  two_present_same   [x, x, eps] - two systems agree, one heard nothing
  agree              [x, x, x]
  exact_2_of_3       [x, x, y] - a token majority. SETTLED, no arm may touch it.
  unresolved_two     [x, y, eps] - occupancy settled 2:1, identity tied
  unresolved_three   [x, y, z] - three different words

PROPERTIES (flags, not classes - a column can carry several):

  strict_homophone   every distinct present token shares a STRICT phonemic key
  loose_homophone    ... shares a LOOSE key (implied by strict)
  partial_homophone  only some pair of the distinct tokens shares a strict key
  max_char_dist      the largest pairwise character Levenshtein among distinct tokens
  split_merge        this column is part of a token-BOUNDARY disagreement: two
                     systems spell the same string across a neighbouring pair of
                     columns but split it differently. Independent per-column edits
                     cannot repair that structure and can worsen it, so both arms
                     abstain here.

ELIGIBILITY, frozen:

  H (homophones)   unresolved_two or unresolved_three, strict_homophone, not
                   split_merge. Epsilon is NEVER offered as a candidate: occupancy
                   was already settled by the vote, so H can only substitute, never
                   delete. Loose-only homophones are a declared secondary variant.
  C (char vote)    unresolved_three, not strict_homophone, max_char_dist <= 2, not
                   split_merge. unresolved_two is excluded on purpose: two strings
                   have no character majority, so a "vote" there is candidate
                   selection wearing a vote's clothes.
  H wins any overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.controlled_eval.greek_phonetics import phon      # noqa: E402
from eval.controlled_eval.scoring import edist             # noqa: E402

CLASSES = ("invalid", "singleton", "two_present_same", "agree", "exact_2_of_3",
           "unresolved_two", "unresolved_three")


def column_class(col) -> str:
    present = [e for e in col if e is not None]
    n, distinct = len(present), set(present)
    if n == 0:
        return "invalid"
    if n == 1:
        return "singleton"
    if len(distinct) == 1:
        return "agree" if n == 3 else "two_present_same"
    if n == 3 and len(distinct) == 2:
        return "exact_2_of_3"
    return "unresolved_three" if n == 3 else "unresolved_two"


def column_flags(col) -> dict:
    distinct = sorted({e for e in col if e is not None})
    if len(distinct) < 2:
        return {"strict_homophone": False, "loose_homophone": False,
                "partial_homophone": False, "max_char_dist": 0}
    ks = {phon(t) for t in distinct}
    kl = {phon(t, loose=True) for t in distinct}
    pairs = [(a, b) for i, a in enumerate(distinct) for b in distinct[i + 1:]]
    return {
        "strict_homophone": len(ks) == 1,
        "loose_homophone": len(kl) == 1,
        "partial_homophone": len(ks) > 1 and any(phon(a) == phon(b) for a, b in pairs),
        "max_char_dist": max(edist(list(a), list(b)) for a, b in pairs),
    }


def split_merge_columns(cols) -> set[int]:
    """Indices of columns caught in a token-boundary disagreement.

    Two systems spell the same character string across one adjacent pair of columns
    but cut it in different places, e.g. (στο, σ, eps) followed by (eps, το, στο).
    Voting the two columns independently cannot reconstruct either spelling.
    """
    bad: set[int] = set()
    for i in range(len(cols) - 1):
        joined = []
        for s in range(3):
            a = cols[i][s] or ""
            b = cols[i + 1][s] or ""
            joined.append((a + b, (cols[i][s], cols[i + 1][s])))
        for x in range(3):
            for y in range(x + 1, 3):
                jx, sx = joined[x]
                jy, sy = joined[y]
                if jx and jx == jy and sx != sy:
                    bad.add(i)
                    bad.add(i + 1)
    return bad


def eligibility(cols, loose: bool = False) -> dict[int, str]:
    """Column index -> "H" or "C" for the columns an arm may touch."""
    sm = split_merge_columns(cols)
    out: dict[int, str] = {}
    for i, col in enumerate(cols):
        if i in sm:
            continue
        klass = column_class(col)
        if klass not in ("unresolved_two", "unresolved_three"):
            continue
        f = column_flags(col)
        if f["strict_homophone"] or (loose and f["loose_homophone"]):
            out[i] = "H"
        elif (klass == "unresolved_three" and not f["strict_homophone"]
              and f["max_char_dist"] <= 2):
            out[i] = "C"
    return out
