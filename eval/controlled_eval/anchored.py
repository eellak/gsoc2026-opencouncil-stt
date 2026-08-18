#!/usr/bin/env python3
"""Anchored re-alignment, and the drift-zone occupancy guard.

Preregistration: `docs/specs/2026-08-18-anchored-realignment-prereg.md`. Everything
frozen there — TOL, MIN_GAP, the stoplist, the anchor definition, the drift-zone
definition — is frozen here, and none of it was tuned on a result.

THE DEFECT. W's alignment is text-only. `msa.align3` minimises sum-of-pairs edit cost
over three token streams and has no notion of time, so when one stream drifts a very
common token («το», «και», «να») matches an identical token from a DIFFERENT moment in
the audio: the spurious match is free, the correct gap costs 1. The mis-pairing chains
until a distinctive word re-syncs the streams, and the singleton columns it leaves
behind are deleted by the occupancy stage of `msa.vote_column`. Measured instance:
columns 84-94 of a 299 s slice, adapter words paired against tokens 2.2-2.8 s away,
four of W's sixteen deletions in five minutes.

TWO INDEPENDENT SWITCHES, so their effects can be measured apart:

  A  anchored segmentation. Find positions where all three streams emit the same
     token at the same time, cut there, align each piece independently with the
     UNMODIFIED frozen aligner, concatenate. Drift cannot propagate past an anchor.
  G  the drift-zone occupancy guard. Inside a run of columns whose timed occupants
     disagree about when, a singleton column is NOT deleted. A "majority" inside a
     mis-alignment is not a majority.

`eval/controlled_eval/msa.py` IS NOT EDITED AND IS NOT COPIED. Its sha256 keys an
18 MB alignment cache. This module imports `align3`, `vote_column` and `compose` and
calls them unchanged; the guard is a wrapper around `vote_column`, not a fork of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.controlled_eval.msa import align3, vote_column          # noqa: E402

# ---------------------------------------------------------------- frozen constants
TOL = 0.5          # seconds; "the same moment" for two streams on the common clock
MIN_GAP = 10       # tokens; minimum separation between consecutive anchors
BAND_FLOOR = 40    # identical to fusion_lab.BAND_FLOOR / exp_composition.py

# High-frequency Greek function words, written from the language and not from these
# transcripts, in the accent-stripped lowercase form `scoring.wtoks` produces. A token
# in here can still anchor, but only inside a run of >= 2 consecutive agreeing tokens.
STOPLIST = frozenset("""
και κι ο η το οι τα του της των τον την τη τους τις να θα δεν δε μην μη σε στο στη
στην στον στους στις στα με για απο που ειναι ως αλλα ομως ετσι αυτο αυτη αυτος ενα
ενας μια μας σας τι οτι ναι οχι
""".split())


def band_for(a, b, c) -> int:
    """The band rule `fusion_lab._band` uses, applied per piece."""
    return max(BAND_FLOOR, max(len(a), len(b), len(c)) - min(len(a), len(b), len(c)) + 20)


def align_piece(a, b, c):
    """One piece through the UNMODIFIED frozen aligner."""
    return align3(a, b, c, band=band_for(a, b, c))


# --------------------------------------------------------------------- anchors
def find_candidates(streams, starts, tol: float = TOL):
    """Position triples where all three streams emit the same token at the same time.

    `streams` is (a, b, c) token lists; `starts` is three parallel lists of start
    seconds on the COMMON clock. Returns a sorted list of (i, j, k).

    Condition 1 (identity) and condition 2 (pairwise time agreement within `tol`) of
    the preregistration. Distinctiveness and separation are applied later, because a
    stoplisted token needs to know whether its NEIGHBOURS are candidates.
    """
    a, b, c = streams
    ta, tb, tc = starts
    pos_b: dict[str, list[int]] = {}
    for j, t in enumerate(b):
        pos_b.setdefault(t, []).append(j)
    pos_c: dict[str, list[int]] = {}
    for k, t in enumerate(c):
        pos_c.setdefault(t, []).append(k)
    out = []
    for i, tok in enumerate(a):
        js, ks = pos_b.get(tok), pos_c.get(tok)
        if not js or not ks:
            continue
        ti = ta[i]
        for j in js:
            tj = tb[j]
            if abs(ti - tj) > tol:
                continue
            for k in ks:
                tk = tc[k]
                if abs(ti - tk) <= tol and abs(tj - tk) <= tol:
                    out.append((i, j, k))
    out.sort()
    return out


def admissible(cands, a, stoplist=STOPLIST):
    """Condition 3: a stoplisted token anchors only inside a run of >= 2 candidates."""
    cs = set(cands)
    out = []
    for (i, j, k) in cands:
        if a[i] not in stoplist:
            out.append((i, j, k))
            continue
        if (i - 1, j - 1, k - 1) in cs or (i + 1, j + 1, k + 1) in cs:
            out.append((i, j, k))
    return out


def _spread(trip, starts):
    ta, tb, tc = starts
    i, j, k = trip
    v = (ta[i], tb[j], tc[k])
    return max(v) - min(v)


def choose_anchors(cands, starts, min_gap: int = MIN_GAP):
    """Conditions 4 and 5: the strictly increasing, >= min_gap separated subset that
    MAXIMISES the anchor count, ties broken by minimum total time spread, then by
    earliest first index.

    Exact O(n^2) chain DP, so the answer does not depend on the order candidates were
    generated. Greedy-by-time would depend on it, and would let one early bad anchor
    block a better chain.
    """
    if not cands:
        return []
    n = len(cands)
    sp = [_spread(t, starts) for t in cands]
    # dp[t] = (count, -total_spread) of the best chain ENDING at t; maximise.
    dp = [(1, -sp[t]) for t in range(n)]
    back = [-1] * n
    for t in range(n):
        it, jt, kt = cands[t]
        best, bi = dp[t], -1
        for u in range(t):
            iu, ju, ku = cands[u]
            if it - iu < min_gap or jt - ju < min_gap or kt - ku < min_gap:
                continue
            cand = (dp[u][0] + 1, dp[u][1] - sp[t])
            if cand > best:
                best, bi = cand, u
        dp[t], back[t] = best, bi
    end = max(range(n), key=lambda t: (dp[t][0], dp[t][1], -cands[t][0]))
    chain = []
    while end != -1:
        chain.append(cands[end])
        end = back[end]
    chain.reverse()
    return chain


def anchors_for(streams, starts, tol: float = TOL, min_gap: int = MIN_GAP,
                stoplist=STOPLIST):
    """The full frozen anchor pipeline: candidates -> admissible -> chosen chain."""
    cands = find_candidates(streams, starts, tol)
    return choose_anchors(admissible(cands, streams[0], stoplist), starts, min_gap)


# ------------------------------------------------------------------ re-alignment
def anchored_columns(streams, anchors, align_fn=align_piece):
    """Cut at the anchors, align each piece independently, concatenate.

    Each anchor contributes one UNANIMOUS column (v, v, v) between its pieces. With
    `anchors == []` this is exactly one call to the frozen aligner over the whole
    window, i.e. byte-identical to W's columns — asserted by the tests, not assumed.
    """
    a, b, c = streams
    cols: list[tuple] = []
    pi = pj = pk = 0
    for (i, j, k) in anchors:
        cols.extend(align_fn(a[pi:i], b[pj:j], c[pk:k]))
        cols.append((a[i], b[j], c[k]))
        pi, pj, pk = i + 1, j + 1, k + 1
    cols.extend(align_fn(a[pi:], b[pj:], c[pk:]))
    return cols


# ------------------------------------------------------------------- drift zones
def column_starts(cols, starts):
    """Per column, the start time of each present occupant (None where absent).

    Occupants are matched to times BY OCCURRENCE INDEX — each column consumes the next
    token of every stream it holds — so a repeated word is never confused with itself.
    """
    idx = [0, 0, 0]
    out = []
    for col in cols:
        row = []
        for s in range(3):
            if col[s] is None:
                row.append(None)
            else:
                st = starts[s]
                row.append(st[idx[s]] if st is not None and idx[s] < len(st) else None)
                idx[s] += 1
        out.append(row)
    return out


def drift_fire(cols, starts, tol: float = TOL):
    """Column indices where the occupancy guard fires.

    1. disagreeing: >= 2 TIMED occupants whose starts span more than `tol`.
    2. thin: exactly 1 occupant -- the columns the occupancy stage deletes.
    3. a drift zone is a maximal run of columns that are each disagreeing or thin AND
       that contains >= 2 disagreeing columns. The floor of two is what separates a
       drift (which chains) from one noisy timestamp.
    4. the guard fires on the thin columns inside such a zone.

    Returns (fire_set, n_zones).
    """
    dis, thin = [], []
    for col, row in zip(cols, column_starts(cols, starts)):
        t = [v for v in row if v is not None]
        dis.append(len(t) >= 2 and (max(t) - min(t)) > tol)
        thin.append(sum(1 for e in col if e is not None) == 1)
    fire: set[int] = set()
    zones = 0
    n = len(cols)
    i = 0
    while i < n:
        if not (dis[i] or thin[i]):
            i += 1
            continue
        j = i
        while j < n and (dis[j] or thin[j]):
            j += 1
        if sum(1 for x in range(i, j) if dis[x]) >= 2:
            zones += 1
            fire.update(x for x in range(i, j) if thin[x])
        i = j
    return fire, zones


# ------------------------------------------------------------------------ voting
def compose_guarded(cols, pivot: int, fire=frozenset(), priority=(0, 1, 2)):
    """`msa.compose`, plus the guard. Every column still goes through the FROZEN
    `msa.vote_column`; the guard only overrides the epsilon outcome inside a drift
    zone, where a single occupant is not evidence that two systems heard silence.

    With `fire == set()` this returns exactly what `msa.compose` returns.
    """
    toks, decisions = [], []
    for n, col in enumerate(cols):
        tok, why = vote_column(col, pivot, priority)
        if tok is None and n in fire:
            present = [e for e in col if e is not None]
            if len(present) == 1:
                tok, why = present[0], "drift_guard"
        decisions.append({"col": n, "token": tok, "reason": why})
        if tok is not None:
            toks.append(tok)
    return toks, decisions
