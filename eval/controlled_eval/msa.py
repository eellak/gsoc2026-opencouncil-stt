#!/usr/bin/env python3
"""Multiple sequence alignment of three ASR hypotheses, and per-column voting.

The whole point of wayfinder #22: stop choosing whole windows and start composing
per word. Everything selection-shaped has a ceiling (the trio's whole-window oracle
is 0.1064 on this substrate); a per-column composition is not bounded by it, because
its output is a text none of the three systems produced.

ALIGNMENT. Exact three-way dynamic programming with unit-cost sum-of-pairs, banded.
Codex job 8112dc72 rejected the first design (progressive pivot-anchored alignment)
for the primary result: aligning the third hypothesis against "the pivot token if
present else the second's" is not a profile alignment, it ignores the other token
already sitting in the column, and it biases column boundaries towards the pivot,
which pulls W back towards the baseline it is supposed to beat. Progressive
alignment survives here only as `progressive_align`, used for the six-ordering
sensitivity that prices exactly that bias.

  state       (i, j, k) = tokens consumed from a, b, c
  transition  advance any non-empty subset; entries are the advanced tokens, epsilon
              elsewhere
  cost        sum over the three pairs: 0 if equal or both epsilon, else 1
  tie-break   fewest columns, then a fixed transition order (frozen, see ORDER)

Banding is an optimisation with a guard, not a silent approximation: `align3` widens
the band and re-runs whenever the recovered path pressed against the band edge. That
guard is a HEURISTIC, not a certificate — a path that stays clear of the edge is
strong evidence the band did not bind, but it does not prove the unbanded optimum was
reachable. Three ASR transcripts of the same two minutes do not need a wide band; the
band is also floored per window at the largest pairwise length difference by the
caller, which is the constraint that actually binds here.

VOTING is hierarchical, also on Codex's correction. A flat vote in which epsilon is
just another candidate deletes speech that two of three systems heard: the column
(epsilon, x, y) has two systems asserting a word and one asserting silence, and a
flat vote makes it silence because x != y. So: vote on OCCUPANCY first (token vs
epsilon), then on identity inside the winning class.
"""
from __future__ import annotations

# Transition order, frozen. Bitmask A=1, B=2, C=4. Ties in cost AND column count
# are broken by the first entry of this list, so it is part of the frozen design.
ORDER = (7, 3, 5, 6, 1, 2, 4)

INF = float("inf")
COLS_BITS = 12                 # score = cost << COLS_BITS | n_columns
COLS_MASK = (1 << COLS_BITS) - 1


def _pair(x, y) -> int:
    """Unit cost of one pair of column entries (None is epsilon)."""
    if x is None and y is None:
        return 0
    if x is None or y is None:
        return 1
    return 0 if x == y else 1


def _col_cost(ea, eb, ec) -> int:
    return _pair(ea, eb) + _pair(ea, ec) + _pair(eb, ec)


def columns_cost(cols) -> int:
    return sum(_col_cost(*c) for c in cols)


def align3(a: list[str], b: list[str], c: list[str], band: int = 40):
    """Exact sum-of-pairs 3-way alignment. Returns a list of (ea, eb, ec) columns.

    `band` constrains |i-j|, |i-k|, |j-k|. The returned path is checked against the
    band and the alignment is redone with a wider band if it ever pressed against it.
    That check is a heuristic guard, not a proof of the unbanded optimum; it becomes
    one only when the band is widened all the way to `lim`, which is the fallback.
    """
    na, nb, nc = len(a), len(b), len(c)
    if na == 0 and nb == 0 and nc == 0:
        return []
    lim = max(na, nb, nc) + 1
    while True:
        cols, touched = _align3_banded(a, b, c, band)
        if not touched or band >= lim:
            return cols
        band = min(band * 2, lim)


def _align3_banded(a, b, c, band):
    na, nb, nc = len(a), len(b), len(c)
    stride = nc + 1
    size = (nb + 1) * stride

    prev = [INF] * size
    cur = [INF] * size
    ops: list[bytearray] = []

    def in_band(i, j, k):
        return abs(i - j) <= band and abs(i - k) <= band and abs(j - k) <= band

    for i in range(na + 1):
        layer = bytearray(size)
        if i:
            prev, cur = cur, [INF] * size
        ai = a[i - 1] if i else None
        jlo, jhi = max(0, i - band), min(nb, i + band)
        klo, khi = max(0, i - band), min(nc, i + band)
        for j in range(jlo, jhi + 1):
            bj = b[j - 1] if j else None
            base = j * stride
            for k in range(klo, khi + 1):
                if i == 0 and j == 0 and k == 0:
                    cur[0] = 0
                    continue
                if abs(j - k) > band:
                    continue
                ck = c[k - 1] if k else None
                best = INF
                bestop = 0
                for m in ORDER:
                    pi = i - 1 if m & 1 else i
                    pj = j - 1 if m & 2 else j
                    pk = k - 1 if m & 4 else k
                    if pi < 0 or pj < 0 or pk < 0:
                        continue
                    src = prev if (m & 1) else cur
                    v = src[pj * stride + pk]
                    if v == INF:
                        continue
                    ea = ai if m & 1 else None
                    eb = bj if m & 2 else None
                    ec = ck if m & 4 else None
                    cost = _col_cost(ea, eb, ec)
                    cand = v + (cost << COLS_BITS) + 1
                    if cand < best:
                        best = cand
                        bestop = m
                cur[base + k] = best
                layer[base + k] = bestop
        ops.append(layer)

    if cur[nb * stride + nc] == INF:
        return [], True

    # backtrace
    cols = []
    i, j, k = na, nb, nc
    touched = False
    while i or j or k:
        if abs(i - j) >= band or abs(i - k) >= band or abs(j - k) >= band:
            touched = True
        m = ops[i][j * stride + k]
        if not m:
            return [], True
        ea = a[i - 1] if m & 1 else None
        eb = b[j - 1] if m & 2 else None
        ec = c[k - 1] if m & 4 else None
        cols.append((ea, eb, ec))
        if m & 1:
            i -= 1
        if m & 2:
            j -= 1
        if m & 4:
            k -= 1
    cols.reverse()
    return cols, touched


# --------------------------------------------------------------- progressive (sens.)
def _align2(x: list, y: list):
    """Pairwise alignment of two token lists; returns list of (ex, ey)."""
    n, m = len(x), len(y)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0], op[i][0] = i, 1
    for j in range(1, m + 1):
        d[0][j], op[0][j] = j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (0 if x[i - 1] == y[j - 1] else 1)
            de, ins = d[i - 1][j] + 1, d[i][j - 1] + 1
            best = min(sub, de, ins)
            d[i][j] = best
            op[i][j] = 3 if best == sub else (1 if best == de else 2)
    out = []
    i, j = n, m
    while i or j:
        m_ = op[i][j]
        if m_ == 3:
            out.append((x[i - 1], y[j - 1]))
            i, j = i - 1, j - 1
        elif m_ == 1:
            out.append((x[i - 1], None))
            i -= 1
        else:
            out.append((None, y[j - 1]))
            j -= 1
    out.reverse()
    return out


def progressive_align(seqs: list[list[str]], order: tuple[int, int, int]):
    """Pivot-anchored progressive alignment, kept ONLY as a sensitivity control.

    `order` names (pivot, second, third) as indices into `seqs`. The third sequence
    is aligned against the column's representative token — the pivot's if present,
    else the second's — which is precisely the approximation Codex flagged. Its
    output is put back in (a, b, c) order so callers can vote on it unchanged.
    """
    p, s, t = order
    pair = _align2(seqs[p], seqs[s])
    rep = [x if x is not None else y for x, y in pair]
    idx = [n for n, r in enumerate(rep) if r is not None]
    rep2 = [rep[n] for n in idx]
    pair2 = _align2(rep2, seqs[t])
    # walk pair2 back onto the column list, inserting new gap columns
    cols_ps: list[list] = [[x, y] for x, y in pair]
    out: list[list] = []
    ptr = 0                       # index into idx / rep2
    ci = 0                        # index into cols_ps
    for ex, ey in pair2:
        if ex is None:
            out.append([None, None, ey])
            continue
        target = idx[ptr]
        ptr += 1
        while ci < target:
            out.append(cols_ps[ci] + [None])
            ci += 1
        out.append(cols_ps[ci] + [ey])
        ci += 1
    while ci < len(cols_ps):
        out.append(cols_ps[ci] + [None])
        ci += 1
    res = []
    for col in out:
        e = [None, None, None]
        e[p], e[s], e[t] = col[0], col[1], col[2]
        res.append(tuple(e))
    return res


# ------------------------------------------------------------------------- voting
def vote_column(col, pivot: int, priority=(0, 1, 2)):
    """Hierarchical vote on one column. Returns (token or None, reason).

    1. occupancy: token vs epsilon, majority of three.
    2. identity inside the winning class: majority.
    3. still tied: the pivot's entry if it belongs to the winning class, else the
       first system in the frozen `priority` order that does.

    `reason` is one of unanimous / majority / tie_pivot / tie_priority / epsilon,
    and is what the LLM arm keys on: it arbitrates tie_* columns only.
    """
    toks = [e for e in col if e is not None]
    if len(toks) < 2:
        return None, "epsilon"
    counts: dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    top = max(counts.values())
    winners = [t for t, n in counts.items() if n == top]
    if len(winners) == 1:
        if top == 3:
            return winners[0], "unanimous"
        return winners[0], "majority"
    if col[pivot] is not None and col[pivot] in winners:
        return col[pivot], "tie_pivot"
    for p in priority:
        if col[p] is not None and col[p] in winners:
            return col[p], "tie_priority"
    return winners[0], "tie_priority"


def compose(cols, pivot: int, priority=(0, 1, 2)):
    """Vote every column. Returns (tokens, per-column decisions)."""
    toks, decisions = [], []
    for n, col in enumerate(cols):
        tok, why = vote_column(col, pivot, priority)
        decisions.append({"col": n, "token": tok, "reason": why})
        if tok is not None:
            toks.append(tok)
    return toks, decisions


# ------------------------------------------------------- alignment-conditional oracle
def oracle_columns(cols, ref: list[str]):
    """The oracle's text: `oracle_select` with the epsilon choices dropped."""
    return [t for t in oracle_select(cols, ref) if t is not None]


def oracle_select(cols, ref: list[str]):
    """Per-COLUMN oracle choice: one entry (or None for epsilon) per input column.

    Added for wayfinder #24. Callers that need to know WHICH column the oracle's
    tokens came from must use this: matching the token list back onto the columns by
    membership silently mis-attributes a token whenever the same word is a candidate
    in two nearby columns, which is common in Greek (articles, «και», repeated
    surnames). CodeRabbit flagged exactly that in the first version of the ceiling
    arms.

    Best text obtainable by choosing one entry per column, against `ref`.

    Exact DP over columns x reference. Codex job 8112dc72 confirmed the recurrence
    and fixed the boundary row, and required that the result be RECONSTRUCTED and
    re-scored by the frozen scorer rather than read off this DP's backtrace, since
    several minimum-cost outputs can exist. Returns the chosen token list.

    This is an ALIGNMENT-CONDITIONAL oracle: the columns come from one particular
    alignment of the hypotheses. It is not an alignment-free per-position ceiling,
    and the six progressive orderings are reported beside it to price that.
    """
    n, m = len(cols), len(ref)
    # candidate sets, frozen order: epsilon first if present, then tokens in
    # (a, b, c) order, de-duplicated
    cands = []
    for col in cols:
        seen, cl = set(), []
        if any(e is None for e in col):
            cl.append(None)
        for e in col:
            if e is not None and e not in seen:
                seen.add(e)
                cl.append(e)
        cands.append(cl)

    NEG = -1
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bk = [[(0, 0, NEG)] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        dp[0][j] = j
        bk[0][j] = (0, j - 1, NEG)
    for i in range(1, n + 1):
        cl = cands[i - 1]
        has_eps = cl and cl[0] is None
        dp[i][0] = dp[i - 1][0] + (0 if has_eps else 1)
        # cands[i-1][0] is epsilon when the column has one, else its first token, so
        # candidate 0 is the right backtrace entry either way
        bk[i][0] = (i - 1, 0, 0)
        for j in range(1, m + 1):
            best = dp[i][j - 1] + 1
            bkb = (i, j - 1, NEG)
            for ci, e in enumerate(cl):
                if e is None:
                    v = dp[i - 1][j]
                    if v < best:
                        best, bkb = v, (i - 1, j, ci)
                    continue
                v = dp[i - 1][j - 1] + (0 if e == ref[j - 1] else 1)
                if v < best:
                    best, bkb = v, (i - 1, j - 1, ci)
                v = dp[i - 1][j] + 1
                if v < best:
                    best, bkb = v, (i - 1, j, ci)
            dp[i][j] = best
            bk[i][j] = bkb
    out: list[str | None] = [None] * n
    i, j = n, m
    while i or j:
        pi, pj, ci = bk[i][j]
        if ci != NEG:
            out[i - 1] = cands[i - 1][ci]
        i, j = pi, pj
    return out
