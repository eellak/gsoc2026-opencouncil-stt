#!/usr/bin/env python3
"""Can the per-column vote see a 2-of-3 near-miss? Measured first, voted second.

The hierarchical vote of `msa.vote_column` settles a column by EXACT token identity.
Where all three systems disagree (`unresolved_three`, 1,104 of 80,659 columns on this
substrate) there is no exact majority, so the vote falls back to the pivot or to the
frozen system priority — it cannot see that two of the three candidates are the same
word off by a character.

This script has two stages and they must be run in this order.

  census   Descriptive only. For every unresolved column: the pairwise character
           distances, whether a 2-of-3 NEAR majority exists at distance 1 or 2,
           whether a 2-of-3 FOLDED (Greek strict-phonemic) majority exists, and how
           often the near/folded majority's representative is the reference word.
           Writes results_near_miss_census.json. No arm is scored here.

  score    Runs the FROZEN arms of docs/reports/2026-08-24-near-miss-vote.md through
           `fusion_lab.evaluate` against W on the same substrate, same scorer, paired
           meeting-clustered bootstrap, 10,000 resamples, plus leave-one-out over
           window / meeting / city. Writes results_near_miss_vote.json.

WHY `max_char_dist` IS NOT THE SAME QUESTION. `column_classes.eligibility` already
admits arm C on `unresolved_three` columns whose MAXIMUM pairwise character distance
is <= 2 (136 columns, `results_column_census.json`). A 2-of-3 near majority is a
MINIMUM-pairwise-distance property: two candidates close together and a third far
away is invisible to the max rule and is exactly the case this asks about. The two
sets overlap but neither contains the other.

REFERENCE ATTRIBUTION is alignment-conditional. `oracle_align` below is
`msa.oracle_select`'s DP with the backtrace exposed, so each column reports both the
candidate the column oracle would choose and the reference token that oracle's path
matched against it (None when the column was aligned as an insertion). A different
valid alignment could attribute differently. Every "is the reference word" number in
the census carries that caveat.

Aggregates only; no transcript text is ever written to disk by this script.

Env: SC (cache dir), N_BOOT (10000), WORKERS
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval.column_classes import (column_class,          # noqa: E402
                                                 split_merge_columns)
from eval.controlled_eval.fusion_lab import (Idea, N_BOOT, Substrate,   # noqa: E402
                                             evaluate, load_substrate, log,
                                             summary_line)
from eval.controlled_eval.greek_phonetics import phon                   # noqa: E402
from eval.controlled_eval.scoring import edist                          # noqa: E402

CENSUS_OUT = Path(__file__).with_name("results_near_miss_census.json")
SCORE_OUT = Path(__file__).with_name("results_near_miss_vote.json")

# Frozen system priority, identical to `msa.compose`'s default: (scribe, soniox, ours).
PRIORITY = (0, 1, 2)


# --------------------------------------------------------------- ref attribution
def oracle_align(cols, ref: list[str]):
    """`msa.oracle_select` with the reference match exposed.

    Returns a list, one entry per column: (chosen_candidate, matched_ref_token).
    `chosen_candidate` is None when the oracle would emit epsilon; `matched_ref_token`
    is None when the oracle's path consumed no reference token at that column, i.e.
    the column is an insertion under that path.

    The DP, the candidate order and the tie-breaking are copied verbatim from
    `msa.oracle_select` so the chosen candidates are identical to the ones the census
    and the ceiling arms already report. Only the extra backtrace field is new.
    """
    n, m = len(cols), len(ref)
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
    out: list[tuple] = [(None, None)] * n
    i, j = n, m
    while i or j:
        pi, pj, ci = bk[i][j]
        if ci != NEG:
            chosen = cands[i - 1][ci]
            reftok = ref[j - 1] if (pj == j - 1 and pi == i - 1) else None
            out[i - 1] = (chosen, reftok)
        i, j = pi, pj
    return out


# --------------------------------------------------------------- near-miss geometry
def _present(col):
    """(system index, token) for the entries that carry a token."""
    return [(s, e) for s, e in enumerate(col) if e is not None]


def near_pair(col, maxd: int):
    """The closest pair of DISTINCT candidates, if within `maxd` characters.

    Returns (system_i, system_j, distance) with i < j, or None. Ties in distance are
    broken by system-index order — (0,1) before (0,2) before (1,2) — which is the same
    frozen priority the exact vote uses, so no threshold is being fitted here.
    """
    pres = _present(col)
    best = None
    for a in range(len(pres)):
        for b in range(a + 1, len(pres)):
            si, ti = pres[a]
            sj, tj = pres[b]
            if ti == tj:
                continue
            d = edist(list(ti), list(tj))
            if d <= maxd and (best is None or d < best[2]):
                best = (si, sj, d)
    return best


def folded_pair(col):
    """The first pair of distinct candidates sharing a STRICT Greek phonemic key."""
    pres = _present(col)
    for a in range(len(pres)):
        for b in range(a + 1, len(pres)):
            si, ti = pres[a]
            sj, tj = pres[b]
            if ti != tj and phon(ti) == phon(tj):
                return (si, sj, 0)
    return None


def represent(col, si, sj):
    """Which member of a near/folded pair the arm would emit: frozen priority order."""
    for p in PRIORITY:
        if p in (si, sj):
            return col[p]
    return col[si]


# ---------------------------------------------------------------------- census
def run_census(sub: Substrate) -> dict:
    n_class = Counter()
    stat = Counter()
    dist_hist = Counter()          # (class, min pairwise distance capped at 6)
    per_meeting = Counter()        # meeting -> eligible columns, for domination
    total_cols = 0

    for w in sub.windows:
        oa = oracle_align(w.cols, w.ref)
        sm = split_merge_columns(w.cols)
        for i, col in enumerate(w.cols):
            total_cols += 1
            klass = column_class(col)
            n_class[klass] += 1
            if klass not in ("unresolved_two", "unresolved_three"):
                continue
            chosen, reftok = oa[i]
            w_tok = w.decisions[i]["token"]
            blocked = i in sm
            if blocked:
                stat[f"{klass}:split_merge"] += 1
            pres = _present(col)
            dists = [edist(list(ti), list(tj))
                     for a, (_, ti) in enumerate(pres)
                     for _, tj in pres[a + 1:]]
            dmin = min(dists) if dists else None
            if dmin is not None:
                dist_hist[(klass, min(dmin, 6))] += 1
            if reftok is not None:
                stat[f"{klass}:ref_attributed"] += 1
                if w_tok == reftok:
                    stat[f"{klass}:W_is_ref"] += 1
                if reftok in [t for _, t in pres]:
                    stat[f"{klass}:ref_among_candidates"] += 1
            if chosen is not None and w_tok == chosen:
                stat[f"{klass}:W_is_oracle_choice"] += 1

            for label, cand in (("d1", near_pair(col, 1)),
                                ("d2", near_pair(col, 2)),
                                ("fold", folded_pair(col))):
                if cand is None:
                    continue
                si, sj, _ = cand
                key = f"{klass}:{label}"
                stat[key] += 1
                if not blocked:
                    stat[key + ":unblocked"] += 1
                    if klass == "unresolved_three":
                        per_meeting[(label, w.meeting)] += 1
                rep = represent(col, si, sj)
                pair_toks = {col[si], col[sj]}
                if rep != w_tok:
                    stat[key + ":would_change_W"] += 1
                if reftok is not None:
                    stat[key + ":ref_attributed"] += 1
                    if reftok in pair_toks:
                        stat[key + ":pair_contains_ref"] += 1
                    if rep == reftok:
                        stat[key + ":rep_is_ref"] += 1
                    if w_tok == reftok:
                        stat[key + ":W_is_ref"] += 1
                if chosen is not None:
                    if rep == chosen:
                        stat[key + ":rep_is_oracle_choice"] += 1
                    if w_tok == chosen:
                        stat[key + ":W_is_oracle_choice"] += 1

    res = {
        "substrate": sub.meta,
        "n_columns": total_cols,
        "classes": {k: {"n": v, "share": v / total_cols} for k, v in n_class.most_common()},
        "clean_share": sum(n_class[k] for k in ("agree", "exact_2_of_3",
                                                "two_present_same")) / total_cols,
        "min_pairwise_distance_histogram": {
            f"{k}:{d}": v for (k, d), v in sorted(dist_hist.items())},
        "stats": dict(sorted(stat.items())),
        "meeting_concentration": {
            label: sorted(((m, n) for (l, m), n in per_meeting.items() if l == label),
                          key=lambda kv: -kv[1])[:5]
            for label in ("d1", "d2", "fold")},
        "note": "DESCRIPTIVE. rep_is_ref / pair_contains_ref read the reference "
                "through the alignment-conditional column oracle of oracle_align(); "
                "no threshold below is fitted on them.",
    }
    CENSUS_OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log(f"-> {CENSUS_OUT}")
    return res


# ------------------------------------------------------------------------- arms
class NearMiss(Idea):
    """FROZEN arm N<d>: 2-of-3 near-miss majority on three-way disagreement columns.

    Scope, all of it frozen before any WER was computed:
      * only `unresolved_three` columns ([x, y, z], all three systems present and all
        three tokens distinct). `unresolved_two` is excluded: two present tokens have
        no majority to find, near or not.
      * not a `split_merge` column.
      * the closest pair of candidates must be within `maxd` characters, and the
        third candidate must be strictly further from both members of that pair than
        they are from each other, so the "majority" is a real cluster of two and not
        three mutually near strings.
      * emit that pair's member from the earliest system in the frozen priority order
        (scribe, soniox, ours).

    The arm never emits epsilon and never consumes a column the vote dropped, so the
    token count and therefore the deletion rate cannot move. It fits no parameter, so
    leave-one-city-out is vacuous by construction.
    """

    fitted = False

    def __init__(self, maxd: int, name: str | None = None):
        self.maxd = maxd
        self.name = name or f"N{maxd}"

    def _pick(self, col):
        pres = _present(col)
        if len(pres) != 3 or len({t for _, t in pres}) != 3:
            return None
        cand = near_pair(col, self.maxd)
        if cand is None:
            return None
        si, sj, d = cand
        sk = ({0, 1, 2} - {si, sj}).pop()
        if min(edist(list(col[sk]), list(col[si])),
               edist(list(col[sk]), list(col[sj]))) <= d:
            return None                      # not a cluster of two
        return represent(col, si, sj)

    def apply(self, w, params):
        sm = split_merge_columns(w.cols)
        out = []
        for i, col in enumerate(w.cols):
            tok = w.decisions[i]["token"]
            if tok is not None and i not in sm:
                rep = self._pick(col)
                if rep is not None:
                    tok = rep
            if tok is not None:
                out.append(tok)
        return out


class FoldedMajority(Idea):
    """FROZEN arm F: 2-of-3 STRICT-homophone majority, same scope as N.

    Identical scope and representative rule to `NearMiss`, except the pair is defined
    by `greek_phonetics.phon` equality rather than by character distance, and the
    third candidate must NOT share that key (otherwise the column is a three-way
    homophone tie and arm H of `exp_char_homophone.py` owns it).

    This is a separate hypothesis from N, not a variant of it: "the two systems wrote
    the same Greek word" is not the same claim as "the two strings are close".
    """

    fitted = False
    name = "F"

    def _pick(self, col):
        pres = _present(col)
        if len(pres) != 3 or len({t for _, t in pres}) != 3:
            return None
        cand = folded_pair(col)
        if cand is None:
            return None
        si, sj, _ = cand
        sk = ({0, 1, 2} - {si, sj}).pop()
        if phon(col[sk]) == phon(col[si]):
            return None
        return represent(col, si, sj)

    apply = NearMiss.apply


class Union(Idea):
    """FROZEN arm NF: arm F first, arm N2 on any column F did not claim."""

    fitted = False
    name = "N2+F"

    def __init__(self):
        self._f = FoldedMajority()
        self._n = NearMiss(2)

    def _pick(self, col):
        return self._f._pick(col) or self._n._pick(col)

    apply = NearMiss.apply


# ------------------------------------------------------------------------- score
def run_score(sub: Substrate) -> dict:
    arms = [NearMiss(1), NearMiss(2), FoldedMajority(), Union()]
    out = {"substrate": sub.meta, "n_boot": N_BOOT,
           "census": "eval/controlled_eval/results_near_miss_census.json",
           "frozen_rule": "docs/reports/2026-08-24-near-miss-vote.md",
           "arms": {}}
    for arm in arms:
        res = evaluate(arm, sub, fold="city", n_boot=N_BOOT)
        res.pop("detail", None)
        out["arms"][arm.name] = res
        log(summary_line(res))
    SCORE_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    log(f"-> {SCORE_OUT}")
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    sub = load_substrate()
    log(json.dumps(sub.meta, indent=1))
    if mode == "census":
        run_census(sub)
    elif mode == "score":
        run_score(sub)
    else:
        raise SystemExit("usage: exp_near_miss_vote.py [census|score]")


if __name__ == "__main__":
    main()
