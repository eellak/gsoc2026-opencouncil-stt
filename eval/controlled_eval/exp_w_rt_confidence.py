#!/usr/bin/env python3
"""Does Soniox per-word confidence beat the plain per-column vote, inside W-rt?

Preregistration (revision 2, Codex `b71f2dca0cad451db62cfb8f65e9d08e`):
`docs/specs/2026-08-16-w-rt-confidence-prereg.md`. Read it before reading any number
this file prints. In particular: this is a DEVELOPMENT run, no arm ships on it, the
confirmatory family is exactly {O, A, M}, and every arm is measured against the W-rt
baseline and never against the frozen old-W numbers.

Arms (all rewrite W-rt's token stream, all evaluated by the unmodified
`fusion_lab.evaluate` under the frozen gates, fold = city):

  O    singleton columns whose lone token is Soniox's: emit it iff conf >= tau_O
  O2   unresolved_two columns with Soniox present: take Soniox's token iff conf >= tau
       (declared variant, NOT in the family)
  A    unresolved_two / unresolved_three: weighted identity vote, Soniox weight =
       conf, the other two systems weight k = 0.5, ties keep W-rt's choice
  M    exact_2_of_3 where Soniox is the minority: override iff conf >= tau_M

Controls that isolate confidence from the structural rewrite (mandatory, preregistered):
  O-all, M-all   the same rewrites with tau = 0
  permutation    conf permuted within (meeting x arm eligibility), complete fitting
                 procedure rerun, 200 replicates, giving a null for each fitted arm

FITTING OBJECTIVE. Thresholds are fitted by minimising pooled training WER. WER's
numerator S+D+I is exactly the unit-cost Levenshtein distance, so the search uses
`rapidfuzz` for the distance and the frozen `sdi` scorer for every reported number.
`test_w_rt.py` asserts the two agree on the real substrate.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import fusion_lab as F                       # noqa: E402
from eval.controlled_eval import w_rt as R                             # noqa: E402
from eval.controlled_eval.column_classes import (column_class,         # noqa: E402
                                                 split_merge_columns)
from eval.controlled_eval.msa import oracle_select                     # noqa: E402
from rapidfuzz.distance import Levenshtein as _LEV                     # noqa: E402

SON = R.SONIOX_IDX
K_OTHER = 0.5                       # preregistered constant weight, see spec S4
K_ENVELOPE = (0.3, 0.4, 0.6, 0.7)   # sensitivity only
N_PERM = int(os.environ.get("N_PERM", "200"))
PERM_SEED = 21
OUT = ROOT / "eval" / "controlled_eval" / "results_w_rt_confidence.json"

# preregistered threshold grid: 0.00..1.00 step 0.05, plus 0.99 / 0.999 (the mass sits
# at the top of the confidence range) and 1.01 so that "never fire" is reachable.
GRID = sorted(set([round(i / 20, 4) for i in range(21)] + [0.99, 0.999, 1.01]))

ARMS = ("O", "O2", "A", "M")
FAMILY = ("O", "A", "M")


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ preparation
class Prep:
    """Per-window frozen precomputation: base column tokens and per-arm eligibility."""

    __slots__ = ("base", "elig", "oracle", "n_cols", "classes")

    def __init__(self, w: F.Window, conf: list):
        self.base = [d["token"] for d in w.decisions]
        assert [t for t in self.base if t is not None] == list(w.w_tokens)
        self.n_cols = len(w.cols)
        sm = split_merge_columns(w.cols)
        self.classes = [column_class(c) for c in w.cols]
        sidx = R.soniox_column_index(w.cols)
        n_son = sum(1 for x in sidx if x is not None)
        assert n_son == len(conf), (
            f"{w.item_id}: {n_son} soniox columns but {len(conf)} confidences")
        self.oracle = oracle_select(w.cols, w.ref)
        elig: dict[str, list] = {a: [] for a in ARMS}
        for i, col in enumerate(w.cols):
            if i in sm:
                continue
            si = sidx[i]
            if si is None:
                continue
            c = conf[si]
            if c is None:
                continue
            klass = self.classes[i]
            tok = col[SON]
            if klass == "singleton":
                elig["O"].append((i, tok, c))
            elif klass == "unresolved_two":
                elig["O2"].append((i, tok, c))
                elig["A"].append((i, tok, c))
            elif klass == "unresolved_three":
                elig["A"].append((i, tok, c))
            elif klass == "exact_2_of_3":
                present = [e for e in col if e is not None]
                if present.count(tok) == 1:          # Soniox is the minority voice
                    elig["M"].append((i, tok, c))
        self.elig = elig


def prepare(sub: F.Substrate, conf: dict) -> dict[str, Prep]:
    return {w.item_id: Prep(w, conf[w.item_id]["conf"]) for w in sub.windows}


# ------------------------------------------------------------------------- arms
def _emit(base: list, changes: dict[int, object]) -> list[str]:
    out = []
    for i, t in enumerate(base):
        t = changes.get(i, t)
        if t is not None:
            out.append(t)
    return out


def apply_threshold_arm(prep: Prep, arm: str, tau: float,
                        conf_override: dict[int, float] | None = None) -> list[str]:
    ch = {}
    for i, tok, c in prep.elig[arm]:
        if conf_override is not None:
            c = conf_override[i]
        if c >= tau:
            ch[i] = tok
    return _emit(prep.base, ch)


def apply_vote_arm(w: F.Window, prep: Prep, k: float,
                   conf_override: dict[int, float] | None = None) -> list[str]:
    ch = {}
    for i, tok, c in prep.elig["A"]:
        if conf_override is not None:
            c = conf_override[i]
        col = w.cols[i]
        weight: dict[str, float] = {}
        for s in range(3):
            e = col[s]
            if e is None:
                continue
            weight[e] = weight.get(e, 0.0) + (c if s == SON else k)
        best = max(weight.values())
        winners = [t for t, v in weight.items() if v == best]
        if len(winners) == 1 and winners[0] != prep.base[i]:
            ch[i] = winners[0]
    return _emit(prep.base, ch)


class ThresholdIdea(F.Idea):
    fitted = True

    def __init__(self, arm: str, preps: dict[str, Prep], fixed: float | None = None):
        self.arm = arm
        self.preps = preps
        self.fixed = fixed
        self.name = arm if fixed is None else f"{arm}-all"
        if fixed is not None:
            self.fitted = False

    def fit(self, train: list[F.Window]):
        if self.fixed is not None:
            return self.fixed
        return fit_threshold(self.arm, train, self.preps)

    def apply(self, w: F.Window, params) -> list[str]:
        return apply_threshold_arm(self.preps[w.item_id], self.arm, params)


class VoteIdea(F.Idea):
    fitted = False

    def __init__(self, preps: dict[str, Prep], k: float = K_OTHER):
        self.preps = preps
        self.k = k
        self.name = "A" if k == K_OTHER else f"A(k={k})"

    def apply(self, w: F.Window, params) -> list[str]:
        return apply_vote_arm(w, self.preps[w.item_id], self.k)


# ----------------------------------------------------------------- fast fitting
def _lev(a: list[str], b: list[str]) -> int:
    return _LEV.distance(a, b)


def curves(windows, preps: dict[str, Prep], arm: str,
           conf_override: dict[str, dict[int, float]] | None = None) -> dict:
    """Per window: (confidences of the eligible columns, descending; edit distance
    when the k highest-confidence of them fire, for k = 0..n).

    Firing "every eligible column with conf >= tau" is exactly "firing the k highest,
    where k counts the ties too", so one curve per window answers the whole grid and
    every fold. Without this the permutation null would recompute the same distances
    ten times per replicate.
    """
    out = {}
    for w in windows:
        p = preps[w.item_id]
        ov = conf_override.get(w.item_id) if conf_override is not None else None
        items = [((ov[i] if ov is not None else c), i, tok)
                 for i, tok, c in p.elig[arm]]
        items.sort(key=lambda t: -t[0])
        ch: dict[int, object] = {}
        d = [_lev(w.ref, _emit(p.base, ch))]
        for _c, i, tok in items:
            ch[i] = tok
            d.append(_lev(w.ref, _emit(p.base, ch)))
        out[w.item_id] = ([c for c, _i, _t in items], d)
    return out


def _k_for_tau(confs: list[float], tau: float) -> int:
    """How many of `confs` (sorted DESCENDING) are >= tau."""
    lo, hi = 0, len(confs)
    while lo < hi:
        mid = (lo + hi) // 2
        if confs[mid] >= tau:
            lo = mid + 1
        else:
            hi = mid
    return lo


def fit_threshold(arm: str, train: list[F.Window], preps: dict[str, Prep],
                  conf_override: dict[str, dict[int, float]] | None = None,
                  cur: dict | None = None) -> float:
    """Grid search for the training-WER-minimising threshold; ties -> LARGER tau.

    The denominator is constant across the grid, so minimising the pooled edit
    numerator is identical to minimising pooled training WER.
    """
    if cur is None:
        cur = curves(train, preps, arm, conf_override)
    best_tau, best_num = None, None
    for tau in GRID:
        num = 0
        for w in train:
            confs, dists = cur[w.item_id]
            num += dists[_k_for_tau(confs, tau)]
        if best_num is None or num < best_num or (num == best_num and tau > best_tau):
            best_tau, best_num = tau, num
    return best_tau


# ------------------------------------------------------------------- permutation
def permuted_conf(sub: F.Substrate, preps: dict[str, Prep], arm: str,
                  rng: random.Random) -> dict[str, dict[int, float]]:
    """Permute conf within (meeting x this arm's eligible set)."""
    by_meeting: dict[str, list[tuple[str, int]]] = {}
    for w in sub.windows:
        for i, _tok, _c in preps[w.item_id].elig[arm]:
            by_meeting.setdefault(w.meeting, []).append((w.item_id, i))
    lookup = {w.item_id: {i: c for i, _t, c in preps[w.item_id].elig[arm]}
              for w in sub.windows}
    out: dict[str, dict[int, float]] = {w.item_id: {} for w in sub.windows}
    for _m, slots in by_meeting.items():
        vals = [lookup[wid][i] for wid, i in slots]
        rng.shuffle(vals)
        for (wid, i), v in zip(slots, vals):
            out[wid][i] = v
    return out


def perm_null(sub: F.Substrate, preps: dict[str, Prep], arm: str, n: int,
              vote_k: float | None = None) -> dict:
    """Null distribution of the arm's out-of-fold delta-WER, refitting inside each
    replicate. Uses the Levenshtein numerator (== S+D+I) rather than the frozen sdi
    split, which is exact for WER and ~200x faster."""
    rng = random.Random(PERM_SEED)
    by_city: dict[str, list[F.Window]] = {}
    for w in sub.windows:
        by_city.setdefault(w.city, []).append(w)
    den = sum(len(w.ref) for w in sub.windows)
    base_num = sum(_lev(w.ref, w.w_tokens) for w in sub.windows)
    deltas = []
    for r in range(n):
        ov = permuted_conf(sub, preps, arm, rng)
        cur = None if vote_k is not None else curves(sub.windows, preps, arm, ov)
        num = 0
        for city, held in by_city.items():
            if vote_k is None:
                train = [w for w in sub.windows if w.city != city]
                tau = fit_threshold(arm, train, preps, ov, cur=cur)
                for w in held:
                    confs, dists = cur[w.item_id]
                    num += dists[_k_for_tau(confs, tau)]
            else:
                for w in held:
                    num += _lev(w.ref, apply_vote_arm(
                        w, preps[w.item_id], vote_k, ov[w.item_id]))
        deltas.append((num - base_num) / den)
        if (r + 1) % 25 == 0:
            log(f"    perm {arm} {r + 1}/{n}")
    deltas.sort()
    return {"n": n, "mean": sum(deltas) / len(deltas),
            "p05": deltas[max(0, int(0.05 * len(deltas)) - 1)],
            "p50": deltas[len(deltas) // 2],
            "p95": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))],
            "min": deltas[0], "max": deltas[-1]}


# ------------------------------------------------------------------- statistics
def adjusted_ci(res: dict, alpha_family: int = 3) -> dict:
    """Bonferroni-adjusted central two-sided percentile interval from the same
    paired meeting-clustered bootstrap: quantiles 0.8333% and 99.1667% for 3 arms."""
    import numpy as np
    d = res["detail"]
    a = np.array([(r[0] + r[1] + r[2], r[3]) for r in d["rows_arm"]], dtype=float)
    b = np.array([(r[0] + r[1] + r[2], r[3]) for r in d["rows_W"]], dtype=float)
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(d["meetings"]):
        groups.setdefault(m, []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(7)
    n_boot = F.N_BOOT
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        den = a[idx, 1].sum()
        diffs[i] = (a[idx, 0].sum() - b[idx, 0].sum()) / den if den else np.nan
    q = 100 * (0.05 / alpha_family) / 2
    lo, hi = np.nanpercentile(diffs, [q, 100 - q])
    return {"level": 1 - 0.05 / alpha_family, "quantiles": [q, 100 - q],
            "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def city_sign_test(res: dict) -> dict:
    """Exact two-sided paired sign test over the 10 cities, ties excluded.

    Low power by construction; reported because a meeting-clustered CI is not
    city-level evidence."""
    pc = res["per_city"]["per_city"]
    neg = sum(1 for v in pc.values() if v["delta"] < 0)
    pos = sum(1 for v in pc.values() if v["delta"] > 0)
    n = neg + pos
    if n == 0:
        return {"n_effective": 0, "p_two_sided": None}
    k = min(neg, pos)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / (2 ** n)
    return {"cities_better": neg, "cities_worse": pos, "n_effective": n,
            "p_two_sided": min(1.0, 2 * tail)}


def domination(res: dict) -> dict:
    d = res["detail"]
    tot = sum(sum(r[:3]) for r in d["rows_W"]) - sum(sum(r[:3]) for r in d["rows_arm"])
    if tot <= 0:
        return {"applicable": False, "total_improvement": tot,
                "note": "no improvement to dominate"}
    by: dict[str, int] = {}
    for m, ra, rw in zip(d["meetings"], d["rows_arm"], d["rows_W"]):
        by[m] = by.get(m, 0) + (sum(rw[:3]) - sum(ra[:3]))
    top = max(by.items(), key=lambda kv: kv[1])
    return {"applicable": True, "total_improvement": tot,
            "top_meeting": top[0], "top_share": top[1] / tot,
            "dominated": bool(top[1] / tot > 0.5)}


# --------------------------------------------------------------------- exposure
def exposure(sub: F.Substrate, preps: dict[str, Prep], arm: str, res: dict,
             fold_params: dict) -> dict:
    """Eligible / firing counts, and (reporting only) agreement with the column
    oracle. The oracle is never visible to fit or apply."""
    n_elig = sum(len(preps[w.item_id].elig[arm]) for w in sub.windows)
    fired = fired_oracle_ok = fired_oracle_bad = 0
    for w in sub.windows:
        p = preps[w.item_id]
        tau = fold_params.get(str(w.city))
        if not isinstance(tau, (int, float)):
            continue
        for i, tok, c in p.elig[arm]:
            if c >= tau:
                fired += 1
                if p.oracle[i] == tok:
                    fired_oracle_ok += 1
                elif p.oracle[i] == p.base[i]:
                    fired_oracle_bad += 1
    return {"eligible_columns": n_elig, "fired": fired,
            "fired_matching_oracle": fired_oracle_ok,
            "fired_against_oracle": fired_oracle_bad,
            "windows_changed_vs_W_rt": res["windows_changed_vs_W"],
            "fold_thresholds": fold_params}


def vote_exposure(sub: F.Substrate, preps: dict[str, Prep], k: float) -> dict:
    n_elig = fired = ok = bad = 0
    for w in sub.windows:
        p = preps[w.item_id]
        n_elig += len(p.elig["A"])
        for i, _tok, c in p.elig["A"]:
            col = w.cols[i]
            weight: dict[str, float] = {}
            for s in range(3):
                e = col[s]
                if e is None:
                    continue
                weight[e] = weight.get(e, 0.0) + (c if s == SON else k)
            best = max(weight.values())
            winners = [t for t, v in weight.items() if v == best]
            if len(winners) == 1 and winners[0] != p.base[i]:
                fired += 1
                if p.oracle[i] == winners[0]:
                    ok += 1
                elif p.oracle[i] == p.base[i]:
                    bad += 1
    return {"eligible_columns": n_elig, "fired": fired,
            "fired_matching_oracle": ok, "fired_against_oracle": bad}


# ------------------------------------------------------------------------- main
def census(sub: F.Substrate) -> dict:
    counts: dict[str, int] = {}
    total = 0
    for w in sub.windows:
        for col in w.cols:
            k = column_class(col)
            counts[k] = counts.get(k, 0) + 1
            total += 1
    return {"total_columns": total,
            "by_class": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "share": {k: v / total for k, v in counts.items()}}


def strip_detail(res: dict) -> dict:
    """`detail` carries verbatim council speech. It never reaches disk."""
    return {k: v for k, v in res.items() if k != "detail"}


def main() -> None:
    n_boot = int(os.environ.get("N_BOOT", str(F.N_BOOT)))
    log("building W-rt substrate")
    sub, conf = R.load_substrate_rt(strict=os.environ.get("STRICT", "1") == "1")
    log(json.dumps(sub.meta, indent=1, ensure_ascii=False))
    preps = prepare(sub, conf)

    results: dict = {
        "spec": "docs/specs/2026-08-16-w-rt-confidence-prereg.md",
        "meta": sub.meta,
        "census_W_rt": census(sub),
        "soniox_rt_stats": {
            k: sum(conf[w.item_id]["stats"].get(k, 0) for w in sub.windows)
            for k in ("n_words", "n_units", "words_without_timestamp",
                      "words_without_confidence", "words_without_lexical_rune",
                      "words_dropped_by_normalization",
                      "words_split_into_several_tokens",
                      "units_with_invalid_confidence", "residual_nonfinal_tokens")},
        "eligible_columns": {a: sum(len(preps[w.item_id].elig[a]) for w in sub.windows)
                             for a in ARMS},
        "grid": GRID,
        "k_other": K_OTHER,
        "arms": {},
        "controls": {},
        "sensitivity": {},
    }

    # ---- baseline identity check and the W-rt baseline itself
    ident = F.evaluate(F.Idea(), sub, n_boot=200, return_detail=True)
    assert ident["out_of_fold"] == ident["baseline_W"], "identity arm is not a no-op"
    results["baseline_W_rt"] = ident["baseline_W"]
    results["oracle_column_W_rt"] = ident["oracle_column"]
    results["baseline_V_rt"] = ident["baseline_V"]

    # ---- descriptive only: the model swap, NOT an experimental comparison
    log("descriptive: old-W baseline (stt-async-v5), model swap only")
    old = F.load_substrate()
    old_ident = F.evaluate(F.Idea(), old, n_boot=200)
    results["descriptive_old_W"] = {
        "note": ("MODEL SWAP, NOT AN EXPERIMENTAL RESULT. Old W uses the paid "
                 "stt-async-v5 Soniox arm; W-rt uses the free stt-rt-v4. The "
                 "difference confounds model, decode path, pacing and "
                 "non-determinism, and is unpaired in any causal sense."),
        "old_W": old_ident["baseline_W"],
        "old_V": old_ident["baseline_V"],
        "old_oracle_column": old_ident["oracle_column"],
        "old_meta": old.meta,
    }
    del old, old_ident

    # ---- the arms
    for arm in ARMS:
        idea = VoteIdea(preps) if arm == "A" else ThresholdIdea(arm, preps)
        log(f"evaluating arm {arm}")
        res = F.evaluate(idea, sub, n_boot=n_boot, return_detail=True)
        entry = strip_detail(res)
        entry["adjusted_ci"] = adjusted_ci(res) if arm in FAMILY else None
        entry["city_sign_test"] = city_sign_test(res)
        entry["domination"] = domination(res)
        entry["exposure"] = (vote_exposure(sub, preps, K_OTHER) if arm == "A"
                             else exposure(sub, preps, arm, res, res["fold_params"]))
        entry["in_family"] = arm in FAMILY
        results["arms"][arm] = entry
        log("  " + F.summary_line(res))

    # ---- ungated controls
    for arm in ("O", "M"):
        idea = ThresholdIdea(arm, preps, fixed=0.0)
        log(f"evaluating control {arm}-all")
        res = F.evaluate(idea, sub, n_boot=n_boot, return_detail=True)
        results["controls"][f"{arm}-all"] = strip_detail(res)
        log("  " + F.summary_line(res))

    # ---- deployable threshold: refit once on all ten cities (reported, not shipped)
    results["deployable_threshold"] = {
        arm: fit_threshold(arm, sub.windows, preps) for arm in ("O", "O2", "M")}

    # ---- permutation nulls
    log(f"permutation nulls, {N_PERM} replicates each")
    results["controls"]["permutation"] = {
        "O": perm_null(sub, preps, "O", N_PERM),
        "M": perm_null(sub, preps, "M", N_PERM),
        "A": perm_null(sub, preps, "A", N_PERM, vote_k=K_OTHER),
    }

    # ---- k envelope, sensitivity only
    for k in K_ENVELOPE:
        res = F.evaluate(VoteIdea(preps, k), sub, n_boot=1000)
        results["sensitivity"][f"A_k={k}"] = {
            "wer": res["out_of_fold"]["wer"],
            "delta": res["vs_W"]["wer"]["delta"],
            "ci95": res["vs_W"]["wer"]["ci95"],
            "gates": res["gates"]}
        log(f"  A(k={k}) dWER={res['vs_W']['wer']['delta']:+.5f}")

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    log(f"wrote {OUT}")

    man = R.manifest(sub, conf)
    mp = R.rt_root() / "manifest.json"
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    log(f"wrote {mp} (stays outside git: it names verbatim-speech caches)")




# ---------------------------------------------------------------- diagnostics
# Everything below is POST-HOC and REPORTING-ONLY. It was written after the arms
# were scored, to explain the result; it is not a gate, it never touched a fitted
# parameter, and no arm is promoted on it. The column oracle it reads is hindsight.
def _auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    rsum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def diagnostics(sub: F.Substrate, preps: dict[str, Prep]) -> dict:
    """Does confidence discriminate the decision each arm has to make, on THIS
    benchmark, against the alignment-conditional column oracle?

    label 1 = the oracle wants the Soniox candidate at that column
    label 0 = the oracle wants what W-rt already has (epsilon for O, the majority
              token for M, W-rt's tie-break for A / O2)
    Columns where the oracle wants a third thing are counted separately and
    excluded from the AUROC, because they are not the decision the arm makes.
    """
    out = {}
    for arm in ARMS:
        scores, labels, other = [], [], 0
        for w in sub.windows:
            p = preps[w.item_id]
            for i, tok, c in p.elig[arm]:
                o = p.oracle[i]
                if o == tok:
                    scores.append(c)
                    labels.append(1)
                elif o == p.base[i]:
                    scores.append(c)
                    labels.append(0)
                else:
                    other += 1
        n1 = sum(labels)
        out[arm] = {
            "eligible": sum(len(preps[w.item_id].elig[arm]) for w in sub.windows),
            "oracle_wants_soniox": n1,
            "oracle_wants_W_rt": len(labels) - n1,
            "oracle_wants_a_third_thing": other,
            "prevalence": n1 / len(labels) if labels else None,
            "auroc_conf_predicts_oracle_wants_soniox": _auroc(scores, labels),
            "conf_mean_when_oracle_wants_soniox":
                (sum(s for s, y in zip(scores, labels) if y) / n1) if n1 else None,
            "conf_mean_when_oracle_wants_W_rt":
                (sum(s for s, y in zip(scores, labels) if not y) / (len(labels) - n1))
                if len(labels) - n1 else None,
        }
    return out


def perm_pvalue_A(sub: F.Substrate, preps: dict[str, Prep], observed: float,
                  n: int = N_PERM) -> dict:
    """Exact left-tail count of the arm-A permutation null (no fitting to redo)."""
    rng = random.Random(PERM_SEED)
    den = sum(len(w.ref) for w in sub.windows)
    base_num = sum(_lev(w.ref, w.w_tokens) for w in sub.windows)
    le = 0
    for _ in range(n):
        ov = permuted_conf(sub, preps, "A", rng)
        num = sum(_lev(w.ref, apply_vote_arm(w, preps[w.item_id], K_OTHER,
                                             ov[w.item_id])) for w in sub.windows)
        if (num - base_num) / den <= observed:
            le += 1
    return {"n": n, "n_at_or_below_observed": le, "p_one_sided": (le + 1) / (n + 1)}


def add_diagnostics() -> None:
    sub, conf = R.load_substrate_rt(strict=True)
    preps = prepare(sub, conf)
    res = json.loads(OUT.read_text())
    res["diagnostics"] = diagnostics(sub, preps)
    res["diagnostics"]["_note"] = (
        "POST-HOC, reporting only. Written after the arms were scored, never a gate, "
        "never visible to fit or apply. The column oracle is hindsight.")
    res["controls"]["permutation"]["A_exact"] = perm_pvalue_A(
        sub, preps, res["arms"]["A"]["vs_W"]["wer"]["delta"])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log(json.dumps(res["diagnostics"], indent=1, ensure_ascii=False))
    log(json.dumps(res["controls"]["permutation"]["A_exact"]))


if __name__ == "__main__":
    if "--diagnostics" in sys.argv:
        add_diagnostics()
    else:
        main()
