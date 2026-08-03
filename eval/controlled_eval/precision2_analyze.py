#!/usr/bin/env python3
"""Analyse the precision-2 corpus pass against the frozen plan.

`docs/specs/precision2-corpus-analysis.md` fixes every rule below, including the
three-way decision bands and the noninferiority margin, before this ran. Numbers not in
that document are exploratory and are labelled so in the output.

M1  paired ratio of detector-defined overlap prevalence, fixed denominator
M2  paired robustness of the bucket association: Cbar per diarizer and their difference,
    both recomputed inside the SAME bootstrap replicate so the covariance survives
M3  turn density vs overlap as PREDICTORS, leave-one-meeting-out, not a causal contrast
M4  event geometry, descriptive

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/precision2_analyze.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as B  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))

R_EQUIV = (0.70, 1.30)      # M1 concordance band, frozen
R_HIGHER = 2.0              # M1 "materially higher", frozen
DELTA = 0.010               # M2 noninferiority margin, 1.0 WER point, frozen
MIN_TURN = 0.25             # M3 turn-boundary convention, frozen


def log(*a):
    print(*a, flush=True)


def boot_groups(clusters, n_boot, seed=7):
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        yield np.concatenate([groups[keys[k]] for k in pick])


def ci(v, lo=2.5, hi=97.5):
    return [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))]


def turn_rate(turns, speech_sec):
    """Speaker changes per minute of speech. Conventions frozen in the plan."""
    t = [x for x in sorted(turns, key=lambda z: z["start"])
         if float(x["end"]) - float(x["start"]) >= MIN_TURN]
    changes = sum(1 for a, b in zip(t, t[1:]) if a["speaker"] != b["speaker"])
    return changes / (speech_sec / 60) if speech_sec > 0 else 0.0


def buckets_from(vals):
    """Zero, then tertiles of the positive values. Frozen from diarization alone."""
    pos = sorted(v for v in vals if v > 0)
    if len(pos) < 3:
        return lambda v: 0 if v <= 0 else 1
    q1, q2 = pos[len(pos) // 3], pos[2 * len(pos) // 3]
    return lambda v: 0 if v <= 0 else (1 if v < q1 else (2 if v < q2 else 3))


def main():
    c1 = json.loads((SC / "overlap_features.json").read_text())["features"]
    p2raw = json.loads((SC / "precision2_corpus.json").read_text())
    p2 = {k: v for k, v in p2raw.items() if "feat" in v}

    report = B.load_report()
    providers = B.provider_ids(report)
    items = [it for it in B.common_items(report, providers)
             if it["item_id"] in c1 and it["item_id"] in p2]
    log(f"{len(items)} windows in both diarizers and all {len(providers)} systems")

    ids = [it["item_id"] for it in items]
    clusters = [it["meeting_id"] for it in items]
    ov1 = np.array([c1[i]["overlap_sec"] for i in ids])
    ov2 = np.array([p2[i]["feat"]["overlap_sec"] for i in ids])
    sp1 = np.array([c1[i]["speech_sec"] for i in ids])
    sp2 = np.array([p2[i]["feat"]["speech_sec"] for i in ids])
    trate = np.array([turn_rate(p2[i]["turns"], p2[i]["feat"]["speech_sec"]) for i in ids])

    # per-window (errors, ref_words) for every system, from the frozen scorer
    import scoring as S
    err = {}
    nref = np.array([len(S.wtoks(it["ref"])) for it in items], dtype=float)
    for p in providers:
        err[p] = np.array([S.edist(S.wtoks(it["ref"]), S.wtoks(it["hyp"][p]))
                           for it in items], dtype=float)

    out = {"plan": "docs/specs/precision2-corpus-analysis.md",
           "n_windows": len(items), "n_meetings": len(set(clusters)),
           "n_boot": N_BOOT, "providers": providers}

    # ------------------------------------------------------------------------- M1
    # Fixed denominator: community-1 speech time. Otherwise a difference in how the two
    # models segment speech would masquerade as a difference in overlap.
    den = sp1.sum()
    r_hat = (ov2.sum() / den) / (ov1.sum() / den)
    rs = []
    for idx in boot_groups(clusters, N_BOOT):
        a, b = ov1[idx].sum(), ov2[idx].sum()
        rs.append(b / a if a > 0 else np.nan)
    r_ci = ci(rs)
    verdict = ("practically concordant" if R_EQUIV[0] <= r_ci[0] and r_ci[1] <= R_EQUIV[1]
               else "materially higher" if r_ci[0] > R_HIGHER else "inconclusive")
    out["M1_prevalence"] = {
        "community1_frac_of_speech": float(ov1.sum() / sp1.sum()),
        "precision2_frac_of_speech": float(ov2.sum() / sp2.sum()),
        "ratio_fixed_denominator": float(r_hat), "ci95": r_ci,
        "equivalence_band": list(R_EQUIV), "higher_threshold": R_HIGHER,
        "verdict": verdict,
        "note": ("NOT an independent estimate. Same audio, same model family, and the "
                 "listening audit showed both count miked speakers only."),
    }

    # ------------------------------------------------------------------------- M2
    bfun1, bfun2 = buckets_from(ov1), buckets_from(ov2)
    b1 = np.array([bfun1(v) for v in ov1])
    b2 = np.array([bfun2(v) for v in ov2])

    def cbar(idx, bk):
        """Mean over systems of (WER in the top bucket - WER in the zero bucket)."""
        hi, zo = idx[bk[idx] == 3], idx[bk[idx] == 0]
        if len(hi) == 0 or len(zo) == 0 or nref[hi].sum() == 0 or nref[zo].sum() == 0:
            return np.nan
        return float(np.mean([err[p][hi].sum() / nref[hi].sum()
                              - err[p][zo].sum() / nref[zo].sum() for p in providers]))

    allidx = np.arange(len(items))
    c1bar, c2bar = cbar(allidx, b1), cbar(allidx, b2)
    d1, d2, dd = [], [], []
    for idx in boot_groups(clusters, N_BOOT):
        x, y = cbar(idx, b1), cbar(idx, b2)
        d1.append(x)
        d2.append(y)
        dd.append(y - x)
    c2_ci, dd_ci = ci(d2), ci(dd)
    supported = c2_ci[0] > 0 and dd_ci[0] > -DELTA
    contradicted = c2_ci[1] <= 0 or dd_ci[1] < -DELTA
    out["M2_bucket_robustness"] = {
        "Cbar_community1": c1bar, "ci95": ci(d1),
        "Cbar_precision2": c2bar, "ci95_precision2": c2_ci,
        "delta_p2_minus_c1": float(c2bar - c1bar), "ci95_delta": dd_ci,
        "margin_delta": DELTA,
        "verdict": ("supported" if supported else
                    "contradicted" if contradicted else "inconclusive"),
        "note": ("Robustness to measurement, NOT independent replication: same windows, "
                 "same references, same hypotheses, related diarizers."),
        "per_system_precision2": {
            p: float(err[p][b2 == 3].sum() / nref[b2 == 3].sum()
                     - err[p][b2 == 0].sum() / nref[b2 == 0].sum()) for p in providers},
    }

    # ---------------------------------------------- M2b continuous, no bucket edges
    def wls_slope(x, idx, p):
        """Weighted slope of window WER on sqrt(overlap seconds). Bucket-free check."""
        w = nref[idx]
        y = err[p][idx] / np.maximum(w, 1)
        X = np.column_stack([np.ones(len(idx)), np.sqrt(x[idx])])
        W = np.diag(w)
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            return np.nan
        return float(beta[1])
    out["M2b_continuous"] = {
        p: {"slope_community1": wls_slope(ov1, allidx, p),
            "slope_precision2": wls_slope(ov2, allidx, p)} for p in providers}

    # ------------------------------------------------------------------------- M3
    # Predictive, not causal. Leave-one-meeting-out; loss is reference-word-weighted
    # squared error on window WER. Base = intercept + speech fraction.
    speech_frac = sp2 / np.array([p2[i]["measured_dur"] for i in ids])
    feats = {
        "base": [np.ones(len(items)), speech_frac],
        "base+overlap": [np.ones(len(items)), speech_frac, np.sqrt(ov2)],
        "base+turns": [np.ones(len(items)), speech_frac, trate],
        "base+overlap+turns": [np.ones(len(items)), speech_frac, np.sqrt(ov2), trate],
    }
    meetings = np.array(clusters)
    # Per-window weighted squared error, kept per model so the incremental values can be
    # resampled by meeting. A point estimate of U with no interval says nothing.
    perwin = {name: np.zeros(len(items)) for name in feats}
    for name, cols in feats.items():
        X = np.column_stack(cols)
        for m in sorted(set(clusters)):
            te = meetings == m
            tr = ~te
            w = nref[tr]
            for p in providers:
                y = err[p] / np.maximum(nref, 1)
                try:
                    beta = np.linalg.solve((X[tr] * w[:, None]).T @ X[tr],
                                           (X[tr] * w[:, None]).T @ y[tr])
                except np.linalg.LinAlgError:
                    continue
                perwin[name][te] += nref[te] * (y[te] - X[te] @ beta) ** 2
    denom = len(providers) * nref.sum()
    loo = {k: float(v.sum() / denom) for k, v in perwin.items()}

    uo = perwin["base+turns"] - perwin["base+overlap+turns"]
    ut = perwin["base+overlap"] - perwin["base+overlap+turns"]
    uo_b, ut_b = [], []
    for idx in boot_groups(clusters, N_BOOT, seed=11):
        d = len(providers) * nref[idx].sum()
        uo_b.append(uo[idx].sum() / d)
        ut_b.append(ut[idx].sum() / d)
    out["M3_prediction"] = {
        "loo_meeting_loss": loo,
        "U_overlap_given_turns": float(uo.sum() / denom), "ci95_U_overlap": ci(uo_b),
        "U_turns_given_overlap": float(ut.sum() / denom), "ci95_U_turns": ci(ut_b),
        "note": ("Out-of-sample incremental value, positive = the variable helps. "
                 "Identifies NEITHER variable as causal; overlap and turn density are "
                 "not independently measured, and turn boundaries are partly created by "
                 "overlap itself."),
    }

    # ------------------------------------------------------------------------- M4
    durs, per_win = [], []
    for i in ids:
        ev = []
        for t in p2[i]["turns"]:
            ev.append((float(t["start"]), float(t["end"]), t["speaker"]))
        # count events where >=2 distinct speakers are active
        pts = sorted({x for s, e, _ in ev for x in (s, e)})
        cur, n_ev = None, 0
        for a, b in zip(pts, pts[1:]):
            mid = (a + b) / 2
            k = len({sp for s, e, sp in ev if s <= mid < e})
            if k >= 2:
                cur = (a if cur is None else cur[0], b) if cur is None else (cur[0], b)
            elif cur is not None:
                durs.append(cur[1] - cur[0])
                n_ev += 1
                cur = None
        if cur is not None:
            durs.append(cur[1] - cur[0])
            n_ev += 1
        per_win.append(n_ev)
    out["M4_event_geometry"] = {
        "n_events": len(durs),
        "event_dur_sec": {"median": float(np.median(durs)) if durs else None,
                          "p25": float(np.percentile(durs, 25)) if durs else None,
                          "p75": float(np.percentile(durs, 75)) if durs else None,
                          "p95": float(np.percentile(durs, 95)) if durs else None},
        "events_per_window": {"mean": float(np.mean(per_win)),
                              "median": float(np.median(per_win))},
        "note": ("Descriptive. Does NOT replace the preregistered uniform(1.5,3.0) dose: "
                 "these durations inherit precision-2's smoothing and its missed "
                 "low-SIR tails."),
    }

    # ---------------------------------------------- disagreement map, for the audit
    strata = collections.Counter()
    for i, iid in enumerate(ids):
        strata[("c1+" if ov1[i] > 0 else "c1-") + ("p2+" if ov2[i] > 0 else "p2-")] += 1
    out["disagreement_strata"] = dict(strata)

    dst = ROOT / "eval/controlled_eval/results_precision2_corpus.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"-> {dst}")
    m1, m2, m3 = out["M1_prevalence"], out["M2_bucket_robustness"], out["M3_prediction"]
    log(f"M1 ratio {m1['ratio_fixed_denominator']:.2f} CI{[round(x,2) for x in m1['ci95']]}"
        f" -> {m1['verdict']}")
    log(f"M2 Cbar c1 {m2['Cbar_community1']:+.4f}  p2 {m2['Cbar_precision2']:+.4f}  "
        f"delta {m2['delta_p2_minus_c1']:+.4f} CI{[round(x,4) for x in m2['ci95_delta']]}"
        f" -> {m2['verdict']}")
    log(f"M3 U_overlap|turns {m3['U_overlap_given_turns']:+.6f} "
        f"CI{[round(x, 6) for x in m3['ci95_U_overlap']]}")
    log(f"   U_turns|overlap {m3['U_turns_given_overlap']:+.6f} "
        f"CI{[round(x, 6) for x in m3['ci95_U_turns']]}")
    log(f"strata {out['disagreement_strata']}")


if __name__ == "__main__":
    main()
