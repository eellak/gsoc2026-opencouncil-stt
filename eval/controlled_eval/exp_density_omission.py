#!/usr/bin/env python3
"""Experiment 2 of `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md`.

Does knowing how many people are talking make the omission detector see the lost
speaker inside simultaneous speech?

The primary comparator is NOT a coin. Beating random intervals would only show that
low-output regions predict transcript deletions, which is nearly circular - both are
functions of missing tokens of ours. So the comparator is the SAME rule with |S| forced
to 1: a pure duration-based under-transcription detector, on the same eligible
intervals, at the same merged-flag budget. The matched-random null is secondary.

Time axis: the whisper-turbo word stream, so these numbers sit beside round 2's A6
(regular 0.361 / 0.107, exclusive 0.376 / 0.134) and NOT beside round 1's 0.320, whose
axis came from Parakeet.

Writes results_density_omission.json (aggregates only, no transcript text).

Env: SC (cache dir) N_BOOT (10000) N_RANDOM (200)
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B                      # noqa: E402
from eval.controlled_eval import density_omission as D                # noqa: E402
from eval.controlled_eval.exp_parakeet_voter import (align_ops,       # noqa: E402
                                                     token_times)
from eval.controlled_eval.exp_speaker_fusion import active_intervals  # noqa: E402
from eval.controlled_eval.overlap_arms import sc, turbo_response      # noqa: E402
from eval.controlled_eval.parakeet_run import RUN_ID, target_items    # noqa: E402
from eval.controlled_eval.scoring import wtoks                        # noqa: E402

OURS = "oc-runpod-fixed-2026-08-10"
N_BOOT = int(os.environ.get("N_BOOT", "10000"))
N_RANDOM = int(os.environ.get("N_RANDOM", "200"))
OUT = Path(__file__).with_name("results_density_omission.json")
THRESHOLDS = {"primary": 1.0, "sens_low": 0.75, "sens_high": 1.25}


def log(m):
    print(m, flush=True)


# ----------------------------------------------------------------- window loading
def load_windows():
    items = target_items()
    out = []
    for it in items:
        d = turbo_response(it["item_id"])
        pk_t, pk_m = [], []
        for w in d["wordLevelTranscription"]:
            tt = wtoks(w["text"])
            if not tt:
                continue
            mid = (float(w["start"]) + float(w["end"])) / 2
            for t in tt:
                pk_t.append(t)
                pk_m.append(mid)
        reg, exc = d["diarization"], d["exclusiveDiarization"]
        audio_end = max([float(s["end"]) for s in reg + exc]
                        + ([pk_m[-1]] if pk_m else []) + [0.0])
        our = wtoks(it["hyp"][OURS])
        ref = wtoks(it["ref"])
        our_times, our_anch = token_times(our, pk_t, pk_m, audio_end)
        ref_times, ref_anch = token_times(ref, pk_t, pk_m, audio_end)
        out.append({
            "item_id": it["item_id"], "city": it["city_id"], "meeting": it["cluster"],
            "reg": reg, "exc": exc, "audio_end": audio_end,
            "our_times": our_times, "ref_times": ref_times,
            "ops": align_ops(ref, our), "n_our": len(our), "n_ref": len(ref),
            "anchors_our": our_anch, "anchors_ref": ref_anch,
        })
    return out


def rho_loco(wins) -> dict[str, float]:
    """Single-speaker token rate, leave-one-city-out.

    Numerator and denominator come from SINGLE-SPEAKER intervals of >= 1.5 s only, whose
    eligibility does not depend on rho, so there is no circularity. Overlap intervals
    are excluded on purpose: the omissions being hunted live there and would depress
    the rate that is supposed to detect them.
    """
    per_city: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    for w in wins:
        for s, e, sp in active_intervals(w["reg"]):
            if len(sp) != 1 or e - s < D.MIN_SPEECH_SEC:
                continue
            per_city[w["city"]][0] += D.count_in(w["our_times"], s, e)
            per_city[w["city"]][1] += e - s
    tot_n = sum(v[0] for v in per_city.values())
    tot_d = sum(v[1] for v in per_city.values())
    return {c: (tot_n - v[0]) / (tot_d - v[1]) for c, v in per_city.items()}


# ------------------------------------------------------------------------ scoring
def score(flag_lists, truth_lists) -> dict:
    tp = nf = nt = 0
    for f, t in zip(flag_lists, truth_lists):
        k, _, _ = D.match(f, t)
        tp += k
        nf += len(f)
        nt += len(t)
    return {"tp": tp, "n_flags": nf, "n_truth": nt,
            "precision": tp / nf if nf else None,
            "recall": tp / nt if nt else None}


def paired_ci(a_flags, b_flags, truths, clusters, metric="precision", seed=7):
    """Meeting-clustered CI on the difference of two corpus rates.

    The two rules flag different intervals, so the denominators are not shared; both
    rates are recomputed inside every replicate from the resampled meetings.
    """
    groups: dict = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)

    def rate(flags, idx):
        tp = num = 0
        for i in idx:
            k, _, _ = D.match(flags[i], truths[i])
            tp += k
            num += len(flags[i]) if metric == "precision" else len(truths[i])
        return tp / num if num else np.nan

    allidx = list(range(len(clusters)))
    point = rate(a_flags, allidx) - rate(b_flags, allidx)
    out = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pk = rng.integers(0, len(keys), len(keys))
        idx = [i for k in pk for i in groups[keys[k]]]
        out[r] = rate(a_flags, idx) - rate(b_flags, idx)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return {"delta": float(point), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_clusters": len(keys)}


def matched_null(obs_rows, flags, rng, caliper=D.CALIPER):
    """One draw of a caliper-matched null: same speaker cardinality, duration within
    +-25%, drawn from eligible intervals that are NOT flagged. Unmatched flags are
    dropped from this contrast and counted rather than being matched across strata."""
    pool = [r for r in obs_rows if (r[0], r[1]) not in set(flags)]
    picked, unmatched = [], 0
    used: set[int] = set()
    for s, e, in flags:
        dur = e - s
        cand = [i for i, r in enumerate(pool) if i not in used
                and abs((r[1] - r[0]) - dur) <= caliper * dur]
        if not cand:
            unmatched += 1
            continue
        j = int(rng.choice(cand))
        used.add(j)
        picked.append((pool[j][0], pool[j][1]))
    return picked, unmatched


def main() -> None:
    wins = load_windows()
    log(f"{len(wins)} windows, {len({w['meeting'] for w in wins})} meetings")
    rho = rho_loco(wins)
    log("rho_single (leave-one-city-out): "
        + json.dumps({k: round(v, 4) for k, v in sorted(rho.items())}))

    clusters = [w["meeting"] for w in wins]
    truths = [D.truth_events(w["ops"], w["ref_times"]) for w in wins]

    res = {
        "spec": "docs/specs/2026-08-16-overlap-speaker-arms-prereg.md",
        "run_id": RUN_ID, "n_windows": len(wins),
        "n_meetings": len(set(clusters)), "ours": OURS,
        "time_axis": "whisper-turbo word stream (comparable to round 2 A6, "
                     "not to round 1's 0.320)",
        "rho_single_loco": {k: round(v, 5) for k, v in sorted(rho.items())},
        "rho_single_pooled": round(
            sum(D.count_in(w["our_times"], s, e)
                for w in wins for s, e, sp in active_intervals(w["reg"])
                if len(sp) == 1 and e - s >= D.MIN_SPEECH_SEC)
            / max(sum(e - s for w in wins for s, e, sp in active_intervals(w["reg"])
                      if len(sp) == 1 and e - s >= D.MIN_SPEECH_SEC), 1e-9), 5),
        "n_truth_events": sum(len(t) for t in truths),
        "rules": {}, "contrasts": {}, "nulls": {}, "diagnostics": {},
    }

    obs = {tl: [D.observed(w[tl], w["our_times"], rho[w["city"]]) for w in wins]
           for tl in ("reg", "exc")}
    res["diagnostics"]["eligible_intervals"] = {
        tl: sum(len(r) for r in obs[tl]) for tl in obs}
    res["diagnostics"]["eligible_overlap_intervals"] = {
        tl: sum(1 for r in obs[tl] for x in r if x[2] >= 2) for tl in obs}

    def build(tl, thr, force_single=False):
        return [D.merge(D.flags_from(obs[tl][i], rho[wins[i]["city"]], thr,
                                     force_single=force_single))
                for i in range(len(wins))]

    old = [D.merge(D.old_rule_flags(w["reg"], w["our_times"])) for w in wins]
    old_exc = [D.merge(D.old_rule_flags(w["exc"], w["our_times"])) for w in wins]
    res["rules"]["old_rule (regular)"] = score(old, truths)
    res["rules"]["old_rule (exclusive)"] = score(old_exc, truths)

    flagsets = {}
    for label, thr in THRESHOLDS.items():
        for tl in ("reg", "exc"):
            f = build(tl, thr)
            flagsets[(label, tl)] = f
            name = f"speaker_aware {label} (thr={thr}) [{tl}]"
            res["rules"][name] = score(f, truths)
            res["rules"][name]["flag_seconds"] = round(
                sum(e - s for w in f for s, e in w), 1)
            res["rules"][name]["flags_touching_overlap"] = sum(
                1 for i, w in enumerate(f) for s, e in w
                if any(x[2] >= 2 and x[0] < e and s < x[1] for x in obs[tl][i]))

    primary = flagsets[("primary", "reg")]
    target = sum(len(x) for x in primary)
    thr1 = D.calibrate_budget([(obs["reg"][i], rho[wins[i]["city"]])
                               for i in range(len(wins))], target)
    dur_only = build("reg", thr1, force_single=True)
    res["rules"]["duration_only comparator [reg]"] = score(dur_only, truths)
    res["rules"]["duration_only comparator [reg]"]["calibrated_threshold"] = thr1
    res["rules"]["duration_only comparator [reg]"]["budget_target"] = target

    # ------------------------------------------------------------- PRIMARY contrast
    res["contrasts"]["PRIMARY speaker_aware - duration_only (precision)"] = paired_ci(
        primary, dur_only, truths, clusters, "precision")
    res["contrasts"]["speaker_aware - duration_only (recall)"] = paired_ci(
        primary, dur_only, truths, clusters, "recall")
    res["contrasts"]["speaker_aware - old_rule (precision)"] = paired_ci(
        primary, old, truths, clusters, "precision")
    res["contrasts"]["speaker_aware - old_rule (recall)"] = paired_ci(
        primary, old, truths, clusters, "recall")

    # ---------------------------------------------------------------- secondary null
    rng = np.random.default_rng(7)
    acc = {"tp": 0.0, "nf": 0.0, "unmatched": 0.0}
    acc_u = {"tp": 0.0, "nf": 0.0}
    for _ in range(N_RANDOM):
        for i, f in enumerate(primary):
            picked, un = matched_null(obs["reg"][i], f, rng)
            k, _, _ = D.match(D.merge(picked), truths[i])
            acc["tp"] += k
            acc["nf"] += len(picked)
            acc["unmatched"] += un
            # round 2's unstratified matched null, for comparability
            pool = obs["reg"][i]
            if f and pool:
                idx = rng.choice(len(pool), size=min(len(f), len(pool)),
                                 replace=False)
                rm = D.merge([(pool[j][0], pool[j][1]) for j in idx])
            else:
                rm = []
            k2, _, _ = D.match(rm, truths[i])
            acc_u["tp"] += k2
            acc_u["nf"] += len(rm)
    res["nulls"]["caliper_matched"] = {
        "precision": acc["tp"] / acc["nf"] if acc["nf"] else None,
        "mean_flags_per_draw": acc["nf"] / N_RANDOM,
        "mean_unmatched_flags_per_draw": acc["unmatched"] / N_RANDOM,
        "note": "exact speaker cardinality is implied by the caliper pool being the "
                "eligible intervals of the same window; flagged intervals excluded"}
    res["nulls"]["unstratified_matched (round 2 comparator)"] = {
        "precision": acc_u["tp"] / acc_u["nf"] if acc_u["nf"] else None,
        "mean_flags_per_draw": acc_u["nf"] / N_RANDOM}

    res["diagnostics"]["anchor_coverage_ours"] = (
        sum(w["anchors_our"] for w in wins) / max(sum(w["n_our"] for w in wins), 1))
    res["diagnostics"]["anchor_coverage_ref"] = (
        sum(w["anchors_ref"] for w in wins) / max(sum(w["n_ref"] for w in wins), 1))

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"-> {OUT}")
    for k, v in res["rules"].items():
        log(f"  {k:44s} flags={v['n_flags']:5d} P={v['precision']} R={v['recall']}")
    for k, v in res["contrasts"].items():
        log(f"  {k:52s} {v['delta']:+.4f} {[round(x,4) for x in v['ci95']]} "
            f"excl0={v['excludes_zero']}")
    log(json.dumps(res["nulls"], indent=1))


if __name__ == "__main__":
    main()
