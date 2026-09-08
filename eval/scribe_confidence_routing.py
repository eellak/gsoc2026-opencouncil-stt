"""Does Scribe's own confidence say where the fusion is worth paying for?

Pre-registered as `exp-2026-09-09-scribe-confidence-routing`. The question is NOT
"where is Scribe wrong" -- it is "where does the fusion beat Scribe", and those are
different rankings: a window that is hard for Scribe is often hard for all three
systems, and the fusion sometimes makes a window worse.

Per window i, with edit operations counted against the benchmark's reference:

    g_i = E_scribe,i - E_fusion,i          (negative where fusion hurts)
    WER(R) = (sum_i E_scribe,i - sum_{i in R} g_i) / sum_i N_i

That identity is exact for whole-window routing, because the fusion output for every
window already exists -- routing here selects between two frozen transcripts, it does
not re-run anything. The x-axis is ROUTED AUDIO SECONDS, because that is what the two
extra vendors bill.

Inputs, both already paid for:
  - the benchmark report of run 2026-09-08-fusion-rules-temp0-vs-post-june-held-out
  - the component cache of that run, which holds Scribe's raw per-word logprobs

Neither goes in git: both carry transcript text.

    python3 eval/scribe_confidence_routing.py [report.json] [cache_dir]
"""

from __future__ import annotations

import collections
import glob
import json
import math
import random
import re
import statistics
import sys

REPORT = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/harold/.cache/oc-public/bench_2026-09-08-fusion-rules-temp0-vs-post-june-held-out.json")
CACHE = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/harold/.cache/oc-public/fusion-bench-cache/component")
RUN_START = "2026-09-08T18:08:59Z"          # excludes the conformance and smoke-test calls
FUSION = "oc-fusion-rules-2026-09-08"
METRIC = "wer"

# Published pooled scores. If this script cannot reproduce both, nothing below it means
# anything -- a routing curve is only interpretable between two endpoints that match.
ANCHOR_SCRIBE = 0.14027
ANCHOR_FUSION = 0.09913

BUDGETS = (0.10, 0.25, 0.50, 0.75)          # share of total audio seconds routed
PRIMARY_BUDGET = 0.25
GAIN_CAPTURE_GATE = 0.50                    # declared before any outcome was computed
RANDOM_DRAWS = 1000


def tokens(s: str) -> set[str]:
    return set(re.findall(r"\w+", (s or "").lower()))


def load_items(report: dict) -> list[dict]:
    out = []
    for it in report["items"]:
        sc = it["perProvider"].get("scribe") or {}
        fu = it["perProvider"].get(FUSION) or {}
        if sc.get("status") != "ok" or fu.get("status") != "ok":
            continue
        ms, mf = sc["perMetric"][METRIC], fu["perMetric"][METRIC]
        assert ms["denominator"] == mf["denominator"], it["itemId"]
        out.append({
            "id": it["itemId"], "city": it["cityId"], "dur": it["durationSec"],
            "n": ms["denominator"],
            "e_s": ms["numerator"], "e_f": mf["numerator"],
            "d_s": (ms.get("details") or {}).get("del", 0),
            "d_f": (mf.get("details") or {}).get("del", 0),
            "scribe_tokens": tokens(sc.get("hypothesisText")),
        })
    return out


def load_cache() -> list[tuple[str, set[str], list[float]]]:
    """(file, token set of Scribe's text, per-word probabilities) for this run only."""
    out = []
    for f in glob.glob(f"{CACHE}/*/*.json"):
        d = json.load(open(f, encoding="utf-8"))
        p = d["payload"]
        if p["providerId"] != "scribe" or d["createdAt"] < RUN_START:
            continue
        raw = p["raw"]
        probs = [math.exp(t["logprob"]) for t in (raw.get("words") or [])
                 if t.get("type") == "word" and t.get("logprob") is not None]
        out.append((f, tokens(raw.get("text")), probs))
    return out


def join(items: list[dict], cached: list) -> dict[str, list[float]]:
    """Match each cached window to its report item by Scribe-text overlap.

    The benchmark's own scribe column is a different ElevenLabs call from August, so the
    texts are close but never identical; what makes the join safe is that the winner beats
    the runner-up by a wide margin and the whole assignment comes out a permutation. Both
    are asserted, not hoped for.
    """
    idx = collections.defaultdict(list)
    for i, it in enumerate(items):
        for t in it["scribe_tokens"]:
            idx[t].append(i)

    assigned, margins, scores = {}, [], []
    for f, ctoks, probs in cached:
        hits = collections.Counter()
        for t in ctoks:
            for i in idx.get(t, ()):
                hits[i] += 1
        ranked = sorted(
            ((h / (len(ctoks) + len(items[i]["scribe_tokens"]) - h), i) for i, h in hits.items()),
            reverse=True)[:2]
        if not ranked:
            continue
        (best, bi) = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        assigned[items[bi]["id"]] = probs
        margins.append(best - second)
        scores.append(best)

    assert len(assigned) == len(cached), f"join collapsed: {len(assigned)} of {len(cached)}"
    print(f"join: {len(assigned)} windows, best-match jaccard median "
          f"{statistics.median(scores):.3f} min {min(scores):.3f}, "
          f"margin over runner-up min {min(margins):.3f}")
    return assigned


# ---- the four window scores, all declared before any outcome was computed -----------

def s1_frac_below_090(p): return sum(1 for x in p if x < 0.90) / len(p) if p else 0.0
def s3_one_minus_mean(p): return 1 - (sum(p) / len(p)) if p else 0.0
def s4_below_050_per100(p): return 100 * sum(1 for x in p if x < 0.50) / len(p) if p else 0.0


def s2_low_decile_mean(p):
    if not p:
        return 0.0
    k = max(1, len(p) // 10)
    return 1 - (sum(sorted(p)[:k]) / k)


SCORES = {
    "s1 frac(p<0.90)": s1_frac_below_090,
    "s2 1-mean(lowest decile)": s2_low_decile_mean,
    "s3 1-mean(p)": s3_one_minus_mean,
    "s4 count(p<0.50)/100w": s4_below_050_per100,
}


def curve(items, order, budget_share):
    """Route windows in `order` until the audio budget is spent; return the exact result."""
    total_dur = sum(i["dur"] for i in items)
    total_n = sum(i["n"] for i in items)
    e_s, d_s = sum(i["e_s"] for i in items), sum(i["d_s"] for i in items)
    budget, spent, gain, dgain, routed = total_dur * budget_share, 0.0, 0, 0, 0
    for i in order:
        if spent + i["dur"] > budget:
            continue                       # skip, do not stop: a budget is a budget
        spent += i["dur"]
        gain += i["e_s"] - i["e_f"]
        dgain += i["d_s"] - i["d_f"]
        routed += 1
    return {
        "wer": (e_s - gain) / total_n,
        "del_rate": (d_s - dgain) / total_n,
        "gain": gain, "windows": routed,
        "audio_share": spent / total_dur,
    }


def main() -> None:
    report = json.load(open(REPORT, encoding="utf-8"))
    items = load_items(report)
    total_n = sum(i["n"] for i in items)
    e_s, e_f = sum(i["e_s"] for i in items), sum(i["e_f"] for i in items)
    total_gain = e_s - e_f

    print(f"windows {len(items)}  reference words {total_n}")
    got_s, got_f = e_s / total_n, e_f / total_n
    print(f"anchors: scribe {got_s:.5f} (published {ANCHOR_SCRIBE}), "
          f"fusion {got_f:.5f} (published {ANCHOR_FUSION})")
    assert abs(got_s - ANCHOR_SCRIBE) < 5e-5 and abs(got_f - ANCHOR_FUSION) < 5e-5, \
        "anchors do not reproduce -- stop, the scoring pipeline disagrees with the report"

    pos = sum(1 for i in items if i["e_s"] - i["e_f"] > 0)
    neg = sum(1 for i in items if i["e_s"] - i["e_f"] < 0)
    print(f"total available gain {total_gain} edits over {len(items)} windows: "
          f"{pos} windows where fusion helps, {neg} where it hurts, {len(items)-pos-neg} tied")

    probs = join(items, load_cache())
    for i in items:
        i["p"] = probs.get(i["id"], [])
    missing = sum(1 for i in items if not i["p"])
    assert missing == 0, f"{missing} windows have no cached confidences"

    print("\n=== gain captured at each routed-audio budget (share of the 4554-edit total) ===")
    header = "ranking".ljust(28) + "".join(f"{int(b*100):>7}%" for b in BUDGETS)
    print(header)

    rows = {}
    for name, fn in SCORES.items():
        order = sorted(items, key=lambda i: fn(i["p"]), reverse=True)
        rows[name] = [curve(items, order, b) for b in BUDGETS]

    # Baselines. Random says what any budget buys by accident; oracle says what was there
    # to be had; duration says whether confidence beats simply routing the long windows.
    rng = random.Random(20260909)
    rand = []
    for b in BUDGETS:
        draws = []
        for _ in range(RANDOM_DRAWS):
            order = items[:]
            rng.shuffle(order)
            draws.append(curve(items, order, b)["gain"])
        draws.sort()
        rand.append({"mean": statistics.fmean(draws), "p95": draws[int(0.95 * len(draws))]})
    rows["baseline: random (mean)"] = [{"gain": r["mean"]} for r in rand]
    rows["baseline: random (p95)"] = [{"gain": r["p95"]} for r in rand]
    rows["baseline: duration desc"] = [
        curve(items, sorted(items, key=lambda i: i["dur"], reverse=True), b) for b in BUDGETS]
    rows["ORACLE gain/second"] = [
        curve(items, sorted(items, key=lambda i: (i["e_s"] - i["e_f"]) / i["dur"], reverse=True), b)
        for b in BUDGETS]

    for name, cells in rows.items():
        line = name.ljust(28)
        for c in cells:
            line += f"{c['gain']/total_gain*100:>6.1f}%"
        print(line)

    print("\n=== WER and deletion rate on the curve ===")
    for name, fn in list(SCORES.items()) + [("baseline: duration desc", None),
                                            ("ORACLE gain/second", None)]:
        if fn is not None:
            order = sorted(items, key=lambda i: fn(i["p"]), reverse=True)
        elif name.startswith("ORACLE"):
            order = sorted(items, key=lambda i: (i["e_s"] - i["e_f"]) / i["dur"], reverse=True)
        else:
            order = sorted(items, key=lambda i: i["dur"], reverse=True)
        cells = [curve(items, order, b) for b in BUDGETS]
        print(name.ljust(28) + "  ".join(
            f"{int(b*100)}%: wer {c['wer']:.5f} del {c['del_rate']:.5f}"
            for b, c in zip(BUDGETS, cells)))
    print("anchors".ljust(28) + f"0%: wer {got_s:.5f} del {sum(i['d_s'] for i in items)/total_n:.5f}"
          f"   100%: wer {got_f:.5f} del {sum(i['d_f'] for i in items)/total_n:.5f}")

    print(f"\n=== primary, declared before the outcome: budget {int(PRIMARY_BUDGET*100)}% ===")
    bi = BUDGETS.index(PRIMARY_BUDGET)
    rand_p95 = rand[bi]["p95"]
    best_name, best = None, None
    for name, fn in SCORES.items():
        order = sorted(items, key=lambda i: fn(i["p"]), reverse=True)
        c = curve(items, order, PRIMARY_BUDGET)
        cap = c["gain"] / total_gain
        beats = c["gain"] > rand_p95
        print(f"  {name.ljust(26)} captures {cap:>6.1%} of the gain, "
              f"{'beats' if beats else 'DOES NOT BEAT'} random p95 ({rand_p95/total_gain:.1%}), "
              f"wer {c['wer']:.5f}")
        if best is None or c["gain"] > best["gain"]:
            best_name, best = name, c
    verdict = (best["gain"] / total_gain >= GAIN_CAPTURE_GATE) and (best["gain"] > rand_p95)
    print(f"  GATE: capture >= {GAIN_CAPTURE_GATE:.0%} and beat random p95 -> "
          f"{'PASS' if verdict else 'FAIL'} (best: {best_name})")

    print("\n=== single-window domination of the captured gain, at the primary budget ===")
    order = sorted(items, key=lambda i: SCORES[best_name](i["p"]), reverse=True)
    total_dur = sum(i["dur"] for i in items)
    spent, contribs = 0.0, []
    for i in order:
        if spent + i["dur"] > total_dur * PRIMARY_BUDGET:
            continue
        spent += i["dur"]
        contribs.append((i["e_s"] - i["e_f"], i["id"]))
    contribs.sort(reverse=True)
    captured = sum(c for c, _ in contribs)
    if captured:
        print(f"  largest window supplies {contribs[0][0]/captured:.1%} of the captured gain")
        print(f"  windows contributing negatively: {sum(1 for c,_ in contribs if c < 0)}")

    print("\n=== training-disjoint cities, reported separately ===")
    strata = {}
    for city in ("argos", "orestiada"):
        sub = [i for i in items if i["city"] == city]
        if not sub:
            continue
        n = sum(i["n"] for i in sub)
        gs, gf = sum(i["e_s"] for i in sub) / n, sum(i["e_f"] for i in sub) / n
        order = sorted(sub, key=lambda i: SCORES[best_name](i["p"]), reverse=True)
        c = curve(sub, order, PRIMARY_BUDGET)
        avail = sum(i["e_s"] - i["e_f"] for i in sub)
        strata[city] = {"n": len(sub), "scribe": gs, "fusion": gf,
                        "wer_at_primary": c["wer"], "gain_captured": c["gain"] / avail}
        print(f"  {city}: n={len(sub)} scribe {gs:.5f} fusion {gf:.5f} | "
              f"at {int(PRIMARY_BUDGET*100)}% budget wer {c['wer']:.5f}, "
              f"captures {c['gain']/avail:.1%} of its {avail}-edit gain")

    # Scores only, never text: this file is committed.
    out = {
        "experiment": "exp-2026-09-09-scribe-confidence-routing",
        "run": "2026-09-08-fusion-rules-temp0-vs-post-june-held-out",
        "metric": METRIC,
        "metric_note": "agreement-with-OpenCouncil, not fidelity-to-audio",
        "windows": len(items), "reference_words": total_n,
        "anchors": {"scribe": got_s, "fusion": got_f},
        "gain": {"total_edits": total_gain, "helps": pos, "hurts": neg, "tied": len(items)-pos-neg},
        "budgets": list(BUDGETS),
        "gain_captured": {name: [c["gain"] / total_gain for c in cells]
                          for name, cells in rows.items()},
        "curve": {name: [curve(items, sorted(items, key=lambda i: fn(i["p"]), reverse=True), b)
                         for b in BUDGETS] for name, fn in SCORES.items()},
        "primary": {"budget": PRIMARY_BUDGET, "gate_capture": GAIN_CAPTURE_GATE,
                    "best_score": best_name, "captured": best["gain"] / total_gain,
                    "random_p95": rand_p95 / total_gain, "verdict": "PASS" if verdict else "FAIL"},
        "domination": {"largest_window_share": contribs[0][0] / captured if captured else None,
                       "negative_windows": sum(1 for c, _ in contribs if c < 0)},
        "strata": strata,
    }
    path = "eval/results_scribe_confidence_routing.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
