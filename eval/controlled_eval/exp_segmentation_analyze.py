#!/usr/bin/env python3
"""Score the segmentation experiment against its frozen decision rule.

`docs/specs/segmentation-experiment-preregistration.md` fixes the primary contrast, the
stratum, the thresholds and the three-way verdict. Nothing here is chosen after seeing a
number.

The primary contrast is **arm 2 minus arm 3**, not arm 2 minus arm 1. Arm 3 has the same
chunk count and the same padding as arm 2 with boundaries placed away from speaker
changes, so 2 − 1 confounds "cut at speaker changes" with "cut at all", and only 2 − 3
isolates the placement.

Usage:
  SC=~/.cache/oc-overlap python eval/controlled_eval/exp_segmentation_analyze.py
Env: SC HYPS N_BOOT
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scoring as S  # noqa: E402

ROOT = Path("/home/harold/opencouncil-fine-tuning")
SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-overlap"))
HYPS = Path(os.environ.get("HYPS", SC / "segmentation_hyps.json"))
N_BOOT = int(os.environ.get("N_BOOT", "10000"))

GATE_OVERALL = 0.010      # 1.0 WER point, frozen
GATE_STRATUM = 0.020      # 2.0 points in the top turn-density tercile, frozen
REGRESSION = 0.005        # "material" regression in the bottom tercile, frozen


def log(*a):
    print(*a, flush=True)


def counts_for(items, hyps, sysname, arm):
    return [(S.edist(S.wtoks(it["ref"]), S.wtoks(hyps[f"{sysname}|{it['item_id']}|{arm}"])),
             len(S.wtoks(it["ref"]))) for it in items]


def contrast(items, hyps, sysname, a, b, label=""):
    """WER(a) - WER(b). Negative means arm a is better."""
    ca, cb = counts_for(items, hyps, sysname, a), counts_for(items, hyps, sysname, b)
    r = S.cluster_bootstrap(ca, cb, [it["meeting_id"] for it in items], n_boot=N_BOOT)
    r.update({"label": label or f"{a} - {b}", "wer_a": S.wer(ca), "wer_b": S.wer(cb),
              "n": len(items)})
    return r


def main():
    plan = json.loads((SC / "segmentation_plan.json").read_text())
    hyps = json.loads(HYPS.read_text())
    systems = sorted({k.split("|")[0] for k in hyps})
    arms = ("arm1", "arm2", "arm3")

    items = [it for it in plan["items"]
             if all(f"{s}|{it['item_id']}|{a}" in hyps for s in systems for a in arms)]
    log(f"{len(items)} windows complete in every arm and system "
        f"({len(plan['items'])} planned)")

    # terciles of turn density, frozen from the diarization alone
    rates = sorted(it["turn_rate_per_min"] for it in items)
    q1, q2 = rates[len(rates) // 3], rates[2 * len(rates) // 3]
    strata = {"low": [it for it in items if it["turn_rate_per_min"] < q1],
              "mid": [it for it in items if q1 <= it["turn_rate_per_min"] < q2],
              "high": [it for it in items if it["turn_rate_per_min"] >= q2]}
    log("turn density terciles: " + ", ".join(
        f"{k} n={len(v)}" for k, v in strata.items()) + f"  (cuts {q1:.1f}, {q2:.1f}/min)")

    out = {"preregistration": "docs/specs/segmentation-experiment-preregistration.md",
           "n_windows": len(items), "n_meetings": len({it["meeting_id"] for it in items}),
           "n_boot": N_BOOT, "tercile_cuts_per_min": [q1, q2], "systems": {}}

    for sysname in systems:
        blk = {
            "wer": {a: S.wer(counts_for(items, hyps, sysname, a)) for a in arms},
            "primary_arm2_minus_arm3": contrast(items, hyps, sysname, "arm2", "arm3"),
            "arm2_minus_arm1": contrast(items, hyps, sysname, "arm2", "arm1"),
            "arm3_minus_arm1": contrast(items, hyps, sysname, "arm3", "arm1"),
            "by_turn_density": {
                k: contrast(v, hyps, sysname, "arm2", "arm3", f"arm2-arm3 [{k}]")
                for k, v in strata.items() if len(v) >= 5},
            "by_city": {},
        }
        cities = collections.defaultdict(list)
        for it in items:
            cities[it["city_id"]].append(it)
        for c, v in sorted(cities.items()):
            if len(v) >= 5:
                blk["by_city"][c] = contrast(v, hyps, sysname, "arm2", "arm3")["delta"]
        # leave-one-city-out, so no single city can carry the result
        loco = {}
        for c in cities:
            sub = [it for it in items if it["city_id"] != c]
            loco[c] = contrast(sub, hyps, sysname, "arm2", "arm3")["delta"]
        blk["leave_one_city_out_worst"] = max(loco.values())
        blk["leave_one_city_out_best"] = min(loco.values())
        out["systems"][sysname] = blk

    # ------------------------------------------------------------------ frozen gate
    for sysname, blk in out["systems"].items():
        p = blk["primary_arm2_minus_arm3"]
        hi = blk["by_turn_density"].get("high")
        lo = blk["by_turn_density"].get("low")
        # improvement is a NEGATIVE delta; the gate is on the upper CI bound
        overall_pass = p["ci95"][1] <= -GATE_OVERALL
        stratum_pass = bool(hi and hi["ci95"][1] <= -GATE_STRATUM)
        regressed = bool(lo and lo["ci95"][0] >= REGRESSION)
        ruled_out = (p["ci95"][0] > -GATE_OVERALL
                     and (not hi or hi["ci95"][0] > -GATE_STRATUM))
        blk["gate"] = {
            "continue": bool((overall_pass or stratum_pass) and not regressed),
            "stop": bool(ruled_out),
            "inconclusive": bool(not (overall_pass or stratum_pass) and not ruled_out),
            "regression_in_low_stratum": regressed,
            "thresholds": {"overall": GATE_OVERALL, "stratum": GATE_STRATUM,
                           "regression": REGRESSION},
            "note": ("Confirmatory pilot on the windows that generated the turn-density "
                     "hypothesis, not external validation."),
        }

    dst = ROOT / "eval/controlled_eval/results_segmentation.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    log(f"-> {dst}\n")
    for sysname, blk in out["systems"].items():
        p = blk["primary_arm2_minus_arm3"]
        log(f"{sysname}: arm1 {blk['wer']['arm1']:.4f}  arm2 {blk['wer']['arm2']:.4f}  "
            f"arm3 {blk['wer']['arm3']:.4f}")
        log(f"  PRIMARY arm2-arm3 {p['delta']:+.4f} CI[{p['ci95'][0]:+.4f},{p['ci95'][1]:+.4f}]")
        for k in ("low", "mid", "high"):
            c = blk["by_turn_density"].get(k)
            if c:
                log(f"    {k:4s} {c['delta']:+.4f} CI[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]")
        g = blk["gate"]
        log(f"  verdict: {'CONTINUE' if g['continue'] else 'STOP' if g['stop'] else 'INCONCLUSIVE'}"
            + ("  (regression in low stratum)" if g["regression_in_low_stratum"] else ""))


if __name__ == "__main__":
    main()
