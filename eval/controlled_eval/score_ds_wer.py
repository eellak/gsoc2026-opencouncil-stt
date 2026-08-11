"""Score every benchmark provider with DS-WER on the frozen 39-window manifest.

`exp-2026-08-12-ds-wer`, proposal Milestone 2. The metric is `scripts/ds_wer.py`,
the term lists are frozen in `research/ds_wer/terms/` and were committed before this
ran, and the windows are `research/eval-freeze-2026-08/manifest.json`.

Two analyses, ordered here **before** either was computed, so that neither can be
chosen for passing:

  primary      all 39 windows
  sensitivity  the 39 minus the two windows with the largest reference term count

The second is the roll-call check. `docs/reports/2026-08-11-error-analysis-vs-scribe.md`
found two name-dense windows (a roll call and an agenda reading) carrying 40 of the
193 errors in the worst eight, so a domain-term metric could be almost entirely a
report on those two windows. "The two largest by reference term count" is a rule
over the reference and the frozen term list only - no provider output enters it.

Hypotheses come from the public benchmark report and stay under `$SC`.

    SC=~/.cache/oc-public .venv-eval/bin/python -m eval.controlled_eval.score_ds_wer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/harold/opencouncil-fine-tuning")
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import bench_data as B  # noqa: E402
from eval.controlled_eval.scoring import cluster_bootstrap  # noqa: E402
from scripts.ds_wer import TermList, aggregate, ds_wer  # noqa: E402

MANIFEST = ROOT / "research/eval-freeze-2026-08/manifest.json"
TERMS_DIR = ROOT / "research/ds_wer/terms"
OUT = Path.home() / ".cache/oc-public/ds-wer"

# The proposal's baseline for Milestone 2, and the systems it is worth reporting
# next to. `oc-runpod-fixed-2026-08-10` is artifact-ct2-fixed on an A4000 - the same
# run the headline benchmark number comes from.
PROVIDERS = {
    "ours (artifact-ct2-fixed)": "oc-runpod-fixed-2026-08-10",
    "Gladia (Milestone 2 baseline)": "gladia-prod",
    "Scribe v2": "scribe-v2-clean",
    "Soniox": "soniox",
    "base whisper-large-v3": "hf-openai-whisper-large-v3",
}
BASELINE = "Gladia (Milestone 2 baseline)"
MILESTONE_TARGET = 0.15  # >=15% relative improvement over Gladia
N_ROLLCALL_EXCLUDED = 2
BOOT, SEED = 4000, 7


def influence(a: list[tuple[int, int]], b: list[tuple[int, int]],
              wids: list[str]) -> dict:
    """Leave-one-window-out on the pooled delta.

    A domain-term metric on 250 term occurrences is exactly where one roll call can
    carry an entire headline. `exp-2026-08-10-packed-training` had a single window
    supply 67% of an effect; this is the check that would have caught it.
    """
    def delta(keep: list[int]) -> float | None:
        na = sum(a[i][1] for i in keep)
        nb = sum(b[i][1] for i in keep)
        if not na or not nb:
            return None
        return sum(a[i][0] for i in keep) / na - sum(b[i][0] for i in keep) / nb

    idx = list(range(len(wids)))
    full = delta(idx)
    shifts = []
    for i in idx:
        d = delta([k for k in idx if k != i])
        if d is None:
            continue
        shifts.append((abs(d - full), wids[i], d))
    shifts.sort(reverse=True)
    worst = shifts[0] if shifts else (0.0, None, full)
    return {"delta": full,
            "max_shift_window": worst[1], "delta_without": worst[2],
            "sign_reversed_by": [w for _, w, d in shifts if (d < 0) != (full < 0)]}


def load_terms() -> dict[str, TermList]:
    return {p.stem: TermList.load(p) for p in sorted(TERMS_DIR.glob("*.json"))}


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    rows = man["eval_windows"]
    terms = load_terms()

    report = B.load_report(man["source_run"])
    items = {it["itemId"]: it for it in report["items"]}

    # Fail loudly rather than score a complete-case subset.
    missing = []
    for r in rows:
        it = items.get(r["window_id"])
        if it is None:
            missing.append((r["window_id"], "window absent from the run"))
            continue
        for label, pid in PROVIDERS.items():
            pp = it["perProvider"].get(pid) or {}
            if not (pp.get("hypothesisText") or "").strip():
                missing.append((r["window_id"], f"no hypothesis from {label}"))
    if missing:
        for w, why in missing[:20]:
            print(f"  MISSING {w}: {why}")
        raise SystemExit(f"{len(missing)} missing hypotheses - refusing to score a "
                         f"subset. Every provider must cover all 39 windows.")

    per_window: dict[str, dict[str, dict]] = {}
    for r in rows:
        it = items[r["window_id"]]
        tl = terms[r["city"]]
        per_window[r["window_id"]] = {
            label: ds_wer(it["referenceText"],
                          it["perProvider"][pid]["hypothesisText"], tl)
            for label, pid in PROVIDERS.items()}

    # N is a property of the reference and the frozen list, identical for every
    # provider. Assert that, then use it to pick the roll-call windows.
    n_by_window = {}
    for wid, byprov in per_window.items():
        ns = {v["N"] for v in byprov.values()}
        assert len(ns) == 1, f"{wid}: providers disagree on the denominator {ns}"
        n_by_window[wid] = ns.pop()

    rollcall = [w for w, _ in sorted(n_by_window.items(), key=lambda kv: (-kv[1], kv[0]))
                ][:N_ROLLCALL_EXCLUDED]

    analyses = {
        "primary": [r for r in rows],
        "sensitivity_no_rollcall": [r for r in rows if r["window_id"] not in rollcall],
    }

    result = {
        "experiment": "exp-2026-08-12-ds-wer",
        "source_run": man["source_run"],
        "terms": {c: {"n_terms": len(t), "version":
                      json.loads((TERMS_DIR / f"{c}.json").read_text())["version"]}
                  for c, t in terms.items()},
        "rollcall_windows_excluded_in_sensitivity": rollcall,
        "n_domain_by_window": n_by_window,
        "analyses": {},
    }

    for name, subset in analyses.items():
        wids = [r["window_id"] for r in subset]
        blocks = [r["meeting_id"] for r in subset]
        by_city: dict[str, list] = {}
        for r in subset:
            by_city.setdefault(r["city"], []).append(r["window_id"])

        arms = {}
        for label in PROVIDERS:
            rowsp = [per_window[w][label] for w in wids]
            agg = aggregate(rowsp)
            agg["per_city"] = {c: aggregate([per_window[w][label] for w in ws])
                               for c, ws in sorted(by_city.items())}
            arms[label] = agg

        base = [( per_window[w][BASELINE]["errors"], per_window[w][BASELINE]["N"])
                for w in wids]
        for label in PROVIDERS:
            if label == BASELINE:
                continue
            ours = [(per_window[w][label]["errors"], per_window[w][label]["N"])
                    for w in wids]
            ci = cluster_bootstrap(ours, base, blocks, n_boot=BOOT, seed=SEED)
            arms[label]["vs_baseline"] = {
                "delta_ds_wer": ci["delta"], "ci95": ci["ci95"],
                "excludes_zero": ci["excludes_zero"], "n_meetings": ci["n_clusters"],
                "influence": influence(ours, base, wids)}

        b = arms[BASELINE]["ds_wer"]
        o = arms["ours (artifact-ct2-fixed)"]["ds_wer"]
        milestone = None
        if b:
            milestone = {"relative_improvement": (b - o) / b,
                         "target": MILESTONE_TARGET,
                         "met": bool((b - o) / b >= MILESTONE_TARGET)}
        else:
            milestone = {"relative_improvement": None,
                         "note": "Gladia DS-WER is 0; the ratio is undefined. "
                                 "Read the absolute delta instead."}

        result["analyses"][name] = {
            "n_windows": len(wids), "n_meetings": len(set(blocks)),
            "n_domain": arms[BASELINE]["N"], "arms": arms, "milestone": milestone}

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "results.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    for name, a in result["analyses"].items():
        print(f"\n== {name}: {a['n_windows']} windows, {a['n_meetings']} meetings, "
              f"N_domain={a['n_domain']}")
        for label, arm in a["arms"].items():
            line = (f"  {label:32s} DS-WER {arm['ds_wer']:.4f}  "
                    f"S{arm['S']:3d} D{arm['D']:3d} I{arm['I']:3d}")
            if "vs_baseline" in arm:
                v = arm["vs_baseline"]
                line += (f"  vs Gladia {v['delta_ds_wer']:+.4f} "
                         f"[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]")
            print(line)
        m = a["milestone"]
        if m.get("relative_improvement") is not None:
            print(f"  Milestone 2: {m['relative_improvement']*100:+.1f}% relative vs "
                  f"Gladia (target >= +15%) -> {'MET' if m['met'] else 'NOT MET'}")
    print(f"\nexcluded as roll-call in the sensitivity analysis: {rollcall}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
