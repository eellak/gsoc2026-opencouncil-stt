"""Derive the mixture-ratio arms from one superset clip build.

Runs on the pod after `train_runpod.py` has built clips from the EXPANDED train
parquet. That build is the superset: every clip either arm needs is already on disk,
so an arm is just a different list over the same files and costs no extra GPU time.

Spec: `docs/specs/mixture-ratio-preregistration.md`.

Three invariants, all asserted rather than assumed — a review of the first version
found that every one of them was merely hoped for, and a silent violation would have
produced a confidently wrong answer overnight:

1. **Equal audio.** Each arm gets the same total unique duration as the control,
   within `TOL_HOURS`. `MAX_STEPS` pins the optimizer budget separately, so row counts
   are free to differ and duration is the quantity worth holding equal.
2. **Equal city mixture.** Each arm matches the control's per-city share of hours,
   within `TOL_CITY`. This is the one that needed new data: corrections concentrate in
   Chania and Athens, so cutting corrections drops those cities unless the no-edit pool
   in them is deep enough to compensate. It now is.
3. **Only the mixture differs.** The correction share hits its target within
   `TOL_SHARE`, and for arm C the corrections that survive are the hand-reviewed ones.

Any violation is a hard failure. A drifted arm is worse than no arm.

    mix_arms.py --work /workspace/whisper-run \
                --parquet   .../train_expanded.parquet \
                --control   .../train_control.parquet \
                --nb2-ids   .../nb2_ids.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

# arm -> (correction share of audio hours, cut algorithm-selected first)
ARMS = {"C": (0.20, True), "C1": (0.20, False)}
DIAGNOSTIC = {"C1"}          # one seed only: isolates ratio from provenance
SEEDS = [13, 29, 47]

TOL_HOURS = 0.02             # 2% of the control's total duration
TOL_SHARE = 0.015            # 1.5 percentage points on the correction share
TOL_CITY = 0.020             # 2.0 percentage points on any single city's share


def load_meta(parquet: str) -> dict:
    """utterance_id -> (source, city, duration). Clip filenames are the utterance id.

    Duration is the RAW span, not the boundary-corrected one, even though clips are
    cut on the corrected span. The pool was sized on raw spans before alignment
    existed, and the corrected spans run wider, so budgeting on them would demand a
    pool that was never selected. Raw is the one basis available at every stage, and
    it is the same basis both arms are measured on."""
    import pandas as pd
    df = pd.read_parquet(parquet, columns=["utterance_id", "source", "city_id",
                                           "start", "end"])
    return {r.utterance_id: (r.source, r.city_id, float(r.end) - float(r.start))
            for r in df.itertuples(index=False)}


def summarize(items: list[dict], meta: dict) -> dict:
    h: collections.Counter = collections.Counter()
    n: collections.Counter = collections.Counter()
    ch: collections.Counter = collections.Counter()
    for c in items:
        src, city, dur = meta[Path(c["audio"]).stem]
        h[src] += dur
        n[src] += 1
        ch[city] += dur
    tot_h = sum(h.values()) or 1.0
    tot_n = sum(n.values()) or 1
    return {"rows": tot_n, "hours": round(tot_h / 3600, 3),
            "rows_correction": n["correction"], "rows_no_edit": n["no_edit"],
            "hours_correction": round(h["correction"] / 3600, 3),
            "hours_no_edit": round(h["no_edit"] / 3600, 3),
            "share_hours_correction": round(h["correction"] / tot_h, 4),
            "share_rows_correction": round(n["correction"] / tot_n, 4),
            "city_share_hours": {c: round(v / tot_h, 4) for c, v in sorted(ch.items())}}


def _fill(pool: list[dict], meta: dict, want_s: float, rng: random.Random,
          deprioritize: set) -> tuple[list[dict], float]:
    """Take clips until `want_s` seconds are covered. Returns (items, seconds taken).

    Ordering is random within tier, so the draw is unbiased, but algorithm-selected
    corrections form a second tier that is only reached once the hand-reviewed ones
    run out. That is the whole point of arm C: the corrections it keeps should be the
    ones a person actually looked at."""
    tier1 = [c for c in pool if Path(c["audio"]).stem not in deprioritize]
    tier2 = [c for c in pool if Path(c["audio"]).stem in deprioritize]
    rng.shuffle(tier1)
    rng.shuffle(tier2)
    out, got = [], 0.0
    for c in tier1 + tier2:
        if got >= want_s:
            break
        out.append(c)
        got += meta[Path(c["audio"]).stem][2]
    return out, got


def build_arm(train: list[dict], meta: dict, target_corr_share: float,
              city_secs: dict, seed: int, deprioritize: set) -> tuple[list[dict], dict]:
    """Fill each (city, source) cell to its own target, on real clip durations.

    Allocating at the city level rather than within each source is what keeps the
    arm-level city mixture fixed. Stratifying inside each source separately does not:
    if corrections and no-edit have different city profiles — and they do — then
    changing their ratio moves the overall city mixture even though both halves were
    'stratified'.

    A city that cannot reach its target (too small a no-edit pool) has the shortfall
    redistributed across cities that still have capacity, and the shortfall is
    reported rather than absorbed silently."""
    by_cell: dict[tuple, list[dict]] = collections.defaultdict(list)
    for c in train:
        src, city, _ = meta[Path(c["audio"]).stem]
        by_cell[(city, src)].append(c)

    rng = random.Random(seed)
    out: list[dict] = []
    deficit: dict[str, float] = {}
    used: set = set()

    for src, share in (("correction", target_corr_share),
                       ("no_edit", 1.0 - target_corr_share)):
        # pass 1: each city's own quota
        short = 0.0
        capacity: dict[str, float] = {}
        for city, secs in sorted(city_secs.items()):
            pool = by_cell.get((city, src), [])
            want = secs * share
            got_items, got = _fill(pool, meta, want, rng, deprioritize)
            out.extend(got_items)
            used.update(id(x) for x in got_items)
            if got + 1e-6 < want:
                short += want - got
                deficit[f"{city}/{src}"] = round((want - got) / 3600, 3)
            capacity[city] = sum(meta[Path(c["audio"]).stem][2]
                                 for c in pool if id(c) not in used)
        # pass 2: spread any shortfall over cities that still have material
        if short > 1.0:
            total_cap = sum(capacity.values()) or 1.0
            for city, cap in sorted(capacity.items()):
                if cap <= 0:
                    continue
                pool = [c for c in by_cell.get((city, src), []) if id(c) not in used]
                extra, got = _fill(pool, meta, short * cap / total_cap, rng,
                                   deprioritize)
                out.extend(extra)
                used.update(id(x) for x in extra)

    rng.shuffle(out)
    return out, deficit


def check(name: str, r: dict, control: dict) -> list[str]:
    """Hard invariants. Returns the list of violations; empty means the arm is sound."""
    bad = []
    if abs(r["hours"] - control["hours"]) / control["hours"] > TOL_HOURS:
        bad.append(f"duration {r['hours']:.2f}h vs control {control['hours']:.2f}h "
                   f"(> {TOL_HOURS:.0%})")
    want_share = 0.20 if name.startswith("C") else control["share_hours_correction"]
    if abs(r["share_hours_correction"] - want_share) > TOL_SHARE:
        bad.append(f"correction share {r['share_hours_correction']:.3f} vs target "
                   f"{want_share:.3f} (> {TOL_SHARE})")
    for city, want in control["city_share_hours"].items():
        got = r["city_share_hours"].get(city, 0.0)
        if abs(got - want) > TOL_CITY:
            bad.append(f"city {city} share {got:.3f} vs control {want:.3f} "
                       f"(> {TOL_CITY})")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/workspace/whisper-run")
    ap.add_argument("--parquet", required=True, help="the EXPANDED train parquet")
    ap.add_argument("--control", required=True,
                    help="the ORIGINAL train parquet — arm A is exactly its rows. "
                         "Must be a different file from --parquet; the first version "
                         "derived it by filename and the two collided.")
    ap.add_argument("--nb2-ids", required=True,
                    help="JSON list of algorithm-selected correction utterance ids. "
                         "Required: without it arm C silently becomes arm C1.")
    args = ap.parse_args()

    work = Path(args.work)
    if Path(args.control).resolve() == Path(args.parquet).resolve():
        raise SystemExit("--control and --parquet are the same file; arm A would be "
                         "the whole expanded pool")

    import pandas as pd
    base = json.load(open(work / "manifest.json"))
    meta = load_meta(args.parquet)
    train = [c for c in base["train"] if Path(c["audio"]).stem in meta]
    if len(train) != len(base["train"]):
        print(f"!! {len(base['train']) - len(train)} clips have no parquet row and "
              f"were dropped from arm construction")

    base_ids = set(pd.read_parquet(args.control, columns=["utterance_id"]).utterance_id)
    arm_a = [c for c in train if Path(c["audio"]).stem in base_ids]
    if not arm_a:
        raise SystemExit("arm A is empty — --control does not match the built clips")
    nb2 = set(json.load(open(args.nb2_ids)))

    report: dict = {"arms": {}, "deficits": {}, "violations": {}}

    def emit(name: str, items: list[dict], deficit: dict | None = None) -> dict:
        p = work / f"manifest_{name}.json"
        p.write_text(json.dumps({"train": items}, ensure_ascii=False))
        report["arms"][name] = r = summarize(items, meta)
        if deficit:
            report["deficits"][name] = deficit
        print(f"{name:8s} {r['rows']:6d} rows  {r['hours']:6.2f} h  "
              f"corrections {r['share_hours_correction']:.3f} of hours  -> {p.name}")
        return r

    control = emit("A", arm_a)
    city_secs = {c: v * control["hours"] * 3600
                 for c, v in control["city_share_hours"].items()}

    for seed in SEEDS:
        for arm, (share, prio) in ARMS.items():
            if arm in DIAGNOSTIC and seed != SEEDS[0]:
                continue
            items, deficit = build_arm(train, meta, share, city_secs, seed,
                                       nb2 if prio else set())
            name = f"{arm}_s{seed}"
            r = emit(name, items, deficit)
            bad = check(arm, r, control)
            if bad:
                report["violations"][name] = bad

    # Provenance readout: the reason arm C exists at all.
    for name in report["arms"]:
        items = json.load(open(work / f"manifest_{name}.json"))["train"]
        hp = alg = 0.0
        for it in items:
            u = Path(it["audio"]).stem
            src, _, dur = meta[u]
            if src == "correction":
                if u in nb2:
                    alg += dur
                else:
                    hp += dur
        report["arms"][name]["hours_handpicked"] = round(hp / 3600, 3)
        report["arms"][name]["hours_algorithmic"] = round(alg / 3600, 3)
        print(f"  {name:8s} hand-reviewed {hp/3600:5.2f} h   algorithmic {alg/3600:5.2f} h")

    (work / "arm_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if report["violations"]:
        for name, bad in report["violations"].items():
            for b in bad:
                print(f"VIOLATION {name}: {b}")
        raise SystemExit("arms violate the preregistered invariants — not training")
    print(f"\nall arms within tolerance; wrote {work / 'arm_report.json'}")


if __name__ == "__main__":
    main()
