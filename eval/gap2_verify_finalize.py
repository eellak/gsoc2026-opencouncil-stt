"""Summarize the gap2 audio-verification manifest -> summary JSON + gap3 ids.

Reads ~/.cache/oc-public/gap-verify/manifest.jsonl (last record per id wins),
writes:
  data/reports/finetune-research/gap2-verify-summary-2026-08-13.json
  data/reports/finetune-research/gap3-ids.json   (VERIFIED in gap2 order,
                                                  then MIDDLE in gap2 order)
Safe to run on a partial manifest (reports progress counts).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDS_PATH = ROOT / "ui/src/lib/server/state/gap2-ids.json"
MANIFEST = Path.home() / ".cache/oc-public/gap-verify/manifest.jsonl"
OUTDIR = ROOT / "data/reports/finetune-research"


def main() -> None:
    ids = json.loads(IDS_PATH.read_text())
    recs: dict[str, dict] = {}
    for l in MANIFEST.read_text().splitlines():
        try:
            r = json.loads(l)
            recs[r["id"]] = r
        except Exception:
            pass

    tiers = collections.Counter()
    fracs = []
    errors = collections.Counter()
    flags = collections.Counter()
    for uid in ids:
        r = recs.get(uid)
        if r is None:
            tiers["(pending)"] += 1
            continue
        if r.get("error"):
            tiers["ERROR"] += 1
            errors[r["error"].split(":")[0]] += 1
            continue
        tiers[r.get("tier") or "(none)"] += 1
        if r.get("found_frac") is not None:
            fracs.append(r["found_frac"])
        for f in r.get("flags") or []:
            flags[f] += 1

    fracs.sort()
    q = lambda p: round(fracs[int(p * (len(fracs) - 1))], 3) if fracs else None
    n_done = len(ids) - tiers["(pending)"]
    verified = [u for u in ids if recs.get(u, {}).get("tier") == "VERIFIED"]
    middle = [u for u in ids if recs.get(u, {}).get("tier") == "MIDDLE"]
    gap3 = verified + middle

    summary = {
        "date": "2026-08-13",
        "input": "ui/src/lib/server/state/gap2-ids.json",
        "n_input": len(ids),
        "n_done": n_done,
        "tiers": dict(tiers),
        "flags": dict(flags),
        "errors_by_kind": dict(errors),
        "found_frac": {"n": len(fracs), "median": q(0.5), "p10": q(0.1),
                       "p25": q(0.25), "p75": q(0.75), "p90": q(0.9)},
        "survival": {
            "verified": len(verified), "middle": len(middle),
            "gap3_total": len(gap3),
            "verified_frac_of_done": round(len(verified) / n_done, 4) if n_done else None,
            "gap3_frac_of_done": round(len(gap3) / n_done, 4) if n_done else None,
        },
        "gap3_order": "VERIFIED in gap2 order, then MIDDLE (uncertain) in gap2 order",
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "gap2-verify-summary-2026-08-13.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    (OUTDIR / "gap3-ids.json").write_text(json.dumps(gap3))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwrote gap3-ids.json ({len(gap3)} ids)")


if __name__ == "__main__":
    main()
