#!/usr/bin/env python3
"""§3.7 of `docs/specs/2026-08-16-overlap-speaker-arms-prereg.md`: the density rule on
the frozen gold cells.

DESCRIPTIVE ONLY, DECLARED IN ADVANCE. 27 cells of 15 scored seconds in 6 meetings is
not a probability sample of anything. No population lower bound is computed, no
confidence-bound method is applied, and the "lower bound" label that
`exp-2026-08-16-gold-set` withdrew from the detector's precision is NOT restored here.

What differs from the 247-window run, and why nothing may be compared across them:

  * the time axis is the adapter's OWN word timestamps, not whisper-turbo anchoring;
  * `rho_single` is frozen at the 247-window pooled value (non-gold data);
  * there is no benchmark reference here, so there are no deletion runs and no
    precision - the only thing on offer is whether a flag lands on speech a human
    heard, which is fidelity-to-audio and is never merged with the other numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.controlled_eval import density_omission as D          # noqa: E402
from eval.controlled_eval.scoring import wtoks                  # noqa: E402

OUT = Path(__file__).with_name("results_density_gold.json")


def sc() -> Path:
    return Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))


def main(rho_single: float) -> None:
    g = sc() / "gold-set-2026-08"
    man = json.loads((g / "manifest.json").read_text())
    cells = {c["id"]: c for c in json.loads((g / "cells-frozen.json").read_text())["cells"]}
    answers = json.loads((g / f"answers-user-2026-08-16.json").read_text())

    rows = []
    for cid in man["scored_cell_ids"]:
        cell = cells[cid]
        # gc_{city}_{meeting}_{core_start_ms}; the meeting id itself carries underscores
        parts = cid.split("_")
        city, mid = parts[1], "_".join(parts[2:-1])
        diar = json.loads((sc() / "gold-set/diar" / f"{city}__{mid}.json")
                          .read_text())["output"]["diarization"]
        lead = float(cell["lead"])
        core_s = int(parts[-1]) / 1000.0
        core_e = core_s + float(cell["core"])
        # meeting time -> clip time, then keep only the scored core
        segs = []
        for s in diar:
            a, b = max(float(s["start"]), core_s), min(float(s["end"]), core_e)
            if b > a:
                segs.append({"speaker": s["speaker"], "start": a - core_s + lead,
                             "end": b - core_s + lead})
        hyp = json.loads((g / "hyp/adapter" / f"{cid}.json").read_text())
        times = []
        for u in hyp["transcription"].get("utterances", []):
            for w in u.get("words", []):
                for _ in wtoks(w["word"]):
                    times.append((float(w["start"]) + float(w["end"])) / 2)
        times.sort()

        flags = D.merge(D.raw_flags(segs, times, rho_single, D.MISSING_THRESHOLD))
        old = D.merge(D.old_rule_flags(segs, times))
        blocks = (answers.get(cid, {}).get("b") or {}).get("blocks") or []
        certain = [(float(b["s"]), float(b["e"])) for b in blocks
                   if not b.get("text_unc")]
        ov_blocks = [(float(b["s"]), float(b["e"])) for b in blocks
                     if b.get("ov_with")]

        def hits(fs, tgt):
            return sum(1 for f in fs
                       if any(f[0] < t[1] and t[0] < f[1] for t in tgt))

        rows.append({
            "cell": cid, "city": city, "meeting": mid,
            "n_flags_new": len(flags), "n_flags_old": len(old),
            "flags_on_certain_human_speech": hits(flags, certain),
            "flags_on_overlap_marked_speech": hits(flags, ov_blocks),
            "old_flags_on_certain_human_speech": hits(old, certain),
            "our_tokens": len(times),
            "overlap_intervals": sum(
                1 for s, e, n in D.eligible(segs, rho_single) if n >= 2),
        })

    tot = {k: sum(r[k] for r in rows) for k in
           ("n_flags_new", "n_flags_old", "flags_on_certain_human_speech",
            "flags_on_overlap_marked_speech", "old_flags_on_certain_human_speech",
            "overlap_intervals")}
    res = {"spec": "docs/specs/2026-08-16-overlap-speaker-arms-prereg.md §3.7",
           "status": "DESCRIPTIVE ONLY - no floor, no bound, no precision",
           "n_cells": len(rows), "n_meetings": len({r["meeting"] for r in rows}),
           "rho_single_frozen_from_247_windows": rho_single,
           "time_axis": "the adapter's own word timestamps (NOT whisper-turbo "
                        "anchoring) - not comparable to the 247-window numbers",
           "totals": tot,
           "cells_with_at_least_one_new_flag": sum(1 for r in rows
                                                   if r["n_flags_new"]),
           "per_cell": rows}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "per_cell"},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 2.4)
