#!/usr/bin/env python3
"""Stage 4: draw the gold cells. FROZEN before anyone listens to anything.

Allocation (frozen 2026-08-16, seed 21):

  required 36 cells   P 14 | I 6 | H 6 | M 4 | R 6
  stretch  18 cells   P  7 | I 3 | H 3 | M 2 | R 3
  warm-up   3 cells   drawn from a SEVENTH meeting, outside the gold frame,
                      so practice never consumes gold.

R is drawn first, uniformly over every eligible cell, independent of stream.
The stream draws then exclude whatever R already took, so a cell is never
double counted; both inclusion probabilities are recorded per cell.

Constraints: at most 7 required cells per meeting (11 cumulative with the stretch tier), and selected cores at
least MIN_GAP seconds apart, so the sample is not a handful of long arguments.

3 of the required cells are flagged `calibration`: the user transcribes those
from a BLANK box before the prefill is ever shown, which is the only measure of
how much the hybrid mode misses.
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 21
REQUIRED = {"P": 14, "I": 6, "H": 6, "M": 4, "R": 6}
STRETCH = {"P": 7, "I": 3, "H": 3, "M": 2, "R": 3}
MAX_PER_MEETING = {"required": 7, "stretch": 11}  # cumulative cap per tier
MIN_GAP = 60.0
N_CALIBRATION = 3
# The required tier was trimmed from 36 to 27 cells once the human-time estimate
# existed (2h13 vs 2h47 against a hard 2-3 h budget), BEFORE anyone listened to
# anything. The trim is deterministic - the first k cell_ids of each stream - so
# it consumes no randomness and cannot have been steered by any result.
CORE27 = {"P": 12, "I": 4, "H": 4, "M": 3, "R": 4}


def pick(pool, k, rng, taken, per_meet, chosen_cores, cap):
    """Greedy draw honouring the meeting cap and the minimum-gap rule."""
    out = []
    for c in rng.sample(pool, len(pool)):
        if len(out) >= k:
            break
        if c["cell_id"] in taken:
            continue
        m = (c["city_id"], c["meeting_id"])
        if per_meet[m] >= cap:
            continue
        if any(abs(float(c["core_start"]) - t) < MIN_GAP for t in chosen_cores[m]):
            continue
        out.append(c)
        taken.add(c["cell_id"])
        per_meet[m] += 1
        chosen_cores[m].append(float(c["core_start"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(ROOT / "research/gold-set-2026-08/cells.csv"))
    ap.add_argument("--out", default=str(ROOT / "research/gold-set-2026-08/selection.csv"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    with open(args.cells, newline="", encoding="utf-8") as fh:
        cells = list(csv.DictReader(fh))
    by_stream = {}
    for c in cells:
        by_stream.setdefault(c["stream"], []).append(c)
    N = {k: len(v) for k, v in by_stream.items()}
    N["R"] = len(cells)

    rng = random.Random(args.seed)
    taken, per_meet, cores = set(), Counter(), {}
    for c in cells:
        cores.setdefault((c["city_id"], c["meeting_id"]), [])

    sel = []
    for tier, alloc in (("required", REQUIRED), ("stretch", STRETCH)):
        # R first: uniform over the whole frame, independent of stream
        for c in pick(cells, alloc["R"], rng, taken, per_meet, cores, MAX_PER_MEETING[tier]):
            sel.append((tier, "R", c))
        for s in ("P", "I", "H", "M"):
            for c in pick(by_stream[s], alloc[s], rng, taken, per_meet, cores, MAX_PER_MEETING[tier]):
                sel.append((tier, s, c))

    kept = set()
    for st, k in CORE27.items():
        ids = sorted(c["cell_id"] for t, ss, c in sel if t == "required" and ss == st)
        kept.update(ids[:k])
    calib = set(rng.sample(sorted(kept), N_CALIBRATION))

    n_draw = {t: {s: sum(1 for tt, ss, _ in sel if tt == t and ss == s)
                  for s in ("P", "I", "H", "M", "R")} for t in ("required", "stretch")}

    rows = []
    for tier, s, c in sel:
        k = n_draw["required"][s] + n_draw["stretch"][s]
        rows.append({
            "cell_id": c["cell_id"], "tier": tier, "draw_stream": s,
            "city_id": c["city_id"], "meeting_id": c["meeting_id"],
            "core_start": c["core_start"], "core_end": c["core_end"],
            "clip_start": c["clip_start"], "clip_end": c["clip_end"],
            "speech_frac": c["speech_frac"],
            "pyannote_overlap_sec": c["pyannote_overlap_sec"],
            "pyannote_speakers": c["pyannote_speakers"],
            "pub_interjections": c["pub_interjections"],
            "frame_stream": c["stream"],
            "pi_cell_in_stream": round(k / N[s], 6),
            "calibration": c["cell_id"] in calib,
        })
    for r in rows:
        r["required27"] = r["tier"] == "required" and r["cell_id"] in kept
    rows.sort(key=lambda r: (not r["required27"], r["tier"] != "required",
                             r["city_id"], float(r["core_start"])))

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "seed": args.seed, "allocation_required": REQUIRED, "allocation_stretch": STRETCH,
        "frame_sizes": N, "max_per_meeting_cumulative": MAX_PER_MEETING, "min_gap_sec": MIN_GAP,
        "n_calibration": N_CALIBRATION, "calibration_cells": sorted(calib),
        "core27": CORE27,
        "n_required27": sum(1 for r in rows if r["required27"]),
        "scored_core_min_required27": round(sum(15.0 for r in rows if r["required27"]) / 60, 2),
        "n_required_drawn": sum(1 for r in rows if r["tier"] == "required"),
        "n_stretch": sum(1 for r in rows if r["tier"] == "stretch"),
        "scored_core_sec_required": round(sum(15.0 for r in rows if r["tier"] == "required"), 1),
    }
    (Path(args.out).parent / "selection_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(Counter((r["tier"], r["draw_stream"]) for r in rows))
    print(Counter(r["city_id"] for r in rows if r["tier"] == "required"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
