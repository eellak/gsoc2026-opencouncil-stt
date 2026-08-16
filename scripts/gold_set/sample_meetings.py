#!/usr/bin/env python3
"""Stage 1 of the gold-set sampling design (issue #21).

Draws the MEETING sample with known inclusion probabilities, frozen by seed,
before any audio is heard and before any overlap signal is computed.

Design (frozen 2026-08-16):
  restriction:  900 <= duration_sec <= 9000, n_speaker_tags >= 3,
                hidden_for_review == False
  stage A:      draw K cities uniformly without replacement from the eligible cities
  stage B:      draw 1 meeting uniformly from each drawn city
  pi(meeting m in city c) = (K / n_cities) * (1 / n_meetings_in_c)

Writes research/gold-set-2026-08/meetings.csv (PII-free: ids, duration, inclusion prob).
"""
import argparse
import csv
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 21
K_CITIES = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=os.path.join(ROOT, "research/gold-set-2026-08/frame.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "research/gold-set-2026-08/meetings.csv"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=K_CITIES)
    args = ap.parse_args()

    with open(args.frame, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    elig = [r for r in rows
            if 900 <= float(r["duration_sec"]) <= 9000
            and int(r["n_speaker_tags"]) >= 3
            and r["hidden_for_review"] == "False"]

    by_city = {}
    for r in elig:
        by_city.setdefault(r["city_id"], []).append(r)
    for v in by_city.values():
        v.sort(key=lambda r: r["meeting_id"])
    cities = sorted(by_city)

    rng = random.Random(args.seed)
    drawn = sorted(rng.sample(cities, args.k))
    out = []
    for c in drawn:
        m = rng.choice(by_city[c])
        out.append({
            "city_id": c,
            "meeting_id": m["meeting_id"],
            "date": m["date"],
            "duration_sec": m["duration_sec"],
            "n_speaker_tags": m["n_speaker_tags"],
            "n_people": m["n_people"],
            "pi_meeting": round((args.k / len(cities)) * (1.0 / len(by_city[c])), 6),
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    meta = {"seed": args.seed, "k_cities": args.k, "n_eligible_meetings": len(elig),
            "n_eligible_cities": len(cities),
            "restriction": "900<=duration_sec<=9000, n_speaker_tags>=3, not hidden_for_review",
            "total_hours_drawn": round(sum(float(r["duration_sec"]) for r in out) / 3600, 2)}
    with open(os.path.join(os.path.dirname(args.out), "meetings_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    for r in out:
        print(f"  {r['city_id']:14s} {r['meeting_id']:18s} {float(r['duration_sec'])/60:6.1f} min  pi={r['pi_meeting']}")


if __name__ == "__main__":
    main()
