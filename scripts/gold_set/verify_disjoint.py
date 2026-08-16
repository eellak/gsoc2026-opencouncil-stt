#!/usr/bin/env python3
"""Prove the gold-set selection touches nothing that is frozen, sealed or spent.

Checks, at MEETING level and at TIME level:
  - the 7 SEALED temporal holdout windows of eval-freeze-2026-08
  - the 39 frozen validation windows
  - every window of the 2026-08-10 benchmark run
  - the 2026-08-04 reference pool / dev allowlist / 2026-08-09 dev windows
  - the training manifest

Exit code 1 on any intersection. Run before anyone listens.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/gold_set"))
from frame import excluded_meetings  # noqa: E402


def main():
    with open(ROOT / "research/gold-set-2026-08/selection.csv", newline="", encoding="utf-8") as fh:
        sel = list(csv.DictReader(fh))
    excl = excluded_meetings()

    bad = []
    for r in sel:
        key = (r["city_id"], r["meeting_id"])
        if key in excl:
            bad.append((r["cell_id"], sorted(excl[key])))

    freeze = json.loads((ROOT / "research/eval-freeze-2026-08/manifest.json").read_text())
    sealed = {(w["city"], w["meeting_id"]) for w in freeze["holdout_windows"]}
    sealed_time = [(w["city"], w["meeting_id"], w["start_sec"], w["start_sec"] + w["duration_sec"])
                   for w in freeze["holdout_windows"]]
    time_hits = []
    for r in sel:
        for c, m, a, b in sealed_time:
            if (r["city_id"], r["meeting_id"]) == (c, m) and \
               float(r["clip_end"]) > a and float(r["clip_start"]) < b:
                time_hits.append(r["cell_id"])

    meetings = {(r["city_id"], r["meeting_id"]) for r in sel}
    print(f"selection: {len(sel)} cells in {len(meetings)} meetings, "
          f"{len({m[0] for m in meetings})} cities")
    print(f"meeting-level collisions with any spent/frozen cohort: {len(bad)}")
    print(f"time-level collisions with the 7 SEALED holdout windows: {len(time_hits)}")
    print(f"sealed holdout meetings in selection: "
          f"{len(meetings & sealed)}")
    if bad or time_hits:
        for x in bad[:20]:
            print("  COLLISION", x)
        print("FAIL")
        return 1
    print("PASS - the selection is disjoint from every frozen, sealed and spent cohort")
    return 0


if __name__ == "__main__":
    sys.exit(main())
