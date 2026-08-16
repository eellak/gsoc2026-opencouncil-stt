#!/usr/bin/env python3
"""Build the sampling frame for the gold set (issue #21, Phase 1).

Eligible meetings = the 151 untouched public meetings of
exp-2026-08-04-public-meetings, minus every meeting that any earlier
evaluation cohort has already touched:

  - the 2026-08-10 benchmark run (253/247 windows)
  - the 39 frozen validation windows of eval-freeze-2026-08
  - the 7 SEALED temporal holdout windows (hard rule)
  - the 2026-08-04 reference pool / dev allowlist / 2026-08-09 dev windows

Writes only PII-free columns. No transcript text, no audio, ever.

Usage:  python3 scripts/gold_set/frame.py [--out research/gold-set-2026-08/frame.csv]
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.expanduser(os.environ.get("SC", "~/.cache/oc-public"))
BENCH_RUN = "2026-08-10-corrected-adapter-label-prefix-fix-vs-ju"


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def excluded_meetings():
    """(city_id, meeting_id) pairs that are off limits, with the reason."""
    out = {}

    def mark(city, mid, reason):
        out.setdefault((city, mid), set()).add(reason)

    fz = os.path.join(ROOT, "research/eval-freeze-2026-08/manifest.json")
    with open(fz, encoding="utf-8") as fh:
        freeze = json.load(fh)
    for w in freeze["eval_windows"]:
        mark(w["city"], w["meeting_id"], "freeze39")
    for w in freeze["holdout_windows"]:
        mark(w["city"], w["meeting_id"], "sealed_holdout")

    bench = os.path.join(SC, f"bench_{BENCH_RUN}.json")
    if os.path.exists(bench):
        with open(bench, encoding="utf-8") as fh:
            rep = json.load(fh)
        for it in rep.get("items", []):
            city = it.get("cityId") or it.get("city_id")
            mid = it.get("meetingId") or it.get("meeting_id")
            if city and mid:
                mark(city, mid, "bench2026-08-10")
    else:
        print(f"WARNING: benchmark report not cached at {bench}", file=sys.stderr)

    for name, reason in (
        ("reference_pool.csv", "reference_pool"),
        ("dev_allowlist.csv", "dev_allowlist"),
        ("dev_priority_may.csv", "dev_priority"),
        ("dev_windows_2026-08-09.csv", "dev_windows_0809"),
    ):
        p = os.path.join(ROOT, "data/public-meetings", name)
        if os.path.exists(p):
            for r in _read_csv(p):
                mark(r["city_id"], r["meeting_id"], reason)

    tm = os.path.join(ROOT, "data/asr/train_manifest.csv")
    if os.path.exists(tm):
        for r in _read_csv(tm):
            if r.get("city_id") and r.get("meeting_id"):
                mark(r["city_id"], r["meeting_id"], "train")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "research/gold-set-2026-08/frame.csv"))
    args = ap.parse_args()

    idx = {(r["city_id"], r["meeting_id"]): r
           for r in _read_csv(os.path.join(ROOT, "data/public-meetings/index.csv"))}
    fetched = _read_csv(os.path.join(ROOT, "data/public-meetings/fetched_summary.csv"))
    excl = excluded_meetings()

    rows, dropped = [], {}
    for r in fetched:
        key = (r["city_id"], r["meeting_id"])
        reasons = excl.get(key)
        if reasons:
            for x in reasons:
                dropped[x] = dropped.get(x, 0) + 1
            continue
        meta = idx.get(key, {})
        dur = float(r["duration_sec"] or 0)
        rows.append({
            "city_id": key[0],
            "meeting_id": key[1],
            "date": meta.get("date", ""),
            "duration_sec": round(dur, 1),
            "n_segments": int(r["n_segments"] or 0),
            "n_utterances": int(r["n_utterances"] or 0),
            "n_people": int(r["n_people"] or 0),
            "n_speaker_tags": int(r["n_speaker_tags"] or 0),
            "hidden_for_review": r["hidden_for_review"],
            "has_audio": meta.get("has_audio", ""),
        })

    rows.sort(key=lambda r: (r["city_id"], r["meeting_id"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"fetched={len(fetched)} eligible={len(rows)} cities={len({r['city_id'] for r in rows})}")
    print("dropped by reason:", dict(sorted(dropped.items())))
    print(f"total eligible audio: {sum(r['duration_sec'] for r in rows)/3600:.1f} h")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
