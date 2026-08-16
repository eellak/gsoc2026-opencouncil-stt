#!/usr/bin/env python3
"""Stage 2: pyannoteAI precision-2 diarization over the WHOLE sampled meetings.

Whole-meeting coverage is deliberate: it makes the inclusion probability of
every cell computable, which a "diarize only the interesting bits" run would
not. No transcription is requested here - this pass only supplies the overlap
and speaker-change SIGNAL used to draw cells. Zero GPU.

Results cached at $SC/gold-set/diar/<city>__<meeting>.json.
Cost: EUR 0.112 per audio hour, ~6.8 h for the six sampled meetings.

Usage: python3 scripts/gold_set/diarize_meetings.py [--dry-run] [--limit N]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.controlled_eval.exclusive_diar_api import api_key, curl, wait_job, API, MODEL  # noqa: E402

SC = Path(os.environ.get("SC", Path.home() / ".cache/oc-public"))
OUT = SC / "gold-set" / "diar"
MEET = SC / "meetings"


def audio_url(city, mid):
    with open(MEET / f"{city}__{mid}.json", encoding="utf-8") as fh:
        return json.load(fh)["meeting"]["audioUrl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default=str(ROOT / "research/gold-set-2026-08/meetings.csv"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    with open(args.meetings, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    hours = sum(float(r["duration_sec"]) for r in rows) / 3600
    print(f"{len(rows)} meetings, {hours:.2f} audio hours, "
          f"projected EUR {hours * 0.112:.2f} diarization-only")
    if args.dry_run:
        for r in rows:
            print("  ", r["city_id"], r["meeting_id"], audio_url(r["city_id"], r["meeting_id"]))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    key = api_key()
    for r in rows:
        city, mid = r["city_id"], r["meeting_id"]
        dst = OUT / f"{city}__{mid}.json"
        if dst.exists():
            print(f"  cached {city}__{mid}")
            continue
        url = audio_url(city, mid)
        body = {"url": url, "model": MODEL, "exclusive": True}
        resp = curl(["-X", "POST", f"{API}/diarize",
                     "-H", f"Authorization: Bearer {key}",
                     "-H", "Content-Type: application/json",
                     "-d", json.dumps(body)])
        job = resp.get("jobId")
        if not job:
            print(f"  FAILED to submit {city}__{mid}: {resp}")
            continue
        print(f"  submitted {city}__{mid} job={job}")
        res = wait_job(job, key, timeout=args.timeout)
        if res.get("status") != "succeeded":
            print(f"  {city}__{mid}: {res.get('status')}")
            continue
        dst.write_text(json.dumps(res), encoding="utf-8")
        d = res["output"].get("diarization", [])
        e = res["output"].get("exclusiveDiarization", [])
        print(f"  ok {city}__{mid}: {len(d)} turns, {len(e)} exclusive turns")


if __name__ == "__main__":
    main()
